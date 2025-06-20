import html
import time
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
import gc
import ipdb
import numpy as np
import torch
import yaml
from torch.utils.tensorboard.writer import SummaryWriter
from typing import List

from grpo import rollout, update_policy
from utils.model import LanguageModel


def main(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    pretrained_model_paths = [Path(path) for path in config["model"]["pretrained_model_path"]]
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(config["model"]["dtype"], torch.bfloat16)
    torch.random.manual_seed(config["training"]["random_seed"])
    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")
    if num_gpus < 3:
        print(f"Error: Need at least 3 GPUs for manual assignment, but only {num_gpus} available")
        return
    
    models = []
    model_names = [Path(path).name for path in config["model"]["pretrained_model_path"]]
    target_devices = [f"cuda:{i}" for i in range(3)]  
    
    for i, (model_path, name, device) in enumerate(zip(pretrained_model_paths, model_names, target_devices)):
        print(f"Loading {name} on {device}...")
        model = LanguageModel(
            model_path=str(model_path),
            target_device=device,
            torch_dtype=dtype,
        )
        model.model.train()
        models.append(model)
        
        torch.cuda.empty_cache()
        gc.collect()
        
        print(f"  {name} loaded successfully on {device}")

    tokenizers = [model.tokenizer for model in models]

    if config["data"]["name"] == "gsm8k":
        from utils.gsm8k import create_gsm8k_dataset, gsm8k_reward_function
        train_dataset = create_gsm8k_dataset()['train']
        reward_function = gsm8k_reward_function
    else:
        print(f"Only GSM8K dataset is supported currently. Got: {config['data']['name']}")
        return

    optimizers = []
    for model in models:
        optimizer = torch.optim.AdamW(
            model.model.parameters(),
            lr=config["training"]["learning_rate"]
        )
        optimizers.append(optimizer)

    for i, (model, name, device) in enumerate(zip(models, model_names, target_devices)):
        print(f"Model {i+1} ({name}) on {device}: {type(model.model)}")
        params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        print(f"  Parameters: {params:,}")

    step = 0
    max_steps = len(train_dataset) * config["training"]["epochs"]
    
    for epoch in range(config["training"]["epochs"]):
        print(f"Epoch {epoch+1} of {config['training']['epochs']}")
        
        for question_idx, question in enumerate(train_dataset):
            step += 1
            print(f"Step {step}/{max_steps} - Question {question_idx+1}/{len(train_dataset)}")
            all_episodes = [] 

            for model_idx, (model, tokenizer, name, device) in enumerate(zip(models, tokenizers, model_names, target_devices)):
                print(f"  {name} ({device}): Processing question...")
                
                #这是做模型的rollout
                rollout_result = rollout(
                    model=model,
                    question=question["prompt"],
                    max_gen_len=config["training"]["max_gen_len"],
                    num_answer_per_question=config["training"]["num_answers_per_question"],
                    temperature=config["training"].get("temperature", 1.0),
                )
                print(f"    {name}: Generated {len(rollout_result['completions'])} episodes")
                model_episodes = []
                model_rewards = []


                for i, result in enumerate(rollout_result["completions"]):
                    reward_result = reward_function(result, question["solution"], tokenizer.eos_token)
                    full_indices = rollout_result["indices"][i]
                    mask = rollout_result["masks"][i]
                    prefix_token_ids = full_indices[~mask].cpu().tolist()
                    generated_token_ids = full_indices[mask].cpu().tolist()
                    episode_data = {
                        "prefix_token_ids": prefix_token_ids,
                        "generated_token_ids": generated_token_ids,
                        "reward": reward_result,
                        "completion": result,
                        "model_name": name,
                        "model_idx": model_idx,
                    }
                    all_episodes.append(episode_data)
                    model_rewards.append(reward_result)

                print(f"    {name}: Mean reward: {np.mean(model_rewards):.3f}")
                
                del rollout_result
                del model_rewards
                torch.cuda.empty_cache()
                gc.collect()
            
            # Now update policy for each model sequentially
            for model_idx, (model, optimizer, tokenizer, name, device) in enumerate(zip(models, optimizers, tokenizers, model_names, target_devices)):
                print(f"  {name} ({device}): Updating policy...")
                
                own_prefix = tokenizer.apply_chat_template(
                        [{"role": "user", "content": question["prompt"]}],
                        tokenize=True,
                        add_generation_prompt=True,
                        return_tensors="pt"
                    )[0].tolist()

                model_episodes = []
                for ep in all_episodes:
                    if ep["model_idx"] == model_idx:
                        model_episodes.append({
                            "prefix_token_ids": ep["prefix_token_ids"], 
                            "generated_token_ids": ep["generated_token_ids"],
                            "reward": ep["reward"],
                        })
                    else:
                        try:
                            completion_tokenized = tokenizer(
                                ep["completion"], 
                                return_tensors="pt", 
                                add_special_tokens=False
                            )
                            new_completion_ids = completion_tokenized["input_ids"][0].tolist()
                            vocab_size = tokenizer.vocab_size
                            if all(tid < vocab_size for tid in new_completion_ids):
                                model_episodes.append({
                                    "prefix_token_ids": own_prefix, 
                                    "generated_token_ids": new_completion_ids,
                                    "reward": ep["reward"],
                                })
                            else:
                                print(f"    Skipping episode: completion tokens out of range")
                                
                        except Exception as e:
                            print(f"    Warning: Failed to retokenize completion: {e}")
                            continue
                
                # Update policy
                update_result = update_policy(
                    model=model.model,
                    optimizer=optimizer,
                    episodes=model_episodes,
                    pad_token_id=tokenizer.pad_token_id,
                    max_grad_norm=config["training"]["max_grad_norm"],
                    device=torch.device(device),
                    dtype=dtype,
                )
                print(f"    {name}: Updated policy - Loss: {update_result['loss']:.4f}, Grad norm: {update_result['grad_norm']:.4f}, Entropy: {update_result['entropy']:.4f}")
                
                del model_episodes
                del update_result
                torch.cuda.empty_cache()
                gc.collect()
            
            del all_episodes
            torch.cuda.empty_cache()
            gc.collect()
            
            if step % 5 == 0:
                print("Performing aggressive memory cleanup...")
                for i in range(3):
                    with torch.cuda.device(i):
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                gc.collect()
            
            if step % 10 == 0:
                print("GPU Memory Usage:")
                for i in range(3):
                    memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
                    memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
                    print(f"  GPU {i}: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
            
            if step % config["training"]["ckpt_save_interval"] == 0:
                for model_idx, (model, name) in enumerate(zip(models, model_names)):
                    ckpt_path = config["training"]["ckpt_dir"] / f"{name}_step_{step}"
                    model.save(str(ckpt_path))
                    print(f"Saved checkpoint for {name} at step {step}")
                    torch.cuda.empty_cache()
                    gc.collect()

    print("Training completed!")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config/grpo_s1.yaml")
    args = parser.parse_args()
    main(args.config)
