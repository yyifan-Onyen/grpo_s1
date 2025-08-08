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
import copy


def prepare_adapter_configs(config):
    """准备适配器配置"""
    fact_config = None
    lora_config = None
    
    if config["model"].get("use_fact", False):
        fact_config = {
            "fact_rank": config["model"].get("fact_rank", 16),
            "fact_alpha": config["model"].get("fact_alpha", 32),
            "fact_dropout": config["model"].get("fact_dropout", 0.05),
            "fact_scale": config["model"].get("fact_scale", 1.0),
            "fact_target_modules": config["model"].get("fact_target_modules", 
                                                       ["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        }
    
    elif config["model"].get("use_lora", False):
        lora_config = {
            "lora_rank": config["model"].get("lora_rank", 16),
            "lora_alpha": config["model"].get("lora_alpha", 32),
            "lora_dropout": config["model"].get("lora_dropout", 0.05),
            "target_modules": config["model"].get("lora_target_modules", 
                                                 ["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        }
    
    return fact_config, lora_config


def main(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # 准备适配器配置
    fact_config, lora_config = prepare_adapter_configs(config)
    
    # 获取threshold配置
    reward_threshold = config["training"].get("reward_threshold", 1.0)
    print(f"Using reward threshold: {reward_threshold}")
    
    pretrained_model_paths = [Path(path) for path in config["model"]["pretrained_model_path"]]
    

    max_supported_models = 3
    num_models = min(len(pretrained_model_paths), max_supported_models)
    pretrained_model_paths = pretrained_model_paths[:num_models]
    

    is_multi_model_mode = num_models > 1
    mode_description = f"{'Multi-model collaborative' if is_multi_model_mode else 'Single-model basic'} GRPO"
    print(f"Running {mode_description} with {num_models} model(s)")
    
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(config["model"]["dtype"], torch.bfloat16)
    torch.random.manual_seed(config["training"]["random_seed"])
    

    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")
    if num_gpus < num_models:
        print(f"Error: Need at least {num_models} GPUs for manual assignment, but only {num_gpus} available")
        return
    

    models = []
    ref_models = []  # Reference models for KL divergence
    model_names = [Path(path).name for path in pretrained_model_paths]
    target_devices = [f"cuda:{i}" for i in range(num_models)]
    
    for i, (model_path, name, device) in enumerate(zip(pretrained_model_paths, model_names, target_devices)):
        print(f"Loading {name} on {device}...")
        
        # Load main model
        model = LanguageModel(
            model_path=str(model_path),
            target_device=device,
            torch_dtype=dtype,
            fact_config=fact_config,
            lora_config=lora_config,
            gradient_checkpointing=config["model"].get("gradient_checkpointing", False),
        )
        model.model.train()
        models.append(model)
        
        # Load reference model (copy of initial model for KL divergence)
        if config["training"].get("use_ref_model", True):
            print(f"  Loading reference model for {name}...")
            ref_model = LanguageModel(
                model_path=str(model_path),
                target_device=device,
                torch_dtype=dtype,
                fact_config=None,  # 参考模型不使用适配器
                lora_config=None,  # 参考模型不使用适配器
                gradient_checkpointing=False,  # 参考模型不需要梯度检查点
            )
            ref_model.model.eval()  # Keep reference model in eval mode
            # Freeze reference model parameters
            for param in ref_model.model.parameters():
                param.requires_grad = False
            ref_models.append(ref_model)
        else:
            ref_models.append(None)
        
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
    
    # GRPO training parameters
    num_completions_per_prompt = config["training"]["num_answers_per_question"]
    epsilon_low = config["training"].get("epsilon_low", 0.2)
    epsilon_high = config["training"].get("epsilon_high", 0.2)
    beta = config["training"].get("beta", 0.01)  # KL regularization
    loss_type = config["training"].get("loss_type", "grpo")
    
    # Storage for old log probabilities (for PPO clipping)
    old_log_probs_storage = [None] * num_models
    
    for epoch in range(config["training"]["epochs"]):
        print(f"Epoch {epoch+1} of {config['training']['epochs']}")
        for question_idx, question in enumerate(train_dataset):
            step += 1
            print(f"Step {step}/{max_steps} - Question {question_idx+1}/{len(train_dataset)}")
            print(f"Question: {question['prompt']}")
            print(f"Target: {question['solution']}")
            print()
            

            all_episodes = [] if is_multi_model_mode else None  
            model_episode_groups = []  

            for model_idx, (model, tokenizer, name, device) in enumerate(zip(models, tokenizers, model_names, target_devices)):
                print(f"  {name} ({device}): Processing question...")
                
                rollout_result = rollout(
                    model=model,
                    question=question["prompt"],
                    max_gen_len=config["training"]["max_gen_len"],
                    num_answer_per_question=num_completions_per_prompt,
                    temperature=config["training"].get("temperature", 1.0),
                )
                print(f"    {name}: Generated {len(rollout_result['completions'])} episodes")
                
                model_episodes = []
                model_rewards = []

                # === 详细的Sample Logging ===
                print(f"    === {name} Rollout Samples ===")
                for i, result in enumerate(rollout_result["completions"]):
                    reward_result = reward_function(result, question["solution"])
                    full_indices = rollout_result["indices"][i]
                    mask = rollout_result["masks"][i]
                    prefix_token_ids = full_indices[~mask].cpu().tolist()
                    generated_token_ids = full_indices[mask].cpu().tolist()
                    
                    # 检测乱码字符
                    has_non_ascii = any(ord(c) > 127 for c in result)
                    non_ascii_chars = [c for c in result if ord(c) > 127]
                    
                    # 详细日志输出
                    print(f"      Sample {i+1}/{len(rollout_result['completions'])}:")
                    print(f"        Reward: {reward_result}")
                    print(f"        Length: {len(result)} chars, {len(generated_token_ids)} tokens")
                    print(f"        Has non-ASCII: {has_non_ascii}")
                    if has_non_ascii:
                        print(f"        Non-ASCII chars: {non_ascii_chars[:10]}...")  # 只显示前10个
                    
                    # 显示文本内容（完整显示）
                    display_text = result.replace('\n', '\\n').replace('\t', '\\t')
                    print(f"        Text: \"{display_text}\"")
                    
                    # 如果reward为0，显示更多调试信息
                    if reward_result == 0:
                        print(f"        [FAILED] Target: {question['solution']}")
                    
                    print()  # 空行分隔
                    
                    episode_data = {
                        "prefix_token_ids": prefix_token_ids,
                        "generated_token_ids": generated_token_ids,
                        "reward": reward_result,
                        "completion": result,
                        "model_name": name,
                        "model_idx": model_idx,
                        "has_non_ascii": has_non_ascii,  # 添加乱码标记
                        "text_length": len(result),      # 添加文本长度
                        "token_count": len(generated_token_ids),  # 添加token数量
                    }
                    
                    model_episodes.append(episode_data)
                    model_rewards.append(reward_result)
                    
          
                    if is_multi_model_mode:
                        all_episodes.append(episode_data)

                model_episode_groups.append(model_episodes)
                
                # 增强的统计信息
                successful_samples = sum(1 for r in model_rewards if r > 0)
                failed_samples = len(model_rewards) - successful_samples
                non_ascii_samples = sum(1 for ep in model_episodes if ep.get("has_non_ascii", False))
                avg_text_length = np.mean([ep["text_length"] for ep in model_episodes])
                avg_token_count = np.mean([ep["token_count"] for ep in model_episodes])
                
                print(f"    {name}: Rollout Summary:")
                print(f"      Mean reward: {np.mean(model_rewards):.3f}")
                print(f"      Success/Failed: {successful_samples}/{failed_samples}")
                print(f"      Non-ASCII samples: {non_ascii_samples}/{len(model_episodes)}")
                print(f"      Avg text length: {avg_text_length:.1f} chars")
                print(f"      Avg token count: {avg_token_count:.1f} tokens")
                print()
                
                del rollout_result
                del model_rewards
                torch.cuda.empty_cache()
                gc.collect()
            

            for model_idx, (model, optimizer, tokenizer, name, device) in enumerate(zip(models, optimizers, tokenizers, model_names, target_devices)):
                print(f"  {name} ({device}): Updating policy...")
                
                final_episodes = []
                
                if is_multi_model_mode:

                    own_prefix = tokenizer.apply_chat_template(
                        [{"role": "user", "content": question["prompt"]}],
                        tokenize=True,
                        add_generation_prompt=True,
                        return_tensors="pt"
                    )[0].tolist()
                    
                    # 统计threshold过滤前后的数量
                    total_other_episodes = 0
                    high_quality_episodes = 0
                    
                    print(f"    === {name} Threshold Filtering ===")
                    
                    for ep in all_episodes:
                        if ep["model_idx"] == model_idx:
                            # 自己的答案直接使用
                            print(f"      [OWN] Using own sample: reward={ep['reward']}, from {ep['model_name']}")
                            final_episodes.append({
                                "prefix_token_ids": ep["prefix_token_ids"], 
                                "generated_token_ids": ep["generated_token_ids"],
                                "reward": ep["reward"],
                            })
                        else:
                            total_other_episodes += 1
                            source_model = ep["model_name"]
                            reward = ep["reward"]
                            has_non_ascii = ep.get("has_non_ascii", False)
                            
                            # 显示过滤决策
                            if reward >= reward_threshold:
                                high_quality_episodes += 1
                                print(f"      [ACCEPT] From {source_model}: reward={reward} (>= {reward_threshold}), non-ASCII: {has_non_ascii}")
                                
                                try:
                                    completion_tokenized = tokenizer(
                                        ep["completion"], 
                                        return_tensors="pt", 
                                        add_special_tokens=False
                                    )
                                    new_completion_ids = completion_tokenized["input_ids"][0].tolist()
                                    vocab_size = tokenizer.vocab_size
                                    if all(tid < vocab_size for tid in new_completion_ids):
                                        final_episodes.append({
                                            "prefix_token_ids": own_prefix, 
                                            "generated_token_ids": new_completion_ids,
                                            "reward": ep["reward"],
                                        })
                                        print(f"        → Successfully retokenized: {len(new_completion_ids)} tokens")
                                    else:
                                        print(f"        → SKIP: tokens out of range (max_id: {max(new_completion_ids)}, vocab: {vocab_size})")
                                        
                                except Exception as e:
                                    print(f"        → SKIP: retokenization failed: {e}")
                                    continue
                            else:
                                print(f"      [REJECT] From {source_model}: reward={reward} (< {reward_threshold}), non-ASCII: {has_non_ascii}")
                                # 显示被拒绝样本的完整文本
                                rejected_text = ep["completion"].replace('\n', '\\n')
                                print(f"        Text: \"{rejected_text}\"")
                    
                    print(f"    {name}: Threshold filtering - {high_quality_episodes}/{total_other_episodes} other episodes passed (threshold={reward_threshold})")
                    print(f"    {name}: Final episodes for training: {len(final_episodes)}")
                    print()
                    
                else:

                    for ep in model_episode_groups[model_idx]:
                        final_episodes.append({
                            "prefix_token_ids": ep["prefix_token_ids"], 
                            "generated_token_ids": ep["generated_token_ids"],
                            "reward": ep["reward"],
                        })
                

                if len(final_episodes) == 0:
                    print(f"    Warning: {name} has no valid episodes for policy update, skipping...")
                    continue
                
                # 调整分组逻辑，允许样本数量不是num_completions_per_prompt的倍数
                original_episode_count = len(final_episodes)
                if len(final_episodes) % num_completions_per_prompt != 0:
                    print(f"    Warning: {name} has {len(final_episodes)} episodes, not divisible by {num_completions_per_prompt}")
                    # 截断到最大的可整除数量
                    final_episodes = final_episodes[:len(final_episodes) - (len(final_episodes) % num_completions_per_prompt)]
                
                if len(final_episodes) == 0:
                    print(f"    Warning: {name} has no valid episodes after grouping, skipping...")
                    continue
                
                print(f"    {name}: Using {len(final_episodes)} episodes for policy update (filtered from {original_episode_count})")
                

                update_result = update_policy(
                    model=model.model,
                    optimizer=optimizer,
                    episodes=final_episodes,
                    pad_token_id=tokenizer.pad_token_id,
                    max_grad_norm=config["training"]["max_grad_norm"],
                    device=torch.device(device),
                    dtype=dtype,
                    num_completions_per_prompt=num_completions_per_prompt,
                    ref_model=ref_models[model_idx].model if ref_models[model_idx] is not None else None,
                    old_log_probs=old_log_probs_storage[model_idx],
                    epsilon_low=epsilon_low,
                    epsilon_high=epsilon_high,
                    beta=beta,
                    loss_type=loss_type,
                )
                
                # Store current log probs for next iteration (PPO clipping)
                if update_result["current_log_probs"] is not None:
                    old_log_probs_storage[model_idx] = update_result["current_log_probs"]
                
                print(f"    {name}: Updated policy - Loss: {update_result['loss']:.4f}, "
                      f"Grad norm: {update_result['grad_norm']:.4f}, "
                      f"Entropy: {update_result['entropy']:.4f}, "
                      f"KL div: {update_result['kl_div']:.4f}, "
                      f"Clip ratio: {update_result['clip_ratio']:.4f}")
                
                del final_episodes
                del update_result
                torch.cuda.empty_cache()
                gc.collect()
            

            if is_multi_model_mode and all_episodes is not None:
                del all_episodes
            del model_episode_groups
            torch.cuda.empty_cache()
            gc.collect()
            
            # === Step Summary ===
            print(f"=== Step {step} Completed ===")
            print(f"All models have finished processing question {question_idx+1}")
            print("=" * 50)
            print()
            

            if step % 5 == 0:
                print("Performing aggressive memory cleanup...")
                for i in range(num_models):
                    with torch.cuda.device(i):
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                gc.collect()
            

            if step % 10 == 0:
                print("GPU Memory Usage:")
                for i in range(num_models):
                    memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
                    memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
                    print(f"  GPU {i}: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")
            

            if step % config["training"]["ckpt_save_interval"] == 0:
                for model_idx, (model, name) in enumerate(zip(models, model_names)):
                    ckpt_path = Path(config["training"]["ckpt_dir"]) / f"{name}_step_{step}"
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
