import html
import time
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
import gc
import ipdb
import numpy as np
import torch
from torch.utils.tensorboard.writer import SummaryWriter
from typing import List
from grpo import rollout, update_policy
try:
    import wandb  # Optional logging
    _WANDB_AVAILABLE = True
except Exception:
    wandb = None
    _WANDB_AVAILABLE = False
from utils.model import LanguageModel
import copy


def prepare_adapter_configs(args):
    """根据命令行参数准备适配器配置"""
    fact_config = None
    lora_config = None
    
    if args.adapter == "fact":
        fact_config = {
            "fact_rank": args.adapter_rank,
            "fact_alpha": args.adapter_alpha,
            "fact_dropout": args.adapter_dropout,
            "fact_scale": args.adapter_scale,
            "fact_target_modules": getattr(args, 'adapter_target_modules')
        }
    elif args.adapter == "lora":
        lora_config = {
            "lora_rank": args.adapter_rank,
            "lora_alpha": args.adapter_alpha,
            "lora_dropout": args.adapter_dropout,
            "target_modules": getattr(args, 'adapter_target_modules')
        }
    
    return fact_config, lora_config


def main():
    parser = ArgumentParser(description="GRPO Multi-Model Training")
    
    # 模型配置
    parser.add_argument("--models", nargs="+", required=True,
                       help="List of pretrained model paths")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"],
                       help="Model dtype")
    
    # 适配器配置
    parser.add_argument("--adapter", choices=["none", "lora", "fact"], default="none",
                       help="Type of adapter to use")
    parser.add_argument("--adapter-rank", type=int, default=16,
                       help="Adapter rank")
    parser.add_argument("--adapter-alpha", type=int, default=32,
                       help="Adapter alpha")
    parser.add_argument("--adapter-dropout", type=float, default=0.05,
                       help="Adapter dropout")
    parser.add_argument("--adapter-scale", type=float, default=1.0,
                       help="Adapter scale (for FACT)")
    parser.add_argument("--adapter-target-modules", nargs="+", 
                       default=["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                       help="Target modules for adapter")
    
    # 训练配置
    parser.add_argument("--lr", type=float, default=1e-5,
                       help="Learning rate")
    parser.add_argument("--epochs", type=int, default=1,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                       help="Number of questions per batch for policy update")
    parser.add_argument("--max-gen-len", type=int, default=512,
                       help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Generation temperature")
    parser.add_argument("--num-answers", type=int, default=4,
                       help="Number of answers per question")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    # GRPO参数
    parser.add_argument("--epsilon-low", type=float, default=0.2,
                       help="PPO clipping lower bound")
    parser.add_argument("--epsilon-high", type=float, default=0.2,
                       help="PPO clipping upper bound")
    parser.add_argument("--beta", type=float, default=0.01,
                       help="KL divergence coefficient")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                       help="Maximum gradient norm")
    parser.add_argument("--loss-type", default="grpo",
                       help="Loss type")
    
    # 数据配置
    parser.add_argument("--dataset", default="gsm8k", choices=["gsm8k"],
                       help="Dataset to use")
    
    # 输出配置
    parser.add_argument("--ckpt-dir", default="checkpoints/grpo_direct",
                       help="Checkpoint directory")
    parser.add_argument("--ckpt-interval", type=int, default=3000,
                       help="Checkpoint save interval")
    parser.add_argument("--max-steps", type=int, default=None,
                       help="Maximum number of training steps (questions) to run. If set, training ends early at this step count.")
    parser.add_argument("--save-final", action="store_true", default=True,
                       help="Save a final checkpoint at the end of training")
    parser.add_argument("--no-save-final", dest="save_final", action="store_false",
                       help="Do not save a final checkpoint at the end of training")
    parser.add_argument("--use-ref-model", action="store_true", default=True,
                       help="Use reference model for KL divergence")
    parser.add_argument("--no-ref-model", dest="use_ref_model", action="store_false",
                       help="Don't use reference model")
    
    args = parser.parse_args()

    global _WANDB_AVAILABLE

    # Initialize Weights & Biases if available (user will login via CLI)
    if _WANDB_AVAILABLE:
        try:
            _run_name = f"GRPO_{'_'.join([Path(p).name for p in (args.models or [])])}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            wandb.init(project="grpo-s1", name=_run_name, config=vars(args))
        except Exception as _e:
            print(f"[wandb] init failed: {_e}. Continue without wandb.")
            _WANDB_AVAILABLE = False
    
    # 准备适配器配置
    fact_config, lora_config = prepare_adapter_configs(args)
    
    pretrained_model_paths = [Path(path) for path in args.models]
    

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
    dtype = dtype_map.get(args.dtype, torch.bfloat16)
    torch.random.manual_seed(args.seed)
    

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
        )
        model.model.train()
        models.append(model)
        
        # Load reference model (copy of initial model for KL divergence)
        if args.use_ref_model:
            print(f"  Loading reference model for {name}...")
            ref_model = LanguageModel(
                model_path=str(model_path),
                target_device=device,
                torch_dtype=dtype,
                fact_config=None,  # 参考模型不使用适配器
                lora_config=None,  # 参考模型不使用适配器
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

    if args.dataset == "gsm8k":
        from utils.gsm8k import create_gsm8k_dataset, gsm8k_reward_function
        train_dataset = create_gsm8k_dataset()['train']
        reward_function = gsm8k_reward_function
    else:
        print(f"Only GSM8K dataset is supported currently. Got: {args.dataset}")
        return

    optimizers = []
    for model in models:
        optimizer = torch.optim.AdamW(
            model.model.parameters(),
            lr=args.lr
        )
        optimizers.append(optimizer)

    for i, (model, name, device) in enumerate(zip(models, model_names, target_devices)):
        print(f"Model {i+1} ({name}) on {device}: {type(model.model)}")
        params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        print(f"  Parameters: {params:,}")

    step = 0
    batch_step = 0  # Batch counter
    planned_total_steps = len(train_dataset) * args.epochs
    max_steps = args.max_steps if args.max_steps is not None else planned_total_steps
    
    # GRPO training parameters
    num_completions_per_prompt = args.num_answers
    epsilon_low = args.epsilon_low
    epsilon_high = args.epsilon_high
    beta = args.beta  # KL regularization
    loss_type = args.loss_type
    batch_size = args.batch_size  # Batch size for policy updates
    
    # Storage for old log probabilities (for PPO clipping)
    old_log_probs_storage = [None] * num_models
    
    # Batch episode storage for each model
    batch_episodes_storage = [[] for _ in range(num_models)]
    
    training_done = False
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1} of {args.epochs}")
        for question_idx, question in enumerate(train_dataset):
            if training_done:
                break
            step += 1
            print(f"Step {step}/{max_steps} - Question {question_idx+1}/{len(train_dataset)}")
            

            all_episodes = [] if is_multi_model_mode else None  
            model_episode_groups = []  

            for model_idx, (model, tokenizer, name, device) in enumerate(zip(models, tokenizers, model_names, target_devices)):
                print(f"  {name} ({device}): Processing question...")
                
                rollout_result = rollout(
                    model=model,
                    question=question["prompt"],
                    max_gen_len=args.max_gen_len,
                    num_answer_per_question=num_completions_per_prompt,
                    temperature=args.temperature,
                )
                print(f"    {name}: Generated {len(rollout_result['completions'])} episodes")
                
                model_episodes = []
                model_rewards = []

                for i, result in enumerate(rollout_result["completions"]):
                    reward_result = reward_function(result, question["solution"])
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
                    
                    model_episodes.append(episode_data)
                    model_rewards.append(reward_result)
                    
          
                    if is_multi_model_mode:
                        all_episodes.append(episode_data)

                model_episode_groups.append(model_episodes)
                print(f"    {name}: Mean reward: {np.mean(model_rewards):.3f}")
                # Log rewards to wandb (mean reward equals success rate for 0/1 rewards)
                if _WANDB_AVAILABLE and len(model_rewards) > 0:
                    _mean_r = float(np.mean(model_rewards))
                    try:
                        wandb.log({
                            f"reward/mean/{name}": _mean_r,
                            f"reward/success_rate/{name}": _mean_r,
                            "train/step": step,
                        })
                    except Exception as _e:
                        print(f"[wandb] log failed: {_e}")
                
                del rollout_result
                del model_rewards
                torch.cuda.empty_cache()
                gc.collect()
            

            # Add episodes to batch storage instead of immediate policy update
            for model_idx in range(num_models):
                if is_multi_model_mode:
                    # Process multi-model episodes like before
                    model_episodes = []
                    tokenizer = tokenizers[model_idx]
                    
                    own_prefix = tokenizer.apply_chat_template(
                        [{"role": "user", "content": question["prompt"]}],
                        tokenize=True,
                        add_generation_prompt=True,
                        return_tensors="pt"
                    )[0].tolist()
                    
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
                            except Exception as e:
                                print(f"    Warning: Failed to retokenize completion: {e}")
                                continue
                    
                    batch_episodes_storage[model_idx].extend(model_episodes)
                else:
                    # Single model mode
                    single_model_episodes = []
                    for ep in model_episode_groups[model_idx]:
                        single_model_episodes.append({
                            "prefix_token_ids": ep["prefix_token_ids"], 
                            "generated_token_ids": ep["generated_token_ids"],
                            "reward": ep["reward"],
                        })
                    batch_episodes_storage[model_idx].extend(single_model_episodes)
            
            # Check if we should perform batch policy update
            should_update = False
            if (question_idx + 1) % batch_size == 0:  # Reached batch size
                should_update = True
                print(f"\n🎯 Batch {batch_step + 1} completed! Performing batch policy update...")
            elif question_idx == len(train_dataset) - 1:  # Last question in epoch
                should_update = True
                print(f"\n🎯 End of epoch! Performing final batch policy update...")
            
            if should_update:
                batch_step += 1
                
                # Perform batch policy update for each model
                for model_idx, (model, optimizer, tokenizer, name, device) in enumerate(zip(models, optimizers, tokenizers, model_names, target_devices)):
                    final_episodes = batch_episodes_storage[model_idx]
                    
                    if len(final_episodes) == 0:
                        print(f"    Warning: {name} has no episodes in batch, skipping...")
                        continue
                    
                    # Ensure episodes are properly grouped for advantage calculation
                    if len(final_episodes) % num_completions_per_prompt != 0:
                        print(f"    Warning: {name} has {len(final_episodes)} episodes, not divisible by {num_completions_per_prompt}")
                        # Truncate to make it divisible
                        final_episodes = final_episodes[:len(final_episodes) - (len(final_episodes) % num_completions_per_prompt)]
                    
                    if len(final_episodes) == 0:
                        print(f"    Warning: {name} has no valid episodes after grouping, skipping...")
                        continue
                    
                    print(f"  {name} ({device}): Batch updating policy with {len(final_episodes)} episodes...")
                    
                    update_result = update_policy(
                        model=model.model,
                        optimizer=optimizer,
                        episodes=final_episodes,
                        pad_token_id=tokenizer.pad_token_id,
                        max_grad_norm=args.max_grad_norm,
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
                    
                    # Store current log probs for next iteration
                    if update_result["current_log_probs"] is not None:
                        old_log_probs_storage[model_idx] = update_result["current_log_probs"]
                    
                    print(f"    {name}: Batch updated - Loss: {update_result['loss']:.4f}, "
                          f"Grad norm: {update_result['grad_norm']:.4f}, "
                          f"Entropy: {update_result['entropy']:.4f}, "
                          f"KL div: {update_result['kl_div']:.4f}, "
                          f"Clip ratio: {update_result['clip_ratio']:.4f}")
                    
                    torch.cuda.empty_cache()
                    gc.collect()
                
                # Clear batch storage
                batch_episodes_storage = [[] for _ in range(num_models)]
                print(f"✅ Batch {batch_step} update completed!\n")
            

            if is_multi_model_mode and all_episodes is not None:
                del all_episodes
            del model_episode_groups
            torch.cuda.empty_cache()
            gc.collect()
            

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
            

            if args.ckpt_interval and args.ckpt_interval > 0 and step % args.ckpt_interval == 0:
                for model_idx, (model, name) in enumerate(zip(models, model_names)):
                    ckpt_path = Path(args.ckpt_dir) / f"{name}_step_{step}"
                    model.save(str(ckpt_path))
                    print(f"Saved checkpoint for {name} at step {step}")
                    torch.cuda.empty_cache()
                    gc.collect()

            # Early stop when reaching the max step budget
            if step >= max_steps:
                training_done = True
                break

    # Save final checkpoint(s) at the end of training if requested
    if args.save_final:
        for model_idx, (model, name) in enumerate(zip(models, model_names)):
            final_ckpt_path = Path(args.ckpt_dir) / f"{name}_final_step_{step}"
            model.save(str(final_ckpt_path))
            print(f"Saved final checkpoint for {name} at step {step}")
            torch.cuda.empty_cache()
            gc.collect()

    # Finish wandb run
    if _WANDB_AVAILABLE:
        try:
            wandb.finish()
        except Exception as _e:
            print(f"[wandb] finish failed: {_e}")

    print("Training completed!")


if __name__ == "__main__":
    main()
