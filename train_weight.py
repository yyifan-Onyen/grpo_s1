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


def compute_ranking_weights(model_rewards, min_weight=0.3, max_weight=2.0):
    """
    基于排名的权重计算
    表现最好 → 最小权重 → 减少更新
    表现最差 → 最大权重 → 增加更新
    
    Args:
        model_rewards: List of mean rewards for each model
        min_weight: Weight for best performing model
        max_weight: Weight for worst performing model
    
    Returns:
        List of weights for each model
    """
    num_models = len(model_rewards)
    
    if num_models == 1:
        return [1.0]
    
    # 获取排名 (0=最好, num_models-1=最差)
    sorted_indices = np.argsort(model_rewards)[::-1]  # 降序排列
    ranks = np.empty_like(sorted_indices)
    ranks[sorted_indices] = np.arange(num_models)
    
    # 线性映射：rank 0 → min_weight, rank (num_models-1) → max_weight
    weights = []
    for rank in ranks:
        weight = min_weight + (max_weight - min_weight) * rank / (num_models - 1)
        weights.append(weight)
    
    return weights


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


def compute_individual_weights(model_rewards, k=7.0, threshold=0.5):
    weights = []
    for reward in model_rewards:
        weight = 1 / (1 + np.exp(k * (reward - threshold)))
        weights.append(weight)
    return weights


def main(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # 准备适配器配置
    fact_config, lora_config = prepare_adapter_configs(config)
    
    pretrained_model_paths = [Path(path) for path in config["model"]["pretrained_model_path"]]
    

    max_supported_models = 3
    num_models = min(len(pretrained_model_paths), max_supported_models)
    pretrained_model_paths = pretrained_model_paths[:num_models]
    

    is_multi_model_mode = num_models > 1
    mode_description = f"{'Multi-model collaborative with ranking weights' if is_multi_model_mode else 'Single-model basic'} GRPO"
    print(f"Running {mode_description} with {num_models} model(s)")
    
    # 动态权重配置
    ranking_config = config["training"].get("ranking_weights", {})
    min_weight = ranking_config.get("min_weight", 0.3)
    max_weight = ranking_config.get("max_weight", 2.0)
    
    # 个体权重配置
    individual_config = config["training"].get("individual_weights", {})
    individual_k = individual_config.get("k", 5.0)
    individual_threshold = individual_config.get("threshold", 0.5)
    
    print(f"Ranking weights: min={min_weight}, max={max_weight}")
    print(f"Individual weights: k={individual_k}, threshold={individual_threshold}")
    
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

    # 创建优化器和保存基础学习率
    base_learning_rate = config["training"]["learning_rate"]
    optimizers = []
    for model in models:
        optimizer = torch.optim.AdamW(
            model.model.parameters(),
            lr=base_learning_rate
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
            

            all_episodes = [] if is_multi_model_mode else None  
            model_episode_groups = []
            model_mean_rewards = []  # 收集所有模型的平均奖励

            # === Rollout Phase ===
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

                for i, result in enumerate(rollout_result["completions"]):
                    reward_result = reward_function(result, question["solution"])
                    full_indices = rollout_result["indices"][i]
                    mask = rollout_result["masks"][i]
                    prefix_token_ids = full_indices[~mask].cpu().tolist()
                    generated_token_ids = full_indices[mask].cpu().tolist()
                    # 生成阶段的旧 log_probs（仅生成段）
                    _old_lps = rollout_result.get("log_probs", None)
                    old_lp = _old_lps[i].detach().cpu().tolist() if isinstance(_old_lps, list) else None
                    
                    episode_data = {
                        "prefix_token_ids": prefix_token_ids,
                        "generated_token_ids": generated_token_ids,
                        "reward": reward_result,
                        "completion": result,
                        "model_name": name,
                        "model_idx": model_idx,
                        "old_log_probs": old_lp,
                    }
                    
                    model_episodes.append(episode_data)
                    model_rewards.append(reward_result)
                    
          
                    if is_multi_model_mode:
                        all_episodes.append(episode_data)

                model_episode_groups.append(model_episodes)
                mean_reward = np.mean(model_rewards)
                model_mean_rewards.append(mean_reward)
                print(f"    {name}: Mean reward: {mean_reward:.3f}")
                
                del rollout_result
                del model_rewards
                torch.cuda.empty_cache()
                gc.collect()
            
            # === Compute Ranking Weights ===
            ranking_weights = [1.0] * num_models  # 默认权重
            if is_multi_model_mode and len(model_mean_rewards) == num_models:
                ranking_weights = compute_individual_weights(
                    model_mean_rewards, 
                    k=individual_k, 
                    threshold=individual_threshold
                )
                
                # 显示排名和权重信息
                print(f"\n=== Step {step} Ranking Weights ===")
                sorted_indices = np.argsort(model_mean_rewards)[::-1]  # 降序
                for i, (name, reward, weight) in enumerate(zip(model_names, model_mean_rewards, ranking_weights)):
                    rank = np.where(sorted_indices == i)[0][0] + 1
                    status = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "🏅"
                    print(f"  {status} {name}: reward={reward:.3f}, rank={rank}/{num_models}, weight={weight:.3f}")
                print()

            # === Policy Update Phase ===
            for model_idx, (model, optimizer, tokenizer, name, device) in enumerate(zip(models, optimizers, tokenizers, model_names, target_devices)):
                # 应用动态学习率
                current_weight = ranking_weights[model_idx]
                dynamic_lr = base_learning_rate * current_weight
                for param_group in optimizer.param_groups:
                    param_group['lr'] = dynamic_lr
                
                print(f"  {name} ({device}): Updating policy (lr={dynamic_lr:.6f}, weight={current_weight:.3f})...")
                
                final_episodes = []
                
                if is_multi_model_mode:

                    own_prefix = tokenizer.apply_chat_template(
                        [{"role": "user", "content": question["prompt"]}],
                        tokenize=True,
                        add_generation_prompt=True,
                        return_tensors="pt"
                    )[0].tolist()
                    
                    for ep in all_episodes:
                        if ep["model_idx"] == model_idx:

                            final_episodes.append({
                                "prefix_token_ids": ep["prefix_token_ids"], 
                                "generated_token_ids": ep["generated_token_ids"],
                                "reward": ep["reward"],
                                "old_log_probs": ep.get("old_log_probs"),
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
                                    final_episodes.append({
                                        "prefix_token_ids": own_prefix, 
                                        "generated_token_ids": new_completion_ids,
                                        "reward": ep["reward"],
                                    })
                                else:
                                    print(f"    Skipping episode: completion tokens out of range")
                                    
                            except Exception as e:
                                print(f"    Warning: Failed to retokenize completion: {e}")
                                continue
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
                
                # Ensure episodes are properly grouped for advantage calculation
                if len(final_episodes) % num_completions_per_prompt != 0:
                    print(f"    Warning: {name} has {len(final_episodes)} episodes, not divisible by {num_completions_per_prompt}")
                    # Truncate to make it divisible
                    final_episodes = final_episodes[:len(final_episodes) - (len(final_episodes) % num_completions_per_prompt)]
                
                if len(final_episodes) == 0:
                    print(f"    Warning: {name} has no valid episodes after grouping, skipping...")
                    continue
                
                print(f"    {name}: Using {len(final_episodes)} episodes for policy update")
                

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
                    ppo_mini_batch_size=config["training"].get("ppo_mini_batch_size", 0),
                    ppo_micro_batch_size=config["training"].get("ppo_micro_batch_size", 4),
                )
                
                # Store current log probs for next iteration (PPO clipping)
                if update_result.get("current_log_probs") is not None:
                    old_log_probs_storage[model_idx] = update_result["current_log_probs"]
                
                print(f"    {name}: Updated policy - Loss: {update_result['loss']:.4f}, "
                      f"Grad norm: {update_result['grad_norm']:.4f}, "
                      f"Weight: {current_weight:.3f}, "
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
            del model_mean_rewards
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
