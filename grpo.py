import dataclasses
import gc
import math
import random
from collections import defaultdict
from typing import Callable, List, Dict, Any, Optional
import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def rollout(
    model,
    question,
    max_gen_len: int,
    num_answer_per_question: int,
    temperature: float = 1.0,
) -> Dict[str, Any]:
    prompts = [question]*num_answer_per_question
    with torch.no_grad():
        _ret = model.generate(
            prompts=prompts,
            limitation=max_gen_len,
            temperature=temperature,
            verbose=True,
            return_log_probs=True,
        )
        # Backward compatible unpacking (support 3-tuple from older generate)
        if isinstance(_ret, tuple) and len(_ret) >= 3:
            completions, indices, masks = _ret[0], _ret[1], _ret[2]
            vllm_lps = _ret[3] if len(_ret) >= 4 else None
        else:
            completions, indices, masks = _ret
            vllm_lps = None
        # compute per-sample generated-token log_probs as rollout-time old policy
        # Prefer vLLM-provided logprobs if available
        if isinstance(vllm_lps, list) and any(lp is not None for lp in vllm_lps):
            per_sample_old_log_probs = []
            for i in range(len(completions)):
                try:
                    lp_i = vllm_lps[i]
                    gen_mask_i = masks[i, 1:]  # align with target_ids
                    if lp_i is not None:
                        # lp_i already contains generated-token logprobs length == num_generated
                        # mask True positions correspond to generated tokens
                        per_sample_old_log_probs.append(lp_i)
                    else:
                        per_sample_old_log_probs.append(None)
                except Exception:
                    per_sample_old_log_probs.append(None)
        else:
            try:
                pad_token_id = model.tokenizer.pad_token_id
                vocab_size = model.model.config.vocab_size
                input_ids = indices[:, :-1]
                target_ids = indices[:, 1:]
                target_masks = masks[:, 1:]

                log_probs_all = get_log_probs(
                    model.model,
                    input_ids,
                    target_ids,
                    target_masks,
                    pad_token_id,
                    vocab_size,
                    return_logits=False,
                )
                per_sample_old_log_probs = []
                for i in range(log_probs_all.size(0)):
                    gen_mask_i = target_masks[i]
                    per_sample_old_log_probs.append(log_probs_all[i][gen_mask_i])
            except Exception:
                per_sample_old_log_probs = None

        return {
            "completions": completions,
            "indices": indices.detach(),
            "masks": masks.detach(),
            "log_probs": per_sample_old_log_probs,
        }


def compute_advantages_groupwise(
    episodes: List[Dict],
    num_completions_per_prompt: int,
    normalize_by_std: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    rewards = torch.tensor([ep["reward"] for ep in episodes], dtype=torch.float32)

    # Reshape to [num_prompts, num_completions_per_prompt]
    num_prompts = len(rewards) // num_completions_per_prompt
    rewards_grouped = rewards.view(num_prompts, num_completions_per_prompt)

    group_means = rewards_grouped.mean(dim=1, keepdim=True)
    if normalize_by_std:
        group_stds = rewards_grouped.std(dim=1, keepdim=True)
        denom = torch.where(group_stds < eps, torch.ones_like(group_stds), group_stds)
        advantages_grouped = (rewards_grouped - group_means) / denom
    else:
        advantages_grouped = rewards_grouped - group_means

    return advantages_grouped.view(-1)


def get_log_probs(model, input_ids, target_ids, target_masks, pad_token_id, vocab_size, return_logits: bool = False):
    # Validate input_ids
    max_token_id = input_ids.max().item()
    if max_token_id >= vocab_size:
        print(f"Warning: Found token ID {max_token_id} >= vocab_size {vocab_size}")
        input_ids = torch.clamp(input_ids, 0, vocab_size - 1)
    
    # Build attention mask and disable KV cache to reduce peak memory
    attention_mask = input_ids != (pad_token_id if pad_token_id < vocab_size else 0)
    # Some models need position_ids when passing attention_mask explicitly
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids[attention_mask == 0] = 0
    logits = model.forward(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
    ).logits
    
    safe_pad_token_id = pad_token_id if pad_token_id < vocab_size else 0
    
    # Compute CE in fp32 for numerical stability (bf16 can amplify errors in softmax/log)
    log_probs = -F.cross_entropy(
        logits.float().reshape(-1, logits.size(-1)),
        target_ids.reshape(-1),
        ignore_index=safe_pad_token_id,
        reduction="none",
    ).reshape(target_ids.shape)
    
    if return_logits:
        return log_probs, logits
    # 默认不返回 logits（减小峰值显存占用）
    del logits
    # 避免在训练热路径调用 empty_cache 以防稀疏非法访问和性能抖动
    return log_probs


def compute_kl_divergence(current_log_probs, ref_log_probs, target_masks):
    log_ratio = current_log_probs - ref_log_probs
    log_ratio = torch.clamp(log_ratio, min=-20.0, max=20.0)
    per_token_kl = torch.exp(log_ratio) - 1.0 - log_ratio
    return per_token_kl


def update_policy(
    model,
    optimizer,
    episodes: List[Dict],
    pad_token_id: int,
    max_grad_norm: float,
    device: torch.device,
    dtype: torch.dtype,
    num_completions_per_prompt: int = 4,
    ref_model: Optional[torch.nn.Module] = None,
    old_log_probs: Optional[torch.Tensor] = None,
    epsilon_low: float = 0.2,
    epsilon_high: float = 0.2,
    beta: float = 0.01,  # KL regularization coefficient
    loss_type: str = "grpo",  # "grpo", "bnpo", or "dr_grpo"
    normalize_adv_by_std: bool = True,
    adv_epsilon: float = 1e-6,
    ppo_mini_batch_size: int = 0,
    ppo_micro_batch_size: int = 4,
    ppo_epochs: int = 1,
    loss_aggregation: str = "token-mean",
    advantage_clip: Optional[float] = None,
    shuffle_episodes: bool = True,
):
    if not episodes:
        return {
            "loss": 0.0,
            "grad_norm": 0.0,
            "entropy": 0.0,
            "kl_div": 0.0,
            "clip_ratio": 0.0,
        }

    if loss_aggregation not in {"token-mean", "episode-mean"}:
        raise ValueError(f"Unsupported loss aggregation: {loss_aggregation}")

    vocab_size = model.config.vocab_size

    total_episode_count = len(episodes)
    total_token_count = sum(len(ep.get("generated_token_ids", [])) for ep in episodes)
    total_token_count = max(total_token_count, 1)

    # Compute group-wise advantages
    advantages = compute_advantages_groupwise(
        episodes,
        num_completions_per_prompt,
        normalize_by_std=normalize_adv_by_std,
        eps=adv_epsilon,
    ).to(device)

    if advantage_clip is not None and advantage_clip > 0:
        advantages = torch.clamp(advantages, -float(advantage_clip), float(advantage_clip))

    avg_entropy_num = torch.tensor(0.0, device=device)
    avg_entropy_den = torch.tensor(0.0, device=device)
    clip_ratio_num = torch.tensor(0.0, device=device)
    clip_ratio_den = torch.tensor(0.0, device=device)
    kl_div_accum = 0.0
    kl_div_batches = 0

    token_loss_accum = 0.0
    token_loss_den = 0.0
    episode_loss_accum = 0.0
    episode_loss_den = 0.0

    # Diagnostics accumulators
    oldlp_cov_num = 0.0
    oldlp_cov_den = 0.0
    ratio_sum = 0.0
    ratio_sq_sum = 0.0
    ratio_den = 0.0
    adv_abs_sum = 0.0
    adv_abs_den = 0.0

    if ppo_mini_batch_size and ppo_mini_batch_size > 0:
        mini_batch_size = min(ppo_mini_batch_size, total_episode_count)
    else:
        mini_batch_size = total_episode_count

    micro_batch_size = max(1, ppo_micro_batch_size)
    ppo_epochs = max(1, int(ppo_epochs))

    episode_indices = list(range(total_episode_count))

    optimizer.zero_grad()

    for epoch_idx in range(ppo_epochs):
        if shuffle_episodes and total_episode_count > mini_batch_size:
            random.shuffle(episode_indices)

        for mini_start in range(0, total_episode_count, mini_batch_size):
            mini_end = min(mini_start + mini_batch_size, total_episode_count)
            mini_slice = episode_indices[mini_start:mini_end]
            if not mini_slice:
                continue
            mini_eps = [episodes[idx] for idx in mini_slice]
            mini_indices_tensor = torch.tensor(mini_slice, device=advantages.device, dtype=torch.long)
            mini_advantages = advantages[mini_indices_tensor]

            mini_length = len(mini_eps)
            for micro_start in range(0, mini_length, micro_batch_size):
                micro_end = min(micro_start + micro_batch_size, mini_length)
                batch_eps = mini_eps[micro_start:micro_end]
                if not batch_eps:
                    continue
                batch_advantages = mini_advantages[micro_start:micro_end]
                chunk_size = micro_end - micro_start

                # Prepare sequences and masks for this micro-batch
                batch_sequences = []
                batch_masks = []
                for episode in batch_eps:
                    full_sequence = episode["prefix_token_ids"] + episode["generated_token_ids"]
                    valid_sequence = []
                    for token_id in full_sequence:
                        if token_id < vocab_size:
                            valid_sequence.append(token_id)
                        else:
                            print(f"Warning: Invalid token ID {token_id} (>= {vocab_size}), replacing with pad token {pad_token_id}")
                            valid_sequence.append(pad_token_id)
                    batch_sequences.append(valid_sequence)
                    prefix_len = len(episode["prefix_token_ids"])
                    generated_len = len(episode["generated_token_ids"])
                    mask = [False] * prefix_len + [True] * generated_len
                    batch_masks.append(mask)

                max_length = max(len(seq) for seq in batch_sequences)
                padded_sequences = []
                padded_masks = []
                for seq, mask in zip(batch_sequences, batch_masks):
                    safe_pad_token_id = pad_token_id if pad_token_id < vocab_size else 0
                    padded_seq = seq + [safe_pad_token_id] * (max_length - len(seq))
                    padded_sequences.append(padded_seq)
                    padded_mask = mask + [False] * (max_length - len(mask))
                    padded_masks.append(padded_mask)

                indices_list = []
                for seq in padded_sequences:
                    validated_seq = [min(max(0, token_id), vocab_size - 1) for token_id in seq]
                    indices_list.append(validated_seq)

                indices = torch.tensor(indices_list, device=device, dtype=torch.long)
                masks = torch.tensor(padded_masks, device=device, dtype=torch.bool)

                with torch.autocast(device_type=device.type, dtype=dtype):
                    input_ids = indices[:, :-1]
                    target_ids = indices[:, 1:]
                    target_masks = masks[:, 1:]

                    current_log_probs = get_log_probs(
                        model,
                        input_ids,
                        target_ids,
                        target_masks,
                        pad_token_id,
                        vocab_size,
                        return_logits=False,
                    )

                    # 使用已计算的目标token对数概率作为轻量级熵代理，避免对整个词表做softmax导致显存激增
                    # 说明：真实分布熵需要对 vocab 维度做 softmax；这里仅做日志统计，不参与反传，
                    # 使用 -log p(token) 的蒙版平均值作为代理，显著降低内存占用。
                    with torch.no_grad():
                        entropy_proxy = -current_log_probs  # shape [B, T]
                        avg_entropy_num += (entropy_proxy * target_masks).sum()
                        avg_entropy_den += target_masks.sum()

                    # logits未返回，无需额外释放

                    # Reference log probs
                    ref_log_probs = None
                    if ref_model is not None and beta > 0:
                        # Use inference_mode for potential allocator/memory savings.
                        with torch.inference_mode():
                            ref_log_probs = get_log_probs(
                                ref_model,
                                input_ids,
                                target_ids,
                                target_masks,
                                pad_token_id,
                                vocab_size,
                                return_logits=False,
                            )

                    # Build old log probs tensor aligned with target_masks; default to current (on-policy no-op)
                    old_log_probs_tensor = current_log_probs.detach().clone()
                    for b_idx, ep in enumerate(batch_eps):
                        if "old_log_probs" in ep and ep["old_log_probs"] is not None:
                            old_vals = ep["old_log_probs"]
                            if not torch.is_tensor(old_vals):
                                old_vals = torch.tensor(old_vals, device=device, dtype=current_log_probs.dtype)
                            else:
                                old_vals = old_vals.to(device=device, dtype=current_log_probs.dtype)
                            pos = torch.nonzero(target_masks[b_idx], as_tuple=False).squeeze(-1)
                            L = min(len(pos), old_vals.shape[0])
                            if L > 0:
                                old_log_probs_tensor[b_idx, pos[:L]] = old_vals[:L]

                    # PPO ratio with stability clamp
                    log_ratio = current_log_probs - old_log_probs_tensor
                    log_ratio = torch.clamp(log_ratio, min=-20.0, max=20.0)
                    ratio = torch.exp(log_ratio)
                    clipped_ratio = torch.clamp(ratio, 1 - epsilon_low, 1 + epsilon_high)

                    advantages_expanded = batch_advantages.unsqueeze(-1)
                    loss_term1 = ratio * advantages_expanded
                    loss_term2 = clipped_ratio * advantages_expanded
                    per_token_loss = -torch.min(loss_term1, loss_term2)

                    # KL regularization (TRL/VERL-style surrogate)
                    if ref_log_probs is not None and beta > 0:
                        per_token_kl = compute_kl_divergence(current_log_probs, ref_log_probs, target_masks)
                        per_token_loss = per_token_loss + beta * per_token_kl
                        # accumulate average kl (masked)
                        _kl = (per_token_kl * target_masks).sum() / target_masks.sum().clamp(min=1)
                        kl_div_accum += _kl.item()
                        kl_div_batches += 1

                    # Loss reduction (support grpo/bnpo) with configurable aggregation
                    token_sum = (per_token_loss * target_masks).sum()
                    token_count = target_masks.sum().item()
                    token_count = max(1.0, float(token_count))

                    if loss_type == "bnpo":
                        micro_loss = token_sum / target_masks.sum().clamp(min=1.0)
                        scale = token_count / max(1.0, float(total_token_count))
                        (micro_loss * scale).backward()
                        token_loss_accum += token_sum.detach().item()
                        token_loss_den += token_count
                    else:
                        if loss_aggregation == "token-mean":
                            micro_loss = token_sum / target_masks.sum().clamp(min=1.0)
                            scale = token_count / max(1.0, float(total_token_count))
                            (micro_loss * scale).backward()
                            token_loss_accum += token_sum.detach().item()
                            token_loss_den += token_count
                        else:  # episode-mean
                            per_episode_loss = (per_token_loss * target_masks).sum(-1) / target_masks.sum(-1).clamp(min=1.0)
                            micro_loss = per_episode_loss.mean()
                            scale = chunk_size / max(1, total_episode_count)
                            (micro_loss * scale).backward()
                            episode_loss_accum += per_episode_loss.sum().item()
                            episode_loss_den += chunk_size

                    # Clipping统计（熵统计已提前计算并释放logits）
                    with torch.no_grad():
                        is_low_clipped = (ratio < 1 - epsilon_low) & (advantages_expanded < 0)
                        is_high_clipped = (ratio > 1 + epsilon_high) & (advantages_expanded > 0)
                        is_clipped = is_low_clipped | is_high_clipped
                        clip_ratio_num += (is_clipped * target_masks).sum()
                        clip_ratio_den += target_masks.sum()

                        # Diagnostics: advantages (episode level), ratio stats, old_log_prob coverage
                        adv_abs_sum += batch_advantages.abs().sum().item()
                        adv_abs_den += batch_advantages.numel()

                        ratio_sum += (ratio * target_masks).sum().item()
                        ratio_sq_sum += ((ratio ** 2) * target_masks).sum().item()
                        ratio_den += float(target_masks.sum().item())

                        provided_mask = (old_log_probs_tensor - current_log_probs).abs() > 1e-9
                        oldlp_cov_num += float((provided_mask * target_masks).sum().item())
                        oldlp_cov_den += float(target_masks.sum().item())

                    # Do not cache current_log_probs across steps to avoid shape mismatch

                    # cleanup micro-batch
                    del indices, masks, input_ids, target_ids, target_masks, current_log_probs
                    if ref_log_probs is not None:
                        del ref_log_probs

    # Optimizer step after accumulating grads
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    grad_norm_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
    optimizer.step()
    optimizer.zero_grad()

    if loss_type == "bnpo" or loss_aggregation == "token-mean":
        loss_den = token_loss_den if token_loss_den > 0 else 1.0
        loss_val = (token_loss_accum / loss_den) if loss_den > 0 else 0.0
    else:
        loss_den = episode_loss_den if episode_loss_den > 0 else 1.0
        loss_val = (episode_loss_accum / loss_den) if loss_den > 0 else 0.0
    avg_entropy_val = (avg_entropy_num / avg_entropy_den.clamp(min=1)).item() if avg_entropy_den.item() > 0 else 0.0
    clip_ratio_val = (clip_ratio_num / clip_ratio_den.clamp(min=1)).item() if clip_ratio_den.item() > 0 else 0.0
    kl_div_val = (kl_div_accum / max(1, kl_div_batches)) if kl_div_batches > 0 else 0.0

    # 训练末尾不强制清空缓存，交由分配器自行管理

    # Diagnostics summary
    ratio_mean = (ratio_sum / ratio_den) if ratio_den > 0 else 0.0
    ratio_var = max(0.0, (ratio_sq_sum / ratio_den) - ratio_mean * ratio_mean) if ratio_den > 0 else 0.0
    ratio_std = math.sqrt(ratio_var)
    old_lp_cov = (oldlp_cov_num / oldlp_cov_den) if oldlp_cov_den > 0 else 0.0
    adv_abs_mean = (adv_abs_sum / adv_abs_den) if adv_abs_den > 0 else 0.0

    return {
        "loss": loss_val,
        "grad_norm": grad_norm_val,
        "entropy": avg_entropy_val,
        "kl_div": kl_div_val,
        "clip_ratio": clip_ratio_val,
        "current_log_probs": None,
        "metrics": {
            "old_lp_cov": float(old_lp_cov),
            "ratio_mean": float(ratio_mean),
            "ratio_std": float(ratio_std),
            "adv_abs_mean": float(adv_abs_mean),
        },
    }
