import dataclasses
import gc
import math
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
        completions, indices, masks = model.generate(
            prompts=prompts,
            limitation=max_gen_len,
            temperature=temperature,
            verbose=True,
        )
        return {
            "completions": completions,
            "indices": indices.detach(),
            "masks": masks.detach(),
        }


def compute_advantages_groupwise(episodes: List[Dict], num_completions_per_prompt: int) -> torch.Tensor:
    """
    Compute advantages using group-wise normalization like official TRL implementation
    """
    rewards = torch.tensor([ep["reward"] for ep in episodes], dtype=torch.float32)
    
    # Reshape to [num_prompts, num_completions_per_prompt]
    num_prompts = len(rewards) // num_completions_per_prompt
    rewards_grouped = rewards.view(num_prompts, num_completions_per_prompt)
    
    # Compute group-wise advantages (relative to group mean)
    group_means = rewards_grouped.mean(dim=1, keepdim=True)
    group_stds = rewards_grouped.std(dim=1, keepdim=True)
    
    # Normalize within each group
    advantages_grouped = (rewards_grouped - group_means) / (group_stds + 1e-8)
    advantages = advantages_grouped.view(-1)
    
    return advantages


def get_log_probs(model, input_ids, target_ids, target_masks, pad_token_id, vocab_size):
    """
    Compute per-token log probabilities
    """
    # Validate input_ids
    max_token_id = input_ids.max().item()
    if max_token_id >= vocab_size:
        print(f"Warning: Found token ID {max_token_id} >= vocab_size {vocab_size}")
        input_ids = torch.clamp(input_ids, 0, vocab_size - 1)
    
    logits = model.forward(input_ids).logits.float()
    
    safe_pad_token_id = pad_token_id if pad_token_id < vocab_size else 0
    
    log_probs = -F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        target_ids.reshape(-1),
        ignore_index=safe_pad_token_id,
        reduction="none",
    ).reshape(target_ids.shape)
    
    return log_probs, logits


def compute_kl_divergence(current_log_probs, ref_log_probs, target_masks):
    """
    Compute KL divergence between current and reference policies
    """
    # KL(ref || current) = exp(ref_log_probs - current_log_probs) - (ref_log_probs - current_log_probs) - 1
    per_token_kl = torch.exp(ref_log_probs - current_log_probs) - (ref_log_probs - current_log_probs) - 1
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
):
    """
    Update policy using improved GRPO logic with PPO clipping and group-wise advantages
    """
    if not episodes:
        return {"loss": 0.0, "grad_norm": 0.0, "entropy": 0.0, "kl_div": 0.0, "clip_ratio": 0.0}
    
    vocab_size = model.config.vocab_size
    
    # Compute group-wise advantages
    advantages = compute_advantages_groupwise(episodes, num_completions_per_prompt).to(device)
    
    # Prepare sequences and masks
    all_sequences = []
    all_masks = []
    for episode in episodes:
        full_sequence = episode["prefix_token_ids"] + episode["generated_token_ids"]
        valid_sequence = []
        for token_id in full_sequence:
            if token_id < vocab_size:
                valid_sequence.append(token_id)
            else:
                print(f"Warning: Invalid token ID {token_id} (>= {vocab_size}), replacing with pad token {pad_token_id}")
                valid_sequence.append(pad_token_id)
        all_sequences.append(valid_sequence)
        prefix_len = len(episode["prefix_token_ids"])
        generated_len = len(episode["generated_token_ids"])
        mask = [False] * prefix_len + [True] * generated_len
        all_masks.append(mask)
    
    # Pad sequences
    max_length = max(len(seq) for seq in all_sequences)
    padded_sequences = []
    padded_masks = []
    
    for seq, mask in zip(all_sequences, all_masks):
        safe_pad_token_id = pad_token_id if pad_token_id < vocab_size else 0
        padded_seq = seq + [safe_pad_token_id] * (max_length - len(seq))
        padded_sequences.append(padded_seq)
        padded_mask = mask + [False] * (max_length - len(mask))
        padded_masks.append(padded_mask)
    
    # Convert to tensors
    indices_list = []
    for seq in padded_sequences:
        validated_seq = [min(max(0, token_id), vocab_size - 1) for token_id in seq]
        indices_list.append(validated_seq)
    
    indices = torch.tensor(indices_list, device=device, dtype=torch.long)
    masks = torch.tensor(padded_masks, device=device, dtype=torch.bool)
    
    # Prepare input and target tensors
    with torch.autocast(device_type=device.type, dtype=dtype):
        input_ids = indices[:, :-1]
        target_ids = indices[:, 1:]
        target_masks = masks[:, 1:]
        
        # Get current log probabilities
        current_log_probs, logits = get_log_probs(model, input_ids, target_ids, target_masks, pad_token_id, vocab_size)
        
        # Get reference log probabilities for KL divergence
        ref_log_probs = None
        if ref_model is not None and beta > 0:
            with torch.no_grad():
                ref_log_probs, _ = get_log_probs(ref_model, input_ids, target_ids, target_masks, pad_token_id, vocab_size)
        
        # Use old log probs if provided, otherwise use current (for first iteration)
        if old_log_probs is None:
            old_log_probs = current_log_probs.detach()
        
        # Compute probability ratios for PPO clipping
        log_ratio = current_log_probs - old_log_probs
        ratio = torch.exp(log_ratio)
        
        # PPO clipping
        clipped_ratio = torch.clamp(ratio, 1 - epsilon_low, 1 + epsilon_high)
        
        # Compute clipped loss terms
        advantages_expanded = advantages.unsqueeze(-1)
        loss_term1 = ratio * advantages_expanded
        loss_term2 = clipped_ratio * advantages_expanded
        per_token_loss = -torch.min(loss_term1, loss_term2)
        
        # Add KL divergence regularization
        kl_div_val = 0.0
        if ref_log_probs is not None and beta > 0:
            per_token_kl = compute_kl_divergence(current_log_probs, ref_log_probs, target_masks)
            per_token_loss = per_token_loss + beta * per_token_kl
            kl_div_val = (per_token_kl * target_masks).sum() / target_masks.sum().clamp(min=1)
            kl_div_val = kl_div_val.item()
        
        # Compute final loss based on loss type
        if loss_type == "grpo":
            loss = ((per_token_loss * target_masks).sum(-1) / target_masks.sum(-1).clamp(min=1.0)).mean()
        elif loss_type == "bnpo":
            loss = (per_token_loss * target_masks).sum() / target_masks.sum().clamp(min=1.0)
        elif loss_type == "dr_grpo":
            max_completion_length = target_masks.size(-1)
            loss = (per_token_loss * target_masks).sum() / (per_token_loss.size(0) * max_completion_length)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        # Compute entropy
        with torch.no_grad():
            probs = torch.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
            avg_entropy = (entropy * target_masks).sum() / target_masks.sum().clamp(min=1)
            avg_entropy_val = avg_entropy.item()
            
            # Compute clipping statistics
            is_low_clipped = (ratio < 1 - epsilon_low) & (advantages_expanded < 0)
            is_high_clipped = (ratio > 1 + epsilon_high) & (advantages_expanded > 0)
            is_clipped = is_low_clipped | is_high_clipped
            clip_ratio = (is_clipped * target_masks).sum() / target_masks.sum().clamp(min=1)
            clip_ratio_val = clip_ratio.item()
    
    # Backward pass
    loss.backward()
    loss_val = loss.item()
    
    # Gradient clipping and optimization step
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    grad_norm_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
    optimizer.step()
    optimizer.zero_grad()
    
    # Memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return {
        "loss": loss_val,
        "grad_norm": grad_norm_val,
        "entropy": avg_entropy_val,
        "kl_div": kl_div_val,
        "clip_ratio": clip_ratio_val,
        "current_log_probs": current_log_probs.detach() if old_log_probs is None else None,  # For next iteration
    }
