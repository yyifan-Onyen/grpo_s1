import dataclasses
import gc
import math
from collections import defaultdict
from typing import Callable, List, Dict, Any
import numpy as np
import torch


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



def update_policy(
    model,
    optimizer,
    episodes: List[Dict],
    pad_token_id: int,
    max_grad_norm: float,
    device: torch.device,
    dtype: torch.dtype,
):
    """Update policy using GRPO logic with proper advantage calculation"""
    if not episodes:
        return {"loss": 0.0, "grad_norm": 0.0, "entropy": 0.0}
    
    vocab_size = model.config.vocab_size
    rewards = torch.tensor([episode["reward"] for episode in episodes], device=device, dtype=torch.float32)
    mean_reward = rewards.mean()
    std_reward = rewards.std()
    advantages = rewards - mean_reward
    if std_reward > 1e-4:
        advantages = advantages / (std_reward + 1e-4)
    del rewards
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
    

    max_length = max(len(seq) for seq in all_sequences)
    padded_sequences = []
    padded_masks = []
    
    for seq, mask in zip(all_sequences, all_masks):
        safe_pad_token_id = pad_token_id if pad_token_id < vocab_size else 0
        padded_seq = seq + [safe_pad_token_id] * (max_length - len(seq))
        padded_sequences.append(padded_seq)
        padded_mask = mask + [False] * (max_length - len(mask))
        padded_masks.append(padded_mask)
    
    del all_sequences, all_masks
    
    indices_list = []
    for seq in padded_sequences:
        validated_seq = [min(max(0, token_id), vocab_size - 1) for token_id in seq]
        indices_list.append(validated_seq)
    
    indices = torch.tensor(indices_list, device=device, dtype=torch.long)
    masks = torch.tensor(padded_masks, device=device, dtype=torch.bool)
    

    del padded_sequences, padded_masks, indices_list
    

    with torch.autocast(device_type=device.type, dtype=dtype):
        input_ids = indices[:, :-1]  # 去掉最后一个token作为输入
        target_ids = indices[:, 1:]  # 去掉第一个token作为目标
        target_masks = masks[:, 1:]  # 对应的mask
        
        # Final validation before forward pass
        max_token_id = input_ids.max().item()
        if max_token_id >= vocab_size:
            print(f"Error: Found token ID {max_token_id} >= vocab_size {vocab_size}")
            input_ids = torch.clamp(input_ids, 0, vocab_size - 1)
        
        logits = model.forward(input_ids).logits.float()
    
    del input_ids
    
    safe_pad_token_id = pad_token_id if pad_token_id < vocab_size else 0
    
    log_probs = -torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        target_ids.reshape(-1),
        ignore_index=safe_pad_token_id,
        reduction="none",
    ).reshape(target_ids.shape)
    
    with torch.no_grad():
        probs = torch.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)
        avg_entropy = (entropy * target_masks).sum() / target_masks.sum().clamp(min=1)
        
        # Store entropy value and clear tensors
        avg_entropy_val = avg_entropy.item()
        del probs, entropy, avg_entropy
    
    del logits
    
    per_token_loss = -log_probs * advantages.unsqueeze(-1) * target_masks
    
    del log_probs, advantages
    
    loss = per_token_loss.sum(-1) / target_masks.sum(-1).clamp(min=1.0)
    loss = loss.mean()
    
    del per_token_loss, target_ids, target_masks, indices, masks
    
    loss.backward()
    
    loss_val = loss.item()
    del loss
    
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    grad_norm_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
    optimizer.step()
    optimizer.zero_grad()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return {
        "loss": loss_val,
        "grad_norm": grad_norm_val,
        "entropy": avg_entropy_val,
    }
