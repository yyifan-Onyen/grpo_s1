import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import math
import copy


class SharedFacTLinear(nn.Module):
    """
    FacT adapter for a Linear layer with shared U/V across same-dim modules
    and a per-layer T. Follows PETL-ViT style:
      - U: in_features -> rank (shared)
      - T: rank -> rank  (per-layer)
      - V: rank -> out_features (shared, init to zeros)
    Output: original(x) + s * (alpha / rank) * V(T(U(x))) with dropout around T.
    """

    def __init__(
        self,
        original_layer: nn.Linear,
        shared_fact_u: nn.Linear,
        shared_fact_v: nn.Linear,
        rank: int = 64,
        alpha: int = 32,
        dropout: float = 0.1,
        scale: float = 1.0,
    ):
        super().__init__()
        self.original_layer = original_layer
        self.rank = int(rank)
        self.alpha = int(alpha)
        self.scale = float(scale)

        # Device / dtype from original layer
        device = next(original_layer.parameters()).device
        dtype = next(original_layer.parameters()).dtype

        # Shared U/V, per-layer T
        self.fact_u = shared_fact_u  # in_features -> rank
        self.fact_v = shared_fact_v  # rank -> out_features (zeros init)
        self.fact_t = nn.Linear(self.rank, self.rank, bias=False, dtype=dtype)

        # Init: PETL-ViT uses zeros for V; for T we use identity
        nn.init.eye_(self.fact_t.weight)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Move to device
        self.fact_t = self.fact_t.to(device)
        self.dropout = self.dropout.to(device)

        # Freeze original weights
        for param in self.original_layer.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_out = self.original_layer(x)

        # U -> T -> V pathway
        u = self.fact_u(x)
        u = self.dropout(u)
        t = self.fact_t(u)
        t = self.dropout(t)
        v = self.fact_v(t)

        # Scale with alpha/rank like LoRA s
        scaling = self.scale * (self.alpha / max(1.0, float(self.rank)))
        return original_out + v * scaling


def apply_shared_fact_to_model(
    model: nn.Module,
    fact_config: Dict,
    target_modules: Optional[list] = None,
) -> nn.Module:
    """
    Apply FacT to specified Linear submodules following PETL-ViT approach.

    - For each unique (in_features, out_features) pair among target modules,
      create shared U and V.
    - Replace each target Linear with a wrapper that adds V(T(U(x))) to output.
    - V weights are zero-initialized; U is Kaiming/Xavier; T is identity.
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    rank = int(fact_config.get("fact_rank", 64))
    alpha = int(fact_config.get("fact_alpha", 32))
    dropout = float(fact_config.get("fact_dropout", 0.1))
    scale = float(fact_config.get("fact_scale", 1.0))

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    print(f"Applying shared FacT to model on device: {device}, dtype: {dtype}")

    # Group Linear modules by dimensionality
    dimension_groups: Dict[tuple[int, int], list[tuple[str, nn.Linear]]] = {}
    for name, module in model.named_modules():
        if any(target in name for target in target_modules) and isinstance(module, nn.Linear):
            dim_key = (module.in_features, module.out_features)
            dimension_groups.setdefault(dim_key, []).append((name, module))

    print(f"Found {len(dimension_groups)} dimension groups:")
    for (in_dim, out_dim), modules in dimension_groups.items():
        print(f"  {in_dim}x{out_dim}: {len(modules)} modules")

    # Create shared U/V per dimension group
    shared_components: Dict[tuple[int, int], tuple[nn.Linear, nn.Linear]] = {}
    for (in_features, out_features), _ in dimension_groups.items():
        fact_u = nn.Linear(in_features, rank, bias=False, dtype=dtype)
        fact_v = nn.Linear(rank, out_features, bias=False, dtype=dtype)

        # Init: U kaiming/xavier, V zeros (as in PETL-ViT)
        nn.init.kaiming_uniform_(fact_u.weight, a=math.sqrt(5))
        nn.init.zeros_(fact_v.weight)

        shared_components[(in_features, out_features)] = (
            fact_u.to(device),
            fact_v.to(device),
        )

    module_dict = dict(model.named_modules())
    replaced_count = 0

    # Replace target Linear layers with FacT wrappers
    for name, module in model.named_modules():
        if any(target in name for target in target_modules) and isinstance(module, nn.Linear):
            module_device = next(module.parameters()).device
            dim_key = (module.in_features, module.out_features)
            shared_u, shared_v = shared_components[dim_key]

            fact_wrapper = SharedFacTLinear(
                original_layer=module,
                shared_fact_u=shared_u,
                shared_fact_v=shared_v,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                scale=scale,
            ).to(module_device)

            # Set back into parent
            parts = name.split('.')
            if len(parts) > 1:
                parent_name = '.'.join(parts[:-1])
                child_name = parts[-1]
                parent = module_dict[parent_name]
                setattr(parent, child_name, fact_wrapper)
            else:
                setattr(model, name, fact_wrapper)

            replaced_count += 1

    print(f"Total shared FacT adapters applied: {replaced_count}")
    print(f"Shared component groups: {len(shared_components)}")
    for (in_dim, out_dim) in shared_components.keys():
        print(f"  Shared components for {in_dim}x{out_dim}")
    return model


def _merge_shared_fact_linear_to_dense(module: "SharedFacTLinear") -> nn.Linear:
    """
    Merge a SharedFacTLinear wrapper into a plain nn.Linear with updated weights.

    Computes W_new = W_orig + scale * (alpha/rank) * V @ T @ U, preserves bias.
    """
    assert isinstance(module, SharedFacTLinear), "Expected SharedFacTLinear"

    original = module.original_layer
    in_features = original.in_features
    out_features = original.out_features
    has_bias = original.bias is not None

    device = next(original.parameters()).device
    dtype = original.weight.dtype

    # Shapes: U[r,in], T[r,r], V[out,r]
    U = module.fact_u.weight.detach().to(device=device, dtype=dtype)
    T = module.fact_t.weight.detach().to(device=device, dtype=dtype)
    V = module.fact_v.weight.detach().to(device=device, dtype=dtype)

    W_orig = original.weight.detach().to(device=device, dtype=dtype)  # [out,in]
    scaling = float(module.scale) * (float(module.alpha) / max(1.0, float(module.rank)))
    # W_add = V @ T @ U
    W_add = (V @ T @ U) * scaling
    W_new = W_orig + W_add

    merged = nn.Linear(in_features, out_features, bias=has_bias, dtype=dtype).to(device)
    with torch.no_grad():
        merged.weight.copy_(W_new)
        if has_bias:
            merged.bias.copy_(original.bias.detach().to(device=device, dtype=dtype))
    return merged


def merge_fact_adapters_to_dense_copy(model: nn.Module) -> nn.Module:
    """
    Return a deep-copied model where all SharedFacTLinear modules
    are replaced by merged nn.Linear layers containing W_orig + V T U.
    """
    new_model = copy.deepcopy(model)
    module_dict = dict(new_model.named_modules())
    replacements = []
    for name, module in new_model.named_modules():
        if isinstance(module, SharedFacTLinear):
            merged = _merge_shared_fact_linear_to_dense(module)
            replacements.append((name, merged))
    for name, merged in replacements:
        parts = name.split('.')
        if len(parts) > 1:
            parent_name = '.'.join(parts[:-1])
            child_name = parts[-1]
            parent = module_dict[parent_name]
            setattr(parent, child_name, merged)
            module_dict[name] = merged
        else:
            setattr(new_model, name, merged)
            module_dict[name] = merged
    if len(replacements) == 0:
        print("[FacT] No SharedFacTLinear modules found to merge; returning a copy unchanged.")
    else:
        print(f"[FacT] Merged {len(replacements)} FacT-wrapped Linear layers into dense weights.")
    return new_model


def get_shared_fact_parameters(model: nn.Module) -> list:
    """
    Get all shared FacT parameters from the model.
    
    Args:
        model: Model with shared FacT applied
        
    Returns:
        List of shared FacT parameters
    """
    fact_params = []
    for name, param in model.named_parameters():
        if 'fact_' in name:
            fact_params.append(param)
    return fact_params


def count_shared_fact_parameters(model: nn.Module) -> int:
    """
    Count the number of shared FacT parameters.
    
    Args:
        model: Model with shared FacT applied
        
    Returns:
        Number of shared FacT parameters
    """
    fact_params = get_shared_fact_parameters(model)
    return sum(p.numel() for p in fact_params)


def analyze_shared_fact_parameters(model: nn.Module) -> Dict:
    """
    Analyze shared FacT parameters in detail.
    
    Args:
        model: Model with shared FacT applied
        
    Returns:
        Dictionary with parameter analysis
    """
    shared_u_params = 0
    shared_v_params = 0
    layer_t_params = 0
    total_fact_params = 0
    
    for name, param in model.named_parameters():
        if 'fact_' in name:
            total_fact_params += param.numel()
            if 'fact_u' in name:
                shared_u_params += param.numel()
            elif 'fact_v' in name:
                shared_v_params += param.numel()
            elif 'fact_t' in name:
                layer_t_params += param.numel()
    
    return {
        'total_fact_params': total_fact_params,
        'shared_u_params': shared_u_params,
        'shared_v_params': shared_v_params,
        'layer_t_params': layer_t_params,
        'shared_ratio': (shared_u_params + shared_v_params) / total_fact_params if total_fact_params > 0 else 0
    }


def apply_fact_to_model(model: nn.Module, fact_config: Dict, target_modules: Optional[list] = None) -> nn.Module:
    """Backward compatibility alias for shared FacT."""
    return apply_shared_fact_to_model(model, fact_config, target_modules)


def get_fact_parameters(model: nn.Module) -> list:
    """Backward compatibility alias."""
    return get_shared_fact_parameters(model)


def count_fact_parameters(model: nn.Module) -> int:
    """Backward compatibility alias."""
    return count_shared_fact_parameters(model)


def analyze_all_trainable_parameters(model: nn.Module) -> Dict:
    """
    Analyze all trainable parameters in the model to understand parameter distribution.
    
    Args:
        model: Model with FacT applied
        
    Returns:
        Dictionary with detailed parameter analysis
    """
    fact_params = 0
    embedding_params = 0
    lm_head_params = 0
    other_params = 0
    total_trainable = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            total_trainable += param.numel()
            
            if 'fact_' in name:
                fact_params += param.numel()
            elif 'embed' in name.lower():
                embedding_params += param.numel()
            elif 'lm_head' in name.lower() or 'output' in name.lower():
                lm_head_params += param.numel()
            else:
                other_params += param.numel()
    
    return {
        'total_trainable': total_trainable,
        'fact_params': fact_params,
        'embedding_params': embedding_params,
        'lm_head_params': lm_head_params,
        'other_params': other_params,
        'fact_ratio': fact_params / total_trainable if total_trainable > 0 else 0
    }


def freeze_non_fact_parameters(model: nn.Module) -> None:
    """
    Freeze all parameters in the model except FacT parameters.
    
    Args:
        model: Model with FacT applied
    """
    frozen_count = 0
    trainable_count = 0
    
    for name, param in model.named_parameters():
        if 'fact_' in name:
            # Keep FacT parameters trainable
            param.requires_grad = True
            trainable_count += param.numel()
        else:
            # Freeze all other parameters
            param.requires_grad = False
            frozen_count += param.numel()
    
    print(f"Frozen {frozen_count:,} non-FacT parameters")
    print(f"Kept {trainable_count:,} FacT parameters trainable")
    print(f"Total parameters: {frozen_count + trainable_count:,}")


if __name__ == "__main__":
    # Test the FacT adapter functions
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print("Loading Qwen2.5-3B model...")
    model_path = "Qwen/Qwen2.5-3B-Instruct"
    
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    print(f"Original model loaded on device: {next(model.parameters()).device}")
    print(f"Original model dtype: {next(model.parameters()).dtype}")
    
    # Count original parameters
    original_params = sum(p.numel() for p in model.parameters())
    print(f"Original model parameters: {original_params:,}")
    
    # Configure FacT
    fact_config = {
        "fact_rank": 64,
        "fact_alpha": 32,
        "fact_dropout": 0.1,
        "fact_scale": 1.0
    }
    
    target_modules = ["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    print(f"\nApplying FacT with config: {fact_config}")
    print(f"Target modules: {target_modules}")
    
    # Apply FacT
    model = apply_fact_to_model(model, fact_config, target_modules)
    
    # Analyze FacT parameters
    print("\n=== FacT Parameter Analysis ===")
    fact_params = count_fact_parameters(model)
    total_params = sum(p.numel() for p in model.parameters())
    fact_analysis = analyze_shared_fact_parameters(model)
    
    print(f"Total FacT parameters: {fact_params:,}")
    print(f"  - Shared U parameters: {fact_analysis['shared_u_params']:,}")
    print(f"  - Shared V parameters: {fact_analysis['shared_v_params']:,}")
    print(f"  - Layer-specific T parameters: {fact_analysis['layer_t_params']:,}")
    print(f"  - Total shared parameters: {fact_analysis['shared_u_params'] + fact_analysis['shared_v_params']:,}")
    print(f"  - Shared ratio: {fact_analysis['shared_ratio']*100:.1f}%")
    print(f"Total model parameters: {total_params:,}")
    print(f"FacT ratio: {fact_params/total_params*100:.2f}%")
    
    # Freeze non-FacT parameters
    print("\n=== Freezing non-FacT parameters ===")
    freeze_non_fact_parameters(model)
    
    # Analyze trainable parameters
    print("\n=== Trainable Parameter Analysis ===")
    trainable_analysis = analyze_all_trainable_parameters(model)
    print(f"Total trainable parameters: {trainable_analysis['total_trainable']:,}")
    print(f"  - FacT parameters: {trainable_analysis['fact_params']:,}")
    print(f"  - Embedding parameters: {trainable_analysis['embedding_params']:,}")
    print(f"  - LM head parameters: {trainable_analysis['lm_head_params']:,}")
    print(f"  - Other parameters: {trainable_analysis['other_params']:,}")
    print(f"  - FacT ratio in trainable: {trainable_analysis['fact_ratio']*100:.2f}%")
    
    print("\n=== Test completed successfully! ===")
