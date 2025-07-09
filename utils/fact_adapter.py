import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
import math

class SharedFacTLinear(nn.Module):
    """
    Linear layer with shared FacT adaptation.
    """
    
    def __init__(
        self,
        original_layer: nn.Linear,
        shared_fact_u: nn.Linear,
        shared_fact_v: nn.Linear,
        rank: int = 64,
        alpha: int = 32,
        dropout: float = 0.1,
        scale: float = 1.0
    ):
        super().__init__()
        self.original_layer = original_layer
        
        # Get the device and dtype of the original layer
        device = next(original_layer.parameters()).device
        dtype = next(original_layer.parameters()).dtype
        
        # Use shared fact_u and fact_v, create layer-specific fact_t
        self.fact_u = shared_fact_u
        self.fact_v = shared_fact_v
        self.fact_t = nn.Linear(rank, rank, bias=False, dtype=dtype)
        
        # Initialize fact_t
        nn.init.eye_(self.fact_t.weight)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        self.scale = scale
        
        # Move fact_t to the correct device
        self.fact_t = self.fact_t.to(device)
        self.dropout = self.dropout.to(device)
        
        # Freeze original weights
        for param in self.original_layer.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: original + shared FacT adaptation.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        original_out = self.original_layer(x)
        
        # Apply shared FacT: x -> U -> T -> V
        u_out = self.fact_u(x)
        u_out = self.dropout(u_out)
        
        t_out = self.fact_t(u_out)
        t_out = self.dropout(t_out)
        
        v_out = self.fact_v(t_out)
        fact_out = v_out * self.scale
        
        return original_out + fact_out


def apply_shared_fact_to_model(
    model: nn.Module,
    fact_config: Dict,
    target_modules: Optional[list] = None
) -> nn.Module:
    """
    Apply shared FacT adaptation to a model.
    
    Args:
        model: The model to apply FacT to
        fact_config: FacT configuration dictionary
        target_modules: List of module names to apply FacT to
        
    Returns:
        Model with shared FacT applied
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    
    rank = fact_config.get("fact_rank", 64)
    alpha = fact_config.get("fact_alpha", 32)
    dropout = fact_config.get("fact_dropout", 0.1)
    scale = fact_config.get("fact_scale", 1.0)
    
    # Get the device and dtype of the model
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    print(f"Applying shared FacT to model on device: {device}, dtype: {dtype}")
    
    # Group modules by their dimensions to create shared components
    dimension_groups = {}
    
    for name, module in model.named_modules():
        if any(target in name for target in target_modules) and isinstance(module, nn.Linear):
            dim_key = (module.in_features, module.out_features)
            if dim_key not in dimension_groups:
                dimension_groups[dim_key] = []
            dimension_groups[dim_key].append((name, module))
    
    print(f"Found {len(dimension_groups)} dimension groups:")
    for (in_dim, out_dim), modules in dimension_groups.items():
        print(f"  {in_dim}x{out_dim}: {len(modules)} modules")
    
    # Create shared components for each dimension group
    shared_components = {}
    
    for (in_features, out_features), modules in dimension_groups.items():
        # Create shared fact_u and fact_v for this dimension group
        shared_fact_u = nn.Linear(in_features, rank, bias=False, dtype=dtype)
        shared_fact_v = nn.Linear(rank, out_features, bias=False, dtype=dtype)
        
        # Initialize shared components
        nn.init.kaiming_uniform_(shared_fact_u.weight, a=math.sqrt(5))
        nn.init.zeros_(shared_fact_v.weight)
        
        # Move shared components to device
        shared_fact_u = shared_fact_u.to(device)
        shared_fact_v = shared_fact_v.to(device)
        
        shared_components[(in_features, out_features)] = (shared_fact_u, shared_fact_v)
    
    # Create a mapping of module names to modules for easier replacement
    module_dict = dict(model.named_modules())
    
    replaced_count = 0
    # Replace target modules with shared FacT versions
    for name, module in model.named_modules():
        if any(target in name for target in target_modules) and isinstance(module, nn.Linear):
            # Get the device and dtype of the current module
            module_device = next(module.parameters()).device
            module_dtype = next(module.parameters()).dtype
            
            # Get the shared components for this module's dimensions
            dim_key = (module.in_features, module.out_features)
            shared_fact_u, shared_fact_v = shared_components[dim_key]
            
            # Create shared FacT wrapper
            fact_module = SharedFacTLinear(
                original_layer=module,
                shared_fact_u=shared_fact_u,
                shared_fact_v=shared_fact_v,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                scale=scale
            )
            
            # Ensure FacT module is on the correct device
            fact_module = fact_module.to(module_device)
            
            # Get the parent module and child name
            name_parts = name.split('.')
            if len(name_parts) > 1:
                parent_name = '.'.join(name_parts[:-1])
                child_name = name_parts[-1]
                parent = module_dict[parent_name]
                setattr(parent, child_name, fact_module)
            else:
                # Root level module
                setattr(model, name, fact_module)
            
            replaced_count += 1
            # print(f"Replaced {name} with shared FacT adapter on device {module_device}, dtype {module_dtype}")
    
    print(f"Total shared FacT adapters applied: {replaced_count}")
    print(f"Shared component groups: {len(shared_components)}")
    for (in_dim, out_dim) in shared_components.keys():
        print(f"  Shared components for {in_dim}x{out_dim}")
    return model


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
