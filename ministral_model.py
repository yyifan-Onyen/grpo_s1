"""
Ministral Model Implementation
Based on Ministral 3B Instruct architecture
Adapted from qwen2_model.py
"""

import math
from pathlib import Path
from typing import Optional, Tuple, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from dataclasses import dataclass


@dataclass
class MinistralConfig:
    attention_dropout: float = 0.0
    bos_token_id: int = 1
    eos_token_id: int = 2
    hidden_act: str = "silu"
    hidden_size: int = 4096
    initializer_range: float = 0.02
    intermediate_size: int = 14336
    max_position_embeddings: int = 131072
    model_type: str = "mistral"
    num_attention_heads: int = 32
    num_hidden_layers: int = 14
    num_key_value_heads: int = 8
    rms_norm_eps: float = 1e-05
    rope_theta: float = 10000.0
    sliding_window: int = 4096
    tie_word_embeddings: bool = False
    torch_dtype: str = "bfloat16"
    use_cache: bool = True
    vocab_size: int = 32000


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        x = self._norm(x).type_as(x)
        x = self.weight * x.to(input_dtype)
        return x


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=2):
    # For GQA (Group Query Attention), q and k may have different numbers of heads
    # We need to handle broadcasting properly
    
    # q shape: [bsz, seqlen, n_q_heads, head_dim]
    # k shape: [bsz, seqlen, n_kv_heads, head_dim]  
    # cos/sin shape: [seqlen, head_dim]
    
    # Instead of unsqueezing, we broadcast cos/sin to match q and k individually
    cos_q = cos[:, None, :].expand(-1, q.size(2), -1)  # [seqlen, n_q_heads, head_dim]
    sin_q = sin[:, None, :].expand(-1, q.size(2), -1)  # [seqlen, n_q_heads, head_dim]
    cos_k = cos[:, None, :].expand(-1, k.size(2), -1)  # [seqlen, n_kv_heads, head_dim]
    sin_k = sin[:, None, :].expand(-1, k.size(2), -1)  # [seqlen, n_kv_heads, head_dim]
    
    q_embed = (q * cos_q) + (rotate_half(q) * sin_q)
    k_embed = (k * cos_k) + (rotate_half(k) * sin_k)
    return q_embed, k_embed


class Attention(nn.Module):
    def __init__(self, args: MinistralConfig):
        super().__init__()
        self.n_kv_heads = (
            args.num_attention_heads
            if args.num_key_value_heads is None
            else args.num_key_value_heads
        )
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = self.n_kv_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = args.hidden_size // args.num_attention_heads

        self.q_proj = nn.Linear(
            args.hidden_size,
            args.num_attention_heads * self.head_dim,
            bias=False,
        )
        self.k_proj = nn.Linear(
            args.hidden_size,
            args.num_key_value_heads * self.head_dim,
            bias=False,
        )
        self.v_proj = nn.Linear(
            args.hidden_size,
            args.num_key_value_heads * self.head_dim,
            bias=False,
        )
        self.o_proj = nn.Linear(
            args.num_attention_heads * self.head_dim,
            args.hidden_size,
            bias=False,
        )
        self.args = args

    def init_kv_cache(
        self,
        max_batch_size: int,
        max_seq_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        cache_shape = (max_batch_size, max_seq_len, self.n_kv_heads, self.head_dim)
        cache_k = torch.zeros(cache_shape, dtype=dtype, device=device)
        cache_v = torch.zeros(cache_shape, dtype=dtype, device=device)
        self.register_buffer("cache_k", cache_k, persistent=False)
        self.register_buffer("cache_v", cache_v, persistent=False)

    def del_kv_cache(self):
        self.cache_k = None
        self.cache_v = None

    def forward(
        self,
        x: torch.Tensor,
        pos_embed: Tuple[torch.Tensor, torch.Tensor],
        start_pos: Optional[Union[int, torch.Tensor]] = None,
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        cos, sin = pos_embed
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin, unsqueeze_dim=2)
        if start_pos is not None:
            # inference mode
            end_pos = start_pos + seqlen
            self.cache_k[:bsz, start_pos:end_pos, :, :] = xk
            self.cache_v[:bsz, start_pos:end_pos, :, :] = xv
            output = torch.nn.functional.scaled_dot_product_attention(
                query=xq.transpose(1, 2),
                key=self.cache_k[:bsz, :end_pos].transpose(1, 2),
                value=self.cache_v[:bsz, :end_pos].transpose(1, 2),
                is_causal=True if seqlen > 1 else False,
                enable_gqa=True,
            ).transpose(1, 2)
        else:
            # training mode
            output = torch.nn.functional.scaled_dot_product_attention(
                query=xq.transpose(1, 2),
                key=xk.transpose(1, 2),
                value=xv.transpose(1, 2),
                is_causal=True,
                enable_gqa=True,
            ).transpose(1, 2)
        output = output.reshape(bsz, seqlen, -1)
        return self.o_proj(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        intermediate_size: int,
    ):
        super().__init__()
        self.up_proj = nn.Linear(dim, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, dim, bias=False)
        self.gate_proj = nn.Linear(dim, intermediate_size, bias=False)

    def forward(self, x):
        x = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return x


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: MinistralConfig):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.dim = args.hidden_size
        self.head_dim = args.hidden_size // args.num_attention_heads
        self.self_attn = Attention(args)
        self.mlp = FeedForward(
            dim=args.hidden_size,
            intermediate_size=args.intermediate_size,
        )
        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        pos_embed: Tuple[torch.Tensor, torch.Tensor],
        start_pos: Optional[Union[int, torch.Tensor]] = None,
    ):
        h = x + self.self_attn(self.input_layernorm(x), pos_embed, start_pos=start_pos)
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out


class MinistralRotaryEmbedding(nn.Module):
    def __init__(self, config: MinistralConfig, device: torch.device):
        super().__init__()
        self.config = config
        base = config.rope_theta
        dim = config.hidden_size // config.num_attention_heads
        with torch.autocast(device_type=device.type, dtype=torch.float32):
            inv_freq = 1.0 / (
                base
                ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim)
            )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.device = device

    @torch.no_grad()
    def forward(self, x, pos):
        if self.inv_freq.device != x.device:
            self.inv_freq = self.inv_freq.to(x.device)
        device_type = x.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = torch.outer(pos.type_as(self.inv_freq), self.inv_freq)
            # Concatenate freqs to match head_dim (similar to Qwen2 implementation)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class Transformer(nn.Module):
    def __init__(self, params: MinistralConfig, device: torch.device):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.num_hidden_layers
        self.device = device

        self.embed_tokens = nn.Embedding(params.vocab_size, params.hidden_size)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.num_hidden_layers):
            self.layers.append(TransformerBlock(layer_id, params))
        self.norm = RMSNorm(params.hidden_size, eps=params.rms_norm_eps)
        # Initialize rotary embedding on actual device to avoid meta tensor errors
        with torch.device(device):
            self.rotary_emb = MinistralRotaryEmbedding(params, device)
        # Create separate language modeling head if embeddings are not tied
        if not params.tie_word_embeddings:
            self.lm_head = nn.Linear(params.hidden_size, params.vocab_size, bias=False)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.params.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.params.initializer_range)

    def output_proj(self, x):
        if self.params.tie_word_embeddings:
            return F.linear(x, self.embed_tokens.weight, bias=None)
        else:
            return self.lm_head(x)

    def forward(self, tokens: torch.Tensor):
        _bsz, seqlen = tokens.shape
        h = self.embed_tokens(tokens)
        pos = torch.arange(0, seqlen, dtype=torch.long, device=tokens.device)
        pos_embed = self.rotary_emb(h, pos)

        for layer in self.layers:
            h = layer(h, pos_embed)
        h = self.norm(h)
        output = self.output_proj(h)
        return output

    def inference(self, tokens: torch.Tensor, start_pos: Union[int, torch.Tensor]):
        _bsz, seqlen = tokens.shape
        h = self.embed_tokens(tokens)
        if isinstance(start_pos, int):
            pos = torch.arange(
                start_pos, start_pos + seqlen, dtype=torch.long, device=tokens.device
            )
        else:
            pos = torch.arange(seqlen, dtype=torch.long, device=tokens.device) + start_pos
        pos_embed = self.rotary_emb(h, pos)

        for layer in self.layers:
            h = layer(h, pos_embed, start_pos=start_pos)
        h = self.norm(h)
        output = self.output_proj(h)
        return output

    def init_kv_cache(
        self,
        max_batch_size: int,
        max_seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        for layer in self.layers:
            layer.self_attn.init_kv_cache(max_batch_size, max_seq_len, dtype, device)

    def del_kv_cache(self):
        for layer in self.layers:
            layer.self_attn.del_kv_cache()

    @classmethod
    def from_pretrained(cls, ckpt_path, device: torch.device):
        ckpt_path = Path(ckpt_path)
        
        # Load config
        with open(ckpt_path / "config.json", "r") as f:
            config_dict = json.load(f)
        
        # Create config object
        config = MinistralConfig(
            attention_dropout=config_dict.get("attention_dropout", 0.0),
            bos_token_id=config_dict.get("bos_token_id", 1),
            eos_token_id=config_dict.get("eos_token_id", 2),
            hidden_act=config_dict.get("hidden_act", "silu"),
            hidden_size=config_dict.get("hidden_size", 4096),
            initializer_range=config_dict.get("initializer_range", 0.02),
            intermediate_size=config_dict.get("intermediate_size", 14336),
            max_position_embeddings=config_dict.get("max_position_embeddings", 131072),
            model_type=config_dict.get("model_type", "mistral"),
            num_attention_heads=config_dict.get("num_attention_heads", 32),
            num_hidden_layers=config_dict.get("num_hidden_layers", 14),
            num_key_value_heads=config_dict.get("num_key_value_heads", 8),
            rms_norm_eps=config_dict.get("rms_norm_eps", 1e-05),
            rope_theta=config_dict.get("rope_theta", 10000.0),
            sliding_window=config_dict.get("sliding_window", 4096),
            tie_word_embeddings=config_dict.get("tie_word_embeddings", False),
            torch_dtype=config_dict.get("torch_dtype", "bfloat16"),
            use_cache=config_dict.get("use_cache", True),
            vocab_size=config_dict.get("vocab_size", 32000)
        )
        
        # Create model on meta device first (no memory allocation)
        with torch.device("meta"):
            model = cls(config, device)
        
        # Load weights using safetensors
        import safetensors.torch
        
        model_weight_files = sorted(ckpt_path.glob("model*.safetensors"))
        weights = {}
        for file in model_weight_files:
            weights.update(safetensors.torch.load_file(file, device="cpu"))
        
        # Remove "model." prefix from keys and handle lm_head
        processed_weights = {}
        for k, v in weights.items():
            if k.startswith("model."):
                new_key = k.replace("model.", "")
                processed_weights[new_key] = v
            elif k == "lm_head.weight":
                # For tied weights, we use embed_tokens.weight
                if not config.tie_word_embeddings:
                    processed_weights["lm_head.weight"] = v
        
        # Load the weights with assign=True for better performance
        model.load_state_dict(processed_weights, strict=True, assign=True)
        return model.to(device)

def main():
    model = Transformer.from_pretrained("Ministral-3b-instruct", device=torch.device("cuda"))
    print(model)

if __name__ == "__main__":
    main() 