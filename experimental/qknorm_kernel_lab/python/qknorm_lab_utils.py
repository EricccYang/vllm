from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import torch


def load_lab_module(so_path: str | Path) -> ModuleType:
    so_path = Path(so_path).resolve()
    spec = importlib.util.spec_from_file_location("qknorm_rope_lab", so_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load extension spec from: {so_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_cos_sin_cache(max_position: int, rotary_dim: int, base: float,
                       device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    half = rotary_dim // 2
    positions = torch.arange(max_position, device=device, dtype=torch.float32)
    inv_freq = base**(-2.0 * torch.arange(half, device=device, dtype=torch.float32) /
                      rotary_dim)
    angles = torch.outer(positions, inv_freq)
    cos = torch.cos(angles).to(dtype)
    sin = torch.sin(angles).to(dtype)
    return torch.cat([cos, sin], dim=1).contiguous()


def make_inputs(num_tokens: int = 512,
                num_heads_q: int = 32,
                num_heads_k: int = 8,
                num_heads_v: int = 8,
                head_dim: int = 128,
                dtype: torch.dtype = torch.float16,
                device: str = "cuda") -> dict[str, torch.Tensor | int | float | bool]:
    dev = torch.device(device)
    total_heads = num_heads_q + num_heads_k + num_heads_v
    qkv = torch.randn(num_tokens, total_heads * head_dim, device=dev,
                      dtype=dtype).contiguous()
    q_weight = torch.randn(head_dim, device=dev, dtype=dtype).contiguous()
    k_weight = torch.randn(head_dim, device=dev, dtype=dtype).contiguous()
    position_ids = torch.arange(num_tokens, device=dev, dtype=torch.int64).contiguous()
    cos_sin_cache = make_cos_sin_cache(num_tokens + 16, head_dim, 10000.0, dev, dtype)
    return {
        "qkv": qkv,
        "num_heads_q": num_heads_q,
        "num_heads_k": num_heads_k,
        "num_heads_v": num_heads_v,
        "head_dim": head_dim,
        "eps": 1e-6,
        "q_weight": q_weight,
        "k_weight": k_weight,
        "cos_sin_cache": cos_sin_cache,
        "is_neox": False,
        "position_ids": position_ids,
        "block_size": 256,
    }
