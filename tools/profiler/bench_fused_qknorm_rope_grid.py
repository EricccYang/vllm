#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Benchmark fused_qk_norm_rope over multiple grid sizes for NCU profiling.
# Launch is fixed: block = (256, 1), grid = (x, 1) with
#   x = ceil((num_tokens * (num_heads_q + num_heads_k)) / 8).
# This script sweeps grid.x from ~100 to ~10000.
#
# Usage (one-shot, all configs):
#   python tools/profiler/bench_fused_qknorm_rope_grid.py
#
# With NCU (profile only fusedQKNormRopeKernel):
#   ncu -k fusedQKNormRopeKernel -o ncu_qknorm_fused python tools/profiler/bench_fused_qknorm_rope_grid.py
# One report: Nsight Compute shows each kernel launch with its Grid/Block, so you can tell configs apart.
# One report per config (easier to compare): use BENCH_CONFIG_INDEX and run_ncu_qknorm_bench_per_config.sh
#
# Optional env:
#   BENCH_DRY_RUN=1        - print configs and grid sizes only, no kernel launch
#   BENCH_ITERATIONS=N     - run each config N times (default 3) for stable NCU sampling
#   BENCH_CONFIG_INDEX=i   - run only config i (0..len(BENCH_CONFIGS)-1); for one report per config

import os
import sys

import torch

# Add project root for imports when run from repo root or tools/profiler
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from vllm._custom_ops import fused_qk_norm_rope


BLOCK_SIZE = 256  # block = (256, 1); fixed in launchFusedQKNormRope


def _grid_x(num_tokens: int, num_heads_q: int, num_heads_k: int) -> int:
    """Grid is (x, 1); x = ceil(total_warps / warps_per_block), warps_per_block=8."""
    warps_per_block = BLOCK_SIZE // 32
    total_warps = num_tokens * (num_heads_q + num_heads_k)
    return (total_warps + warps_per_block - 1) // warps_per_block


def _make_cos_sin_cache(
    max_position: int,
    rotary_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    # Simple cos/sin cache: [max_position, rotary_dim] with cos/sin interleaved
    # (neox style: cos0, sin0, cos1, sin1, ... along last dim)
    half = rotary_dim // 2
    position = torch.arange(max_position, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (10000.0 ** (torch.arange(half, dtype=torch.float32, device=device) / half))
    freqs = torch.outer(position, inv_freq)
    cos = freqs.cos().to(dtype)
    sin = freqs.sin().to(dtype)
    # interleave cos/sin
    cache = torch.stack([cos, sin], dim=-1).flatten(-2)
    return cache


# Target grid.x: ~100, ~500, ~1k, ~2k, ~5k, ~10k (grid = (x, 1), block = (256, 1))
# (num_tokens, num_heads_q, num_heads_k); num_heads_v = num_heads_k
BENCH_CONFIGS = [
    (100, 16, 4),    # grid.x ~100
    (500, 16, 4),    # grid.x ~500
    (1000, 16, 4),   # grid.x ~1000
    (2000, 16, 4),   # grid.x ~2000
    (5000, 16, 4),   # grid.x ~5000
    (10000, 16, 4),  # grid.x ~10000
    (250, 32, 8),    # grid.x ~1000 (alternate head config)
    (2500, 32, 8),   # grid.x ~10000 (alternate head config)
]


def main() -> None:
    dry_run = os.environ.get("BENCH_DRY_RUN", "0") == "1"
    iterations = int(os.environ.get("BENCH_ITERATIONS", "3"))
    config_index_str = os.environ.get("BENCH_CONFIG_INDEX", "")
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    head_dim = 128
    eps = 1e-6
    is_neox = True
    max_position = 4096
    rotary_dim = head_dim

    if dry_run:
        print("BENCH_DRY_RUN=1: configs only (block=(256,1), grid=(x,1))\n")
        for idx, (num_tokens, num_heads_q, num_heads_k) in enumerate(BENCH_CONFIGS):
            x = _grid_x(num_tokens, num_heads_q, num_heads_k)
            print(f"  [{idx}] num_tokens={num_tokens}, num_heads_q={num_heads_q}, num_heads_k={num_heads_k} -> grid=({x}, 1)")
        return

    configs_to_run = BENCH_CONFIGS
    if config_index_str != "":
        idx = int(config_index_str)
        if idx < 0 or idx >= len(BENCH_CONFIGS):
            raise SystemExit(f"BENCH_CONFIG_INDEX must be 0..{len(BENCH_CONFIGS) - 1}, got {idx}")
        configs_to_run = [BENCH_CONFIGS[idx]]
        grid_x = _grid_x(*configs_to_run[0])
        print(f"Single config: index={idx} -> grid=({grid_x}, 1)\n")

    q_weight = torch.randn(head_dim, dtype=dtype, device=device) * 0.1 + 1.0
    k_weight = torch.randn(head_dim, dtype=dtype, device=device) * 0.1 + 1.0
    cos_sin_cache = _make_cos_sin_cache(max_position, rotary_dim, dtype, device)

    print("fused_qk_norm_rope grid sweep: block=(256,1), grid=(x,1) (NCU: -k fusedQKNormRopeKernel)\n")
    for num_tokens, num_heads_q, num_heads_k in configs_to_run:
        num_heads_v = num_heads_k
        total_dim = (num_heads_q + num_heads_k + num_heads_v) * head_dim
        grid_x = _grid_x(num_tokens, num_heads_q, num_heads_k)

        qkv = torch.randn(num_tokens, total_dim, dtype=dtype, device=device)
        position_ids = torch.arange(num_tokens, dtype=torch.long, device=device)

        print(f"  grid=({grid_x}, 1) (num_tokens={num_tokens}, totalQKHeads={num_heads_q + num_heads_k}) ... ", end="", flush=True)
        for _ in range(iterations):
            fused_qk_norm_rope(
                qkv,
                num_heads_q,
                num_heads_k,
                num_heads_v,
                head_dim,
                eps,
                q_weight,
                k_weight,
                cos_sin_cache,
                is_neox,
                position_ids,
            )
        torch.cuda.synchronize()
        print("done")

    print("\nAll configs finished.")


if __name__ == "__main__":
    main()
