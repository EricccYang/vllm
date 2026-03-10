#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Benchmark fused_qk_norm_rope over multiple grid sizes for NCU/Nsys profiling.
# Launch: block = (block_size, 1), grid = (x, 1) with
#   x = ceil((num_tokens * (num_heads_q + num_heads_k)) / (block_size/32)).
# This script sweeps grid.x and block_size (128/256/512).
#
# Usage (one-shot, all configs):
#   python tools/profiler/bench_fused_qknorm_rope_grid.py
#
# With NCU (profile only fusedQKNormRopeKernel):
#   ncu -k fusedQKNormRopeKernel -o ncu_qknorm_fused python tools/profiler/bench_fused_qknorm_rope_grid.py
#
# With Nsys (single kernel, one NVTX range "FusedQKNormRope"):
#   nsys profile -t cuda,nvtx -o qknorm_rope python tools/profiler/bench_fused_qknorm_rope_grid.py
#
# Per-part timing (QKNorm vs RoPE) *inside* the same kernel: use Nsight Compute (ncu).
# The kernel has comments "Part 1: QK Norm" and "Part 2: RoPE"; ncu Source view attributes
# cycles to source lines, so you get the ratio for different block sizes / load.
#   ncu -k fusedQKNormRopeKernel -o ncu_qknorm python tools/profiler/bench_fused_qknorm_rope_grid.py
#   Then in ncu report: Source view, sum cycles for lines in Part 1 vs Part 2.
#
# Optional env:
#   BENCH_DRY_RUN=1        - print configs and grid sizes only, no kernel launch
#   BENCH_ITERATIONS=N     - run each config N times (default 3) for stable NCU sampling
#   BENCH_CONFIG_INDEX=i   - run only config i (0..len(BENCH_CONFIGS)-1); for one report per config
#   BENCH_BLOCK_SIZES=128,256,512  - comma-separated block sizes to sweep (default: 128,256,512)
#   BENCH_IMPROVE=1                 - run fused_qk_norm_rope_improve (Improve kernel) instead of baseline
#   BENCH_VERIFY_2_HEADS=1          - run correctness check: 2-token-heads kernel vs baseline (improve) kernel
#   BENCH_2_HEADS=1                 - run N-token-heads kernel (same configs/compute as improve)
#   BENCH_TOKEN_HEADS_PER_WARP=2|4  - when BENCH_2_HEADS=1, use 2 or 4 token-heads per warp (default 2)
#   BENCH_COMPUTE=1                 - run fused_qk_norm_rope_compute (on-the-fly sin/cos, no cache)
#   BENCH_COMPUTE_N_HEADS=1         - run fused_qk_norm_rope_compute_n_token_heads
#   BENCH_VERIFY_COMPUTE=1          - correctness check: compute kernel vs improve kernel
#   BENCH_ROPE_BASE=10000.0         - rope_base for compute kernels (default 10000.0)
#   BENCH_CSV=1                     - print one CSV line per (config, block_size): num_tokens,num_heads_q,num_heads_k,block_size,grid_x,avg_ms
# If you see "undefined symbol: nvtxRangePushA", run with NVTX preloaded, e.g.:
#   LD_PRELOAD=$(python -c "import nvidia.nvtx,os; print(os.path.join(os.path.dirname(nvidia.nvtx.__file__),'lib','libnvToolsExt.so.1'))") BENCH_VERIFY_2_HEADS=1 python tools/profiler/bench_fused_qknorm_rope_grid.py
#
# NCU comparison (same compute: 1-token-head vs 2-token-head):
#   ncu -k fusedQKNormRopeImproveKernel -o ncu_1head -- python tools/profiler/bench_fused_qknorm_rope_grid.py   # BENCH_IMPROVE=1
#   ncu -k fusedQKNormRopeImproveKernel2TokenHeads -o ncu_2head -- python tools/profiler/bench_fused_qknorm_rope_grid.py   # BENCH_2_HEADS=1
#   Then: ncu -i ncu_1head.ncu-rep; ncu -i ncu_2head.ncu-rep (or use ncu-ui)

import os
import sys
import time

import torch

# Add project root for imports when run from repo root or tools/profiler
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from vllm._custom_ops import (
    fused_qk_norm_rope,
    fused_qk_norm_rope_compute,
    fused_qk_norm_rope_compute_n_token_heads,
    fused_qk_norm_rope_improve,
    fused_qk_norm_rope_improve_2_token_heads,
)


def _grid_x(num_tokens: int, num_heads_q: int, num_heads_k: int,
            block_size: int) -> int:
    """Grid is (x, 1); x = ceil(total_warps / warps_per_block)."""
    warps_per_block = block_size // 32
    total_warps = num_tokens * (num_heads_q + num_heads_k)
    return (total_warps + warps_per_block - 1) // warps_per_block


def _grid_x_n_heads(
    num_tokens: int,
    num_heads_q: int,
    num_heads_k: int,
    block_size: int,
    token_heads_per_warp: int,
) -> int:
    """Grid for N-token-heads kernel: total_warps = ceil(total_work / token_heads_per_warp)."""
    warps_per_block = block_size // 32
    total_qk_heads = num_heads_q + num_heads_k
    total_work = num_tokens * total_qk_heads
    total_warps = (total_work + token_heads_per_warp - 1) // token_heads_per_warp
    return (total_warps + warps_per_block - 1) // warps_per_block


def _grid_x_2_heads(num_tokens: int, num_heads_q: int, num_heads_k: int,
                    block_size: int) -> int:
    """Grid for 2-token-heads kernel (convenience)."""
    return _grid_x_n_heads(num_tokens, num_heads_q, num_heads_k, block_size, 2)


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


def _make_cos_sin_cache_noninterleaved(
    max_position: int,
    rotary_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    rope_base: float = 10000.0,
) -> torch.Tensor:
    """Non-interleaved cache: [cos[0..half-1], sin[0..half-1]] per row.
    Matches the layout the improve kernel reads (cos_ptr = cache_ptr,
    sin_ptr = cache_ptr + embed_dim)."""
    half = rotary_dim // 2
    position = torch.arange(max_position, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (rope_base ** (
        torch.arange(half, dtype=torch.float32, device=device) / half))
    freqs = torch.outer(position, inv_freq)
    cos = freqs.cos().to(dtype)
    sin = freqs.sin().to(dtype)
    return torch.cat([cos, sin], dim=-1)


# Target grid.x: ~100, ~500, ~1k, ~2k, ~5k, ~10k (grid = (x, 1), block = (block_size, 1))
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

# Configs used for 2-token-heads correctness verification (small + a bit larger).
VERIFY_CONFIGS = [
    (8, 16, 4),
    (32, 16, 4),
    (64, 32, 8),
]


def verify_n_token_heads_correctness(
    device: torch.device,
    dtype: torch.dtype,
    head_dim: int,
    eps: float,
    is_neox: bool,
    max_position: int,
    rotary_dim: int,
    block_sizes: list[int],
    token_heads_per_warp_list: list[int],
    rtol: float = 1e-2,
    atol: float = 1e-2,
) -> bool:
    """Compare N-token-heads kernel output to baseline (improve kernel). Returns True if all pass."""
    q_weight = torch.randn(head_dim, dtype=dtype, device=device) * 0.1 + 1.0
    k_weight = torch.randn(head_dim, dtype=dtype, device=device) * 0.1 + 1.0
    cos_sin_cache = _make_cos_sin_cache(max_position, rotary_dim, dtype, device)

    all_ok = True
    for n in token_heads_per_warp_list:
        for num_tokens, num_heads_q, num_heads_k in VERIFY_CONFIGS:
            num_heads_v = num_heads_k
            total_dim = (num_heads_q + num_heads_k + num_heads_v) * head_dim

            for block_size in block_sizes:
                qkv_baseline = torch.randn(
                    num_tokens, total_dim, dtype=dtype, device=device
                )
                qkv_n = qkv_baseline.clone()
                position_ids = torch.arange(num_tokens, dtype=torch.long, device=device)

                fused_qk_norm_rope_improve(
                    qkv_baseline,
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
                    block_size=block_size,
                )
                fused_qk_norm_rope_improve_2_token_heads(
                    qkv_n,
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
                    block_size=block_size,
                    token_heads_per_warp=n,
                )

                ok = torch.allclose(qkv_baseline, qkv_n, rtol=rtol, atol=atol)
                if not ok:
                    diff = (qkv_baseline - qkv_n).abs()
                    max_diff = diff.max().item()
                    mean_diff = diff.float().mean().item()
                    print(
                        f"  FAIL N={n} num_tokens={num_tokens} num_heads_q={num_heads_q} "
                        f"num_heads_k={num_heads_k} block_size={block_size}: "
                        f"max_diff={max_diff:.4e} mean_diff={mean_diff:.4e}"
                    )
                    all_ok = False
                else:
                    max_diff = (qkv_baseline - qkv_n).abs().max().item()
                    print(
                        f"  OK   N={n} num_tokens={num_tokens} num_heads_q={num_heads_q} "
                        f"num_heads_k={num_heads_k} block_size={block_size} "
                        f"(max_diff={max_diff:.2e})"
                    )
    return all_ok


def verify_compute_correctness(
    device: torch.device,
    dtype: torch.dtype,
    head_dim: int,
    eps: float,
    is_neox: bool,
    max_position: int,
    block_sizes: list[int],
    token_heads_per_warp_list: list[int],
    rope_base: float = 10000.0,
    rtol: float = 1e-2,
    atol: float = 1e-2,
) -> bool:
    """Compare compute kernels against the improve (cache) kernel. Returns True if all pass."""
    q_weight = torch.randn(head_dim, dtype=dtype, device=device) * 0.1 + 1.0
    k_weight = torch.randn(head_dim, dtype=dtype, device=device) * 0.1 + 1.0
    # Use non-interleaved cache so improve kernel reads correct cos/sin values,
    # matching what the compute kernel computes from rope_base.
    cos_sin_cache = _make_cos_sin_cache_noninterleaved(
        max_position, head_dim, dtype, device, rope_base=rope_base)

    all_ok = True
    for num_tokens, num_heads_q, num_heads_k in VERIFY_CONFIGS:
        num_heads_v = num_heads_k
        total_dim = (num_heads_q + num_heads_k + num_heads_v) * head_dim

        for block_size in block_sizes:
            qkv_ref = torch.randn(num_tokens, total_dim, dtype=dtype, device=device)
            position_ids = torch.arange(num_tokens, dtype=torch.long, device=device)

            # Reference: improve kernel (uses pre-computed cache)
            qkv_baseline = qkv_ref.clone()
            fused_qk_norm_rope_improve(
                qkv_baseline, num_heads_q, num_heads_k, num_heads_v, head_dim,
                eps, q_weight, k_weight, cos_sin_cache, is_neox, position_ids,
                block_size=block_size,
            )

            # Single-head compute kernel
            qkv_c = qkv_ref.clone()
            fused_qk_norm_rope_compute(
                qkv_c, num_heads_q, num_heads_k, num_heads_v, head_dim,
                eps, q_weight, k_weight, is_neox, position_ids,
                block_size=block_size, rope_base=rope_base,
            )
            ok_single = torch.allclose(qkv_baseline, qkv_c, rtol=rtol, atol=atol)
            tag = "compute-1h"
            if ok_single:
                print(f"  OK   {tag} num_tokens={num_tokens} hq={num_heads_q} hk={num_heads_k} bs={block_size} "
                      f"(max_diff={( qkv_baseline - qkv_c).abs().max().item():.2e})")
            else:
                print(f"  FAIL {tag} num_tokens={num_tokens} hq={num_heads_q} hk={num_heads_k} bs={block_size} "
                      f"max_diff={(qkv_baseline - qkv_c).abs().max().item():.4e}")
                all_ok = False

            # N-token-heads compute kernel
            for n in token_heads_per_warp_list:
                qkv_cn = qkv_ref.clone()
                fused_qk_norm_rope_compute_n_token_heads(
                    qkv_cn, num_heads_q, num_heads_k, num_heads_v, head_dim,
                    eps, q_weight, k_weight, is_neox, position_ids,
                    block_size=block_size, token_heads_per_warp=n,
                    rope_base=rope_base,
                )
                ok_n = torch.allclose(qkv_baseline, qkv_cn, rtol=rtol, atol=atol)
                tag_n = f"compute-{n}h"
                if ok_n:
                    print(f"  OK   {tag_n} num_tokens={num_tokens} hq={num_heads_q} hk={num_heads_k} bs={block_size} "
                          f"(max_diff={(qkv_baseline - qkv_cn).abs().max().item():.2e})")
                else:
                    print(f"  FAIL {tag_n} num_tokens={num_tokens} hq={num_heads_q} hk={num_heads_k} bs={block_size} "
                          f"max_diff={(qkv_baseline - qkv_cn).abs().max().item():.4e}")
                    all_ok = False
    return all_ok


def main() -> None:
    dry_run = os.environ.get("BENCH_DRY_RUN", "0") == "1"
    verify_2_heads = os.environ.get("BENCH_VERIFY_2_HEADS", "0") == "1"
    verify_compute = os.environ.get("BENCH_VERIFY_COMPUTE", "0") == "1"
    iterations = int(os.environ.get("BENCH_ITERATIONS", "3"))
    config_index_str = os.environ.get("BENCH_CONFIG_INDEX", "")
    block_sizes_str = os.environ.get("BENCH_BLOCK_SIZES", "128,256,512")
    block_sizes = [int(x.strip()) for x in block_sizes_str.split(",")]
    use_improve = os.environ.get("BENCH_IMPROVE", "0") == "1"
    use_2_heads = os.environ.get("BENCH_2_HEADS", "0") == "1"
    use_compute = os.environ.get("BENCH_COMPUTE", "0") == "1"
    use_compute_n = os.environ.get("BENCH_COMPUTE_N_HEADS", "0") == "1"
    token_heads_per_warp = int(os.environ.get("BENCH_TOKEN_HEADS_PER_WARP", "2"))
    rope_base = float(os.environ.get("BENCH_ROPE_BASE", "10000.0"))
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    head_dim = 128
    eps = 1e-6
    is_neox = True
    max_position = 4096
    rotary_dim = head_dim

    if verify_compute:
        print("BENCH_VERIFY_COMPUTE=1: correctness of compute kernels vs improve (cache) kernel\n")
        ok = verify_compute_correctness(
            device=device, dtype=dtype, head_dim=head_dim, eps=eps,
            is_neox=is_neox, max_position=max_position,
            block_sizes=block_sizes,
            token_heads_per_warp_list=[2, 4, 8],
            rope_base=rope_base,
        )
        print("\nCompute kernel verification: PASS" if ok else "\nCompute kernel verification: FAIL")
        sys.exit(0 if ok else 1)

    if verify_2_heads:
        if not hasattr(torch.ops._C, "fused_qk_norm_rope_improve"):
            print(
                "Custom ops not found (fused_qk_norm_rope_improve missing).\n"
                "Install vLLM from source to build C++ extensions:\n"
                "  pip install -e .\n"
                "Then run this script again."
            )
            sys.exit(1)
        print("BENCH_VERIFY_2_HEADS=1: correctness of N-token-heads (N=2,4) vs baseline (improve)\n")
        ok = verify_n_token_heads_correctness(
            device=device,
            dtype=dtype,
            head_dim=head_dim,
            eps=eps,
            is_neox=is_neox,
            max_position=max_position,
            rotary_dim=rotary_dim,
            block_sizes=block_sizes,
            token_heads_per_warp_list=[2, 4, 8],
        )
        print("\nN-token-heads verification: PASS" if ok else "\nN-token-heads verification: FAIL")
        sys.exit(0 if ok else 1)

    if dry_run:
        print("BENCH_DRY_RUN=1: configs only (block=(block_size,1), grid=(x,1))\n")
        for idx, (num_tokens, num_heads_q, num_heads_k) in enumerate(BENCH_CONFIGS):
            for bs in block_sizes:
                x = _grid_x(num_tokens, num_heads_q, num_heads_k, bs)
                print(f"  [{idx}] block={bs} num_tokens={num_tokens}, num_heads_q={num_heads_q}, num_heads_k={num_heads_k} -> grid=({x}, 1)")
        return

    configs_to_run = BENCH_CONFIGS
    if config_index_str != "":
        idx = int(config_index_str)
        if idx < 0 or idx >= len(BENCH_CONFIGS):
            raise SystemExit(f"BENCH_CONFIG_INDEX must be 0..{len(BENCH_CONFIGS) - 1}, got {idx}")
        configs_to_run = [BENCH_CONFIGS[idx]]
        bs0 = block_sizes[0]
        grid_x = _grid_x(*configs_to_run[0], bs0)
        print(f"Single config: index={idx} block={bs0} -> grid=({grid_x}, 1)\n")

    q_weight = torch.randn(head_dim, dtype=dtype, device=device) * 0.1 + 1.0
    k_weight = torch.randn(head_dim, dtype=dtype, device=device) * 0.1 + 1.0
    cos_sin_cache = _make_cos_sin_cache(max_position, rotary_dim, dtype, device)

    if use_compute_n:
        op_name = f"fused_qk_norm_rope_compute_{token_heads_per_warp}_token_heads"
        ncu_kernel = "fusedQKNormRopeComputeKernelNTokenHeads"
        grid_fn = lambda t, hq, hk, bs: _grid_x_n_heads(t, hq, hk, bs, token_heads_per_warp)
    elif use_compute:
        op_name = "fused_qk_norm_rope_compute"
        ncu_kernel = "fusedQKNormRopeComputeKernel"
        grid_fn = _grid_x
    elif use_2_heads:
        op_name = f"fused_qk_norm_rope_improve_{token_heads_per_warp}_token_heads"
        ncu_kernel = "fusedQKNormRopeImproveKernelNTokenHeads"
        grid_fn = lambda t, hq, hk, bs: _grid_x_n_heads(t, hq, hk, bs, token_heads_per_warp)
    elif use_improve:
        op_name = "fused_qk_norm_rope_improve"
        ncu_kernel = "fusedQKNormRopeImproveKernel"
        grid_fn = _grid_x
    else:
        op_name = "fused_qk_norm_rope"
        ncu_kernel = "fusedQKNormRopeKernel"
        grid_fn = _grid_x

    bench_csv = os.environ.get("BENCH_CSV", "0") == "1"
    if not bench_csv:
        print(f"{op_name} grid/block sweep (NCU: -k {ncu_kernel})\n")
    for block_size in block_sizes:
        if not bench_csv:
            print(f"--- block_size={block_size} ---")
        for num_tokens, num_heads_q, num_heads_k in configs_to_run:
            num_heads_v = num_heads_k
            total_dim = (num_heads_q + num_heads_k + num_heads_v) * head_dim
            grid_x = grid_fn(num_tokens, num_heads_q, num_heads_k, block_size)

            qkv = torch.randn(num_tokens, total_dim, dtype=dtype, device=device)
            position_ids = torch.arange(num_tokens, dtype=torch.long, device=device)

            if os.environ.get("BENCH_CSV", "0") != "1":
                print(f"  block=({block_size},1) grid=({grid_x}, 1) (num_tokens={num_tokens}, totalQKHeads={num_heads_q + num_heads_k}) ... ", end="", flush=True)
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(iterations):
                if use_compute_n:
                    fused_qk_norm_rope_compute_n_token_heads(
                        qkv, num_heads_q, num_heads_k, num_heads_v, head_dim,
                        eps, q_weight, k_weight, is_neox, position_ids,
                        block_size=block_size,
                        token_heads_per_warp=token_heads_per_warp,
                        rope_base=rope_base,
                    )
                elif use_compute:
                    fused_qk_norm_rope_compute(
                        qkv, num_heads_q, num_heads_k, num_heads_v, head_dim,
                        eps, q_weight, k_weight, is_neox, position_ids,
                        block_size=block_size, rope_base=rope_base,
                    )
                elif use_2_heads:
                    fused_qk_norm_rope_improve_2_token_heads(
                        qkv, num_heads_q, num_heads_k, num_heads_v, head_dim,
                        eps, q_weight, k_weight, cos_sin_cache, is_neox,
                        position_ids, block_size=block_size,
                        token_heads_per_warp=token_heads_per_warp,
                    )
                elif use_improve:
                    fused_qk_norm_rope_improve(
                        qkv, num_heads_q, num_heads_k, num_heads_v, head_dim,
                        eps, q_weight, k_weight, cos_sin_cache, is_neox,
                        position_ids, block_size=block_size,
                    )
                else:
                    fused_qk_norm_rope(
                        qkv, num_heads_q, num_heads_k, num_heads_v, head_dim,
                        eps, q_weight, k_weight, cos_sin_cache, is_neox,
                        position_ids, block_size=block_size,
                    )
            torch.cuda.synchronize()
            elapsed_ms = (time.perf_counter() - start) * 1000
            avg_ms = elapsed_ms / iterations
            if os.environ.get("BENCH_CSV", "0") == "1":
                print(f"{num_tokens},{num_heads_q},{num_heads_k},{block_size},{grid_x},{avg_ms:.4f}")
            else:
                print(f"done  avg={avg_ms:.2f} ms/call")

    if not os.environ.get("BENCH_CSV", "0") == "1":
        print("\nAll configs finished.")


if __name__ == "__main__":
    main()
