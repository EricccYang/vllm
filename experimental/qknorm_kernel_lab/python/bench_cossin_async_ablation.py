"""Ablation: cos/sin cp.async smem vs global LDG in NTokenHeads kernel.

Compares:
  A) NTokenHeads with cos/sin cp.async → smem (production, force_hpw)
  B) NTokenHeads with cos/sin via VLLM_LDG from global (no_cossin_async)

Both use QKV cp.async → smem. The ONLY difference is how cos/sin is loaded.

Modes:
  --mode timing     : CUDA-event timing sweep
  --mode ncu-single : single kernel launch for ncu capture
  --mode ncu-gen    : print shell commands for ncu batch
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

NUM_HEADS_Q = 32
NUM_HEADS_K = 4
NUM_HEADS_V = 4
HEAD_DIM = 128
TOTAL_QK = NUM_HEADS_Q + NUM_HEADS_K
BLOCK_SIZE = 256
WARPS_PER_BLOCK = BLOCK_SIZE // 32


def compute_grid(num_tokens: int, hpw: int) -> int:
    head_chunks = math.ceil(TOTAL_QK / hpw)
    total_warps = num_tokens * head_chunks
    return math.ceil(total_warps / WARPS_PER_BLOCK)


def make_inputs(num_tokens: int) -> dict:
    dev = torch.device("cuda")
    total_heads = NUM_HEADS_Q + NUM_HEADS_K + NUM_HEADS_V
    qkv = torch.randn(num_tokens, total_heads * HEAD_DIM,
                       dtype=torch.bfloat16, device=dev)
    q_weight = torch.randn(HEAD_DIM, dtype=torch.bfloat16, device=dev)
    k_weight = torch.randn(HEAD_DIM, dtype=torch.bfloat16, device=dev)
    max_pos = max(num_tokens, 8192)
    cos_sin_cache = torch.randn(max_pos, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    position_ids = torch.arange(num_tokens, dtype=torch.long, device=dev)
    return dict(qkv=qkv, num_heads_q=NUM_HEADS_Q, num_heads_k=NUM_HEADS_K,
                num_heads_v=NUM_HEADS_V, head_dim=HEAD_DIM, eps=1e-6,
                q_weight=q_weight, k_weight=k_weight,
                cos_sin_cache=cos_sin_cache, is_neox=True,
                position_ids=position_ids, block_size=BLOCK_SIZE)


def call_with_cossin_async(mod, inputs, hpw):
    """Production NTokenHeads: cos/sin cp.async → smem."""
    mod.fused_qk_norm_rope_improve_force_hpw(
        token_heads_per_warp=hpw, **inputs)


def call_no_cossin_async(mod, inputs, hpw):
    """Ablation: cos/sin via VLLM_LDG from global."""
    mod.fused_qk_norm_rope_no_cossin_async(
        token_heads_per_warp=hpw, **inputs)


def bench(fn, warmup=200, iters=1000):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters * 1000  # µs


def run_timing(mod, args):
    token_counts = [int(x) for x in args.tokens.split(",")]
    hpw_values = [int(x) for x in args.hpw.split(",")]
    iters = args.iters

    print(f"{'tokens':>8} {'hpw':>4} {'grid':>6} "
          f"{'async_smem_us':>14} {'global_ldg_us':>14} {'diff_us':>10} {'speedup%':>9}")
    print("-" * 75)

    for T in token_counts:
        for hpw in hpw_values:
            grid = compute_grid(T, hpw)
            inp_a = make_inputs(T)
            inp_b = make_inputs(T)

            t_async = bench(lambda: call_with_cossin_async(mod, inp_a, hpw),
                            warmup=200, iters=iters)
            t_global = bench(lambda: call_no_cossin_async(mod, inp_b, hpw),
                             warmup=200, iters=iters)
            diff = t_global - t_async
            pct = diff / t_global * 100 if t_global > 0 else 0
            print(f"{T:>8} {hpw:>4} {grid:>6} "
                  f"{t_async:>14.2f} {t_global:>14.2f} {diff:>10.2f} {pct:>8.1f}%")


def run_ncu_single(mod, args):
    """Execute one kernel for ncu to capture."""
    hpw = args.ncu_hpw
    T = args.ncu_tokens
    variant = args.ncu_variant  # "async" or "global"
    inp = make_inputs(T)
    torch.cuda.synchronize()

    if variant == "async":
        call_with_cossin_async(mod, inp, hpw)
    else:
        call_no_cossin_async(mod, inp, hpw)
    torch.cuda.synchronize()


def run_ncu_gen(args):
    """Print ncu commands for all configs."""
    so = args.so_path
    py = sys.executable
    script = __file__
    token_counts = [int(x) for x in args.tokens.split(",")]
    hpw_values = [int(x) for x in args.hpw.split(",")]

    metrics = ",".join([
        "gpu__time_duration.sum",
        "dram__bytes.sum",
        "l1tex__t_bytes.sum",
        "sm__inst_executed.sum",
        "sm__ctas_launched.sum",
        "smsp__warps_issue_stalled_dispatch_stall.sum",
        "smsp__warps_issue_stalled_drain.sum",
        "smsp__warps_issue_stalled_lg_throttle.sum",
        "smsp__warps_issue_stalled_long_scoreboard.sum",
        "smsp__warps_issue_stalled_short_scoreboard.sum",
        "smsp__warps_issue_stalled_wait.sum",
        "sm__inst_executed.avg.per_cycle_active",
        "sm__inst_executed.avg.per_cycle_elapsed",
        "sm__warps_active.avg.per_cycle_active",
        "sm__warps_active.avg.per_cycle_elapsed",
    ])

    print("#!/bin/bash")
    print("set -e")
    print()
    for T in token_counts:
        for hpw in hpw_values:
            for variant in ["async", "global"]:
                tag = f"hpw{hpw}_t{T}_{variant}"
                # Kernel name filter differs: async uses fusedQKNormRopeImproveKernelNTokenHeads,
                # global uses fusedQKNormRopeNTokenHeadsNoCossinAsync
                if variant == "async":
                    kname = "fusedQKNormRopeImproveKernelNTokenHeads"
                else:
                    kname = "fusedQKNormRopeNTokenHeadsNoCossinAsync"
                cmd = (
                    f'sudo /usr/local/cuda/bin/ncu '
                    f'--metrics {metrics} '
                    f'--kernel-name "regex:{kname}" '
                    f'--launch-skip 0 --launch-count 1 --csv '
                    f'{py} {script} --so-path {so} --mode ncu-single '
                    f'--ncu-hpw {hpw} --ncu-tokens {T} --ncu-variant {variant}'
                )
                print(f'echo "=== {tag} ==="')
                print(cmd)
                print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--so-path", required=True)
    p.add_argument("--mode", choices=["timing", "ncu-single", "ncu-gen"], default="timing")
    p.add_argument("--tokens", default="100,500,1000,2000")
    p.add_argument("--hpw", default="4,8")
    p.add_argument("--iters", type=int, default=1000)
    p.add_argument("--ncu-hpw", type=int, default=4)
    p.add_argument("--ncu-tokens", type=int, default=500)
    p.add_argument("--ncu-variant", choices=["async", "global"], default="async")
    args = p.parse_args()

    if args.mode == "ncu-gen":
        run_ncu_gen(args)
        return

    sys.path.insert(0, str(Path(args.so_path).parent))
    import importlib
    mod = importlib.import_module(Path(args.so_path).stem)

    if args.mode == "timing":
        run_timing(mod, args)
    elif args.mode == "ncu-single":
        run_ncu_single(mod, args)


if __name__ == "__main__":
    main()
