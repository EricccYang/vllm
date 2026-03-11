"""Instruction Density Experiment: 1h/warp vs Nh/warp at matched grid sizes.

Goal
----
Determine whether multi-head-per-warp (Nh) gains from instruction density
(cos/sin smem reuse, register weight reuse) or is purely block-count driven.

Method
------
Fix grid size <= H100 SM count (132) so all blocks fit in ONE wave.
For 1h @ T tokens  => grid = ceil(T * total_qk / warps_per_block).
For 4h @ ~4T tokens => same grid, but 4x total work.
For 8h @ ~8T tokens => same grid, but ~8x total work.

Key metric: time_ratio = time(Nh) / time(1h)
  ~ work_ratio (e.g. 4.0) => bandwidth limited, NO density benefit
  < work_ratio            => instruction density DOES help
  ~ 1.0                   => block/launch limited (absurd for 1 wave)

Modes
-----
  --mode timing   : CUDA-event timing (default), high iters
  --mode ncu-gen  : print ncu commands for profiling each config
  --mode ncu-single : run ONE kernel launch (for ncu to capture)
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Model config  (user-specified: 32Q + 4K, bf16, head_dim=128)
# ---------------------------------------------------------------------------
NUM_HEADS_Q = 32
NUM_HEADS_K = 4
NUM_HEADS_V = 4
HEAD_DIM = 128
TOTAL_QK = NUM_HEADS_Q + NUM_HEADS_K  # 36
BLOCK_SIZE = 256
WARPS_PER_BLOCK = BLOCK_SIZE // 32     # 8
H100_SMS = 132


def compute_grid(num_tokens: int, hpw: int) -> int:
    if hpw == 1:
        total_warps = num_tokens * TOTAL_QK
    else:
        head_chunks = math.ceil(TOTAL_QK / hpw)
        total_warps = num_tokens * head_chunks
    return math.ceil(total_warps / WARPS_PER_BLOCK)


def tokens_for_matched_grid(base_tokens: int, hpw: int) -> tuple[int, int]:
    """Find token count for *hpw* that gives the same grid as 1h@base_tokens."""
    target_grid = compute_grid(base_tokens, 1)
    head_chunks = math.ceil(TOTAL_QK / hpw)
    # grid = ceil(T * head_chunks / WARPS_PER_BLOCK)  ~= target_grid
    t = round(target_grid * WARPS_PER_BLOCK / head_chunks)
    t = max(1, t)
    # fine-tune: walk up/down to match
    while compute_grid(t, hpw) < target_grid:
        t += 1
    while t > 1 and compute_grid(t - 1, hpw) >= target_grid:
        t -= 1
    return t, compute_grid(t, hpw)


# ---------------------------------------------------------------------------
# Tensor helpers
# ---------------------------------------------------------------------------
def make_inputs(num_tokens: int) -> dict:
    dev = torch.device("cuda")
    total_heads = NUM_HEADS_Q + NUM_HEADS_K + NUM_HEADS_V
    qkv = torch.randn(num_tokens, total_heads * HEAD_DIM,
                       device=dev, dtype=torch.bfloat16).contiguous()
    q_weight = torch.randn(HEAD_DIM, device=dev, dtype=torch.bfloat16).contiguous()
    k_weight = torch.randn(HEAD_DIM, device=dev, dtype=torch.bfloat16).contiguous()
    position_ids = torch.arange(num_tokens, device=dev, dtype=torch.int64).contiguous()
    rotary_dim = HEAD_DIM
    half = rotary_dim // 2
    positions = torch.arange(num_tokens + 16, device=dev, dtype=torch.float32)
    inv_freq = 10000.0 ** (
        -2.0 * torch.arange(half, device=dev, dtype=torch.float32) / rotary_dim
    )
    angles = torch.outer(positions, inv_freq)
    cos_sin_cache = torch.cat(
        [torch.cos(angles), torch.sin(angles)], dim=1
    ).to(torch.bfloat16).contiguous()
    return dict(
        qkv=qkv, num_heads_q=NUM_HEADS_Q, num_heads_k=NUM_HEADS_K,
        num_heads_v=NUM_HEADS_V, head_dim=HEAD_DIM, eps=1e-6,
        q_weight=q_weight, k_weight=k_weight, cos_sin_cache=cos_sin_cache,
        is_neox=False, position_ids=position_ids, block_size=BLOCK_SIZE,
    )


# ---------------------------------------------------------------------------
# Kernel runners
# ---------------------------------------------------------------------------
def call_1h(mod, params):
    qkv = params["qkv"].clone()
    mod.fused_qk_norm_rope_improve(
        qkv, params["num_heads_q"], params["num_heads_k"],
        params["num_heads_v"], params["head_dim"], params["eps"],
        params["q_weight"], params["k_weight"], params["cos_sin_cache"],
        params["is_neox"], params["position_ids"], params["block_size"])


def call_nh(mod, params, hpw):
    qkv = params["qkv"].clone()
    mod.fused_qk_norm_rope_improve_force_hpw(
        qkv, params["num_heads_q"], params["num_heads_k"],
        params["num_heads_v"], params["head_dim"], params["eps"],
        params["q_weight"], params["k_weight"], params["cos_sin_cache"],
        params["is_neox"], params["position_ids"], params["block_size"], hpw)


def bench(mod, params, iters, warmup, hpw=None):
    """Return avg kernel time in ms."""
    fn = (lambda: call_1h(mod, params)) if hpw is None else (lambda: call_nh(mod, params, hpw))
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
    return start.elapsed_time(end) / iters


# ---------------------------------------------------------------------------
# load module
# ---------------------------------------------------------------------------
def load_lab_module(so_path):
    import importlib.util
    so_path = Path(so_path).resolve()
    spec = importlib.util.spec_from_file_location("qknorm_rope_lab", so_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Mode: timing
# ---------------------------------------------------------------------------
def run_timing(args):
    mod = load_lab_module(args.so_path)
    hpw_list = [int(x) for x in args.hpw.split(",")]
    base_tokens_list = [int(x) for x in args.base_tokens.split(",")]

    print(f"Config: {NUM_HEADS_Q}Q + {NUM_HEADS_K}K = {TOTAL_QK} QK heads, "
          f"head_dim={HEAD_DIM}, bf16")
    print(f"block_size={BLOCK_SIZE}, warps/block={WARPS_PER_BLOCK}, "
          f"H100 SMs={H100_SMS}")
    print(f"Iters={args.iters}, Warmup={args.warmup}")

    # ---- Grid plan ----
    print("\n" + "=" * 78)
    print("GRID PLAN")
    print("=" * 78)
    plan = []  # [(base_t, [(hpw, tokens, grid)])]
    for base_t in base_tokens_list:
        grid_1h = compute_grid(base_t, 1)
        entries = [(1, base_t, grid_1h)]
        line = f"  1h @ {base_t:4d}t  grid={grid_1h:4d}"
        for hpw in hpw_list:
            if hpw == 1:
                continue
            t_nh, g_nh = tokens_for_matched_grid(base_t, hpw)
            entries.append((hpw, t_nh, g_nh))
            line += f"  |  {hpw}h @ {t_nh:4d}t  grid={g_nh:4d}"
        print(line)
        plan.append((base_t, entries))

    # ---- Timing ----
    print("\n" + "=" * 78)
    print("TIMING RESULTS")
    print("=" * 78)
    results = []
    for base_t, entries in plan:
        grid_1h = entries[0][2]
        print(f"\n--- Grid target ≈ {grid_1h} (≤ {H100_SMS} SMs = 1 wave) ---")

        # 1h baseline
        p1 = make_inputs(base_t)
        t1 = bench(mod, p1, args.iters, args.warmup, hpw=None)
        print(f"  1h @ {base_t:4d}t : {t1:.6f} ms   grid={grid_1h}")

        for hpw, t_nh, g_nh in entries:
            if hpw == 1:
                continue
            p_nh = make_inputs(t_nh)
            t_ms = bench(mod, p_nh, args.iters, args.warmup, hpw=hpw)
            work_ratio = t_nh / base_t
            time_ratio = t_ms / t1
            density_saving = (1.0 - time_ratio / work_ratio) * 100
            print(f"  {hpw}h @ {t_nh:4d}t : {t_ms:.6f} ms   grid={g_nh}   "
                  f"work={work_ratio:.1f}x  time={time_ratio:.2f}x  "
                  f"density_saving={density_saving:+.1f}%")
            results.append(dict(
                base_tokens=base_t, hpw=hpw, tokens_nh=t_nh,
                grid_1h=grid_1h, grid_nh=g_nh,
                time_1h_ms=t1, time_nh_ms=t_ms,
                work_ratio=work_ratio, time_ratio=time_ratio,
                density_saving_pct=density_saving,
            ))

    # ---- Verdict ----
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print("density_saving > 0%  => instruction density DOES help (cos/sin reuse)")
    print("density_saving ~ 0%  => bandwidth limited, no density benefit")
    print("time_ratio ~ 1.0     => block/launch limited\n")
    for r in results:
        tag = ""
        if r["density_saving_pct"] > 5:
            tag = " << DENSITY BENEFIT"
        elif abs(r["density_saving_pct"]) <= 5:
            tag = " (noise / no benefit)"
        elif r["time_ratio"] < 1.5:
            tag = " << LAUNCH LIMITED"
        print(f"  Grid≈{r['grid_1h']:3d}  {r['hpw']}h  "
              f"time_ratio={r['time_ratio']:.2f}x vs work={r['work_ratio']:.1f}x  "
              f"saving={r['density_saving_pct']:+.1f}%{tag}")

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2))
        print(f"\nSaved JSON: {out}")


# ---------------------------------------------------------------------------
# Mode: ncu-single  (launch exactly ONE kernel for ncu to capture)
# ---------------------------------------------------------------------------
def run_ncu_single(args):
    mod = load_lab_module(args.so_path)
    hpw = int(args.ncu_hpw)
    tokens = int(args.ncu_tokens)
    grid = compute_grid(tokens, hpw)
    print(f"ncu-single: hpw={hpw}, tokens={tokens}, grid={grid}", flush=True)
    params = make_inputs(tokens)
    # warmup
    for _ in range(3):
        if hpw == 1:
            call_1h(mod, params)
        else:
            call_nh(mod, params, hpw)
    torch.cuda.synchronize()
    # single captured launch
    if hpw == 1:
        call_1h(mod, params)
    else:
        call_nh(mod, params, hpw)
    torch.cuda.synchronize()
    print("done", flush=True)


# ---------------------------------------------------------------------------
# Mode: ncu-gen  (print ncu commands)
# ---------------------------------------------------------------------------
def run_ncu_gen(args):
    hpw_list = [int(x) for x in args.hpw.split(",")]
    base_tokens_list = [int(x) for x in args.base_tokens.split(",")]
    script = Path(__file__).resolve()
    out_dir = Path(args.ncu_out_dir).resolve()

    print("#!/usr/bin/env bash")
    print("set -euo pipefail")
    print(f"mkdir -p {out_dir}\n")

    ncu_metrics = ",".join([
        "gpu__time_duration.sum",
        "dram__bytes.sum",
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        "l1tex__t_bytes.sum",
        "sm__inst_executed.sum",
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "sm__warps_active.avg.pct_of_peak_sustained_elapsed",
    ])

    for base_t in base_tokens_list:
        grid_1h = compute_grid(base_t, 1)
        configs = [(1, base_t, grid_1h)]
        for hpw in hpw_list:
            if hpw == 1:
                continue
            t_nh, g_nh = tokens_for_matched_grid(base_t, hpw)
            configs.append((hpw, t_nh, g_nh))

        for hpw, tokens, grid in configs:
            tag = f"g{grid}_hpw{hpw}_t{tokens}"
            rep = out_dir / f"density_{tag}.ncu-rep"
            print(f"echo '=== {tag}: grid={grid} hpw={hpw} tokens={tokens} ==='")
            print(f"ncu --set full "
                  f"--metrics {ncu_metrics} "
                  f"--kernel-name 'fusedQKNormRope' "
                  f"--launch-skip 3 --launch-count 1 "
                  f"-o {rep} "
                  f"python {script} "
                  f"--so-path {args.so_path} "
                  f"--mode ncu-single "
                  f"--ncu-hpw {hpw} --ncu-tokens {tokens}")
            print()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="Instruction density: 1h vs Nh at matched grid")
    p.add_argument("--so-path", required=True, help="Path to qknorm_rope_lab.so")
    p.add_argument("--mode", choices=["timing", "ncu-gen", "ncu-single"],
                   default="timing")
    # timing params
    p.add_argument("--iters", type=int, default=5000)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--hpw", default="4,8",
                   help="Comma-sep HPW values to compare against 1h")
    p.add_argument("--base-tokens", default="1,4,10,20,26",
                   help="1h token counts (grid ≤ 132)")
    p.add_argument("--output-json", default="")
    # ncu-single params
    p.add_argument("--ncu-hpw", type=int, default=1)
    p.add_argument("--ncu-tokens", type=int, default=1)
    # ncu-gen params
    p.add_argument("--ncu-out-dir", default="ncu_density_profiles")
    args = p.parse_args()

    if args.mode == "timing":
        run_timing(args)
    elif args.mode == "ncu-gen":
        run_ncu_gen(args)
    elif args.mode == "ncu-single":
        run_ncu_single(args)


if __name__ == "__main__":
    main()
