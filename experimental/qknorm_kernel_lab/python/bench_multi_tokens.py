"""Benchmark the lab kernel at multiple token counts, output JSON summary."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from qknorm_lab_utils import load_lab_module, make_inputs


def run_kernel(mod, fn_name: str, params: dict, iters: int, warmup: int) -> float:
    fn = getattr(mod, fn_name)
    for _ in range(warmup):
        qkv = params["qkv"].clone()
        fn(qkv, params["num_heads_q"], params["num_heads_k"], params["num_heads_v"],
           params["head_dim"], params["eps"], params["q_weight"], params["k_weight"],
           params["cos_sin_cache"], params["is_neox"], params["position_ids"],
           params["block_size"])
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        qkv = params["qkv"].clone()
        fn(qkv, params["num_heads_q"], params["num_heads_k"], params["num_heads_v"],
           params["head_dim"], params["eps"], params["q_weight"], params["k_weight"],
           params["cos_sin_cache"], params["is_neox"], params["position_ids"],
           params["block_size"])
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--so-path", required=True)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--tokens", type=str, default="1,8,64,256,512,2048")
    p.add_argument("--fn", type=str, default="fused_qk_norm_rope_improve")
    p.add_argument("--output-json", type=str, default="")
    args = p.parse_args()

    token_counts = [int(x.strip()) for x in args.tokens.split(",")]
    mod = load_lab_module(args.so_path)

    results = []
    for nt in token_counts:
        params = make_inputs(num_tokens=nt)
        ms = run_kernel(mod, args.fn, params, args.iters, args.warmup)
        entry = {"tokens": nt, "avg_ms": ms}
        results.append(entry)
        print(f"tokens={nt:6d}  avg_ms={ms:.6f}")

    summary = {"fn": args.fn, "iters": args.iters, "warmup": args.warmup, "results": results}
    print(json.dumps(summary, indent=2))

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
