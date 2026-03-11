from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import vllm._custom_ops  # noqa: F401

from qknorm_lab_utils import load_lab_module, make_inputs


def run_lab(mod, params: dict, iters: int, warmup: int) -> float:
    fn = mod.fused_qk_norm_rope_improve
    for _ in range(warmup):
        qkv = params["qkv"].clone()
        fn(qkv, params["num_heads_q"], params["num_heads_k"], params["num_heads_v"],
           params["head_dim"], params["eps"], params["q_weight"], params["k_weight"],
           params["cos_sin_cache"], params["is_neox"], params["position_ids"],
           params["block_size"])
    torch.cuda.synchronize()
    st = torch.cuda.Event(enable_timing=True)
    ed = torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(iters):
        qkv = params["qkv"].clone()
        fn(qkv, params["num_heads_q"], params["num_heads_k"], params["num_heads_v"],
           params["head_dim"], params["eps"], params["q_weight"], params["k_weight"],
           params["cos_sin_cache"], params["is_neox"], params["position_ids"],
           params["block_size"])
    ed.record()
    torch.cuda.synchronize()
    return st.elapsed_time(ed) / iters


def run_prod(params: dict, iters: int, warmup: int) -> float:
    fn = torch.ops._C.fused_qk_norm_rope_improve
    for _ in range(warmup):
        qkv = params["qkv"].clone()
        fn(qkv, params["num_heads_q"], params["num_heads_k"], params["num_heads_v"],
           params["head_dim"], params["eps"], params["q_weight"], params["k_weight"],
           params["cos_sin_cache"], params["is_neox"], params["position_ids"],
           params["block_size"])
    torch.cuda.synchronize()
    st = torch.cuda.Event(enable_timing=True)
    ed = torch.cuda.Event(enable_timing=True)
    st.record()
    for _ in range(iters):
        qkv = params["qkv"].clone()
        fn(qkv, params["num_heads_q"], params["num_heads_k"], params["num_heads_v"],
           params["head_dim"], params["eps"], params["q_weight"], params["k_weight"],
           params["cos_sin_cache"], params["is_neox"], params["position_ids"],
           params["block_size"])
    ed.record()
    torch.cuda.synchronize()
    return st.elapsed_time(ed) / iters


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--so-path", required=True)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--tokens", type=int, default=512)
    p.add_argument("--output-json", type=str, default="")
    args = p.parse_args()

    mod = load_lab_module(args.so_path)
    params = make_inputs(num_tokens=args.tokens)
    lab_ms = run_lab(mod, params, args.iters, args.warmup)
    prod_ms = run_prod(params, args.iters, args.warmup)

    result = {
        "tokens": args.tokens,
        "iters": args.iters,
        "warmup": args.warmup,
        "lab_avg_ms": lab_ms,
        "prod_avg_ms": prod_ms,
        "speedup_vs_prod": (prod_ms / lab_ms) if lab_ms > 0 else 0.0,
        "delta_pct_vs_prod": ((lab_ms - prod_ms) / prod_ms * 100.0)
        if prod_ms > 0 else 0.0,
    }
    print(json.dumps(result, indent=2))
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
