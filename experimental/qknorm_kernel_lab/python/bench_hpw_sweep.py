"""Sweep HEADS_PER_WARP values and token counts."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from qknorm_lab_utils import load_lab_module, make_inputs

def run_kernel(mod, fn_name, params, iters, warmup, hpw=2):
    fn = getattr(mod, fn_name)
    for _ in range(warmup):
        qkv = params["qkv"].clone()
        fn(qkv, params["num_heads_q"], params["num_heads_k"], params["num_heads_v"],
           params["head_dim"], params["eps"], params["q_weight"], params["k_weight"],
           params["cos_sin_cache"], params["is_neox"], params["position_ids"],
           params["block_size"], hpw)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        qkv = params["qkv"].clone()
        fn(qkv, params["num_heads_q"], params["num_heads_k"], params["num_heads_v"],
           params["head_dim"], params["eps"], params["q_weight"], params["k_weight"],
           params["cos_sin_cache"], params["is_neox"], params["position_ids"],
           params["block_size"], hpw)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--so-path", required=True)
    p.add_argument("--iters", type=int, default=500)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--tokens", type=str, default="1,8,64,256,512,2048")
    p.add_argument("--hpw-values", type=str, default="2,4,8")
    p.add_argument("--block-size", type=int, default=256)
    p.add_argument("--output-json", type=str, default="")
    args = p.parse_args()
    token_counts = [int(x.strip()) for x in args.tokens.split(",")]
    hpw_values = [int(x.strip()) for x in args.hpw_values.split(",")]
    mod = load_lab_module(args.so_path)
    results = []
    for hpw in hpw_values:
        print(f"\n--- HEADS_PER_WARP={hpw}, block_size={args.block_size} ---")
        for nt in token_counts:
            params = make_inputs(num_tokens=nt)
            params["block_size"] = args.block_size
            ms = run_kernel(mod, "fused_qk_norm_rope_improve_2_token_heads",
                          params, args.iters, args.warmup, hpw)
            results.append({"hpw": hpw, "tokens": nt, "avg_ms": ms})
            print(f"  tokens={nt:6d}  avg_ms={ms:.6f}")
    summary = {"block_size": args.block_size, "iters": args.iters, "results": results}
    print("\n" + json.dumps(summary, indent=2))
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
