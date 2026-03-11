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
    parser = argparse.ArgumentParser()
    parser.add_argument("--so-path", required=True)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--output-json", type=str, default="")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark.")

    mod = load_lab_module(args.so_path)
    params = make_inputs(num_tokens=args.tokens)
    ms = run_kernel(mod, "fused_qk_norm_rope_improve", params, args.iters,
                    args.warmup)
    result = {
        "kernel": "fused_qk_norm_rope_improve",
        "avg_ms": ms,
        "tokens": args.tokens,
        "iters": args.iters,
        "warmup": args.warmup,
    }
    print(json.dumps(result, indent=2))

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
