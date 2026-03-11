from __future__ import annotations

import argparse

import torch

from qknorm_lab_utils import load_lab_module, make_inputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--so-path", required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this smoke test.")

    mod = load_lab_module(args.so_path)
    params = make_inputs(num_tokens=64)
    qkv = params["qkv"].clone()
    mod.fused_qk_norm_rope_improve(
        qkv, params["num_heads_q"], params["num_heads_k"], params["num_heads_v"],
        params["head_dim"], params["eps"], params["q_weight"], params["k_weight"],
        params["cos_sin_cache"], params["is_neox"], params["position_ids"],
        params["block_size"])
    torch.cuda.synchronize()

    if not torch.isfinite(qkv).all():
        raise AssertionError("Output contains non-finite values.")

    print("smoke test passed")


if __name__ == "__main__":
    main()
