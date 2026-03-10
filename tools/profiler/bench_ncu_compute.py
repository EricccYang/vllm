#!/usr/bin/env python3
# Benchmark the on-the-fly sin/cos compute kernels under NCU.
# Mirrors bench_ncu_5configs.py but uses fused_qk_norm_rope_compute /
# fused_qk_norm_rope_compute_n_token_heads (no cos/sin cache).
#
# Usage:
#   # 1-head-per-warp variant:
#   ncu -k fusedQKNormRopeComputeKernel -o ncu_compute_1h \
#       python tools/profiler/bench_ncu_compute.py
#
#   # 4-heads-per-warp variant:
#   BENCH_NTOKEN_VARIANT=4 \
#   ncu -k fusedQKNormRopeComputeKernelNTokenHeads -o ncu_compute_4h \
#       python tools/profiler/bench_ncu_compute.py
#
# Env:
#   BENCH_NTOKEN_VARIANT=1|2|4|8  (default 1)
#   BENCH_ITERATIONS=2             (default)
#   BENCH_BLOCK_SIZE=256           (default)
#   BENCH_SINGLE_NTOKEN=<int>      If set, run only this num_tokens
#   BENCH_ROPE_BASE=10000.0        (default)

import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from vllm._custom_ops import (
    fused_qk_norm_rope_compute,
    fused_qk_norm_rope_compute_n_token_heads,
)


def main():
    variant    = int(os.environ.get("BENCH_NTOKEN_VARIANT", "1"))  # 1, 2, 4, 8
    iterations = int(os.environ.get("BENCH_ITERATIONS", "2"))
    block_size = int(os.environ.get("BENCH_BLOCK_SIZE", "256"))
    rope_base  = float(os.environ.get("BENCH_ROPE_BASE", "10000.0"))

    device = "cuda:0"
    dtype  = torch.bfloat16
    head_dim, eps        = 128, 1e-6
    is_neox              = True
    num_heads_q          = 16
    num_heads_k          = 4
    num_heads_v          = 4
    total_dim            = (num_heads_q + num_heads_k + num_heads_v) * head_dim

    num_tokens_list = [1000, 2000, 5000, 6000, 8000]
    if os.environ.get("BENCH_SINGLE_NTOKEN"):
        num_tokens_list = [int(os.environ["BENCH_SINGLE_NTOKEN"])]

    q_weight = torch.randn(head_dim, dtype=dtype, device=device) * 0.1 + 1.0
    k_weight = torch.randn(head_dim, dtype=dtype, device=device) * 0.1 + 1.0

    for num_tokens in num_tokens_list:
        qkv         = torch.randn(num_tokens, total_dim, dtype=dtype, device=device)
        position_ids = torch.arange(num_tokens, dtype=torch.long, device=device)
        for _ in range(iterations):
            if variant == 1:
                fused_qk_norm_rope_compute(
                    qkv, num_heads_q, num_heads_k, num_heads_v, head_dim,
                    eps, q_weight, k_weight, is_neox, position_ids,
                    block_size, rope_base,
                )
            else:
                fused_qk_norm_rope_compute_n_token_heads(
                    qkv, num_heads_q, num_heads_k, num_heads_v, head_dim,
                    eps, q_weight, k_weight, is_neox, position_ids,
                    block_size, variant, rope_base,
                )

    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
