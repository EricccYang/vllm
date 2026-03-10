#!/usr/bin/env python3
# Run one variant (1t, 4t, or 8t) for num_tokens in [1000, 2000, 5000, 6000, 8000].
# Used under NCU to capture kernel for all 5 configs in one .ncu-rep.
# Env:
#   BENCH_NTOKEN_VARIANT=1|4|8 (default 1)
#   BENCH_ITERATIONS=2 (default)
#   BENCH_SINGLE_NTOKEN=<int>  If set, run only this num_tokens (e.g. 5000) for clean NCU Compare.

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import torch
import tools.profiler.bench_fused_qknorm_rope_grid as B
from vllm._custom_ops import fused_qk_norm_rope_improve, fused_qk_norm_rope_improve_2_token_heads

def main():
    variant = int(os.environ.get("BENCH_NTOKEN_VARIANT", "1"))  # 1, 4, or 8
    iterations = int(os.environ.get("BENCH_ITERATIONS", "2"))
    device = "cuda:0"
    dtype = torch.bfloat16
    head_dim, eps = 128, 1e-6
    is_neox, max_position, rotary_dim = True, 4096, 128
    num_heads_q, num_heads_k, num_heads_v = 32, 4, 4
    block_size = int(os.environ.get("BENCH_BLOCK_SIZE", "256"))
    num_tokens_list = [1000, 2000, 5000, 6000, 8000]
    if os.environ.get("BENCH_SINGLE_NTOKEN"):
        num_tokens_list = [int(os.environ["BENCH_SINGLE_NTOKEN"])]

    q_weight = torch.randn(head_dim, dtype=dtype, device=device) * 0.1 + 1.0
    k_weight = torch.randn(head_dim, dtype=dtype, device=device) * 0.1 + 1.0
    cos_sin_cache = B._make_cos_sin_cache(max_position, rotary_dim, dtype, torch.device(device))
    total_dim = (num_heads_q + num_heads_k + num_heads_v) * head_dim

    for num_tokens in num_tokens_list:
        qkv = torch.randn(num_tokens, total_dim, dtype=dtype, device=device)
        position_ids = torch.arange(num_tokens, dtype=torch.long, device=device)
        for _ in range(iterations):
            if variant == 1:
                fused_qk_norm_rope_improve(qkv, num_heads_q, num_heads_k, num_heads_v, head_dim, eps, q_weight, k_weight, cos_sin_cache, is_neox, position_ids, block_size)
            else:
                fused_qk_norm_rope_improve_2_token_heads(qkv, num_heads_q, num_heads_k, num_heads_v, head_dim, eps, q_weight, k_weight, cos_sin_cache, is_neox, position_ids, block_size, variant)
    torch.cuda.synchronize()

if __name__ == "__main__":
    main()

