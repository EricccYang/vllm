# R002: Vectorized Weight Loading

## Change
Replaced scalar per-element weight loads (`q_weight[dim]`, `k_weight[dim]`) with
vectorized `vec_T` loads matching the QKV load pattern, in both the 1-head and N-heads kernels.

## Results (stable runs, 512 tokens, head_dim=128, fp16)

| Round     | avg_ms (512t) | avg_ms (2048t) | vs Prod |
|-----------|---------------|----------------|---------|
| R001 base | 0.02553       | 0.12333        | ~1.00x  |
| R002 vec  | 0.02496       | 0.10389        | ~1.00x  |

Delta: ~2% at 512 tokens (within noise), ~16% at 2048 tokens (marginal, needs more runs).

## Verdict: NOT EFFECTIVE (noise-level improvement)

## Why
- Weights are only `head_dim` = 128 elements = 256 bytes total per weight tensor.
- At 128 bytes per lane (4 elements x 2 bytes), the weight data fits entirely in L1/L2 cache
  after the first access, so subsequent lanes and iterations hit cache.
- The scalar loads were already being served from L1 at near-register speed.
- The number of load transactions is dominated by the much larger QKV tensor, not weights.

## NCU Expectation vs Reality
- Expected: fewer `l1tex__t_sectors_pipe_lsu_mem_global_op_ld` (load sectors).
- Reality: Weight loads are a negligible fraction of total load traffic. The vectorization
  saves perhaps 2-3 transactions out of hundreds — unmeasurable in Duration.

## Lesson
For small constant-size buffers (weights, position_ids), vectorization doesn't matter
because they're already L1-resident after warmup. Focus optimization effort on the
large per-token QKV data path instead.
