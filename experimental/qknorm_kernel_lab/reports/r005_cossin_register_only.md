# R005: Eliminate smem cos/sin — Register-Only RoPE via On-the-Fly __sincosf

## Change
Removed the entire cos/sin shared memory region and cp.async group 1. Instead, compute
cos/sin values on-the-fly using `__sincosf(pos_f * freq)` and `powf()` for frequency
computation. Values stored in per-thread register arrays `cos_reg[]`, `sin_reg[]`.

## Results (stable runs)

| Round     | avg_ms (512t) | avg_ms (2048t) | vs Prod (512t) |
|-----------|---------------|----------------|----------------|
| R001 base | 0.02553       | 0.12333        | ~1.00x         |
| R005 reg  | 0.02481       | 0.12266        | ~1.03x         |
| Delta     | -2.8%         | -0.5%          |                |

## Verdict: MARGINAL (~3% at 512t, noise-level at 2048t)

## Why the Limited Impact
1. The cos/sin data is tiny: rotary_dim=128 → 256 bytes per position. This is 1 cp.async
   transaction (16B granule × 16 copies ≈ 256B). Eliminating this saves ~1 transaction
   out of the much larger QKV copy (32 lanes × 8B = 256B per head, × N heads).
2. The cp.async for cos/sin overlapped with QKV processing anyway (group 1 was issued
   *after* group 0, and the wait<0> at the end caught both).
3. `__sincosf` + `powf` cost ~60 SFU cycles per pair. With head_dim=128, each lane
   computes 2 pairs = ~120 SFU cycles, which is comparable to the latency of a single
   global load. So we traded memory for compute but didn't win either way.
4. Register pressure increased: cos_reg[4] + sin_reg[4] = 8 extra FP32 registers per
   lane. This may slightly reduce occupancy in high-token scenarios.

## When This WOULD Help
- If smem were the bottleneck (many heads per warp using large smem tiles).
- If cos_sin_cache had poor cache locality (e.g., random position IDs causing cache misses).
- On architectures where SFU throughput is higher relative to memory bandwidth.

## Lesson
For small, reused data (rotary_dim ≤ 256), the cp.async + smem path is already nearly
free due to overlapping. Compute-based alternatives trade memory traffic for ALU/SFU
cycles without clear wins. The optimization is architecturally interesting but not
effective for this specific kernel configuration.
