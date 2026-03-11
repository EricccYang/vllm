# R012: Block Size Sweep (128/256/512)

## Change
Tested the baseline kernel with block_size=128, 256, and 512 (passed at runtime to
the dispatch function; no code change needed).

## Results (500 iters, 100 warmup)

| Block Size | avg_ms (256t) | avg_ms (512t) | avg_ms (2048t) |
|------------|---------------|---------------|----------------|
| 128        | 0.01815       | 0.02653       | 0.11555        |
| 256        | 0.01877       | 0.02777       | 0.12054        |
| Delta      | -3.3%         | -4.5%         | -4.1%          |

block_size=512 is slightly slower than 256 at all counts.

## Verdict: EFFECTIVE — block_size=128 gives consistent ~4% speedup

## Why
1. **Better load balancing.** With block_size=128 (4 warps/block), each block is smaller.
   For 512 tokens × ~18 head-chunks/token = ~9216 warps, block_size=128 produces 2304
   blocks vs 1152 blocks at 256. More blocks means finer-grained distribution across 132
   SMs, reducing the "tail" effect where a few SMs have extra work.
2. **Reduced warp scheduling overhead per block.** Fewer warps per block means the warp
   scheduler has fewer competing warps, reducing instruction issue contention.
3. **Smem pressure is lower.** With 4 warps × (HEADS_PER_WARP × 256B per QKV tile +
   cos/sin), total smem per block decreases, allowing more concurrent blocks per SM.
4. **No register impact.** The kernel template is the same regardless of block_size;
   register count stays at 40.

## When This WOULDN'T Help
- At very high token counts (8K+), block_size=256 may catch up as the grid is large
  enough to amortize tail effects.
- If the kernel had significant per-block initialization cost.

## Recommendation
Change the default block_size from 256 to 128 in production.
