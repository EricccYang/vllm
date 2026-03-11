# R013: HEADS_PER_WARP Sweep (2/4/8)

## Change
Tested HEADS_PER_WARP = 2, 4, 8 via the _2_token_heads dispatch function.
Combined with block_size = 128 and 256.

## Results (500 iters, 100 warmup, fresh build)

### block_size=256
| HPW | 256t     | 512t     | 2048t    |
|-----|----------|----------|----------|
| 2   | 0.008470 | 0.011710 | 0.054383 |
| 4   | 0.008561 | 0.011707 | 0.054354 |
| 8   | 0.008450 | 0.011619 | 0.054001 |

### block_size=128
| HPW | 256t     | 512t     | 2048t    |
|-----|----------|----------|----------|
| 2   | 0.008432 | 0.011679 | 0.054072 |
| 4   | 0.008418 | 0.011683 | 0.054101 |
| 8   | 0.008479 | 0.011716 | 0.054025 |

## Verdict: NO SIGNIFICANT DIFFERENCE across HPW values

All configurations within <1% of each other. HPW=8 shows a slight (~0.7%) edge at 2048t.

## Why
1. The dispatch heuristic already selects an appropriate HPW based on token count.
   The _2_token_heads function with HPW=2 is the default in production — for low token
   counts it maps to 1 head/warp internally.
2. With 36 Q+K heads per token, HPW=8 means 5 head-chunks per token vs 18 at HPW=2.
   This reduces the number of warps (and blocks) but each warp does more work. The
   net effect on total execution time is nearly zero because the bottleneck is memory
   bandwidth, not warp scheduling.
3. HPW=8's slight edge at 2048t likely comes from better cos/sin smem reuse (8 heads
   sharing one cos/sin load vs 2).

## Lesson
For this kernel with head_dim=128, the HPW parameter is a wash. The internal
dispatch heuristic that chooses HPW based on token count is working correctly.
