# R008: __launch_bounds__ for Register Control

## Change
Added __launch_bounds__(256, 8) to both kernels.

## Register Impact
- N-heads (hd=128, interleave): 40 -> 32 registers. NO spills (LOCAL:0).
- 1-head: 27 -> 27 (unchanged).
- Theoretical occupancy: 6 -> 8 max blocks/SM.

## Results
- 512t: 0.02568 ms (vs baseline 0.02553) = within noise
- 2048t: 0.11996 ms (vs baseline 0.12333) = ~2.7% improvement, marginal

## Verdict: NOT EFFECTIVE at 512t, MARGINAL at 2048t

## Why
Kernel is bandwidth-bound, not occupancy-bound. At 512t, grid only fills ~2 blocks/SM.
Extra occupancy is wasted. At 2048t, slight benefit as grid approaches 7 blocks/SM.
