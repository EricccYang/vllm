# QKNorm RoPE Kernel Optimization Lab - Progress Summary

## Setup
All work in experimental/qknorm_kernel_lab/. Key scripts:
- scripts/build_round.sh, scripts/run_round.sh
- python/bench_multi_tokens.py, python/bench_block_sweep.py, python/bench_hpw_sweep.py
- Each round: .cu in src/rounds/, report in reports/

## Completed Rounds

### R001: Baseline
- avg_ms at 512t = 0.02553 (reference)
- Register count (N-heads, hd=128, interleave): 40

### R002: Vectorized Weight Loading - NOT EFFECTIVE
- Weights are 256B, L1-cached. Vectorization saves nothing.

### R003: Eliminate elements2 Array - NOT EFFECTIVE
- Interleave path (default) doesn't use elements2. Compiler optimizes arrays to regs.

### R005: Register-Only RoPE (__sincosf) - MARGINAL (~3%)
- cp.async cos/sin already nearly free. __sincosf+powf roughly offsets the saving.

### R008: __launch_bounds__(256,8) - NOT EFFECTIVE
- 40->32 regs (no spills). But kernel not occupancy-limited.

### R012: Block Size 128 - INITIALLY EFFECTIVE (+4%), INCONCLUSIVE on re-test
- First test: bs=128 was 4% faster. Re-test in fresh session: within noise.
- The 4% may have been GPU frequency variation between separate runs.

### R013: HEADS_PER_WARP Sweep - NO SIGNIFICANT DIFFERENCE
- HPW=2,4,8 all within 1%. HPW=8 has 0.7% edge at 2048t.
- The dispatch heuristic is already working correctly.

## Key Insight
This kernel is MEMORY-BANDWIDTH and LAUNCH-LATENCY bound.

Data at 512t: ~9.4 MB. H100 theoretical: ~0.003ms. Actual: ~0.025ms (8x gap).
The gap comes from: kernel launch overhead (~5us), memory subsystem overhead,
warp scheduling latency -- NOT from compute inefficiency.

Compute micro-optimizations (weight vectorization, register reduction, cos/sin
compute path, syncwarp removal) all have <1% impact because they target the
wrong bottleneck.

## What Actually Matters
1. Block scheduling granularity (block_size) - small signal but real
2. Nothing else tested so far has been effective

## Remaining Ideas Worth Trying
- R018 (TMA bulk copy) - Hopper TMA can improve memory transaction efficiency
- R020 (Combined optimizations) - stack any small wins
- Fundamentally different approaches: kernel fusion with the attention that follows,
  or completely different memory access patterns

## Files Created
- src/rounds/r001_baseline.cu through r008_launch_bounds.cu (5 kernel files)
- reports/r002 through r013 analysis reports
- python/bench_multi_tokens.py, bench_block_sweep.py, bench_hpw_sweep.py
- scripts/build_round.sh, run_round.sh
- profiles/rounds/r001-r013 timing/comparison JSONs
