# Instruction Density Experiment: 1h/warp vs Nh/warp

## Question
In multi-token scenarios, does the multi-head-per-warp (Nh) kernel gain from
instruction density (cos/sin smem reuse, register weight reuse)?

## Setup
- 32 Q heads + 4 K heads = 36 total QK heads, head_dim=128, bf16
- block_size=256 (8 warps/block), H100 (132 SMs)
- Grid constrained to 1 wave (≤ 132 blocks) to isolate per-block effects
- Token counts chosen so 1h@T and 4h@4T produce the SAME grid size

**Critical fix**: Previous R013 HPW sweep was invalid because auto-select
heuristic (line 941-950 in kernel) overwrote user-supplied HPW. Added
`force_hpw` entry point to bypass it.

## Phase 1: CUDA Event Timing (5000 iters)

All configs show ~0.008ms regardless of 1x-8x work difference:

| Grid | 1h time | 4h time (4x work) | 8h time (~7x work) |
|------|---------|-------------------|---------------------|
| 5    | 0.009ms | 0.008ms (0.89x)   | 0.008ms (0.89x)     |
| 18   | 0.008ms | 0.008ms (1.00x)   | 0.008ms (0.99x)     |
| 45   | 0.008ms | 0.008ms (1.00x)   | 0.008ms (1.01x)     |
| 90   | 0.008ms | 0.008ms (1.01x)   | 0.011ms (1.41x)     |
| 117  | 0.008ms | 0.008ms (1.04x)   | 0.009ms (1.13x)     |

**Verdict**: Host launch overhead (~8µs) dominates. Cannot distinguish.

## Phase 2: NCU Profiling (GPU-only time, no host overhead)

| Grid | Config   | tokens | GPU time (ns) | DRAM (B)  | L1 (B)   | Instructions |
|------|----------|--------|---------------|-----------|-----------|-------------|
| 5    | 1h@1t    | 1      | 6,656         | 15,872    | 74,880    | 6,236       |
| 5    | 4h@4t    | 4      | 7,808         | 49,408    | 157,824   | 19,644      |
| 5    | 8h@7t    | 7      | 9,248         | 77,568    | 210,784   | 29,141      |
| 45   | 1h@10t   | 10     | 6,848         | 100,864   | 748,800   | 60,840      |
| 45   | 4h@40t   | 40     | 8,000         | 390,144   | 1,578,240 | 194,760     |
| 45   | 8h@71t   | 71     | 9,632         | 684,544   | 2,137,952 | 293,653     |
| 117  | 1h@26t   | 26     | 7,136         | 252,416   | 1,946,880 | 158,184     |
| 117  | 4h@104t  | 104    | 8,352         | 996,864   | 4,103,424 | 506,376     |
| 117  | 8h@186t  | 186    | 10,240        | 1,774,592 | 5,600,832 | 768,990     |

### Derived: per-QK-unit efficiency (Grid=117)

| Config | QK units | Inst/QK | DRAM B/QK | GPU ns/QK |
|--------|----------|---------|-----------|-----------|
| 1h@26t | 936      | 168.9   | 269.7     | 7.62      |
| 4h@104t| 3,744    | 135.2   | 266.3     | 2.23      |
| 8h@186t| 6,696    | 114.8   | 265.0     | 1.53      |

### Key ratios (Grid=117, same grid size):

| Metric       | 4h/1h | 8h/1h | Ideal if no benefit |
|-------------|-------|-------|---------------------|
| Work (QK)   | 4.0x  | 7.15x | -                   |
| GPU time    | 1.17x | 1.44x | 4.0x / 7.15x       |
| DRAM bytes  | 3.95x | 7.03x | 4.0x / 7.15x       |
| Instructions| 3.20x | 4.86x | 4.0x / 7.15x       |

## Analysis

### Instruction density benefit: YES, measurable

Nh kernel executes fewer instructions per QK unit:
- 4h: 135.2 inst/QK vs 1h: 168.9 inst/QK = **20% fewer instructions**
- 8h: 114.8 inst/QK vs 1h: 168.9 inst/QK = **32% fewer instructions**

Sources of instruction savings:
1. cos/sin loaded once from smem, reused across N heads (vs N separate global loads)
2. q_weight/k_weight loaded to registers once, reused across N heads
3. Amortized position_id load, tokenIdx computation

### DRAM bandwidth savings: NEGLIGIBLE

- 4h: 266.3 B/QK vs 1h: 269.7 B/QK = only **1.3% less DRAM**
- Reason: cos/sin is tiny (256 bytes/token) vs QKV (9216 bytes/token per QK set).
  Reusing cos/sin across 4 heads saves ~192B out of ~9400B per QK unit = ~2%.
  QKV read/write dominates DRAM and scales linearly with work.

### GPU time scaling: dominated by fixed overhead

- 1h@26t (936 QK units): 7,136 ns
- 4h@104t (3,744 QK units): 8,352 ns (+1,216 ns for 3x more work)
- 8h@186t (6,696 QK units): 10,240 ns (+3,104 ns for 6.15x more work)

The ~6.5-7µs base is **GPU-side fixed overhead** (block scheduling, warp setup,
first memory transaction latency). Only the incremental ~1-3µs scales with work.

At grid=117, SM utilization is only 5-16%. The GPU has enough SMs to process all
blocks in parallel within a single wave. Time = max(per-SM latency) not sum(work).

## Conclusion

**Multi-head-per-warp provides a real 20-32% instruction density benefit**, but
this benefit is **practically invisible** in the single-wave regime because:

1. GPU execution is dominated by fixed scheduling overhead (~6.5µs floor)
2. DRAM savings are negligible (~1-2%) since cos/sin is tiny vs QKV data
3. The kernel is neither compute-bound nor bandwidth-bound at these scales --
   it's **latency-bound** (waiting for first data to arrive, warp scheduling)

**When does the density benefit actually matter?**

In the **multi-wave regime** (grid >> 132), the per-wave cost increases
linearly. Each wave pays the per-block compute cost, and the 20-32% instruction
savings compound across waves. But in that regime, the kernel becomes
bandwidth-limited and instruction density matters less than memory throughput.

The fused kernel's primary benefit over separate QKNorm+RoPE kernels remains
**kernel launch elimination** (saving ~5-8µs per avoided launch), not
per-instruction efficiency.
