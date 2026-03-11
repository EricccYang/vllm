# R003: Eliminate elements2 Array

## Change
Replaced `float elements2[numElemsPerThread]` array with inline `float elem2` temporary
in the neox RoPE path, for both 1-head and N-heads kernels.

## Results (512 tokens, head_dim=128, fp16)

| Round     | avg_ms (512t) | avg_ms (2048t) |
|-----------|---------------|----------------|
| R001 base | 0.02553       | 0.12333        |
| R003 no-e2| 0.02520       | 0.11332        |
| Delta     | -1.3%         | -8.1%          |

## Verdict: MARGINAL / NOT EFFECTIVE at 512t, slight improvement at 2048t

## Why
1. Our default test uses `is_neox=false` → `interleave=true` path, which doesn't use
   `elements2` at all. The neox path code is dead code in this configuration.
2. Even if neox were active, the CUDA compiler already optimizes small fixed-size arrays
   into individual registers. The "array" vs "scalar" distinction at source level
   usually compiles to identical SASS.
3. The 2048t improvement is likely noise — register count stayed at 40 (verified by
   compile output not changing).

## Lesson
Source-level "register pressure" changes often don't map to actual register allocation
changes. The NVCC register allocator works on SSA form, not source-level arrays.
To actually reduce register count, need `__launch_bounds__` or algorithmic changes.
