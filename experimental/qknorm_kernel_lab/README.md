# QKNorm RoPE Kernel Lab

This folder is an isolated optimization sandbox for
`fused_qknorm_rope_improve_kernel.cu`, so experiments do not affect production
vLLM code paths.

## Layout

- `src/fused_qknorm_rope_improve_kernel.cu`: copied kernel source to optimize.
- `src/qknorm_rope_lab_binding.cpp`: minimal pybinding interface.
- `CMakeLists.txt`: standalone CMake build for this module only.
- `python/bench_once.py`: simple timing benchmark.
- `python/analyze_round.py`: parse NCU output into a compact JSON summary.
- `python/compare_rounds.py`: compare two round summaries.
- `tests/test_smoke.py`: minimal smoke test.
- `scripts/build_lab.sh`: build standalone `.so`.
- `scripts/run_round1.sh`: run benchmark + NCU + CSV export.

## Quick Start

```bash
cd experimental/qknorm_kernel_lab
scripts/build_lab.sh
PYTHONPATH=python python tests/test_smoke.py --so-path build/qknorm_rope_lab.so
scripts/run_round1.sh
PYTHONPATH=python python python/analyze_round.py \
  --details-csv profiles/round1_details.csv \
  --timing-json profiles/round1_timing.json \
  --out-json profiles/round1_summary.json
```

If NCU cannot access GPU counters in your environment, `profiles/round1_ncu.log`
will contain `ERR_NVGPUCTRPERM` and `round1_details.csv` may be empty.

## Iteration Loop

1. Create a **new** round file under `src/rounds/` for each micro-optimization.
2. Do not overwrite previous round files; keep them as immutable history.
3. Point `CMakeLists.txt` to the selected round file.
4. Rebuild with `scripts/build_lab.sh`.
5. Profile with `scripts/run_round1.sh` (or a round-specific script).
6. Summarize using `python/analyze_round.py`.
7. Compare rounds using `python/compare_rounds.py`.
