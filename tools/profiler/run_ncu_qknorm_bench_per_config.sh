#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Run NCU once per benchmark config so each report file corresponds to one grid size.
# Output: tools/profiler/ncu_qknorm_grid{N}.ncu-rep for each config (N = grid.x).
#
# If you see ERR_NVGPUCTRPERM, run with sudo:
#   sudo env "PATH=$PATH" ./tools/profiler/run_ncu_qknorm_bench_per_config.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
NCU="${NCU:-ncu}"

# Number of configs (must match len(BENCH_CONFIGS) in bench_fused_qknorm_rope_grid.py)
NUM_CONFIGS=8

cd "$REPO_ROOT"
echo "Running NCU once per config (${NUM_CONFIGS} reports); output in $SCRIPT_DIR"
for i in $(seq 0 $((NUM_CONFIGS - 1))); do
  out="$SCRIPT_DIR/ncu_qknorm_config${i}"
  echo ""
  echo "=== Config $i -> ${out}.ncu-rep ==="
  BENCH_CONFIG_INDEX=$i "$NCU" -k fusedQKNormRopeKernel -o "$out" python tools/profiler/bench_fused_qknorm_rope_grid.py
done
echo ""
echo "Done. Reports: $SCRIPT_DIR/ncu_qknorm_config*.ncu-rep"
ls -la "$SCRIPT_DIR"/ncu_qknorm_config*.ncu-rep 2>/dev/null || true
