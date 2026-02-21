#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Run NCU on bench_fused_qknorm_rope_grid.py and produce a report file.
#
# Output: REPO_ROOT/tools/profiler/ncu_qknorm_fused.ncu-rep (and .ncu-rep dir if applicable)
#
# If you see ERR_NVGPUCTRPERM, run with sudo:
#   sudo ./tools/profiler/run_ncu_qknorm_bench.sh
# Or once to allow current user: sudo ncu --allow-remote-access ...
# See: https://developer.nvidia.com/ERR_NVGPUCTRPERM

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUTPUT="$SCRIPT_DIR/ncu_qknorm_fused"
NCU="${NCU:-ncu}"
# If ERR_NVGPUCTRPERM, run: sudo env "PATH=$PATH" ./tools/profiler/run_ncu_qknorm_bench.sh
# or: sudo env "PATH=$PATH" NCU=/usr/local/cuda/bin/ncu ./tools/profiler/run_ncu_qknorm_bench.sh

cd "$REPO_ROOT"
"$NCU" -k fusedQKNormRopeKernel -o "$OUTPUT" python tools/profiler/bench_fused_qknorm_rope_grid.py

echo ""
echo "NCU report written to: ${OUTPUT}.ncu-rep (or ${OUTPUT}/)"
ls -la "${OUTPUT}.ncu-rep" 2>/dev/null || ls -la "${OUTPUT}" 2>/dev/null || true
