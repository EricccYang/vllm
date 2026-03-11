#!/usr/bin/env bash
set -euo pipefail

LAB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${LAB_DIR}/build"
PROFILE_DIR="${LAB_DIR}/profiles"
SO_PATH="${BUILD_DIR}/qknorm_rope_lab.so"
BENCH_PY="${LAB_DIR}/python/bench_once.py"

mkdir -p "${PROFILE_DIR}"

python "${BENCH_PY}" \
  --so-path "${SO_PATH}" \
  --iters 100 \
  --warmup 20 \
  --tokens 512 \
  --output-json "${PROFILE_DIR}/round1_timing.json"

NCU_LOG="${PROFILE_DIR}/round1_ncu.log"
ncu --target-processes all \
  -f \
  --set full \
  --kernel-name-base demangled \
  --kernel-name "regex:fusedQKNormRopeImproveKernel.*" \
  --launch-skip 20 \
  --launch-count 1 \
  --export "${PROFILE_DIR}/round1" \
  python "${BENCH_PY}" \
    --so-path "${SO_PATH}" \
    --iters 80 \
    --warmup 30 \
    --tokens 512 > "${NCU_LOG}" 2>&1 || true

if python - "${NCU_LOG}" <<'PY'
from pathlib import Path
import sys
log = Path(sys.argv[1]).read_text(encoding="utf-8", errors="ignore")
raise SystemExit(0 if "ERR_NVGPUCTRPERM" in log else 1)
PY
then
  echo "NCU counters permission is missing (ERR_NVGPUCTRPERM)." | tee -a "${NCU_LOG}"
  echo "Run with elevated counter permissions to collect full metrics."
fi

ncu --import "${PROFILE_DIR}/round1.ncu-rep" --page details --csv \
  > "${PROFILE_DIR}/round1_details.csv" || true

echo "Round1 profile outputs:"
echo "  - ${PROFILE_DIR}/round1.ncu-rep"
echo "  - ${PROFILE_DIR}/round1_details.csv"
echo "  - ${PROFILE_DIR}/round1_timing.json"
  echo "  - ${PROFILE_DIR}/round1_ncu.log"
