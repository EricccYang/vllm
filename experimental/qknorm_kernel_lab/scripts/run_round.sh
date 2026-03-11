#!/usr/bin/env bash
set -euo pipefail

# Usage: scripts/run_round.sh <round_id> <round_cu_file>
# Example: scripts/run_round.sh r001 src/rounds/r001_baseline.cu

LAB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${LAB_DIR}/build"
SO_PATH="${BUILD_DIR}/qknorm_rope_lab.so"
ROUND_ID="${1:?Usage: run_round.sh <round_id> <round_cu>}"
ROUND_CU="${2:?Usage: run_round.sh <round_id> <round_cu>}"
PROFILE_DIR="${LAB_DIR}/profiles/rounds/${ROUND_ID}"
TOKENS="${BENCH_TOKENS:-1,8,64,256,512,2048}"

mkdir -p "${PROFILE_DIR}"

echo "=== Building ${ROUND_ID} from ${ROUND_CU} ==="
"${LAB_DIR}/scripts/build_round.sh" "${ROUND_CU}"

echo "=== Smoke test ==="
PYTHONPATH="${LAB_DIR}/python" python "${LAB_DIR}/tests/test_smoke.py" --so-path "${SO_PATH}"

echo "=== Multi-token benchmark ==="
PYTHONPATH="${LAB_DIR}/python" python "${LAB_DIR}/python/bench_multi_tokens.py" \
  --so-path "${SO_PATH}" \
  --iters 200 \
  --warmup 50 \
  --tokens "${TOKENS}" \
  --output-json "${PROFILE_DIR}/timing.json"

echo "=== Compare with production op ==="
PYTHONPATH="${LAB_DIR}/python" python "${LAB_DIR}/python/bench_compare_prod.py" \
  --so-path "${SO_PATH}" \
  --iters 200 \
  --warmup 50 \
  --tokens 512 \
  --output-json "${PROFILE_DIR}/compare_prod.json"

echo "=== ${ROUND_ID} complete ==="
echo "Outputs: ${PROFILE_DIR}/"
ls "${PROFILE_DIR}/"
