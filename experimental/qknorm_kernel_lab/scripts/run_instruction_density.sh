#!/usr/bin/env bash
set -euo pipefail

LAB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${LAB_DIR}/build"
SO_PATH="${BUILD_DIR}/qknorm_rope_lab.so"

# ---------- 1. Build ----------
echo "=== Building lab module ==="
bash "${LAB_DIR}/scripts/build_lab.sh"
echo ""

# ---------- 2. Timing benchmark ----------
echo "=== Running timing benchmark ==="
cd "${LAB_DIR}/python"
python bench_instruction_density.py \
    --so-path "${SO_PATH}" \
    --mode timing \
    --iters 5000 \
    --warmup 1000 \
    --hpw "4,8" \
    --base-tokens "1,4,10,20,26" \
    --output-json "${LAB_DIR}/profiles/rounds/instruction_density/timing.json"

echo ""
echo "=== Generating NCU commands ==="
python bench_instruction_density.py \
    --so-path "${SO_PATH}" \
    --mode ncu-gen \
    --hpw "4,8" \
    --base-tokens "1,10,26" \
    --ncu-out-dir "${LAB_DIR}/profiles/rounds/instruction_density/ncu" \
    > "${LAB_DIR}/scripts/ncu_instruction_density.sh"
chmod +x "${LAB_DIR}/scripts/ncu_instruction_density.sh"
echo "NCU script written to: ${LAB_DIR}/scripts/ncu_instruction_density.sh"
echo "Run it with:  bash ${LAB_DIR}/scripts/ncu_instruction_density.sh"
