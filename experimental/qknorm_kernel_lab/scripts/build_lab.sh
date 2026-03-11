#!/usr/bin/env bash
set -euo pipefail

LAB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${LAB_DIR}/build"

mkdir -p "${BUILD_DIR}"

PYTHON_CMAKE_PREFIX="$(python - <<'PY'
import torch
print(torch.utils.cmake_prefix_path)
PY
)"

cmake -S "${LAB_DIR}" -B "${BUILD_DIR}" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="${PYTHON_CMAKE_PREFIX}" \
  -DCMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES:-90}"

cmake --build "${BUILD_DIR}" -j "${BUILD_JOBS:-8}"
echo "Built module at: ${BUILD_DIR}/qknorm_rope_lab.so"
