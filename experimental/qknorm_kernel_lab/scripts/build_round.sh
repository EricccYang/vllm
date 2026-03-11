#!/usr/bin/env bash
set -euo pipefail

# Usage: scripts/build_round.sh <round_cu_file>
# Example: scripts/build_round.sh src/rounds/r001_baseline.cu

LAB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${LAB_DIR}/build"
KERNEL_SRC="${1:?Usage: build_round.sh <round_cu_file>}"

mkdir -p "${BUILD_DIR}"

PYTHON_CMAKE_PREFIX="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')"

cmake -S "${LAB_DIR}" -B "${BUILD_DIR}" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="${PYTHON_CMAKE_PREFIX}" \
  -DCMAKE_CUDA_ARCHITECTURES="${CMAKE_CUDA_ARCHITECTURES:-90}" \
  -DLAB_KERNEL_SOURCE="${KERNEL_SRC}" 2>&1 | tail -5

cmake --build "${BUILD_DIR}" -j "${BUILD_JOBS:-8}" 2>&1
echo "Built: ${BUILD_DIR}/qknorm_rope_lab.so (kernel=${KERNEL_SRC})"
