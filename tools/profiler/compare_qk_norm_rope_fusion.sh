#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Compare vLLM performance with and without QKNormRoPEFusion optimization.
# This script runs two profile sessions and benchmark tests:
#   1. Without fusion (baseline)
#   2. With fusion (optimized)
#
# Prerequisites: nsys in PATH, vllm CLI available
#
# Usage:
#   ./tools/profiler/compare_qk_norm_rope_fusion.sh <model> [extra vllm args...]
#
# Examples:
#   ./tools/profiler/compare_qk_norm_rope_fusion.sh /ephemeral/model/Qwen3-Coder-30B-A3B-Instruct-FP8
#   REQUEST_RATE=10 NUM_PROMPTS=80 ./tools/profiler/compare_qk_norm_rope_fusion.sh /path/to/model
#
# Optional env:
#   NSYS_DELAY_BEFORE  - seconds to wait before starting nsys trace (default: 50)
#   NSYS_DURATION      - nsys trace duration in seconds (default: 60)
#   BENCH_DELAY        - seconds to wait before starting benchmark (default: 50)
#   REQUEST_RATE       - benchmark request rate (default: 10)
#   NUM_PROMPTS        - number of benchmark prompts (default: 80)
#   PORT               - server port (default: 8000)

set -e

MODEL="${1:?Usage: $0 <model> [vllm args...]}"
shift || true

# Configuration
NSYS_DELAY_BEFORE="${NSYS_DELAY_BEFORE:-50}"
NSYS_DURATION="${NSYS_DURATION:-60}"
BENCH_DELAY="${BENCH_DELAY:-50}"
REQUEST_RATE="${REQUEST_RATE:-10}"
NUM_PROMPTS="${NUM_PROMPTS:-80}"
PORT="${PORT:-8000}"
BASE_URL="http://127.0.0.1:${PORT}"

OUTPUT_NO_FUSION="qknorm_rope_fusion_no"
OUTPUT_WITH_FUSION="qknorm_rope_fusion"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/${TIMESTAMP}_fusion_comparison"

echo "========================================"
echo "QKNormRoPE Fusion Comparison Test"
echo "========================================"
echo "Model: $MODEL"
echo "Nsys: delay=${NSYS_DELAY_BEFORE}s, duration=${NSYS_DURATION}s"
echo "Benchmark: delay=${BENCH_DELAY}s, rate=${REQUEST_RATE} req/s, prompts=${NUM_PROMPTS}"
echo "Port: ${PORT}"
echo "Results will be saved to: ${RESULTS_DIR}"
echo ""

mkdir -p "$RESULTS_DIR"

# Function to wait for server to be ready
wait_for_server() {
  local timeout=${1:-60}
  local elapsed=0
  echo "[health] Waiting for server at ${BASE_URL}/health ..."
  while [[ $elapsed -lt $timeout ]]; do
    if curl -s -o /dev/null -w "%{http_code}" --connect-timeout 2 "${BASE_URL}/health" 2>/dev/null | grep -q 200; then
      echo "[health] ✓ Server is ready!"
      return 0
    fi
    sleep 2
    elapsed=$((elapsed + 2))
  done
  echo "[health] ✗ Server failed to become ready within ${timeout}s"
  return 1
}

# Function to run benchmark during trace window
run_benchmark_during_trace() {
  local delay=$1
  echo "[benchmark] Sleeping ${delay}s before starting benchmark..."
  sleep "$delay"
  echo ""
  echo ">>> STARTING BENCHMARK (rate=${REQUEST_RATE}, prompts=${NUM_PROMPTS}) <<<"
  echo ""
  vllm bench serve --model "$MODEL" --request-rate "$REQUEST_RATE" --num-prompts "$NUM_PROMPTS"
  echo ""
  echo "[benchmark] ✓ Benchmark completed"
}

# Function to cleanup server process
cleanup_server() {
  echo ""
  echo "[cleanup] Stopping server and nsys processes..."
  # Kill all vllm and nsys processes for this session
  pkill -f "vllm serve" || true
  pkill -f "nsys profile" || true
  sleep 3
  echo "[cleanup] ✓ Cleanup done"
}

# Trap to ensure cleanup on exit
trap cleanup_server EXIT INT TERM

#
# TEST 1: WITHOUT FUSION (BASELINE)
#
echo ""
echo "========================================"
echo "TEST 1/2: WITHOUT FUSION (baseline)"
echo "========================================"
echo ""

# Start benchmark in background (it will wait BENCH_DELAY, then run)
run_benchmark_during_trace "$BENCH_DELAY" &
BENCH_PID=$!

# Run server under nsys
echo "[nsys] Starting profile WITHOUT fusion..."
nsys profile \
  -t cuda,nvtx \
  -f true \
  -o "$OUTPUT_NO_FUSION" \
  --trace-fork-before-exec=true \
  --cuda-graph-trace=node \
  --delay "$NSYS_DELAY_BEFORE" \
  --duration "$NSYS_DURATION" \
  --stats=true \
  -- \
  vllm serve "$MODEL" \
  -cc.pass_config.enable_qk_norm_rope_fusion=false \
  "$@" \
  --port "$PORT" || true

# Wait for benchmark to finish
wait "$BENCH_PID" 2>/dev/null || true

echo ""
echo "[test1] ✓ Profile saved to: ${OUTPUT_NO_FUSION}.nsys-rep"

# Move results to results directory
mv "${OUTPUT_NO_FUSION}.nsys-rep" "$RESULTS_DIR/" 2>/dev/null || true
mv "${OUTPUT_NO_FUSION}.sqlite" "$RESULTS_DIR/" 2>/dev/null || true

# Ensure server is fully stopped
cleanup_server
sleep 5

#
# TEST 2: WITH FUSION (OPTIMIZED)
#
echo ""
echo "========================================"
echo "TEST 2/2: WITH FUSION (optimized)"
echo "========================================"
echo ""

# Start benchmark in background
run_benchmark_during_trace "$BENCH_DELAY" &
BENCH_PID=$!

# Run server under nsys
echo "[nsys] Starting profile WITH fusion..."
nsys profile \
  -t cuda,nvtx \
  -f true \
  -o "$OUTPUT_WITH_FUSION" \
  --trace-fork-before-exec=true \
  --cuda-graph-trace=node \
  --delay "$NSYS_DELAY_BEFORE" \
  --duration "$NSYS_DURATION" \
  --stats=true \
  -- \
  vllm serve "$MODEL" \
  -cc.pass_config.enable_qk_norm_rope_fusion=true \
  "$@" \
  --port "$PORT" || true

# Wait for benchmark to finish
wait "$BENCH_PID" 2>/dev/null || true

echo ""
echo "[test2] ✓ Profile saved to: ${OUTPUT_WITH_FUSION}.nsys-rep"

# Move results to results directory
mv "${OUTPUT_WITH_FUSION}.nsys-rep" "$RESULTS_DIR/" 2>/dev/null || true
mv "${OUTPUT_WITH_FUSION}.sqlite" "$RESULTS_DIR/" 2>/dev/null || true

# Final cleanup
cleanup_server

#
# SUMMARY
#
echo ""
echo "========================================"
echo "✓ COMPARISON TEST COMPLETED"
echo "========================================"
echo ""
echo "Results saved to: ${RESULTS_DIR}/"
echo ""
echo "Files generated:"
echo "  - ${OUTPUT_NO_FUSION}.nsys-rep  (without fusion)"
echo "  - ${OUTPUT_WITH_FUSION}.nsys-rep  (with fusion)"
echo ""
echo "To view profiles:"
echo "  nsys-ui ${RESULTS_DIR}/${OUTPUT_NO_FUSION}.nsys-rep"
echo "  nsys-ui ${RESULTS_DIR}/${OUTPUT_WITH_FUSION}.nsys-rep"
echo ""
echo "To verify fusion, search for 'fused_qk_norm_rope' or 'fusedQKNormRopeKernel' in the WITH fusion timeline."
echo ""
