#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Profile vLLM with QKNormRoPEFusionPass enabled using NVIDIA Nsight Systems.
# Starts the server under nsys and sends completion requests during the trace
# so that fused_qk_norm_rope kernels appear in the timeline.
#
# Prerequisites: nsys in PATH, curl (for sending requests).
#
# Usage:
#   ./tools/profiler/nsys_profile_qk_norm_rope_fusion.sh <model> [extra vllm args...]
#
# Examples:
#   ./tools/profiler/nsys_profile_qk_norm_rope_fusion.sh Qwen/Qwen2-7B-Instruct
#   ./tools/profiler/nsys_profile_qk_norm_rope_fusion.sh /path/to/Qwen2-7B-Instruct --max-model-len 2048
#   NSYS_DELAY=90 NSYS_DURATION=120 ./tools/profiler/nsys_profile_qk_norm_rope_fusion.sh Qwen/Qwen2-7B-Instruct
#
# Optional env:
#   NSYS_DELAY    - seconds before nsys starts tracing (default: 60; allow time for model load + compile)
#   NSYS_DURATION - trace duration in seconds (default: 60)
#   NSYS_OUTPUT   - output .nsys-rep path (default: qknorm_rope_fusion)
#   NSYS_PORT     - server port for health/requests (default: 8000; must match vllm --port if you pass it)

set -e

MODEL="${1:?Usage: $0 <model> [vllm args...]}"
shift || true

# nsys options
DELAY="${NSYS_DELAY:-60}"
DURATION="${NSYS_DURATION:-60}"
OUTPUT="${NSYS_OUTPUT:-qknorm_rope_fusion}"
PORT="${NSYS_PORT:-8000}"
BASE_URL="http://127.0.0.1:${PORT}"

# Background job: wait for server to be ready, then send completion requests during the trace window.
# This ensures the fused path runs so nsys can capture fused_qk_norm_rope kernels.
send_requests_during_trace() {
  local delay=$1 duration=$2
  echo "[client] Waiting ${delay}s before checking server..."
  sleep "$delay"
  echo ""
  echo ">>> NSYS TRACE WINDOW STARTED (duration: ${duration}s) <<<"
  echo "[client] Checking server at ${BASE_URL}/health ..."
  for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30; do
    if curl -s -o /dev/null -w "%{http_code}" --connect-timeout 2 "${BASE_URL}/health" 2>/dev/null | grep -q 200; then
      echo "[client] Server ready, sending completion requests for ${duration}s..."
      break
    fi
    if [[ $i -eq 30 ]]; then
      echo "[client] WARNING: Server not ready after 30 attempts; no requests sent. Increase NSYS_DELAY?"
      return
    fi
    sleep 2
  done
  local end=$(($(date +%s) + duration))
  while [[ $(date +%s) -lt $end ]]; do
    curl -s -X POST "${BASE_URL}/v1/completions" \
      -H "Content-Type: application/json" \
      -d '{"model":"","prompt":"Hello world","max_tokens":16}' \
      -o /dev/null || true
    sleep 1
  done
  echo "[client] Done sending requests."
}

echo "Model: $MODEL"
echo "Nsys: delay=${DELAY}s, duration=${DURATION}s, output=${OUTPUT}.nsys-rep, port=${PORT}"
echo "QKNormRoPE fusion: enabled; requests will be sent during trace so fused kernels are captured."
echo ""

# Start request sender in background (waits DELAY, then sends requests for DURATION)
send_requests_during_trace "$DELAY" "$DURATION" &
CLIENT_PID=$!

# Run server under nsys. Trace starts after DELAY and lasts DURATION; client sends requests in that window.
nsys profile \
  -t cuda,nvtx \
  -o "$OUTPUT" \
  -f true \
  --trace-fork-before-exec=true \
  --cuda-graph-trace=node \
  --delay "$DELAY" \
  --duration "$DURATION" \
  --stats=true \
  -- \
  vllm serve "$MODEL" \
  -cc.pass_config.enable_qk_norm_rope_fusion=true \
  "$@" \
  --port "$PORT"

# Client may still be running if server exited early
kill "$CLIENT_PID" 2>/dev/null || true
wait "$CLIENT_PID" 2>/dev/null || true

echo ""
echo "Profile saved to: ${OUTPUT}.nsys-rep"
echo "Open with: nsys-ui ${OUTPUT}.nsys-rep"
echo "To verify fusion: search for 'fused_qk_norm_rope' or 'fusedQKNormRopeKernel' in the timeline."
