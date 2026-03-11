#!/usr/bin/env bash
set -euo pipefail

LAB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SO="${LAB_DIR}/build/qknorm_rope_lab.so"
PY="/home/ubuntu/work/myenv/bin/python"
SCRIPT="${LAB_DIR}/python/bench_instruction_density.py"
NCU="/usr/local/cuda/bin/ncu"
METRICS="gpu__time_duration.sum,dram__bytes.sum,l1tex__t_bytes.sum,sm__inst_executed.sum"
OUTFILE="${LAB_DIR}/profiles/rounds/instruction_density/ncu_results.txt"
mkdir -p "$(dirname "$OUTFILE")"

run_ncu() {
    local hpw=$1 tokens=$2 tag=$3
    echo "=== $tag: hpw=$hpw tokens=$tokens ==="
    sudo "$NCU" --metrics "$METRICS" \
        --kernel-name "regex:fusedQKNormRopeImprove" \
        --launch-skip 3 --launch-count 1 --csv \
        "$PY" "$SCRIPT" --so-path "$SO" \
        --mode ncu-single --ncu-hpw "$hpw" --ncu-tokens "$tokens" 2>&1 \
        | grep -E '^"' | tail -n +2
    echo ""
}

{
echo "NCU Instruction Density Results"
echo "================================"
echo "Config: 32Q + 4K = 36 QK heads, head_dim=128, bf16, block_size=256"
echo ""

echo "--- Grid ≈ 5 (1 wave, 5/132 SMs) ---"
run_ncu 1  1  "1h@1t_g5"
run_ncu 4  4  "4h@4t_g5"
run_ncu 8  7  "8h@7t_g5"

echo "--- Grid ≈ 45 (1 wave, 45/132 SMs) ---"
run_ncu 1  10 "1h@10t_g45"
run_ncu 4  40 "4h@40t_g45"
run_ncu 8  71 "8h@71t_g45"

echo "--- Grid ≈ 117 (1 wave, 117/132 SMs) ---"
run_ncu 1  26  "1h@26t_g117"
run_ncu 4  104 "4h@104t_g117"
run_ncu 8  186 "8h@186t_g117"
} 2>&1 | tee "$OUTFILE"

echo "Results saved to: $OUTFILE"
