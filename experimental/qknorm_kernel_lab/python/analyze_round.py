from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


TARGET_METRICS = [
    "SM Frequency",
    "Elapsed Cycles",
    "Duration",
    "Achieved Occupancy",
    "Memory Throughput",
    "DRAM Throughput",
    "L2 Cache Throughput",
    "Compute (SM) Throughput",
    "Registers Per Thread",
    "Shared Memory Configuration Size",
]


def parse_metric_line(line: str) -> tuple[str, str] | None:
    # ncu details CSV lines are quoted.
    parts = [p.strip().strip('"') for p in re.split(r",(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)", line)]
    if len(parts) < 2:
        return None
    # Heuristic: metric name appears in col 1/2 and value near the end.
    for idx in range(min(4, len(parts))):
        name = parts[idx]
        if any(key in name for key in TARGET_METRICS):
            for value in reversed(parts):
                if value and value not in {"Metric Value", "N/A"}:
                    return name, value
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--details-csv", required=True)
    parser.add_argument("--timing-json", required=True)
    parser.add_argument("--out-json", required=True)
    args = parser.parse_args()

    lines = Path(args.details_csv).read_text(encoding="utf-8", errors="ignore").splitlines()
    metrics: dict[str, str] = {}
    for line in lines:
        parsed = parse_metric_line(line)
        if parsed is not None:
            k, v = parsed
            metrics.setdefault(k, v)

    timing = json.loads(Path(args.timing_json).read_text(encoding="utf-8"))
    result = {
        "timing": timing,
        "ncu_metrics": metrics,
        "notes": [
            "Round1 baseline from isolated lab kernel copy.",
            "Use this as reference before editing src/fused_qknorm_rope_improve_kernel.cu in lab.",
        ],
    }
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
