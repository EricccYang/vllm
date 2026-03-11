from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--new", required=True)
    args = parser.parse_args()

    base = json.loads(Path(args.base).read_text(encoding="utf-8"))
    new = json.loads(Path(args.new).read_text(encoding="utf-8"))

    base_ms = float(base["timing"]["avg_ms"])
    new_ms = float(new["timing"]["avg_ms"])
    speedup = base_ms / new_ms if new_ms > 0 else 0.0
    delta_pct = (new_ms - base_ms) / base_ms * 100.0

    print(f"base avg_ms: {base_ms:.4f}")
    print(f"new  avg_ms: {new_ms:.4f}")
    print(f"delta      : {delta_pct:+.2f}%")
    print(f"speedup    : {speedup:.3f}x")


if __name__ == "__main__":
    main()
