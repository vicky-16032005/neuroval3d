"""Run the end-to-end smoke benchmark and dump a one-line summary to RUN_LOG.md."""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path


def main() -> int:
    from neuroval3d.evaluation import run_benchmark

    out_dir = Path("outputs/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    result = run_benchmark(use_synthetic=True, n_samples=24, n_per_report=3, out_dir=out_dir)
    table = result.summary_table()

    print(table)

    log_line = (
        f"\n## {datetime.now().strftime('%Y-%m-%d %H:%M')} — smoke benchmark\n"
        f"- run_id: `{result.run_id}`\n"
        f"- n_records: {result.n_records}\n"
        f"- AUROC fusion: {result.auroc_overall.get('fusion', float('nan')):.4f}\n"
        f"- artifacts: `outputs/results/{result.run_id}/`\n"
    )
    log_path = Path("RUN_LOG.md")
    if log_path.exists():
        log_path.write_text(log_path.read_text(encoding="utf-8") + log_line, encoding="utf-8")
    else:
        log_path.write_text(log_line, encoding="utf-8")

    summary = {
        "run_id": result.run_id,
        "n_records": result.n_records,
        "auroc_overall": result.auroc_overall,
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
