"""Run only the two cross-dataset transfer benchmarks (held-out runs already done).

Single Python process so BioClinicalBERT loads once.
Each run persists its result to outputs/results/<run_id>/ immediately, so even if the
second run is interrupted the first is preserved.

Outputs a consolidated `outputs/results/cross_dataset_summary_<timestamp>.md`.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from neuroval3d.data import radgenome_reports_only, textbrats_reports_only
from neuroval3d.evaluation.benchmark import run_cross_dataset_benchmark
from neuroval3d.utils.logging import get_logger

log = get_logger("cross_only")


def main() -> int:
    log.info("Loading reports...")
    tb = textbrats_reports_only()
    rg = radgenome_reports_only(section="global_finding")
    log.info("TextBraTS: %d, RadGenome: %d", len(tb), len(rg))

    log.info("============== 1/2: Cross TextBraTS -> RadGenome ==============")
    r1 = run_cross_dataset_benchmark(
        train_reports=tb, test_reports=rg,
        train_label="textbrats", test_label="radgenome",
    )

    log.info("============== 2/2: Cross RadGenome -> TextBraTS ==============")
    r2 = run_cross_dataset_benchmark(
        train_reports=rg, test_reports=tb,
        train_label="radgenome", test_label="textbrats",
    )

    out_dir = Path("outputs/results")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"cross_dataset_summary_{timestamp}.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# NeuroVal-3D — Cross-Dataset Transfer Results\n\n")
        f.write(f"_Generated {datetime.now().isoformat(timespec='seconds')}_\n\n")
        f.write("| Direction | n_train | n_test | Fusion Train AUROC | Fusion Test AUROC |\n")
        f.write("|-----------|--------:|-------:|-------------------:|------------------:|\n")
        for name, r in [("TextBraTS → RadGenome", r1), ("RadGenome → TextBraTS", r2)]:
            f.write(f"| {name} | {r.n_train} | {r.n_test} | "
                    f"{r.auroc_train.get('fusion', float('nan')):.4f} | "
                    f"**{r.auroc_overall.get('fusion', float('nan')):.4f}** |\n")
        f.write("\n## Detail\n\n")
        for name, r in [("TextBraTS → RadGenome", r1), ("RadGenome → TextBraTS", r2)]:
            f.write(f"\n### {name}\n\n")
            f.write(r.summary_table())
            f.write(f"\n\nrun_id: `{r.run_id}` · out_dir: `{r.out_dir}`\n")

    print(f"\nWrote {out_path}")
    print(f"  TextBraTS → RadGenome: fusion test AUROC = {r1.auroc_overall.get('fusion', float('nan')):.4f}")
    print(f"  RadGenome → TextBraTS: fusion test AUROC = {r2.auroc_overall.get('fusion', float('nan')):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
