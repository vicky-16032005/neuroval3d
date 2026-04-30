"""Fire all four headline benchmarks in one process to share BioClinicalBERT load.

  1. Held-out TextBraTS (n=369, train_frac=0.7)
  2. Held-out RadGenome (n=1,007, train_frac=0.7)
  3. Cross TextBraTS → RadGenome (train fusion on TextBraTS, test on RadGenome)
  4. Cross RadGenome → TextBraTS

Each result is persisted under outputs/results/<run_id>/ and a checkpoint registered.
A consolidated report is written to outputs/results/headline_table_<timestamp>.md.
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

from neuroval3d.data import radgenome_reports_only, textbrats_reports_only
from neuroval3d.evaluation.benchmark import run_benchmark, run_cross_dataset_benchmark
from neuroval3d.utils.logging import get_logger

log = get_logger("all_benchmarks")


def main() -> int:
    runs: list[tuple[str, dict, "BenchmarkResult | None"]] = []

    log.info("============== 1/4: Held-out TextBraTS ==============")
    tb_reports = textbrats_reports_only()
    log.info("loaded %d TextBraTS reports", len(tb_reports))
    r1 = run_benchmark(reports=tb_reports, use_synthetic=False,
                        n_samples=len(tb_reports), train_frac=0.7)
    runs.append(("Held-out TextBraTS", {"n": len(tb_reports), "train_frac": 0.7}, r1))

    log.info("============== 2/4: Held-out RadGenome ==============")
    rg_reports = radgenome_reports_only(section="global_finding")
    log.info("loaded %d RadGenome reports", len(rg_reports))
    r2 = run_benchmark(reports=rg_reports, use_synthetic=False,
                        n_samples=len(rg_reports), train_frac=0.7)
    runs.append(("Held-out RadGenome", {"n": len(rg_reports), "train_frac": 0.7}, r2))

    log.info("============== 3/4: Cross TextBraTS -> RadGenome ==============")
    r3 = run_cross_dataset_benchmark(
        train_reports=tb_reports, test_reports=rg_reports,
        train_label="textbrats", test_label="radgenome",
    )
    runs.append(("Cross TextBraTS -> RadGenome", {"n_train": len(tb_reports), "n_test": len(rg_reports)}, r3))

    log.info("============== 4/4: Cross RadGenome -> TextBraTS ==============")
    r4 = run_cross_dataset_benchmark(
        train_reports=rg_reports, test_reports=tb_reports,
        train_label="radgenome", test_label="textbrats",
    )
    runs.append(("Cross RadGenome -> TextBraTS", {"n_train": len(rg_reports), "n_test": len(tb_reports)}, r4))

    # Consolidated headline table
    out_dir = Path("outputs/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"headline_table_{timestamp}.md"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# NeuroVal-3D — Consolidated Headline Results\n\n")
        f.write(f"_Generated {datetime.now().isoformat(timespec='seconds')}_\n\n")
        f.write("| Benchmark | Setting | Fusion AUROC | Train AUROC | n_test |\n")
        f.write("|-----------|---------|--------------|-------------|--------|\n")
        for name, setting, r in runs:
            test = r.auroc_overall.get("fusion", float("nan"))
            train = r.auroc_train.get("fusion", float("nan"))
            f.write(f"| {name} | {setting} | **{test:.4f}** | {train:.4f} | {r.n_test} |\n")
        f.write("\n## Per-benchmark detail\n\n")
        for name, _, r in runs:
            f.write(f"\n### {name}\n\n")
            f.write(r.summary_table())
            f.write(f"\n\nrun_id: `{r.run_id}` · out_dir: `{r.out_dir}`\n")

    print(f"\nWrote consolidated table to {out_path}")
    for name, _, r in runs:
        print(f"  {name}: fusion test AUROC = {r.auroc_overall.get('fusion', float('nan')):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
