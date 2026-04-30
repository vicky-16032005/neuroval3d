"""Stage 8 — end-to-end perturbation benchmark.

Loop:
  1. (synthetic mode) sample N random BraTS-like masks → templated reports
     (real-data mode) load TextBraTS / RadGenome reports
  2. build perturbation set (N×K records)
  3. score each (original, perturbed) with our validator + baselines
  4. compute AUROC for hallucination detection per error type
  5. emit a markdown table to outputs/results/<run_id>/auroc_table.md
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np

from neuroval3d.evaluation.perturbation import (
    PerturbationOp,
    PerturbationRecord,
    build_perturbation_set,
)
from neuroval3d.utils.checkpoint import save_checkpoint
from neuroval3d.utils.io import write_jsonl
from neuroval3d.utils.logging import get_logger
from neuroval3d.validators.fusion import FusionValidator
from neuroval3d.validators.lexical import LexicalValidator
from neuroval3d.validators.semantic import SemanticValidator
from neuroval3d.validators.structural import StructuralValidator

log = get_logger("benchmark")


@dataclass
class BenchmarkResult:
    run_id: str
    n_records: int
    auroc_overall: dict[str, float] = field(default_factory=dict)  # validator_name → AUROC
    auroc_by_op: dict[str, dict[str, float]] = field(default_factory=dict)  # op → name → AUROC
    out_dir: Path = field(default_factory=lambda: Path("outputs/results"))

    def summary_table(self) -> str:
        names = sorted({k for d in [self.auroc_overall, *self.auroc_by_op.values()] for k in d})
        ops = sorted(self.auroc_by_op.keys())
        lines = [
            "| Validator | Overall AUROC | " + " | ".join(ops) + " |",
            "|-----------|---------------|" + "|".join(["----------"] * len(ops)) + "|",
        ]
        for name in names:
            row = [name, f"{self.auroc_overall.get(name, float('nan')):.4f}"]
            row.extend(f"{self.auroc_by_op.get(op, {}).get(name, float('nan')):.4f}" for op in ops)
            lines.append("| " + " | ".join(row) + " |")
        return "\n".join(lines)


def run_benchmark(
    reports: Iterable[str] | None = None,
    use_synthetic: bool = True,
    n_samples: int = 120,
    n_per_report: int = 4,
    out_dir: Path | str = Path("outputs/results"),
    seed: int = 7,
) -> BenchmarkResult:
    out_dir = Path(out_dir)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if reports is None:
        if use_synthetic:
            reports_list = list(_synthetic_reports(n_samples, seed=seed))
        else:
            raise ValueError("reports=None requires use_synthetic=True (no real data path provided)")
    else:
        reports_list = list(reports)

    log.info("Building perturbation set on %d reports (n_per_report=%d)...", len(reports_list), n_per_report)
    pset = build_perturbation_set(reports_list, n_per_report=n_per_report, seed=seed)
    pset.save_jsonl(str(run_dir / "perturbation_set.jsonl"))
    log.info("Perturbation records: %d (clean+perturbed)", len(pset))

    semantic = SemanticValidator()
    lexical = LexicalValidator().fit([r.original for r in pset.records] + [r.perturbed for r in pset.records])
    structural = StructuralValidator()
    from neuroval3d.validators import ModalityValidator, NumericValidator, RaTEScoreLite
    numeric = NumericValidator()
    modality = ModalityValidator()
    ratescore_lite = RaTEScoreLite()

    rows: list[dict[str, object]] = []
    for rec in pset.records:
        is_clean = rec.op_detail == "<clean>"
        sem = semantic.score(rec.original, rec.perturbed)
        lex = lexical.score(rec.original, rec.perturbed)
        struct = structural.score(rec.original, rec.perturbed)
        num = numeric.score(rec.original, rec.perturbed)
        mod = modality.score(rec.original, rec.perturbed)
        rs = ratescore_lite.score(rec.original, rec.perturbed)
        rows.append({
            "original_id": rec.original_id,
            "op_type": rec.op_type.value,
            "op_detail": rec.op_detail,
            "is_clean": is_clean,
            "label": 1 if is_clean else 0,    # 1 = valid, 0 = hallucinated
            "semantic": sem,
            "lexical": lex,
            "structural": struct,
            "numeric": num,
            "modality": mod,
            "ratescore_lite": rs,
        })
    write_jsonl(str(run_dir / "scores.jsonl"), rows)

    # Train fusion on the scored set; lazy-import sklearn only here
    fusion = FusionValidator()
    sub_scores = [
        (r["semantic"], r["lexical"], r["structural"], r["numeric"], r["modality"])
        for r in rows
    ]
    labels = [r["label"] for r in rows]
    fusion.fit(sub_scores, labels)
    fused = [fusion.predict(*s).fused for s in sub_scores]

    auroc_overall: dict[str, float] = {}
    auroc_by_op: dict[str, dict[str, float]] = {}
    for name, scores in [
        ("fusion", fused),
        ("structural", [r["structural"] for r in rows]),
        ("lexical", [r["lexical"] for r in rows]),
        ("numeric", [r["numeric"] for r in rows]),
        ("modality", [r["modality"] for r in rows]),
        ("semantic", [r["semantic"] for r in rows]),
        ("ratescore_lite (baseline)", [r["ratescore_lite"] for r in rows]),
    ]:
        auroc_overall[name] = _auroc(labels, scores)
        for op in PerturbationOp:
            mask = [(r["op_type"] == op.value or r["is_clean"]) for r in rows]
            sub_labels = [labels[i] for i, m in enumerate(mask) if m]
            sub_scores_op = [scores[i] for i, m in enumerate(mask) if m]
            if len(set(sub_labels)) >= 2:
                auroc_by_op.setdefault(op.value, {})[name] = _auroc(sub_labels, sub_scores_op)

    result = BenchmarkResult(
        run_id=run_id,
        n_records=len(rows),
        auroc_overall=auroc_overall,
        auroc_by_op=auroc_by_op,
        out_dir=run_dir,
    )

    (run_dir / "auroc_table.md").write_text(result.summary_table(), encoding="utf-8")
    (run_dir / "result.json").write_text(json.dumps(asdict(result), default=str, indent=2), encoding="utf-8")

    cp_id = f"CP-{datetime.now().strftime('%Y%m%d')}-bench-{run_id[-6:]}"
    save_checkpoint(
        cp_id=cp_id,
        phase="Phase 4 (validator+benchmark)",
        description=f"Perturbation benchmark with n={n_samples}",
        metric_name="auroc_fusion",
        metric_value=auroc_overall.get("fusion", float("nan")),
        config={"n_samples": n_samples, "n_per_report": n_per_report, "seed": seed,
                "synthetic": use_synthetic},
        extra={"by_op": auroc_by_op, "run_dir": str(run_dir)},
    )

    log.info("Benchmark complete; AUROC fusion = %.4f", auroc_overall.get("fusion", float("nan")))
    return result


# ----------------------------------------------------------------------------- helpers

def _auroc(labels: list[int], scores: list[float]) -> float:
    """Standard binary AUROC via sklearn; falls back to a numpy implementation if needed."""
    if len(set(labels)) < 2:
        return float("nan")
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(labels, scores))
    except ImportError:
        return _np_auroc(labels, scores)


def _np_auroc(labels: list[int], scores: list[float]) -> float:
    pos = [s for l, s in zip(labels, scores, strict=True) if l == 1]
    neg = [s for l, s in zip(labels, scores, strict=True) if l == 0]
    if not pos or not neg:
        return float("nan")
    wins = sum(1 for p in pos for n in neg if p > n)
    ties = sum(1 for p in pos for n in neg if p == n)
    return float((wins + 0.5 * ties) / (len(pos) * len(neg)))


def _synthetic_reports(n: int, seed: int = 7) -> list[str]:
    """Generate N synthetic reports from random BraTS-like masks for offline benchmarking."""
    from neuroval3d.data.synthetic import SyntheticReportGenerator
    rng = np.random.default_rng(seed)
    gen = SyntheticReportGenerator()
    reports: list[str] = []
    for i in range(n):
        # Build a small random "tumor" mask: 64³ volume with a blob of ET + ED + NCR.
        D = 64
        mask = np.zeros((D, D, D), dtype=np.int16)
        cx, cy, cz = (rng.integers(D // 4, 3 * D // 4) for _ in range(3))
        rad_ed = rng.integers(8, 14)
        rad_et = rng.integers(3, max(4, rad_ed - 3))
        rad_ncr = max(0, rad_et - 2)
        zz, yy, xx = np.ogrid[:D, :D, :D]
        dist2 = (xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2
        mask[dist2 <= rad_ed ** 2] = 2
        mask[dist2 <= rad_et ** 2] = 4
        if rad_ncr > 0:
            mask[dist2 <= rad_ncr ** 2] = 1
        rep = gen.from_mask(mask, voxel_volume_mm3=1.0).text
        reports.append(rep)
    return reports
