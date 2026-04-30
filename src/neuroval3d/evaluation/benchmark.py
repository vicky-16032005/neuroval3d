"""Stage 8 — end-to-end perturbation benchmark.

Loop:
  1. (synthetic mode) sample N random BraTS-like masks → templated reports
     (real-data mode) load TextBraTS / RadGenome reports
  2. build perturbation set (N×K records)
  3. score each (original, perturbed) with our validator + baselines
  4. **70/30 split by original_id** → train fusion on train-half, evaluate on test-half
  5. compute AUROC for hallucination detection per error type, on the held-out test set
  6. emit a markdown table to outputs/results/<run_id>/auroc_table.md

`run_cross_dataset_benchmark` does the obvious cross-dataset transfer: fit fusion on
`train_reports`'s perturbation set, evaluate on `test_reports`'s perturbation set. This is
the answer to the reviewer question "does the validator transfer across datasets?"
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
    PerturbationSet,
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

AXIS_COLS = ("semantic", "lexical", "structural", "numeric", "modality", "negation", "lesion_type")


@dataclass
class BenchmarkResult:
    run_id: str
    n_records: int
    auroc_overall: dict[str, float] = field(default_factory=dict)        # held-out test
    auroc_by_op: dict[str, dict[str, float]] = field(default_factory=dict)
    auroc_train: dict[str, float] = field(default_factory=dict)          # train-set AUROC (transparency)
    n_train: int = 0
    n_test: int = 0
    train_frac: float = 0.7
    cross_dataset: dict[str, str] = field(default_factory=dict)          # {"train_label": ..., "test_label": ...}
    out_dir: Path = field(default_factory=lambda: Path("outputs/results"))

    def summary_table(self) -> str:
        names = sorted({k for d in [self.auroc_overall, *self.auroc_by_op.values()] for k in d})
        ops = sorted(self.auroc_by_op.keys())
        header_kind = "Held-out Test AUROC" if self.train_frac < 1.0 else "AUROC"
        lines = [
            f"| Validator | {header_kind} | Train AUROC | " + " | ".join(ops) + " |",
            "|-----------|---------------|-----------|" + "|".join(["----------"] * len(ops)) + "|",
        ]
        for name in names:
            row = [
                name,
                f"{self.auroc_overall.get(name, float('nan')):.4f}",
                f"{self.auroc_train.get(name, float('nan')):.4f}",
            ]
            row.extend(f"{self.auroc_by_op.get(op, {}).get(name, float('nan')):.4f}" for op in ops)
            lines.append("| " + " | ".join(row) + " |")
        if self.cross_dataset:
            lines.append("")
            lines.append(f"_Cross-dataset: train={self.cross_dataset.get('train_label')} → "
                         f"test={self.cross_dataset.get('test_label')}_")
        return "\n".join(lines)


# ============================================================================ public API

def run_benchmark(
    reports: Iterable[str] | None = None,
    use_synthetic: bool = True,
    n_samples: int = 120,
    n_per_report: int = 4,
    train_frac: float = 0.7,
    out_dir: Path | str = Path("outputs/results"),
    seed: int = 7,
) -> BenchmarkResult:
    """Run the perturbation benchmark on a single dataset with a held-out test split."""
    out_dir = Path(out_dir)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    reports_list = _resolve_reports(reports, use_synthetic, n_samples, seed)
    log.info("Building perturbation set on %d reports (n_per_report=%d)...", len(reports_list), n_per_report)
    pset = build_perturbation_set(reports_list, n_per_report=n_per_report, seed=seed)
    pset.save_jsonl(str(run_dir / "perturbation_set.jsonl"))

    rows = score_records(pset)

    # Held-out split by original_id (so a base report's clean + perturbations stay together)
    train_ids, test_ids = _split_ids(rows, train_frac=train_frac, seed=seed)
    train_rows = [r for r in rows if r["original_id"] in train_ids]
    test_rows = [r for r in rows if r["original_id"] in test_ids]
    log.info("Held-out split: %d train rows, %d test rows (by %d/%d base reports)",
             len(train_rows), len(test_rows), len(train_ids), len(test_ids))

    fusion = _fit_fusion(train_rows)
    fused_train = _fused_scores(fusion, train_rows)
    fused_test = _fused_scores(fusion, test_rows)

    # Decorate each row with split label + fusion score so downstream notebooks can plot
    # ROC / PR / confusion-matrix / score-distribution charts directly from scores.jsonl.
    for r, fs in zip(train_rows, fused_train, strict=True):
        r["split"] = "train"
        r["fused"] = fs
    for r, fs in zip(test_rows, fused_test, strict=True):
        r["split"] = "test"
        r["fused"] = fs
    write_jsonl(str(run_dir / "scores.jsonl"), train_rows + test_rows)

    auroc_train = _aurocs(train_rows, fused_train)
    auroc_test, auroc_by_op_test = _aurocs_with_breakdown(test_rows, fused_test)

    result = BenchmarkResult(
        run_id=run_id,
        n_records=len(rows),
        auroc_overall=auroc_test,
        auroc_by_op=auroc_by_op_test,
        auroc_train=auroc_train,
        n_train=len(train_rows),
        n_test=len(test_rows),
        train_frac=train_frac,
        out_dir=run_dir,
    )

    _persist_result(result, run_dir, n_samples, n_per_report, seed, use_synthetic)
    log.info("Benchmark complete; held-out AUROC fusion = %.4f (train AUROC = %.4f)",
             auroc_test.get("fusion", float("nan")), auroc_train.get("fusion", float("nan")))
    return result


def run_cross_dataset_benchmark(
    train_reports: list[str],
    test_reports: list[str],
    train_label: str = "train_dataset",
    test_label: str = "test_dataset",
    n_per_report: int = 4,
    out_dir: Path | str = Path("outputs/results"),
    seed: int = 7,
) -> BenchmarkResult:
    """Train the fusion on `train_reports`'s perturbation set; evaluate on `test_reports`'s.

    The reviewer-critical question: does the validator generalise across datasets?
    """
    out_dir = Path(out_dir)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = out_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    log.info("Cross-dataset benchmark: train=%s (n=%d) → test=%s (n=%d)",
             train_label, len(train_reports), test_label, len(test_reports))

    pset_train = build_perturbation_set(train_reports, n_per_report=n_per_report, seed=seed)
    pset_test = build_perturbation_set(test_reports, n_per_report=n_per_report, seed=seed + 1)

    train_rows = score_records(pset_train)
    test_rows = score_records(pset_test)

    fusion = _fit_fusion(train_rows)
    fused_train = _fused_scores(fusion, train_rows)
    fused_test = _fused_scores(fusion, test_rows)

    for r, fs in zip(train_rows, fused_train, strict=True):
        r["split"] = "train"
        r["fused"] = fs
    for r, fs in zip(test_rows, fused_test, strict=True):
        r["split"] = "test"
        r["fused"] = fs
    write_jsonl(str(run_dir / "scores_train.jsonl"), train_rows)
    write_jsonl(str(run_dir / "scores_test.jsonl"), test_rows)

    auroc_train = _aurocs(train_rows, fused_train)
    auroc_test, auroc_by_op_test = _aurocs_with_breakdown(test_rows, fused_test)

    result = BenchmarkResult(
        run_id=run_id,
        n_records=len(train_rows) + len(test_rows),
        auroc_overall=auroc_test,
        auroc_by_op=auroc_by_op_test,
        auroc_train=auroc_train,
        n_train=len(train_rows),
        n_test=len(test_rows),
        train_frac=1.0,
        cross_dataset={"train_label": train_label, "test_label": test_label},
        out_dir=run_dir,
    )

    _persist_result(result, run_dir, len(train_reports) + len(test_reports), n_per_report, seed,
                    use_synthetic=False)
    log.info("Cross-dataset benchmark complete; fusion AUROC train=%.4f, test=%.4f",
             auroc_train.get("fusion", float("nan")), auroc_test.get("fusion", float("nan")))
    return result


# ============================================================================ helpers

def score_records(pset: PerturbationSet) -> list[dict[str, object]]:
    """Score every record on the 7 NeuroVal-3D axes + RaTEScore-lite baseline."""
    semantic = SemanticValidator()
    log.info("Semantic validator on device: %s", semantic.config.device)
    lexical = LexicalValidator().fit(
        [r.original for r in pset.records] + [r.perturbed for r in pset.records]
    )
    structural = StructuralValidator()
    from neuroval3d.validators import (
        LesionTypeValidator, ModalityValidator, NegationValidator, NumericValidator,
        RaTEScoreLite,
    )
    numeric = NumericValidator()
    modality = ModalityValidator()
    negation = NegationValidator()
    lesion_type = LesionTypeValidator()
    ratescore_lite = RaTEScoreLite()

    # tqdm if available, else identity — keeps the loop visible to the user when
    # BioClinicalBERT scoring is dominating wall-clock.
    try:
        from tqdm.auto import tqdm
        iterator = tqdm(pset.records, desc="scoring", unit="rec")
    except ImportError:
        iterator = pset.records

    rows: list[dict[str, object]] = []
    for rec in iterator:
        is_clean = rec.op_detail == "<clean>"
        rows.append({
            "original_id": rec.original_id,
            "op_type": rec.op_type.value,
            "op_detail": rec.op_detail,
            "is_clean": is_clean,
            "label": 1 if is_clean else 0,
            "semantic": semantic.score(rec.original, rec.perturbed),
            "lexical": lexical.score(rec.original, rec.perturbed),
            "structural": structural.score(rec.original, rec.perturbed),
            "numeric": numeric.score(rec.original, rec.perturbed),
            "modality": modality.score(rec.original, rec.perturbed),
            "negation": negation.score(rec.original, rec.perturbed),
            "lesion_type": lesion_type.score(rec.original, rec.perturbed),
            "ratescore_lite": ratescore_lite.score(rec.original, rec.perturbed),
        })
    return rows


def _split_ids(rows: list[dict[str, object]], train_frac: float, seed: int) -> tuple[set[str], set[str]]:
    rng = np.random.default_rng(seed)
    ids = sorted({str(r["original_id"]) for r in rows})
    rng.shuffle(ids)
    n_train = int(round(len(ids) * train_frac))
    return set(ids[:n_train]), set(ids[n_train:])


def _fit_fusion(rows: list[dict[str, object]]) -> FusionValidator:
    sub_scores = [tuple(r[c] for c in AXIS_COLS) for r in rows]
    labels = [r["label"] for r in rows]
    return FusionValidator().fit(sub_scores, labels)


def _fused_scores(fusion: FusionValidator, rows: list[dict[str, object]]) -> list[float]:
    if fusion._lr is None:
        return [float(sum(r[c] for c in AXIS_COLS) / len(AXIS_COLS)) for r in rows]
    X = np.asarray([[r[c] for c in AXIS_COLS] for r in rows], dtype=np.float32)
    return list(map(float, fusion._lr.predict_proba(X)[:, 1]))


def _aurocs(rows: list[dict[str, object]], fused: list[float]) -> dict[str, float]:
    labels = [r["label"] for r in rows]
    out: dict[str, float] = {"fusion": _auroc(labels, fused)}
    for col in AXIS_COLS + ("ratescore_lite",):
        out[col + (" (baseline)" if col == "ratescore_lite" else "")] = _auroc(labels, [r[col] for r in rows])
    return out


def _aurocs_with_breakdown(
    rows: list[dict[str, object]],
    fused: list[float],
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    overall = _aurocs(rows, fused)
    by_op: dict[str, dict[str, float]] = {}
    labels = [r["label"] for r in rows]
    for name, scores in [
        ("fusion", fused),
        *((c, [r[c] for r in rows]) for c in AXIS_COLS),
        ("ratescore_lite (baseline)", [r["ratescore_lite"] for r in rows]),
    ]:
        for op in PerturbationOp:
            mask = [(r["op_type"] == op.value or r["is_clean"]) for r in rows]
            sub_l = [labels[i] for i, m in enumerate(mask) if m]
            sub_s = [scores[i] for i, m in enumerate(mask) if m]
            if len(set(sub_l)) >= 2:
                by_op.setdefault(op.value, {})[name] = _auroc(sub_l, sub_s)
    return overall, by_op


def _persist_result(
    result: BenchmarkResult,
    run_dir: Path,
    n_samples: int,
    n_per_report: int,
    seed: int,
    use_synthetic: bool,
) -> None:
    (run_dir / "auroc_table.md").write_text(result.summary_table(), encoding="utf-8")
    (run_dir / "result.json").write_text(json.dumps(asdict(result), default=str, indent=2), encoding="utf-8")
    cp_id = f"CP-{datetime.now().strftime('%Y%m%d')}-bench-{result.run_id[-6:]}"
    save_checkpoint(
        cp_id=cp_id,
        phase="Phase 4 (validator+benchmark)",
        description=(
            f"Cross-dataset bench {result.cross_dataset}"
            if result.cross_dataset
            else f"Held-out bench n={n_samples} train_frac={result.train_frac}"
        ),
        metric_name="auroc_fusion_test",
        metric_value=result.auroc_overall.get("fusion", float("nan")),
        config={
            "n_samples": n_samples,
            "n_per_report": n_per_report,
            "seed": seed,
            "synthetic": use_synthetic,
            "train_frac": result.train_frac,
            "cross_dataset": result.cross_dataset,
        },
        extra={"by_op": result.auroc_by_op, "auroc_train": result.auroc_train,
               "run_dir": str(run_dir)},
    )


def _resolve_reports(reports, use_synthetic: bool, n_samples: int, seed: int) -> list[str]:
    if reports is not None:
        return list(reports)
    if use_synthetic:
        return list(_synthetic_reports(n_samples, seed=seed))
    raise ValueError("reports=None requires use_synthetic=True (no real data path provided)")


def _auroc(labels: list[int], scores: list[float]) -> float:
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
    for _ in range(n):
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
        reports.append(gen.from_mask(mask, voxel_volume_mm3=1.0).text)
    return reports
