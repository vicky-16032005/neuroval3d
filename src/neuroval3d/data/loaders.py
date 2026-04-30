"""Loaders for real paired-report datasets (TextBraTS, RadGenome-Brain MRI, …).

Each loader returns a list of dicts with at minimum:
    - "subject_id": str  (canonical identifier, e.g. BraTS20_Training_001)
    - "report":     str  (the radiology report, free text)

Extras like "modality", "volume_path", "seg_path", "license" can also appear.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

DEFAULT_TEXTBRATS_ROOT = Path("data/raw/TextBraTS/reports")


def load_textbrats(
    root: str | Path = DEFAULT_TEXTBRATS_ROOT,
    limit: int | None = None,
) -> list[dict[str, str]]:
    """Load TextBraTS reports from disk.

    Expects `root/<subject_id>.txt` files (as produced by
    `scripts/download_textbrats_reports.py`).
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(
            f"TextBraTS reports not found at {root.resolve()}. "
            f"Run `python scripts/download_textbrats_reports.py` first."
        )
    files = sorted(root.glob("*.txt"))
    if limit:
        files = files[:limit]
    out: list[dict[str, str]] = []
    for f in files:
        text = f.read_text(encoding="utf-8").strip()
        if not text:
            continue
        out.append({
            "subject_id": f.stem,
            "report": text,
            "source": "TextBraTS",
            "license": "MIT",
            "modality": "FLAIR",
        })
    return out


def textbrats_reports_only(
    root: str | Path = DEFAULT_TEXTBRATS_ROOT,
    limit: int | None = None,
) -> list[str]:
    """Return TextBraTS reports as a flat list of strings, suitable for `run_benchmark`."""
    return [r["report"] for r in load_textbrats(root=root, limit=limit)]


def iter_reports_jsonl(path: str | Path) -> Iterable[dict[str, str]]:
    """Generic JSONL reader for any report dataset that has been pre-flattened."""
    import json
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


# ---------------------------------------------------------------------- RadGenome-Brain MRI

DEFAULT_RADGENOME_ROOT = Path("data/raw/RadGenome-BrainMRI")
RADGENOME_SUBSETS = ("BraTS_GLI", "BraTS_MEN", "BraTS_MET", "ISLES22", "WMH")
RADGENOME_DISEASE = {
    "BraTS_GLI": "glioma",
    "BraTS_MEN": "meningioma",
    "BraTS_MET": "metastasis",
    "ISLES22": "infarction",
    "WMH": "white_matter_hyperintensity",
}


def load_radgenome(
    root: str | Path = DEFAULT_RADGENOME_ROOT,
    section: str = "global_finding",
    subsets: tuple[str, ...] | None = None,
    limit: int | None = None,
) -> list[dict[str, str]]:
    """Load RadGenome-Brain MRI reports from disk.

    Args:
        section: which JSON to read per subset — `global_finding`, `impression`, or
                 `modal_wise_finding`. Defaults to `global_finding` (the richest single text).
        subsets: tuple of subset names; defaults to all five.
    """
    import json
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(
            f"RadGenome reports not found at {root.resolve()}. "
            f"Run `python scripts/download_radgenome_reports.py` first."
        )
    subsets = subsets or RADGENOME_SUBSETS
    out: list[dict[str, str]] = []
    for subset in subsets:
        path = root / subset / f"{section}.json"
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            continue
        for subj_id, payload in data.items():
            # Three known layouts:
            # 1. global_finding / modal_wise_finding: payload is the report string
            # 2. impression: payload is {"disease": [...], "impression": "..."}
            text: str | None = None
            disease_labels: list[str] = []
            if isinstance(payload, str):
                text = payload
            elif isinstance(payload, dict):
                text = payload.get("impression") or payload.get("finding") or payload.get("report")
                if isinstance(payload.get("disease"), list):
                    disease_labels = [str(d) for d in payload["disease"]]
            if not text or not text.strip():
                continue
            entry: dict[str, str] = {
                "subject_id": subj_id,
                "report": text,
                "source": "RadGenome-Brain_MRI",
                "subset": subset,
                "disease": RADGENOME_DISEASE.get(subset, "unknown"),
                "section": section,
                "license": "research-only (AutoRG-Brain repo)",
            }
            if disease_labels:
                entry["disease_labels"] = "|".join(disease_labels)
            out.append(entry)
    if limit:
        out = out[:limit]
    return out


def radgenome_reports_only(
    root: str | Path = DEFAULT_RADGENOME_ROOT,
    section: str = "global_finding",
    subsets: tuple[str, ...] | None = None,
    limit: int | None = None,
) -> list[str]:
    """Flat list of RadGenome reports for `run_benchmark`."""
    return [r["report"] for r in load_radgenome(root=root, section=section,
                                                 subsets=subsets, limit=limit)]
