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
