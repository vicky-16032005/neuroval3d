"""Phase 1 prep — scan `data/raw/` and report which BraTS subsets are present.

Run after the user has registered at Synapse and downloaded any BraTS subset:
    python scripts/check_brats.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

EXPECTED_MODALITIES = {"t1", "t1ce", "t2", "flair"}
NIFTI_PATTERN = re.compile(r"_(t1|t1ce|t2|flair|seg)\.nii(\.gz)?$", re.IGNORECASE)


def scan(root: Path) -> dict[str, list[Path]]:
    if not root.exists():
        return {}
    out: dict[str, list[Path]] = {}
    for p in sorted(root.rglob("*.nii*")):
        if NIFTI_PATTERN.search(p.name):
            subj = p.parent.name
            out.setdefault(subj, []).append(p)
    return out


def main() -> int:
    raw = Path("data/raw")
    if not raw.exists():
        print(f"[!] data/raw not found at {raw.resolve()} — nothing to check.")
        return 1

    targets = [d for d in raw.iterdir() if d.is_dir()]
    if not targets:
        print(f"[!] data/raw is empty. Register at https://www.synapse.org/ and download BraTS.")
        return 1

    grand_total_complete = 0
    for tgt in targets:
        subjects = scan(tgt)
        complete = 0
        partial = 0
        for files in subjects.values():
            modalities = {NIFTI_PATTERN.search(f.name).group(1).lower() for f in files}  # type: ignore
            if EXPECTED_MODALITIES.issubset(modalities):
                complete += 1
            elif modalities & EXPECTED_MODALITIES:
                partial += 1
        print(f"  {tgt.name}: {complete} complete subjects ({partial} partial)")
        grand_total_complete += complete

    print(f"\nTotal complete subjects across all sets: {grand_total_complete}")
    if grand_total_complete == 0:
        print("\n[!] No complete BraTS subjects found. Check the directory structure:")
        print("    data/raw/<dataset_name>/<subject_id>/<subject_id>_t1.nii.gz, _t1ce.nii.gz, _t2.nii.gz, _flair.nii.gz")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
