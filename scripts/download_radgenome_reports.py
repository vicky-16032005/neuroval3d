"""Download RadGenome-Brain MRI reports from HuggingFace JiayuLei/RadGenome-Brain_MRI.

Structure on HF:
    BraTS_GLI/  global_finding.json  impression.json  modal_wise_finding.json   (glioma)
    BraTS_MEN/  ...                                                              (meningioma)
    BraTS_MET/  ...                                                              (metastasis)
    ISLES22/    ...                                                              (stroke)
    WMH/        ...                                                              (white matter hyperintensities)
    train_val_test_split.json

We pull all JSON files (text reports only — no volumes) into
`data/raw/RadGenome-BrainMRI/`. Total size: a few MB.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

OUT_DIR = Path("data/raw/RadGenome-BrainMRI")
REPO_ID = "JiayuLei/RadGenome-Brain_MRI"


def main() -> int:
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError:
        print("[!] huggingface_hub not installed. Run: uv sync --extra data")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = list_repo_files(REPO_ID, repo_type="dataset")
    json_files = [f for f in files if f.endswith(".json")]
    print(f"Found {len(json_files)} JSON files in {REPO_ID}.")

    token = os.environ.get("HF_TOKEN") or None
    n_done = 0
    for f in sorted(json_files):
        try:
            local = hf_hub_download(
                repo_id=REPO_ID, filename=f, repo_type="dataset", token=token,
                local_dir=str(OUT_DIR / "_hf_cache"),
            )
            target = OUT_DIR / f
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(Path(local).read_bytes())
            n_done += 1
            print(f"  {f}: {target.stat().st_size:,} bytes")
        except Exception as e:  # noqa: BLE001
            print(f"  [!] failed {f}: {e}")

    print(f"\nDone: {n_done}/{len(json_files)} files under {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
