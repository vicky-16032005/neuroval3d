"""Download BraTS 2020 volumes from the Kaggle community mirror.

Path of least resistance for volume access — no IRB, no DUC, just a Kaggle account.
Mirror: https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation

Prerequisites:
    1. Kaggle account at https://www.kaggle.com (free)
    2. Mint an API token at https://www.kaggle.com/settings/account → "Create New Token"
       This downloads kaggle.json — place it at C:\\Users\\Admin\\.kaggle\\kaggle.json
       (or set KAGGLE_USERNAME + KAGGLE_KEY env vars in .env)
    3. pip install kaggle  (or: uv add kaggle)

Run:
    python scripts/download_brats_kaggle.py

Outputs to data/raw/BraTS2020-kaggle/. About 10 GB.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

OUT_DIR = Path("data/raw/BraTS2020-kaggle")
KAGGLE_DATASET = "awsaf49/brats20-dataset-training-validation"


def main() -> int:
    try:
        import kaggle  # noqa: F401
    except ImportError:
        print("[!] kaggle not installed. Run: uv add kaggle  OR  pip install kaggle")
        return 1

    cred_path = Path.home() / ".kaggle" / "kaggle.json"
    if not cred_path.exists() and not (os.environ.get("KAGGLE_KEY") and os.environ.get("KAGGLE_USERNAME")):
        print(f"[!] Kaggle credentials missing. Either:")
        print(f"    - place kaggle.json at {cred_path}")
        print(f"    - or set KAGGLE_USERNAME + KAGGLE_KEY in .env")
        return 2

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {KAGGLE_DATASET} → {OUT_DIR}/")
    print("This is ~10 GB. Will skip if already present.")

    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(KAGGLE_DATASET, path=str(OUT_DIR), unzip=True, quiet=False)

    n_subjects = len(list(OUT_DIR.rglob("BraTS20_Training_*")))
    print(f"\n[ok] {n_subjects} subject directories under {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
