"""Download TextBraTS reports from HuggingFace Jupitern52/TextBraTS.

The dataset has 369 paired reports (one per BraTS 2020 training subject), each in
`TextBraTSData/BraTS20_Training_XXX/BraTS20_Training_XXX_flair_text.txt`.

Saves them to `data/raw/TextBraTS/reports/<subject_id>.txt` for downstream use.
Idempotent: skips files that already exist.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

OUT_DIR = Path("data/raw/TextBraTS/reports")


def main() -> int:
    try:
        from huggingface_hub import hf_hub_download, list_repo_files
    except ImportError:
        print("[!] huggingface_hub not installed. Run: uv sync --extra data")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    repo_id = "Jupitern52/TextBraTS"

    print(f"Listing files in {repo_id}...")
    files = list_repo_files(repo_id, repo_type="dataset")
    txt_files = [f for f in files if f.endswith("_flair_text.txt")]
    print(f"Found {len(txt_files)} report files.")

    token = os.environ.get("HF_TOKEN") or None  # public dataset, token optional
    n_done = 0
    n_skip = 0
    n_fail = 0

    for f in sorted(txt_files):
        subject = Path(f).parent.name      # BraTS20_Training_XXX
        out_path = OUT_DIR / f"{subject}.txt"
        if out_path.exists() and out_path.stat().st_size > 0:
            n_skip += 1
            continue
        try:
            local = hf_hub_download(
                repo_id=repo_id, filename=f, repo_type="dataset", token=token,
                local_dir=Path("data/raw/TextBraTS/_hf_cache"),
            )
            content = Path(local).read_text(encoding="utf-8")
            out_path.write_text(content, encoding="utf-8")
            n_done += 1
            if n_done % 50 == 0:
                print(f"  ...{n_done} downloaded")
        except Exception as e:  # noqa: BLE001
            n_fail += 1
            print(f"  [!] failed {f}: {e}")

    print(f"\nDone: {n_done} downloaded, {n_skip} already-present, {n_fail} failed.")
    print(f"Total reports under {OUT_DIR}: {len(list(OUT_DIR.glob('*.txt')))}")
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
