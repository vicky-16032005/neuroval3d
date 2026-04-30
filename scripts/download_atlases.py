"""Download the small atlases (SRI24, MNI152, AAL v3) used by Stages 1 + 6.

Skeleton — fill in the actual URLs and checksums in Phase 1. We deliberately do not
ship atlas binaries in this repo.
"""
from __future__ import annotations

from pathlib import Path

ATLAS_TARGET = Path("data/raw/atlases")

PLANNED_ATLASES = {
    "SRI24": "https://www.nitrc.org/projects/sri24",
    "MNI152": "https://www.bic.mni.mcgill.ca/ServicesAtlases/ICBM152NLin2009",
    "AAL3": "https://www.gin.cnrs.fr/en/tools/aal/",
}


def main() -> None:
    ATLAS_TARGET.mkdir(parents=True, exist_ok=True)
    print(f"Atlases will be placed under {ATLAS_TARGET}")
    for name, url in PLANNED_ATLASES.items():
        target = ATLAS_TARGET / name
        target.mkdir(parents=True, exist_ok=True)
        manifest = target / "MANIFEST.txt"
        manifest.write_text(
            f"{name}\nofficial source: {url}\n"
            f"download instructions: register at the source page and place files here.\n",
            encoding="utf-8",
        )
        print(f"  {name}: {target}/MANIFEST.txt — see source page for download steps")


if __name__ == "__main__":
    main()
