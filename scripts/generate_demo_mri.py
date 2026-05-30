"""Generate representative 4-modality brain-MRI axial-slice panels for the dashboard demo.

These are PROCEDURAL phantom slices (not the raw BraTS voxels, which live on Kaggle).
Each panel shows T1, T1ce, T2, FLAIR for one held-out subject with the tumour placed on
the side stated in that subject's reference report. They are clearly labelled in the UI as
representative reconstructions so the demo flow (input -> report -> validation) is legible.

Output: dashboard/assets/<subject_id>.png  (one 4-up panel per subject)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "dashboard" / "assets"
OUT.mkdir(parents=True, exist_ok=True)

H = W = 220

# subject_id -> (lesion_side, lesion_region, seed)
# side: "left" or "right" in radiological terms; on an axial slice viewed from below,
# patient-left is on the image's RIGHT. We place accordingly for realism.
SUBJECTS = {
    "BraTS20_Training_081": ("left", "frontal", 81),
    "BraTS20_Training_094": ("left", "parietal", 94),
    "BraTS20_Training_096": ("right", "frontal", 96),
    "BraTS20_Training_098": ("right", "parietal", 98),
}

MODALITIES = ["T1", "T1ce", "T2", "FLAIR"]


def _coords():
    yy, xx = np.mgrid[0:H, 0:W]
    cx, cy = W / 2.0, H / 2.0
    return xx, yy, cx, cy


def _ellipse_mask(xx, yy, cx, cy, rx, ry):
    return ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0


def build_brain(side: str, region: str, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    xx, yy, cx, cy = _coords()

    skull_outer = _ellipse_mask(xx, yy, cx, cy, 92, 108)
    skull_inner = _ellipse_mask(xx, yy, cx, cy, 84, 100)
    brain = _ellipse_mask(xx, yy, cx, cy, 80, 96)

    # parenchyma texture
    texture = rng.normal(0.0, 1.0, (H, W))
    # light blur via separable box
    for _ in range(3):
        texture = (texture
                   + np.roll(texture, 1, 0) + np.roll(texture, -1, 0)
                   + np.roll(texture, 1, 1) + np.roll(texture, -1, 1)) / 5.0

    # lateral ventricles (two slanted ellipses near centre)
    vent_l = _ellipse_mask(xx, yy, cx - 14, cy - 4, 10, 26)
    vent_r = _ellipse_mask(xx, yy, cx + 14, cy - 4, 10, 26)
    ventricles = vent_l | vent_r

    # lesion placement: patient-left -> image right (+x); patient-right -> image left (-x)
    sx = +1 if side == "left" else -1
    region_dy = {"frontal": -34, "parietal": +18, "temporal": +30, "occipital": +48}.get(region, -20)
    lx = cx + sx * 34
    ly = cy + region_dy
    lesion_core = _ellipse_mask(xx, yy, lx, ly, 15, 13)
    lesion_rim = _ellipse_mask(xx, yy, lx, ly, 20, 18) & ~lesion_core
    edema = _ellipse_mask(xx, yy, lx, ly, 34, 31) & ~_ellipse_mask(xx, yy, lx, ly, 20, 18)

    out: dict[str, np.ndarray] = {}
    for mod in MODALITIES:
        img = np.zeros((H, W), dtype=np.float32)
        # base parenchyma
        paren = 0.45 + 0.05 * texture
        img[brain] = paren[brain]
        # grey/white differentiation ring
        gw = _ellipse_mask(xx, yy, cx, cy, 80, 96) & ~_ellipse_mask(xx, yy, cx, cy, 62, 74)
        img[gw] += 0.06

        if mod in ("T1", "T1ce"):
            img[ventricles] = 0.12          # CSF dark on T1
            img[edema] = paren[edema] - 0.05
            img[lesion_core] = 0.20
            img[lesion_rim] = 0.55 if mod == "T1" else 0.92  # enhancement on T1ce
            skull_val = 0.30
        elif mod == "T2":
            img[ventricles] = 0.95          # CSF bright on T2
            img[edema] = 0.80
            img[lesion_core] = 0.70
            img[lesion_rim] = 0.85
            skull_val = 0.15
        else:  # FLAIR
            img[ventricles] = 0.18          # CSF suppressed
            img[edema] = 0.92               # edema very bright
            img[lesion_core] = 0.55
            img[lesion_rim] = 0.80
            skull_val = 0.12

        # skull ring
        img[skull_inner & ~brain] = 0.0
        img[skull_outer & ~skull_inner] = skull_val
        img[~skull_outer] = 0.0
        # gentle noise
        img[brain] += rng.normal(0, 0.015, img[brain].shape)
        out[mod] = np.clip(img, 0, 1)
    return out


def render_panel(subject_id: str, side: str, region: str, seed: int) -> Path:
    mods = build_brain(side, region, seed)
    fig, axes = plt.subplots(1, 4, figsize=(8.4, 2.3))
    for ax, mod in zip(axes, MODALITIES):
        ax.imshow(mods[mod], cmap="gray", vmin=0, vmax=1)
        ax.set_title(mod, fontsize=11, color="#0f172a", fontweight="bold", pad=4)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor("#cbd5e1")
    fig.suptitle("", y=0.0)
    plt.tight_layout(pad=0.4)
    dst = OUT / f"{subject_id}.png"
    fig.savefig(dst, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return dst


def main() -> None:
    for sid, (side, region, seed) in SUBJECTS.items():
        p = render_panel(sid, side, region, seed)
        print(f"wrote {p}  ({p.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
