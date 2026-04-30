"""Stage 7 — coronal/axial/sagittal triptych overlay rendering.

Saves a single PNG with three orthogonal mid-slices, the volume in greyscale and the heatmap
as a translucent magma colormap. Useful in notebooks and in the paper.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def save_triptych(
    volume: np.ndarray,
    heatmap: np.ndarray | None,
    out_path: str | Path,
    title: str = "",
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        # Smoke-test fallback: just dump the shape info as a tiny manifest.
        out_path.with_suffix(".txt").write_text(
            f"volume_shape={volume.shape}, heatmap={'yes' if heatmap is not None else 'no'}",
            encoding="utf-8",
        )
        return out_path

    if volume.ndim == 4:
        volume = volume[0]  # take first channel
    D, H, W = volume.shape
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(volume[D // 2], cmap="gray")
    axes[0].set_title("Axial")
    axes[1].imshow(volume[:, H // 2, :], cmap="gray", aspect="auto")
    axes[1].set_title("Coronal")
    axes[2].imshow(volume[:, :, W // 2], cmap="gray", aspect="auto")
    axes[2].set_title("Sagittal")

    if heatmap is not None:
        if heatmap.ndim == 4:
            heatmap = heatmap[0]
        axes[0].imshow(heatmap[D // 2], cmap="magma", alpha=0.4)
        axes[1].imshow(heatmap[:, H // 2, :], cmap="magma", alpha=0.4, aspect="auto")
        axes[2].imshow(heatmap[:, :, W // 2], cmap="magma", alpha=0.4, aspect="auto")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path
