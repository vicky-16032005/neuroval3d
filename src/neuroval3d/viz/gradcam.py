"""Stage 7 — 3D Grad-CAM. Skeleton; full impl wires into the encoder during Phase 2."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GradCAMConfig:
    target_layer: str = "encoder.layers3.0"  # MONAI Swin-UNETR conv3 block
    upsample: bool = True
    smooth_sigma: float = 1.0


class GradCAM3D:
    """Compute a 3D class-activation heatmap for a given encoder + input volume.

    For Phase 0 this is an interface stub; the working implementation hooks
    forward/backward gradients on the chosen layer during Phase 2.
    """

    def __init__(self, model=None, config: GradCAMConfig | None = None) -> None:
        self.model = model
        self.config = config or GradCAMConfig()

    def attribute(self, volume, target=None):
        import numpy as np
        # Skeleton: returns a uniform heatmap. Replace during Phase 2.
        if hasattr(volume, "shape"):
            shape = tuple(volume.shape[-3:])
        else:
            shape = (64, 64, 64)
        return np.ones(shape, dtype="float32") / float(np.prod(shape))
