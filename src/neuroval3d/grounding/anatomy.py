"""Stage 6 — anatomical anchoring.

Given a tumor centroid (in image-space voxel coordinates) and a registration to MNI152,
returns the AAL v3 anatomical region label and a confidence score. For the Phase-0 build
we use a small built-in heuristic (axis-thirds → coarse lobe label) so the pipeline is
runnable without downloading the AAL atlas.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AnchoredSpan:
    sentence: str
    region_label: str
    voxel: tuple[int, int, int]
    confidence: float


class AnatomicalAnchorer:
    """Map (centroid, image_shape) → coarse anatomical region label.

    Replace with a real AAL v3 lookup once `data/raw/atlases/AAL3.nii.gz` is downloaded.
    """

    REGIONS = {
        "frontal_left", "frontal_right",
        "parietal_left", "parietal_right",
        "temporal_left", "temporal_right",
        "occipital_left", "occipital_right",
        "cerebellar_left", "cerebellar_right",
        "brainstem_central",
        "thalamus_left", "thalamus_right",
        "insular_left", "insular_right",
    }

    def anchor(
        self,
        sentence: str,
        centroid: tuple[float, float, float],
        image_shape: tuple[int, int, int],
    ) -> AnchoredSpan:
        cx, cy, cz = centroid
        dx, dy, dz = image_shape
        side = "left" if cx >= dx / 2 else "right"

        # Coarse axial-third assignment.
        third_z = cz / dz
        third_y = cy / dy

        if third_z < 0.20:
            region = "brainstem"
            side_suffix = "central"
        elif third_z < 0.30:
            region = "cerebellar"
            side_suffix = side
        elif third_z < 0.55:
            region = "temporal" if third_y < 0.5 else "occipital"
            side_suffix = side
        elif third_z < 0.85:
            region = "frontal" if third_y < 0.5 else "parietal"
            side_suffix = side
        else:
            region = "frontal"
            side_suffix = side

        label = f"{region}_{side_suffix}"
        # Confidence proxy: distance-from-axis-boundary normalized to [0, 1]
        confidence = float(min(abs(third_z - 0.5), abs(third_y - 0.5)) * 2.0)
        confidence = max(0.5, 1.0 - confidence * 0.5)

        return AnchoredSpan(
            sentence=sentence,
            region_label=label,
            voxel=(int(cx), int(cy), int(cz)),
            confidence=confidence,
        )
