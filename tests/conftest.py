"""Pytest fixtures for NeuroVal-3D smoke tests."""
from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def small_seg_mask() -> np.ndarray:
    """Tiny BraTS-shaped segmentation mask with a synthetic lesion."""
    D = 32
    mask = np.zeros((D, D, D), dtype=np.int16)
    cz, cy, cx = 16, 14, 18
    zz, yy, xx = np.ogrid[:D, :D, :D]
    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2
    mask[dist2 <= 9 ** 2] = 2  # edema
    mask[dist2 <= 5 ** 2] = 4  # enhancing
    mask[dist2 <= 2 ** 2] = 1  # necrosis
    return mask


@pytest.fixture
def report_pair() -> tuple[str, str]:
    gen = (
        "FINDINGS:\n"
        "- There is a 3.5 cm intra-axial lesion in the right frontal lobe with avid enhancement.\n"
        "- Surrounding T2/FLAIR hyperintensity reflects moderate vasogenic oedema.\n"
        "- No restricted diffusion or hemorrhage.\n\n"
        "IMPRESSION:\n"
        "3.5 cm enhancing intra-axial mass in the right frontal lobe with moderate oedema, "
        "consistent with high-grade glioma."
    )
    ref = (
        "FINDINGS:\n"
        "- A 3.5 cm right frontal intra-axial mass demonstrates avid post-contrast enhancement.\n"
        "- Moderate peritumoral edema is present.\n"
        "- Diffusion-weighted imaging shows no restricted diffusion.\n\n"
        "IMPRESSION:\n"
        "Right frontal high-grade glial neoplasm, 3.5 cm, with moderate edema."
    )
    return gen, ref
