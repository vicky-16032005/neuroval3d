"""Synthetic-report generator — turns BraTS-style segmentation masks into VASARI-templated reports.

Used for two purposes:
1. Expanding TextBraTS (~369) to the larger BraTS21/23 corpus (~2,700) without radiologist time.
2. Generating reproducible "ground-truth" reports for the Stage 8 perturbation benchmark.

The templating is rule-based and deterministic: same mask + same config → same report. We do NOT
use an LLM here, because we want determinism and zero API cost. The TextBraTS authors used GPT-4o
for richness; we trade richness for reproducibility.

BraTS label convention (2020/2021/2023):
    0 = background
    1 = NCR / NET (necrotic + non-enhancing tumor core)
    2 = ED (peritumoral edema)
    4 = ET (enhancing tumor)
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class SyntheticReport:
    text: str
    findings: list[str]
    impression: str
    vasari_features: dict[str, str]
    seed: int


class SyntheticReportGenerator:
    """Mask-conditional templated radiology report generator.

    The output is structured into Findings + Impression sections, mirroring radiology
    convention. Text is sampled from a small phrasebook that is *deterministic* given the
    feature vector; this gives reproducible synthetic data without an LLM.
    """

    LABELS = {0: "background", 1: "necrotic_core", 2: "edema", 4: "enhancing_tumor"}

    REGION_NAMES = ["frontal", "parietal", "temporal", "occipital", "cerebellar", "brainstem"]

    def __init__(self, modality_set: tuple[str, ...] = ("T1", "T1ce", "T2", "FLAIR")) -> None:
        self.modalities = modality_set

    # ------------------------------------------------------------------ public
    def from_mask(self, seg: np.ndarray, voxel_volume_mm3: float = 1.0) -> SyntheticReport:
        feats = self._extract_features(seg, voxel_volume_mm3)
        seed = int(hashlib.md5(seg.tobytes()).hexdigest()[:8], 16) % (2**31)
        rng = np.random.default_rng(seed)

        findings = self._compose_findings(feats, rng)
        impression = self._compose_impression(feats)
        text = "FINDINGS:\n" + "\n".join(f"- {f}" for f in findings) + f"\n\nIMPRESSION:\n{impression}"

        return SyntheticReport(
            text=text,
            findings=findings,
            impression=impression,
            vasari_features=feats,
            seed=seed,
        )

    def from_specs(self, specs: Iterable[dict[str, object]]) -> list[SyntheticReport]:
        out: list[SyntheticReport] = []
        for s in specs:
            mask = s["mask"]
            assert isinstance(mask, np.ndarray)
            vox = float(s.get("voxel_volume_mm3", 1.0))  # type: ignore[arg-type]
            out.append(self.from_mask(mask, vox))
        return out

    # ------------------------------------------------------------------ internals
    def _extract_features(self, seg: np.ndarray, vox_mm3: float) -> dict[str, str]:
        """Compute VASARI-style features from a BraTS mask."""
        feats: dict[str, str] = {}

        ncr_vol = float((seg == 1).sum() * vox_mm3)
        ed_vol = float((seg == 2).sum() * vox_mm3)
        et_vol = float((seg == 4).sum() * vox_mm3)
        total_tumor = ncr_vol + et_vol
        total_lesion = total_tumor + ed_vol

        if total_lesion < 1e-6:
            feats.update({
                "tumor_present": "no",
                "side": "n/a",
                "region": "n/a",
                "enhancement": "none",
                "edema": "none",
                "necrosis": "none",
                "size_cm": "0.0",
            })
            return feats

        # Side: based on centroid X coordinate (assumes RAS orientation). For BraTS data in default
        # BraTS coordinate space, X<dim/2 is right hemisphere; >= dim/2 is left.
        coords = np.argwhere(seg > 0)
        cx = coords[:, 0].mean() if coords.size else seg.shape[0] / 2
        cy = coords[:, 1].mean() if coords.size else seg.shape[1] / 2
        cz = coords[:, 2].mean() if coords.size else seg.shape[2] / 2
        side = "left" if cx >= seg.shape[0] / 2 else "right"

        # Region: crude lobe by centroid position along axes.
        region = self._region_from_centroid((cx, cy, cz), seg.shape)

        # Enhancement proportion
        enh_prop = et_vol / max(total_tumor, 1e-6)
        if enh_prop < 0.05:
            enhancement = "non-enhancing"
        elif enh_prop < 0.34:
            enhancement = "minimal enhancement"
        elif enh_prop < 0.67:
            enhancement = "moderate enhancement"
        else:
            enhancement = "avid enhancement"

        # Necrosis
        necrosis_prop = ncr_vol / max(total_tumor, 1e-6)
        necrosis = "absent" if necrosis_prop < 0.05 else (
            "minimal" if necrosis_prop < 0.34 else
            "moderate" if necrosis_prop < 0.67 else "extensive"
        )

        # Edema
        edema_prop = ed_vol / max(total_lesion, 1e-6)
        edema = "absent" if edema_prop < 0.05 else (
            "minimal" if edema_prop < 0.34 else
            "moderate" if edema_prop < 0.67 else "marked"
        )

        # Size (longest axis through tumor)
        if coords.size:
            ranges = coords.max(0) - coords.min(0)
            size_voxels = float(ranges.max())
            size_cm = round(size_voxels * (vox_mm3 ** (1 / 3)) / 10, 2)
        else:
            size_cm = 0.0

        # Multifocality
        from scipy import ndimage  # local import to keep top-level light
        try:
            tumor_mask = (seg == 1) | (seg == 4)
            _, n_components = ndimage.label(tumor_mask)
            multifocal = "no" if n_components <= 1 else "yes"
        except ImportError:
            multifocal = "unknown"

        # Midline crossing — does tumor mask span both halves of axis 0?
        if coords.size:
            spans_midline = bool((coords[:, 0] < seg.shape[0] / 2).any()
                                 and (coords[:, 0] >= seg.shape[0] / 2).any())
        else:
            spans_midline = False

        feats.update({
            "tumor_present": "yes",
            "side": side,
            "region": region,
            "enhancement": enhancement,
            "edema": edema,
            "necrosis": necrosis,
            "size_cm": str(size_cm),
            "multifocal": multifocal,
            "crosses_midline": "yes" if spans_midline else "no",
            "enhancing_volume_mm3": f"{et_vol:.0f}",
            "edema_volume_mm3": f"{ed_vol:.0f}",
            "necrosis_volume_mm3": f"{ncr_vol:.0f}",
        })
        return feats

    @staticmethod
    def _region_from_centroid(centroid: tuple[float, float, float], shape: tuple[int, ...]) -> str:
        cx, cy, cz = centroid
        dx, dy, dz = shape[:3]
        # Crude lobar mapping (BraTS axes ≈ RAS):
        #   axial third low → cerebellum/brainstem
        #   axial third middle → temporal/occipital depending on AP
        #   axial third high → frontal/parietal depending on AP
        third_z = cz / dz
        third_y = cy / dy
        if third_z < 0.30:
            return "cerebellar" if third_y < 0.5 else "brainstem"
        if third_z < 0.65:
            return "temporal" if third_y < 0.5 else "occipital"
        return "frontal" if third_y < 0.5 else "parietal"

    def _compose_findings(self, feats: dict[str, str], rng: np.random.Generator) -> list[str]:
        if feats["tumor_present"] == "no":
            return [
                "No discrete intra-axial mass lesion is identified.",
                "No abnormal parenchymal enhancement.",
                "No restricted diffusion or hemorrhage.",
                "Ventricles are normal in size; no midline shift.",
            ]

        out: list[str] = []
        out.append(
            f"There is a {feats['size_cm']} cm intra-axial lesion centred in the {feats['side']} "
            f"{feats['region']} lobe."
        )
        out.append(
            f"The lesion demonstrates {feats['enhancement']} on post-contrast T1 imaging."
        )
        if feats["edema"] != "absent":
            out.append(f"Surrounding T2/FLAIR hyperintensity reflects {feats['edema']} vasogenic oedema.")
        else:
            out.append("There is no significant peritumoral oedema.")

        if feats["necrosis"] != "absent":
            out.append(f"Central low-T1 / high-T2 signal is consistent with {feats['necrosis']} necrosis.")
        if feats.get("multifocal") == "yes":
            out.append("Additional satellite lesions are noted, indicating multifocal disease.")
        if feats.get("crosses_midline") == "yes":
            out.append("The mass extends across the midline.")

        # Random radiologist-style fillers, deterministically selected by seed
        fillers = [
            "Cortical sulci adjacent to the lesion appear effaced.",
            "There is no evidence of acute hemorrhage on susceptibility imaging.",
            "Diffusion-weighted imaging does not show restricted diffusion.",
            "The remainder of the parenchyma is unremarkable.",
            "No leptomeningeal enhancement is identified.",
        ]
        idxs = rng.choice(len(fillers), size=2, replace=False)
        out.extend(fillers[i] for i in idxs)
        return out

    @staticmethod
    def _compose_impression(feats: dict[str, str]) -> str:
        if feats["tumor_present"] == "no":
            return "Unremarkable brain MRI. No acute intracranial abnormality."
        return (
            f"{feats['size_cm']} cm {feats['enhancement'].replace(' enhancement', '')} "
            f"intra-axial mass in the {feats['side']} {feats['region']} lobe with "
            f"{feats['edema']} surrounding oedema. Findings are most consistent with a "
            "high-grade glial neoplasm; recommend correlation with clinical context and "
            "consideration of histopathological confirmation."
        )
