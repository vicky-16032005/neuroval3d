"""Stage 5c — structural validator: VASARI feature-level F1 against a ground-truth feature vector.

If a segmentation mask is available, we derive the ground-truth VASARI feature vector via
the SyntheticReportGenerator's feature extractor, then parse the generated report into its
own feature vector and compute F1 over the union of features.

If no segmentation is available, we fall back to parser-vs-parser F1 between the generated
and reference report (same approach, less ground-truthy).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neuroval3d.grounding.vasari import VASARIFeatureVector, VASARIParser


@dataclass
class StructuralValidatorConfig:
    feature_subset: tuple[str, ...] | None = None
    """If set, restrict scoring to these VASARI short codes (e.g. ('side', 'tumor_location'))."""


class StructuralValidator:
    """Computes VASARI feature-level F1, with two backends:

    1. (preferred) generated_report ↔ mask_derived_feature_vector
    2. (fallback)  generated_report ↔ reference_report (both parsed)
    """

    def __init__(self, config: StructuralValidatorConfig | None = None) -> None:
        self.config = config or StructuralValidatorConfig()
        self.parser = VASARIParser()

    # ------------------------------------------------------------------ public
    def score_against_mask(self, generated: str, mask: np.ndarray, voxel_volume_mm3: float = 1.0) -> float:
        from neuroval3d.data.synthetic import SyntheticReportGenerator

        feats_gt = SyntheticReportGenerator()._extract_features(mask, voxel_volume_mm3)
        feats_gt_vec = self._to_vector(feats_gt)
        feats_gen_vec = self.parser.parse(generated)
        return self._f1(feats_gen_vec, feats_gt_vec)

    def score(self, generated: str, reference: str) -> float:
        feats_gen_vec = self.parser.parse(generated)
        feats_ref_vec = self.parser.parse(reference)
        return self._f1(feats_gen_vec, feats_ref_vec)

    # ------------------------------------------------------------------ internals
    def _to_vector(self, feats: dict[str, str]) -> VASARIFeatureVector:
        out: dict[str, str] = {}
        # Keep keys that line up with VASARIFeature.short
        keymap = {
            "side": "side",
            "region": "tumor_location",
            "enhancement": "enhancement_quality",
            "edema": "prop_edema",
            "necrosis": "prop_necrosis",
            "multifocal": "multifocal",
            "crosses_midline": "et_crosses_midline",
        }
        # Convert categorical bins into VASARI canonical values where straightforward
        if feats.get("enhancement"):
            v = feats["enhancement"]
            if v == "non-enhancing": out["enhancement_quality"] = "none"
            elif v == "minimal enhancement": out["enhancement_quality"] = "mild"
            elif v == "moderate enhancement": out["enhancement_quality"] = "mild"
            elif v == "avid enhancement": out["enhancement_quality"] = "marked"
        if feats.get("edema"):
            v = feats["edema"]
            out["prop_edema"] = {"absent": "0", "minimal": "1-25", "moderate": "26-50", "marked": "51-75"}.get(v, "26-50")
        if feats.get("necrosis"):
            v = feats["necrosis"]
            out["prop_necrosis"] = {"absent": "0", "minimal": "1-25", "moderate": "26-50", "extensive": "76-100"}.get(v, "1-25")
        if feats.get("region"):
            v = feats["region"]
            out["tumor_location"] = v.lower() if v in {"frontal", "parietal", "temporal", "occipital", "cerebellar", "brainstem"} else "frontal"
        if feats.get("side"):
            v = feats["side"]
            if v in {"left", "right", "bilateral"}: out["side"] = v
        if feats.get("multifocal"):
            v = feats["multifocal"]
            out["multifocal"] = {"yes": "multifocal", "no": "focal"}.get(v, "focal")
        if feats.get("crosses_midline"):
            out["et_crosses_midline"] = feats["crosses_midline"]
        del keymap  # unused alias map, kept for clarity
        return VASARIFeatureVector(values=out)

    def _f1(self, a: VASARIFeatureVector, b: VASARIFeatureVector) -> float:
        if self.config.feature_subset is None:
            return a.f1_against(b)
        sub = set(self.config.feature_subset)
        a_sub = VASARIFeatureVector(values={k: v for k, v in a.values.items() if k in sub})
        b_sub = VASARIFeatureVector(values={k: v for k, v in b.values.items() if k in sub})
        return a_sub.f1_against(b_sub)
