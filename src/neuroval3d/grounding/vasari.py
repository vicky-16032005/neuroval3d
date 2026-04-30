"""VASARI feature lexicon, parser, and vocabulary export.

VASARI (Visually AcceSAble Rembrandt Images) is the consensus controlled vocabulary for
glioma reporting. We use the 30-feature key as published in:
- Wiki for the VASARI feature set: https://wiki.cancerimagingarchive.net/display/Public/VASARI+Research+Project
- VASARI 2.0 (Frontiers in Oncology 2024)
- "Ten Years of VASARI Glioma Features" (AJNR 2024)

This module is the single source of truth for VASARI in the project. The validators (lexical,
structural) and the perturbation benchmark all import from here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import ClassVar


@dataclass(frozen=True)
class VASARIFeature:
    """A single VASARI feature definition."""

    code: str                    # e.g. "F1"
    name: str                    # e.g. "Tumor Location"
    short: str                   # e.g. "tumor_location"
    values: tuple[str, ...]      # canonical categorical values
    aliases: dict[str, tuple[str, ...]] = field(default_factory=dict)  # value → free-text patterns


# ---------------------------------------------------------------------------
# Canonical 30-feature VASARI lexicon (abbreviated where the value list would
# explode; aliases capture common report phrasing).
# ---------------------------------------------------------------------------

VASARI_FEATURES: tuple[VASARIFeature, ...] = (
    VASARIFeature(
        code="F1", name="Tumor Location", short="tumor_location",
        values=("frontal", "parietal", "temporal", "occipital", "insular",
                "thalamus", "basal_ganglia", "brainstem", "cerebellum", "corpus_callosum"),
        aliases={
            "frontal": ("frontal lobe", "frontal"),
            "parietal": ("parietal lobe", "parietal"),
            "temporal": ("temporal lobe", "temporal"),
            "occipital": ("occipital lobe", "occipital"),
            "insular": ("insular", "insula"),
            "brainstem": ("brainstem", "pons", "medulla", "midbrain"),
            "cerebellum": ("cerebellar", "cerebellum"),
            "corpus_callosum": ("corpus callosum", "splenium", "genu"),
        },
    ),
    VASARIFeature(
        code="F2", name="Side of Tumor Epicenter", short="side",
        values=("left", "right", "bilateral", "central"),
        aliases={
            "left": ("left", "left-sided", "left hemisphere"),
            "right": ("right", "right-sided", "right hemisphere"),
            "bilateral": ("bilateral", "both hemispheres", "midline"),
        },
    ),
    VASARIFeature(
        code="F3", name="Eloquent Brain", short="eloquent",
        values=("none", "speech_motor", "speech_sensory", "motor", "vision"),
    ),
    VASARIFeature(
        code="F4", name="Enhancement Quality", short="enhancement_quality",
        values=("none", "mild", "marked"),
        aliases={
            "none": ("non-enhancing", "no enhancement", "without enhancement"),
            "mild": ("mild enhancement", "minimal enhancement", "faint enhancement"),
            "marked": ("avid enhancement", "marked enhancement", "intense enhancement", "strong enhancement"),
        },
    ),
    VASARIFeature(
        code="F5", name="Proportion Enhancing", short="prop_enhancing",
        values=("0", "1-25", "26-50", "51-75", "76-100"),
    ),
    VASARIFeature(
        code="F6", name="Proportion non-Contrast Enhancing Tumor (nCET)", short="prop_ncet",
        values=("0", "1-25", "26-50", "51-75", "76-100"),
    ),
    VASARIFeature(
        code="F7", name="Proportion Necrosis", short="prop_necrosis",
        values=("0", "1-25", "26-50", "51-75", "76-100"),
        aliases={"0": ("no necrosis", "without necrosis"), "76-100": ("extensive necrosis",)},
    ),
    VASARIFeature(
        code="F8", name="Cysts", short="cysts",
        values=("yes", "no"),
        aliases={"yes": ("cyst", "cystic component"), "no": ("non-cystic",)},
    ),
    VASARIFeature(
        code="F9", name="Multifocal or Multicentric", short="multifocal",
        values=("focal", "multifocal", "multicentric", "gliomatosis"),
        aliases={"focal": ("solitary", "single lesion"),
                 "multifocal": ("multifocal", "multiple lesions", "satellite lesions")},
    ),
    VASARIFeature(
        code="F10", name="T1/FLAIR Ratio", short="t1_flair_ratio",
        values=("expansive", "mixed", "infiltrative"),
    ),
    VASARIFeature(
        code="F11", name="Thickness of Enhancing Margin", short="margin_thickness",
        values=("none", "thin", "thick"),
    ),
    VASARIFeature(
        code="F12", name="Definition of Enhancing Margin", short="enhancing_margin_def",
        values=("well-defined", "poorly-defined"),
        aliases={"well-defined": ("well-circumscribed", "well-marginated"),
                 "poorly-defined": ("ill-defined", "ill-marginated", "infiltrative margin")},
    ),
    VASARIFeature(
        code="F13", name="Definition of the Non-Enhancing Margin", short="ncet_margin_def",
        values=("well-defined", "poorly-defined"),
    ),
    VASARIFeature(
        code="F14", name="Proportion of Edema", short="prop_edema",
        values=("0", "1-25", "26-50", "51-75", "76-100"),
        aliases={"0": ("no edema", "without edema", "no oedema")},
    ),
    VASARIFeature(
        code="F15", name="Edema Crosses Midline", short="edema_crosses_midline",
        values=("yes", "no"),
    ),
    VASARIFeature(
        code="F16", name="Hemorrhage", short="hemorrhage",
        values=("yes", "no"),
        aliases={"yes": ("hemorrhage", "haemorrhage", "blood products", "hemorrhagic"),
                 "no": ("no hemorrhage", "no haemorrhage", "no blood products")},
    ),
    VASARIFeature(
        code="F17", name="Diffusion", short="diffusion",
        values=("facilitated", "restricted"),
        aliases={"restricted": ("restricted diffusion", "diffusion restriction", "low ADC"),
                 "facilitated": ("no restricted diffusion", "facilitated diffusion", "high ADC")},
    ),
    VASARIFeature(
        code="F18", name="Pial Invasion", short="pial_invasion",
        values=("yes", "no"),
    ),
    VASARIFeature(
        code="F19", name="Ependymal Extension", short="ependymal_extension",
        values=("yes", "no"),
    ),
    VASARIFeature(
        code="F20", name="Cortical Involvement", short="cortical_involvement",
        values=("yes", "no"),
    ),
    VASARIFeature(
        code="F21", name="Deep WM Involvement", short="deep_wm_involvement",
        values=("yes", "no"),
    ),
    VASARIFeature(
        code="F22", name="nCET Crosses Midline", short="ncet_crosses_midline",
        values=("yes", "no"),
    ),
    VASARIFeature(
        code="F23", name="Enhancing Tumor Crosses Midline", short="et_crosses_midline",
        values=("yes", "no"),
    ),
    VASARIFeature(
        code="F24", name="Satellites", short="satellites",
        values=("yes", "no"),
    ),
    VASARIFeature(
        code="F25", name="Calvarial Remodeling", short="calvarial_remodeling",
        values=("yes", "no"),
    ),
    VASARIFeature(
        code="F26", name="Lesion Size (longest axis, cm)", short="size_long_cm",
        values=("0-2", "2-4", "4-6", "6-8", ">8"),
    ),
    VASARIFeature(
        code="F27", name="Lesion Size (perpendicular axis, cm)", short="size_perp_cm",
        values=("0-2", "2-4", "4-6", "6-8", ">8"),
    ),
    VASARIFeature(
        code="F28", name="Extent of Resection — enhancing tumor (post-op)", short="resection_et",
        values=("100", "76-99", "0-75", "n/a"),
    ),
    VASARIFeature(
        code="F29", name="Extent of Resection — nCET (post-op)", short="resection_ncet",
        values=("100", "76-99", "0-75", "n/a"),
    ),
    VASARIFeature(
        code="F30", name="Extent of Resection — Edema (post-op)", short="resection_edema",
        values=("100", "76-99", "0-75", "n/a"),
    ),
)


def vasari_vocabulary() -> list[str]:
    """Return the flat token vocabulary for VASARI-restricted TF-IDF.

    The vocabulary covers:
      - VASARI feature short codes + canonical values + aliases (the structured layer)
      - Pathology anchor terms (glioma, infarct, …)
      - Anatomical region terms
      - Modality tokens (T1 / T2 / FLAIR / DWI / ADC …) — needed because Stage 8 perturbations
        flip modalities and the validator has to notice
      - Numeric size descriptors and units (cm, mm, x.x cm …) — also Stage 8 territory
      - Counting/quantifier tokens (one, two, multiple, solitary …) — for count-perturbation detection
    """
    vocab: set[str] = set()
    for feat in VASARI_FEATURES:
        vocab.add(feat.short.replace("_", " "))
        for v in feat.values:
            v_norm = v.replace("_", " ").replace("-", " ")
            vocab.add(v_norm)
        for _value, aliases in feat.aliases.items():
            for a in aliases:
                vocab.add(a.lower())
    # Pathology anchor terms
    vocab.update({
        "glioma", "glioblastoma", "meningioma", "metastasis", "metastases",
        "infarct", "infarction", "ischemic stroke",
        "white matter hyperintensity", "small vessel disease", "leukoaraiosis",
        "midline shift", "mass effect", "ventricular compression",
        "leptomeningeal", "intra-axial", "extra-axial",
    })
    # Anatomical region terms (extended)
    vocab.update({
        "frontal", "parietal", "temporal", "occipital",
        "cerebellar", "cerebellum",
        "insular", "insula",
        "thalamus", "thalamic",
        "basal ganglia",
        "brainstem", "pons", "medulla", "midbrain",
        "corpus callosum", "splenium", "genu",
    })
    # Modality tokens — Stage 8 modality-perturbation requires these to be in the vocab
    vocab.update({
        "t1", "t1ce", "t1-weighted", "post-contrast t1",
        "t2", "t2-weighted", "t2/flair",
        "flair",
        "dwi", "diffusion weighted", "diffusion-weighted",
        "adc", "apparent diffusion coefficient",
        "swi", "susceptibility weighted",
        "mra", "mr angiography",
        "perfusion",
    })
    # Size + unit tokens — Stage 8 size-perturbation requires these
    vocab.update({
        "cm", "mm", "centimeters", "millimeters",
        "small", "large", "tiny", "massive",
        "diameter", "longest axis", "perpendicular axis",
    })
    # Counting + quantifier tokens — Stage 8 count-perturbation requires these
    vocab.update({
        "one", "two", "three", "four", "five", "six", "seven",
        "single", "solitary", "multiple", "few", "several", "numerous",
        "lesion", "lesions", "mass", "masses", "nodule", "nodules",
        "satellite", "satellites",
    })
    return sorted(vocab)


@dataclass
class VASARIFeatureVector:
    """Sparse vector of VASARI feature → value, for use in F1 scoring."""
    values: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, str]:
        return dict(self.values)

    def f1_against(self, other: "VASARIFeatureVector") -> float:
        """F1 over the union of features, treating each feature as a single-class match."""
        keys = set(self.values) | set(other.values)
        if not keys:
            return 1.0
        tp = sum(1 for k in keys if self.values.get(k) == other.values.get(k) and k in self.values and k in other.values)
        fp = sum(1 for k in self.values if k not in other.values or self.values[k] != other.values.get(k))
        fn = sum(1 for k in other.values if k not in self.values or other.values[k] != self.values.get(k))
        if tp == 0:
            return 0.0
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        return 2 * precision * recall / (precision + recall + 1e-9)


class VASARIParser:
    """Rule-based VASARI feature extractor over free-text radiology reports.

    Scope: exact + alias substring matches; does not (yet) call scispaCy. Good enough for the
    Phase-0 perturbation benchmark; replace with scispaCy NER + UMLS linker for paper-grade runs.
    """

    NEGATION_TOKENS: ClassVar[tuple[str, ...]] = ("no", "without", "absent", "not", "negative for")
    NEGATION_WINDOW: ClassVar[int] = 6  # tokens

    def __init__(self) -> None:
        self._compiled: dict[str, list[tuple[str, re.Pattern[str]]]] = {}
        for feat in VASARI_FEATURES:
            patterns: list[tuple[str, re.Pattern[str]]] = []
            for value, aliases in feat.aliases.items():
                for a in aliases:
                    pat = re.compile(rf"\b{re.escape(a.lower())}\b")
                    patterns.append((value, pat))
            for value in feat.values:
                v_norm = value.replace("_", " ").replace("-", " ")
                if len(v_norm) > 1:
                    pat = re.compile(rf"\b{re.escape(v_norm.lower())}\b")
                    patterns.append((value, pat))
            self._compiled[feat.short] = patterns

    def parse(self, text: str) -> VASARIFeatureVector:
        text_lower = text.lower()
        tokens = re.findall(r"\w+", text_lower)
        out = VASARIFeatureVector()

        for feat in VASARI_FEATURES:
            for value, pat in self._compiled[feat.short]:
                m = pat.search(text_lower)
                if m is None:
                    continue
                start_token = len(re.findall(r"\w+", text_lower[: m.start()]))
                window_start = max(0, start_token - self.NEGATION_WINDOW)
                window = tokens[window_start:start_token]
                negated = any(any(neg.split()[0] == w for neg in self.NEGATION_TOKENS) for w in window)
                if negated:
                    if feat.values and "no" in feat.values:
                        out.values[feat.short] = "no"
                    continue
                if feat.short not in out.values:
                    out.values[feat.short] = value
                    break
        return out

    def is_negated(self, text: str, span_start: int) -> bool:
        text_lower = text.lower()
        before = text_lower[: span_start]
        tokens = re.findall(r"\w+", before)
        window = tokens[-self.NEGATION_WINDOW :]
        return any(any(neg.split()[0] == w for neg in self.NEGATION_TOKENS) for w in window)
