"""Stage 8 — controlled-error perturbation generator.

Given a clean radiology report, produce N labeled perturbed variants. Each perturbation is one
of the eight error types listed in `docs/strategy.md`:

    1. laterality      left ↔ right
    2. lesion_type     glioma → meningioma → metastasis → infarction → WMH
    3. size            diameter scaled 0.3× or 3×
    4. negation        "no edema" ↔ "edema is present"
    5. region          frontal ↔ parietal ↔ temporal ↔ occipital ↔ cerebellar
    6. vasari_flip     enhancing ↔ non-enhancing, well-defined ↔ ill-defined
    7. count           "two lesions" ↔ "three lesions"
    8. modality        T1 finding described as T2 finding

Output: a `PerturbationSet` of `PerturbationRecord(original, perturbed, op_type, op_detail)`
that downstream validators can score.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

import numpy as np


class PerturbationOp(str, Enum):
    LATERALITY = "laterality"
    LESION_TYPE = "lesion_type"
    SIZE = "size"
    NEGATION = "negation"
    REGION = "region"
    VASARI_FLIP = "vasari_flip"
    COUNT = "count"
    MODALITY = "modality"


@dataclass
class PerturbationRecord:
    original: str
    perturbed: str
    op_type: PerturbationOp
    op_detail: str
    original_id: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "original": self.original,
            "perturbed": self.perturbed,
            "op_type": self.op_type.value,
            "op_detail": self.op_detail,
            "original_id": self.original_id,
        }


@dataclass
class PerturbationSet:
    records: list[PerturbationRecord] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.records)

    def by_op(self) -> dict[PerturbationOp, list[PerturbationRecord]]:
        out: dict[PerturbationOp, list[PerturbationRecord]] = {}
        for r in self.records:
            out.setdefault(r.op_type, []).append(r)
        return out

    def save_jsonl(self, path: str) -> None:
        from neuroval3d.utils.io import write_jsonl
        write_jsonl(path, (r.to_dict() for r in self.records))


# ----------------------------------------------------------------------------- operations

LATERALITY_PAIRS = (("left", "right"), ("left-sided", "right-sided"))

LESION_FAMILIES = {
    "glioma": ("glioma", "glioblastoma", "high-grade glial neoplasm", "low-grade glioma"),
    "meningioma": ("meningioma", "meningothelial mass"),
    "metastasis": ("metastasis", "metastatic deposit", "metastases"),
    "infarction": ("infarct", "infarction", "ischemic stroke"),
    "wmh": ("white matter hyperintensity", "small vessel disease", "leukoaraiosis"),
}

REGION_LIST = ("frontal", "parietal", "temporal", "occipital", "cerebellar", "insular")

VASARI_FLIPS = (
    ("enhancing", "non-enhancing"),
    ("avid enhancement", "no enhancement"),
    ("well-defined", "ill-defined"),
    ("well-circumscribed", "infiltrative"),
    ("restricted diffusion", "no restricted diffusion"),
)

COUNT_WORDS = {
    "one": "two", "two": "three", "three": "four", "four": "five",
    "single": "multiple", "solitary": "multiple",
}

MODALITY_FLIPS = (("T1", "T2"), ("T2", "T1"), ("FLAIR", "T1"), ("DWI", "ADC"))


def perturb(report: str, op: PerturbationOp, rng: np.random.Generator | None = None) -> PerturbationRecord | None:
    """Apply one specific perturbation. Returns None if op cannot be applied to this text."""
    rng = rng or np.random.default_rng()
    fn = {
        PerturbationOp.LATERALITY: _perturb_laterality,
        PerturbationOp.LESION_TYPE: _perturb_lesion_type,
        PerturbationOp.SIZE: _perturb_size,
        PerturbationOp.NEGATION: _perturb_negation,
        PerturbationOp.REGION: _perturb_region,
        PerturbationOp.VASARI_FLIP: _perturb_vasari_flip,
        PerturbationOp.COUNT: _perturb_count,
        PerturbationOp.MODALITY: _perturb_modality,
    }[op]
    new_text, detail = fn(report, rng)
    if new_text is None or new_text == report:
        return None
    return PerturbationRecord(original=report, perturbed=new_text, op_type=op, op_detail=detail)


def build_perturbation_set(
    reports: Iterable[str],
    n_per_report: int = 4,
    seed: int = 7,
    ops: tuple[PerturbationOp, ...] | None = None,
) -> PerturbationSet:
    rng = np.random.default_rng(seed)
    ops_pool = list(ops or list(PerturbationOp))
    out = PerturbationSet()
    for i, rep in enumerate(reports):
        # Add the clean original as a "no-op" record (label 0 = valid in the benchmark).
        out.records.append(
            PerturbationRecord(original=rep, perturbed=rep, op_type=PerturbationOp.LATERALITY, op_detail="<clean>", original_id=str(i))
        )
        attempts = 0
        added = 0
        ops_shuffled = list(ops_pool)
        rng.shuffle(ops_shuffled)
        while added < n_per_report and attempts < n_per_report * 4:
            op = ops_shuffled[attempts % len(ops_shuffled)]
            rec = perturb(rep, op, rng)
            attempts += 1
            if rec is not None:
                rec.original_id = str(i)
                out.records.append(rec)
                added += 1
    return out


# ----------------------------------------------------------------------------- per-op impls

def _perturb_laterality(text: str, rng: np.random.Generator) -> tuple[str | None, str]:
    for left, right in LATERALITY_PAIRS:
        pat_l = re.compile(rf"\b{left}\b", re.IGNORECASE)
        pat_r = re.compile(rf"\b{right}\b", re.IGNORECASE)
        if pat_l.search(text):
            return pat_l.sub(_match_case(right), text, count=1), f"{left}->{right}"
        if pat_r.search(text):
            return pat_r.sub(_match_case(left), text, count=1), f"{right}->{left}"
    return None, "no laterality token"


def _perturb_lesion_type(text: str, rng: np.random.Generator) -> tuple[str | None, str]:
    families = list(LESION_FAMILIES.keys())
    for fam in families:
        for term in LESION_FAMILIES[fam]:
            pat = re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE)
            if pat.search(text):
                others = [f for f in families if f != fam]
                target_fam = rng.choice(others)
                target_term = LESION_FAMILIES[target_fam][0]
                return pat.sub(target_term, text, count=1), f"{term}->{target_term}"
    return None, "no lesion-type token"


def _perturb_size(text: str, rng: np.random.Generator) -> tuple[str | None, str]:
    pat = re.compile(r"(\d+(?:\.\d+)?)\s*(cm|mm)")
    m = pat.search(text)
    if m is None:
        return None, "no size measurement"
    val = float(m.group(1))
    unit = m.group(2)
    factor = float(rng.choice([0.3, 0.5, 2.0, 3.0]))
    new_val = round(val * factor, 1)
    return pat.sub(f"{new_val} {unit}", text, count=1), f"{val}{unit}*{factor}"


def _perturb_negation(text: str, rng: np.random.Generator) -> tuple[str | None, str]:
    flip_pairs = (
        (re.compile(r"\bno (\w+ )?(edema|oedema)\b", re.IGNORECASE), "marked oedema"),
        (re.compile(r"\bno (\w+ )?hemorrhage\b", re.IGNORECASE), "areas of hemorrhage"),
        (re.compile(r"\bno (\w+ )?enhancement\b", re.IGNORECASE), "avid enhancement"),
        (re.compile(r"\bno restricted diffusion\b", re.IGNORECASE), "restricted diffusion"),
        (re.compile(r"\bno mass effect\b", re.IGNORECASE), "mass effect"),
        (re.compile(r"\bunremarkable\b", re.IGNORECASE), "abnormal"),
    )
    for pat, replacement in flip_pairs:
        if pat.search(text):
            return pat.sub(replacement, text, count=1), f"flip:{replacement}"

    # The other direction: turn an affirmative finding into a negation.
    affirmative_pairs = (
        (re.compile(r"\b(marked|moderate) (oedema|edema)\b", re.IGNORECASE), "no oedema"),
        (re.compile(r"\bavid enhancement\b", re.IGNORECASE), "no enhancement"),
        (re.compile(r"\bmass effect\b", re.IGNORECASE), "no mass effect"),
    )
    for pat, replacement in affirmative_pairs:
        if pat.search(text):
            return pat.sub(replacement, text, count=1), f"flip:{replacement}"

    return None, "no negation candidate"


def _perturb_region(text: str, rng: np.random.Generator) -> tuple[str | None, str]:
    for region in REGION_LIST:
        pat = re.compile(rf"\b{region}\b", re.IGNORECASE)
        if pat.search(text):
            others = [r for r in REGION_LIST if r != region]
            target = rng.choice(others)
            return pat.sub(target, text, count=1), f"{region}->{target}"
    return None, "no region token"


def _perturb_vasari_flip(text: str, rng: np.random.Generator) -> tuple[str | None, str]:
    for a, b in VASARI_FLIPS:
        pat_a = re.compile(rf"\b{re.escape(a)}\b", re.IGNORECASE)
        pat_b = re.compile(rf"\b{re.escape(b)}\b", re.IGNORECASE)
        if pat_a.search(text):
            return pat_a.sub(b, text, count=1), f"{a}->{b}"
        if pat_b.search(text):
            return pat_b.sub(a, text, count=1), f"{b}->{a}"
    return None, "no VASARI flip candidate"


def _perturb_count(text: str, rng: np.random.Generator) -> tuple[str | None, str]:
    for word, replacement in COUNT_WORDS.items():
        pat = re.compile(rf"\b{word}\b", re.IGNORECASE)
        if pat.search(text):
            return pat.sub(replacement, text, count=1), f"{word}->{replacement}"
    pat = re.compile(r"\b(\d+) (lesion|mass|nodule)s?\b", re.IGNORECASE)
    m = pat.search(text)
    if m:
        n = int(m.group(1))
        new_n = n + 1 if n < 10 else max(1, n - 1)
        return pat.sub(f"{new_n} {m.group(2)}s", text, count=1), f"{n}->{new_n}"
    return None, "no count token"


def _perturb_modality(text: str, rng: np.random.Generator) -> tuple[str | None, str]:
    for a, b in MODALITY_FLIPS:
        pat = re.compile(rf"\b{a}\b")
        if pat.search(text):
            return pat.sub(b, text, count=1), f"{a}->{b}"
    return None, "no modality token"


def _match_case(replacement: str) -> str:
    return replacement
