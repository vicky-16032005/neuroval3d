"""Modality-mention agreement validator.

Detects modality terms (T1, T1ce, T2, FLAIR, DWI, ADC, SWI, MRA, perfusion, etc.) in each
text and returns set-Jaccard agreement.

Catches Stage 8 modality perturbations (T1 finding described as T2) that surface-cosine
encoders miss because the token-level edit distance is small.
"""
from __future__ import annotations

import re

# Group of synonyms collapse to a single canonical key.
MODALITY_GROUPS: dict[str, tuple[str, ...]] = {
    "t1": ("t1-weighted", "t1 weighted", "t1wi", r"\bt1\b"),
    "t1ce": ("t1ce", "post-contrast t1", "post contrast t1", "t1c", "t1 post"),
    "t2": ("t2-weighted", "t2 weighted", "t2wi", r"\bt2\b"),
    "flair": ("flair", "fluid attenuated"),
    "dwi": ("dwi", "diffusion-weighted", "diffusion weighted"),
    "adc": ("adc", "apparent diffusion coefficient"),
    "swi": ("swi", "susceptibility weighted"),
    "mra": ("mra", "mr angiography"),
    "perfusion": ("perfusion", "dsc", "dce"),
}


class ModalityValidator:
    """Returns set-Jaccard over canonical modality keys mentioned in each text."""

    def __init__(self) -> None:
        self._compiled: dict[str, list[re.Pattern[str]]] = {
            canon: [re.compile(rf"\b{p}\b" if not p.startswith(r"\b") else p, re.IGNORECASE)
                    for p in patterns]
            for canon, patterns in MODALITY_GROUPS.items()
        }

    def detect(self, text: str) -> set[str]:
        out: set[str] = set()
        for canon, pats in self._compiled.items():
            for p in pats:
                if p.search(text):
                    out.add(canon)
                    break
        return out

    def score(self, generated: str, reference: str) -> float:
        gen = self.detect(generated)
        ref = self.detect(reference)
        if not gen and not ref:
            return 1.0
        if not gen or not ref:
            return 0.0
        return len(gen & ref) / len(gen | ref)
