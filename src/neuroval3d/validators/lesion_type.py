"""Lesion-type validator — recognises clinically distinct lesion families and detects flips.

Maps every text to the set of lesion families mentioned ({glioma, meningioma, metastasis,
infarction, wmh, abscess, hematoma, demyelination, ms_lesion}). Returns set Jaccard.
"""
from __future__ import annotations

import re
from typing import Iterable

# Each family → list of synonym phrases
LESION_FAMILIES: dict[str, tuple[str, ...]] = {
    "glioma": ("glioma", "glioblastoma", "glial neoplasm", "high-grade glioma",
               "low-grade glioma", "astrocytoma", "oligodendroglioma"),
    "meningioma": ("meningioma", "meningothelial"),
    "metastasis": ("metastasis", "metastases", "metastatic deposit", "metastatic disease"),
    "infarction": ("infarct", "infarction", "ischemic stroke", "acute infarct",
                   "chronic infarct"),
    "wmh": ("white matter hyperintensity", "white matter hyperintensities",
            "small vessel disease", "leukoaraiosis"),
    "abscess": ("abscess", "pyogenic abscess", "ring-enhancing abscess"),
    "hematoma": ("hematoma", "haematoma", "intracerebral haemorrhage",
                 "intracerebral hemorrhage"),
    "demyelination": ("multiple sclerosis", "demyelinating", "demyelination"),
    "ms_lesion": ("ms plaque", "ms lesion"),
}


class LesionTypeValidator:
    def __init__(self, families: Iterable[str] | None = None) -> None:
        self.families = tuple(families) if families else tuple(LESION_FAMILIES.keys())
        self._compiled: dict[str, list[re.Pattern[str]]] = {}
        for fam in self.families:
            patterns = LESION_FAMILIES.get(fam, ())
            self._compiled[fam] = [re.compile(rf"\b{re.escape(p)}\b", re.IGNORECASE)
                                    for p in patterns]

    def detect(self, text: str) -> set[str]:
        out: set[str] = set()
        for fam, pats in self._compiled.items():
            for p in pats:
                if p.search(text):
                    out.add(fam)
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
