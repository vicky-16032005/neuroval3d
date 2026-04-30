"""Numeric-equivalence validator — exists because TF-IDF and BERT both swallow size measurements.

Pulls numeric measurements (like "3.5 cm", "12 mm") from both texts. Compares the multi-set of
extracted (value, unit) tuples and returns a similarity score in [0, 1].

Concretely:
  - extract every numeric+unit token from both texts
  - normalise to mm (cm × 10, mm × 1)
  - return the size-agreement Jaccard with a tolerance band
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# (?:\b|^) numeric (\.\d+)? (cm|mm|millimet*|centimet*)
NUMERIC_PAT = re.compile(
    r"\b(\d+(?:\.\d+)?)\s*(cm|mm|millim(?:eter|etre)s?|centim(?:eter|etre)s?)\b",
    re.IGNORECASE,
)


@dataclass
class NumericValidatorConfig:
    tolerance_mm: float = 1.0
    """Two values are considered the same if they're within this many mm."""


class NumericValidator:
    """Compute numeric-mention agreement between two reports."""

    def __init__(self, config: NumericValidatorConfig | None = None) -> None:
        self.config = config or NumericValidatorConfig()

    def score(self, generated: str, reference: str) -> float:
        gen = self._extract_mm(generated)
        ref = self._extract_mm(reference)
        if not gen and not ref:
            return 1.0
        if not gen or not ref:
            return 0.0
        gen_sorted = sorted(gen)
        ref_sorted = sorted(ref)
        n_match = 0
        used = [False] * len(ref_sorted)
        for g in gen_sorted:
            for i, r in enumerate(ref_sorted):
                if used[i]:
                    continue
                if abs(g - r) <= self.config.tolerance_mm:
                    n_match += 1
                    used[i] = True
                    break
        return n_match / max(len(gen_sorted), len(ref_sorted), 1)

    @staticmethod
    def _extract_mm(text: str) -> list[float]:
        out: list[float] = []
        for m in NUMERIC_PAT.finditer(text):
            val = float(m.group(1))
            unit = m.group(2).lower()
            if unit.startswith("cm") or unit.startswith("centim"):
                out.append(val * 10.0)
            else:
                out.append(val)
        return out
