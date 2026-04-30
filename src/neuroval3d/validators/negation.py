"""Negation-polarity validator — catches "no edema" ↔ "edema" flips that lexical TF-IDF misses.

Approach: scan each text for (term, polarity) tuples by sliding a negation window over the
tokens, much like NegEx. Compare the multi-set of tuples — if the polarity has flipped for
the same term, the score drops.

Closed clinical-finding vocabulary (no LLM needed for offline use):
  edema, oedema, hemorrhage, haemorrhage, enhancement, restricted diffusion,
  mass effect, midline shift, hydrocephalus, calcification, cyst,
  necrosis, satellite, infiltration, mass, lesion
"""
from __future__ import annotations

import re
from typing import Iterable

NEGATION_TOKENS = ("no", "without", "absent", "not", "denies", "negative", "free of")
WINDOW = 6

CLINICAL_TERMS: tuple[str, ...] = (
    "edema", "oedema", "hemorrhage", "haemorrhage", "enhancement",
    "restricted diffusion", "mass effect", "midline shift", "hydrocephalus",
    "calcification", "cyst", "necrosis", "satellite", "infiltration",
    "mass", "lesion", "nodule", "tumor", "tumour", "abnormality", "lesions",
)


class NegationValidator:
    """Compute (term, polarity) agreement between two texts."""

    def __init__(self, terms: Iterable[str] = CLINICAL_TERMS) -> None:
        self.terms = tuple(terms)
        self._term_pats = {t: re.compile(rf"\b{re.escape(t)}\b", re.IGNORECASE) for t in self.terms}

    def extract(self, text: str) -> set[tuple[str, bool]]:
        """Return {(term, is_negated)} for every clinical term mention in `text`."""
        text_lower = text.lower()
        out: set[tuple[str, bool]] = set()
        for term, pat in self._term_pats.items():
            for m in pat.finditer(text_lower):
                negated = self._is_negated(text_lower, m.start())
                out.add((term, negated))
        return out

    def score(self, generated: str, reference: str) -> float:
        gen = self.extract(generated)
        ref = self.extract(reference)
        if not gen and not ref:
            return 1.0
        if not gen or not ref:
            return 0.0
        # Set Jaccard over (term, polarity) tuples
        return len(gen & ref) / len(gen | ref)

    @staticmethod
    def _is_negated(text_lower: str, span_start: int) -> bool:
        # Look only within the current clause: stop at any punctuation that ends a clause.
        # This prevents "No edema, marked hemorrhage" from negating "hemorrhage".
        before = text_lower[: span_start]
        # Find the last clause-terminator (., ;, ,, or "but"/"and" at a token boundary)
        last_break = max(
            before.rfind("."),
            before.rfind(";"),
            before.rfind(","),
        )
        # Also break on coordinating conjunctions
        for conj in (" but ", " and ", " however ", " whereas "):
            idx = before.rfind(conj)
            if idx > last_break:
                last_break = idx + len(conj) - 1
        clause = before[last_break + 1 :] if last_break >= 0 else before
        tokens = re.findall(r"[\w\-]+", clause)
        window = tokens[-WINDOW:]
        for neg in NEGATION_TOKENS:
            head = neg.split()[0]
            if any(w == head for w in window):
                return True
        return False
