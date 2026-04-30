"""Stage 5b — lexical validator (VASARI-restricted TF-IDF + negation-aware penalty).

Generic English TF-IDF rewards fluent-but-wrong reports (a fluent fabricated
report shares many tokens with the reference). Restricting the vocabulary to
VASARI feature values + UMLS brain-MRI anchor terms makes TF-IDF clinically
meaningful: only tokens that actually carry diagnostic content count.

Negation handling: "no edema" and "edema is present" should differ even though
they share the head noun. We detect negations within a 6-token window and
emit a paired marker token (e.g. `~edema`) so they are different in the bag-of-words.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from neuroval3d.grounding.vasari import vasari_vocabulary

NEGATION_TOKENS = ("no", "without", "absent", "not", "denies", "negative for")
NEG_WINDOW = 6


@dataclass
class LexicalValidatorConfig:
    ngram_range: tuple[int, int] = (1, 3)
    use_idf: bool = True
    sublinear_tf: bool = True


class LexicalValidator:
    """VASARI-restricted TF-IDF cosine + negation-flip penalty."""

    def __init__(self, config: LexicalValidatorConfig | None = None) -> None:
        self.config = config or LexicalValidatorConfig()
        self.vocabulary: list[str] = vasari_vocabulary()
        self._vectorizer = None
        self._ready = False

    # ------------------------------------------------------------------ public
    def fit(self, corpus: Iterable[str] | None = None) -> "LexicalValidator":
        from sklearn.feature_extraction.text import TfidfVectorizer

        docs = list(corpus) if corpus is not None else self.vocabulary
        if not docs:
            docs = self.vocabulary

        self._vectorizer = TfidfVectorizer(
            vocabulary=self.vocabulary,
            ngram_range=self.config.ngram_range,
            use_idf=self.config.use_idf,
            sublinear_tf=self.config.sublinear_tf,
            lowercase=True,
            token_pattern=r"(?u)\b[\w\-]+\b",
        )
        self._vectorizer.fit(docs)
        self._ready = True
        return self

    def score(self, generated: str, reference: str) -> float:
        if not self._ready:
            self.fit()
        gen = _negation_normalize(generated)
        ref = _negation_normalize(reference)
        v = self._vectorizer.transform([gen, ref]).toarray()  # type: ignore[union-attr]
        if v[0].sum() == 0 and v[1].sum() == 0:
            return 1.0
        if v[0].sum() == 0 or v[1].sum() == 0:
            return 0.0
        cos = float(np.dot(v[0], v[1]) / (np.linalg.norm(v[0]) * np.linalg.norm(v[1]) + 1e-9))
        flip = _negation_flip_penalty(generated, reference)
        return max(0.0, cos - flip)

    def score_batch(self, generated: Iterable[str], reference: Iterable[str]) -> list[float]:
        gens = list(generated)
        refs = list(reference)
        if len(gens) != len(refs):
            raise ValueError("generated and reference must have equal length")
        if not self._ready:
            self.fit(gens + refs)
        return [self.score(g, r) for g, r in zip(gens, refs, strict=True)]


# ----------------------------------------------------------------------------- helpers

def _negation_normalize(text: str) -> str:
    """Insert `~` before a head noun if a negation token sits within NEG_WINDOW tokens before it.

    This is a cheap analogue of NegEx — accurate enough for the validator: makes "no edema"
    distinct from "edema present" in the bag-of-words.
    """
    text = text.lower()
    tokens = re.findall(r"\b[\w\-]+\b|[\.,;!?]", text)
    out_tokens: list[str] = []
    last_neg = -10
    for i, tok in enumerate(tokens):
        if tok in NEGATION_TOKENS or " ".join(tokens[max(0, i - 1) : i + 1]) in NEGATION_TOKENS:
            last_neg = i
            out_tokens.append(tok)
            continue
        if i - last_neg <= NEG_WINDOW and re.match(r"^[\w\-]+$", tok):
            out_tokens.append("~" + tok)
        else:
            out_tokens.append(tok)
    return " ".join(out_tokens)


def _negation_flip_penalty(gen: str, ref: str) -> float:
    """Penalize when a negation polarity has been flipped on a clinically meaningful word.

    We look at words that appear in both texts and check whether one says "no <X>" and the
    other says "<X>" within the negation window. Each flipped word adds 0.10 (clamped at 0.5).
    """
    gen_neg = _extract_negated_terms(gen)
    ref_neg = _extract_negated_terms(ref)
    flipped = (gen_neg ^ ref_neg)  # symmetric difference of negated terms
    return min(0.5, 0.10 * len(flipped))


def _extract_negated_terms(text: str) -> set[str]:
    text = text.lower()
    tokens = re.findall(r"\b[\w\-]+\b", text)
    negated: set[str] = set()
    for i, tok in enumerate(tokens):
        if tok in NEGATION_TOKENS:
            for j in range(i + 1, min(len(tokens), i + 1 + NEG_WINDOW)):
                if tokens[j] in {"is", "are", "was", "were", "of", "the", "a", "an", "any"}:
                    continue
                if re.match(r"^[a-z][a-z\-]+$", tokens[j]):
                    negated.add(tokens[j])
                    break
    return negated
