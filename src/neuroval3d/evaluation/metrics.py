"""Standard NLG metrics — wrappers around well-known libraries with graceful degradation.

These are reported alongside our validator AUROC, not as the headline metric.
"""
from __future__ import annotations

import warnings
from collections.abc import Iterable


def compute_nlg_metrics(generated: list[str], reference: list[str]) -> dict[str, float]:
    """Compute BLEU-1/2/3/4, ROUGE-L, METEOR (where available)."""
    out: dict[str, float] = {}
    out.update(_safe_bleu(generated, reference))
    out.update(_safe_rouge(generated, reference))
    out.update(_safe_meteor(generated, reference))
    return out


def _safe_bleu(gen: Iterable[str], ref: Iterable[str]) -> dict[str, float]:
    try:
        import sacrebleu
        gen_l, ref_l = list(gen), list(ref)
        # BLEU-N via sentence-bleu averages
        scores: dict[str, list[float]] = {f"bleu{n}": [] for n in (1, 2, 3, 4)}
        for g, r in zip(gen_l, ref_l, strict=True):
            for n in (1, 2, 3, 4):
                bleu = sacrebleu.sentence_bleu(g, [r]).score
                scores[f"bleu{n}"].append(bleu)
        return {k: float(sum(v) / max(len(v), 1)) for k, v in scores.items()}
    except ImportError:
        warnings.warn("sacrebleu not installed; BLEU skipped")
        return {}


def _safe_rouge(gen: Iterable[str], ref: Iterable[str]) -> dict[str, float]:
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        rouges: list[float] = []
        for g, r in zip(list(gen), list(ref), strict=True):
            rouges.append(scorer.score(r, g)["rougeL"].fmeasure)
        return {"rougeL": float(sum(rouges) / max(len(rouges), 1))}
    except ImportError:
        warnings.warn("rouge_score not installed; ROUGE skipped")
        return {}


def _safe_meteor(gen: Iterable[str], ref: Iterable[str]) -> dict[str, float]:
    try:
        import nltk
        from nltk.translate.meteor_score import meteor_score
        try:
            nltk.data.find("wordnet")
        except LookupError:
            nltk.download("wordnet", quiet=True)
            nltk.download("punkt", quiet=True)
        scores: list[float] = []
        for g, r in zip(list(gen), list(ref), strict=True):
            scores.append(meteor_score([r.split()], g.split()))
        return {"meteor": float(sum(scores) / max(len(scores), 1))}
    except ImportError:
        warnings.warn("nltk not installed; METEOR skipped")
        return {}
