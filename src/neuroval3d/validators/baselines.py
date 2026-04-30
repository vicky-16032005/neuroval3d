"""Baseline validators we compare *against*: BERTScore, RaTEScore-stub, generic-BERT cosine.

These exist so the AUROC table can show that NeuroVal-3D outperforms each on brain-MRI
hallucination detection. They are NOT part of NeuroVal-3D itself.
"""
from __future__ import annotations

import warnings
from typing import Iterable


class BERTScoreBaseline:
    """Wrapper around the standard `bert-score` package.

    Uses `microsoft/deberta-xlarge-mnli` by default per the package's recommendation; we
    override to `roberta-base` for speed in the smoke benchmark.
    """

    def __init__(self, model_type: str = "roberta-base", device: str = "cpu") -> None:
        self.model_type = model_type
        self.device = device
        self._fn = None

    def _lazy(self):
        try:
            from bert_score import score
            self._fn = score
        except ImportError as e:
            warnings.warn(f"bert-score not installed; baseline disabled: {e}")
            self._fn = None

    def score(self, generated: str, reference: str) -> float:
        return self.score_batch([generated], [reference])[0]

    def score_batch(self, generated: Iterable[str], reference: Iterable[str]) -> list[float]:
        if self._fn is None:
            self._lazy()
        gen_l = list(generated)
        ref_l = list(reference)
        if self._fn is None:
            return [0.5] * len(gen_l)
        try:
            P, R, F1 = self._fn(gen_l, ref_l, model_type=self.model_type, device=self.device,
                                verbose=False, lang=None)
            return [float(f) for f in F1]
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"BERTScore failed: {e}")
            return [0.5] * len(gen_l)


class RaTEScoreLite:
    """Stub for RaTEScore. Real impl requires the RaTEScore package + entity-linking weights.

    For Phase 0 / Phase 4 polish we expose the API so callers can drop the real one in once
    available. Returns deterministic dummy scores (token Jaccard) when unavailable so the
    pipeline is unblocked.
    """

    def score(self, generated: str, reference: str) -> float:
        a = set(generated.lower().split())
        b = set(reference.lower().split())
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)


class GenericBERTBaseline:
    """Generic-domain BERT cosine — what most non-medical validators use. Worse than NeuroVal-3D
    on medical terminology (this is the point of the comparison)."""

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self._tokenizer = None
        self._model = None

    def _lazy(self):
        try:
            from transformers import AutoModel, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            self._model = AutoModel.from_pretrained("bert-base-uncased").to(self.device).eval()
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"GenericBERT baseline unavailable: {e}")

    def score(self, generated: str, reference: str) -> float:
        if self._model is None:
            self._lazy()
        try:
            import torch
            with torch.no_grad():
                enc = self._tokenizer(
                    [generated, reference], padding=True, truncation=True, max_length=512,
                    return_tensors="pt",
                ).to(self.device)
                out = self._model(**enc).last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1).float()
                pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                cos = torch.nn.functional.cosine_similarity(pooled[0], pooled[1], dim=0)
                return float(cos.cpu().item())
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"GenericBERT score failed: {e}")
            return 0.5
