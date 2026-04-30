"""Stage 5a — semantic validator.

Uses a frozen biomedical text encoder to compute cosine similarity between a generated
report and a reference (or retrieved nearest neighbor). Default backbone is
BioClinicalBERT (emilyalsentzer/Bio_ClinicalBERT). Falls back to sentence-transformers
all-MiniLM-L6-v2 if the medical model can't be loaded (e.g. offline smoke test).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Iterable

import numpy as np

DEFAULT_MODEL = "emilyalsentzer/Bio_ClinicalBERT"
FALLBACK_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@dataclass
class SemanticValidatorConfig:
    model_name: str = DEFAULT_MODEL
    fallback_model: str = FALLBACK_MODEL
    max_length: int = 512
    pooling: str = "mean"
    device: str = "cpu"
    use_hash_fallback: bool = True


class SemanticValidator:
    """Compute (generated, reference) cosine similarity.

    Loads the model lazily so the import path does not pay a network/disk cost.
    """

    def __init__(self, config: SemanticValidatorConfig | None = None) -> None:
        self.config = config or SemanticValidatorConfig()
        self._model = None
        self._tokenizer = None
        self._fallback_active = False

    # ------------------------------------------------------------------ public
    def score(self, generated: str, reference: str) -> float:
        v_gen = self.embed(generated)
        v_ref = self.embed(reference)
        return float(_cosine(v_gen, v_ref))

    def score_batch(self, generated: Iterable[str], reference: Iterable[str]) -> list[float]:
        gens = list(generated)
        refs = list(reference)
        if len(gens) != len(refs):
            raise ValueError("generated and reference must have equal length")
        emb_g = self.embed_batch(gens)
        emb_r = self.embed_batch(refs)
        return [float(_cosine(g, r)) for g, r in zip(emb_g, emb_r, strict=True)]

    def embed(self, text: str) -> np.ndarray:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        if self._model is None:
            self._lazy_load()

        if self._fallback_active and self._model == "hash":
            return np.stack([_hash_embed(t) for t in texts], axis=0)

        try:
            import torch
            with torch.no_grad():
                enc = self._tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_length,
                    return_tensors="pt",
                ).to(self.config.device)
                out = self._model(**enc)
                hidden = out.last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1)
                if self.config.pooling == "mean":
                    summed = (hidden * mask).sum(1)
                    counts = mask.sum(1).clamp(min=1e-9)
                    pooled = summed / counts
                elif self.config.pooling == "cls":
                    pooled = hidden[:, 0]
                else:
                    raise ValueError(f"Unknown pooling {self.config.pooling}")
                return pooled.cpu().numpy()
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"Semantic encoder forward failed; using hash fallback: {e}")
            self._model = "hash"
            self._fallback_active = True
            return np.stack([_hash_embed(t) for t in texts], axis=0)

    # ------------------------------------------------------------------ internals
    def _lazy_load(self) -> None:
        try:
            from transformers import AutoModel, AutoTokenizer
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
                self._model = AutoModel.from_pretrained(self.config.model_name)
            except Exception:
                self._tokenizer = AutoTokenizer.from_pretrained(self.config.fallback_model)
                self._model = AutoModel.from_pretrained(self.config.fallback_model)
            import torch
            self._model = self._model.to(self.config.device).eval()  # type: ignore[union-attr]
        except Exception as e:  # noqa: BLE001
            if not self.config.use_hash_fallback:
                raise
            warnings.warn(f"Semantic encoder unavailable; using hash fallback: {e}")
            self._model = "hash"
            self._fallback_active = True


# ----------------------------------------------------------------------------- utilities

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a) + 1e-9)
    nb = float(np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b) / (na * nb))


def _hash_embed(text: str, dim: int = 256) -> np.ndarray:
    """Deterministic hash-based bag-of-words embedding — for offline smoke tests."""
    import hashlib
    vec = np.zeros(dim, dtype=np.float32)
    for tok in text.lower().split():
        h = int(hashlib.md5(tok.encode()).hexdigest()[:8], 16)
        vec[h % dim] += 1.0
    norm = float(np.linalg.norm(vec) + 1e-9)
    return vec / norm
