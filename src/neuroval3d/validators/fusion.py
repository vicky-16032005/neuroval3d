"""Stage 5d — fusion: combine semantic + lexical + structural sub-scores into a single decision.

Default fusion is an sklearn LogisticRegression calibrated on (clean, perturbed) pairs from the
Stage 8 perturbation benchmark. If sklearn is unavailable, we fall back to a fixed-weight blend
(0.5/0.3/0.2) per the strategic blueprint.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass
class ValidationScore:
    semantic: float
    lexical: float
    structural: float
    numeric: float = 1.0
    modality: float = 1.0
    fused: float = 0.0
    decision: str = "FLAGGED"
    explanation: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "semantic": self.semantic,
            "lexical": self.lexical,
            "structural": self.structural,
            "numeric": self.numeric,
            "modality": self.modality,
            "fused": self.fused,
            "decision": self.decision,
            "explanation": dict(self.explanation),
        }


@dataclass
class FusionConfig:
    threshold: float = 0.5
    # weights apply to (semantic, lexical, structural, numeric, modality) when use_logistic=False
    weights: tuple[float, float, float, float, float] = (0.20, 0.30, 0.20, 0.15, 0.15)
    use_logistic: bool = True


class FusionValidator:
    """Train a logistic regressor over the five sub-scores → P(valid).

    Sub-scores (in order): semantic, lexical, structural, numeric, modality.

    Backwards-compatible: callers passing only 3-tuples still work; the fit picks up the
    column count from the input.
    """

    def __init__(self, config: FusionConfig | None = None) -> None:
        self.config = config or FusionConfig()
        self._lr = None
        self._n_features: int | None = None

    # ------------------------------------------------------------------ public
    def fit(
        self,
        sub_scores: Sequence[tuple[float, ...]],
        labels: Sequence[int],
    ) -> "FusionValidator":
        if not self.config.use_logistic:
            return self
        try:
            from sklearn.linear_model import LogisticRegression
            X = np.asarray(sub_scores, dtype=np.float32)
            self._n_features = int(X.shape[1])
            y = np.asarray(labels, dtype=np.int64)
            self._lr = LogisticRegression(class_weight="balanced", solver="lbfgs", max_iter=200)
            self._lr.fit(X, y)
        except (ImportError, ValueError):
            self._lr = None
        return self

    def predict(
        self,
        semantic: float,
        lexical: float,
        structural: float,
        numeric: float = 1.0,
        modality: float = 1.0,
    ) -> ValidationScore:
        full = np.array([[semantic, lexical, structural, numeric, modality]], dtype=np.float32)
        if self._lr is not None:
            try:
                # Slice to the feature count we were trained on
                X = full[:, : self._n_features or 5]
                p = float(self._lr.predict_proba(X)[0, 1])
            except Exception:  # noqa: BLE001
                p = self._weighted_blend(semantic, lexical, structural, numeric, modality)
        else:
            p = self._weighted_blend(semantic, lexical, structural, numeric, modality)

        decision = "VALID" if p >= self.config.threshold else "FLAGGED"
        return ValidationScore(
            semantic=semantic,
            lexical=lexical,
            structural=structural,
            numeric=numeric,
            modality=modality,
            fused=p,
            decision=decision,
            explanation={
                "weight_semantic": self.config.weights[0],
                "weight_lexical": self.config.weights[1],
                "weight_structural": self.config.weights[2],
                "weight_numeric": self.config.weights[3],
                "weight_modality": self.config.weights[4],
                "threshold": self.config.threshold,
            },
        )

    def _weighted_blend(self, semantic: float, lexical: float, structural: float,
                        numeric: float, modality: float) -> float:
        w = self.config.weights
        return float(w[0] * semantic + w[1] * lexical + w[2] * structural
                     + w[3] * numeric + w[4] * modality)

    # ------------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> None:
        if self._lr is None:
            return
        try:
            import joblib
        except ImportError:  # joblib ships with sklearn but be defensive
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._lr, str(path))

    def load(self, path: str | Path) -> "FusionValidator":
        try:
            import joblib
            self._lr = joblib.load(str(path))
        except (ImportError, FileNotFoundError):
            self._lr = None
        return self
