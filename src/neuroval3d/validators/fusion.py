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
    fused: float
    decision: str
    explanation: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "semantic": self.semantic,
            "lexical": self.lexical,
            "structural": self.structural,
            "fused": self.fused,
            "decision": self.decision,
            "explanation": dict(self.explanation),
        }


@dataclass
class FusionConfig:
    threshold: float = 0.5
    weights: tuple[float, float, float] = (0.5, 0.3, 0.2)
    use_logistic: bool = True


class FusionValidator:
    """Train a simple logistic regressor over (semantic, lexical, structural) → P(valid)."""

    def __init__(self, config: FusionConfig | None = None) -> None:
        self.config = config or FusionConfig()
        self._lr = None

    # ------------------------------------------------------------------ public
    def fit(
        self,
        sub_scores: Sequence[tuple[float, float, float]],
        labels: Sequence[int],
    ) -> "FusionValidator":
        if not self.config.use_logistic:
            return self
        try:
            from sklearn.linear_model import LogisticRegression
            X = np.asarray(sub_scores, dtype=np.float32)
            y = np.asarray(labels, dtype=np.int64)
            self._lr = LogisticRegression(class_weight="balanced", solver="lbfgs", max_iter=200)
            self._lr.fit(X, y)
        except (ImportError, ValueError):
            self._lr = None
        return self

    def predict(self, semantic: float, lexical: float, structural: float) -> ValidationScore:
        if self._lr is not None:
            try:
                p = float(self._lr.predict_proba(np.array([[semantic, lexical, structural]]))[0, 1])
            except Exception:  # noqa: BLE001
                p = self._weighted_blend(semantic, lexical, structural)
        else:
            p = self._weighted_blend(semantic, lexical, structural)

        decision = "VALID" if p >= self.config.threshold else "FLAGGED"
        return ValidationScore(
            semantic=semantic,
            lexical=lexical,
            structural=structural,
            fused=p,
            decision=decision,
            explanation={
                "weight_semantic": self.config.weights[0],
                "weight_lexical": self.config.weights[1],
                "weight_structural": self.config.weights[2],
                "threshold": self.config.threshold,
            },
        )

    def _weighted_blend(self, semantic: float, lexical: float, structural: float) -> float:
        ws, wl, wst = self.config.weights
        return float(ws * semantic + wl * lexical + wst * structural)

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
