"""Stage 3 — multimodal projector: maps 3D image features to the decoder's input space."""
from __future__ import annotations


class MLPProjector:
    """A thin MLP that projects [B, F_img] → [B, T, D_text].

    For Phase 0 we keep the implementation skeletal; the contract is fixed so downstream
    code can call `.project(image_features)` and receive token-shaped output.
    """

    def __init__(self, in_dim: int = 48, out_dim: int = 768, n_tokens: int = 32) -> None:
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_tokens = n_tokens
        self._net = None

    def _build(self):
        try:
            import torch.nn as nn
            self._net = nn.Sequential(
                nn.Linear(self.in_dim, self.out_dim * self.n_tokens),
            )
        except ImportError:
            self._net = None

    def project(self, image_features):
        """image_features: [B, in_dim] → tokens [B, n_tokens, out_dim]."""
        if self._net is None:
            self._build()
        try:
            import torch
            x = image_features if hasattr(image_features, "shape") else torch.as_tensor(image_features)
            B = x.shape[0]
            out = self._net(x).view(B, self.n_tokens, self.out_dim)
            return out
        except Exception:  # noqa: BLE001
            import numpy as np
            B = image_features.shape[0]
            return np.zeros((B, self.n_tokens, self.out_dim), dtype="float32")
