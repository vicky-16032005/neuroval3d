"""Stage 2 — 3D feature extractor.

Three backends:
  - "swin_unetr"  : MONAI Swin-UNETR. Recommended Colab T4 default.
  - "brainsegfounder" : Pretrained brain-MRI 3D encoder (MIA 2024).
  - "vit3d_inflated" : 2D BiomedCLIP ViT inflated along depth — Brain3D-style.

For Phase 0 we build the configs and a `build_encoder` factory; the heavy weights only
download when actually called. Falls back to a tiny CPU-friendly dummy encoder if MONAI
or torch are unavailable so the smoke test always runs.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EncoderConfig:
    backbone: str = "swin_unetr"
    in_channels: int = 4
    feature_size: int = 48
    image_size: tuple[int, int, int] = (128, 128, 128)
    use_checkpoint: bool = True
    pretrained_weights: str | None = None


def build_encoder(config: EncoderConfig | None = None):
    config = config or EncoderConfig()
    if config.backbone == "swin_unetr":
        return _build_swin_unetr(config)
    if config.backbone == "brainsegfounder":
        return _build_brainsegfounder(config)
    if config.backbone == "vit3d_inflated":
        return _build_inflated_vit3d(config)
    if config.backbone == "dummy":
        return _build_dummy(config)
    raise ValueError(f"Unknown encoder backbone: {config.backbone}")


def _build_swin_unetr(config: EncoderConfig):
    try:
        import torch.nn as nn
        from monai.networks.nets import SwinUNETR

        model = SwinUNETR(
            img_size=config.image_size,
            in_channels=config.in_channels,
            out_channels=3,
            feature_size=config.feature_size,
            use_checkpoint=config.use_checkpoint,
        )
        return model
    except ImportError:
        return _build_dummy(config)


def _build_brainsegfounder(config: EncoderConfig):
    """Skeleton — real impl pulls weights from https://github.com/lab-smile/BrainSegFounder."""
    return _build_dummy(config)


def _build_inflated_vit3d(config: EncoderConfig):
    """Skeleton — real impl follows Brain3D's 2D→3D ViT inflation
    (https://github.com/PRAISELab-PicusLab/BrainGemma3D)."""
    return _build_dummy(config)


def _build_dummy(config: EncoderConfig):
    """Tiny CPU-only encoder used for smoke tests; returns mean-pooled features."""
    try:
        import torch.nn as nn

        class DummyEncoder(nn.Module):
            def __init__(self, in_ch: int, feat: int) -> None:
                super().__init__()
                self.conv = nn.Conv3d(in_ch, feat, kernel_size=3, padding=1)
                self.pool = nn.AdaptiveAvgPool3d(1)

            def forward(self, x):
                z = self.conv(x)
                return self.pool(z).flatten(1)

        return DummyEncoder(config.in_channels, config.feature_size)
    except ImportError:
        # Last-resort: return a callable that gives a fixed-shape numpy vector.
        import numpy as np

        def _fn(x):
            n = x.shape[0] if hasattr(x, "shape") else 1
            return np.zeros((n, config.feature_size), dtype="float32")

        return _fn
