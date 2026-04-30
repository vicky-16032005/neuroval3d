"""Stage 4 — text decoder. BART-base default; Llama-3.2-3B-QLoRA / M3D-LaMed are config-swappable.

Phase 0 contract: a `.generate(visual_tokens, prompt)` method that returns a string. The
heavy backbone only loads when `generate` is first called; smoke tests can use the
template-only generator from `data.synthetic`.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DecoderConfig:
    backbone: str = "bart-base"     # or "t5-small", "llama-3.2-3b-qlora", "m3d-lamed"
    max_new_tokens: int = 256
    num_beams: int = 4
    no_repeat_ngram_size: int = 3
    do_sample: bool = False
    device: str = "cpu"


class ReportDecoder:
    def __init__(self, config: DecoderConfig | None = None) -> None:
        self.config = config or DecoderConfig()
        self._model = None
        self._tokenizer = None

    def _lazy_load(self) -> None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        name = {"bart-base": "facebook/bart-base", "t5-small": "google-t5/t5-small"}[
            self.config.backbone
        ]
        self._tokenizer = AutoTokenizer.from_pretrained(name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(name).to(self.config.device).eval()

    def generate(self, visual_tokens=None, prompt: str = "Describe the brain MRI:") -> str:
        """Stub generate. For real runs, visual_tokens are concatenated as soft prompts."""
        try:
            if self._model is None:
                self._lazy_load()
            import torch
            with torch.no_grad():
                inputs = self._tokenizer(prompt, return_tensors="pt").to(self.config.device)  # type: ignore[union-attr]
                out = self._model.generate(  # type: ignore[union-attr]
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    num_beams=self.config.num_beams,
                    no_repeat_ngram_size=self.config.no_repeat_ngram_size,
                    do_sample=self.config.do_sample,
                )
                return self._tokenizer.decode(out[0], skip_special_tokens=True)  # type: ignore[union-attr]
        except Exception as e:  # noqa: BLE001
            return f"[decoder unavailable: {e!s}; prompt: {prompt}]"
