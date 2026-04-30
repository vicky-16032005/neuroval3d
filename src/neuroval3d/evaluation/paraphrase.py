"""Deterministic, meaning-preserving paraphrase pairs for radiology phrasings.

Why: in the smoke benchmark, byte-identical "clean" examples make semantic AUROC trivially
1.0 — we need surface variation that *preserves* meaning so the validator has a real job.

Approach: a small curated synonym table of equivalent medical phrasings. Apply N random
substitutions per sentence with a deterministic seed. We deliberately avoid an LLM here:
deterministic + free + reproducible.

This is for the smoke benchmark only. The Phase-4 paper benchmark uses real generated reports.
"""
from __future__ import annotations

import re
from typing import Iterable

import numpy as np

# Each tuple is a set of mutually-substitutable phrasings. Order does not matter.
EQUIV_GROUPS: tuple[tuple[str, ...], ...] = (
    ("oedema", "edema"),
    ("haemorrhage", "hemorrhage"),
    ("intra-axial", "intraaxial"),
    ("avid enhancement", "marked enhancement", "intense enhancement"),
    ("post-contrast", "post contrast", "after contrast"),
    ("on T1", "on T1-weighted imaging", "on T1WI"),
    ("on T2", "on T2-weighted imaging", "on T2WI"),
    ("vasogenic oedema", "vasogenic edema", "perilesional oedema", "peritumoral oedema"),
    ("there is", "there is identified", "we note", "is identified"),
    ("demonstrates", "shows", "exhibits"),
    ("a discrete", "a well-defined", "a focal"),
    ("mass", "lesion"),
    ("no restricted diffusion", "no diffusion restriction", "without restricted diffusion"),
    ("no hemorrhage", "no haemorrhage", "no blood products"),
    ("findings are most consistent with", "appearances are most consistent with",
     "imaging features favour"),
    ("recommend correlation with clinical context", "clinical correlation is recommended"),
    ("high-grade glial neoplasm", "high-grade glioma"),
    ("FLAIR hyperintensity", "FLAIR signal abnormality", "hyperintense FLAIR signal"),
    ("the lesion", "this lesion", "the mass", "this mass"),
    ("approximately", "roughly", "around"),
    ("in the", "centred in the", "located in the"),
    ("intra-axial lesion", "intra-axial mass", "intraparenchymal lesion"),
)


def paraphrase(text: str, seed: int = 0, max_changes: int = 3) -> str:
    """Return a meaning-preserving paraphrase of `text`.

    Deterministic given the seed: same (text, seed) → same output.
    """
    rng = np.random.default_rng(seed)
    changes = 0
    out = text

    # Shuffle group order so different seeds explore different substitutions first.
    groups = list(EQUIV_GROUPS)
    rng.shuffle(groups)

    for group in groups:
        if changes >= max_changes:
            break
        # Find which alias is currently present (longest first → avoids partial matches).
        present = sorted(group, key=len, reverse=True)
        for alias in present:
            pat = re.compile(rf"\b{re.escape(alias)}\b", re.IGNORECASE)
            if pat.search(out):
                # Pick a different alias as replacement
                alternatives = [a for a in group if a != alias]
                if not alternatives:
                    break
                replacement = alternatives[int(rng.integers(0, len(alternatives)))]
                out = pat.sub(replacement, out, count=1)
                changes += 1
                break

    return out


def paraphrase_batch(texts: Iterable[str], seed: int = 0, max_changes: int = 3) -> list[str]:
    return [paraphrase(t, seed=seed + i, max_changes=max_changes) for i, t in enumerate(texts)]
