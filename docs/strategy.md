# Strategy: How NeuroVal-3D Demolishes the Open SOTA

The instinct is to chase BLEU/METEOR on M3D-Cap. That's a trap — those leaderboards are won by labs with A100×8 clusters and 100K+ paired samples. We will lose head-to-head, full stop.

Instead we attack three open frontiers where **no one has published anything yet** for 3D brain MRI specifically.

## Frontier 1 — Hallucination Detection

**Status of the field:** Chest X-ray hallucination detection has matured (RaTEScore ACL/EMNLP 2024; ReXTrust arXiv 2412.15264 AUROC 0.875 on MIMIC-CXR; RadFlag arXiv 2411.00299; Process Reward Models arXiv 2510.23217). Brain MRI: **zero published hallucination detectors as of 2026-04**. The brain-MRI generation papers (AutoRG-Brain, Brain3D, PIRTA, BrainGPT) all evaluate via NLG metrics + human review — none ship a per-report numeric clinical-correctness score.

**Our move:** Construct a perturbation benchmark — 200–500 reports drawn from TextBraTS + RadGenome-Brain MRI, each given a controlled error from one of:

1. **Laterality flip** — left ↔ right hemisphere
2. **Lesion-type substitution** — glioma → meningioma → metastasis → infarction → WMH
3. **Size error** — diameter scaled by 0.3× or 3×
4. **Negation flip** — "no edema" ↔ "edema is present"
5. **Anatomical region swap** — frontal ↔ parietal ↔ temporal ↔ occipital ↔ cerebellum
6. **VASARI-feature flip** — enhancing ↔ non-enhancing, well-defined ↔ ill-defined
7. **Count change** — "two lesions" ↔ "three lesions"
8. **Modality confusion** — T1 finding described as T2 finding

A validator that achieves AUROC > 0.85 detecting these against the original report **is publishable on its own** because no such benchmark exists yet for brain MRI.

## Frontier 2 — VASARI-Grounded Structural Validation

**Status of the field:** VASARI 2.0 (Frontiers in Oncology 2024) is the consensus controlled vocabulary for glioma reporting. AJNR 2024 systematic review covers ten years of VASARI usage. Yet **zero published 3D brain MRI report generators enforce or evaluate VASARI consistency**. AutoRG-Brain explicitly does not.

**Our move:** Two-way VASARI grounding:
- **Forward**: parse generated reports into the 30-feature VASARI vector via scispaCy NER + rules.
- **Backward**: derive ground-truth VASARI features from BraTS segmentation masks (centroid → laterality, mask volumes → enhancement %, edema % etc.).
- Score each report on VASARI feature-level F1.

This is the first VASARI-aware validator. Even modest performance (F1 ≥ 0.65) is the first published number on this axis.

## Frontier 3 — Reproducibility on Free-Tier Compute

**Status of the field:** Every published 3D brain MRI generator was trained on A100/H100. None ships a Colab-runnable end-to-end demo. Reviewers complain about this constantly.

**Our move:** A complete pipeline that runs end-to-end on Colab Pro / Kaggle, with:
- paired notebooks for every stage
- synthetic-data smoke test that needs zero downloads
- a single Colab notebook (`notebooks/00_smoke_test.ipynb`) that reproduces the headline AUROC number in <30 minutes

This is a methodological contribution: democratizing 3D brain MRI report-generation research.

## What we are NOT trying to do

- ❌ Beat M3D-LaMed / Med3DVLM / Brain3D on METEOR. Wrong fight.
- ❌ Train Llama-3-8B from scratch.
- ❌ Compete with foundation-model labs on raw scale.
- ❌ Replace radiologists.

## The headline numbers we are gunning for

| Metric | Target | Beats |
|--------|--------|-------|
| Hallucination AUROC (perturbation set) | ≥ 0.85 | RaTEScore (~0.75), BERTScore (~0.70), F1RadGraph (~0.65) on this brain-MRI bench |
| VASARI feature F1 (vs segmentation-derived GT) | ≥ 0.65 | First published number — no baseline |
| Latency per-report validation | < 200 ms on T4 | Practical for inference-time use |
| End-to-end demo wall-clock on Colab | < 30 min | First reproducible setup |

If we land all four, the paper writes itself.

## Why this beats the SOTA

The SOTA generators are bigger, more accurate, more fluent. Fine. But they cannot tell you whether their output is correct on a per-report basis. That's the gap. We don't compete with the generators — we make them safe to use. That's a complementary contribution, and reviewers will see it as a positive-sum addition to the field rather than a head-to-head challenge they have to gatekeep.

The narrative for the paper: **"3D brain MRI report generation is mature. 3D brain MRI report VALIDATION is wide open. We own it."**
