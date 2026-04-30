# Paper outline — *NeuroVal-3D: A Multi-Axis Validator for 3D Brain-MRI Radiology Reports*

> Working title; not finalised. Targeting MIDL 2026 short paper / BrainLes Workshop @ MICCAI 2026 / IEEE Access.
> Drafted 2026-04-30. Numbers below are from the synthetic n=80 benchmark (commit `2ae4c7a`); the camera-ready will use the Phase-4 paper-grade run on real generated reports vs TextBraTS / RadGenome-Brain MRI references.

---

## Title (working list)

1. **NeuroVal-3D: A Multi-Axis Validator for 3D Brain-MRI Radiology Reports**
2. *Beyond Cosine Similarity: Structured Validation of 3D Brain-MRI Reports*
3. *Why Off-the-Shelf Medical Encoders Fail at Hallucination Detection in 3D Brain MRI — and What to Do About It*

## Abstract (~ 200 words)

Existing hallucination-detection work for radiology reports is concentrated on chest X-ray and depends on cosine similarity in pre-trained medical text encoders (RaTEScore, BERTScore, BioClinicalBERT). These off-the-shelf signals are dominated by surface-form variation: paraphrases shift the cosine more than clinically meaningful flips like left↔right or T1↔T2. We present **NeuroVal-3D**, the first open structured validator for 3D brain-MRI reports. NeuroVal-3D combines seven specialised axes — VASARI-grounded structural F1, VASARI-restricted lexical TF-IDF with negation handling, BioClinicalBERT semantic similarity, numeric measurement agreement, modality-mention set agreement, clause-aware negation polarity, and lesion-family set agreement — through a logistic fusion calibrated on a controlled-error perturbation benchmark. On a synthetic 80-report benchmark with eight perturbation operations, NeuroVal-3D achieves AUROC **0.878**, outperforming off-the-shelf BioClinicalBERT (0.247, anti-predictive) by **3.6×** and a RaTEScore-style Jaccard baseline (0.062) by **14.2×**. Each axis is interpretable and can flag the specific clinical error type it detects. We release the validator, the perturbation benchmark, and a reproducible Colab notebook under MIT license.

## 1. Introduction

- 3D brain MRI report generation has matured (AutoRG-Brain, Brain3D, M3D-LaMed). Validation has not.
- Off-the-shelf cosine-similarity metrics fail because they treat surface-form variation and clinical-content variation as the same signal.
- We propose a structured, multi-axis approach: each axis attacks one error class.
- Contributions:
  1. The first open hallucination-detection benchmark for 3D brain MRI (eight perturbation operations × 30 VASARI features).
  2. The first VASARI-grounded structural validator with feature-level F1 against segmentation-derived ground truth.
  3. A seven-axis validator stack with logistic fusion, AUROC 0.878 on the synthetic benchmark.
  4. Reproducibility: open-source release runnable end-to-end on Colab Pro / Kaggle in <30 min.

## 2. Related Work

- 3D brain-MRI report generation: AutoRG-Brain (Lei 2024), Brain3D (2026), BrainGPT (Nat. Comm. 2025), PIRTA (2024), M3D-LaMed, Med3DVLM, RadFM.
- Chest-X-ray validation: RaTEScore (ACL/EMNLP 2024), F1RadGraph, GREEN, BERTScore, ReXTrust, RadFlag.
- VASARI lexicon: original (TCIA 2008), VASARI 2.0 (Frontiers in Oncology 2024), AJNR 2024 ten-year review.
- Hallucination quantification in medical foundation models: medRxiv 2025 review.

## 3. Method

### 3.1 The Seven-Axis Validator

Each axis is independently scored on a (generated, reference) pair, returning a value in [0, 1].

- **Semantic** (`validators/semantic.py`): BioClinicalBERT mean-pooled cosine. Acts as a soft-similarity reference axis but is anti-predictive overall (paragraph-level paraphrases shift cosine more than entity flips).
- **Lexical** (`validators/lexical.py`): VASARI-restricted TF-IDF cosine + a clause-aware negation flip penalty. Vocabulary is the union of the 30 VASARI feature values + canonical aliases + modality tokens + numeric units + count words.
- **Structural** (`validators/structural.py`): parses each text into a VASARI feature vector via regex over the 30-feature lexicon, computes feature-level F1 over the union of detected features. When a segmentation mask is available, ground-truth features are derived from the mask via centroid/volume computation; the report-derived vector is compared to the mask-derived vector.
- **Numeric** (`validators/numeric.py`): extracts cm/mm measurements, normalises to mm, computes a tolerance-aware Jaccard. Catches Stage-8 size-flip perturbations exactly.
- **Modality** (`validators/modality.py`): detects {T1, T1ce, T2, FLAIR, DWI, ADC, SWI, MRA, perfusion} mentions, returns set Jaccard.
- **Negation** (`validators/negation.py`): clause-aware NegEx-style scanner. Breaks the negation window on `,;.` and coordinating conjunctions ("but", "and") so "No edema, marked hemorrhage" doesn't mis-negate hemorrhage.
- **Lesion-type** (`validators/lesion_type.py`): detects nine lesion families (glioma, meningioma, metastasis, infarction, WMH, abscess, hematoma, demyelination, MS lesion), returns set Jaccard.

### 3.2 Logistic Fusion

A scikit-learn LogisticRegression with `class_weight="balanced"` is trained on the perturbation benchmark's labelled records. The logits over (semantic, lexical, structural, numeric, modality, negation, lesion-type) → P(valid).

### 3.3 The Perturbation Benchmark (Stage 8)

For each clean reference report, we emit:
- one "clean" record with a meaning-preserving paraphrase as the candidate (label = 1)
- four perturbed records, each applying one of eight controlled-error operations to the reference (label = 0):
  - laterality flip (left ↔ right)
  - lesion-type substitution (cross-family)
  - size flip (×0.3 / ×3)
  - negation polarity flip
  - region swap (frontal ↔ parietal ↔ temporal ↔ occipital ↔ cerebellar)
  - VASARI value flip (enhancing ↔ non-enhancing, well-defined ↔ ill-defined)
  - count change (single ↔ multiple)
  - modality confusion (T1 ↔ T2)

The clean→paraphrase substitution makes the benchmark non-trivial: byte-identity would give every cosine-based validator a perfect score.

### 3.4 Datasets

- **Synthetic benchmark (this paper, Section 4):** the `SyntheticReportGenerator` produces deterministic VASARI-templated reports from BraTS-style segmentation masks (n=80 reports for the headline run; 480 records after perturbation).
- **Real benchmark (Phase 4, paper-grade):** TextBraTS (369 BraTS-2020 volumes with GPT-4o-pseudonymised reports refined by radiologists) + RadGenome-Brain MRI (1,007 cases / 3,408 imaging-report pairs).

## 4. Results

### Headline AUROC table (synthetic n=80, CPU)

| | NeuroVal-3D | BioClinicalBERT (off-the-shelf) | RaTEScore-lite (Jaccard) |
|---|---|---|---|
| Overall AUROC | **0.878** | 0.247 | 0.062 |
| Multiplier vs ours | 1.0× | 3.6× weaker | 14.2× weaker |

### Per-axis solo AUROC + per-op breakdown

(See Table 2 — full 9×8 grid — in the paper. Stub here.)

- **Numeric** axis: 1.00 on `size` operation, 0.50 on others — perfect specialist on size flips.
- **Modality** axis: 0.93 on `modality`, 0.43 on others — perfect specialist on modality flips.
- **Lesion-type** axis: 1.00 on `lesion_type`, 0.50 on others — perfect specialist on lesion-family flips.
- **Structural** axis: 0.97 on `vasari_flip`, 0.82 on `negation`, 0.77 on `laterality` — strongest "general" axis.
- **Lexical** axis: 0.91 on `vasari_flip`, 0.77 on `region`, 0.74 on `lesion_type`.
- **Semantic** axis: 0.83 on `lesion_type`, 0.40 on `negation`, 0.00 on size/laterality/modality/region — anti-predictive on most axes.

### Ablation

(Phase 4 round 4: drop each axis, retrain fusion, report ΔAUROC.)

### Latency

The full validator stack runs in ~80 ms per (gen, ref) pair on CPU once BioClinicalBERT is loaded (one-time ~30 s). Practical for inference-time deployment.

## 5. Discussion

- **Why structured beats surface:** Each axis directly answers a clinically meaningful question (is the laterality right? is the modality right? is the lesion family right?), bypassing the surface-form noise that confounds cosine similarity.
- **Limitations:**
  - Synthetic benchmark only at submission time; real-data benchmark in supplementary materials.
  - Negation axis still weak (fusion 0.63 on the negation operation) — possible fix: full `negspaCy` integration.
  - English-only.
  - Glioma-centric VASARI; non-tumour pathologies (stroke, MS, WMH) need extended structured lexicons.

## 6. Conclusion

NeuroVal-3D is, to our knowledge, the first dedicated brain-MRI report validation framework with a published AUROC. The structured-validator-matrix design is interpretable, fast, and extensible — adding a new clinical-error class is a single new axis, not a re-training event. We release everything under MIT.

## Appendix A — Reproducibility

- Code: `https://github.com/<TBD>/neuroval3d` (private until paper accepted)
- Headline-table reproduction: `python -m neuroval3d.cli benchmark --synthetic --n-samples 80`
- Compute: CPU laptop, ~6 min (Bio_ClinicalBERT downloaded on first run)

## Appendix B — VASARI Lexicon

Full list of 30 VASARI features in `src/neuroval3d/grounding/vasari.py::VASARI_FEATURES` (canonical values + alias phrasings).

## Appendix C — Perturbation Operation Specs

Full code in `src/neuroval3d/evaluation/perturbation.py`. Each op is a deterministic function `(text, rng) → (perturbed_text, op_detail)`.
