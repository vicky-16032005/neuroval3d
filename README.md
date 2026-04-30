# NeuroVal-3D

**A brain-MRI-specific validation framework for volumetric radiology report generation.**

> Minor project — Naveen Rajdev · Pooja P · Vikneshwaran Marimuthu · Vaishnavi Pagad
> Guide: Prashant Narayankar
> Submission targets: MIDL 2026 short paper · BrainLes Workshop @ MICCAI 2026 · IEEE Access

---

## What this project actually claims

We do **not** claim SOTA on raw NLG metrics (BLEU/METEOR/ROUGE) on M3D-Cap or RadGenome-Brain MRI. Those leaderboards are dominated by foundation models trained on A100×8 clusters (M3D-LaMed, Med3DVLM, Brain3D, AutoRG-Brain, RadFM). Beating them head-to-head with Colab Pro is not a credible 24-week goal.

We **do** claim three first-of-their-kind contributions on open territory:

1. **First open hallucination-detection benchmark for 3D brain MRI reports.** A perturbation set with controlled errors (laterality flips, lesion-type swaps, size errors, negation flips, region swaps, count errors, modality confusions, VASARI flips) constructed from TextBraTS + RadGenome-Brain MRI. Evaluated as AUROC against RaTEScore, BERTScore, F1RadGraph, GREEN.
2. **First VASARI-grounded structural validator** that scores each generated report against the 30-feature VASARI lexicon, with feature-level F1 against segmentation-derived ground truth.
3. **First Colab-trainable end-to-end brain-MRI reporting + validation stack** — a complete reproducible pipeline that runs on free-tier compute, with paired notebooks for every stage.

The publishable claim is **brain-MRI report VALIDATION**, not generation. The generator is a respectable baseline; the validator is the contribution.

### Headline benchmark numbers

**Real data — 369 TextBraTS radiologist-refined reports (n=1,829 records, CPU)**

| | NeuroVal-3D | Off-the-shelf BioClinicalBERT | RaTEScore-lite (Jaccard) |
|---|---|---|---|
| Overall AUROC | **0.998** | 0.088 | 0.017 |
| Multiplier | 1.0× | **11.3× weaker** | **57× weaker** |

Per-op fusion: count 1.00, vasari-flip 1.00, region 1.00, laterality 1.00, negation 0.85.

**Synthetic warmup — 80 templated reports (n=480 records, CPU)**

| | NeuroVal-3D | BioClinicalBERT | RaTEScore-lite |
|---|---|---|---|
| Overall AUROC | 0.878 | 0.247 | 0.062 |
| Multiplier | 1.0× | 3.6× weaker | 14.2× weaker |

Seven specialised validator axes (semantic, lexical, structural, numeric, modality, negation, lesion-type) feed a logistic fusion. Different axes catch different perturbation types — `numeric → size 1.00`, `lesion-type → lesion-type 1.00`, `modality → modality 0.93`, `structural → vasari-flip 1.00`.

## Pipeline (8 stages)

```
[1] 3D Volumetric Preprocessing       HD-BET → N4 → SRI24 → z-score → 128³ patches
[2] 3D-Inflated Feature Extraction    Swin-UNETR / BrainSegFounder / 3D-ViT-inflated
[3] Multimodal Alignment              InfoNCE → MLP projector → LoRA
[4] Modality-Aware Report Generation  BART-base / Llama-3.2-3B-QLoRA / M3D-LaMed
[5] Evidence-Based RAG Validation     RadLex/UMLS/RadGraph KB + FAISS + SciSpaCy
[6] Anatomical Coordinate Anchoring   AAL v3 + MNI152 → [region, (x,y,z), conf]
[7] Explainability & Verification     3D-Grad-CAM + saliency overlay + IoU vs expert
[8] Hallucination Perturbation Bench  Controlled-error generator + AUROC vs baselines
```

Stages 5, 6, 8 are the contribution; 1–4, 7 are the supporting pipeline.

## Quickstart

### Local (Windows / Mac / Linux, CPU OK)

```bash
# 1. Install dependencies (uv recommended; pip works too)
uv sync

# 2. Run the smoke test — no data, no GPU, completes in <60s
uv run pytest -q

# 3. Run the perturbation benchmark on synthetic reports
uv run python -m neuroval3d.cli benchmark --synthetic
```

### Colab / Kaggle

Open `notebooks/00_smoke_test.ipynb` and run all cells. The notebook:
- installs deps via `pip install -e .[dev,eval,nlp,rag]`
- runs the perturbation benchmark on built-in synthetic data
- prints AUROC for the validator vs baselines

### Real data (BraTS / TextBraTS / RadGenome-Brain MRI)

See `docs/datasets.md` for registration steps. None of the data is committed here — it's all downloaded on first use into `data/raw/`.

## Repo layout

```
src/neuroval3d/
├── data/                 # Stage 1 — datasets, preprocessing, synthetic-report generation
├── models/               # Stage 2-4 — encoder, projector, decoder
├── grounding/            # Stage 6 — VASARI lexicon, AAL anatomy, anchoring
├── validators/           # Stage 5 — semantic / lexical / structural / fusion (THE CONTRIBUTION)
├── evaluation/           # Stage 8 — perturbation benchmark, RadEval-style metrics
├── viz/                  # Stage 7 — Grad-CAM, overlays
├── utils/                # checkpoint persistence, IO, logging
└── cli.py                # `neuroval3d` command-line entrypoint

configs/                  # Hydra-style YAML configs per stage
notebooks/                # paired .ipynb for every key module
scripts/                  # data download, Colab/Kaggle bootstrap, batch runs
tests/                    # pytest, runs on dummy data
docs/                     # architecture, strategy, datasets, checkpoints
data/{raw,processed,synthetic}/   # gitignored
outputs/{checkpoints,logs,results}/   # gitignored
```

## Artifact persistence policy

**Every** code execution and resulting artifact is saved redundantly:
- All Python source lives in `src/neuroval3d/` AND has a paired notebook in `notebooks/`.
- Every run logs to `RUN_LOG.md` with timestamp, command, stdout snippet, artifact paths.
- Every checkpoint persists to `outputs/checkpoints/` AND has a manifest entry in `docs/CHECKPOINTS.md`.
- Major milestones are also persisted to long-term Claude memory (`~/.claude/projects/E--MINOR-PROJECT/memory/`).

This guarantees that when reviewers ask "where is the code? where did this number come from?" the answer is in three places at once.

## License

MIT. See `LICENSE`. Not for clinical use.
