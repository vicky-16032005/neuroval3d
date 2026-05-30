# NeuroVal-3D + Dashboard UI — Complete Context File

> **What this file is:** A single, self-contained reference covering BOTH (A) the NeuroVal-3D project itself and (B) the dashboard website UI built on top of it. Read this and you will understand the science, the code, the results, the website, how to run everything, and how to deploy/update it — with zero prior context.
>
> **Last updated:** 2026-05-02
> **Working directory:** `E:\MINOR PROJECT` (Windows 10)
> **GitHub:** https://github.com/vicky-16032005/neuroval3d
> **Dashboard live URL (once GitHub Pages is enabled):** https://vicky-16032005.github.io/neuroval3d/

---

# PART A — NeuroVal-3D (the project)

## A.1 One-paragraph summary

NeuroVal-3D is a structured, multi-axis hallucination validator for AI-generated 3D brain MRI radiology reports. It is a B.Tech minor project (KLE Technological University, team of 4, guide Prof. Prashant Narayankar). The core thesis: off-the-shelf medical text encoders (BioClinicalBERT cosine) fail to catch clinically dangerous errors in AI-generated reports — they score a wrong report (e.g. "left" written as "right") at ~0.99 ("looks great") — while our 7-axis structured validator catches it (scores it ~0.18–0.52, "FLAGGED"). The project is functionally complete through Phase 2. Phase 3 (Concept Bottleneck) is authored-but-not-run.

## A.2 Team & identity

| Field | Value |
|-------|-------|
| Project | NeuroVal-3D ("Medical Report Generation and Validation System") |
| Type | B.Tech Minor Project, 6th sem, 2025–26, CSE(AI) |
| University | KLE Technological University, Hubballi, India |
| Team | Naveen Rajdev (01FE23BCI061), Pooja P / Pooja Pattar (01FE23BCI058), Vikneshwaran Marimuthu (01FE23BCI060), Vaishnavi Pagad (01FE23BCI063) |
| Guide | Prof. Prashant Narayankar |
| GitHub | https://github.com/vicky-16032005/neuroval3d |
| Kaggle | https://www.kaggle.com/code/vikneshwaran16032005/minor-project1 |
| Compute | Kaggle T4×2 / Colab Pro / local Windows CPU. Design MUST be free-tier feasible — no A100×8. |

## A.3 The pivot history (important)

1. **2D chest X-ray → 3D brain MRI** (confirmed 2026-04-30). Original was ResNet50 + LSTM + BERT-cosine on MIMIC-CXR. Pivoted because that 2022-era design can't beat current SOTA; 3D brain MRI *validation* is open territory.
2. **Generator-first → Validator-first.** We don't try to beat SOTA generators (need A100×8). The **validator is the contribution** — it makes generators safe to use. Complementary, positive-sum.
3. **Naming:** internally the validator work was "Phase 4 round N" (memory files use this). In final deliverables it's **Phase 1 (validator)** + **Phase 2 (generator + loop closure)**.

## A.4 The problem

A brain MRI study = four co-registered 3D volumes (T1, T1ce, T2, FLAIR). AI generators write fluent reports but hallucinate: laterality flips (left↔right), region swaps (frontal↔temporal), negation flips (no oedema↔marked oedema), lesion-type swaps (glioma↔meningioma), size errors, modality confusion. No published validator exists for 3D brain MRI; chest X-ray validators (F1RadGraph, RaTEScore, GREEN, ReXTrust) don't transfer.

## A.5 The 8-stage pipeline

| Stage | Name | Responsibility | Status |
|-------|------|----------------|--------|
| 1 | Preprocessing | Otsu skull-strip, N4 bias (optional), z-score, resample to 4×64³ | Complete |
| 2 | 3D Encoder | 3D CNN (4→64→128→256, GroupNorm+GELU, stride-2) / SwinUNETR option | Phase 2 impl |
| 3 | Projector | Linear 256→768 + cross-attn over 32 learned queries | Phase 2 impl |
| 4 | Decoder | BART-base (139M), image tokens as cross-attn K/V | Phase 2 impl |
| 5 | **Validator (NeuroVal Core)** | **7 specialists + logistic fusion** | **Complete — THE CONTRIBUTION** |
| 6 | Anatomical Anchor | Tag findings with AAL v3 / MNI152 | Scaffold |
| 7 | Grad-CAM | 3D heatmap overlays | Scaffold |
| 8 | Perturbation Benchmark | 8 controlled-error ops → labelled data | Complete |

## A.6 The 7 specialist validators (Stage 5)

| # | Axis | Module | Technique | Catches |
|---|------|--------|-----------|---------|
| 1 | Semantic | `validators/semantic.py` | BioClinicalBERT mean-pooled cosine | (baseline; anti-predictive) |
| 2 | Lexical | `validators/lexical.py` | VASARI-restricted 200-word TF-IDF + negation penalty | clinical keyword overlap |
| 3 | Structural | `validators/structural.py` | regex VASARI parser → 30-feature set-F1 | laterality, region, features |
| 4 | Numeric | `validators/numeric.py` | cm/mm regex → mm → Jaccard ±1mm | size flips |
| 5 | Modality | `validators/modality.py` | 9-modality set Jaccard | T1↔T2 confusion |
| 6 | Negation | `validators/negation.py` | clause-aware NegEx → negspaCy (Phase 2) | polarity flips |
| 7 | Lesion-type | `validators/lesion_type.py` | 9-family disease set Jaccard | glioma↔meningioma↔metastasis |
| — | **Fusion** | `validators/fusion.py` | sklearn LogisticRegression, class_weight="balanced", threshold 0.5 | → P(valid) |

Fusion math: `P(valid) = σ(w·[s1..s7] + b)`. Decision: ≥0.5 → VALID, else FLAGGED.

## A.7 Perturbation benchmark (Stage 8)

Per clean reference report → 1 clean (paraphrased) + 4 perturbed records. Clean is paraphrased via a **22-group meaning-preserving map** (oedema↔edema) so cosine can't score 1.0 trivially. **8 ops:** laterality flip, lesion-type swap, size error (×0.3/×3), negation flip, region swap, VASARI-feature flip, count change, modality confusion. Metric = AUROC (clean=1 vs perturbed=0). Held-out split = **70/30 by base report ID** (no leakage). VASARI = ~30-feature glioma reporting lexicon, the grounding vocab for lexical + structural axes.

## A.8 Datasets

| Source | Content | Count | License |
|--------|---------|------:|---------|
| `Jupitern52/TextBraTS` (HF) | GPT-4o-drafted, radiologist-refined reports | 369 | MIT |
| `JiayuLei/RadGenome-Brain_MRI` (HF) | 5 disease subsets (glioma, meningioma, metastasis, stroke, WMH) | 1,007 | Research |
| `awsaf49/brats20-dataset-training-validation` (Kaggle) | 3D volumes (T1, T1ce, T2, FLAIR) | 369 | Research |

Total: **1,376 paired reports + 369 volumes**. After perturbation: TextBraTS → 1,829 records, RadGenome → 4,891 records. All public mirrors — no IRB/Synapse needed.

## A.9 ALL RESULTS (the numbers that appear in every deliverable)

### Synthetic benchmark evolution (n=80)
| Round | Added | Fusion AUROC |
|-------|-------|-------------:|
| Bootstrap | skeleton + fusion | trivial 1.0 |
| Round 1 | paraphrase + baselines | 0.682 |
| Round 2 | numeric + modality | 0.787 |
| Round 3 | negation + lesion-type (full 7-axis) | **0.878** |

At 0.878: 3.6× better than BioClinicalBERT (0.247, anti-predictive), 14.2× better than RaTEScore-lite (0.062).

### The four-row paper-grade table (the headline)
| Setting | n_test | Train AUROC | **Test AUROC** | Gap |
|---------|-------:|------------:|---------------:|----:|
| TextBraTS held-out (70/30) | 551 | 0.9990 | **0.9961** | +0.003 |
| RadGenome held-out (70/30) | 1,472 | 0.9699 | **0.9715** | −0.002 |
| TextBraTS → RadGenome | 4,891 | 0.9982 | **0.9358** | +0.062 |
| RadGenome → TextBraTS | 1,829 | 0.9728 | **1.0000** | −0.027 |

All > 0.93. Beats BioClinicalBERT **12.1×** (TextBraTS) / **3.4×** (RadGenome).

### TextBraTS held-out per-validator
NeuroVal-3D fused **0.9961** · Structural 0.6242 · Lexical 0.4218 · BioClinicalBERT 0.0821 · RaTEScore-lite 0.0099

### RadGenome held-out per-validator
NeuroVal-3D fused **0.9715** · Lexical 0.7345 · Structural 0.7244 · Modality 0.6062 · Numeric 0.5927 · BioClinicalBERT 0.2891 · RaTEScore-lite 0.2203

### Phase 2 generator (143.1M params, Kaggle T4, 80 train / 20 test, 5 epochs)
Train loss falls 2.51→1.06 (~58%); test loss plateaus ~1.55–1.58 from epoch 2–3 then drifts up = mild overfit (data-limited at 80 samples, not architecture-limited).

### Phase 2 loop closure — validator scores on 20 real generations
Semantic (BioClinicalBERT) **0.987** · Structural **0.518** · Lexical 0.384 · Negation 0.888 · Numeric/Modality/Lesion-type 1.000 (correctly silent — TextBraTS rarely mentions those) · RaTEScore-lite 0.319

### Phase 2 discrimination gap (clean − hallucinated mean, held-out perturbation set)
NeuroVal-3D fused **+0.488** · structural +0.107 · BioClinicalBERT **−0.001** · lexical −0.027 · RaTEScore-lite −0.052

### Phase 2 detection rate on 9 real hallucinations (threshold 0.5)
Lexical **89%** (8/9) · Structural 33% (3/9) · BioClinicalBERT **0%** (0/9) · RaTEScore-lite 100% (trivial — calls everything wrong)

**The punchline:** BioClinicalBERT misses 100% of real anatomical hallucinations made by a real BART generator; NeuroVal-3D lexical catches 89%.

### Real generation examples
- BraTS20_Training_081: REF "left **frontal**" → GEN "left **temporal**" (region error)
- BraTS20_Training_082: REF "**bilateral**" → GEN "left" (laterality/multiplicity error)
- BraTS20_Training_096: REF "**right** frontal" → GEN "**left** parietal/occipital" (laterality flip)

### Reproducibility
Cross-platform deterministic to **±0.0001** AUROC (Linux GPU Kaggle vs Windows CPU). Phase 1 ~20 min Kaggle T4; Phase 2 full ~75–90 min.

## A.10 Repo structure & how to run

```
src/neuroval3d/
  data/        preprocessing.py, loaders.py, synthetic.py, datasets.py
  models/      encoder.py, projector.py, decoder.py
  validators/  semantic, lexical, structural, numeric, modality, negation,
               lesion_type, fusion, baselines
  evaluation/  perturbation.py, paraphrase.py, benchmark.py, metrics.py
  grounding/   vasari.py, anatomy.py
  viz/         gradcam.py, overlay.py
  utils/       io, checkpoint, logging
  cli.py       (neuroval3d CLI)
tests/         36 pytest tests, 12 files — all green
notebooks/     00_smoke..04_benchmark + kaggle_phase2 / kaggle_phase2_full / kaggle_phase3_concept_bottleneck
docs/          REVIEW_2_REPORT.*, PHASE_1_2_REPORT.*, PROJECT_EXPLAINED.*, paper_outline.md,
               architecture.md, strategy.md, datasets.md, CHECKPOINTS.md, HUMANIZATION_GUIDE.md,
               Minor Project Review 1 (1).pptx, Minor Project Review 2.pptx
dashboard/     ← the UI (see PART B)
```

Run commands (use `.venv/Scripts/python.exe` locally; that venv has NO pip — use `py -3.11 -m pip` to install):
```
.venv/Scripts/python.exe -m neuroval3d.cli benchmark --textbrats --train-frac 0.7
.venv/Scripts/python.exe -m neuroval3d.cli benchmark --radgenome --train-frac 0.7
.venv/Scripts/python.exe -m neuroval3d.cli cross-dataset --train textbrats --test radgenome
.venv/Scripts/python.exe -m neuroval3d.cli cross-dataset --train radgenome --test textbrats
.venv/Scripts/python.exe -m pytest -q          # 36 tests
```
Windows note: always run text-generating scripts with `py -3.11 -X utf8` (cp1252 crashes on → / ≈). Git commits need explicit identity: `git -c user.name="Vikneshwaran Marimuthu" -c user.email="vikneshwaranmarimuthu@rioaconsulting.com" commit ...`

## A.11 Errors & incidents (resolved)

1. **Synapse recovery-codes commit** — `s.pdf` accidentally committed at `4becd8b`; reset, gitignored, reflog-purged, re-committed clean at `83c8399`. STILL PENDING: move `synapse recov codes/s.pdf` out of tree + revoke the Synapse PAT.
2. **cp1252 Unicode crashes** — `→`/`≈` broke Windows console; use `-X utf8` or ASCII.
3. **Triptych NameError** (subject→subject_id) — fixed `f2a6f22`.
4. **Kaggle namespace shadowing** — import-path hardening `7b4b7b3`.
5. **BraTS path resolution** (.nii/.nii.gz) — `c22bc7c`.
6. **n=8 held-out gives AUROC 0.25** — smoke-test artifact, not a bug.
7. **PPT blank divider pages** — fixed `6487932`.
8. **localhost:8080 taken** — use port 5500 for the dashboard.
9. **git identity unknown** — needs explicit `-c user.name/email` flags.

---

# PART B — The Dashboard UI

## B.1 What it is

A self-contained **static showcase website** for the NeuroVal-3D project. Pure vanilla HTML/CSS/JS — no framework, no build step. It presents the whole project (problem → architecture → results → live demo) in a polished, scrollable single page, and includes a **live in-browser validator demo** where you paste a reference + generated report and get real per-axis scores.

Built this session. Lives at `E:\MINOR PROJECT\dashboard\`. Also pushed to a `gh-pages` branch for GitHub Pages hosting.

## B.2 Tech stack

| Layer | Tech | Source |
|-------|------|--------|
| Markup | HTML5, single page | `index.html` |
| Styling | Tailwind CSS (utility classes) + custom CSS | Tailwind via CDN `cdn.tailwindcss.com`; custom in `css/styles.css` |
| Charts | Chart.js 4.4.0 | CDN `cdn.jsdelivr.net` |
| Fonts | Inter + JetBrains Mono | Google Fonts |
| Logic | Vanilla JS (3 files, no framework) | `js/data.js`, `js/validators.js`, `js/app.js` |

No backend. No npm. No build. Opens in any modern browser. Everything except the CDN libs (Tailwind, Chart.js, fonts) is local — so it needs internet for those CDNs to render fully.

## B.3 File-by-file breakdown

```
dashboard/
├── index.html        (503 lines) — all markup: 6 sections + nav + footer
├── css/styles.css    (345 lines) — custom styles layered on Tailwind
├── js/
│   ├── data.js       (224 lines) — ALL baked results + preset examples
│   ├── validators.js (265 lines) — JS reimplementations of the 7 validators + fusion
│   └── app.js        (382 lines) — UI logic, Chart.js rendering, live-demo wiring
├── README.md         (53 lines)  — how to run / deploy
├── .nojekyll                     — tells GitHub Pages to serve files as-is
└── ~$Minor Project Review 2.pptx — STRAY LOCK FILE, should be deleted
```

### `index.html` — 6 `<section>` anchors + footer
1. `#overview` — hero with 4 KPI cards (0.9961 · 12.1× · +0.488 · 89%) + the "left vs right, BERT 0.99 vs NeuroVal 0.18" hook box
2. `#architecture` — the two-pipeline diagram (Generation pipeline + Validation pipeline as stage cards) + a grid of the 7 specialist cards (rendered from `VALIDATORS_INFO`)
3. `#phase1` — tabbed AUROC results: 4 tabs (TextBraTS held-out, RadGenome held-out, TB→RG, RG→TB), each renders a horizontal bar chart + a stats panel; below it the static four-row paper-grade HTML table
4. `#phase2` — 3 Chart.js charts (training loss curves, discrimination gap, detection rate) + the "headline finding" red callout card with the REF/GEN example and the 0.99 vs 0.27 comparison
5. `#demo` — the LIVE validator: preset dropdown + two textareas (reference, generated) + Validate button → verdict banner + 7 per-axis score bars
6. `#team` — 4 students with USNs + guide + tech-stack pills + 4 resource links (GitHub, Kaggle, TextBraTS, RadGenome)
7. `#footer` — disclaimer ("research only — not for clinical use")

Top nav bar is fixed, links scroll-spy to each section. A "N" logo chip top-left, GitHub button top-right.

### `css/styles.css`
Custom classes layered on Tailwind: `.kpi-card`, `.stage-card` (5 color variants: input/prep/model/analysis/output), `.validator-card`, `.tab-btn`, `.chart-card`, `.tech-pill`, `.resource-link`, `.axis-row` / `.axis-bar-fill` (the live-demo score bars, green=pass/red=fail), `.verdict-pass` / `.verdict-fail` banners, plus a `slideUp` animation and responsive `@media` rules. Brand palette: teal `#2a9d8f`, navy `#1f3a4f`, red `#e63946`.

### `js/data.js` — everything baked in (NO live Python connection)
- `VALIDATORS_INFO` — the 7 specialists + fusion (name, tech, what-it-catches, icon) for the architecture grid
- `PHASE1_RESULTS` — object with 4 keys (`textbrats`, `radgenome`, `cross1`, `cross2`); each has label, description, headline number, per-validator AUROC array, and a stats array. Drives the Phase 1 tabbed charts.
- `PHASE2_LOSS` — epochs + train + test loss arrays (Run B numbers: train 2.8674→1.0906, test 1.7292→1.6228)
- `PHASE2_DISCRIMINATION` — labels, cleanScores, hallucScores, gaps, isOurs flags
- `PHASE2_DETECTION` — labels, detectionRate, overallAccuracy, isOurs, trustworthy
- `PRESET_EXAMPLES` — 7 preset (reference, generated) pairs for the live demo: clean, laterality flip, region swap, negation flip, size error, modality confusion, lesion-type swap

### `js/validators.js` — JS reimplementations of the 7 Python validators
These are **simplified JavaScript ports** of the real Python validators so the live demo gives realistic (not fake) scores without a server:
- `semanticScore(gen, ref)` — word-overlap proxy for BioClinicalBERT cosine (intentionally high-baseline 0.85–1.0, mirroring how BERT is surface-dominated)
- `lexicalScore` — VASARI-vocab Jaccard
- `structuralScore` — extracts VASARI features (laterality, region, enhancement, oedema, necrosis, haemorrhage, count) → set-F1
- `numericScore` — cm/mm regex → mm → Jaccard ±1mm
- `modalityScore` — 9-modality set Jaccard
- `negationScore` — clause-aware NegEx over clinical terms → (term, polarity) Jaccard
- `lesionTypeScore` — 9-family Jaccard
- `fusionScore(axisScores)` — weighted sum (structural 0.35, lexical 0.20, others smaller) → sigmoid. Weights roughly calibrated from the real trained model so structural+lexical dominate.
- `runAllValidators(gen, ref)` — master entry; returns all 7 + fused
- Vocabularies baked in: `VASARI_VOCAB`, `MODALITIES`, `LESION_FAMILIES`, `NEGATION_CUES`, `NEGATION_TERMS`, `REGIONS`

### `js/app.js` — UI wiring
- `renderValidatorGrid()` — builds the 7 architecture cards
- `initPhase1Tabs()` + `renderPhase1Tab()` — tab switching + Chart.js horizontal bar per dataset
- `renderLossChart()` / `renderDiscriminationChart()` / `renderDetectionChart()` — the 3 Phase 2 Chart.js charts
- `initLiveDemo()` — preset loader, Validate button → `runAllValidators` → `renderResults()` (verdict banner + animated per-axis bars). Loads the "clean" preset on page load and auto-validates.
- `initSmoothScroll()` — nav anchor smooth scrolling
- Boots on `DOMContentLoaded`

## B.4 The live demo (how it works)

1. User picks a preset from the dropdown (or types custom ref + gen text).
2. Click **Validate** → `runAllValidators(gen, ref)` runs all 7 JS validators + fusion entirely in the browser.
3. Output: a **verdict banner** (VALID green / FLAGGED red with the P(valid) number) + **7 horizontal score bars**, each green (≥0.5) or red (<0.5), with a dashed threshold line at 0.5.
4. Preset examples demonstrate each error class — e.g. the "Laterality flip" preset makes the structural axis drop and the report get FLAGGED, while "Clean" stays VALID.

## B.5 How to run locally

```powershell
py -3.11 -m http.server -d dashboard 5500
```
Then open **http://localhost:5500**.

**IMPORTANT:** use port **5500**, NOT 8080 — on this machine port 8080 is already taken by another app and returns 404s. Any free port works (5500, 3000, 8000); 5500 is what we verified. Opening `index.html` directly by double-click mostly works too, but a local server avoids any `file://` quirks.

## B.6 How to deploy (GitHub Pages)

The dashboard is already pushed to a **`gh-pages`** branch (via `git subtree push --prefix dashboard origin gh-pages`). To make it live, someone with repo access must do this ONCE in the browser:

1. Go to https://github.com/vicky-16032005/neuroval3d/settings/pages
2. Under **Build and deployment → Source**, choose **Deploy from a branch**
3. **Branch** = `gh-pages`, **Folder** = `/ (root)`, click **Save**
4. Wait ~1–2 min → live at **https://vicky-16032005.github.io/neuroval3d/**

The `.nojekyll` file is already in place so Pages serves the JS/CSS unmodified.

## B.7 How to update / redeploy

After editing anything in `dashboard/`:
```powershell
git add dashboard/
git -c user.name="Vikneshwaran Marimuthu" -c user.email="vikneshwaranmarimuthu@rioaconsulting.com" commit -m "Update dashboard"
git push origin main
git subtree push --prefix dashboard origin gh-pages
```
To change a number on the site, edit `dashboard/js/data.js` (all results live there). To change the live-demo logic, edit `dashboard/js/validators.js`. To change layout/copy, edit `dashboard/index.html`.

## B.8 Honest limitations (be transparent about these)

- **The dashboard is NOT connected to the Python code.** All numbers are hardcoded in `js/data.js` from the real Phase 1/2 runs. The 7 validators in `js/validators.js` are simplified JS reimplementations — they give realistic scores for the demo but are not byte-identical to the Python validators (no BioClinicalBERT in the browser; the semantic axis is a word-overlap proxy).
- For a truly live-connected demo you'd need a Flask/FastAPI backend serving the real `neuroval3d` package — out of scope for a static GitHub Pages site (Pages can't run Python).
- This is appropriate for a review demo / showcase, not a production validator endpoint.

## B.9 Known issues / cleanup

- `dashboard/~$Minor Project Review 2.pptx` — a stray PowerPoint lock file accidentally created in this folder. **Delete it** (and it should never be committed). It is NOT part of the dashboard.
- The dashboard depends on three CDNs (Tailwind, Chart.js, Google Fonts). Offline, the page renders unstyled / chartless. For a fully offline version, vendor those libs locally into `dashboard/`.

## B.10 Commit reference

- Dashboard source committed to `main` at **`9a29d98`** ("Add NeuroVal-3D dashboard — static showcase site with live validator demo").
- `gh-pages` branch holds the dashboard files at root for Pages.

---

# Quick-start for a fresh session

1. **Understand the science:** PART A. The thesis is in A.1, the numbers in A.9.
2. **Understand the UI:** PART B. It's a static site in `dashboard/`; run with `py -3.11 -m http.server -d dashboard 5500`.
3. **Confirm git state:** `git status` + `git log --oneline -5`. Latest is `9a29d98` (dashboard). `main` is pushed; `gh-pages` exists.
4. **Most likely next tasks:** enable GitHub Pages (B.6), fix the Review 2 PPT (user flagged it as "not correct" without specifics), delete the stray `~$...pptx` lock file, or extend the dashboard.
5. **Environment gotchas:** Windows; `py -3.11 -X utf8` for text scripts; `.venv` python has no pip; git needs explicit identity flags; dashboard on port 5500 not 8080.

*End of context file. This single file fully describes both NeuroVal-3D and its dashboard UI.*
