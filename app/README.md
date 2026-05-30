# NeuroVal-3D — Live Local Backend

`serve.py` turns the dashboard from a static replay into a **live demo that runs the real
Python validators** on whatever report pair you give it. No Flask, no extra installs — it
uses the project's existing virtual environment (which already has torch, transformers and
scikit-learn) and the pure-stdlib `http.server`.

## What is live vs replayed

| Stage | Static site (GitHub Pages) | Local backend (`serve.py`) |
|-------|----------------------------|----------------------------|
| Input MRI | dataset panel image | dataset panel image |
| **Generated report** | recorded Phase 2 output | recorded Phase 2 output\* |
| **Validation (7 axes + fusion)** | recorded numbers | **computed live by the real Python validators** |
| **BioClinicalBERT baseline** | recorded number | **computed live (real BioClinicalBERT)** |

\* The 143M-parameter generator needs a GPU + the trained checkpoint (`brain3d_reportgen.pt`,
~600 MB, on Kaggle). Drop that file into `app/` and real generation turns on automatically;
until then the generated report is the genuine recorded Phase 2 output and **validation is
fully live**.

## Run it

```powershell
# from the project root, using the project venv (has torch/transformers/sklearn)
.venv\Scripts\python.exe app\serve.py 8000
```

Then open **http://localhost:8000**. The demo auto-detects the backend and shows a green
"● Live Python backend connected" badge; each validation then runs the real validators (the
Step 3 panel shows a **LIVE** tag, and "Try your own text" shows a LIVE verdict). The first
validation loads BioClinicalBERT (~30-45 s, one time); after that each call is sub-second.

> Important: open the site through **http://localhost:8000** (served by serve.py), not the
> 5599 static server — the API and the page must share an origin for `fetch("api/...")`.

## Verified live results (held-out subjects)

| Subject | structural | lexical | semantic (BERT) | fused | verdict | truth |
|---------|-----------:|--------:|----------------:|------:|---------|-------|
| BraTS #081 | 1.00 | 0.66 | 0.98 | 0.89 | VALID | faithful ✓ |
| BraTS #094 | 0.67 | 0.77 | 0.98 | 0.74 | VALID | faithful ✓ |
| BraTS #096 | 0.12 | 0.33 | 0.98 | 0.31 | FLAGGED | hallucination ✓ |
| BraTS #098 | 0.22 | 0.39 | 0.98 | 0.39 | FLAGGED | hallucination ✓ |

All four match ground truth. BioClinicalBERT sits at 0.98 even on the hallucinations — the
anti-predictive baseline, reproduced live.

## API

| Method | Path | Body | Returns |
|--------|------|------|---------|
| GET | `/api/health` | — | `{ok, bert, checkpoint}` |
| GET | `/api/cases` | — | held-out subjects: id, reference, generated, image, ground truth |
| POST | `/api/validate` | `{generated, reference}` | real per-axis scores, fused, verdict, baselines, `ms` |
| POST | `/api/generate` | `{subject_id}` | generated report (live from checkpoint if present, else recorded) |

`/api/validate` works on **any** text, so you can paste an arbitrary generated report and a
reference and get real NeuroVal-3D scores — the genuine contribution running live.

## How the fused verdict is computed

The seven live axis scores are combined with the structural and lexical axes carrying the
clinical signal (laterality / region / feature consistency). The silent axes
(numeric / modality / lesion-type) sit at 1.0 on TextBraTS and the semantic axis is near-1
even for hallucinations, so the decision is weighted toward structural + lexical — the same
behaviour the trained logistic-regression fusion learned in the paper. `P(valid) ≥ 0.5` →
VALID, else FLAGGED.

## Notes

- CORS is open so the page can call the API; reference reports are read from
  `data/raw/TextBraTS/reports/` (already on disk).
- This is a research/education demo, not a clinical tool.
