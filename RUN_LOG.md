# NeuroVal-3D — Run Log

Every command executed against this project is logged here in reverse-chronological order.
Format:
```
## YYYY-MM-DD HH:MM — <description>
- cmd: `<command>`
- exit: <code>
- stdout: <≤ 200 char snippet>
- artifacts: <paths created/modified>
- notes: <any context>
```

---

## 2026-04-30 — Phase 0 bootstrap

### Steps completed
1. Probed environment: Python 3.14 default + 3.11.9 secondary, no local GPU, git 2.51, uv 0.11.1, 127 GB free on E:
2. Created full directory tree: `src/neuroval3d/{data,models,grounding,validators,evaluation,viz,utils}/`, `configs/`, `tests/`, `notebooks/`, `scripts/`, `docs/`, `data/{raw,processed,synthetic}/`, `outputs/{checkpoints,logs,results}/`
3. Authored project config: `pyproject.toml` (uv-managed, Python 3.10–3.12, MIT, hatchling build), `.gitignore`, `LICENSE`, `Makefile`
4. Authored docs: `README.md`, `docs/{architecture,strategy,datasets,CHECKPOINTS}.md`
5. Wrote 25 source modules (~1,800 LOC) covering all 8 stages:
   - `utils/`: io, logging, checkpoint manager
   - `data/`: dataset registry, Stage 1 preprocessor, synthetic-report generator
   - `grounding/`: 30-feature VASARI lexicon, parser, anatomical anchorer
   - `validators/`: semantic (BioClinicalBERT), lexical (VASARI TF-IDF + negation), structural (segmentation-grounded F1), fusion (logistic)
   - `evaluation/`: 8-op perturbation generator, NLG metrics wrappers, end-to-end benchmark
   - `models/`: encoder factory (Swin-UNETR / dummy fallback), MLP projector, BART decoder
   - `viz/`: 3D Grad-CAM stub, triptych overlay
   - `cli.py`: typer-based CLI (`neuroval3d info|version|benchmark|vasari-demo`)
6. Wrote 6 pytest test files: smoke (full-pipeline), vasari, perturbation, validator, synthetic, preprocessing
7. Wrote 5 configs: `default.yaml`, `data.yaml`, `model.yaml`, `validator.yaml`, `eval.yaml`
8. Wrote 5 scripts: `setup_colab.sh`, `setup_kaggle.sh`, `run_smoke.py`, `generate_notebooks.py`, `download_atlases.py`
9. Generated 5 paired notebooks: smoke, vasari, perturbation, validator, benchmark
10. Syntax-checked all source via `py -3.11 -m compileall src tests scripts` — clean
11. `git init -b main`; staged all files

### In progress at pause
- `uv sync` (bg task `blawbwfoz`) — **completed** (exit 0). Venv at `.venv/`, all critical pkgs verified.
- `pytest -q` (bg task `bn0k0czyt`) — kicked off, not awaited. Output: `C:\Users\Admin\AppData\Local\Temp\claude\E--MINOR-PROJECT\02e55765-00d8-4c24-bbad-6046fe81cf2c\tasks\bn0k0czyt.output`

### PAUSE 2026-04-30 12:58
User signaled 96% session-limit; build paused. Pickup script in `outputs/logs/session_pause_2026-04-30T12-58.md` and memory `project_pause_state.md`. Resume after 5-hour cap reset.

---

## 2026-04-30 14:52 — RESUME — n=80 benchmark run

### cmd
`.venv/Scripts/python.exe -m neuroval3d.cli benchmark --synthetic --n-samples 80`

### artifacts
- `outputs/results/20260430_145218/auroc_table.md`
- `outputs/results/20260430_145218/perturbation_set.jsonl` (400 records)
- `outputs/results/20260430_145218/scores.jsonl` (per-record validator scores)
- `outputs/results/20260430_145218/result.json`
- `outputs/checkpoints/CP-20260430-bench-145218/` (auto-registered)

### result table

| Validator | Overall AUROC | laterality | lesion_type | modality | negation | region | size | vasari_flip |
|-----------|---------------|----------|----------|----------|----------|----------|----------|----------|
| fusion | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| lexical | 0.8598 | 1.0000 | 1.0000 | 0.5226 | 0.8125 | 1.0000 | 0.6648 | 1.0000 |
| semantic | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| structural | 0.6695 | 0.7707 | 0.4938 | 0.5044 | 0.8232 | 0.6885 | 0.4966 | 0.9660 |

### honest interpretation
This is a smoke benchmark, **not** the headline paper number. "Clean" examples here are byte-identical to the reference — that's why `semantic` and `fusion` trivially hit 1.0. The genuinely informative signal is the per-op breakdown of the *individual* axes:

- **Lexical (0.86)** — strong on laterality / lesion_type / region / vasari_flip / negation; weak on **modality (0.52)** and **size (0.66)** because "T1/T2" tokens and numeric sizes aren't in the VASARI vocabulary. We can fix this in Phase 4 by adding a small modality + numerics sub-vocabulary.
- **Structural (0.67)** — VASARI parser is regex-based; misses lesion-type tokens that aren't in the lexicon and can't reason about modality. Replace with scispaCy + UMLS linker in Phase 4.
- **Semantic (1.00 trivially)** — BioClinicalBERT loaded and ran. The task is too easy because clean and reference are identical. The real test (Phase 4) replaces "clean" with model-generated reports vs reference.

The pipeline runs end-to-end on a CPU laptop; BioClinicalBERT downloads + processes 800 forwards in ~6 min on this machine. Phase 0 done.

