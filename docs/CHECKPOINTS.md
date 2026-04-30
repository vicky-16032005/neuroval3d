# NeuroVal-3D — Checkpoint Ledger

Every milestone, model checkpoint, and reproducible result lands here. Append-only.

| ID | Date | Phase | What | Path / Memory ref |
|----|------|-------|------|-------------------|
| CP-000 | 2026-04-30 | Phase 0 | Repo skeleton + dependency lock + artifact persistence policy | `pyproject.toml`, `README.md`, `RUN_LOG.md`, memory: `project_phase0_complete.md` |

---

## Conventions

- **CP-XXX** — sequential, never reused
- **Phase** — Phase 0 (bootstrap), Phase 1 (preprocessing), Phase 2 (encoder), Phase 3 (alignment+gen), Phase 4 (validator+benchmark), Phase 5 (paper)
- **Path / Memory ref** — file path under `outputs/checkpoints/` AND/OR memory file under `~/.claude/projects/E--MINOR-PROJECT/memory/`

When a model checkpoint is saved, the entry MUST include:
- the path to the `.pt` / `.safetensors` file (gitignored, but logged here)
- the hash of the training config used
- the metric score that justified saving
- the git SHA at training time

| CP-20260430-bench-130437 | 2026-04-30 | Phase 4 (validator+benchmark) | Perturbation benchmark with n=8 (`auroc_fusion`=1.0000) | `outputs\checkpoints\CP-20260430-bench-130437` |

| CP-20260430-bench-145218 | 2026-04-30 | Phase 4 (validator+benchmark) | Perturbation benchmark with n=80 (`auroc_fusion`=1.0000) | `outputs\checkpoints\CP-20260430-bench-145218` |
