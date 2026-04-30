# NeuroVal-3D — Checkpoint Ledger

Every milestone, model checkpoint, and reproducible result lands here. Append-only.

| ID | Date | Phase | What | Path / Memory ref |
|----|------|-------|------|-------------------|
| CP-000 | 2026-04-30 | Phase 0 | Repo skeleton + dependency lock + artifact persistence policy | `pyproject.toml`, `README.md`, `RUN_LOG.md` |
| CP-001 | 2026-04-30 | Phase 0 | Phase 0 complete — 8-stage scaffold, validator + perturbation benchmark, pytest exit 0, n=80 benchmark AUROC table emitted, first git commit `633dd46` | git: `633dd46` · memory: `project_phase0_complete.md` |
| CP-20260430-bench-130437 | 2026-04-30 | Phase 4 (validator+benchmark) | Smoke perturbation benchmark, n=8 (`auroc_fusion`=1.0000) | `outputs/checkpoints/CP-20260430-bench-130437/` |
| CP-20260430-bench-145218 | 2026-04-30 | Phase 4 (validator+benchmark) | Production perturbation benchmark, n=80 (`auroc_fusion`=1.0000, lexical=0.860, structural=0.670, semantic=1.000 with BioClinicalBERT) | `outputs/checkpoints/CP-20260430-bench-145218/` |
| PAUSE-1 | 2026-04-30 12:58 | Phase 0 | 96% session cap pause (resolved at 14:52 by CP-001) | `outputs/logs/session_pause_2026-04-30T12-58.md` |
| CP-20260430-bench-150358 | 2026-04-30 | Phase 4 (validator+benchmark) | Perturbation benchmark with n=8 (`auroc_fusion`=1.0000) | `outputs\checkpoints\CP-20260430-bench-150358` |
| CP-20260430-bench-150510 | 2026-04-30 | Phase 4 (validator+benchmark) | Perturbation benchmark with n=8 (`auroc_fusion`=1.0000) | `outputs\checkpoints\CP-20260430-bench-150510` |
| CP-20260430-bench-150644 | 2026-04-30 | Phase 4 (validator+benchmark) | Perturbation benchmark with n=8 (`auroc_fusion`=1.0000) | `outputs\checkpoints\CP-20260430-bench-150644` |
| CP-20260430-bench-150812 | 2026-04-30 | Phase 4 (validator+benchmark) | Perturbation benchmark with n=8 (`auroc_fusion`=1.0000) | `outputs\checkpoints\CP-20260430-bench-150812` |
| CP-20260430-bench-151239 | 2026-04-30 | Phase 4 (validator+benchmark) | Perturbation benchmark with n=8 (`auroc_fusion`=0.7344) | `outputs\checkpoints\CP-20260430-bench-151239` |
| CP-20260430-bench-151328 | 2026-04-30 | Phase 4 (validator+benchmark) | Perturbation benchmark with n=80 (`auroc_fusion`=0.6822) | `outputs\checkpoints\CP-20260430-bench-151328` |
| CP-20260430-bench-152440 | 2026-04-30 | Phase 4 (validator+benchmark) | Perturbation benchmark with n=8 (`auroc_fusion`=0.8281) | `outputs\checkpoints\CP-20260430-bench-152440` |
| CP-20260430-bench-152526 | 2026-04-30 | Phase 4 (validator+benchmark) | Perturbation benchmark with n=80 (`auroc_fusion`=0.7866) | `outputs\checkpoints\CP-20260430-bench-152526` |
| CP-20260430-bench-153507 | 2026-04-30 | Phase 4 (validator+benchmark) | Perturbation benchmark with n=8 (`auroc_fusion`=0.8594) | `outputs\checkpoints\CP-20260430-bench-153507` |
| CP-20260430-bench-153654 | 2026-04-30 | Phase 4 (validator+benchmark) | Perturbation benchmark with n=8 (`auroc_fusion`=0.8594) | `outputs\checkpoints\CP-20260430-bench-153654` |
| CP-20260430-bench-153730 | 2026-04-30 | Phase 4 (validator+benchmark) | Perturbation benchmark with n=80 (`auroc_fusion`=0.8775) | `outputs\checkpoints\CP-20260430-bench-153730` |
| CP-20260430-bench-171052 | 2026-04-30 | Phase 4 (validator+benchmark) | Perturbation benchmark with n=369 (`auroc_fusion`=0.9982) | `outputs\checkpoints\CP-20260430-bench-171052` |
| CP-20260430-bench-173116 | 2026-04-30 | Phase 4 (validator+benchmark) | Perturbation benchmark with n=8 (`auroc_fusion`=0.8594) | `outputs\checkpoints\CP-20260430-bench-173116` |

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

<!-- auto-rows below have been merged into the main table above; this region is reserved for future automatic appends -->

