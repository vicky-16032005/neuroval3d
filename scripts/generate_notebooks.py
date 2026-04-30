"""Generate paired demo notebooks from the curated `notebooks/_specs.py` definitions.

We don't auto-convert every module to a notebook because that produces noise. Instead we
hand-curate the demos that the team actually needs (smoke test, VASARI parser tour,
perturbation generator, validator benchmark, preprocessing).

Each spec is a list of (cell_type, source) tuples; this script renders them to .ipynb JSON.
"""
from __future__ import annotations

import json
from pathlib import Path

NOTEBOOKS_DIR = Path("notebooks")


def make_notebook(cells: list[tuple[str, str]]) -> dict:
    nb = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    for cell_type, source in cells:
        nb["cells"].append(
            {
                "cell_type": cell_type,
                "metadata": {},
                "source": source.splitlines(keepends=True),
                **({"outputs": [], "execution_count": None} if cell_type == "code" else {}),
            }
        )
    return nb


def write(name: str, cells: list[tuple[str, str]]) -> Path:
    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    path = NOTEBOOKS_DIR / name
    path.write_text(json.dumps(make_notebook(cells), indent=1), encoding="utf-8")
    return path


SMOKE = [
    ("markdown", "# 00 — NeuroVal-3D smoke test\n\nVerifies the pipeline imports + the perturbation benchmark runs end-to-end on synthetic reports.\n"),
    ("code", "%load_ext autoreload\n%autoreload 2\n\nimport sys, pathlib\nsys.path.insert(0, str(pathlib.Path('..').resolve() / 'src'))\n"),
    ("code", "import neuroval3d\nprint('neuroval3d', neuroval3d.__version__)\n"),
    ("code", "from neuroval3d.evaluation import run_benchmark\nresult = run_benchmark(use_synthetic=True, n_samples=24, n_per_report=3)\nprint(result.summary_table())\n"),
    ("markdown", "If the table renders with finite AUROC numbers, Phase 0 is healthy. Move to `01_vasari_demo.ipynb` next.\n"),
]

VASARI_DEMO = [
    ("markdown", "# 01 — VASARI lexicon + parser demo\n\nShows the 30-feature VASARI lexicon and parses a sample report into a feature vector.\n"),
    ("code", "from neuroval3d.grounding import VASARI_FEATURES, VASARIParser, vasari_vocabulary\nprint(len(VASARI_FEATURES), 'features')\nfor f in VASARI_FEATURES[:6]:\n    print(f.code, f.name, '->', f.values)\n"),
    ("code", "vocab = __import__('neuroval3d.grounding.vasari', fromlist=['vasari_vocabulary']).vasari_vocabulary()\nprint(len(vocab), 'vocab tokens')\nprint(vocab[:30])\n"),
    ("code", "report = '''There is a 3.5 cm avidly enhancing intra-axial mass in the left frontal lobe.\nMarked surrounding oedema. No restricted diffusion.'''\nfeats = VASARIParser().parse(report).to_dict()\nfeats\n"),
]

PERTURBATION_DEMO = [
    ("markdown", "# 02 — Perturbation generator (Stage 8)\n\nApply each error type to a sample report and compare outputs.\n"),
    ("code", "from neuroval3d.evaluation.perturbation import PerturbationOp, perturb\nimport numpy as np\nrng = np.random.default_rng(0)\nreport = ('There is a 3.5 cm avidly enhancing left frontal mass with marked oedema. '\n          'Two satellite lesions are noted. No restricted diffusion. T1 imaging shows the lesion clearly.')\nfor op in PerturbationOp:\n    rec = perturb(report, op, rng)\n    print(op.value, '->', (rec.perturbed if rec else '<no candidate>'))\n"),
]

VALIDATOR_DEMO = [
    ("markdown", "# 03 — Validator scores on a clean vs perturbed report\n\nDemonstrates that the validator gives lower scores to hallucinated reports.\n"),
    ("code", "from neuroval3d.validators import LexicalValidator, SemanticValidator, StructuralValidator, FusionValidator\n"),
    ("code", "ref = 'Right frontal high-grade glioma, 3.5 cm, with moderate edema. No restricted diffusion.'\nclean = ref\nflipped = ref.replace('Right', 'Left').replace('No restricted diffusion', 'Restricted diffusion is present')\n\nsem = SemanticValidator()\nlex = LexicalValidator().fit([ref, flipped, clean])\nst = StructuralValidator()\n\nprint('clean :', sem.score(ref, clean), lex.score(ref, clean), st.score(ref, clean))\nprint('flip  :', sem.score(ref, flipped), lex.score(ref, flipped), st.score(ref, flipped))\n"),
]

BENCHMARK_DEMO = [
    ("markdown", "# 04 — Perturbation benchmark + AUROC table\n\nThis is the headline notebook for the paper: it produces the AUROC table comparing each validator axis.\n"),
    ("code", "from neuroval3d.evaluation import run_benchmark\nresult = run_benchmark(use_synthetic=True, n_samples=80, n_per_report=4)\nprint(result.summary_table())\n"),
    ("code", "import json\nprint(json.dumps(result.auroc_overall, indent=2))\nprint(json.dumps(result.auroc_by_op, indent=2))\n"),
]


def main() -> None:
    write("00_smoke_test.ipynb", SMOKE)
    write("01_vasari_demo.ipynb", VASARI_DEMO)
    write("02_perturbation_demo.ipynb", PERTURBATION_DEMO)
    write("03_validator_demo.ipynb", VALIDATOR_DEMO)
    write("04_benchmark_demo.ipynb", BENCHMARK_DEMO)
    print("notebooks written:", sorted(p.name for p in NOTEBOOKS_DIR.glob("*.ipynb")))


if __name__ == "__main__":
    main()
