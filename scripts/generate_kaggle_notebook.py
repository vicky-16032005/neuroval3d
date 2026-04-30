"""Generate `notebooks/kaggle_phase2.ipynb` — the Phase-2 Kaggle runner.

The notebook clones our GitHub repo, downloads TextBraTS + RadGenome reports from HF,
runs both held-out benchmarks, demonstrates Stage 1 preprocessing on one BraTS volume
from the attached Kaggle dataset, and produces an AUROC comparison bar chart.

Edit GITHUB_URL at the top of cell 1 before running.
"""
from __future__ import annotations

import json
from pathlib import Path

NB_PATH = Path("notebooks/kaggle_phase2.ipynb")


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


CELLS = [
    md(
        "# NeuroVal-3D — Phase 2 on Kaggle\n"
        "\n"
        "End-to-end demo of the validator stack on real radiology data, plus Stage 1 preprocessing on one BraTS 2020 volume.\n"
        "\n"
        "**What this notebook does:**\n"
        "1. Clones the project from your GitHub mirror\n"
        "2. Downloads 369 TextBraTS reports (HuggingFace, MIT)\n"
        "3. Downloads 1,007 RadGenome-Brain MRI reports (HuggingFace, research-only)\n"
        "4. Runs the held-out perturbation benchmark on **TextBraTS**\n"
        "5. Runs the held-out perturbation benchmark on **RadGenome-Brain MRI**\n"
        "6. Runs Stage 1 preprocessing on one BraTS 2020 volume from the attached Kaggle dataset\n"
        "7. Plots the headline AUROC bar chart vs BioClinicalBERT and RaTEScore-lite\n"
        "\n"
        "Expected wall-clock with T4 GPU: ~20 minutes. CPU only: ~1.5 hours.\n"
    ),
    md(
        "## Prerequisites\n"
        "\n"
        "**Kaggle setup before running:**\n"
        "1. Settings (right rail) → **Internet: ON**\n"
        "2. Settings → **Accelerator: GPU T4 ×2** (or P100, either works)\n"
        "3. **Add Data** (right rail) → search **`brats20-dataset-training-validation`** by user **awsaf49** → Add\n"
        "4. Edit `GITHUB_URL` in the cell below to point to your fork/clone of the repo\n"
        "\n"
        "**Direct dataset link:**\n"
        "https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation\n"
    ),
    code(
        "# Project lives at vicky-16032005/neuroval3d (set 2026-04-30)\n"
        "GITHUB_URL = \"https://github.com/vicky-16032005/neuroval3d.git\"\n"
        "\n"
        "import os, subprocess, sys\n"
        "PROJECT_DIR = \"/kaggle/working/neuroval3d\"\n"
        "\n"
        "# 1. Heavy installs Kaggle doesn't ship by default\n"
        "subprocess.run([sys.executable, \"-m\", \"pip\", \"install\", \"-q\", \"--upgrade\", \"pip\"], check=True)\n"
        "subprocess.run([sys.executable, \"-m\", \"pip\", \"install\", \"-q\", \"monai>=1.3\", \"nibabel>=5.2\", \"SimpleITK>=2.3\", \"python-dotenv\"], check=True)\n"
        "\n"
        "# 2. Clone the project\n"
        "if not os.path.isdir(PROJECT_DIR):\n"
        "    subprocess.run([\"git\", \"clone\", GITHUB_URL, PROJECT_DIR], check=True)\n"
        "os.chdir(PROJECT_DIR)\n"
        "\n"
        "# 3. Install our package + extras (data) for huggingface_hub etc.\n"
        "subprocess.run([sys.executable, \"-m\", \"pip\", \"install\", \"-q\", \"-e\", \".[data,eval]\"], check=True)\n"
        "\n"
        "print(\"Project at:\", PROJECT_DIR)\n"
        "print(os.listdir(PROJECT_DIR))\n"
    ),
    code(
        "# Sanity check imports + GPU\n"
        "import torch, sys\n"
        "print(f\"python: {sys.version.split()[0]}\")\n"
        "print(f\"torch:  {torch.__version__}, cuda available: {torch.cuda.is_available()}\")\n"
        "if torch.cuda.is_available():\n"
        "    print(f\"  device: {torch.cuda.get_device_name(0)}, vram: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB\")\n"
        "\n"
        "import neuroval3d\n"
        "print(f\"neuroval3d: v{neuroval3d.__version__}\")\n"
    ),
    md(
        "## 1 · Download real radiology reports from HuggingFace\n"
        "\n"
        "TextBraTS (369) and RadGenome-Brain MRI (1,007) — both fully public, no auth needed.\n"
    ),
    code(
        "# This pulls all 369 TextBraTS reports into data/raw/TextBraTS/reports/\n"
        "import subprocess, sys\n"
        "subprocess.run([sys.executable, \"scripts/download_textbrats_reports.py\"], check=True)\n"
    ),
    code(
        "# This pulls 16 JSON files (5 disease subsets × 3 sections + split) into data/raw/RadGenome-BrainMRI/\n"
        "import subprocess, sys\n"
        "subprocess.run([sys.executable, \"scripts/download_radgenome_reports.py\"], check=True)\n"
    ),
    md(
        "## 2 · Held-out benchmark on **TextBraTS** (369 reports)\n"
        "\n"
        "70/30 split by `original_id` — fusion is trained on 258 base reports, evaluated on 111 unseen ones.\n"
    ),
    code(
        "from neuroval3d.evaluation import run_benchmark\n"
        "from neuroval3d.data import textbrats_reports_only\n"
        "\n"
        "tb_reports = textbrats_reports_only()\n"
        "print(f\"Loaded {len(tb_reports)} TextBraTS reports.\")\n"
        "\n"
        "result_tb = run_benchmark(\n"
        "    reports=tb_reports,\n"
        "    use_synthetic=False,\n"
        "    n_samples=len(tb_reports),\n"
        "    train_frac=0.7,\n"
        ")\n"
        "print(result_tb.summary_table())\n"
    ),
    md(
        "## 3 · Held-out benchmark on **RadGenome-Brain MRI** (1,007 reports)\n"
        "\n"
        "Larger, richer dataset — 5 disease subsets, explicit T1/T2/FLAIR mentions, all 7 perturbation ops fire.\n"
    ),
    code(
        "from neuroval3d.data import radgenome_reports_only\n"
        "\n"
        "rg_reports = radgenome_reports_only(section=\"global_finding\")\n"
        "print(f\"Loaded {len(rg_reports)} RadGenome reports.\")\n"
        "\n"
        "result_rg = run_benchmark(\n"
        "    reports=rg_reports,\n"
        "    use_synthetic=False,\n"
        "    n_samples=len(rg_reports),\n"
        "    train_frac=0.7,\n"
        ")\n"
        "print(result_rg.summary_table())\n"
    ),
    md(
        "## 4 · Stage 1 preprocessing on a real BraTS 2020 volume\n"
        "\n"
        "Loads four MRI modalities (T1, T1ce, T2, FLAIR) for one subject, runs N4 bias correction, brain masking, z-score normalisation, and reshapes to a 128³ cube. Saves a triptych preview of all four modalities.\n"
    ),
    code(
        "from pathlib import Path\n"
        "from neuroval3d.data import Stage1Preprocessor, PreprocessingConfig\n"
        "\n"
        "# Path inside Kaggle container after attaching `brats20-dataset-training-validation`:\n"
        "BRATS_ROOT = Path(\"/kaggle/input/brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData\")\n"
        "if not BRATS_ROOT.exists():\n"
        "    raise SystemExit(\n"
        "        \"BraTS dataset not attached. In Kaggle right-rail: Add Data → \"\n"
        "        \"awsaf49/brats20-dataset-training-validation\"\n"
        "    )\n"
        "\n"
        "subject = \"BraTS20_Training_001\"\n"
        "subject_dir = BRATS_ROOT / subject\n"
        "paths = {\n"
        "    \"T1\":    str(subject_dir / f\"{subject}_t1.nii\"),\n"
        "    \"T1ce\":  str(subject_dir / f\"{subject}_t1ce.nii\"),\n"
        "    \"T2\":    str(subject_dir / f\"{subject}_t2.nii\"),\n"
        "    \"FLAIR\": str(subject_dir / f\"{subject}_flair.nii\"),\n"
        "}\n"
        "seg_path = str(subject_dir / f\"{subject}_seg.nii\")\n"
        "\n"
        "pre = Stage1Preprocessor(PreprocessingConfig(\n"
        "    target_shape=(128, 128, 128),\n"
        "    bias_correct=True,\n"
        "    normalize=\"zscore\",\n"
        "))\n"
        "out = pre.run(paths, seg_path=seg_path)\n"
        "\n"
        "print(f\"Preprocessed volume shape: {out.volume.shape}\")\n"
        "print(f\"Brain-mask voxel count:   {int(out.brain_mask.sum()):,}\")\n"
        "print(\"Per-modality stats:\")\n"
        "for m, s in out.modality_stats.items():\n"
        "    print(f\"  {m}: mean_pre_norm={s['mean']:.2f}, std_pre_norm={s['std']:.2f}\")\n"
    ),
    code(
        "# Triptych: each modality, axial + coronal mid-slice\n"
        "import matplotlib.pyplot as plt\n"
        "import numpy as np\n"
        "\n"
        "vol = out.volume\n"
        "modalities = [\"T1\", \"T1ce\", \"T2\", \"FLAIR\"]\n"
        "fig, axes = plt.subplots(2, 4, figsize=(16, 8))\n"
        "for i, m in enumerate(modalities):\n"
        "    D, H, W = vol.shape[1:]\n"
        "    axes[0, i].imshow(vol[i, D // 2], cmap=\"gray\")\n"
        "    axes[0, i].set_title(f\"{m} — axial\")\n"
        "    axes[0, i].axis(\"off\")\n"
        "    axes[1, i].imshow(vol[i, :, H // 2, :], cmap=\"gray\", aspect=\"auto\")\n"
        "    axes[1, i].set_title(f\"{m} — coronal\")\n"
        "    axes[1, i].axis(\"off\")\n"
        "fig.suptitle(f\"Stage 1 preprocessing output — {subject}\")\n"
        "plt.tight_layout()\n"
        "plt.savefig(\"/kaggle/working/preprocessing_triptych.png\", dpi=120, bbox_inches=\"tight\")\n"
        "plt.show()\n"
    ),
    md(
        "## 5 · Synthetic report generated from the segmentation mask\n"
        "\n"
        "Demonstrates the templated VASARI feature extractor — same code that produces the synthetic warmup benchmark.\n"
    ),
    code(
        "from neuroval3d.data import SyntheticReportGenerator\n"
        "import nibabel as nib, numpy as np\n"
        "\n"
        "seg = np.asarray(nib.load(seg_path).dataobj, dtype=np.int16)\n"
        "synth = SyntheticReportGenerator().from_mask(seg, voxel_volume_mm3=1.0)\n"
        "print(\"=== Generated radiology report from the segmentation ===\")\n"
        "print(synth.text)\n"
        "print(\"\\n=== Extracted VASARI features ===\")\n"
        "for k, v in synth.vasari_features.items():\n"
        "    print(f\"  {k:25s}: {v}\")\n"
    ),
    md(
        "## 6 · The headline chart — NeuroVal-3D vs baselines\n"
    ),
    code(
        "import matplotlib.pyplot as plt\n"
        "import numpy as np\n"
        "\n"
        "datasets = [\"TextBraTS (n=369)\", \"RadGenome (n=1007)\"]\n"
        "ours = [\n"
        "    result_tb.auroc_overall.get(\"fusion\", 0.0),\n"
        "    result_rg.auroc_overall.get(\"fusion\", 0.0),\n"
        "]\n"
        "bioclin = [\n"
        "    result_tb.auroc_overall.get(\"semantic\", 0.0),\n"
        "    result_rg.auroc_overall.get(\"semantic\", 0.0),\n"
        "]\n"
        "ratescore = [\n"
        "    result_tb.auroc_overall.get(\"ratescore_lite (baseline)\", 0.0),\n"
        "    result_rg.auroc_overall.get(\"ratescore_lite (baseline)\", 0.0),\n"
        "]\n"
        "\n"
        "x = np.arange(len(datasets))\n"
        "width = 0.27\n"
        "fig, ax = plt.subplots(figsize=(11, 6))\n"
        "ax.bar(x - width, ours,      width, label=\"NeuroVal-3D fused (ours)\", color=\"#2a9d8f\")\n"
        "ax.bar(x,         bioclin,   width, label=\"BioClinicalBERT (off-the-shelf)\", color=\"#e9c46a\")\n"
        "ax.bar(x + width, ratescore, width, label=\"RaTEScore-lite (Jaccard baseline)\", color=\"#e76f51\")\n"
        "for i, v in enumerate(ours):\n"
        "    ax.text(x[i] - width, v + 0.02, f\"{v:.3f}\", ha=\"center\", fontsize=10, fontweight=\"bold\")\n"
        "for i, v in enumerate(bioclin):\n"
        "    ax.text(x[i], v + 0.02, f\"{v:.3f}\", ha=\"center\", fontsize=10)\n"
        "for i, v in enumerate(ratescore):\n"
        "    ax.text(x[i] + width, v + 0.02, f\"{v:.3f}\", ha=\"center\", fontsize=10)\n"
        "\n"
        "ax.axhline(0.5, ls=\":\", color=\"gray\", lw=1)\n"
        "ax.text(-0.45, 0.51, \"random\", color=\"gray\", fontsize=9)\n"
        "ax.set_ylabel(\"Held-out AUROC\")\n"
        "ax.set_title(\"Brain-MRI hallucination detection — NeuroVal-3D vs off-the-shelf baselines\")\n"
        "ax.set_xticks(x)\n"
        "ax.set_xticklabels(datasets)\n"
        "ax.legend(loc=\"lower right\")\n"
        "ax.set_ylim([0, 1.10])\n"
        "plt.tight_layout()\n"
        "plt.savefig(\"/kaggle/working/auroc_bar_chart.png\", dpi=120, bbox_inches=\"tight\")\n"
        "plt.show()\n"
    ),
    md(
        "## 7 · Where the artefacts landed\n"
        "\n"
        "Everything Kaggle keeps after this notebook commits ends up in `/kaggle/working/`. Two key files:\n"
        "\n"
        "- `preprocessing_triptych.png` — Stage 1 output for one BraTS volume\n"
        "- `auroc_bar_chart.png` — the headline chart\n"
        "\n"
        "Plus all per-run AUROC tables under `/kaggle/working/neuroval3d/outputs/results/`.\n"
        "\n"
        "## 8 · What's next\n"
        "\n"
        "- **Cross-dataset transfer** (TextBraTS → RadGenome and reverse) is the next paper-table entry. Run `python scripts/run_cross_only.py` after this notebook completes; it shares the loaded BERT model.\n"
        "- **Phase 2 proper**: train a small report decoder (BART-base) using the preprocessed volumes from Stage 1 above. Use the resulting generated reports as the test corpus for NeuroVal-3D — this closes the generator-validator loop.\n"
        "- **Negation axis improvement** (round 6) — the fusion negation AUROC is the weakest spot at 0.63 on RadGenome. Integrating `negspaCy` should lift it past 0.85.\n"
    ),
]


def main() -> None:
    NB_PATH.parent.mkdir(parents=True, exist_ok=True)
    nb = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
            "kaggle": {"accelerator": "gpu", "dataSources": [
                {"sourceType": "datasetVersion", "datasetId": 751906,
                 "sourceId": 1296470, "datasetVersionNumber": 2}
            ]},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NB_PATH.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"wrote {NB_PATH}")


if __name__ == "__main__":
    main()
