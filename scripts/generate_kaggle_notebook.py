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
        "import sys, torch, importlib\n"
        "print(f\"python: {sys.version.split()[0]}\")\n"
        "print(f\"torch:  {torch.__version__}, cuda available: {torch.cuda.is_available()}\")\n"
        "if torch.cuda.is_available():\n"
        "    print(f\"  device: {torch.cuda.get_device_name(0)}, vram: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB\")\n"
        "\n"
        "# Some Kaggle environments resolve `neuroval3d` to an empty namespace package despite\n"
        "# the editable install succeeding. Force-prepend our src/ to sys.path and drop any\n"
        "# stale module cache before the first real import.\n"
        "_PROJECT_SRC = \"/kaggle/working/neuroval3d/src\"\n"
        "if _PROJECT_SRC not in sys.path:\n"
        "    sys.path.insert(0, _PROJECT_SRC)\n"
        "for _m in list(sys.modules):\n"
        "    if _m == \"neuroval3d\" or _m.startswith(\"neuroval3d.\"):\n"
        "        del sys.modules[_m]\n"
        "\n"
        "import neuroval3d\n"
        "print(f\"neuroval3d: v{getattr(neuroval3d, '__version__', '?')} from {neuroval3d.__file__}\")\n"
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
        "# The awsaf49 BraTS 2020 mirror normally mounts as\n"
        "#   /kaggle/input/brats20-dataset-training-validation/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/\n"
        "# but if Kaggle ever re-nests it slightly we still want to find the right subjects dir.\n"
        "def _find_brats_training_root() -> Path | None:\n"
        "    base = Path(\"/kaggle/input\")\n"
        "    if not base.exists():\n"
        "        return None\n"
        "    # Preferred: directory literally called MICCAI_BraTS2020_TrainingData\n"
        "    for p in base.rglob(\"MICCAI_BraTS2020_TrainingData\"):\n"
        "        if p.is_dir():\n"
        "            return p\n"
        "    # Fallback: walk up from any BraTS20_Training_NNN subject dir\n"
        "    for p in base.rglob(\"BraTS20_Training_001\"):\n"
        "        if p.is_dir():\n"
        "            return p.parent\n"
        "    return None\n"
        "\n"
        "BRATS_ROOT = _find_brats_training_root()\n"
        "if BRATS_ROOT is None:\n"
        "    raise SystemExit(\n"
        "        \"BraTS dataset not found under /kaggle/input/. \"\n"
        "        \"In Kaggle right-rail: Add Data \\u2192 \"\n"
        "        \"https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation\"\n"
        "    )\n"
        "print(f\"BraTS training root: {BRATS_ROOT}\")\n"
        "n_subjects = len(list(BRATS_ROOT.glob(\"BraTS20_Training_*\")))\n"
        "print(f\"Subject directories found: {n_subjects}\")\n"
        "\n"
        "subject_id = \"BraTS20_Training_001\"\n"
        "subject_dir = BRATS_ROOT / subject_id\n"
        "\n"
        "# Files might be .nii or .nii.gz depending on the mirror; detect either.\n"
        "def _modality_path(modality: str) -> str:\n"
        "    for ext in (\".nii\", \".nii.gz\"):\n"
        "        cand = subject_dir / f\"{subject_id}_{modality}{ext}\"\n"
        "        if cand.exists():\n"
        "            return str(cand)\n"
        "    raise FileNotFoundError(\n"
        "        f\"Missing {subject_id} {modality} under {subject_dir}; \"\n"
        "        f\"contents: {sorted(p.name for p in subject_dir.iterdir())[:8] if subject_dir.exists() else 'dir missing'}\"\n"
        "    )\n"
        "\n"
        "paths = {\n"
        "    \"T1\":    _modality_path(\"t1\"),\n"
        "    \"T1ce\":  _modality_path(\"t1ce\"),\n"
        "    \"T2\":    _modality_path(\"t2\"),\n"
        "    \"FLAIR\": _modality_path(\"flair\"),\n"
        "}\n"
        "seg_path = _modality_path(\"seg\")\n"
        "print(\"Resolved modality paths:\")\n"
        "for k, v in paths.items():\n"
        "    print(f\"  {k:5s}: {v}\")\n"
        "\n"
        "pre = Stage1Preprocessor(PreprocessingConfig(\n"
        "    target_shape=(128, 128, 128),\n"
        "    bias_correct=True,\n"
        "    normalize=\"zscore\",\n"
        "))\n"
        "out = pre.run(paths, seg_path=seg_path)\n"
        "\n"
        "print(f\"\\nPreprocessed volume shape: {out.volume.shape}\")\n"
        "print(f\"Brain-mask voxel count:    {int(out.brain_mask.sum()):,}\")\n"
        "print(\"Per-modality stats (pre-zscore):\")\n"
        "for m, s in out.modality_stats.items():\n"
        "    print(f\"  {m}: mean={s['mean']:.2f}, std={s['std']:.2f}\")\n"
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
        "## 7 · Diagnostic plots — ROC, PR, confusion matrix, score distributions, per-op heatmap\n"
        "\n"
        "Loads the per-record scores (saved by `run_benchmark`) and plots the standard set of binary-classification diagnostics for both datasets.\n"
    ),
    code(
        "# Load per-record scores from both held-out runs into pandas DataFrames\n"
        "import json\n"
        "from pathlib import Path\n"
        "import pandas as pd\n"
        "\n"
        "def load_scores(run_id: str) -> pd.DataFrame:\n"
        "    p = Path(f\"/kaggle/working/neuroval3d/outputs/results/{run_id}/scores.jsonl\")\n"
        "    rows = [json.loads(l) for l in open(p, encoding=\"utf-8\") if l.strip()]\n"
        "    return pd.DataFrame(rows)\n"
        "\n"
        "df_tb = load_scores(result_tb.run_id)\n"
        "df_rg = load_scores(result_rg.run_id)\n"
        "for name, df in [(\"TextBraTS\", df_tb), (\"RadGenome\", df_rg)]:\n"
        "    splits = df[\"split\"].value_counts().to_dict()\n"
        "    pos = (df[df.split == \"test\"][\"label\"] == 1).sum()\n"
        "    neg = (df[df.split == \"test\"][\"label\"] == 0).sum()\n"
        "    print(f\"{name}: {len(df)} total, splits={splits}, test positives={pos}, test negatives={neg}\")\n"
    ),
    md(
        "### 7.1 ROC curves — fusion vs each axis vs baselines (held-out test split)\n"
    ),
    code(
        "import matplotlib.pyplot as plt\n"
        "from sklearn.metrics import roc_curve, auc\n"
        "\n"
        "VALIDATOR_STYLE = [\n"
        "    (\"fused\",          \"NeuroVal-3D fused (ours)\",       \"#2a9d8f\", 3.0),\n"
        "    (\"structural\",     \"structural\",                      \"#264653\", 1.5),\n"
        "    (\"lexical\",        \"lexical\",                         \"#287271\", 1.5),\n"
        "    (\"numeric\",        \"numeric\",                         \"#76b39d\", 1.0),\n"
        "    (\"modality\",       \"modality\",                        \"#a8dadc\", 1.0),\n"
        "    (\"negation\",       \"negation\",                        \"#7f4f7b\", 1.0),\n"
        "    (\"lesion_type\",    \"lesion_type\",                     \"#bb9bb0\", 1.0),\n"
        "    (\"semantic\",       \"semantic (BioClinicalBERT)\",      \"#e9c46a\", 1.5),\n"
        "    (\"ratescore_lite\", \"RaTEScore-lite (Jaccard baseline)\", \"#e76f51\", 1.5),\n"
        "]\n"
        "\n"
        "def plot_roc(df: pd.DataFrame, title: str, ax) -> None:\n"
        "    test = df[df.split == \"test\"]\n"
        "    if len(test) == 0:\n"
        "        ax.text(0.5, 0.5, \"no test rows\", ha=\"center\", transform=ax.transAxes); return\n"
        "    for col, label, color, lw in VALIDATOR_STYLE:\n"
        "        if col not in test.columns:\n"
        "            continue\n"
        "        fpr, tpr, _ = roc_curve(test[\"label\"], test[col])\n"
        "        ax.plot(fpr, tpr, label=f\"{label}  (AUC={auc(fpr, tpr):.3f})\", color=color, linewidth=lw)\n"
        "    ax.plot([0, 1], [0, 1], \"--\", color=\"gray\", alpha=0.5, label=\"random (AUC=0.50)\")\n"
        "    ax.set_xlabel(\"False Positive Rate\")\n"
        "    ax.set_ylabel(\"True Positive Rate\")\n"
        "    ax.set_title(title)\n"
        "    ax.legend(loc=\"lower right\", fontsize=8)\n"
        "    ax.grid(alpha=0.3)\n"
        "\n"
        "fig, axes = plt.subplots(1, 2, figsize=(15, 6))\n"
        "plot_roc(df_tb, \"ROC — TextBraTS held-out test\", axes[0])\n"
        "plot_roc(df_rg, \"ROC — RadGenome-Brain MRI held-out test\", axes[1])\n"
        "plt.tight_layout()\n"
        "plt.savefig(\"/kaggle/working/roc_curves.png\", dpi=140, bbox_inches=\"tight\")\n"
        "plt.show()\n"
    ),
    md(
        "### 7.2 Precision-Recall curves\n"
        "\n"
        "Better than ROC when the classes are imbalanced (we have ~5× more perturbed records than clean ones).\n"
    ),
    code(
        "from sklearn.metrics import precision_recall_curve, average_precision_score\n"
        "\n"
        "def plot_pr(df: pd.DataFrame, title: str, ax) -> None:\n"
        "    test = df[df.split == \"test\"]\n"
        "    if len(test) == 0:\n"
        "        ax.text(0.5, 0.5, \"no test rows\", ha=\"center\", transform=ax.transAxes); return\n"
        "    base_rate = test[\"label\"].mean()\n"
        "    for col, label, color, lw in VALIDATOR_STYLE:\n"
        "        if col not in test.columns:\n"
        "            continue\n"
        "        precision, recall, _ = precision_recall_curve(test[\"label\"], test[col])\n"
        "        ap = average_precision_score(test[\"label\"], test[col])\n"
        "        ax.plot(recall, precision, label=f\"{label}  (AP={ap:.3f})\", color=color, linewidth=lw)\n"
        "    ax.axhline(base_rate, ls=\"--\", color=\"gray\", alpha=0.5, label=f\"random (AP={base_rate:.2f})\")\n"
        "    ax.set_xlabel(\"Recall\")\n"
        "    ax.set_ylabel(\"Precision\")\n"
        "    ax.set_title(title)\n"
        "    ax.legend(loc=\"lower left\", fontsize=8)\n"
        "    ax.grid(alpha=0.3)\n"
        "    ax.set_ylim([0, 1.02])\n"
        "\n"
        "fig, axes = plt.subplots(1, 2, figsize=(15, 6))\n"
        "plot_pr(df_tb, \"Precision-Recall — TextBraTS held-out test\", axes[0])\n"
        "plot_pr(df_rg, \"Precision-Recall — RadGenome held-out test\", axes[1])\n"
        "plt.tight_layout()\n"
        "plt.savefig(\"/kaggle/working/pr_curves.png\", dpi=140, bbox_inches=\"tight\")\n"
        "plt.show()\n"
    ),
    md(
        "### 7.3 Confusion matrices @ threshold 0.5\n"
        "\n"
        "How the fusion validator labels each held-out test record at the default 0.5 threshold. Includes accuracy, precision, recall, F1.\n"
    ),
    code(
        "from sklearn.metrics import confusion_matrix\n"
        "\n"
        "def plot_cm(df: pd.DataFrame, title: str, ax, threshold: float = 0.5) -> None:\n"
        "    test = df[df.split == \"test\"]\n"
        "    if len(test) == 0:\n"
        "        ax.text(0.5, 0.5, \"no test rows\", ha=\"center\", transform=ax.transAxes); return\n"
        "    y_true = test[\"label\"].values\n"
        "    y_pred = (test[\"fused\"] >= threshold).astype(int)\n"
        "    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])\n"
        "    tn, fp, fn, tp = cm.ravel()\n"
        "    acc = (tp + tn) / max(tp + tn + fp + fn, 1)\n"
        "    prec = tp / max(tp + fp, 1)\n"
        "    rec = tp / max(tp + fn, 1)\n"
        "    f1 = 2 * prec * rec / max(prec + rec, 1e-9)\n"
        "\n"
        "    im = ax.imshow(cm, cmap=\"Blues\")\n"
        "    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])\n"
        "    ax.set_xticklabels([\"Predicted\\nFLAGGED\", \"Predicted\\nVALID\"])\n"
        "    ax.set_yticklabels([\"Actually\\nhallucinated\", \"Actually\\nclean\"])\n"
        "    for i in range(2):\n"
        "        for j in range(2):\n"
        "            color = \"white\" if cm[i, j] > cm.max() * 0.5 else \"black\"\n"
        "            ax.text(j, i, str(cm[i, j]), ha=\"center\", va=\"center\",\n"
        "                    color=color, fontsize=18, fontweight=\"bold\")\n"
        "    ax.set_title(f\"{title}\\nAcc={acc:.3f} · Prec={prec:.3f} · Rec={rec:.3f} · F1={f1:.3f}\")\n"
        "\n"
        "fig, axes = plt.subplots(1, 2, figsize=(13, 6))\n"
        "plot_cm(df_tb, \"Confusion Matrix — TextBraTS\", axes[0])\n"
        "plot_cm(df_rg, \"Confusion Matrix — RadGenome\", axes[1])\n"
        "plt.tight_layout()\n"
        "plt.savefig(\"/kaggle/working/confusion_matrices.png\", dpi=140, bbox_inches=\"tight\")\n"
        "plt.show()\n"
    ),
    md(
        "### 7.4 Score distribution (clean vs hallucinated, held-out test)\n"
        "\n"
        "Visualises separability — the further apart the two histograms, the easier the validator's job.\n"
    ),
    code(
        "import numpy as np\n"
        "\n"
        "def plot_dist(df: pd.DataFrame, title: str, ax) -> None:\n"
        "    test = df[df.split == \"test\"]\n"
        "    if len(test) == 0:\n"
        "        ax.text(0.5, 0.5, \"no test rows\", ha=\"center\", transform=ax.transAxes); return\n"
        "    bins = np.linspace(0, 1, 41)\n"
        "    pos = test[test.label == 1][\"fused\"]\n"
        "    neg = test[test.label == 0][\"fused\"]\n"
        "    ax.hist(neg, bins=bins, alpha=0.6, color=\"#e76f51\", label=f\"hallucinated (n={len(neg)})\", density=True)\n"
        "    ax.hist(pos, bins=bins, alpha=0.6, color=\"#2a9d8f\", label=f\"clean (n={len(pos)})\", density=True)\n"
        "    ax.axvline(0.5, ls=\"--\", color=\"gray\", lw=1, label=\"threshold = 0.5\")\n"
        "    ax.set_xlabel(\"NeuroVal-3D fused score\")\n"
        "    ax.set_ylabel(\"Density\")\n"
        "    ax.set_title(title)\n"
        "    ax.legend(loc=\"upper center\")\n"
        "    ax.grid(alpha=0.3)\n"
        "\n"
        "fig, axes = plt.subplots(1, 2, figsize=(15, 5))\n"
        "plot_dist(df_tb, \"Score distribution — TextBraTS test\", axes[0])\n"
        "plot_dist(df_rg, \"Score distribution — RadGenome test\", axes[1])\n"
        "plt.tight_layout()\n"
        "plt.savefig(\"/kaggle/working/score_distributions.png\", dpi=140, bbox_inches=\"tight\")\n"
        "plt.show()\n"
    ),
    md(
        "### 7.5 Per-op AUROC heatmap (validator × perturbation type)\n"
        "\n"
        "Reads the test-set AUROC for every (validator, perturbation operation) combination. Tells you which validator catches which error class. Each axis should be near-perfect on its specialty.\n"
    ),
    code(
        "VALIDATOR_ORDER = [\n"
        "    \"fusion\", \"structural\", \"lexical\", \"numeric\", \"modality\",\n"
        "    \"negation\", \"lesion_type\", \"semantic\", \"ratescore_lite (baseline)\",\n"
        "]\n"
        "\n"
        "def per_op_heatmap(result, title: str, ax):\n"
        "    by_op = result.auroc_by_op\n"
        "    ops = sorted(by_op.keys())\n"
        "    M = np.full((len(VALIDATOR_ORDER), len(ops)), np.nan)\n"
        "    for i, v in enumerate(VALIDATOR_ORDER):\n"
        "        for j, op in enumerate(ops):\n"
        "            if v in by_op[op]:\n"
        "                M[i, j] = by_op[op][v]\n"
        "    im = ax.imshow(M, cmap=\"RdYlGn\", vmin=0.0, vmax=1.0, aspect=\"auto\")\n"
        "    ax.set_xticks(range(len(ops)))\n"
        "    ax.set_xticklabels(ops, rotation=30, ha=\"right\")\n"
        "    ax.set_yticks(range(len(VALIDATOR_ORDER)))\n"
        "    ax.set_yticklabels(VALIDATOR_ORDER)\n"
        "    for i in range(len(VALIDATOR_ORDER)):\n"
        "        for j in range(len(ops)):\n"
        "            v = M[i, j]\n"
        "            if not np.isnan(v):\n"
        "                color = \"white\" if (v < 0.35 or v > 0.85) else \"black\"\n"
        "                ax.text(j, i, f\"{v:.2f}\", ha=\"center\", va=\"center\", fontsize=9, color=color)\n"
        "    ax.set_title(title)\n"
        "    return im\n"
        "\n"
        "fig, axes = plt.subplots(1, 2, figsize=(16, 7))\n"
        "im_a = per_op_heatmap(result_tb, \"Per-op AUROC — TextBraTS\", axes[0])\n"
        "im_b = per_op_heatmap(result_rg, \"Per-op AUROC — RadGenome\", axes[1])\n"
        "fig.colorbar(im_b, ax=axes, location=\"right\", shrink=0.7, label=\"AUROC\")\n"
        "plt.savefig(\"/kaggle/working/per_op_heatmap.png\", dpi=140, bbox_inches=\"tight\")\n"
        "plt.show()\n"
    ),
    md(
        "### 7.6 Train vs Test AUROC — the analog of train/test loss\n"
        "\n"
        "**Why no `loss curves`?** The fusion validator is a single sklearn `LogisticRegression` fit — closed-form on a small sample, not an iterative gradient-descent loop. There's no per-epoch loss to plot. The right diagnostic for overfit detection here is **gap between train AUROC and held-out test AUROC** — if the fusion is memorising, train AUROC ≫ test AUROC. We want them close.\n"
        "\n"
        "When we train the BART or Llama-3 decoder in Phase 2, that's where proper per-epoch loss curves show up.\n"
    ),
    code(
        "def plot_train_vs_test(result, title: str, ax) -> None:\n"
        "    train = [result.auroc_train.get(v, np.nan) for v in VALIDATOR_ORDER]\n"
        "    test  = [result.auroc_overall.get(v, np.nan) for v in VALIDATOR_ORDER]\n"
        "    x = np.arange(len(VALIDATOR_ORDER))\n"
        "    width = 0.38\n"
        "    ax.bar(x - width/2, train, width, label=\"Train AUROC\", color=\"#264653\")\n"
        "    ax.bar(x + width/2, test,  width, label=\"Held-out Test AUROC\", color=\"#2a9d8f\")\n"
        "    ax.axhline(0.5, ls=\":\", color=\"gray\", alpha=0.7)\n"
        "    ax.set_xticks(x)\n"
        "    ax.set_xticklabels(VALIDATOR_ORDER, rotation=30, ha=\"right\")\n"
        "    ax.set_ylabel(\"AUROC\")\n"
        "    ax.set_title(title)\n"
        "    ax.set_ylim([0, 1.05])\n"
        "    ax.legend()\n"
        "    ax.grid(alpha=0.3, axis=\"y\")\n"
        "\n"
        "fig, axes = plt.subplots(1, 2, figsize=(16, 6))\n"
        "plot_train_vs_test(result_tb, \"Train vs Test AUROC — TextBraTS\", axes[0])\n"
        "plot_train_vs_test(result_rg, \"Train vs Test AUROC — RadGenome\", axes[1])\n"
        "plt.tight_layout()\n"
        "plt.savefig(\"/kaggle/working/train_vs_test.png\", dpi=140, bbox_inches=\"tight\")\n"
        "plt.show()\n"
        "\n"
        "# Print numeric gaps for the paper\n"
        "print(\"\\nFusion train→test gap (should be near zero — confirms no overfit):\")\n"
        "for name, r in [(\"TextBraTS\", result_tb), (\"RadGenome\", result_rg)]:\n"
        "    gap = r.auroc_train.get(\"fusion\", float(\"nan\")) - r.auroc_overall.get(\"fusion\", float(\"nan\"))\n"
        "    print(f\"  {name:12s}: train={r.auroc_train.get('fusion', float('nan')):.4f}, test={r.auroc_overall.get('fusion', float('nan')):.4f}, gap={gap:+.4f}\")\n"
    ),
    md(
        "## 8 · Where the artefacts landed\n"
        "\n"
        "All saved under `/kaggle/working/`:\n"
        "\n"
        "- `preprocessing_triptych.png` — Stage 1 output for one BraTS volume\n"
        "- `auroc_bar_chart.png` — headline NeuroVal-3D vs baselines\n"
        "- `roc_curves.png` — ROC for fusion + each axis on both datasets\n"
        "- `pr_curves.png` — Precision-Recall curves\n"
        "- `confusion_matrices.png` — at threshold 0.5 with Acc/Prec/Rec/F1\n"
        "- `score_distributions.png` — clean vs hallucinated histograms\n"
        "- `per_op_heatmap.png` — validator × perturbation-op heatmap\n"
        "- `train_vs_test.png` — train/test AUROC bars (no overfit check)\n"
        "\n"
        "Per-run AUROC tables + raw scores under `/kaggle/working/neuroval3d/outputs/results/`.\n"
        "\n"
        "## 9 · What's next\n"
        "\n"
        "- **Cross-dataset transfer** (TextBraTS → RadGenome and reverse) — run `python scripts/run_cross_only.py` for the transfer numbers (~2 hr more).\n"
        "- **Phase 2 proper**: train a small report decoder (BART-base) using the preprocessed volumes from Stage 1 above. Use the resulting generated reports as the test corpus for NeuroVal-3D — closes the generator-validator loop. **This is where real per-epoch train/test loss curves live.**\n"
        "- **Negation axis improvement** (round 6) — fusion negation AUROC is the weakest spot at 0.63 on RadGenome. Integrate `negspaCy` to lift it past 0.85.\n"
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
