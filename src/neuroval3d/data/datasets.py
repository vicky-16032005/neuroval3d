"""Dataset registry — names, paths, modality lists, license info.

We do not commit any data. This module records *where data lives once downloaded*
and provides loaders. Real data lives under `data/raw/<dataset_name>/`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

DatasetSplit = Literal["train", "val", "test", "all"]
DataKind = Literal["paired_report", "segmentation_only", "caption", "unlabeled"]


@dataclass(frozen=True)
class BrainMRIDatasetSpec:
    name: str
    kind: DataKind
    modalities: tuple[str, ...]
    n_subjects: int | None
    license: str
    url: str
    notes: str = ""
    register_required: bool = False
    local_path: Path = field(default_factory=lambda: Path("data/raw"))


# The canonical registry.
DATASETS: dict[str, BrainMRIDatasetSpec] = {
    "TextBraTS": BrainMRIDatasetSpec(
        name="TextBraTS",
        kind="paired_report",
        modalities=("T1", "T1ce", "T2", "FLAIR"),
        n_subjects=369,
        license="MIT (reports); BraTS DUA (volumes)",
        url="https://github.com/Jupitern52/TextBraTS",
        notes="GPT-4o pseudo-reports refined by radiologists; volumes from BraTS 2020.",
        register_required=True,
    ),
    "RadGenome-BrainMRI": BrainMRIDatasetSpec(
        name="RadGenome-BrainMRI",
        kind="paired_report",
        modalities=("T1", "T2", "DWI", "FLAIR", "ADC", "T1ce"),
        n_subjects=1007,
        license="Research-only (see AutoRG-Brain repo)",
        url="https://github.com/ljy19970415/AutoRG-Brain",
        notes="3,408 image-report pairs; pixel-level grounding; 5 disease categories.",
        register_required=True,
    ),
    "BraTS2020": BrainMRIDatasetSpec(
        name="BraTS2020",
        kind="segmentation_only",
        modalities=("T1", "T1ce", "T2", "FLAIR"),
        n_subjects=369,
        license="BraTS 2020 DUA",
        url="https://www.med.upenn.edu/cbica/brats2020/",
        notes="Foundation segmentation dataset; required to load TextBraTS volumes.",
        register_required=True,
    ),
    "BraTS2021": BrainMRIDatasetSpec(
        name="BraTS2021",
        kind="segmentation_only",
        modalities=("T1", "T1ce", "T2", "FLAIR"),
        n_subjects=1470,
        license="BraTS 2021 DUA",
        url="https://www.synapse.org/#!Synapse:syn25829067",
        notes="Larger BraTS — used for synthetic-report generation.",
        register_required=True,
    ),
    "BraTS2023-AdultGlioma": BrainMRIDatasetSpec(
        name="BraTS2023-AdultGlioma",
        kind="segmentation_only",
        modalities=("T1", "T1ce", "T2", "FLAIR"),
        n_subjects=1251,
        license="BraTS 2023 DUA",
        url="https://www.synapse.org/#!Synapse:syn51156910",
        notes="Adult glioma 2023.",
        register_required=True,
    ),
    "ISLES2022": BrainMRIDatasetSpec(
        name="ISLES2022",
        kind="segmentation_only",
        modalities=("DWI", "ADC", "FLAIR"),
        n_subjects=250,
        license="ISLES Challenge",
        url="https://isles24.grand-challenge.org/",
        notes="Acute stroke; for multi-pathology extension.",
        register_required=True,
    ),
    "ATLAS-v2.0": BrainMRIDatasetSpec(
        name="ATLAS-v2.0",
        kind="segmentation_only",
        modalities=("T1",),
        n_subjects=304,
        license="CC-BY",
        url="https://atlas.grand-challenge.org/",
        notes="Post-stroke T1 lesion tracings.",
    ),
    "UPENN-GBM": BrainMRIDatasetSpec(
        name="UPENN-GBM",
        kind="segmentation_only",
        modalities=("T1", "T1ce", "T2", "FLAIR"),
        n_subjects=630,
        license="CC-BY",
        url="https://www.cancerimagingarchive.net/collection/upenn-gbm/",
        notes="GBM with structured molecular fields → synthetic-report templates.",
    ),
    "ROCOv2-brain": BrainMRIDatasetSpec(
        name="ROCOv2-brain",
        kind="caption",
        modalities=("various",),
        n_subjects=None,
        license="CC-BY",
        url="https://huggingface.co/datasets/eltorio/ROCOv2-radiology",
        notes="Filter ROCOv2 by modality=MRI + region=brain UMLS concepts.",
    ),
    "IXI": BrainMRIDatasetSpec(
        name="IXI",
        kind="unlabeled",
        modalities=("T1", "T2", "PD", "MRA", "DTI"),
        n_subjects=600,
        license="CC-BY-SA",
        url="https://brain-development.org/ixi-dataset/",
        notes="Healthy-subject pretraining.",
    ),
}


class DatasetRegistry:
    """Thin lookup + path resolution helper."""

    def __init__(self, root: str | Path = "data/raw") -> None:
        self.root = Path(root)

    def get(self, name: str) -> BrainMRIDatasetSpec:
        if name not in DATASETS:
            raise KeyError(f"Unknown dataset: {name}. Known: {sorted(DATASETS)}")
        spec = DATASETS[name]
        return BrainMRIDatasetSpec(
            **{**spec.__dict__, "local_path": self.root / spec.name}
        )

    def status(self, name: str) -> dict[str, object]:
        spec = self.get(name)
        present = spec.local_path.exists() and any(spec.local_path.iterdir())
        return {
            "name": spec.name,
            "expected_path": str(spec.local_path),
            "present": present,
            "license": spec.license,
            "register_required": spec.register_required,
        }

    def list_present(self) -> list[str]:
        return [n for n in DATASETS if self.status(n)["present"]]
