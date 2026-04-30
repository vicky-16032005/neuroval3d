# Datasets — How to Obtain

> None of the data lives in this repo. Everything below is registration / download instructions.

## Primary paired-report datasets

### TextBraTS (PRIMARY)
- 369 BraTS 2020 multi-modal volumes paired with GPT-4o pseudo-reports refined by radiologists
- License: MIT (the report layer); BraTS volumes have their own DUA
- Reports: https://github.com/Jupitern52/TextBraTS, https://huggingface.co/datasets/Jupitern52/TextBraTS
- Volumes: register at https://www.synapse.org/ and download BraTS 2020 training set
- Place at `data/raw/TextBraTS/`

### RadGenome-Brain MRI
- 1,007 cases / 3,408 imaging-report pairs
- 5 disease categories, pixel-level grounding
- Source: https://github.com/ljy19970415/AutoRG-Brain (release info in repo)
- Place at `data/raw/RadGenome-BrainMRI/`

## Segmentation-only sources (for synthetic reports)

### BraTS (entire suite)
- BraTS 2020 (369), 2021 (1,470), 2023 adult glioma (~1,250), 2024 post-treatment glioma + sub-challenges
- Register: https://www.synapse.org/ and https://www.med.upenn.edu/cbica/brats2024/
- Place at `data/raw/BraTS/{2020,2021,2023,2024}/`

### ISLES 2022 / 2024 (stroke)
- 250 longitudinal acute stroke cases with multimodal CT + follow-up MRI
- https://isles24.grand-challenge.org/

### ATLAS v2.0 (post-stroke MRI)
- 304 T1-weighted with manual lesion tracings
- https://atlas.grand-challenge.org/

### UPENN-GBM (TCIA)
- 630 GBM patients, mpMRI + segmentations + clinical metadata
- No native reports, but rich VASARI-templatable structured fields
- https://www.cancerimagingarchive.net/collection/upenn-gbm/

## Caption-style proxies (smaller, easier)

### ROCOv2 — brain MRI subset
- Filter the full ROCOv2 by modality/MRI + region/brain UMLS concepts
- https://huggingface.co/datasets/eltorio/ROCOv2-radiology

## Atlases (committed-into-source-of-truth)

- **SRI24** — BraTS standard. Used for inter-subject registration.
- **MNI152** — used for anatomical anchoring.
- **AAL v3** — 116 anatomical region labels in MNI152 space.

These are downloaded by `scripts/download_atlases.py` on first use into `data/raw/atlases/`.

## Self-supervised pretraining (optional)

- **IXI** — 600 healthy subjects, T1/T2/PD/MRA/DTI. Free.
- **OASIS-3** — 1,098 participants, longitudinal. Free with NITRC registration.
- **ADNI** — gated, multi-week application. https://adni.loni.usc.edu/

We do **not** depend on UK Biobank — instead we use the public BrainSegFounder pretrained weights (https://github.com/lab-smile/BrainSegFounder).

## Compliance

- Cite all source DUAs in the paper.
- Never redistribute downloaded volumes.
- Use de-identified data only.
- Add the "not for clinical use" disclaimer to all artifacts.
