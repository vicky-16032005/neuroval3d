# NeuroVal-3D Architecture

## Stage interface contract

Every stage exposes a single class with the same shape:

```python
class StageX:
    """Stateful or stateless transform with a deterministic .run()."""

    def __init__(self, config: dict): ...

    def run(self, inputs: StageXInputs) -> StageXOutputs: ...

    @classmethod
    def from_config(cls, path: str | Path) -> "StageX": ...
```

This means each stage can be swapped, mocked, or invoked standalone for testing.

## Stages 1–4 (the pipeline)

### Stage 1 — Preprocessing
**In:** path to a 4-modality MRI volume (NIfTI; T1, T1ce, T2, FLAIR)
**Out:** registered, normalized 5D tensor `[1, 4, D, H, W]` plus optional segmentation mask
**Module:** `src.neuroval3d.data.preprocessing`
**Deps:** SimpleITK (N4), nibabel (IO), MONAI (transforms), HD-BET (skull strip; lazy import)

### Stage 2 — 3D Encoder
**In:** preprocessed volume tensor
**Out:** dense feature map `[B, T, D]` where `D` is the hidden dim
**Module:** `src.neuroval3d.models.encoder`
**Default backbone:** Swin-UNETR via MONAI

### Stage 3 — Multimodal Projector
**In:** image features + (optionally) text embeddings
**Out:** aligned tokens in the decoder's input space
**Module:** `src.neuroval3d.models.projector`
**Default:** MLP projector + LoRA adapter on the decoder's first cross-attention layer

### Stage 4 — Decoder
**In:** projected tokens
**Out:** generated report text
**Module:** `src.neuroval3d.models.decoder`
**Default:** BART-base. Swappable to T5-small / Llama-3.2-3B-QLoRA / M3D-LaMed via config.

## Stages 5–6 (the validator — THE CONTRIBUTION)

### Stage 5a — Semantic Validator
**In:** generated report text + reference text (or retrieved nearest neighbor)
**Out:** cosine similarity ∈ [0, 1]
**Module:** `src.neuroval3d.validators.semantic`
**Encoder:** BioClinicalBERT (default) or RadBERT or BiomedCLIP-text

### Stage 5b — Lexical Validator (VASARI-restricted TF-IDF)
**In:** same
**Out:** weighted n-gram cosine over a VASARI + UMLS-brain vocabulary, with negation-aware penalty
**Module:** `src.neuroval3d.validators.lexical`

### Stage 5c — Structural Validator (segmentation-grounded VASARI F1)
**In:** generated report + segmentation mask (or its derived VASARI feature vector)
**Out:** feature-level F1 ∈ [0, 1]
**Module:** `src.neuroval3d.validators.structural`

### Stage 5d — Fusion
**In:** the three sub-scores above
**Out:** a single calibrated probability "report is valid" + per-axis explanation
**Module:** `src.neuroval3d.validators.fusion`
**Default:** sklearn `LogisticRegression` calibrated on the perturbation training split

### Stage 6 — Anatomical Anchoring
**In:** generated report + tumor centroid (from segmentation)
**Out:** annotated report `[(sentence, region_label, (x, y, z), confidence)]`
**Module:** `src.neuroval3d.grounding.anatomy`
**Atlases:** AAL v3 in MNI152 space

## Stages 7–8 (supporting evaluation)

### Stage 7 — Explainability
**In:** model + image
**Out:** 3D Grad-CAM volume + 2D coronal/axial/sagittal overlays
**Module:** `src.neuroval3d.viz.gradcam`

### Stage 8 — Perturbation Benchmark
**In:** clean report + (optional) modality
**Out:** N perturbed variants with labeled error types
**Module:** `src.neuroval3d.evaluation.perturbation`

This stage produces the gold dataset for evaluating Stage 5 — that's the loop that closes the contribution.

## Data flow at inference

```
[volume.nii.gz, mask.nii.gz?]
        ↓ Stage 1
   [B, 4, 128, 128, 128]
        ↓ Stage 2
   [B, T, 768]
        ↓ Stage 3
   [B, T, 4096]   (projected to decoder space)
        ↓ Stage 4
   "There is a left frontal lobe enhancing mass..."  (the generated report)
        ↓ Stage 5 (the validator)
   {semantic: 0.81, lexical: 0.74, structural: 0.69, fused: 0.86 → VALID}
        ↓ Stage 6
   [(sentence, "Frontal_L", (x,y,z), 0.92), ...]
        ↓ Stage 7
   gradcam_overlay.png + axial/coronal/sagittal triptych
```

## Data flow at evaluation

```
TextBraTS / RadGenome-Brain MRI ground-truth reports
        ↓ Stage 8
   500 perturbed reports labeled by error type
        ↓ Stage 5 (semantic + lexical + structural)
   500 score triples
        ↓ AUROC per error type
   Compare vs RaTEScore / BERTScore / F1RadGraph / GREEN
```

The headline number is the AUROC table from this evaluation flow.
