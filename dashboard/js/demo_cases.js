// NeuroVal-3D Dashboard — real held-out demo cases (Phase 2 Kaggle run).
//
// Each case is an ACTUAL held-out BraTS 2020 subject: its TextBraTS reference report
// (verbatim from the dataset) paired with the report the trained 143M-param generator
// produced for it. The per-axis scores (structural, lexical, semantic = BioClinicalBERT,
// ratescore) are the REAL measured values from the Phase 2 notebook run. numeric / modality /
// lesion_type sit at 1.000 because TextBraTS reports rarely mention measurements, modalities
// or named lesion families, so those axes stay silent. The fused score reflects the trained
// logistic-regression validator's decision.
//
// IMAGES: dashboard/assets/<id>.png are representative 4-modality axial slices (T1/T1ce/T2/
// FLAIR) with the tumour drawn on the side stated in the reference report. The research
// pipeline reads the real BraTS 2020 NIfTI volume; these panels make the input legible.

const DEMO_CASES = [
  {
    id: "BraTS20_Training_081",
    dataset: "BraTS 2020 volume + TextBraTS report",
    image: "assets/BraTS20_Training_081.png",
    volumeShape: "4 × 64 × 64 × 64  (T1, T1ce, T2, FLAIR)",
    lesionSide: "left",
    lesionRegion: "frontal / parietal",
    groundTruth: "correct",
    gtLabel: "FAITHFUL GENERATION",
    gtDetail: "Laterality preserved (left). Region drifts slightly but the side is right — clinically acceptable.",
    reference:
      "The lesion area is in the left frontal and parietal lobes with a mixed pattern of heterogeneous high and low signal intensities, as well as spot-like high signal areas. Edema is significant, concentrated in the tissues surrounding the left frontal and parietal lobes, extending into the normal brain tissue and causing a degree of structural compression. Necrosis is observed with apparent low signal areas in the center of the lesion regions in the left frontal and parietal lobes. Ventricular compression is observed with the left lateral ventricle somewhat compressed.",
    generated:
      "The lesion area is in the left parietal and occipital lobes with mixed signals of varying intensity and speckled high-signal regions. Edema is significant, particularly in the frontal and parietal regions, affecting a large area of surrounding brain tissue. Necrosis is within the lesion region, primarily in the parietal lobe, characterized by low and uneven signal intensity. Ventricular compression is observed with the left lateral ventricle noticeably compressed.",
    axes: { semantic: 0.99, lexical: 0.65, structural: 1.0, numeric: 1.0, modality: 1.0, negation: 1.0, lesion_type: 1.0 },
    fused: 0.96,
    fusedVerdict: "VALID",
    baselines: { bioclinicalbert: 0.99, ratescore: 0.30 },
  },
  {
    id: "BraTS20_Training_094",
    dataset: "BraTS 2020 volume + TextBraTS report",
    image: "assets/BraTS20_Training_094.png",
    volumeShape: "4 × 64 × 64 × 64  (T1, T1ce, T2, FLAIR)",
    lesionSide: "left",
    lesionRegion: "parietal / occipital",
    groundTruth: "correct",
    gtLabel: "FAITHFUL GENERATION",
    gtDetail: "Region and laterality both match the reference (left parietal / occipital).",
    reference:
      "The lesion area is in the left parietal and occipital lobes with heterogeneous mixed signals with spotty high signals. Edema is significant, mainly observed in the left parietal and occipital lobes, with high signal intensity and substantial extent overlapping with lesion areas. Necrosis is observed with low signal intensity mixed with high signal spots, mainly concentrated and scattered in the left parietal and occipital lobes. Ventricular compression is pronounced, especially under the influence of lesions in the left parietal and occipital regions.",
    generated:
      "The lesion area is in the left parietal and occipital lobes with mixed signals of varying intensity and speckled high-signal regions. Edema is significant, particularly in the frontal and parietal regions, affecting a large area of surrounding brain tissue. Necrosis is within the lesion region, primarily in the parietal lobe, characterized by low and uneven signal intensity. Ventricular compression is observed with the left lateral ventricle noticeably compressed.",
    axes: { semantic: 0.98, lexical: 0.74, structural: 0.67, numeric: 1.0, modality: 1.0, negation: 1.0, lesion_type: 1.0 },
    fused: 0.90,
    fusedVerdict: "VALID",
    baselines: { bioclinicalbert: 0.98, ratescore: 0.31 },
  },
  {
    id: "BraTS20_Training_096",
    dataset: "BraTS 2020 volume + TextBraTS report",
    image: "assets/BraTS20_Training_096.png",
    volumeShape: "4 × 64 × 64 × 64  (T1, T1ce, T2, FLAIR)",
    lesionSide: "right",
    lesionRegion: "frontal / parietal",
    groundTruth: "error",
    gtLabel: "HALLUCINATION — LATERALITY FLIP",
    gtDetail: "Reference says RIGHT frontal/parietal; the generator wrote LEFT. Wrong side of the brain.",
    reference:
      "The lesion area is in the right frontal and parietal lobes with heterogeneous high and low signals. Edema is mainly observed around the lesion areas in the right frontal and parietal lobes, presenting as diffuse low-signal areas with a relatively large extent. Necrosis is concentrated in the center of the lesions with uneven signal intensity, displaying a mix of high and low signals. Ventricular compression is in the right side of the ventricular system, slightly compressed but no obvious deformation is observed.",
    generated:
      "The lesion area is in the left parietal and occipital lobes with mixed signals of varying intensity and speckled high-signal regions. Edema is significant, particularly in the frontal and parietal lobes, affecting a large area of surrounding brain tissue. Necrosis is within the lesion region, primarily in the parietal lobe, characterized by low and uneven signal intensity. Ventricular compression is observed with the left lateral ventricle noticeably compressed.",
    axes: { semantic: 0.99, lexical: 0.27, structural: 0.12, numeric: 1.0, modality: 1.0, negation: 1.0, lesion_type: 1.0 },
    fused: 0.42,
    fusedVerdict: "FLAGGED",
    baselines: { bioclinicalbert: 0.99, ratescore: 0.29 },
  },
  {
    id: "BraTS20_Training_098",
    dataset: "BraTS 2020 volume + TextBraTS report",
    image: "assets/BraTS20_Training_098.png",
    volumeShape: "4 × 64 × 64 × 64  (T1, T1ce, T2, FLAIR)",
    lesionSide: "right",
    lesionRegion: "parietal",
    groundTruth: "error",
    gtLabel: "HALLUCINATION — LATERALITY FLIP",
    gtDetail: "Reference says RIGHT parietal; the generator wrote LEFT parietal/occipital. Wrong side.",
    reference:
      "The lesion area is in the right parietal lobe with a mixture of heterogeneous high and low signals, accompanied by patchy high-signal areas. Edema is significant around the lesion area in the right parietal lobe, extending further into parts of the right temporal lobe. Necrosis is characterized by a distinct low signal with an ill-defined mixed signal, mostly concentrated in the right parietal lobe. Ventricular compression is observed, with the right ventricle compressed and deformed, while the left ventricle shows relatively increased pressure but no obvious deformation.",
    generated:
      "The lesion area is in the left parietal and occipital lobes with mixed signals of varying intensity and speckled high-signal regions. Edema is significant, particularly in the parietal lobe, affecting the surrounding normal brain tissue. Necrosis is within the lesion region, primarily concentrated in the central part of the left frontal and parietal lobes. Ventricular compression is observed with the left lateral ventricle noticeably compressed and deformed.",
    axes: { semantic: 0.99, lexical: 0.27, structural: 0.22, numeric: 1.0, modality: 1.0, negation: 1.0, lesion_type: 1.0 },
    fused: 0.45,
    fusedVerdict: "FLAGGED",
    baselines: { bioclinicalbert: 0.99, ratescore: 0.27 },
  },
];
