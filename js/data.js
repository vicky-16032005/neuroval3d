// NeuroVal-3D Dashboard — all numerical results, baked in for static deployment

// ============== 7 Validator Specialists ==============
const VALIDATORS_INFO = [
  {
    id: "semantic",
    name: "Semantic",
    tech: "BioClinicalBERT cosine",
    catches: "Sentence-level meaning similarity",
    icon: "S",
  },
  {
    id: "lexical",
    name: "Lexical",
    tech: "VASARI TF-IDF + negation",
    catches: "Clinical keyword overlap",
    icon: "L",
  },
  {
    id: "structural",
    name: "Structural",
    tech: "VASARI parser → set F1",
    catches: "Feature-level consistency",
    icon: "T",
  },
  {
    id: "numeric",
    name: "Numeric",
    tech: "cm/mm regex + Jaccard",
    catches: "Size errors (3.5cm → 1.0cm)",
    icon: "#",
  },
  {
    id: "modality",
    name: "Modality",
    tech: "9-modality set Jaccard",
    catches: "T1 ↔ T2 ↔ FLAIR confusion",
    icon: "M",
  },
  {
    id: "negation",
    name: "Negation",
    tech: "Clause-aware NegEx / negspaCy",
    catches: "Polarity flips (no ↔ marked)",
    icon: "¬",
  },
  {
    id: "lesion_type",
    name: "Lesion Type",
    tech: "9-family disease Jaccard",
    catches: "glioma ↔ meningioma ↔ metastasis",
    icon: "Δ",
  },
  {
    id: "fusion",
    name: "Fusion (final)",
    tech: "sklearn LogisticRegression",
    catches: "Calibrated P(valid) ∈ [0,1]",
    icon: "Σ",
  },
];

// ============== Phase 1 Results ==============
const PHASE1_RESULTS = {
  textbrats: {
    label: "TextBraTS Held-Out",
    description: "369 real radiologist-refined reports → 1,829 (clean + perturbed) records → 70/30 split → 258 train / 111 test base reports.",
    headline: "0.9961",
    headlineSub: "Fusion test AUROC",
    validators: [
      { name: "NeuroVal-3D Fused (ours)", auroc: 0.9961, isOurs: true, isFusion: true },
      { name: "Structural", auroc: 0.6242, isOurs: true },
      { name: "Lexical", auroc: 0.4218, isOurs: true },
      { name: "BioClinicalBERT", auroc: 0.0821, isOurs: false },
      { name: "RaTEScore-lite", auroc: 0.0099, isOurs: false },
    ],
    stats: [
      { label: "Test AUROC", value: "0.9961", color: "text-teal-600" },
      { label: "Train AUROC", value: "0.9990", color: "text-slate-700" },
      { label: "Train-Test Gap", value: "+0.003", color: "text-emerald-600" },
      { label: "Multiplier vs BERT", value: "12.1×", color: "text-amber-600" },
    ],
  },
  radgenome: {
    label: "RadGenome-Brain MRI Held-Out",
    description: "1,007 reports across 5 disease subsets → ~5,000 records → 705 train / 302 test base reports. All 7 active perturbation ops fired.",
    headline: "0.9715",
    headlineSub: "Fusion test AUROC",
    validators: [
      { name: "NeuroVal-3D Fused (ours)", auroc: 0.9715, isOurs: true, isFusion: true },
      { name: "Lexical", auroc: 0.7345, isOurs: true },
      { name: "Structural", auroc: 0.7244, isOurs: true },
      { name: "Modality", auroc: 0.6062, isOurs: true },
      { name: "Numeric", auroc: 0.5927, isOurs: true },
      { name: "BioClinicalBERT", auroc: 0.2891, isOurs: false },
      { name: "RaTEScore-lite", auroc: 0.2203, isOurs: false },
    ],
    stats: [
      { label: "Test AUROC", value: "0.9715", color: "text-teal-600" },
      { label: "Train AUROC", value: "0.9699", color: "text-slate-700" },
      { label: "Train-Test Gap", value: "−0.002", color: "text-emerald-600" },
      { label: "Multiplier vs BERT", value: "3.4×", color: "text-amber-600" },
    ],
  },
  cross1: {
    label: "TextBraTS → RadGenome (cross-dataset)",
    description: "Fusion trained on TextBraTS, tested on RadGenome — the validator never saw RadGenome during training.",
    headline: "0.9358",
    headlineSub: "Cross-dataset test AUROC",
    validators: [
      { name: "Fusion (transfer test)", auroc: 0.9358, isOurs: true, isFusion: true },
      { name: "Lexical", auroc: 0.7354, isOurs: true },
      { name: "Structural", auroc: 0.7221, isOurs: true },
      { name: "Modality", auroc: 0.6021, isOurs: true },
      { name: "Numeric", auroc: 0.5950, isOurs: true },
      { name: "Negation", auroc: 0.3250, isOurs: true },
      { name: "BioClinicalBERT", auroc: 0.2766, isOurs: false },
      { name: "RaTEScore-lite", auroc: 0.2053, isOurs: false },
    ],
    stats: [
      { label: "Test AUROC", value: "0.9358", color: "text-teal-600" },
      { label: "Train AUROC", value: "0.9982", color: "text-slate-700" },
      { label: "n_train", value: "1,829", color: "text-slate-700" },
      { label: "n_test", value: "4,891", color: "text-slate-700" },
    ],
  },
  cross2: {
    label: "RadGenome → TextBraTS (cross-dataset)",
    description: "Fusion trained on RadGenome, tested on TextBraTS — perfect 1.000 generalisation.",
    headline: "1.0000",
    headlineSub: "Perfect transfer AUROC",
    validators: [
      { name: "Fusion (transfer test)", auroc: 1.0000, isOurs: true, isFusion: true },
      { name: "Structural", auroc: 0.6584, isOurs: true },
      { name: "Lexical", auroc: 0.4332, isOurs: true },
      { name: "Modality", auroc: 0.5000, isOurs: true },
      { name: "Numeric", auroc: 0.5000, isOurs: true },
      { name: "BioClinicalBERT", auroc: 0.0908, isOurs: false },
      { name: "RaTEScore-lite", auroc: 0.0148, isOurs: false },
    ],
    stats: [
      { label: "Test AUROC", value: "1.0000", color: "text-teal-600" },
      { label: "Train AUROC", value: "0.9728", color: "text-slate-700" },
      { label: "n_train", value: "4,891", color: "text-slate-700" },
      { label: "n_test", value: "1,829", color: "text-slate-700" },
    ],
  },
};

// ============== Phase 2 Loss Curves ==============
const PHASE2_LOSS = {
  epochs: [1, 2, 3, 4, 5],
  train: [2.8674, 1.6066, 1.3593, 1.2062, 1.0906],
  test:  [1.7292, 1.5758, 1.5364, 1.5971, 1.6228],
};

// ============== Phase 2 Discrimination Gap ==============
const PHASE2_DISCRIMINATION = {
  labels: [
    "NeuroVal-3D fused",
    "NeuroVal-3D structural",
    "BioClinicalBERT cosine",
    "NeuroVal-3D lexical",
    "RaTEScore-lite",
  ],
  cleanScores:  [0.951, 1.000, 0.999, 0.907, 0.931],
  hallucScores: [0.463, 0.893, 1.000, 0.934, 0.983],
  gaps:         [0.488, 0.107, -0.001, -0.027, -0.052],
  isOurs:       [true, true, false, true, false],
};

// ============== Phase 2 Detection Rate ==============
const PHASE2_DETECTION = {
  labels: [
    "NeuroVal-3D lexical",
    "NeuroVal-3D structural",
    "RaTEScore-lite",
    "BioClinicalBERT",
  ],
  detectionRate:  [89, 33, 100, 0],
  overallAccuracy: [70, 60, 45, 55],
  isOurs:          [true, true, false, false],
  // RaTEScore caught 100% only because it calls everything wrong (45% overall accuracy)
  trustworthy: [true, true, false, true],
};

// ============== Live Demo Preset Examples ==============
const PRESET_EXAMPLES = [
  {
    name: "Clean: matching reports",
    ref: "There is a 3.5 cm enhancing lesion in the left frontal lobe with marked surrounding oedema observed on T1ce. No haemorrhage. Mass effect on the lateral ventricle is noted.",
    gen: "A 3.5 cm enhancing mass is seen in the left frontal lobe with significant edema on T1ce. No bleeding present. Lateral ventricle shows mass effect.",
  },
  {
    name: "Laterality flip (left ↔ right)",
    ref: "There is a 3.5 cm enhancing lesion in the left frontal lobe with marked surrounding oedema observed on T1ce. No haemorrhage.",
    gen: "There is a 3.5 cm enhancing lesion in the right frontal lobe with marked surrounding oedema observed on T1ce. No haemorrhage.",
  },
  {
    name: "Region swap (frontal ↔ temporal)",
    ref: "There is a 3.5 cm enhancing lesion in the left frontal lobe with marked surrounding oedema observed on T1ce. No haemorrhage.",
    gen: "There is a 3.5 cm enhancing lesion in the left temporal lobe with marked surrounding oedema observed on T1ce. No haemorrhage.",
  },
  {
    name: "Negation flip (no ↔ marked)",
    ref: "There is a 3.5 cm enhancing lesion in the left frontal lobe with marked oedema observed on T1ce. No haemorrhage.",
    gen: "There is a 3.5 cm enhancing lesion in the left frontal lobe with no oedema observed on T1ce. Marked haemorrhage.",
  },
  {
    name: "Size error (3.5cm → 1.0cm)",
    ref: "There is a 3.5 cm enhancing lesion in the left frontal lobe with marked surrounding oedema observed on T1ce. No haemorrhage.",
    gen: "There is a 1.0 cm enhancing lesion in the left frontal lobe with marked surrounding oedema observed on T1ce. No haemorrhage.",
  },
  {
    name: "Modality confusion (T1 ↔ T2)",
    ref: "There is a 3.5 cm enhancing lesion in the left frontal lobe with marked surrounding oedema observed on T1ce. No haemorrhage.",
    gen: "There is a 3.5 cm enhancing lesion in the left frontal lobe with marked surrounding oedema observed on T2 FLAIR. No haemorrhage.",
  },
  {
    name: "Lesion-type swap (glioma → meningioma)",
    ref: "Findings consistent with a high-grade glioma in the left frontal lobe with marked surrounding oedema. Necrosis is observed.",
    gen: "Findings consistent with a meningioma in the left frontal lobe with marked surrounding oedema. Necrosis is observed.",
  },
];
