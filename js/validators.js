// NeuroVal-3D Dashboard — JavaScript reimplementations of the 7 validator specialists
// These mirror the Python implementations in src/neuroval3d/validators/
// so the live demo gives real-ish scores (not just fake numbers).

// ============== Vocabularies (mirror the Python VASARI lexicon) ==============
const VASARI_VOCAB = new Set([
  // anatomical regions
  "frontal", "parietal", "temporal", "occipital", "cerebellar", "cerebellum",
  "brainstem", "thalamus", "basal ganglia", "ventricle", "ventricles",
  // laterality
  "left", "right", "bilateral", "midline", "ipsilateral", "contralateral",
  // lesion descriptors
  "enhancing", "non-enhancing", "well-defined", "ill-defined", "necrosis",
  "haemorrhage", "hemorrhage", "calcification", "cyst", "mass", "lesion",
  "edema", "oedema", "swelling", "hyperintense", "hypointense", "isointense",
  "heterogeneous", "homogeneous", "diffuse", "focal", "infiltrative",
  // mass effect
  "midline shift", "mass effect", "compression", "displacement", "herniation",
  // count
  "single", "solitary", "multiple", "multifocal", "two", "three",
]);

const MODALITIES = ["t1", "t1ce", "t1gd", "t2", "flair", "dwi", "adc", "swi", "mra"];

const LESION_FAMILIES = {
  glioma: ["glioma", "glioblastoma", "astrocytoma", "oligodendroglioma", "gbm"],
  meningioma: ["meningioma"],
  metastasis: ["metastasis", "metastases", "metastatic"],
  infarction: ["infarct", "infarction", "stroke", "ischemia"],
  wmh: ["white matter hyperintensit", "wmh", "leukoaraiosis"],
  abscess: ["abscess"],
  hematoma: ["hematoma", "haematoma"],
  demyelination: ["demyelination", "demyelinat"],
  ms_lesion: ["ms lesion", "multiple sclerosis"],
};

const NEGATION_CUES = ["no", "without", "absent", "denies", "negative for", "not", "free of", "lacks"];

const NEGATION_TERMS = [
  "edema", "oedema", "haemorrhage", "hemorrhage", "enhancement",
  "restricted diffusion", "mass effect", "midline shift", "hydrocephalus",
  "calcification", "cyst", "necrosis", "satellite", "infiltration",
  "mass", "lesion", "nodule", "tumor", "tumour", "abnormality",
];

const REGIONS = ["frontal", "parietal", "temporal", "occipital", "cerebellar", "cerebellum", "brainstem"];

// ============== Utility functions ==============
function tokenize(text) {
  return text.toLowerCase().match(/[a-z][a-z\-]*/g) || [];
}

function jaccard(setA, setB) {
  if (setA.size === 0 && setB.size === 0) return 1.0;
  if (setA.size === 0 || setB.size === 0) return 0.0;
  let intersect = 0;
  for (const x of setA) if (setB.has(x)) intersect++;
  const union = setA.size + setB.size - intersect;
  return intersect / union;
}

// ============== 1. Semantic (BioClinicalBERT proxy via word-overlap) ==============
// Real implementation uses BioClinicalBERT. JS proxy: weighted word overlap on
// clinical vocab. This intentionally behaves like BERT cosine — high on surface
// similarity, struggles with content flips.
function semanticScore(gen, ref) {
  const gTokens = new Set(tokenize(gen));
  const rTokens = new Set(tokenize(ref));
  if (gTokens.size === 0 || rTokens.size === 0) return 0;
  // Word overlap ratio
  let common = 0;
  for (const t of gTokens) if (rTokens.has(t)) common++;
  // BERT cosine is dominated by surface similarity — high baseline
  const overlap = common / Math.max(gTokens.size, rTokens.size);
  // Normalize to BERT-like range (0.85 - 1.00) for typical clinical text
  return 0.85 + overlap * 0.15;
}

// ============== 2. Lexical (VASARI-restricted TF-IDF cosine) ==============
function lexicalScore(gen, ref) {
  const gVasari = new Set(tokenize(gen).filter(t => VASARI_VOCAB.has(t)));
  const rVasari = new Set(tokenize(ref).filter(t => VASARI_VOCAB.has(t)));
  if (gVasari.size === 0 && rVasari.size === 0) return 1.0;
  if (gVasari.size === 0 || rVasari.size === 0) return 0.0;
  return jaccard(gVasari, rVasari);
}

// ============== 3. Structural (VASARI feature parser → set F1) ==============
function extractStructuralFeatures(text) {
  const features = new Set();
  const lower = text.toLowerCase();
  // Laterality
  if (/\bleft\b/.test(lower)) features.add("laterality:left");
  if (/\bright\b/.test(lower)) features.add("laterality:right");
  if (/\bbilateral\b/.test(lower)) features.add("laterality:bilateral");
  // Regions
  for (const r of REGIONS) if (lower.includes(r)) features.add("region:" + r);
  // Enhancement
  if (/\benhancing\b/.test(lower) && !/\bnon-enhancing\b/.test(lower)) features.add("enhance:yes");
  if (/\bnon-enhancing\b/.test(lower)) features.add("enhance:no");
  // Oedema
  if (/\b(?:oedema|edema)\b/.test(lower) && !/\bno (?:oedema|edema)\b/.test(lower))
    features.add("edema:yes");
  // Necrosis
  if (lower.includes("necrosis")) features.add("necrosis:yes");
  // Haemorrhage (positive only)
  if (/\b(?:haemorrhage|hemorrhage|bleed)\b/.test(lower) &&
      !/\bno (?:haemorrhage|hemorrhage|bleed)\b/.test(lower))
    features.add("haemorrhage:yes");
  // Count
  if (/\b(?:single|solitary|one)\b/.test(lower)) features.add("count:single");
  if (/\b(?:multiple|multifocal|two|three)\b/.test(lower)) features.add("count:multi");
  return features;
}
function structuralScore(gen, ref) {
  const gF = extractStructuralFeatures(gen);
  const rF = extractStructuralFeatures(ref);
  if (gF.size === 0 && rF.size === 0) return 1.0;
  if (gF.size === 0 || rF.size === 0) return 0.0;
  // F1 over feature set
  let tp = 0;
  for (const f of gF) if (rF.has(f)) tp++;
  const precision = tp / gF.size;
  const recall = tp / rF.size;
  if (precision + recall === 0) return 0;
  return (2 * precision * recall) / (precision + recall);
}

// ============== 4. Numeric (cm/mm extraction + tolerance Jaccard) ==============
function extractMeasurements(text) {
  // Extract numbers followed by cm or mm. Normalise to mm.
  const out = new Set();
  const matches = text.matchAll(/(\d+(?:\.\d+)?)\s*(cm|mm)\b/gi);
  for (const m of matches) {
    let val = parseFloat(m[1]);
    if (m[2].toLowerCase() === "cm") val *= 10;
    // Round to nearest mm (tolerance)
    out.add(Math.round(val));
  }
  return out;
}
function numericScore(gen, ref) {
  const gM = extractMeasurements(gen);
  const rM = extractMeasurements(ref);
  if (gM.size === 0 && rM.size === 0) return 1.0;
  if (gM.size === 0 || rM.size === 0) return 0.0;
  // Tolerance-aware Jaccard: 1mm tolerance
  let matched = 0;
  for (const g of gM) {
    for (const r of rM) {
      if (Math.abs(g - r) <= 1) { matched++; break; }
    }
  }
  const union = gM.size + rM.size - matched;
  return matched / union;
}

// ============== 5. Modality (set Jaccard over modality tokens) ==============
function extractModalities(text) {
  const out = new Set();
  const lower = text.toLowerCase();
  for (const m of MODALITIES) {
    if (new RegExp(`\\b${m}\\b`).test(lower)) out.add(m);
  }
  return out;
}
function modalityScore(gen, ref) {
  const gM = extractModalities(gen);
  const rM = extractModalities(ref);
  if (gM.size === 0 && rM.size === 0) return 1.0;
  return jaccard(gM, rM);
}

// ============== 6. Negation (clause-aware NegEx) ==============
function extractNegationPolarity(text) {
  const out = new Set();
  const lower = text.toLowerCase();
  for (const term of NEGATION_TERMS) {
    const termRegex = new RegExp(`\\b${term}\\b`, "g");
    let match;
    while ((match = termRegex.exec(lower)) !== null) {
      const startIdx = match.index;
      // Look back within the current clause (up to last . ; , but/and)
      let clauseStart = 0;
      for (let i = startIdx - 1; i >= 0; i--) {
        const ch = lower[i];
        if (ch === "." || ch === ";" || ch === ",") { clauseStart = i + 1; break; }
        // also break on " but ", " and "
        if (i >= 5 && lower.slice(i - 4, i + 1) === " but ") { clauseStart = i + 1; break; }
        if (i >= 5 && lower.slice(i - 4, i + 1) === " and ") { clauseStart = i + 1; break; }
      }
      const clause = lower.slice(clauseStart, startIdx);
      const tokens = clause.match(/[a-z]+/g) || [];
      const negated = tokens.slice(-6).some(t => NEGATION_CUES.includes(t));
      out.add(`${term}:${negated ? "neg" : "pos"}`);
    }
  }
  return out;
}
function negationScore(gen, ref) {
  const gN = extractNegationPolarity(gen);
  const rN = extractNegationPolarity(ref);
  if (gN.size === 0 && rN.size === 0) return 1.0;
  if (gN.size === 0 || rN.size === 0) return 0.0;
  return jaccard(gN, rN);
}

// ============== 7. Lesion type (9-family set Jaccard) ==============
function extractLesionFamilies(text) {
  const out = new Set();
  const lower = text.toLowerCase();
  for (const [family, terms] of Object.entries(LESION_FAMILIES)) {
    for (const t of terms) {
      if (lower.includes(t)) { out.add(family); break; }
    }
  }
  return out;
}
function lesionTypeScore(gen, ref) {
  const gL = extractLesionFamilies(gen);
  const rL = extractLesionFamilies(ref);
  if (gL.size === 0 && rL.size === 0) return 1.0;
  return jaccard(gL, rL);
}

// ============== Fusion (logistic-style weighted combination) ==============
// Mirrors the Python sklearn LogisticRegression. Weights are roughly calibrated
// from the actual trained model on TextBraTS — structural and lexical dominate.
const FUSION_WEIGHTS = {
  semantic: 0.05,
  lexical: 0.20,
  structural: 0.35,
  numeric: 0.10,
  modality: 0.10,
  negation: 0.10,
  lesion_type: 0.10,
};

function fusionScore(axisScores) {
  let weighted = 0;
  for (const [axis, score] of Object.entries(axisScores)) {
    weighted += (FUSION_WEIGHTS[axis] || 0) * score;
  }
  // Apply sigmoid-like nonlinearity to push toward 0/1
  // shift around 0.5 threshold, scale by 4
  const x = (weighted - 0.5) * 6;
  const sigmoid = 1 / (1 + Math.exp(-x));
  return sigmoid;
}

// ============== Master entry point ==============
function runAllValidators(gen, ref) {
  if (!gen.trim() || !ref.trim()) return null;
  const scores = {
    semantic: semanticScore(gen, ref),
    lexical: lexicalScore(gen, ref),
    structural: structuralScore(gen, ref),
    numeric: numericScore(gen, ref),
    modality: modalityScore(gen, ref),
    negation: negationScore(gen, ref),
    lesion_type: lesionTypeScore(gen, ref),
  };
  scores.fused = fusionScore(scores);
  return scores;
}
