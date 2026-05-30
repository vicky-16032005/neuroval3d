// NeuroVal-3D Dashboard — interactive demo flow (real dataset cases + custom text).
// Builds the full pipeline walkthrough into #demo-root:
//   Step 1 Input MRI -> Step 2 Generated report -> Step 3 NeuroVal-3D validation -> Step 4 Comparison
// Real-case data: DEMO_CASES (demo_cases.js). Custom mode: runAllValidators (validators.js).

const DEMO_AXIS_LABELS = {
  structural: "Structural (VASARI F1)",
  lexical: "Lexical (VASARI TF-IDF)",
  semantic: "Semantic (BioClinicalBERT)",
  numeric: "Numeric (mm)",
  modality: "Modality (T1/T2/FLAIR)",
  negation: "Negation (NegEx)",
  lesion_type: "Lesion-type",
};
const DEMO_AXIS_ORDER = ["structural", "lexical", "semantic", "numeric", "modality", "negation", "lesion_type"];

function demoAxisBars(axes, measured) {
  return DEMO_AXIS_ORDER.map((k) => {
    const v = axes[k];
    if (v === undefined) return "";
    const pct = Math.round(v * 100);
    const pass = v >= 0.5;
    const cls = pass ? "axis-bar-pass" : "axis-bar-fail";
    let tag = "";
    if (measured && measured.includes(k)) {
      tag = '<span class="demo-axis-tag tag-measured">measured</span>';
    } else if (v === 1.0) {
      tag = '<span class="demo-axis-tag tag-silent">silent</span>';
    }
    return `
      <div class="axis-row">
        <div class="axis-name">${DEMO_AXIS_LABELS[k]}${tag}</div>
        <div class="axis-bar-track"><div class="axis-bar-threshold"></div>
          <div class="axis-bar-fill ${cls}" style="width:${pct}%;"></div></div>
        <div class="axis-score ${pass ? "text-emerald-600" : "text-red-600"}">${v.toFixed(2)}</div>
      </div>`;
  }).join("");
}

function demoVerdictPill(score) {
  return score >= 0.5
    ? '<span class="px-2 py-0.5 rounded-full bg-emerald-100 text-emerald-700 text-xs font-bold">VALID</span>'
    : '<span class="px-2 py-0.5 rounded-full bg-red-100 text-red-700 text-xs font-bold">FLAGGED</span>';
}

function renderRealCase(idx) {
  const c = DEMO_CASES[idx];
  const correctVerdict = c.groundTruth === "correct" ? "VALID" : "FLAGGED";
  const gtIsError = c.groundTruth === "error";

  const validators = [
    { name: "NeuroVal-3D (fused)", score: c.fused, ours: true },
    { name: "BioClinicalBERT cosine", score: c.baselines.bioclinicalbert, ours: false },
    { name: "RaTEScore-lite", score: c.baselines.ratescore, ours: false },
  ];
  const compRows = validators.map((v) => {
    const verdict = v.score >= 0.5 ? "VALID" : "FLAGGED";
    const right = verdict === correctVerdict;
    return `
      <tr class="${v.ours ? "bg-teal-50" : ""}">
        <td class="p-2 font-medium ${v.ours ? "text-teal-800" : "text-slate-700"}">${v.name}${v.ours ? " ★" : ""}</td>
        <td class="p-2 text-center font-mono">${v.score.toFixed(2)}</td>
        <td class="p-2 text-center">${demoVerdictPill(v.score)}</td>
        <td class="p-2 text-center font-bold ${right ? "text-emerald-600" : "text-red-600"}">${right ? "✓ correct" : "✗ wrong"}</td>
      </tr>`;
  }).join("");

  const takeaway = gtIsError
    ? `BioClinicalBERT scored this hallucinated report <b>${c.baselines.bioclinicalbert.toFixed(2)}</b> and called it VALID &mdash; it missed the laterality flip entirely. NeuroVal-3D flagged it (fused ${c.fused.toFixed(2)}), driven by the structural axis at ${c.axes.structural.toFixed(2)}.`
    : `A faithful generation. NeuroVal-3D (${c.fused.toFixed(2)}) and BioClinicalBERT (${c.baselines.bioclinicalbert.toFixed(2)}) both correctly accept it. RaTEScore-lite flags it anyway &mdash; it tends to reject almost everything.`;

  const gtColor = gtIsError ? "bg-red-500" : "bg-emerald-500";

  return `
    <div class="demo-flow-card">
      <div class="demo-step">
        <div class="demo-step-head"><span class="demo-step-num">1</span> Input &mdash; 3D Brain MRI Volume</div>
        <div class="grid md:grid-cols-3 gap-4 items-center">
          <div class="md:col-span-2">
            <img src="${c.image}" alt="${c.id} four-modality MRI" class="w-full rounded-lg border border-slate-200">
          </div>
          <div class="text-sm space-y-1.5">
            <div><span class="text-slate-500">Subject:</span> <span class="font-mono">${c.id}</span></div>
            <div><span class="text-slate-500">Source:</span> ${c.dataset}</div>
            <div><span class="text-slate-500">Input tensor:</span> <span class="font-mono">${c.volumeShape}</span></div>
            <div><span class="text-slate-500">Lesion (truth):</span> <b>${c.lesionSide} ${c.lesionRegion}</b></div>
            <div class="text-xs text-slate-400 pt-1">Representative axial slices; the pipeline reads the real BraTS NIfTI volume.</div>
          </div>
        </div>
      </div>

      <div class="demo-arrow">&#9660;&nbsp; 3D CNN encoder &rarr; projector &rarr; BART-base decoder (143M params)</div>

      <div class="demo-step">
        <div class="demo-step-head"><span class="demo-step-num">2</span> Generated Report vs Reference</div>
        <div class="grid md:grid-cols-2 gap-4">
          <div class="report-box report-ref">
            <div class="report-label text-emerald-700">REFERENCE (radiologist)</div>
            <p>${c.reference}</p>
          </div>
          <div class="report-box report-gen">
            <div class="report-label text-slate-700">GENERATED (BART)</div>
            <p>${c.generated}</p>
          </div>
        </div>
        <div class="mt-3 p-3 rounded-lg text-white text-sm font-semibold ${gtColor}">
          GROUND TRUTH: ${c.gtLabel} &mdash; <span class="font-normal">${c.gtDetail}</span>
        </div>
      </div>

      <div class="demo-arrow">&#9660;&nbsp; NeuroVal-3D scores the (generated, reference) pair on 7 axes</div>

      <div class="demo-step">
        <div class="demo-step-head"><span class="demo-step-num">3</span> NeuroVal-3D Validation</div>
        <div class="space-y-2 mb-4">${demoAxisBars(c.axes, ["structural", "lexical", "semantic"])}</div>
        <div class="flex items-center justify-between p-4 rounded-xl ${c.fusedVerdict === "VALID" ? "verdict-pass" : "verdict-fail"}">
          <div>
            <div class="text-xs font-semibold uppercase ${c.fusedVerdict === "VALID" ? "verdict-text-pass" : "verdict-text-fail"}">Fused verdict (logistic regression over 7 axes)</div>
            <div class="text-2xl font-extrabold ${c.fusedVerdict === "VALID" ? "verdict-text-pass" : "verdict-text-fail"}">${c.fusedVerdict}</div>
          </div>
          <div class="text-right">
            <div class="font-mono text-2xl font-bold ${c.fusedVerdict === "VALID" ? "verdict-text-pass" : "verdict-text-fail"}">${c.fused.toFixed(2)}</div>
            <div class="text-xs ${c.fusedVerdict === "VALID" ? "verdict-text-pass" : "verdict-text-fail"} opacity-70">P(valid)</div>
          </div>
        </div>
      </div>

      <div class="demo-arrow">&#9660;&nbsp; How do existing validators score the same report?</div>

      <div class="demo-step">
        <div class="demo-step-head"><span class="demo-step-num">4</span> Comparison with Existing Validators</div>
        <table class="w-full text-sm">
          <thead class="bg-slate-900 text-white">
            <tr><th class="p-2 text-left">Validator</th><th class="p-2">Score</th><th class="p-2">Verdict</th><th class="p-2">vs ground truth</th></tr>
          </thead>
          <tbody class="divide-y divide-slate-100">${compRows}</tbody>
        </table>
        <div class="mt-3 text-sm text-slate-700 bg-slate-50 border border-slate-200 rounded-lg p-3">${takeaway}</div>
      </div>
    </div>`;
}

function renderRealMode(root) {
  const chips = DEMO_CASES.map((c, i) => {
    const err = c.groundTruth === "error";
    return `<button class="demo-case-chip" data-idx="${i}">
      <span class="font-mono text-xs">${c.id.replace("BraTS20_Training_", "BraTS #")}</span>
      <span class="demo-case-tag ${err ? "tag-err" : "tag-ok"}">${err ? "hallucinated" : "faithful"}</span>
    </button>`;
  }).join("");

  root.innerHTML = `
    <p class="text-center text-sm text-slate-500 mb-4">Choose a held-out case &mdash; two faithful generations, two with a hallucinated tumour side:</p>
    <div class="flex flex-wrap gap-2 justify-center mb-6">${chips}</div>
    <div id="demo-case-view"></div>`;

  const chipEls = root.querySelectorAll(".demo-case-chip");
  const view = root.querySelector("#demo-case-view");
  function select(i) {
    chipEls.forEach((el, j) => el.classList.toggle("active", j === i));
    view.innerHTML = renderRealCase(i);
  }
  chipEls.forEach((el) => el.addEventListener("click", () => select(parseInt(el.dataset.idx))));
  select(0);
}

function renderCustomMode(root) {
  const presetOpts = PRESET_EXAMPLES.map((p, i) => `<option value="${i}">${p.name}</option>`).join("");
  root.innerHTML = `
    <div class="bg-white rounded-2xl shadow-md p-6 border border-slate-200">
      <p class="text-sm text-slate-500 mb-4">Paste any reference + generated report (or load a perturbation example). Scores are computed live in your browser.</p>
      <div class="mb-4">
        <label class="text-sm font-semibold text-slate-700 mr-3">Perturbation example:</label>
        <select id="c-preset" class="rounded-lg border border-slate-300 text-sm px-3 py-2 bg-slate-50">
          <option value="">— custom —</option>${presetOpts}
        </select>
      </div>
      <div class="grid md:grid-cols-2 gap-4 mb-4">
        <div>
          <label class="block text-sm font-semibold text-slate-700 mb-2">Reference <span class="text-emerald-600">(ground truth)</span></label>
          <textarea id="c-ref" rows="6" class="w-full px-4 py-3 border border-slate-300 rounded-lg text-sm font-mono"></textarea>
        </div>
        <div>
          <label class="block text-sm font-semibold text-slate-700 mb-2">Generated <span class="text-red-600">(to validate)</span></label>
          <textarea id="c-gen" rows="6" class="w-full px-4 py-3 border border-slate-300 rounded-lg text-sm font-mono"></textarea>
        </div>
      </div>
      <button id="c-run" class="w-full bg-gradient-to-r from-teal-500 to-cyan-600 hover:from-teal-600 hover:to-cyan-700 text-white font-semibold py-3 rounded-lg transition">Validate &rarr;</button>
      <div id="c-results" class="mt-6 hidden">
        <div id="c-verdict" class="mb-5 p-5 rounded-xl flex items-center justify-between"></div>
        <h4 class="font-bold text-slate-800 mb-3">Per-axis scores</h4>
        <div id="c-axes" class="space-y-2"></div>
        <p class="text-xs text-slate-400 mt-3">These JS validators are simplified ports of the Python ones; the semantic axis is a word-overlap proxy for BioClinicalBERT.</p>
      </div>
    </div>`;

  const ref = root.querySelector("#c-ref");
  const gen = root.querySelector("#c-gen");
  const preset = root.querySelector("#c-preset");
  const results = root.querySelector("#c-results");

  preset.addEventListener("change", () => {
    const i = parseInt(preset.value);
    if (isNaN(i) || !PRESET_EXAMPLES[i]) return;
    ref.value = PRESET_EXAMPLES[i].ref;
    gen.value = PRESET_EXAMPLES[i].gen;
    run();
  });
  root.querySelector("#c-run").addEventListener("click", run);

  preset.value = "1";
  ref.value = PRESET_EXAMPLES[1].ref;
  gen.value = PRESET_EXAMPLES[1].gen;

  function run() {
    if (!ref.value.trim() || !gen.value.trim()) { alert("Paste both a reference and a generated report."); return; }
    const s = runAllValidators(gen.value, ref.value);
    results.classList.remove("hidden");
    const valid = s.fused >= 0.5;
    const banner = root.querySelector("#c-verdict");
    banner.className = "mb-5 p-5 rounded-xl flex items-center justify-between " + (valid ? "verdict-pass" : "verdict-fail");
    banner.innerHTML = `
      <div><div class="text-xs font-semibold uppercase ${valid ? "verdict-text-pass" : "verdict-text-fail"}">Fused verdict</div>
        <div class="text-2xl font-extrabold ${valid ? "verdict-text-pass" : "verdict-text-fail"}">${valid ? "VALID" : "FLAGGED"}</div></div>
      <div class="font-mono text-2xl font-bold ${valid ? "verdict-text-pass" : "verdict-text-fail"}">${s.fused.toFixed(2)}</div>`;
    root.querySelector("#c-axes").innerHTML = demoAxisBars({
      structural: s.structural, lexical: s.lexical, semantic: s.semantic,
      numeric: s.numeric, modality: s.modality, negation: s.negation, lesion_type: s.lesion_type,
    });
  }
  run();
}

function initRealDemo() {
  const host = document.getElementById("demo-root");
  if (!host) return;
  host.innerHTML = `
    <div class="flex gap-2 justify-center mb-8">
      <button class="demo-mode-btn active" data-mode="real">Real dataset cases</button>
      <button class="demo-mode-btn" data-mode="custom">Try your own text</button>
    </div>
    <div id="demo-real"></div>
    <div id="demo-custom" class="hidden"></div>`;

  const realPane = host.querySelector("#demo-real");
  const customPane = host.querySelector("#demo-custom");
  const btns = host.querySelectorAll(".demo-mode-btn");
  let customBuilt = false;

  renderRealMode(realPane);

  btns.forEach((b) => b.addEventListener("click", () => {
    btns.forEach((x) => x.classList.toggle("active", x === b));
    const real = b.dataset.mode === "real";
    realPane.classList.toggle("hidden", !real);
    customPane.classList.toggle("hidden", real);
    if (!real && !customBuilt) { renderCustomMode(customPane); customBuilt = true; }
  }));
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initRealDemo);
} else {
  initRealDemo();
}
