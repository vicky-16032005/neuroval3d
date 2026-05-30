// NeuroVal-3D Dashboard — interactive demo flow.
// Step 0: insert an input MRI (upload a dataset panel OR pick from the gallery)
//   -> Step 1 Input  -> Step 2 Generate  -> Step 3 Validate  -> Step 4 Compare
// Real-case data: DEMO_CASES (demo_cases.js). Custom text mode: runAllValidators (validators.js).
//
// NOTE ON TRANSPARENCY: a browser cannot run the 143M-param BART generator or BioClinicalBERT
// (those need PyTorch + GPU + the trained checkpoint). So for an uploaded/selected dataset
// subject we replay the REAL precomputed report + validation scores from the Phase 2 Kaggle run.
// The "Try your own text" tab runs the live JS validators on any text you paste.

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

// Set true when the local Python backend (app/serve.py) is reachable -> live real validation.
let DEMO_BACKEND = false;

function demoAxisBars(axes, measured) {
  return DEMO_AXIS_ORDER.map((k) => {
    const v = axes[k];
    if (v === undefined) return "";
    const pct = Math.round(v * 100);
    const pass = v >= 0.5;
    const cls = pass ? "axis-bar-pass" : "axis-bar-fail";
    let tag = "";
    if (measured && measured.includes(k)) tag = '<span class="demo-axis-tag tag-measured">measured</span>';
    else if (v === 1.0) tag = '<span class="demo-axis-tag tag-silent">silent</span>';
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

// ---------------- Pipeline render for one case ----------------

function renderRealCase(idx, inputSrc, live) {
  const c = DEMO_CASES[idx];
  const correctVerdict = c.groundTruth === "correct" ? "VALID" : "FLAGGED";
  const gtIsError = c.groundTruth === "error";
  const imgSrc = inputSrc || c.image;

  // Prefer live backend scores when present; otherwise use the recorded Phase 2 values.
  const axes = live ? live.axes : c.axes;
  const fused = live ? live.fused : c.fused;
  const fusedVerdict = live ? live.verdict : c.fusedVerdict;
  const baselines = live ? live.baselines : c.baselines;
  const liveBadge = live
    ? `<span class="ml-2 px-2 py-0.5 rounded-full bg-teal-600 text-white text-[10px] font-bold align-middle">LIVE · real Python validators${live.bert_live ? " + BioClinicalBERT" : ""}</span>`
    : "";

  const validators = [
    { name: "NeuroVal-3D (fused)", score: fused, ours: true },
    { name: "BioClinicalBERT cosine", score: baselines.bioclinicalbert, ours: false },
    { name: "RaTEScore-lite", score: baselines.ratescore, ours: false },
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
    ? `BioClinicalBERT scored this hallucinated report <b>${baselines.bioclinicalbert.toFixed(2)}</b> and called it VALID &mdash; it missed the laterality flip entirely. NeuroVal-3D flagged it (fused ${fused.toFixed(2)}), driven by the structural axis at ${axes.structural.toFixed(2)}.`
    : `A faithful generation. NeuroVal-3D (${fused.toFixed(2)}) and BioClinicalBERT (${baselines.bioclinicalbert.toFixed(2)}) both correctly accept it. RaTEScore-lite flags it anyway &mdash; it tends to reject almost everything.`;

  const gtColor = gtIsError ? "bg-red-500" : "bg-emerald-500";

  return `
    <div class="demo-flow-card">
      <div class="demo-step">
        <div class="demo-step-head"><span class="demo-step-num">1</span> Input &mdash; 3D Brain MRI Volume</div>
        <div class="grid md:grid-cols-3 gap-4 items-center">
          <div class="md:col-span-2">
            <img src="${imgSrc}" alt="${c.id} four-modality MRI" class="w-full rounded-lg border border-slate-200">
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
        <div class="demo-step-head"><span class="demo-step-num">3</span> NeuroVal-3D Validation${liveBadge}</div>
        <div class="space-y-2 mb-4">${demoAxisBars(axes, ["structural", "lexical", "semantic"])}</div>
        <div class="flex items-center justify-between p-4 rounded-xl ${fusedVerdict === "VALID" ? "verdict-pass" : "verdict-fail"}">
          <div>
            <div class="text-xs font-semibold uppercase ${fusedVerdict === "VALID" ? "verdict-text-pass" : "verdict-text-fail"}">Fused verdict (logistic regression over 7 axes)</div>
            <div class="text-2xl font-extrabold ${fusedVerdict === "VALID" ? "verdict-text-pass" : "verdict-text-fail"}">${fusedVerdict}</div>
          </div>
          <div class="text-right">
            <div class="font-mono text-2xl font-bold ${fusedVerdict === "VALID" ? "verdict-text-pass" : "verdict-text-fail"}">${fused.toFixed(2)}</div>
            <div class="text-xs ${fusedVerdict === "VALID" ? "verdict-text-pass" : "verdict-text-fail"} opacity-70">P(valid)</div>
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
        <p class="text-[11px] text-slate-400 mt-2">${live
          ? "Validator scores computed live by the real Python validators on this machine. The generated report is the recorded Phase 2 output (the 143M-param generator runs on GPU, not per-request)."
          : "Report + scores are the actual recorded values from our Phase 2 Kaggle run for this subject (a browser cannot run the GPU model or BioClinicalBERT live — start the local backend for live validation)."}</p>
      </div>
    </div>`;
}

// ---------------- Step 0: input picker (upload + gallery) ----------------

function matchCaseByName(filename) {
  const f = (filename || "").toLowerCase();
  for (let i = 0; i < DEMO_CASES.length; i++) {
    const m = DEMO_CASES[i].id.match(/(\d+)\s*$/);
    const num = m ? m[1] : null;
    if (num && f.includes(num)) return i;
    if (f.includes(DEMO_CASES[i].id.toLowerCase())) return i;
  }
  return -1;
}

function renderRealMode(root) {
  const gallery = DEMO_CASES.map((c, i) => {
    const err = c.groundTruth === "error";
    return `<button class="demo-thumb" data-idx="${i}">
      <img src="${c.image}" alt="${c.id}">
      <div class="demo-thumb-meta">
        <span class="font-mono">${c.id.replace("BraTS20_Training_", "#")}</span>
        <span class="demo-case-tag ${err ? "tag-err" : "tag-ok"}">${err ? "hallucinated" : "faithful"}</span>
      </div>
    </button>`;
  }).join("");

  root.innerHTML = `
    <div class="demo-step demo-input-card">
      <div class="demo-step-head"><span class="demo-step-num">0</span> Insert an input MRI (from the dataset)</div>
      <div class="grid md:grid-cols-2 gap-6">
        <label class="demo-upload" id="demo-drop">
          <input type="file" id="demo-file" accept="image/*" style="display:none">
          <div class="demo-upload-inner">
            <div class="demo-upload-icon">&#8682;</div>
            <div class="font-semibold text-slate-700">Upload a BraTS MRI panel</div>
            <div class="text-xs text-slate-400 mt-1">drag &amp; drop or click &mdash; e.g. <span class="font-mono">BraTS20_Training_096.png</span></div>
            <div class="text-[11px] text-slate-400 mt-2">Files are in <span class="font-mono">dashboard/assets/</span></div>
          </div>
        </label>
        <div>
          <div class="text-sm font-semibold text-slate-600 mb-2">&hellip;or click a dataset case:</div>
          <div class="demo-gallery">${gallery}</div>
        </div>
      </div>
      <div id="demo-upload-msg" class="text-sm mt-3"></div>
    </div>
    <div id="demo-pipeline" class="mt-6"></div>`;

  const pipeline = root.querySelector("#demo-pipeline");
  const msg = root.querySelector("#demo-upload-msg");
  const fileInput = root.querySelector("#demo-file");
  const drop = root.querySelector("#demo-drop");

  function runFlow(idx, inputSrc) {
    const c = DEMO_CASES[idx];
    msg.innerHTML = `<span class="text-teal-700">Loaded <b>${c.id}</b>. Running pipeline&hellip;</span>`;
    // Stage 1: show input + processing spinner
    pipeline.innerHTML = `
      <div class="demo-step">
        <div class="demo-step-head"><span class="demo-step-num">1</span> Input received</div>
        <img src="${inputSrc || c.image}" class="w-full max-w-2xl mx-auto rounded-lg border border-slate-200">
      </div>
      <div class="demo-processing" id="demo-proc">
        <div class="demo-spinner"></div>
        <div id="demo-proc-text">Encoding MRI volume&hellip;</div>
      </div>`;
    pipeline.scrollIntoView({ behavior: "smooth", block: "start" });
    const txt = pipeline.querySelector("#demo-proc-text");
    const steps = ["Encoding MRI volume…", "Generating report (BART)…", "Scoring 7 validator axes…", "Comparing with baselines…"];
    let si = 0;
    const tick = setInterval(() => { si++; if (txt && steps[si]) txt.textContent = steps[si]; }, 420);
    const finish = (live) => {
      clearInterval(tick);
      pipeline.innerHTML = renderRealCase(idx, inputSrc, live);
      pipeline.scrollIntoView({ behavior: "smooth", block: "start" });
      msg.innerHTML = `<span class="text-emerald-700">Done &mdash; pipeline complete for <b>${c.id}</b>${live ? " &middot; <b>live validation</b>" : ""}.</span>`;
    };
    if (DEMO_BACKEND) {
      fetch("api/validate", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ generated: c.generated, reference: c.reference }),
      })
        .then((r) => r.json())
        .then((live) => setTimeout(() => finish(live && live.axes ? live : null), 500))
        .catch(() => setTimeout(() => finish(null), 500));
    } else {
      setTimeout(() => finish(null), 1900);
    }
  }

  root.querySelectorAll(".demo-thumb").forEach((el) =>
    el.addEventListener("click", () => runFlow(parseInt(el.dataset.idx))));

  function handleFile(file) {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (e) => {
      const idx = matchCaseByName(file.name);
      if (idx >= 0) {
        runFlow(idx, e.target.result);
      } else {
        msg.innerHTML = `<span class="text-amber-600">We have precomputed pipeline results for subjects 081, 094, 096 and 098. ` +
          `Your file "<b>${file.name}</b>" wasn't recognised &mdash; upload one of those panels (in <span class="font-mono">dashboard/assets/</span>) or click a case above.</span>`;
        pipeline.innerHTML = `
          <div class="demo-step">
            <div class="demo-step-head"><span class="demo-step-num">1</span> Your uploaded image</div>
            <img src="${e.target.result}" class="w-full max-w-2xl mx-auto rounded-lg border border-slate-200">
            <p class="text-sm text-slate-500 mt-3 text-center">Not a recognised held-out subject &mdash; pick one of the dataset cases above to run the full generate &rarr; validate flow.</p>
          </div>`;
      }
    };
    reader.readAsDataURL(file);
  }

  fileInput.addEventListener("change", (e) => handleFile(e.target.files[0]));
  ["dragover", "dragenter"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("dragging"); }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("dragging"); }));
  drop.addEventListener("drop", (e) => { if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]); });

  // auto-run the first case so the section is never empty
  runFlow(0);
}

// ---------------- Custom-text mode (live JS validators) ----------------

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

  function paint(s, fused, isLive) {
    results.classList.remove("hidden");
    const valid = fused >= 0.5;
    const banner = root.querySelector("#c-verdict");
    banner.className = "mb-5 p-5 rounded-xl flex items-center justify-between " + (valid ? "verdict-pass" : "verdict-fail");
    banner.innerHTML = `
      <div><div class="text-xs font-semibold uppercase ${valid ? "verdict-text-pass" : "verdict-text-fail"}">Fused verdict${isLive ? ' · <span class="text-teal-700">LIVE</span>' : ""}</div>
        <div class="text-2xl font-extrabold ${valid ? "verdict-text-pass" : "verdict-text-fail"}">${valid ? "VALID" : "FLAGGED"}</div></div>
      <div class="font-mono text-2xl font-bold ${valid ? "verdict-text-pass" : "verdict-text-fail"}">${fused.toFixed(2)}</div>`;
    root.querySelector("#c-axes").innerHTML = demoAxisBars(s);
  }

  function run() {
    if (!ref.value.trim() || !gen.value.trim()) { alert("Paste both a reference and a generated report."); return; }
    if (DEMO_BACKEND) {
      fetch("api/validate", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ generated: gen.value, reference: ref.value }),
      })
        .then((r) => r.json())
        .then((live) => {
          if (live && live.axes) paint(live.axes, live.fused, true);
          else { const s = runAllValidators(gen.value, ref.value); paint(s, s.fused, false); }
        })
        .catch(() => { const s = runAllValidators(gen.value, ref.value); paint(s, s.fused, false); });
    } else {
      const s = runAllValidators(gen.value, ref.value);
      paint(s, s.fused, false);
    }
  }
  run();
}

// ---------------- Boot ----------------

function initRealDemo() {
  const host = document.getElementById("demo-root");
  if (!host) return;
  host.innerHTML = `
    <div id="demo-backend-status" class="text-center text-xs mb-3"></div>
    <div class="flex gap-2 justify-center mb-8">
      <button class="demo-mode-btn active" data-mode="real">Insert dataset image</button>
      <button class="demo-mode-btn" data-mode="custom">Try your own text</button>
    </div>
    <div id="demo-real"></div>
    <div id="demo-custom" class="hidden"></div>`;

  const realPane = host.querySelector("#demo-real");
  const customPane = host.querySelector("#demo-custom");
  const status = host.querySelector("#demo-backend-status");
  const btns = host.querySelectorAll(".demo-mode-btn");
  let customBuilt = false;

  btns.forEach((b) => b.addEventListener("click", () => {
    btns.forEach((x) => x.classList.toggle("active", x === b));
    const real = b.dataset.mode === "real";
    realPane.classList.toggle("hidden", !real);
    customPane.classList.toggle("hidden", real);
    if (!real && !customBuilt) { renderCustomMode(customPane); customBuilt = true; }
  }));

  // Detect the local Python backend; render once we know (or after 1.5s).
  let rendered = false;
  const go = () => { if (rendered) return; rendered = true;
    status.innerHTML = DEMO_BACKEND
      ? '<span class="px-3 py-1 rounded-full bg-teal-100 text-teal-800 font-semibold">● Live Python backend connected — validation runs the real validators</span>'
      : '<span class="px-3 py-1 rounded-full bg-slate-100 text-slate-500">Static mode — showing recorded Phase 2 results. Run <span class="font-mono">app/serve.py</span> for live validation.</span>';
    renderRealMode(realPane);
  };
  fetch("api/health").then((r) => r.json()).then((j) => { DEMO_BACKEND = !!(j && j.ok); go(); }).catch(go);
  setTimeout(go, 1500);
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initRealDemo);
} else {
  initRealDemo();
}
