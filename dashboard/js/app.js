// NeuroVal-3D Dashboard — main app logic

// ============== Render validator grid (Architecture section) ==============
function renderValidatorGrid() {
  const grid = document.getElementById("validator-grid");
  if (!grid) return;
  grid.innerHTML = VALIDATORS_INFO.map(v => `
    <div class="validator-card">
      <div class="v-name">
        <span class="v-icon">${v.icon}</span>
        ${v.name}
      </div>
      <div class="v-tech">${v.tech}</div>
      <div class="v-catches">${v.catches}</div>
    </div>
  `).join("");
}

// ============== Phase 1 tabs + chart ==============
let phase1Chart = null;

function renderPhase1Tab(tabKey) {
  const data = PHASE1_RESULTS[tabKey];
  if (!data) return;

  // Description
  document.getElementById("tab-description").textContent = data.description;

  // Stats panel
  const statsHtml = `
    <div class="bg-gradient-to-br from-teal-50 to-cyan-50 rounded-xl p-6 mb-3 text-center border border-teal-200">
      <div class="text-5xl font-extrabold text-teal-600">${data.headline}</div>
      <div class="text-sm text-slate-600 mt-2">${data.headlineSub}</div>
    </div>
    ${data.stats.map(s => `
      <div class="flex justify-between items-center bg-slate-50 rounded-lg p-3 border border-slate-200">
        <span class="text-sm text-slate-600">${s.label}</span>
        <span class="font-mono font-bold ${s.color}">${s.value}</span>
      </div>
    `).join("")}
  `;
  document.getElementById("tab-stats").innerHTML = statsHtml;

  // Chart
  const ctx = document.getElementById("phase1-chart").getContext("2d");
  if (phase1Chart) phase1Chart.destroy();

  phase1Chart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: data.validators.map(v => v.name),
      datasets: [{
        label: "Test AUROC",
        data: data.validators.map(v => v.auroc),
        backgroundColor: data.validators.map(v =>
          v.isFusion ? "#0d9488" : (v.isOurs ? "#2dd4bf" : "#e9c46a")
        ),
        borderColor: data.validators.map(v =>
          v.isFusion ? "#0f766e" : (v.isOurs ? "#0d9488" : "#d97706")
        ),
        borderWidth: 1.5,
        borderRadius: 4,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      indexAxis: "y",
      scales: {
        x: {
          beginAtZero: true,
          max: 1.0,
          title: { display: true, text: "AUROC" },
          grid: { color: "#e2e8f0" },
        },
        y: {
          grid: { display: false },
          ticks: { font: { size: 11 } },
        },
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => `AUROC: ${ctx.parsed.x.toFixed(4)}`,
          }
        },
      }
    }
  });
}

function initPhase1Tabs() {
  const buttons = document.querySelectorAll(".tab-btn");
  buttons.forEach(btn => {
    btn.addEventListener("click", () => {
      buttons.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      renderPhase1Tab(btn.dataset.tab);
    });
  });
  renderPhase1Tab("textbrats");
}

// ============== Phase 2 charts ==============
function renderLossChart() {
  const ctx = document.getElementById("loss-chart").getContext("2d");
  new Chart(ctx, {
    type: "line",
    data: {
      labels: PHASE2_LOSS.epochs,
      datasets: [
        {
          label: "Train Loss",
          data: PHASE2_LOSS.train,
          borderColor: "#1f3a4f",
          backgroundColor: "rgba(31, 58, 79, 0.1)",
          tension: 0.3,
          fill: false,
          pointRadius: 5,
          pointBackgroundColor: "#1f3a4f",
          borderWidth: 2.5,
        },
        {
          label: "Test Loss",
          data: PHASE2_LOSS.test,
          borderColor: "#2a9d8f",
          backgroundColor: "rgba(42, 157, 143, 0.1)",
          tension: 0.3,
          fill: false,
          pointRadius: 5,
          pointBackgroundColor: "#2a9d8f",
          borderWidth: 2.5,
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { title: { display: true, text: "Epoch" } },
        y: { title: { display: true, text: "Cross-Entropy Loss" }, beginAtZero: false },
      },
      plugins: {
        legend: { position: "top" },
      }
    }
  });
}

function renderDiscriminationChart() {
  const ctx = document.getElementById("discrimination-chart").getContext("2d");
  new Chart(ctx, {
    type: "bar",
    data: {
      labels: PHASE2_DISCRIMINATION.labels,
      datasets: [{
        label: "Discrimination Gap (clean - hallucinated)",
        data: PHASE2_DISCRIMINATION.gaps,
        backgroundColor: PHASE2_DISCRIMINATION.isOurs.map(o => o ? "#2a9d8f" : "#e9c46a"),
        borderColor: PHASE2_DISCRIMINATION.isOurs.map(o => o ? "#0f766e" : "#d97706"),
        borderWidth: 1.5,
        borderRadius: 4,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      indexAxis: "y",
      scales: {
        x: {
          title: { display: true, text: "Gap (higher = better detector)" },
          grid: { color: "#e2e8f0" },
        },
        y: {
          grid: { display: false },
          ticks: { font: { size: 10 } },
        },
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => {
              const i = ctx.dataIndex;
              const clean = PHASE2_DISCRIMINATION.cleanScores[i];
              const halluc = PHASE2_DISCRIMINATION.hallucScores[i];
              return [`Gap: ${ctx.parsed.x.toFixed(3)}`,
                      `Clean: ${clean.toFixed(3)}`,
                      `Hallucinated: ${halluc.toFixed(3)}`];
            }
          }
        }
      }
    }
  });
}

function renderDetectionChart() {
  const ctx = document.getElementById("detection-chart").getContext("2d");
  new Chart(ctx, {
    type: "bar",
    data: {
      labels: PHASE2_DETECTION.labels,
      datasets: [
        {
          label: "Detection Rate on 9 Hallucinations (%)",
          data: PHASE2_DETECTION.detectionRate,
          backgroundColor: PHASE2_DETECTION.isOurs.map(o => o ? "#2a9d8f" : "#e9c46a"),
          borderColor: PHASE2_DETECTION.isOurs.map(o => o ? "#0f766e" : "#d97706"),
          borderWidth: 1.5,
          borderRadius: 4,
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      indexAxis: "y",
      scales: {
        x: {
          beginAtZero: true,
          max: 100,
          title: { display: true, text: "Detection Rate (%)" },
          grid: { color: "#e2e8f0" },
        },
        y: {
          grid: { display: false },
          ticks: { font: { size: 11 } },
        },
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => {
              const i = ctx.dataIndex;
              const overall = PHASE2_DETECTION.overallAccuracy[i];
              const notes = PHASE2_DETECTION.trustworthy[i] ? "" :
                " (calls everything wrong — overall acc just " + overall + "%)";
              return `Detection: ${ctx.parsed.x}%, Overall acc: ${overall}%${notes}`;
            }
          }
        }
      }
    }
  });
}

// ============== Live Demo ==============
function initLiveDemo() {
  const refInput = document.getElementById("ref-input");
  const genInput = document.getElementById("gen-input");
  const presetSelect = document.getElementById("preset-select");
  const validateBtn = document.getElementById("validate-btn");
  const resultsPanel = document.getElementById("results-panel");

  // Preset loader
  presetSelect.addEventListener("change", () => {
    const idx = parseInt(presetSelect.value);
    if (isNaN(idx) || !PRESET_EXAMPLES[idx]) return;
    refInput.value = PRESET_EXAMPLES[idx].ref;
    genInput.value = PRESET_EXAMPLES[idx].gen;
    // Auto-validate
    runValidation();
  });

  // Load default preset (clean)
  presetSelect.value = "0";
  refInput.value = PRESET_EXAMPLES[0].ref;
  genInput.value = PRESET_EXAMPLES[0].gen;

  validateBtn.addEventListener("click", runValidation);

  function runValidation() {
    const ref = refInput.value;
    const gen = genInput.value;
    if (!ref.trim() || !gen.trim()) {
      alert("Please paste both a reference and a generated report.");
      return;
    }
    const scores = runAllValidators(gen, ref);
    if (!scores) return;
    renderResults(scores);
  }

  function renderResults(scores) {
    resultsPanel.classList.remove("hidden");
    resultsPanel.classList.add("animate-slideUp");

    // Verdict banner
    const verdict = scores.fused >= 0.5 ? "VALID" : "FLAGGED";
    const isPass = verdict === "VALID";
    const banner = document.getElementById("verdict-banner");
    banner.className = "mb-6 p-6 rounded-xl flex items-center justify-between " +
      (isPass ? "verdict-pass" : "verdict-fail");
    banner.innerHTML = `
      <div>
        <div class="text-xs font-semibold uppercase tracking-wider ${isPass ? "verdict-text-pass" : "verdict-text-fail"} mb-1">
          Fusion Verdict
        </div>
        <div class="text-3xl font-extrabold ${isPass ? "verdict-text-pass" : "verdict-text-fail"}">
          ${verdict}
        </div>
        <div class="text-sm ${isPass ? "verdict-text-pass" : "verdict-text-fail"} mt-1 opacity-80">
          ${isPass
            ? "The generated report agrees with the reference across most axes."
            : "The generated report disagrees with the reference — needs human review."}
        </div>
      </div>
      <div class="text-right">
        <div class="verdict-icon ${isPass ? "verdict-text-pass" : "verdict-text-fail"}">
          ${isPass ? "&#10003;" : "&#9888;"}
        </div>
        <div class="font-mono text-2xl font-bold ${isPass ? "verdict-text-pass" : "verdict-text-fail"}">
          ${scores.fused.toFixed(3)}
        </div>
        <div class="text-xs opacity-70 ${isPass ? "verdict-text-pass" : "verdict-text-fail"}">
          P(valid)
        </div>
      </div>
    `;

    // Per-axis bars
    const axisOrder = ["semantic", "lexical", "structural", "numeric", "modality", "negation", "lesion_type"];
    const axisLabels = {
      semantic: "Semantic (BERT)",
      lexical: "Lexical (VASARI)",
      structural: "Structural (VASARI)",
      numeric: "Numeric (cm/mm)",
      modality: "Modality (T1/T2)",
      negation: "Negation (NegEx)",
      lesion_type: "Lesion Type",
    };
    const html = axisOrder.map(axis => {
      const score = scores[axis];
      const pct = (score * 100).toFixed(0);
      const pass = score >= 0.5;
      const cls = pass ? "axis-bar-pass" : "axis-bar-fail";
      return `
        <div class="axis-row">
          <div class="axis-name">${axisLabels[axis]}</div>
          <div class="axis-bar-track">
            <div class="axis-bar-threshold"></div>
            <div class="axis-bar-fill ${cls}" style="width:${pct}%;"></div>
          </div>
          <div class="axis-score ${pass ? "text-emerald-600" : "text-red-600"}">${score.toFixed(2)}</div>
        </div>
      `;
    }).join("");
    document.getElementById("axis-results").innerHTML = html;
  }

  // Render initial results
  runValidation();
}

// ============== Smooth scroll for nav links ==============
function initSmoothScroll() {
  document.querySelectorAll('a[href^="#"]').forEach(link => {
    link.addEventListener("click", e => {
      const targetId = link.getAttribute("href");
      if (targetId === "#") return;
      const target = document.querySelector(targetId);
      if (target) {
        e.preventDefault();
        target.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    });
  });
}

// ============== Boot ==============
document.addEventListener("DOMContentLoaded", () => {
  renderValidatorGrid();
  initPhase1Tabs();
  renderLossChart();
  renderDiscriminationChart();
  renderDetectionChart();
  initLiveDemo();
  initSmoothScroll();
});
