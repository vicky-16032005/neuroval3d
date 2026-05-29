# NeuroVal-3D Dashboard

A self-contained static dashboard website showcasing the NeuroVal-3D project — built with vanilla HTML/CSS/JS for portability and zero-dependency local serving.

## What's in here

- **Overview** — Headline KPI cards + the "BioClinicalBERT 0.99 vs NeuroVal 0.18" hook
- **Architecture** — Two-pipeline visualization (generation + validation) + the 7 specialist validators
- **Phase 1 Results** — Interactive AUROC charts across 4 settings (TextBraTS held-out, RadGenome held-out, both cross-dataset directions) + the four-row paper-grade table
- **Phase 2 Loop Closure** — Loss curves + discrimination gap chart + detection rate on real hallucinations + the headline finding card
- **Live Demo** — Paste a reference + generated report (or pick a preset perturbation) and get real per-axis validator scores live, computed in-browser via vanilla JS reimplementations of the Python validators
- **Team** — All four students + guide + tech stack + resource links

## How to run locally

### Option 1 — Just open `index.html`
Double-click `dashboard/index.html` in Explorer. Most browsers will render it fully — Tailwind and Chart.js load from CDN, and the JS modules load from relative paths.

### Option 2 — Local HTTP server (recommended for the review demo)
A simple HTTP server avoids any file:// CORS quirks:

```powershell
# from project root, in PowerShell
py -3.11 -m http.server -d dashboard 8080
```

Then open http://localhost:8080 in any browser.

## How to deploy

Static site — drop into GitHub Pages, Netlify, Vercel, Cloudflare Pages, or any static host.

For GitHub Pages: enable Pages on the `vicky-16032005/neuroval3d` repo with source = `main` branch / `dashboard` folder.

## Files

```
dashboard/
├── index.html         # main page with all sections
├── css/styles.css     # custom styles (works alongside Tailwind CDN)
├── js/
│   ├── data.js        # all hardcoded results (AUROC tables, examples)
│   ├── validators.js  # JS reimplementations of the 7 specialists
│   └── app.js         # UI logic + Chart.js rendering + live demo
└── assets/            # (reserved for any image assets)
```

## Tech

- **Tailwind CSS** via CDN
- **Chart.js 4** via CDN for interactive charts
- **Inter** + **JetBrains Mono** from Google Fonts
- Vanilla JS — no framework, no build step
