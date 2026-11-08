# Design brief — "NBA internship ready" (wave 3)

Goal: three apps that a basketball-operations analyst would be comfortable opening on a phone in a meeting,
plus a hub page. Shipped as one static GitHub Pages site, installable to a phone home screen (PWA).

## What the research says (distilled)
1. **Dark-first, restrained.** 2026 pro dashboards (Linear, Vercel, Grafana, Stripe) use near-black surfaces,
   **one accent color**, and reserve color for *meaning* (good/bad, win/loss, up/down). Neon glow everywhere reads
   as a hobby project. Keep each app's identity (amber / lime / magenta) as the single accent; drop text-shadow
   glows, scanlines, film grain, marquee tickers. Use subtle 1px borders and 2-level surface elevation.
2. **Density with hierarchy.** KPI tiles first (3–4 numbers that answer "so what"), then the interactive module,
   then supporting evidence (tables, model lab, methodology). Progressive disclosure: collapse methodology/model
   lab behind "Details" on mobile.
3. **Basketball-ops conventions** (CourtSketch, Cleaning the Glass, NBA.com stats): league-rank context next to
   every rating, recent-form dots, side-by-side comparisons, Four-Factors-style groupings, team color chips,
   tabular numerals, percentages to 1 decimal, ratings to 1 decimal, probabilities as whole %.
4. **Communicate insights to non-technical staff.** Every panel gets a one-sentence "Read:" line that says what
   the chart means in plain English. Model limitations are stated once in a "Method & limits" footer, not sprinkled.
5. **Mobile-first.** 375px is the primary viewport. Single column, sticky top bar with app switcher, 44px touch
   targets, horizontal scroll only for wide tables (with a fade hint), charts sized to width via SVG viewBox.
   Desktop (≥900px) goes to 2–3 columns. Test both.
6. **Typography.** System UI stack (-apple-system, Inter fallback) for text, tabular-nums monospace for figures.
   Headline weights 700–800, tight tracking. 14px body on mobile, 13px tables.
7. **Honest labeling.** "Sample league · synthetic data" chip in the top bar of each app; the hub explains why
   and how to switch to real data.

## Shared shell (hub + every app)
- Top bar: hub logo mark + app name · "Sample data" chip · app switcher (3 icons) — same across all pages.
- Color tokens: `--bg #0b0d12`, `--surface #12151c`, `--surface-2 #181c25`, `--line #242a36`,
  `--text #e6e9f0`, `--muted #8a93a6`, `--pos #3ddc84`, `--neg #ff5c7a`, `--warn #ffb020`.
- Per-app accent: Jumpshot `#ffa030`, Lineup `#a3ff3c`, Predictor `#ff3d8e`. Accent used for: active states,
  the single hero number, primary buttons, selected tiles. Nowhere else.
- Light mode not required; honor `prefers-color-scheme` minimally by keeping contrast ≥ 4.5:1 in dark.
- Icons: inline SVG, 20px, 1.75 stroke.

## PWA
- `manifest.webmanifest` at site root (`name`, `short_name` "NBA Projects", `start_url` "./", `display`
  "standalone", `background_color`/`theme_color` `#0b0d12`, icons 192/512 PNG + maskable).
- `sw.js` cache-first for same-origin assets with a versioned cache name; registered from every page.
- iOS: `<link rel="apple-touch-icon">`, `apple-mobile-web-app-capable`, `apple-mobile-web-app-status-bar-style`
  black-translucent, `viewport-fit=cover` with safe-area padding.
- All paths relative (`./`) so it works under `/nba-projects/` on GitHub Pages.

## Done means
- Lighthouse-style sanity: no console errors, no external requests, loads from `file://` and from a sub-path.
- Screenshot at 375×812 and 1280×900 for each page, inspected.
- Every interactive feature from wave 2 still works.
