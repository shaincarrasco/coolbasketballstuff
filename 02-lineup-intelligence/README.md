# Five IQ — Lineup Intelligence

**Five players. One better fit.**

A model that learns which five-man lineups play well *together* — not just which
five players are individually best — then scores **every** possible five on every
roster and hands you an interactive page: an interactive lineup builder, pairwise
synergy heatmaps, what-if swaps, a minutes-budgeted rotation, an opponent-aware
(heuristic) mode and trade scenarios. Everything runs offline from one HTML file.

> **Sample data.** stats.nba.com was not reachable from the build machine, so the
> shipped dataset is a seeded, fictional 8-team league (invented cities, nicknames
> and player names). **Fit scores are for demonstration only.** A real-data path
> via `nba_api` is included and documented below.

![mobile](../site/lineup/_preview_mobile.png)
![desktop](../site/lineup/_preview_desktop.png)

## What / why

Box scores describe players; lineups are where basketball actually happens.
Two 38% shooters next to a rim-running centre and a real creator are worth more
than the sum of their BPMs; three non-shooters together are worth less. Coaches
know this intuitively — this project makes it a measurable, searchable quantity.

Pipeline: **INPUT → LEARN → COMPARE → SELECT → BUILD**

1. **Input** — player season stats + historical 5-man lineup results (minutes, OffRtg, DefRtg, NetRtg).
2. **Learn** — 39 order-invariant lineup features → XGBoost predicts lineup net rating, weighted by minutes (a second XGBoost predicts OffRtg so net can be split into off / def).
3. **Compare** — GroupKFold by *team* (held-out rosters): XGBoost vs HistGradientBoosting vs Ridge vs predict-the-mean, plus a learning curve.
4. **Select** — enumerate all C(13,5) = 1287 fives per roster, apply constraints, rank by predicted net → Fit score, explain with per-feature contributions.
5. **Build** — every scored five (10,296 rows) is embedded in the report; the browser does the rest.

## Features

| # | feature | where |
|---|---|---|
| 1 | **Interactive lineup builder** — pick a team, tap five roster tiles; fit, predicted net / off / def, rank among all C(n,5) fives, top SHAP-style contributions, court diagram and history all update instantly (bitmask lookup into the embedded table) | builder |
| 2 | **Pairwise synergy heatmap** — for every pair, mean fit of lineups containing both minus the team mean; hover for the value, tap to load the duo; top-5 best / worst duos | synergy, `optimize.pair_synergy`, `output/synergy_<team>.csv` |
| 3 | **What-if swaps** — the 5 best single substitutions for the current five ("swap X for Y: +4.2 fit"); tap to apply | swaps, `optimize.best_swaps`, CLI `recommend` |
| 4 | **Rotation mode** — starters + bench five with minimal overlap under a minutes budget, plus a closing five that ignores minutes | rotation, `optimize.build_rotation`, `recommend --rotation` |
| 5 | **Opponent-aware mode** — `--vs <team>` re-ranks with a documented matchup **heuristic** (hand-set coefficients, not learned); also an opponent dropdown in the builder | builder + method footer, `optimize.matchup_adjust` |
| 6 | **Injury / trade scenario** — `--exclude` removes a player, `--add-player` pulls one in from another team and recomputes all C(14,5) = 2002 fives | CLI `recommend` |
| 7 | **Model lab** — third model (HistGradientBoosting), per-fold grouped bars, learning curve (training rows vs held-out Spearman, ±1 sd) | model lab, `model.learning_curve` |
| 8 | **Exports** — `output/all_lineups_<team>.csv` (every five with rank, feasibility, net / off / def, fit, S/H/B) and `output/synergy_<team>.csv` | `optimize.export_team` |

## Stack

Python 3.13 · pandas · numpy · scipy · scikit-learn (Ridge, HistGradientBoosting, GroupKFold) ·
XGBoost (model + `pred_contribs` attributions) · `nba_api` (optional, real data) ·
hand-written HTML/CSS/SVG + vanilla JS report (no CDN, no build step) · Playwright (verification only).

## Run

```bash
cd 02-lineup-intelligence
pip install -r requirements.txt          # pandas/numpy/sklearn/xgboost (+ nba_api, playwright optional)

python cli.py demo --site ../site/lineup  # generate -> train -> recommend + rotations -> exports -> report + site page (~60 s)
python verify_report.py                   # headless Playwright check at 375×812 and 1280×900 + _preview_*.png

# step by step
python cli.py generate [--seed 5150 --per-team 70]
python cli.py train
python cli.py recommend --team "Harbor City" --top 10
python cli.py recommend --team "Northgate" --min-bigs 2 --must Xiong --exclude Pembrook --json

# new modes
python cli.py recommend --team Northgate --rotation                   # starters / bench / closing five + minutes
python cli.py recommend --team Northgate --vs Ironvale                # opponent-aware re-rank (heuristic, labelled)
python cli.py recommend --team Northgate --add-player Kettering --exclude Pembrook   # trade scenario, 14-man roster
```

Example trade scenario output:

```
Northgate Voltage — 540 feasible of 2002 fives (exhaustive), constraints {... 'exclude': ('Pembrook',)}
Trade scenario: added Emeka Kettering (from Ironvale Foundry) -> 14-man roster
#    Fit    Net    Off    Def  S/H/B     Ht  rank  Lineup
1     82  +15.1  119.7  104.6  2/1/2   79.0     4  Marchetti, Iverness, Kettering, Zabala, Xiong
```

Outputs

| path | what |
|---|---|
| `data/players.csv`, `data/lineups.csv` | 104 players, 560 historical five-man lineups (sample) |
| `models/model.json`, `models/model_off.json`, `models/ridge.pkl`, `models/hgb.pkl`, `models/meta.json` | XGBoost net + off boosters, Ridge, HistGB, feature list / metrics / learning curve / fit-score calibration |
| `output/recommendations.json` | top-10 fives, swaps and rotation for every team |
| `output/all_lineups_<team>.csv` · `output/synergy_<team>.csv` | every scored five · every pair's synergy, per team |
| `output/report.html` | standalone page (shared shell inlined, ~810 KB with the embedded table) |
| `../site/lineup/index.html` | static-site page written by `--site` (references `../assets/shell.*`) |
| `../site/lineup/_preview_mobile.png`, `_preview_desktop.png` | 375×812 and 1280×900 screenshots written by `verify_report.py` |

## Design notes — shared "NBA Projects" shell (wave 3)

The page follows `../DESIGN_BRIEF.md` and uses the shared shell in `../site/assets/`
(`shell.css` tokens + components, `shell.js` top bar with app switcher and
service-worker registration). Nothing is loaded from a CDN.

* **Mobile-first.** 375 px is the primary viewport: single column, KPI tiles first,
  then the builder, then evidence. Desktop (≥ 900 px) goes to two columns (roster
  tiles beside the court, heatmap beside the duos) and opens the model lab.
* **One accent.** Lime `var(--accent)` is used only for the hero fit score, selected
  roster tiles, active controls and the five markers on the court. Everything else
  is colour-as-meaning: `--pos` / `--neg` for contributions, swap deltas, duos and
  the diverging synergy scale (`--neg → --surface-2 → --pos`); `--warn` only for the
  "heuristic" label. Player role tags (S / H / B / DNP, positions) are plain `.chip`s.
* **Hierarchy.** Fit /100 with "rank x of 1 287", predicted net, off / def and
  S / H / B answer "so what" before any interaction. Every panel ends with a
  one-sentence **Read:** line generated from the data (main driver / drag, best
  swap, duo lift, rotation split, league board, model lift).
* **Progressive disclosure.** Model lab (metrics table, per-fold bars, learning
  curve, gain share) sits in `<details class="more">`, collapsed on phones. Method &
  limits — including the opponent heuristic formula — is stated once in the footer.
* **Touch.** Tiles, swaps, buttons, selects, summary and board rows are ≥ 40 px;
  wide tables and the 13×13 heatmap scroll horizontally with a fade hint; charts are
  SVG sized by `viewBox`.
* **Payload.** The embedded table is ~780 KB: per combo `[bitmask, net (2 dp),
  off (1 dp), feasible, rank, [feature_idx, contribution (1 dp), …]]`; players carry
  only the fields the page uses.

### Static site output (`--site`)

```bash
python cli.py demo --site ../site/lineup                 # also writes ../site/lineup/index.html
python cli.py recommend --team Northgate --site ../site/lineup   # rebuilds the page after printing
python verify_report.py                                  # Playwright: 375×812 + 1280×900, writes _preview_*.png
```

`output/report.html` is standalone (shell CSS + JS inlined, `data-depth` points the
top-bar links at `../../site/`). `<site>/index.html` references `../assets/shell.css`
and `../assets/shell.js`, declares `<body data-app="lineup" data-title="Lineup
Intelligence" data-mark="LI">` and carries the PWA meta tags; the manifest, service
worker and icons live at the site root and are owned by the hub.

## Switching to real data (nba_api)

```bash
python cli.py generate --real --season 2024-25     # tries stats.nba.com, falls back to sample data
python cli.py train && python cli.py recommend --team BOS
```

`data/nba_api_loader.py` pulls `leaguedashlineups` (GroupQuantity=5, Advanced:
MIN / OFF_RATING / DEF_RATING / NET_RATING) and `leaguedashplayerstats`
(Advanced + Base) plus `leaguedashplayerbiostats` for height/position, joins on
player IDs parsed from `GROUP_ID`, and writes the same two CSVs the sample
generator does. Every request has a timeout and the whole thing is wrapped in
`try/except`; on any failure it prints
`stats.nba.com unreachable — using sample data` and returns `(None, None)`.

Caveats with the live site: hard rate limiting / cloud-IP blocking (keep the
browser-like headers, pass `timeout`, sleep ~1 s between calls, cache to disk);
far more and noisier rows (the loader filters `MIN >= 10`, the model weights by
minutes, expect lower R²); no BPM on stats.nba.com (`obpm`/`dbpm` are proxied
from on-court OFF/DEF rating deltas ×0.4); positions come as `G`/`F`/`C`/`G-F`
(the "big" flag also uses height ≥ 81 in). Rosters above 17 players switch the
optimizer to a beam search for the top fives; the full-table features (builder,
heatmap) need the exhaustive path, so trim rosters to the rotation first.

## Methodology

### Synthetic data (`data/sample_generator.py`)
* 8 fictional teams × 13 players drawn from 9 archetypes (lead guard, combo
  guard, sniper wing, 3&D, slasher, stretch big, rim protector, post big,
  energy big) with per-player minutes, usage, 3P%, 3PA rate, AST%, REB%, STL%,
  BLK%, TOV%, height, position, a defensive-rating proxy and BPM-like
  offence/defence values. Minutes are tilted toward better players; each team
  has one zero-minute (two-way/injured) player.
* 70 historical lineups per team, sampled with probability ∝ product of player
  minutes (rotation players play together), 12–700 minutes each.
* Ratings come from a **hidden synergy function** the model never sees:
  `0.35 × ΣBPM` (talent) + `0.9 ×` fit effects — non-linear spacing bonus by
  shooter count, three-non-shooter penalty, ball-handler redundancy, usage-sum
  penalty, turnover drag that doubles with no creator, shooters × best-passer
  and spacing × rim-protection interactions, saturating rim protection, hinge
  penalty for small lineups, size-variety and small-lineup rebounding penalties,
  steals that only pay with a backstop — plus Gaussian noise with
  `σ = 24/√minutes + 1`. The league is centred so minutes-weighted net = 0.

### Features (`features.py`, 39, all order-invariant, vectorised)
Sums / means / stds / mins / maxes of shooting, creation, size, defence,
impact and turnover stats; counts of shooters (3P% > .36 & 3PA rate > .30),
ball handlers (AST% > 20 & USG > 20) and bigs (≥ 81 in or C); pairwise features
over all 10 pairs; continuous interactions (shooters × bigs, shooters × best
AST%, handlers × top usage, steals × rim protection, spacing × rim protection).
`features_for_combos` is a pure-numpy implementation over an `(n_combos, 5)`
index array — all 1287 fives of a roster in ~7 ms, which is what makes scoring
every lineup of every team cheap enough to embed. **Deliberately excluded:**
hand-coded step flags like "exactly three non-shooters"; discovering the steps
is the tree model's job.

### Models (`model.py`)
* **XGBoost** (production): depth-2 trees, 1500 rounds at η = 0.01, 70% row /
  50% column subsampling, L2 = 2, `min_child_weight = 3` in lineup units.
  A second booster with the same settings (800 rounds) predicts OffRtg;
  DefRtg = Off − Net so the three always agree.
* **HistGradientBoosting** (sanity check): 300 iterations, η = 0.04, depth 3,
  6 leaves, `min_samples_leaf = 10`, L2 = 1.
* **Ridge** baseline (α = 30) on standardised features.
* All **sample-weighted by minutes** (normalised to mean 1), fits pinned to one
  thread (`threadpoolctl`) — on a 2-core box OpenMP oversubscription between
  XGBoost and HistGB made fits 20× slower.
* **GroupKFold by team** (8 folds): every score below is on rosters the model
  never saw, which is the regime the optimizer runs in.
* **Learning curve**: 4 team-folds × {25, 50, 75, 100}% of the training rows.
* **Fit score** = `100 · sigmoid((net_pred − μ) / σ)` with μ, σ the
  minutes-weighted mean / std of historical net rating (50 = average, 73 = +1σ,
  88 = +2σ, 95 = +3σ).

### Optimizer (`optimize.py`)
`score_team()` builds a `TeamTable`: every C(n,5) combination with predicted
net / off / def, fit, constraint feasibility, rank among all fives and exact
SHAP-style attributions from XGBoost's `pred_contribs`. Everything else is a
view over that table:

* **recommend** — top-N feasible fives (`min_bigs`, `max_zero_minute`,
  `must_include`, `exclude`), coach's most-used five for comparison, best swaps
  for #1, optional rotation, optional `--vs`, optional `--add-player`.
* **best_swaps** — 5 × (n − 5) single substitutions, sorted by Δfit.
* **pair_synergy** — mean fit of the C(n−2,3) lineups containing a pair minus
  the team mean over all C(n,5).
* **build_rotation** — budget `b_i = min(36, 240 × season-minute share)`
  (0 for zero-minute players); starters need `b_i ≥ 18` and play
  `clip(b_i, 30, 34)` where budgets allow (else `min(b_i, 34)`), bench need
  `b_i ≥ 8` and play `clip(b_i, 14, 18)`; search the top-60 starter candidates ×
  the best disjoint bench (overlap 1, then 2, only if no disjoint bench exists;
  a shared player's minutes are scaled to his budget), maximising the
  minutes-weighted predicted net of the two units. Closing five = best feasible
  five, budgets ignored. If the model rates the bench unit higher than the
  budget-forced starters, the report says so.
* **matchup_adjust** (`--vs`) — **heuristic, not learned**:
  `adj = −0.6 · (opp rim BLK% − league) · max(0, our non-shooters − 1)
  − 0.08 · (opp STL% sum − league) · (our TOV% sum − league)
  + 0.5 · (our avg height − opp avg height)`, clipped to ±4. "Opp" is the
  opponent's best feasible five, "league" is the mean of all eight teams' best
  fives. It re-ranks by `net + adj` and reports both numbers. The builder's
  opponent dropdown computes the same thing in JS.
* **export_team** — the two CSVs per team.

### Report (`report.py`)
Python emits the page, the static SVG charts (per-fold bars, learning curve) and a
compact JSON payload: per team the roster, every combo as
`[bitmask, net, off, feasible, rank, [feature_idx, contribution, …]]`, the
history map, the rotation and the matchup profile. ~45 lines of vanilla JS per
feature does the lookups (`Map<bitmask, row>`), draws the court and heatmap SVGs
and wires the tiles, swaps, duos and rotation buttons. `verify_report.py` drives
it headlessly with Playwright and asserts that selecting a different five changes
the fit, that swaps / heatmap / rotation populate and that no JS errors fire.

## Results (this run: seed 5150, 560 lineups, 8 held-out folds)

| model | RMSE (min-weighted) | R² (min-weighted) | Spearman (pooled OOF) | Spearman (mean per fold) | folds won |
|---|---:|---:|---:|---:|---:|
| Predict the mean | 10.02 | 0.000 | 0.000 | 0.000 | 0 |
| Ridge (standardised) | 5.75 | 0.671 | 0.773 | 0.716 | 2 |
| **XGBoost** | **5.39** | **0.711** | **0.789** | **0.742** | **5** |
| HistGradientBoosting | 5.60 | 0.688 | 0.776 | 0.734 | 1 |

Per held-out team (Spearman, Ridge → XGBoost → HistGB): Sable Ridge 0.611 → 0.773 → 0.763 ·
Redmesa 0.577 → 0.620 → 0.610 · Port Lumen 0.814 → 0.777 → 0.746 · Northgate 0.664 → 0.649 → 0.647 ·
Ironvale 0.724 → 0.726 → 0.722 · Harbor City 0.758 → 0.771 → 0.768 · Glasswater 0.816 → 0.843 → 0.824 ·
Copperfield 0.764 → 0.779 → 0.788.

Learning curve (held-out Spearman, mean of 4 team-folds):

| training rows | 105 | 210 | 315 | 420 |
|---|---:|---:|---:|---:|
| Ridge | 0.726 | 0.728 | 0.734 | 0.730 |
| XGBoost | 0.719 | 0.720 | 0.746 | 0.754 |
| HistGB | 0.687 | 0.692 | 0.714 | 0.734 |

The linear model is flat from 100 rows on; both tree models are still climbing at
420 — the expected picture for a target with interaction structure and ~500 rows.

Top features by gain share: shooters × bigs 17%, spacing × rim protection 13%,
shooters 9%, non-shooters 7%, bigs 6%, shooters × best passer 5%.

**Honesty note.** The XGBoost-over-Ridge gap is real but modest (+0.016 pooled,
+0.026 per-fold Spearman) and seed-sensitive: across six generator seeds at 70
lineups/team XGBoost won 6/6 with mean lift ≈ +0.02; at 60 lineups/team it won
5/6. HistGB lands between the two (+0.003 pooled over Ridge). With ~500 noisy
rows a well-featured linear model is a strong baseline, and that is the expected
finding, not a failure. The target itself is noisy (σ ∝ 1/√minutes), so R² = 1 is
unreachable even for the true function.

## Limitations

* **Synthetic data.** The league, players and synergy function are invented.
  Nothing here says anything about real NBA lineups.
* **Small sample.** 560 lineups / 8 teams. Real seasons have far more rows but
  far more noise per row.
* **Illustrative model.** Hyper-parameters were chosen with a handful of seeds,
  not a rigorous search; no uncertainty on predictions; no pace, garbage-time or
  schedule adjustments.
* **The opponent mode is a heuristic.** Three hand-set terms, clipped to ±4
  points. It is labelled as such in the CLI, the JSON and the report and should
  be read as "a direction to think in", not a prediction.
* **The rotation is a heuristic too.** Minutes budgets are season-minute shares,
  not real availability; the sample league's players under-play, so most teams
  allocate ~200 of 240 minutes and the remainder is simply "rest of roster".
* **Synergy is marginal, not causal.** A pair's synergy is the average over all
  lineups that contain it, so a bad pair on a deep roster can still look fine.
* **Fit scores are for demonstration only.** The 0–100 scale is a calibrated
  sigmoid of a point prediction, not a probability. Off / def predictions come
  from a separate booster and are rougher than net.
* Recommendations ignore foul trouble, fatigue, contracts and every other reason
  a coach actually picks a five.

## Resume bullets

* Built an end-to-end lineup-optimisation pipeline: vectorised order-invariant
  feature engineering over 5-player groups (39 features incl. pairwise and
  interaction terms), minutes-weighted XGBoost regression against Ridge and
  HistGradientBoosting baselines, team-grouped cross-validation with RMSE / R² /
  Spearman and a learning-curve study.
* Scored every C(13,5) lineup of an 8-team league and shipped it as a
  self-contained interactive report (vanilla JS, no CDN): a lineup builder with
  SHAP-style attributions, pairwise synergy heatmaps, what-if substitutions, a
  minutes-budgeted rotation optimiser and a clearly-labelled opponent heuristic.
* Designed a seeded synthetic data generator with a hidden non-additive synergy
  function and 1/√minutes noise to benchmark non-linear vs linear models in a
  no-network environment; documented the regime where trees beat linear models.
* Shipped a CLI with injury / trade scenarios, an optional `nba_api` adapter
  with graceful offline fallback, CSV exports and a Playwright-verified report
  on a shared mobile-first design system.
