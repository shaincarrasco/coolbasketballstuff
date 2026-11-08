# Five IQ — Lineup Intelligence

**Five players. One better fit.**

A model that learns which five-man lineups play well *together* — not just which
five players are individually best — then scores **every** possible five on every
NBA roster and hands you an interactive page: a lineup builder, pairwise synergy
heatmaps, what-if swaps, a minutes-budgeted rotation, an opponent-aware (heuristic)
mode and trade scenarios. Everything runs in the browser from static files.

> **Real data · ESPN.** The default build pulls real box scores for all 30 teams from
> ESPN's free public API (no key), with `stats.nba.com` (true 5-man stints via
> `nba_api`) tried first when reachable and the seeded synthetic league kept as
> `--source sample`. The page is labelled with the source and timestamp.
> **ESPN lineups are starting-five / closing-five approximations built from box
> scores, not stint-level data** — see [Data](#data-sources) for exactly how.

![mobile](../site/lineup/_preview_mobile.png)
![desktop](../site/lineup/_preview_desktop.png)

## What / why

Box scores describe players; lineups are where basketball actually happens.
Two 38% shooters next to a rim-running centre and a real creator are worth more
than the sum of their BPMs; three non-shooters together are worth less. Coaches
know this intuitively — this project makes it a measurable, searchable quantity.

Pipeline: **INPUT → LEARN → COMPARE → SELECT → BUILD**

1. **Input** — per-player season stats + historical five-man lineup results (minutes, OffRtg, DefRtg, NetRtg). Source picked automatically: `nba_api` → ESPN → sample.
2. **Learn** — 39 order-invariant lineup features → XGBoost predicts lineup net rating, weighted by minutes (a second XGBoost predicts OffRtg so net can be split into off / def).
3. **Compare** — GroupKFold by *team* (held-out rosters): XGBoost vs HistGradientBoosting vs Ridge vs predict-the-mean, plus a learning curve. On real data Ridge is not beaten — the report says so.
4. **Select** — enumerate all C(n,5) fives per roster (n ≤ 17 → up to 6 188), apply constraints, rank by predicted net → Fit score, explain with per-feature contributions.
5. **Build** — every scored five (≈179 k rows across 30 teams) is written as one compact JSON per team (`data/<ABBR>.json`, ≤ 160 KB); the page embeds every team's top-50 and lazily fetches the full table when you pick a team.

## Features

| # | feature | where |
|---|---|---|
| 1 | **Interactive lineup builder** — pick a team, tap five roster tiles; fit, predicted net / off / def, rank among all C(n,5) fives, top SHAP-style contributions, court diagram and history all update instantly (bitmask lookup; the team's full table is fetched on select) | builder |
| 2 | **Pairwise synergy heatmap** — for every pair, mean fit of lineups containing both minus the team mean; hover for the value, tap to load the duo; top-5 best / worst duos | synergy, `optimize.pair_synergy`, `output/synergy_<team>.csv` |
| 3 | **What-if swaps** — the 5 best single substitutions for the current five ("swap X for Y: +4.2 fit"); tap to apply | swaps, `optimize.best_swaps`, CLI `recommend` |
| 4 | **Rotation mode** — starters + bench five with minimal overlap under a minutes budget, plus a closing five that ignores minutes | rotation, `optimize.build_rotation`, `recommend --rotation` |
| 5 | **Opponent-aware mode** — `--vs <team>` re-ranks with a documented matchup **heuristic** (hand-set coefficients, not learned); also an opponent dropdown in the builder | builder + method footer, `optimize.matchup_adjust` |
| 6 | **Injury / trade scenario** — `--exclude` removes a player, `--add-player` pulls one in from another team and recomputes every five; roster tiles carry ESPN injury chips (OUT / DTD) | CLI `recommend`, builder |
| 7 | **Model lab** — third model (HistGradientBoosting), per-fold grouped bars, learning curve (training rows vs held-out Spearman, ±1 sd) | model lab, `model.learning_curve` |
| 8 | **Exports** — `output/all_lineups_<team>.csv` (every five with rank, feasibility, net / off / def, fit, S/H/B) and `output/synergy_<team>.csv` | `optimize.export_team` |

## Stack

Python 3.13 · pandas · numpy · scipy · scikit-learn (Ridge, HistGradientBoosting, GroupKFold) ·
XGBoost (model + `pred_contribs` attributions) · `nba_api` (optional, real data) ·
hand-written HTML/CSS/SVG + vanilla JS report (no CDN, no build step) · Playwright (verification only).

## Run

```bash
cd 02-lineup-intelligence
pip install -r requirements.txt          # pandas/numpy/sklearn/xgboost/requests (+ nba_api, playwright optional)

python cli.py demo --site ../site/lineup  # generate (auto source) -> train -> recommend + rotations -> exports -> report + site page
python verify_report.py                   # headless Playwright check at 390×844 and 1280×900, light + dark, + _preview_*.png

# data
python cli.py generate                    # auto: stats.nba.com (15 s timeout) -> ESPN -> sample
python cli.py generate --source espn      # ESPN only; incremental: fetches only games missing from data/cache/
python cli.py generate --source espn --prev-games 0   # whole previous season instead of each team's last 40 games
python cli.py generate --no-refresh       # rebuild players.csv / lineups.csv from the cache, no network
python cli.py generate --source sample [--seed 5150 --per-team 70]   # the fictional league
python -m data.espn_loader --prev-games 40                            # same refresh, standalone

# step by step
python cli.py train
python cli.py recommend --team "Celtics" --top 10
python cli.py recommend --team "Knicks" --min-bigs 2 --must Towns --exclude Hart --json
python cli.py recommend --team Thunder --rotation                   # starters / bench / closing five + minutes
python cli.py recommend --team Thunder --vs Nuggets                 # opponent-aware re-rank (heuristic, labelled)
python cli.py recommend --team Thunder --add-player "Antetokounmpo" --exclude Hartenstein   # trade scenario
```

First run on an empty cache: ~5 min (≈800 game summaries at 3 req/s) + ~1.5 min modelling.
Daily runs: only new games are fetched (seconds), then ~1.5 min.

Outputs

| path | what |
|---|---|
| `data/cache/espn_games.csv`, `espn_boxscores.csv.gz`, `espn_team_totals.csv.gz`, `espn_rosters.json`, `espn_skip.csv` | **committed incremental cache** keyed by ESPN game id (box-score lines, team totals, current rosters, exhibition ids to skip) |
| `data/players.csv`, `data/lineups.csv`, `data/source.json` | derived tables (641 players, 1 351 lineups this run) + provenance (source, timestamp, seasons, counts) |
| `models/model.json`, `models/model_off.json`, `models/ridge.pkl`, `models/hgb.pkl`, `models/meta.json` | XGBoost net + off boosters, Ridge, HistGB, feature list / metrics / learning curve / fit calibration / data source |
| `output/recommendations.json` | top-10 fives, swaps and rotation for every team + `data_source`, `metrics` (read by `../build_site.py`) |
| `output/all_lineups_<team>.csv` · `output/synergy_<team>.csv` | every scored five · every pair's synergy, per team |
| `output/report.html` + `output/data/<ABBR>.json` | standalone page (app foundation inlined, default team embedded) + per-team tables; serve the folder over http for the lazy loads (`python -m http.server`) |
| `../site/lineup/index.html` + `../site/lineup/data/<ABBR>.json` | static-site page written by `--site` (references `../assets/app.css` + `app-shell.js`) + per-team tables |
| `../site/lineup/_preview_mobile.png`, `_preview_desktop.png` | 390×844 and 1280×900 screenshots (light) written by `verify_report.py` |

## Design notes — app-polish redesign

The page follows the app-polish skill and the shared foundation in `../site/assets/`
(`app.css` tokens, grouped lists, stats, buttons, badges, sheets, toast; `app-shell.js`
translucent top bar with large-title collapse, bottom tab bar on phones, inline nav on desktop,
`App.toast`, service worker). Page-only CSS lives in the generated page's `<style>`. No CDN.
Reference apps: Apple Sports (centred score hero, team as the context), ESPN Fantasy and Sleeper
(roster grouped by position, checkmark picks, max five, live score while you pick).

* **Screen order (390 px first).** Large title "Lineups" -> team picker (a native `<select>`
  styled as a field showing the team name) -> hero: the five as initials avatars, the fit score
  with a plain verdict ("Top 3% of possible lineups for the Cavaliers", "Ranked 57th of 6,188"),
  net rating with a coloured ▲/▼ delta, points scored and allowed as a score-style "121–110",
  and the observed history in one sentence -> "Matchup" row (sheet; result badged "Estimate")
  -> roster grouped into Guards / Forwards / Centers with height, 3P% and Out / Day-to-day
  badges -> "Why this works" (3–4 plain-English traits with centred +/− bars) -> "Best swaps"
  (tap applies, toast "Swapped in X") -> "Best pairs" (top / bottom three; tap tries the pair;
  the full heatmap is in a "See all" sheet that scrolls sideways) -> "Rotation" (starters, bench,
  closing five as three compact lists with "Use this five") -> "Around the league" (top 10,
  all 30 in a sheet) -> "About this model".
* **Live score bar.** Once the hero scrolls away, a blurred bar above the tab bar shows five
  pips, "3 of 5" and the live fit, so swaps and picks further down give instant feedback.
  Tap it to jump back to the hero.
* **Colour.** One brand colour (basketball orange from `app.css`) for the fit score,
  checkmarks, "See all" / text buttons and the selected team row. Green / red only for
  contributions, swap and pair deltas, net rating and the heatmap. Light and dark follow the
  system setting.
* **Copy.** Sentence case, no eyebrows, no monospace, no "Read:" lines. Model names, rank
  correlation, the honest result (ridge regression edges XGBoost on held-out teams), the data
  source, the matchup and rotation heuristics and every box-score approximation live in the
  "About this model" sheet (opened from the top-bar info button or the last row).
* **States.** Empty (five open slots, "Pick five players to score a lineup"), partial
  ("Pick 2 more players"), loading (skeleton rows shaped like the lists while a team's full
  table downloads), error ("Couldn't load every Nuggets lineup..." with Try again), full five
  (toast + haptic). Team choice is kept in the URL hash.
* **Payload.** Unchanged: every team's roster, rotation, matchup profile and top-50 fives are
  embedded; each `data/<ABBR>.json` (≤ 160 KB, columnar, delta-encoded bitmasks, packed
  contributions) is fetched when the team is selected. Feature labels in the payload are the
  plain-English names from `report.PLAIN_LABELS`.

### Static site output (`--site`)

```bash
python cli.py demo --site ../site/lineup --no-refresh    # retrain from cache + write ../site/lineup/index.html
python report.py ../site/lineup                          # rebuild only the page from models/ + output/recommendations.json
python verify_report.py                                  # Playwright: 390×844 + 1280×900, light + dark, writes _preview_*.png
```

`output/report.html` is standalone (`app.css` + `app-shell.js` inlined, PWA meta tags without the
`../` links, `data-depth` points the nav at `../../site/`); it embeds the default team and loads
the others from `output/data/`, so open it over http (Chrome blocks `fetch` on `file://`; the page
then shows the error state with Try again and keeps working with the embedded top fives).
`<site>/index.html` uses the tags from `site/_head_snippet.html`, `../assets/app.css` and
`../assets/app-shell.js` (defer) and declares `<body data-app="lineup" data-title="Lineups">`.

## Data sources

`cli.py generate` (default `--source auto`) tries, in order:

1. **`stats.nba.com` via `nba_api`** (`data/nba_api_loader.py`) — true 5-man stint
   lineups (`leaguedashlineups`, GroupQuantity 5, Advanced) + player Advanced / Base /
   bio tables. 15 s timeout; on any failure it prints one line and moves on. It is
   blocked from the build machine but may work from GitHub Actions; when it does, the
   page is labelled "Live data · NBA Stats".
2. **ESPN public API** (`data/espn_loader.py`) — always reachable, no key, CORS `*`.
   **This is what the shipped build uses.**
3. **Synthetic sample league** (`data/sample_generator.py`) — offline fallback and
   `--source sample`.

### ESPN endpoints

| call | purpose |
|---|---|
| `site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates=YYYYMM&limit=1000` | one call per month: final games with id, date, `season.year/type`, competitors (the `YYYYMMDD-YYYYMMDD` range form returns 400) |
| `…/nba/summary?event=<id>` | box score: `boxscore.players[team].statistics[0]` (`labels` MIN PTS FG 3PT FT REB AST TO STL BLK OREB DREB PF +/-, `athletes[]` with id / name / position / `starter` / `didNotPlay`), `boxscore.teams[].statistics` (team FGA, FTA, TOV, OREB …), `injuries` |
| `…/nba/teams?limit=50` | the 30 franchises (ESPN ids 1–30; All-Star sides use other ids and are dropped) |
| `…/nba/teams/<id>/roster` | current roster: `height` (inches), `weight`, `age`, `position.abbreviation`, `injuries[0].status`, `status.type` |

Politeness: a token bucket at ≤ 3 requests/s, 4 retries with exponential backoff, a
descriptive User-Agent. Summaries that 404 (exhibitions) go to `espn_skip.csv` and are
never retried.

### Cache and refresh

`data/cache/` is committed. `espn_games.csv` is the index; a run lists the season's
final games from the monthly scoreboards and fetches only ids not in the index
(checkpointing every 200 games). Rosters are refetched every run (30 calls). Default
scope: the **current season in full plus each team's last 40 games of the previous
season** (`--prev-games 40`; `0` = whole previous season, ~1 230 more summaries ≈ 7 min
once). Seasons are detected from the scoreboard (`leagues[0].season.year`), so the
same command keeps working next October.

```bash
python cli.py generate --source espn        # daily: new games only
python -m data.espn_loader --prev-games 0   # one-off: backfill the full previous season
```

### Player table (ESPN → `features.py` columns)

Per-player sums pool the current and previous season (two seasons for stability early
in the year); rates with < 150 minutes are shrunk toward the league mean.

| column | derivation |
|---|---|
| `minutes` | minutes **this season only** → rotation budget and the DNP flag (0 = has not played this season) |
| `usage_pct` | 100 · Σ[(FGA + .44 FTA + TOV) · TmMIN/5] / Σ[MIN · (TmFGA + .44 TmFTA + TmTOV)] |
| `fg3_pct`, `fg3a_rate` | 3PM / 3PA, 3PA / FGA |
| `ast_pct` | 100 · Σ AST / Σ[(MIN / (TmMIN/5)) · TmFGM − FGM] |
| `reb_pct` | 100 · Σ[REB · TmMIN/5] / Σ[MIN · (TmREB + OppREB)] |
| `stl_pct`, `blk_pct` | 100 · Σ[STL · TmMIN/5] / Σ[MIN · OppPoss], 100 · Σ[BLK · TmMIN/5] / Σ[MIN · (OppFGA − Opp3PA)], OppPoss = .96 (FGA + .44 FTA + TOV − OREB) |
| `tov_pct` | 100 · TOV / (FGA + .44 FTA + TOV) |
| `obpm`, `dbpm` | **proxies, not BPM**: offensive / defensive halves of Hollinger's Game Score per 36 (off: PTS + .4 FGM − .7 FGA − .4 (FTA − FTM) + .7 OREB + .7 AST − TOV; def: .3 DREB + STL + .7 BLK − .4 PF), centred on the minutes-weighted league mean and scaled to sd 2.5, averaged 50/50 with half the player's shrunk +/− per 36 (pm36 · MIN / (MIN + 300)) |
| `def_rating_proxy` | minutes-weighted mean of the team's game defensive rating over the games the player appeared in (team context, not on/off) |
| `height_in`, `position` | roster endpoint (fallback: box-score position, median height by position) |
| `injury_status` | roster `injuries[0].status` ("Out", "Day-To-Day", …) at build time → tile chip |
| `active` | 1 = on the current ESPN roster, capped at 17 per team by minutes (so C(n,5) ≤ 6 188 and the builder stays exhaustive); former players stay in the table with `active = 0` only to featurise historical lineups |

### Lineup table (ESPN) — a documented approximation

Box scores do not say who was on the floor together, so each team-game contributes
two observations:

* `kind = starting` — the five flagged `starter`;
* `kind = closing` — the five with the most minutes (skipped when identical to the starters).

For each: `minutes` = mean of the five players' minutes; `net_rating` =
(Σ of the five players' +/−) / 5, scaled to per-100 possessions with
`lineup_poss = game_poss · minutes / game_minutes` and `game_poss` = mean over both teams
of .96 (FGA + .44 FTA + TOV − OREB) from the team totals (fallback 100 poss / 48 min);
`off_rating` = team game ORtg + (lineup net − team game net) / 2 and `def = off − net`
(the off/def split is an even-split assumption and is rougher than net). Identical fives
are aggregated across games (minutes-weighted mean net, total minutes, games); rows with
< 20 total minutes are dropped. The report, the JSON and the CLI all label these as
starting/closing-five approximations. The `nba_api` path replaces all of this with real
stint data and the same CSV schema (`games`/`kind` columns absent).

## Methodology

### Synthetic data (`data/sample_generator.py`, `--source sample`)
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
* **XGBoost** (production): depth-2 trees, 600 rounds at η = 0.02, 70% row /
  50% column subsampling, L2 = 5, `min_child_weight = 10` in lineup units (retuned
  lightly for ~1 300 noisy real lineups; a 6-config sweep moved held-out Spearman by
  < 0.01, so the choice is "fewer rounds, more regularisation", not a win).
  A second booster with the same settings (500 rounds) predicts OffRtg;
  DefRtg = Off − Net so the three always agree.
* **HistGradientBoosting** (sanity check): 300 iterations, η = 0.04, depth 3,
  6 leaves, `min_samples_leaf = 20`, L2 = 2.
* **Ridge** baseline (α = 100) on standardised features.
* All **sample-weighted by minutes** (normalised to mean 1), fits pinned to one
  thread (`threadpoolctl`) — on a 2-core box OpenMP oversubscription between
  XGBoost and HistGB made fits 20× slower.
* **GroupKFold by team** (leave-one-team-out up to 10 folds; 3 held-out teams per
  fold with 30): every score below is on rosters the model never saw, which is the
  regime the optimizer runs in.
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

## Results (this run: ESPN, 2025-26 last-40 + 2026-27 to date, 791 games, 1 351 lineups, 30 teams, 10 team-folds)

| model | RMSE (min-weighted) | R² (min-weighted) | Spearman (pooled OOF) | Spearman (mean per fold) | folds won |
|---|---:|---:|---:|---:|---:|
| Predict the mean | 15.90 | 0.000 | 0.000 | 0.000 | 0 |
| **Ridge (standardised)** | **13.33** | **0.297** | **0.509** | **0.446** | **6** |
| XGBoost · production | 13.39 | 0.290 | 0.499 | 0.442 | 3 |
| HistGradientBoosting | 13.75 | 0.253 | 0.473 | 0.401 | 1 |

Per held-out fold (Spearman, Ridge → XGBoost → HistGB): MEM/MIN/NY .47 → .50 → .45 ·
CHA/NO/WSH .52 → .51 → .51 · HOU/IND/PHX .42 → .41 → .31 · ATL/CLE/UTAH .50 → .56 → .56 ·
DEN/GS/LAL .45 → .39 → .34 · BKN/PHI/POR .43 → .39 → .35 · DAL/DET/LAC .53 → .51 → .48 ·
MIA/MIL/TOR .42 → .42 → .40 · CHI/SAC/SA .52 → .48 → .41 · BOS/OKC/ORL .22 → .25 → .19.

Learning curve (held-out Spearman, mean of 4 team-folds):

| training rows | 253 | 506 | 759 | 1013 |
|---|---:|---:|---:|---:|
| Ridge | 0.503 | 0.480 | 0.511 | 0.506 |
| XGBoost | 0.453 | 0.468 | 0.477 | 0.481 |
| HistGB | 0.420 | 0.441 | 0.444 | 0.463 |

Top features by gain share: total BPM 15%, defensive BPM sum 13%, offensive BPM sum 9%,
avg def rating 8%, weakest offensive player 4%; all interaction features together ≈ 8%.

**Honest read.** On real box-score lineups **XGBoost does not beat Ridge** (−0.011 pooled
Spearman, Ridge wins 6 of 10 folds) and the trees are still below the linear model at
every point of the learning curve. The signal is mostly additive talent (the BPM-proxy
sums dominate the gain); the fit / interaction effects the synthetic league was built
around are second-order here, and with σ(net) ≈ 16 points per lineup-observation the
noise floor is high. XGBoost stays the production model because the builder's per-feature
attributions come from `pred_contribs`; the lab table and its Read line say Ridge is the
benchmark. Sample-league numbers (XGBoost +0.02 over Ridge) are in the git history and
reproducible with `--source sample`. Fit = 100 · sigmoid((net − 0.35) / 15.9) this run.

## Limitations

* **Not stint data.** ESPN lineups are starting / closing fives inferred from box scores;
  a starter's +/− includes minutes with bench players, so lineup net ratings are blurred
  team-context signals. `nba_api` (when reachable) replaces them with real stints.
* **Target leakage, mildly.** The BPM proxies include each player's +/− per 36 from the
  same games that produce the lineup targets, so held-out-team metrics are optimistic.
* **Early-season scope.** 139 games of the current season plus each team's last 40 of
  the previous one; rotations and rosters have changed in between (former players are
  kept only to featurise history). `--prev-games 0` backfills the full previous season.
* **Proxies.** No BPM, no on/off, no tracking; the off / def split of a lineup is an
  even-split assumption; positions are ESPN's coarse G / F / C.
* **Rosters.** ESPN's current roster capped at 17 by minutes (two-way players with
  no minutes drop first); DNP = no minutes this season; injury chips are the status at
  build time and are shown, not used by the optimizer (`--exclude` does that).
* **Modest model.** Pooled Spearman ≈ 0.5 on held-out teams, ≈ 0.45 within a team;
  no uncertainty on predictions; no pace, garbage-time or schedule adjustments.
* **The opponent mode is a heuristic.** Three hand-set terms, clipped to ±4 points,
  labelled as such in the CLI, the JSON and the report.
* **The rotation is a heuristic too.** Minutes budgets are current-season minute
  shares, not availability.
* **Synergy is marginal, not causal.** A pair's synergy is the average over all
  lineups that contain it, so a bad pair on a deep roster can still look fine.
* Recommendations ignore foul trouble, fatigue, contracts and every other reason
  a coach actually picks a five.

## Resume bullets

* Built an end-to-end lineup-optimisation pipeline: vectorised order-invariant
  feature engineering over 5-player groups (39 features incl. pairwise and
  interaction terms), minutes-weighted XGBoost regression against Ridge and
  HistGradientBoosting baselines, team-grouped cross-validation with RMSE / R² /
  Spearman and a learning-curve study.
* Built a polite, incremental ESPN ingestion layer (≤ 3 req/s, backoff, committed
  gzip cache keyed by game id) covering all 30 teams, with a documented mapping from
  box scores to usage / assist / rebound / steal / block rates and lineup net ratings,
  and automatic source selection (`nba_api` → ESPN → synthetic) labelled in the output.
* Scored every C(n,5) lineup of all 30 rosters (~179 k fives) and shipped it as a
  static interactive page (vanilla JS, no CDN) with per-team lazy-loaded columnar
  tables ≤ 160 KB: a lineup builder with SHAP-style attributions, pairwise synergy
  heatmaps, what-if substitutions, a minutes-budgeted rotation optimiser, injury chips
  and a clearly-labelled opponent heuristic.
* Reported honestly that on real box-score lineups a regularised linear model matches
  gradient boosting (held-out-team Spearman 0.51 vs 0.50) and explained why; kept a
  seeded synthetic league with a hidden non-additive synergy function as the regime
  where trees win.
* Shipped a CLI with injury / trade scenarios, CSV exports and a Playwright-verified
  report (375 / 1280 px, lazy team loads) on a shared mobile-first design system.
