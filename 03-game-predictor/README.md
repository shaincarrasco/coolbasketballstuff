# 03 · NBA Game Predictor

Pre-game win probabilities from recent performance: Elo, rolling form, rest and
matchup features feeding a calibrated XGBoost classifier, evaluated the only
honest way for a sports model — forward in time. Shipped as a mobile-first
analytics page on the shared NBA Projects shell: KPI tiles, an interactive slate
picker, power rankings, a Monte Carlo season simulator, team drawers, an upset
detector and a three-model lab.

> **This is a learning project, not a betting tool.** It runs on *simulated*
> sample data with fictional teams. Not betting advice. Nothing here is guaranteed.

## What / why

Game prediction is the classic "it looks easy until you measure it properly"
problem. The hard parts are not the classifier; they are (1) building features
that use *only* information available before tip-off, (2) evaluating on games
the model has never seen in time order, and (3) reporting probabilities that
are actually calibrated instead of just picking winners. This project does all
three and shows its work: baselines, reliability table, walk-forward curves,
and a product-style view of the same facts (confidence tiers, hindcast slates).

## Stack

Python 3.13 · pandas · numpy · scikit-learn · XGBoost · (optional) `nba_api`.
No JS build; the report is one HTML file with inline SVG charts, one embedded
JSON blob (~600 KB, floats rounded to 3 dp) and ~150 lines of vanilla JS. Styling
comes from the shared shell (`../site/assets/shell.css` + `shell.js`) plus a
small page-level `<style>` block. No CDN, no webfonts.

## Run

```bash
cd 03-game-predictor
pip install -r requirements.txt
python cli.py demo                        # generate -> train -> predict -> simulate -> report (~8 s on a 2-core box)
open output/report.html                   # standalone copy: shell.css/js inlined, works from file://

# also publish into the shared static site (../site/predictor/index.html references ../assets/shell.*)
python cli.py demo --site ../site/predictor
python cli.py report --site ../site/predictor   # re-render only

# individual steps
python cli.py generate --seed 42          # data/games.csv + data/teams.csv (30 fictional teams, two conferences)
python cli.py train                       # XGBoost + Elo-logit + HistGradientBoosting, models/, metrics.json
python cli.py predict --date 2026-01-10   # one slate: win probabilities + key drivers (hindcast if played)
python cli.py simulate --date 2026-03-01 --sims 2000 --team PNE   # Monte Carlo rest of season
python cli.py rankings --date 2026-01-24  # Elo power rankings as of a date
python cli.py team HAV                    # team drill-down in the terminal
python cli.py report                      # output/report.html (recomputes analytics + simulation)
```

Outputs in `output/`: `report.html`, `predictions.json`,
`metrics.json`, `simulation.json`, and CSV exports `season_predictions.csv`,
`rankings.csv`, `simulation.csv`, `upsets.csv`, `confidence_tiers.csv`.
Models in `models/`: `model.json` (XGBoost), `calibrator.pkl`, `elo_logit.pkl`,
`hgb.pkl`, `feature_columns.json`, `metrics.json`.

## Report features

| Section | What it does | Where |
|---|---|---|
| **KPI tiles** | Test accuracy (hero, vs home baseline), log loss, season hindcast x/1230, #1 Elo team | top |
| **Slate** | All 172 game days of the test season pre-scored by the frozen model and embedded as JSON. Date select, prev/next buttons and arrow keys switch the board instantly. Per game: away @ home, away→home probability bar, both probabilities, pick chip, final score and Hit/Miss chip; on desktop the same rows also show pre-game Elo and three SHAP-style drivers. Day-accuracy and season-to-date chips plus a running-accuracy sparkline update with the date. Cards on mobile, table rows on desktop (one DOM, CSS grid reflow). | `analytics.season_predictions`, JS |
| **Season simulator** | Monte Carlo from a chosen date (default ~55% through the season): standings from played games, Elo as of that date, remaining games drawn N=2000 times from Elo probabilities with in-path Elo updates. Per team: expected wins, 5–95% interval, playoff odds (top-6 per conference, random tiebreak), #1-seed odds — plus the actual final record and IN/out, because the sample season is fully played (hindcast; playoff-odds Brier reported). Win-total histogram for one team with the actual result highlighted. | `simulate.py` |
| **Power rankings** | Elo table with rank change vs 30 days earlier, Δ30d (green/red), L10 form dots, Net10, W-L, Off10/Def10 (points for/against per game over the last 10 — rating proxies). Horizontal scroll with fade hint on mobile. | `analytics.power_rankings` |
| **Team drawer** | Tap any team code anywhere → bottom sheet (mobile) / side panel (desktop): record, home/away, back-to-back vs rested, as favourite / as underdog, model accuracy on that team, Elo curve, biggest upset wins/losses, full schedule with model p(win) and hit/miss. Esc or scrim closes; focus returns to the tapped code. | `analytics.team_pages`, JS |
| **Upset detector** *(in "Upsets & confidence tiers" details)* | Test-season games where the model gave the eventual winner < 35%, with "what the model missed" read from the winner's side (favourite on B2B, hotter L10, streak, H2H) — or the honest label *noise*. | `analytics.upsets` |
| **Confidence tiers** | Bins of pick confidence → actual hit rate, n, gap, pred/actual bars. Calibration, product view. | `analytics.tiers` |
| **Model lab** *(details)* | XGBoost vs Elo-logit vs sklearn HistGradientBoosting on the same split: accuracy / log loss / Brier / AUC, XGBoost gain share, HGB permutation importance, walk-forward curves for all models on one chart, reliability diagram + table. | `model.py` |

Every section ends with a one-sentence **Read:** line in plain English. The
"Sample data" chip sits in the shell top bar; "Not betting advice. Nothing here
is guaranteed." is stated once in the Method & limits footer and in the drawer.

## Design notes

Follows `../DESIGN_BRIEF.md` ("NBA internship ready", wave 3) and the shared
shell in `../site/assets/`:

- **Mobile-first.** 375 px is the primary viewport: single column, KPI tiles in
  a 2×2 grid, each game a compact card; at ≥ 900 px the same DOM reflows into
  a 10-column table via CSS grid (`.game` areas → `display:contents`). Wide
  tables scroll horizontally with a fade hint. All controls, team codes and
  the drawer close button are ≥ 40 px touch targets (team codes get padding +
  negative margin so the hit area is larger than the glyph).
- **One accent, color = meaning.** Magenta `var(--accent)` is used only for
  the hero number, the selected date, the active app icon and focus rings.
  Win/loss, hit/miss, up/down use `var(--pos)` / `var(--neg)`; back-to-back
  flags use `var(--warn)`. Charts are greyscale except where a value means
  something (actual final in the histogram, Elo curve direction).
- **Hierarchy.** Eyebrow → h1 → lede → 4 KPI tiles → interactive slate →
  rankings → simulator → progressive disclosure (`<details class="more">`) for
  upsets/tiers and the model lab → Method & limits → footer.
- **Numbers.** Probabilities as whole %, ratings (Net10/Off10/Def10, expected
  wins) to 1 dp, Elo as integers, tabular monospace figures.
- **No chrome of its own.** The top bar comes from `shell.js` (`data-app`,
  `data-title`, `data-mark` on `<body>`); no ticker, scanlines or cursor.
- **Two outputs, one body.** `output/report.html` inlines shell.css/js so it
  opens from `file://`; `--site DIR` writes `DIR/index.html` that links
  `../assets/shell.css` and `../assets/shell.js` for the GitHub Pages site.

## Switching to real data

`data/nba_api_loader.py` pulls regular-season games with `nba_api`
(`LeagueGameLog`, falling back to `LeagueGameFinder`) and writes the **same**
`games.csv` schema, so every downstream step is unchanged:

```bash
python cli.py generate --real --real-seasons 2022-23 2023-24 2024-25
python cli.py train && python cli.py demo
```

Every network call has a 20 s timeout and is wrapped in `try/except`; on any
failure it prints `stats.nba.com unreachable — using sample data` and falls back
to the simulator. (stats.nba.com was not reachable from the build machine, so
the shipped demo and all numbers below are from sample data.) With real data
the two conferences in `teams.csv` would need the real East/West mapping.

## The sample data

`data/sample_generator.py` simulates a 30-team league (fictional names and
3-letter codes, 15 "East" + 15 "West"), 3 seasons x 1230 games, balanced
home/away schedule, with:

- hidden team strength ~ N(0, 5) points, drifting daily as a random walk and
  regressing toward the mean between seasons;
- home-court advantage +2.5 points;
- rest effects: back-to-back −1.6, 3+ days rest +0.5 (B2B rate ≈ 20% of games);
- injuries as temporary strength dips (≈3.5 per team-season, 5–35 days, 1.5–6.5 pts);
- game noise ~ N(0, 12) points on the margin, totals ~ N(226, 14).

Those constants were chosen so the problem is *as hard as the real one*: a
perfect-knowledge oracle wins ~68% of games here and real NBA models land at
65–68%. Seeded (`--seed`), so runs are reproducible.

## Features (all strictly pre-game)

Computed in `features.py` by streaming games one calendar day at a time;
features for every game on a day are produced before any result from that day
is folded into team state.

| Group | Per team | Difference feature (home − away) |
|---|---|---|
| Elo | K = 20, home adv 80 Elo, margin-of-victory multiplier, 25% regression to 1500 each season | `elo_diff`, plus `elo_prob` (Elo win probability) |
| Form | win% over last 5 / 10 / 20 games, streak | `wp5_diff`, `wp10_diff`, `wp20_diff`, `streak_diff` |
| Margin / ratings proxy | L10 point differential, L10 pts for, L10 pts against | `pdiff10_diff`, `off10_diff`, `def10_diff` |
| Season | season win%, games played | `season_wp_diff`, `gp_diff` |
| Splits | home team's home win%, away team's away win% | `split_wp_diff` |
| Schedule | rest days (capped at 7), back-to-back flag | `rest_diff`, `b2b_diff`, `home_rest`, `away_rest`, `home_b2b`, `away_b2b` |
| Matchup | head-to-head over the last 4 meetings | `h2h_home_wp`, `h2h_n` |

20 features total. Early-season values are `NaN` where there is no history
(both GBMs handle missing values natively; Elo is always defined).

## Leakage safeguards

1. **Day-batched state updates.** Features for all games on date *d* are built
   from state as of *d − 1*; results from *d* are applied only afterwards.
2. **Independent re-derivation check.** `features.assert_no_leakage()` recomputes
   L10 win%, L10 margin and rest days for a random sample of games with an
   explicit `games.date < game_date` filter and asserts equality to 1e-9. It
   runs on every `train`.
3. **Time-based split.** Train on seasons 1–2, test on season 3 — no shuffling.
4. **Validation slice inside the training window.** Hyper-parameters and the
   calibrator were chosen on the last 60 days of season 2, never on season 3.
5. **Walk-forward evaluation.** For each month of season 3 every model is refit on
   everything before that month and scored on that month only.
6. **Hindcast honesty.** The slate picker, team pages, upsets and tiers all use
   the model trained on seasons 1–2 only, scoring season-3 games whose features
   come from prior games only; actual results are shown next to the picks as a
   check, never as an input.
7. **Simulator uses pre-date state only.** Standings and Elo as of the chosen
   date come from the same streaming feature pass; the "actual final" column is
   display-only.

## Results (sample data, seed 42, test season 2025-26, n = 1230)

| Model | Accuracy | Log loss | Brier | ROC AUC |
|---|---|---|---|---|
| Home team always wins | 57.6% | 0.6817 | 0.2443 | 0.500 |
| Logistic regression, Elo diff only | 64.5% | 0.6365 | 0.2225 | 0.671 |
| HistGradientBoosting (20 features) | 64.0% | 0.6424 | 0.2252 | 0.663 |
| XGBoost (raw) | 64.6% | 0.6382 | 0.2232 | 0.670 |
| **XGBoost + sigmoid calibration** | **64.5%** | **0.6380** | **0.2232** | **0.670** |

Expected calibration error (10 bins): XGBoost 0.033, Elo logistic 0.022.
Season hindcast (slate picker, all 172 game days): 793/1230 = 64.5%.

Walk-forward accuracy by month (season 3), XGBoost: 57% → 63% → 66% → 69% →
65% → 65% → 70%; HGB and Elo-logit track it within ±3 points each month.

Confidence tiers (pick confidence → hit rate): 50–55% → 54% (n=265),
55–60% → 60% (251), 60–65% → 61% (191), 65–70% → 74% (202), 70–75% → 68% (97),
75–80% → 74% (104), 80%+ → 77% (120). 138 of 1230 games were "upsets" where the
model gave the winner < 35%.

Simulator hindcast from 2026-01-24 (536 games remaining, N=2000): playoff-odds
Brier vs the actual final standings 0.077; all 10 teams given ≥ 70% playoff
odds made it, and two long shots (BRK at 28%, SLT at 11%) sneaked in — about
what a calibrated simulator should produce.

Top XGBoost features by gain share: `elo_prob` 16.7%, `elo_diff` 16.2%,
`pdiff10_diff` 12.6%, `season_wp_diff` 9.9%, `split_wp_diff` 6.0%. HGB
permutation importance agrees: `elo_diff` dominates, everything else is
≤ 0.005 log loss.

**Honest read:** every learner beats the home-team baseline by ~7 points of
accuracy and 0.04–0.045 log loss, and every learner *ties* a one-feature
logistic regression on Elo. That is the realistic outcome, not a failure: in a
world where team strength is mostly stable, a well-tuned Elo already captures
most of the predictable signal, and the extra features (rest, form, splits)
carry a small residual that ~2,500 training games cannot pin down reliably.
Real NBA models see the same thing. If this project had reported 80%+ accuracy,
that would have been the bug.

## Limitations

- **Simulated data.** Team names, conferences, schedules and scores are
  synthetic. The numbers above describe the simulator, not the NBA.
- Only team-level inputs: no player availability, lineups, travel, altitude,
  playoff seeding incentives or betting-market lines (the latter would dominate
  any of these features on real data).
- Small sample for a 20-feature GBM; hyper-parameters are heavily regularised
  and still gain nothing over Elo. More seasons would help.
- Elo constants (K, home advantage, carry-over) are fixed, not tuned.
- The season simulator uses Elo probabilities, not the XGBoost model (XGBoost
  needs rolling-form features that would have to be re-simulated game by game;
  Elo-only keeps 2000 paths under 2 s). Tiebreaks are random, not NBA rules.
- Off/Def "ratings" are points per game, not per possession — no pace data.
- "What the model missed" in the upset detector is heuristic context, not a
  causal attribution; most upsets are labelled noise, correctly.
- LightGBM is not used (not installed on the build machine); sklearn's
  HistGradientBoosting plays the third-model role.
- Default predict date is the last *full* slate (≥ 6 games) of the final season,
  because the last few calendar days of a simulated season are 1–2 game days.
- Not betting advice. Nothing here is guaranteed.

## Resume bullets

- Built a leak-free sports prediction pipeline (streaming day-batched features,
  Elo with margin-of-victory and season carry-over, independent re-derivation
  assertions) and evaluated it with a time-based split plus monthly walk-forward
  across three learners.
- Trained and calibrated an XGBoost win-probability model against baselines,
  reporting accuracy, log loss, Brier, AUC, 10-bin reliability and confidence
  tiers; selected the calibrator (none / isotonic / Platt) on a held-out slice.
- Shipped a vectorised Monte Carlo season simulator (expected wins, playoff and
  top-seed odds with a hindcast Brier check), per-game SHAP-style explanations,
  a CLI, a graceful real-data loader (`nba_api`) with offline fallback, and a
  mobile-first interactive HTML report (embedded JSON + vanilla JS) on a shared
  design system.
