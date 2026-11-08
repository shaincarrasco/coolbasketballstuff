# 03 · NBA Game Predictor

Pre-game win probabilities from recent performance: Elo, rolling form, rest and
matchup features feeding a calibrated XGBoost classifier, evaluated the only
honest way for a sports model — forward in time. Shipped as a mobile-first
analytics page on the shared NBA Projects shell: KPI tiles, an interactive slate
picker, power rankings, a Monte Carlo season simulator, team drawers, an upset
detector and a three-model lab.

> **This is a learning project, not a betting tool.** It runs on real NBA games
> pulled from ESPN's free, *unofficial* public API (a synthetic sample league is
> the offline fallback). Not betting advice. Nothing here is guaranteed.

## What / why

Game prediction is the classic "it looks easy until you measure it properly"
problem. The hard parts are not the classifier; they are (1) building features
that use *only* information available before tip-off, (2) evaluating on games
the model has never seen in time order, and (3) reporting probabilities that
are actually calibrated instead of just picking winners. This project does all
three and shows its work: baselines, reliability table, walk-forward curves,
and a product-style view of the same facts (confidence tiers, hindcast slates).

## Stack

Python 3.13 · pandas · numpy · scikit-learn · XGBoost · stdlib `urllib` +
`zoneinfo` for the ESPN loader (legacy `nba_api` loader kept, optional).
No JS build; the report is one HTML file with inline SVG charts, one embedded
JSON blob (~150 KB, floats rounded to 3 dp) and ~200 lines of vanilla JS. Styling
comes from the shared shell (`../site/assets/shell.css` + `shell.js`) plus a
small page-level `<style>` block. No CDN, no webfonts.

## Run

```bash
cd 03-game-predictor
pip install -r requirements.txt
python cli.py demo                        # ESPN refresh -> train -> predict -> simulate -> report (~15 s warm, ~35 s cold cache)
open output/report.html                   # standalone copy: shell.css/js inlined, works from file://

# also publish into the shared static site (../site/predictor/index.html references ../assets/shell.*)
python cli.py demo --site ../site/predictor
python cli.py report --site ../site/predictor   # re-render only

# data
python cli.py generate                    # = --source espn: incremental refresh of data/cache/espn_games.csv -> games.csv, teams.csv
python cli.py generate --force            # ignore the cache, backfill from 2023-10-24 (~44 requests, ~20 s)
python cli.py generate --source sample --seed 42   # offline: 30 fictional teams, 3 synthetic seasons
python cli.py demo --source sample        # whole pipeline on the sample league
python cli.py generate --real             # LEGACY nba_api / stats.nba.com path (usually blocked from cloud boxes)

# individual steps (dates are US/Eastern game dates)
python cli.py train                       # XGBoost + Elo-logit + HistGradientBoosting, models/, metrics.json
python cli.py predict                     # today's slate (or the next day with games); --date YYYY-MM-DD for a hindcast
python cli.py simulate --sims 2000 --team BOS   # Monte Carlo over the published remaining schedule (from today)
python cli.py rankings                    # Elo power rankings as of the last played date
python cli.py team OKC                    # team drill-down in the terminal
python cli.py report                      # output/report.html (recomputes analytics + simulation)
```

If ESPN cannot be reached, `generate`/`demo` print a banner and (1) rebuild from
the committed cache if it exists (real but stale data), else (2) fall back to the
sample league. The page footer and top-bar chip always say which data it runs on.

## Data source: ESPN public API

Everything comes from ESPN's site API — free, no key, CORS `*`, but **unofficial
and undocumented**: it can change or disappear without notice.

| Use | Endpoint |
|---|---|
| Games by day | `https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates=YYYYMMDD&limit=50` |
| Games by month (backfill + published schedule) | same with `dates=YYYYMM&limit=500` (verified; the `YYYYMMDD-YYYYMMDD` range form returns 400 for the NBA) |
| Conferences (teams.csv) | `https://site.api.espn.com/apis/v2/sports/basketball/nba/standings` |
| Live scores (browser only) | the scoreboard endpoint for today's US/Eastern date, fetched by the page itself |

From each event: `id`, UTC tip-off (`date`), `season.year/type`, both
competitors (`abbreviation`, `displayName`, `homeAway`, `score`),
`status.type.name`, `neutralSite`. Season label = `season.year` mapped to
`"2025-26"`. Game dates are converted to **US/Eastern** so rest days and
back-to-backs follow the NBA calendar (a 10 pm ET tip is the same "day" as the
7 pm one even though it is tomorrow in UTC).

**Politeness:** ≤ 3 requests/s (0.35 s spacing), a descriptive `User-Agent`, 4
retries with exponential backoff on 429 / 5xx / network errors, 20 s timeout.

**Cache strategy** (`data/espn_loader.py`, cache committed to the repo):

- `data/cache/espn_games.csv` — every event fetched since 2023-10-24 (regular
  season, playoffs, play-in, preseason, All-Star; ~5,400 rows, 0.8 MB).
- Each run fetches only **from the last FINAL game minus 3 days** (late finals,
  postponements) **through today + 7** as single days; the rest of the current
  season's published schedule is re-fetched at most once per 24 h. A warm
  refresh is ~13 requests / 5 s; a cold backfill is 44 month requests / ~20 s.
- `data/games.csv` — the pipeline schema `game_id, date, season, home_team,
  away_team, home_pts, away_pts` plus `status, neutral, cup_final, home_name,
  away_name, season_type, tip_utc, espn_id`. Regular season **and** playoffs
  are written; playoffs are flagged (`season_type = 3`) and excluded from Elo
  and training by `features.load_games()` unless asked for. Unplayed games keep
  `NaN` scores so the same feature pass serves hindcast and forecast.
- `data/cache/espn_scheduled.csv` — unplayed regular-season games in the next
  7 days (the slate picker's "upcoming" dates and the default predict date).
- `data/teams.csv` — the 30 real franchises with ESPN abbreviations
  (`GS`, `NY`, `NO`, `SA`, `UTAH`, `WSH` …), names, East/West from the standings
  endpoint, ESPN team id.

Refresh command: `python cli.py generate` (or just `python cli.py demo`, which
refreshes first). Build-time snapshot in `data/meta.json`
(`updated_at`, counts, request count); the report prints it in the top-bar chip
and footer.

**Data quirks handled:** the NBA Cup final is tagged as a regular-season game by
ESPN but does not count in the standings — it is kept for Elo/training
(`cup_final = True`) and dropped from the simulator's W-L; neutral-site games
(Paris, Mexico City, Cup semis) are flagged but treated as home games for the
listed home side; ~30 games of the current season (Cup knockout rounds and the
make-up games for non-finalists) are not on ESPN's schedule until December, so
the simulator pro-rates expected wins to 82 and says so.

Outputs in `output/`: `report.html`, `predictions.json`,
`metrics.json`, `simulation.json`, and CSV exports `season_predictions.csv`,
`rankings.csv`, `simulation.csv`, `upsets.csv`, `confidence_tiers.csv`.
Models in `models/`: `model.json` (XGBoost), `calibrator.pkl`, `elo_logit.pkl`,
`hgb.pkl`, `feature_columns.json`, `metrics.json`.

## Report features

| Section | What it does | Where |
|---|---|---|
| **Top-bar chip / footer** | "Live data · ESPN" with the data timestamp (UTC) and final-game count; footer "Data: ESPN public API · updated …". On the sample league both say so instead. | `report.py`, JS |
| **KPI tiles** | Current-season accuracy (hero, with n and the full prior-season holdout underneath), log loss, season hindcast x/n with small-sample flag, #1 Elo team | top |
| **Slate** | Every played game day of the current season plus the next 7 days of scheduled games, pre-scored by the frozen model and embedded as JSON. Defaults to **today (US/Eastern)** or the next day with games; upcoming dates are marked ▸. Date select, prev/next buttons and arrow keys switch the board instantly. Per game: away @ home, away→home probability bar, both probabilities, pick chip, final score and Hit/Miss chip — or the ET tip time and no chip for unplayed games; on desktop the same rows also show pre-game Elo and three SHAP-style drivers. Day/season-to-date chips plus a running-accuracy sparkline. Cards on mobile, table rows on desktop (one DOM, CSS grid reflow). | `analytics.season_predictions`, JS |
| **Live scores strip** | In the browser the page fetches today's ESPN scoreboard (CORS `*`), shows "n live · n final · n upcoming", and overlays live status/score (and a Hit/Miss chip once final) on the matching cards via `data-espn-id`; refreshes every 60 s while the tab is visible, silent when offline. No logos or images. | JS |
| **Season simulator** | Monte Carlo from today over ESPN's published remaining schedule: standings from final games (Cup final excluded), Elo as of today, remaining games drawn N=2000 times from Elo probabilities with in-path Elo updates. Per team: expected wins (pro-rated to 82 while Cup games are unscheduled), 5–95% interval, top-6-seed odds per conference (random tiebreak), #1-seed odds. Win-total histogram for one team. On the sample league it is a mid-season hindcast with the actual finish and a playoff-odds Brier. | `simulate.py` |
| **Power rankings** | Elo table with rank change vs 30 days earlier, Δ30d (green/red), L10 form dots, Net10, W-L, Off10/Def10 (points for/against per game over the last 10 — rating proxies). Horizontal scroll with fade hint on mobile. | `analytics.power_rankings` |
| **Team drawer** | Tap any team code anywhere → bottom sheet (mobile) / side panel (desktop): record, home/away, back-to-back vs rested, as favourite / as underdog, model accuracy on that team, Elo curve, biggest upset wins/losses, full schedule with model p(win) and hit/miss. Esc or scrim closes; focus returns to the tapped code. | `analytics.team_pages`, JS |
| **Upset detector** *(in "Upsets & confidence tiers" details)* | Current-season games where the model gave the eventual winner < 35%, with "what the model missed" read from the winner's side (favourite on B2B, hotter L10, streak, H2H) — or the honest label *noise*. | `analytics.upsets` |
| **Confidence tiers** | Bins of pick confidence → actual hit rate, n, gap, pred/actual bars. Calibration, product view. | `analytics.tiers` |
| **Model lab** *(details)* | XGBoost vs Elo-logit vs sklearn HistGradientBoosting on the same split: accuracy / log loss / Brier / AUC, XGBoost gain share, HGB permutation importance, walk-forward curves for all models on one chart, reliability diagram + table. | `model.py` |

Every section ends with a one-sentence **Read:** line in plain English.
"Not betting advice. Nothing here is guaranteed." is stated once in the Method &
limits footer and in the drawer.

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

## Offline fallback: the sample league

`data/sample_generator.py` simulates a 30-team league (fictional names and
3-letter codes, 15 "East" + 15 "West"), 3 seasons x 1230 games, with hidden
drifting team strength, +2.5 home court, rest effects (B2B −1.6), injuries and
N(0, 12) game noise — tuned so a perfect oracle wins ~68%, the same regime as
the real thing. `--source sample` runs the identical pipeline on it (numbers in
the previous README revision: 64.5% test accuracy, i.e. the simulator is a
fair stand-in). `data/nba_api_loader.py` (`generate --real`) is the legacy
stats.nba.com path, kept for reference.

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
3. **Time-based split.** Train on every season before the latest one
   (2023-24 + 2024-25 + 2025-26), test on the in-progress 2026-27 — no
   shuffling. Elo carries over across the real seasons (25% regression to 1500).
   With < 200 test games the run is flagged `small_sample` everywhere it is shown.
4. **Prior-season holdout.** The same recipe one season back — train on
   2023-24 + 2024-25, test on the complete 2025-26 (n = 1231) — so there is
   always a full-season out-of-sample number next to the noisy live one.
5. **Validation slice inside the training window.** Hyper-parameters and the
   calibrator were chosen on the last 60 days of the final training season,
   never on the test season.
6. **Walk-forward evaluation.** For each month from the start of 2025-26 through
   the current month every model is refit on everything before that month and
   scored on that month only.
7. **Hindcast honesty.** The slate picker, team pages, upsets and tiers use the
   frozen model (trained on prior seasons only), scoring 2026-27 games whose
   features come from prior games only; actual results are shown next to the
   picks as a check, never as an input. Upcoming games show the same model's
   probability with no result chip until ESPN reports a final.
8. **Simulator uses pre-date state only.** Standings and Elo as of today come
   from the same streaming feature pass; the remaining games are ESPN's
   published schedule, never invented.
9. **Scheduled games advance rest only.** An unplayed game updates a team's
   "last game date" (the schedule is public before tip-off) but never its
   results or Elo, so the B2B flag for a game two days out is right.

## Results (real NBA data via ESPN, run of 2026-11-08)

Data: 3,832 final regular-season games 2023-10-24 → 2026-11-07
(1,231 + 1,231 + 1,231 + 139), 1,061 scheduled games of 2026-27 still to play.

**Prior full-season holdout — train 2023-24 + 2024-25, test 2025-26 (n = 1231):**

| Model | Accuracy | Log loss | Brier | ROC AUC |
|---|---|---|---|---|
| Home team always wins | 55.5% | 0.6874 | 0.2471 | 0.500 |
| Logistic regression, Elo diff only | 67.7% | 0.5992 | 0.2064 | 0.734 |
| HistGradientBoosting (20 features) | 67.4% | 0.6026 | 0.2076 | 0.731 |
| XGBoost (raw) | 68.4% | 0.5989 | 0.2059 | 0.740 |
| **XGBoost + sigmoid calibration** | **68.4%** | **0.5988** | **0.2058** | **0.740** |

**Current season — train 2023-24 … 2025-26, test 2026-27 to date (n = 139, small sample):**

| Model | Accuracy | Log loss | Brier | ROC AUC |
|---|---|---|---|---|
| Home team always wins | 51.8% | 0.6942 | 0.2505 | 0.500 |
| Logistic regression, Elo diff only | 67.6% | 0.5985 | 0.2076 | 0.741 |
| HistGradientBoosting (20 features) | 66.2% | 0.6067 | 0.2105 | 0.732 |
| XGBoost (raw) | 65.5% | 0.6083 | 0.2104 | 0.731 |
| **XGBoost + isotonic calibration** | **64.7%** | **0.6093** | **0.2106** | **0.727** |

Pooled out-of-sample (both sets, n = 1370): XGBoost 68.0% / log loss 0.600 /
ECE 0.024; Elo-logit 67.7% / 0.599 / ECE 0.029. Season hindcast on the slate
picker: 90/139 = 64.8% over 18 game days.

Walk-forward accuracy by month (XGBoost; Elo-logit in brackets): 2025-10 65%
(61) → 11 70% (68) → 12 58% (59) → 2026-01 62% (62) → 02 69% (67) → 03 77% (77)
→ 04 78% (81) → 2026-10 66% (64) → 11 71% (75). The dip in December and the
climb in March/April are real-NBA features: tanking and rest decisions make the
late season *easier* to call.

Top XGBoost features by gain share: `elo_prob` 16.3%, `season_wp_diff` 15.4%,
`elo_diff` 15.2%, `pdiff10_diff` 7.2%, `split_wp_diff` 6.3%, `wp20_diff` 6.1%.
HGB permutation importance on the full holdout: `elo_diff` +0.052 log loss,
`pdiff10_diff` +0.027, everything else ≤ 0.004.

Simulator (from 2026-11-08, 139 played / 1,061 scheduled, N = 2000): 10 teams at
≥ 70% for a top-6 seed; no hindcast Brier yet because the season is live.

**Honest read:** on a full real season the calibrated XGBoost beats "home team
wins" by 13 points and the one-feature Elo regression by 0.6 points — a real
but small edge, exactly the 65–68% band real NBA team-level models live in. The
current-season number (64.7% on 139 games) is within noise of that; it will move
by several points before December and the page says so. Market lines would beat
all of this; that is not what the project is for.

## Limitations

- **ESPN's API is unofficial and undocumented.** No SLA, no schema contract; a
  field rename or a bot-protection change breaks the loader (the committed cache
  and the sample league keep the demo runnable). Scores are taken only from
  `STATUS_FINAL` events; postponed/cancelled events are dropped.
- **Small current-season sample.** 139 games as of this run; every page element
  that uses it carries a small-sample flag, and the full prior-season holdout is
  shown alongside.
- **Live scores run in the visitor's browser** against ESPN directly; some
  bot filters (Akamai) reject headless browsers, in which case the strip simply
  stays empty — the page never depends on it.
- Only team-level inputs: no player availability, lineups, travel, altitude,
  playoff seeding incentives or betting-market lines (the latter would dominate
  any of these features on real data).
- ~3,700 training games for a 20-feature GBM; hyper-parameters are heavily
  regularised and the gain over Elo is under one point. More seasons would help.
- Elo constants (K, home advantage, carry-over) are fixed, not tuned.
- The season simulator uses Elo probabilities, not the XGBoost model (XGBoost
  needs rolling-form features that would have to be re-simulated game by game;
  Elo-only keeps 2000 paths under 2 s). Tiebreaks are random, not NBA rules.
- Off/Def "ratings" are points per game, not per possession — no pace data.
- "What the model missed" in the upset detector is heuristic context, not a
  causal attribution; most upsets are labelled noise, correctly.
- LightGBM is not used (not installed on the build machine); sklearn's
  HistGradientBoosting plays the third-model role.
- Playoff odds mean "top-6 seed" (direct spot); the play-in and NBA tiebreak
  rules are not modelled. Cup knockout games are added to the schedule by ESPN
  in December, until then expected wins are pro-rated to 82.
- Not betting advice. Nothing here is guaranteed.

## Resume bullets

- Built a leak-free NBA prediction pipeline on real games from ESPN's public API
  (polite incremental loader with cache, US/Eastern game dates, streaming
  day-batched features, Elo with margin-of-victory and season carry-over,
  independent re-derivation assertions) and evaluated it with time-based splits
  plus monthly walk-forward across three learners: 68.4% on a full held-out
  season vs 55.5% home baseline.
- Trained and calibrated an XGBoost win-probability model against baselines,
  reporting accuracy, log loss, Brier, AUC, 10-bin reliability and confidence
  tiers; selected the calibrator (none / isotonic / Platt) on a held-out slice.
- Shipped a vectorised Monte Carlo season simulator (expected wins, playoff and
  top-seed odds with a hindcast Brier check), per-game SHAP-style explanations,
  a CLI, graceful offline fallbacks (cached real data, then a synthetic league),
  and a mobile-first interactive HTML report (embedded JSON + vanilla JS, live
  ESPN score overlay) on a shared design system.
