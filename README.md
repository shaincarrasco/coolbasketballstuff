# NBA Coding Projects

Three portfolio-grade NBA × tech projects, built from the "NBA coding projects for your resume" idea list
(Jumpshot Analyzer · Lineup Intelligence · Game Predictor). Each project is self-contained, runs offline on
realistic sample data, and has an optional `nba_api` loader for real data when you run it at home.

| # | Project | Stack | One-liner |
|---|---------|-------|-----------|
| 01 | [Jumpshot Analyzer — "Film Room"](01-jumpshot-analyzer/) | Python · OpenCV · MediaPipe · PyTorch | Pose-track your jumper, interactive frame scrubber with ghost reference, mechanics radar, session history, drill-based coaching plan, batch mode |
| 02 | [Lineup Intelligence — "Tactical Board"](02-lineup-intelligence/) | Python · Pandas · scikit-learn · XGBoost | Tap-to-build lineup picker scored over every C(n,5), pairwise synergy heatmap, what-if swaps, starters/bench/closing rotation, opponent + trade scenarios, 3-model lab |
| 03 | [Game Predictor — "NBA//DESK"](03-game-predictor/) | Python · Pandas · scikit-learn · XGBoost | Leak-free rolling form + Elo → calibrated win probabilities; any-date slate picker, power rankings, Monte Carlo playoff odds, team drill-downs, upset detector, 3-model lab |

## Quick start

```bash
pip install -r requirements.txt          # or per-project requirements
./run.sh 1   # jumpshot analyzer demo
./run.sh 2   # lineup intelligence demo
./run.sh 3   # game predictor demo
./run.sh all
```

Each demo writes a self-contained interactive `output/report.html` (no CDN, works from file://) inside its project folder. Each project has its own visual identity: amber Film Room, lime/violet Tactical Board, magenta/green terminal.

`NO_OPEN=1 ./run.sh all` skips auto-opening the reports (headless/CI).

## Real data

`stats.nba.com` was not reachable from the build environment, so every project ships with a seeded sample
generator and **no real team or player names**. To use real data, see each README's "Real data" section
(`nba_api` loaders are included and fail gracefully). Fit scores and win probabilities are for learning and
demonstration only — nothing here is a guarantee or betting advice.

## Layout

```
shared/theme.css          original shared tokens (each project now ships its own distinct theme in output/)
01-jumpshot-analyzer/
02-lineup-intelligence/
03-game-predictor/
run.sh                    one-command launcher (wrap it in a Desktop .command/.app for one-click)
```

## Website / GitHub Pages

The three demos plus a hub page ship as one static, installable (PWA) site in `site/`.

```bash
python build_site.py              # runs each app's `cli.py demo --site site/<app>`, injects PWA head tags,
                                  # fills hub KPIs from output JSON, writes site/index.html
python build_site.py --skip-demos # only rebuild the hub from site/index.template.html
```

The script is idempotent and re-runnable; a failing per-app demo is reported but does not fail the build
(the hub is written with "—" for any missing KPI). `site/_head_snippet.html` holds the PWA head tags
(`../`-relative) that get injected into each `site/<app>/index.html` if the page has no `rel="manifest"`.

**Publishing.** The built `site/` is committed; CI does not run Python. `.github/workflows/pages.yml`
uploads `site/` with `actions/upload-pages-artifact` and deploys with `actions/deploy-pages` on every push to
`main`. One-time setup: repo **Settings → Pages → Build and deployment → Source: GitHub Actions**.
The live URL is `https://<user>.github.io/nba-projects/` (e.g. `https://shaincarrasco.github.io/nba-projects/`);
all paths are `./`-relative so the site also works from `file://` and under any sub-path.

**Add to home screen.** iPhone/iPad: open the URL in Safari → Share → *Add to Home Screen*.
Android: open in Chrome → ⋮ menu → *Install app* (or *Add to Home screen*). After the first visit the
service worker (`site/sw.js`, cache `np-v1`) serves the site offline; bump the cache name when shipping
breaking changes.

Site layout:

```
site/index.template.html   hub template ({{TOKEN}} placeholders) -> site/index.html (generated)
site/assets/shell.css|js   shared dark shell: tokens, top bar, app switcher, SW registration
site/manifest.webmanifest  PWA manifest (start_url "./", standalone, #0b0d12)
site/sw.js                 cache-first service worker
site/icons/                PNG icons (192/512 + maskable, apple-touch-icon 180); make_icons.py regenerates
site/jumpshot|lineup|predictor/index.html   per-app demos, written by each project's cli.py
site/404.html, site/.nojekyll
```
