"""
Render the Game Predictor page ("Games") on the shared app foundation
(../site/assets/app.css + app-shell.js, app-polish skill).

Two outputs from the same body:
  * output/report.html          standalone: app.css + app-shell.js inlined, opens from file://
  * <site>/predictor/index.html references ../assets/app.css and ../assets/app-shell.js
                                (python cli.py report --site ../site/predictor)

Screen (mobile first, Apple Sports as the reference):
  large title "Games" + one summary line -> date strip (day pills; calendar sheet for far jumps;
  arrow keys) -> the day's games as row cards (away over home, win chance per team, thin two-tone
  bar, status: tip time / Live / Final + score + "Picked right"/"Missed") -> Power rankings (top 10,
  See all sheet) -> Playoff odds (East/West segmented) -> Biggest upsets -> footer note.
  Sheets: game ("why" in plain English, links to both teams), team (record, splits, rating chart,
  projected wins, schedule), all rankings, date picker, About this model (accuracy vs baselines,
  calibration, month by month, confidence tiers, inputs, method, data source).

Live scores: the page fetches today's ESPN scoreboard in the browser and overlays status and score
on matching games. Not betting advice. Nothing here is guaranteed.
"""
from __future__ import annotations

import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"
SITE = ROOT.parent / "site"
SITE_ASSETS = SITE / "assets"
DISCLAIMER = "Not betting advice. Nothing here is guaranteed."
CHEV = '<svg class="chev" viewBox="0 0 8 13" aria-hidden="true"><path d="M1.5 1.5 6.5 6.5 1.5 11.5" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>'


def esc(s) -> str:
    return html.escape(str(s))


def _round(o, nd: int = 3):
    """Recursively round floats (trims the embedded JSON)."""
    if isinstance(o, float):
        return round(o, nd)
    if isinstance(o, dict):
        return {k: _round(v, nd) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_round(v, nd) for v in o]
    return o


def nick(name: str) -> str:
    """'Golden State Warriors' -> 'Warriors' (team nickname for sentences)."""
    if name.endswith("Trail Blazers"):
        return "Trail Blazers"
    parts = name.split()
    return parts[-1] if len(parts) > 1 else name


def pct(p: float) -> str:
    if p >= 0.995:
        return ">99%"
    if p < 0.005:
        return "<1%"
    return f"{p:.0%}"


def short_date(d: str) -> str:
    dt = datetime.strptime(d, "%Y-%m-%d")
    return f"{dt:%b} {dt.day}"


def month_label(m: str) -> str:
    dt = datetime.strptime(m + "-01", "%Y-%m-%d")
    return f"{dt:%b}"


# ---- plain-English "why" lines from the model's top drivers ------------------
_NUM = r"([+-]?\.?\d+(?:\.\d+)?)"


def why_lines(g: dict, names: dict) -> list[list[str]]:
    """Turn driver strings like 'Elo +435 ↑' (values are home minus away) into
    [category, sentence] pairs a fan can read. Unknown drivers are skipped."""
    H, A = nick(names.get(g["h"], g["h"])), nick(names.get(g["a"], g["a"]))
    out: list[list[str]] = []
    seen = set()

    def side(v):  # team the fact favours when v = home - away
        return H if v >= 0 else A

    for s in g.get("drv", []):
        s = s.rstrip(" ↑↓").strip()
        m = re.match(r"Elo p\(home\) (\d+)%", s)
        if m:
            p = int(m.group(1)) / 100
            cat, txt = "Team strength", (f"On ratings alone the {H} win {p:.0%} of the time" if p >= .5 else f"On ratings alone the {A} win {1 - p:.0%} of the time")
        elif (m := re.match(r"Elo " + _NUM, s)):
            v = float(m.group(1))
            cat, txt = "Team strength", f"The {side(v)} are rated {abs(v):.0f} points stronger"
        elif (m := re.match(r"season win% " + _NUM, s)):
            v = float(m.group(1))
            cat, txt = "Season record", f"The {side(v)} have {'a much' if abs(v) >= .25 else 'the'} better record this season"
        elif (m := re.match(r"L10 margin " + _NUM, s)):
            v = float(m.group(1))
            cat, txt = "Recent form", f"The {side(v)} have been {abs(v):.0f} points a game better over the last 10"
        elif (m := re.match(r"L(5|10|20) form " + _NUM, s)):
            v = float(m.group(2))
            cat, txt = "Recent form", f"The {side(v)} have won more of their last {m.group(1)}"
        elif (m := re.match(r"L10 defence ([+-])(\d+(?:\.\d+)?)", s)):
            # formatted with the sign flipped: '-' means home allows more (away defends better)
            better = A if m.group(1) == "-" else H
            cat, txt = "Recent defence", f"The {better} have allowed {float(m.group(2)):.0f} fewer points a game lately"
        elif (m := re.match(r"L10 offence " + _NUM, s)):
            v = float(m.group(1))
            cat, txt = "Recent offence", f"The {side(v)} have scored {abs(v):.0f} more points a game lately"
        elif (m := re.match(r"home/away split " + _NUM, s)):
            v = float(m.group(1))
            cat, txt = "Home and away", (f"The {H} have been better at home than the {A} on the road" if v >= 0
                                         else f"The {A} have been better on the road than the {H} at home")
        elif (m := re.match(r"streak " + _NUM, s)):
            v = float(m.group(1))
            cat, txt = "Momentum", f"The {side(v)} are on the better run of results"
        elif (m := re.match(r"H2H (\d+)%", s)):
            p = int(m.group(1)) / 100
            cat, txt = "Head to head", (f"The {H} have won {p:.0%} of recent meetings" if p >= .5 else f"The {A} have won {1 - p:.0%} of recent meetings")
        else:
            continue  # rest is always added below from the raw rest days
        if cat in seen and cat != "Recent form":
            continue
        seen.add(cat)
        out.append([cat, txt])
    rh, ra = g.get("rh"), g.get("ra")
    if rh is not None and ra is not None:
        if rh == 1 and ra == 1:
            out.append(["Rest", "Both teams played yesterday"])
        elif rh == 1:
            out.append(["Rest", f"The {H} played yesterday, the {A} are rested"])
        elif ra == 1:
            out.append(["Rest", f"The {A} played yesterday, the {H} are rested"])
        elif min(rh, ra) < 7:
            out.append(["Rest", "Both teams are rested" if rh == ra else f"Both teams are rested ({A} {ra} days, {H} {rh} days since last game)"])
    return out


# ---- SVG charts (CSS variables, so they follow light/dark) -------------------
def walk_svg(walk: list[dict]) -> str:
    W, H, L, R, T, B = 360, 190, 30, 64, 12, 24
    pw, ph = W - L - R, H - T - B
    n = len(walk)
    lo, hi = 0.45, 0.80
    xs = [L + (i + 0.5) * pw / n for i in range(n)]
    sy = lambda v: T + (1 - (min(max(v, lo), hi) - lo) / (hi - lo)) * ph
    p = [f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" aria-label="Share of games picked right, month by month">']
    for v in (0.5, 0.6, 0.7):
        p.append(f'<line class="grid" x1="{L}" y1="{sy(v):.1f}" x2="{W - R + 4}" y2="{sy(v):.1f}"/><text x="{L - 6}" y="{sy(v) + 4:.1f}" text-anchor="end">{v:.0%}</text>')
    series = (("home_accuracy", "Home team", False), ("elo_accuracy", "Ratings only", False), ("xgb_accuracy", "This model", True))
    labels = []
    for key, label, main in series:
        d = " ".join(f"{'M' if i == 0 else 'L'}{xs[i]:.1f},{sy(w[key]):.1f}" for i, w in enumerate(walk))
        p.append(f'<path class="series{"" if main else " muted"}" d="{d}"/>')
        if main:
            for i, w in enumerate(walk):
                p.append(f'<circle cx="{xs[i]:.1f}" cy="{sy(w[key]):.1f}" r="3" style="fill:var(--brand)"><title>{w["month"]}: {w[key]:.0%} of {w["n"]} games</title></circle>')
        labels.append([sy(walk[-1][key]), label, main])
    labels.sort()
    for k in range(1, len(labels)):  # keep direct labels from colliding
        if labels[k][0] - labels[k - 1][0] < 13:
            labels[k][0] = labels[k - 1][0] + 13
    for y, label, main in labels:
        p.append(f'<text x="{xs[-1] + 8:.1f}" y="{y + 4:.1f}" class="{"lab-main" if main else ""}">{label}</text>')
    prev = None
    for i, w in enumerate(walk):
        mo = month_label(w["month"])
        yr = w["month"][:4]
        txt = mo if yr == prev else f"{mo} {yr[2:]}"
        prev = yr
        if i % 2 == 0 or i == n - 1:
            p.append(f'<text x="{xs[i]:.1f}" y="{H - 6}" text-anchor="middle">{txt}</text>')
    p.append("</svg>")
    return "".join(p)


def calibration_svg(table_xgb: list[dict], table_elo: list[dict]) -> str:
    W, H, L, B, T, R = 360, 330, 36, 36, 10, 12
    pw, ph = W - L - R, H - T - B
    sx = lambda v: L + v * pw
    sy = lambda v: T + (1 - v) * ph
    p = [f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" aria-label="Predicted chance against how often it happened">']
    for v in (0, .25, .5, .75, 1):
        p.append(f'<line class="grid" x1="{sx(0)}" y1="{sy(v):.1f}" x2="{sx(1)}" y2="{sy(v):.1f}"/>')
        p.append(f'<text x="{L - 6}" y="{sy(v) + 4:.1f}" text-anchor="end">{v:.0%}</text><text x="{sx(v):.1f}" y="{T + ph + 16}" text-anchor="middle">{v:.0%}</text>')
    p.append(f'<line x1="{sx(0)}" y1="{sy(0)}" x2="{sx(1)}" y2="{sy(1)}" style="stroke:var(--label-3)" stroke-dasharray="4 4"/>')
    p.append(f'<text x="{sx(.86):.1f}" y="{sy(.94):.1f}" text-anchor="end">Perfect</text>')
    maxn = max([r["n"] for r in table_xgb] + [1])
    for table, main in ((table_elo, False), (table_xgb, True)):
        pts = [(sx(r["pred_mean"]), sy(r["actual_rate"]), r["n"]) for r in table if r["n"]]
        if len(pts) > 1:
            p.append('<path class="series' + ("" if main else " muted") + '" d="' + " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y, _) in enumerate(pts)) + '"/>')
        if main:
            for x, y, n in pts:
                p.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{2.5 + 5 * (n / maxn) ** 0.5:.1f}" style="fill:var(--brand);fill-opacity:.85"><title>{n} games</title></circle>')
    p.append(f'<text x="{L + pw / 2}" y="{H - 4}" text-anchor="middle">Predicted chance for the home team</text></svg>')
    return "".join(p)


# ---- static sections (server-rendered, taps handled by the page script) ------
def form_dots(l10: str) -> str:
    w = l10.count("W")
    return (f'<span class="form" role="img" aria-label="Last {len(l10)}: {w} wins, {len(l10) - w} losses">'
            + "".join(f'<i class="{"w" if c == "W" else "l"}"></i>' for c in l10) + "</span>")


def spark_svg(pts: list[int], up: bool) -> str:
    if len(pts) < 2:
        return '<span class="spark"></span>'
    W, H = 56, 24
    lo, hi = min(pts), max(pts)
    rng = max(hi - lo, 1)
    xy = [(i / (len(pts) - 1) * (W - 4) + 2, H - 3 - (v - lo) / rng * (H - 6)) for i, v in enumerate(pts)]
    line = " ".join(f"{x:.1f},{y:.1f}" for x, y in xy)
    x, y = xy[-1]
    return (f'<svg class="spark" viewBox="0 0 {W} {H}" aria-hidden="true"><polyline points="{line}"/>'
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.5" class="{"pos" if up else "neg"}"/></svg>')


def rank_row(r: dict) -> str:
    dr = r["d_rank"]
    mv = (f'<span class="mv pos" aria-label="up {dr}">&#9650;{dr}</span>' if dr > 0 else
          f'<span class="mv neg" aria-label="down {-dr}">&#9660;{-dr}</span>' if dr < 0 else "")
    return (f'<button type="button" class="row rk" data-team="{esc(r["team"])}">'
            f'<span class="rk-n num">{r["rank"]}</span>'
            f'<span class="grow"><span class="title">{esc(r["name"])}</span>'
            f'<span class="sub"><span class="num">{r["w"]}&ndash;{r["l"]}</span>{mv}{form_dots(r["l10"])}</span></span>'
            f'{spark_svg(r["spark"][-20:], r["d_elo_30"] >= 0)}{CHEV}</button>')


def odds_list(sim: dict, conf: str, visible: bool) -> str:
    rows = sorted([r for r in sim["teams"] if r["conf"] == conf], key=lambda r: (-r["playoff"], -r["exp_w"]))
    out = []
    for k, r in enumerate(rows):
        ew = round(r["exp_w"])
        if sim["season_complete"]:
            sub = f'Finished {r["final_w"]}&ndash;{r["final_l"]}, {"made it" if r["made_playoffs"] else "missed out"}'
        else:
            sub = f'Projected {ew}&ndash;{82 - ew}'
        out.append(f'<button type="button" class="row po{" extra" if k >= 8 else ""}" data-team="{esc(r["team"])}">'
                   f'<span class="grow"><span class="title">{esc(r["name"])}</span><span class="sub num">{sub}</span></span>'
                   f'<span class="meter" aria-hidden="true"><i style="width:{r["playoff"] * 100:.1f}%"></i></span>'
                   f'<span class="value num">{pct(r["playoff"])}</span>{CHEV}</button>')
    more = (f'<button type="button" class="row show-all" data-act="show-all"><span class="grow">Show all {len(rows)}</span></button>'
            if len(rows) > 8 else "")
    return f'<div class="list odds" data-conf="{esc(conf)}"{"" if visible else " hidden"}>{"".join(out)}{more}</div>'


def upset_rows(ups: list[dict], names: dict) -> str:
    out = []
    for u in ups[:5]:
        hp, ap = (int(x) for x in u["score"].split("-"))  # score is home-away
        ws, ls = (hp, ap) if u["home_won"] else (ap, hp)
        out.append(f'<button type="button" class="row" data-game="{esc(u["d"])}|{esc(u["winner"])}|{esc(u["loser"])}">'
                   f'<span class="grow"><span class="title">{esc(nick(names.get(u["winner"], u["winner"])))} beat {esc(nick(names.get(u["loser"], u["loser"])))}</span>'
                   f'<span class="sub num">{short_date(u["d"])}, {ws}&ndash;{ls}</span></span>'
                   f'<span class="value num">{pct(u["p_winner"])}</span>{CHEV}</button>')
    return "".join(out)


FEATURE_NAMES = {
    "elo_diff": "Team rating gap", "elo_prob": "Odds from ratings", "rest_diff": "Days of rest",
    "home_rest": "Home team rest", "away_rest": "Away team rest", "b2b_diff": "Back-to-backs",
    "home_b2b": "Home team back-to-back", "away_b2b": "Away team back-to-back", "wp5_diff": "Last 5 games",
    "wp10_diff": "Last 10 games", "wp20_diff": "Last 20 games", "pdiff10_diff": "Point margin, last 10",
    "off10_diff": "Scoring, last 10", "def10_diff": "Defence, last 10", "season_wp_diff": "Season record",
    "split_wp_diff": "Home and road records", "streak_diff": "Win streaks", "gp_diff": "Games played",
    "h2h_home_wp": "Head-to-head record", "h2h_n": "Head-to-head meetings",
}


def about_html(met: dict, sim: dict, tiers: list[dict], live: bool, updated: str, n_final: int) -> str:
    m, cal, walk, d = met["models"], met["calibration"], met["walk_forward"], met["data"]
    prior, pooled = met.get("prior_holdout"), met.get("pooled")
    small = d.get("small_sample")

    def acc_rows(models: dict, n: int) -> str:
        rows = [("This model", models["xgb"]["accuracy"], True),
                ("Team ratings only", models["elo_logit"]["accuracy"], False),
                ("Always pick the home team", models["home_baseline"]["accuracy"], False)]
        return "".join(f'<div class="row"><span class="grow title{" b" if main else ""}">{t}</span>'
                       f'<span class="value num{" strong" if main else ""}">{a:.1%}</span></div>' for t, a, main in rows)

    this_season = (f'<div class="sub-h">This season so far</div><div class="list">{acc_rows(m, d["n_test"])}</div>'
                   f'<p class="section-foot">{d["n_test"]} games{". Early in the season a few results swing these numbers by several points" if small else ""}.</p>')
    last_season = ""
    if prior:
        last_season = (f'<div class="sub-h">Last full season ({esc(prior["test_season"])})</div><div class="list">{acc_rows(prior["models"], prior["n_test"])}</div>'
                       f'<p class="section-foot">{prior["n_test"]:,} games the model never saw while training.</p>')
    rel_x, rel_e, rel_n = cal["reliability_xgb"], cal["reliability_elo"], d["n_test"]
    if pooled:
        rel_x, rel_e, rel_n = pooled["reliability_xgb"], pooled["reliability_elo"], pooled["n"]

    tier_rows = []
    for t in tiers:
        if not t["n"]:
            continue
        lo_hi = t["tier"].replace("%-", "&ndash;")
        tier_rows.append(f'<div class="row"><span class="grow"><span class="title">{lo_hi} sure</span><span class="sub">{t["n"]} games</span></span>'
                         f'<span class="value num">won {t["actual"]:.0%}</span></div>')

    imp = met["feature_importance"][:6]
    mx = imp[0]["gain_share"] if imp else 1
    imp_rows = "".join(f'<div class="row"><span class="grow title">{esc(FEATURE_NAMES.get(r["feature"], r["feature"]))}</span>'
                       f'<span class="meter" aria-hidden="true"><i style="width:{r["gain_share"] / mx * 100:.0f}%"></i></span>'
                       f'<span class="value num">{r["gain_share"]:.0%}</span></div>' for r in imp)

    def trow(name, r, best=False):
        return (f'<tr{" class=b" if best else ""}><td>{esc(name)}</td><td class="r">{r["accuracy"]:.1%}</td><td class="r">{r["log_loss"]:.3f}</td>'
                f'<td class="r">{r["brier"]:.3f}</td><td class="r">{r["roc_auc"]:.3f}</td></tr>')
    detail = "".join([trow("Always home", m["home_baseline"]), trow("Elo logistic", m["elo_logit"]), trow("Hist. gradient boosting", m["hgb"]),
                      trow("XGBoost, raw", m["xgb_raw"]), trow(f"XGBoost + {cal['chosen']}", m["xgb"], True)])

    src = (f'Real NBA games from ESPN&rsquo;s public scoreboard (unofficial, can change without notice), {n_final:,} finished games, updated {esc(updated)}.'
           if live else "A synthetic sample league (30 fictional teams), used when ESPN can&rsquo;t be reached.")
    sim_note = (f' {esc(sim["schedule_note"])}' if sim.get("schedule_note") else "")
    return f"""
  <h3 id="sheet-title">About this model</h3>
  <p class="lead">It estimates each team&rsquo;s chance to win before tip-off, using team ratings, recent results and rest. It was trained on {esc(" + ".join(d["train_seasons"]))} and is checked on games it has never seen.</p>
  {this_season}
  {last_season}
  <div class="sub-h">Are the percentages honest?</div>
  <p class="body-s">When the model says 70%, that team should win about 7 times in 10. Orange dots close to the dashed line mean it does; the grey line is team ratings alone. Bigger dots cover more games ({rel_n:,} in total).</p>
  {calibration_svg(rel_x, rel_e)}
  <div class="sub-h">Month by month</div>
  <p class="body-s">Each month the model is retrained on everything before it, then scored on that month.</p>
  {walk_svg(walk)}
  <div class="sub-h">How sure vs how often right</div>
  <div class="list">{"".join(tier_rows)}</div>
  <div class="sub-h">What it looks at most</div>
  <div class="list">{imp_rows}</div>
  <div class="sub-h">Detailed scores</div>
  <div class="tbl-wrap"><table class="table"><thead><tr><th>Model</th><th class="r">Accuracy</th><th class="r">Log loss</th><th class="r">Brier</th><th class="r">AUC</th></tr></thead><tbody>{detail}</tbody></table></div>
  <p class="section-foot">This season, {d["n_test"]} games. Lower log loss and Brier are better.</p>
  <div class="sub-h">How it works</div>
  <p class="body-s">Features use only games played before each date, so nothing from game day leaks in. The team rating is an Elo rating (K=20, 80 points for home court, 25% pulled back to average each summer). The learner is XGBoost with {esc(cal["chosen"])} calibration; on its own the rating gap explains most of what can be predicted, which is why the two land so close.</p>
  <p class="body-s">Playoff odds come from {sim["n_sims"]:,} simulated seasons over the published schedule, using ratings with home court; the top 6 in each conference count as making the playoffs (the play-in isn&rsquo;t modelled).{sim_note}</p>
  <p class="body-s">No player injuries, travel or betting lines are used.</p>
  <div class="sub-h">Data</div>
  <p class="body-s">{src}</p>
  <p class="body-s strong">{esc(DISCLAIMER)}</p>"""


# ---- page CSS (page-only; app.css carries the system) ------------------------
CSS = r"""
.summary{font:400 var(--t-sub)/1.4 var(--font);color:var(--label-2);margin:0}
.summary a{white-space:nowrap}
.linkbtn{appearance:none;border:0;background:none;padding:0;font:inherit;color:var(--brand-text);cursor:pointer}
.linkbtn:active{opacity:.6}
.section-h .linkbtn{min-height:44px;padding:0 8px;margin:-12px -8px}
html{scroll-padding:calc(var(--topbar-h) + var(--sat) + 8px) 0 calc(var(--tabbar-h) + var(--sab) + 8px)}

/* Date strip */
.strip{position:relative;margin:16px calc(-1 * var(--margin)) 0}
.days{display:flex;align-items:center;gap:8px;overflow-x:auto;scroll-snap-type:x proximity;padding:4px var(--margin) 6px;scrollbar-width:none;-webkit-overflow-scrolling:touch;overscroll-behavior-x:contain}
.days::-webkit-scrollbar{display:none}
.day{flex:none;appearance:none;border:0;height:36px;min-width:44px;padding:0 14px;border-radius:var(--r-pill);background:var(--surface);color:var(--label);box-shadow:var(--shadow-1);
  font:600 var(--t-sub)/1 var(--font);font-variant-numeric:tabular-nums;scroll-snap-align:center;cursor:pointer;user-select:none;transition:background-color 160ms var(--ease),color 160ms var(--ease),transform 100ms var(--ease)}
.day .wd{font-weight:400;color:var(--label-2);margin-right:4px}
.day.on{background:var(--brand);color:var(--on-brand)}
.day.on .wd{color:inherit;opacity:.85}
.day:active{transform:scale(.96)}
.mon{flex:none;font:600 var(--t-foot)/1 var(--font);color:var(--label-2);padding:0 2px 0 6px}
.strip .nav{display:none}
@media (min-width:900px){
  .strip{margin:20px 0 0;display:flex;align-items:center;gap:4px}
  .days{flex:1;padding:4px 2px 6px;-webkit-mask-image:linear-gradient(90deg,transparent,#000 24px,#000 calc(100% - 24px),transparent);mask-image:linear-gradient(90deg,transparent,#000 24px,#000 calc(100% - 24px),transparent);padding-left:24px;padding-right:24px}
  .strip .nav{display:grid}
  .strip .nav[disabled]{opacity:.3;pointer-events:none}
}
@media (hover:hover){.day:not(.on):hover{background:var(--surface-2)}}

/* Slate */
.slate-h{margin-top:24px}
.slate-h .meta{font:400 var(--t-foot)/1.2 var(--font);color:var(--label-2);white-space:nowrap}
.games{display:grid;grid-template-columns:minmax(0,1fr);gap:10px}
.game{display:block;width:100%;appearance:none;border:0;text-align:left;color:inherit;font:inherit;background:var(--surface);border-radius:var(--r-card);box-shadow:var(--shadow-1);
  padding:12px var(--s4) 14px;cursor:pointer;user-select:none;transition:transform 120ms var(--ease),background-color 120ms var(--ease)}
.game:active{transform:scale(.985);background:var(--surface-2)}
@media (hover:hover){.game:hover{background:var(--surface-2)}}
.g-top{display:flex;align-items:center;gap:8px;min-height:22px;margin-bottom:6px;font:400 var(--t-foot)/1 var(--font);color:var(--label-2)}
.g-top .clock{font-variant-numeric:tabular-nums;color:var(--label)}
.g-top .badge:last-child{margin-left:auto}
.g-team{display:flex;align-items:center;gap:10px;min-height:34px}
.g-ab{flex:none;width:52px;font:700 var(--t-body)/1 var(--font-display);letter-spacing:-.01em}
.g-nm{flex:1;min-width:0;font:400 var(--t-sub)/1.2 var(--font);color:var(--label-2);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.g-pct{flex:none;min-width:44px;text-align:right;font:400 var(--t-sub)/1 var(--font);color:var(--label-2);font-variant-numeric:tabular-nums}
.g-pct.fav{color:var(--label);font-weight:600}
.g-pct.big{font-size:var(--t-body)}
.g-sc{flex:none;min-width:40px;text-align:right;font:600 var(--t-2)/1 var(--font-display);font-variant-numeric:tabular-nums;letter-spacing:-.01em}
.g-sc.lose{color:var(--label-2);font-weight:500}
.wbar{display:flex;gap:2px;height:4px;margin-top:10px}
.wbar i{display:block;height:100%;border-radius:2px;background:var(--fill-2);transition:flex-grow var(--dur) var(--ease)}
.wbar i.fav{background:var(--brand)}
.empty-day{background:var(--surface);border-radius:var(--r-card);box-shadow:var(--shadow-1)}

/* Lists */
.list .row .title.b,.value.strong,.strong{font-weight:600;color:var(--label)}
.rk-n{flex:none;width:22px;text-align:center;font:600 var(--t-sub)/1 var(--font);color:var(--label-2)}
.row .title{display:block}
.row .sub{display:flex;align-items:center;gap:8px}
.mv{font:600 var(--t-cap)/1 var(--font)}
.form{display:inline-flex;gap:3px}
.form i{width:6px;height:6px;border-radius:3px;background:var(--fill-2)}
.form i.w{background:var(--pos)}
svg.spark{flex:none;width:56px;height:24px;overflow:visible}
svg.spark polyline{fill:none;stroke:var(--label-3);stroke-width:1.5;stroke-linejoin:round;stroke-linecap:round}
svg.spark circle.pos{fill:var(--pos)} svg.spark circle.neg{fill:var(--neg)}
.row .meter{flex:none;width:64px}
.meter + .value{min-width:40px;text-align:right}
.odds .row.extra{display:none}
.odds.all .row.extra{display:flex}
.odds.all .show-all{display:none}
.show-all{color:var(--brand-text)}
.seg{margin-bottom:12px}
@media (min-width:900px){.side-col .seg{max-width:none}}

/* Desktop layout */
.main-cols{margin-top:0;grid-template-columns:minmax(0,1fr)}
.side-col > .section:first-child{margin-top:24px}
@media (min-width:900px){
  .main-cols{grid-template-columns:minmax(0,1.25fr) minmax(0,1fr);gap:32px;align-items:start}
}

/* Sheet internals */
.sheet[inert]{visibility:hidden;transition:transform 320ms var(--ease),visibility 0s 320ms}
@media (min-width:900px){.sheet[inert]{transition:opacity var(--dur) var(--ease),transform var(--dur) var(--ease),visibility 0s var(--dur)}}
.sh-bar{position:sticky;top:-8px;z-index:2;display:flex;align-items:center;min-height:44px;margin:-8px calc(-1 * var(--margin)) 0;padding:8px calc(var(--margin) - 8px) 0;background:var(--surface)}
.sh-bar .grabber{position:absolute;left:50%;top:6px;translate:-50% 0;margin:0}
.sh-bar .done{margin-left:auto;font-weight:600}
.sh-bar [hidden]{display:none}
.sh-bar .btn.plain{height:44px}
.sheet .lead{font:400 var(--t-sub)/1.45 var(--font);color:var(--label-2);margin:0 0 8px}
.sheet .body-s{font:400 var(--t-sub)/1.45 var(--font);color:var(--label-2);margin:0 0 10px}
.sheet .sub-h{font:600 var(--t-body)/1.3 var(--font);margin:28px 0 8px}
.sheet .list{background:var(--surface-2);box-shadow:none}
.sheet .list .row + .row::before{background:var(--separator)}
.sheet a.row:active,.sheet button.row:active{background:var(--surface-3)}
.sheet svg.chart{margin:4px 0 0}
.chart text{font-size:12px}
.chart .lab-main{fill:var(--brand-text);font-weight:600}
@media (min-width:900px){.sheet svg.chart{max-width:420px}}
.tbl-wrap{overflow-x:auto;margin:0 calc(-1 * var(--margin));padding:0 var(--margin)}
.tbl-wrap .table{min-width:340px;font-size:var(--t-foot)}
.tbl-wrap .table th,.tbl-wrap .table td{padding:10px 6px}
.tbl-wrap .table th:first-child,.tbl-wrap .table td:first-child{padding-left:0}
.tbl-wrap .table tr.b td{font-weight:600}
.sheet h3 small{display:block;font:400 var(--t-sub)/1.3 var(--font);color:var(--label-2);margin-top:4px}
.sheet .stats{margin:16px 0 4px}
.sheet .stats.three{grid-template-columns:repeat(3,1fr)}
.sheet .stat .v{font-size:var(--t-1)}
.matchup{display:flex;align-items:flex-end;justify-content:space-between;gap:16px;margin:16px 0 0}
.matchup .side{min-width:0}
.matchup .side.r{text-align:right}
.matchup .v{font:600 var(--t-large)/1 var(--font-display);letter-spacing:-.02em;font-variant-numeric:tabular-nums}
.matchup .side:not(.fav) .v{color:var(--label-2);font-weight:500}
.matchup .c{font:400 var(--t-foot)/1.3 var(--font);color:var(--label-2);margin-top:6px}
.sheet .wbar{height:6px;margin:12px 0 4px}
.why .row{align-items:flex-start}
.why .title{white-space:normal}
.rng{margin-top:4px}
.date-sheet .field{margin:4px 0 16px}
"""


JS = r"""
(function(){
  const D = JSON.parse(document.getElementById('gp-data').textContent);
  const SP = D.sp, TP = D.teams, RK = {}, SIM = {};
  D.rankings.forEach(r => RK[r.team] = r);
  D.sim.teams.forEach(r => SIM[r.team] = r);
  const $ = s => document.querySelector(s);
  const pct = p => p >= .995 ? '>99%' : p < .005 ? '<1%' : Math.round(p * 100) + '%';
  const nm = c => (TP[c] && TP[c].name) || c;
  const nick = c => { const n = nm(c); if (/Trail Blazers$/.test(n)) return 'Trail Blazers'; const p = n.split(' '); return p.length > 1 ? p[p.length - 1] : n; };
  const CHEV = '<svg class="chev" viewBox="0 0 8 13" aria-hidden="true"><path d="M1.5 1.5 6.5 6.5 1.5 11.5" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>';
  const dt = d => new Date(d + 'T12:00:00Z');
  const fmt = (d, o) => new Intl.DateTimeFormat('en-US', Object.assign({ timeZone: 'UTC' }, o)).format(dt(d));
  const dash = '–';
  let i = SP.dates.indexOf(D.default_date); if (i < 0) i = SP.dates.length - 1;

  // ---------------- date strip ----------------
  const days = $('#days');
  function buildStrip(){
    let h = '', prevMon = null;
    SP.dates.forEach((d, k) => {
      const mon = fmt(d, { month: 'short' });
      if (mon !== prevMon && k > 0) h += `<span class="mon" aria-hidden="true">${mon}</span>`;
      if (k === 0) h += `<span class="mon" aria-hidden="true">${mon}</span>`;
      prevMon = mon;
      const label = d === D.today ? 'Today' : `<span class="wd">${fmt(d, { weekday: 'short' })}</span>${fmt(d, { day: 'numeric' })}`;
      const aria = fmt(d, { weekday: 'long', month: 'long', day: 'numeric' }) + (d === D.today ? ', today' : '');
      h += `<button type="button" class="day" role="tab" data-i="${k}" aria-label="${aria}">${label}</button>`;
    });
    days.innerHTML = h;
  }
  function syncStrip(smooth){
    days.querySelectorAll('.day').forEach(b => { const on = +b.dataset.i === i; b.classList.toggle('on', on); b.setAttribute('aria-selected', on); b.tabIndex = on ? 0 : -1; });
    const el = days.querySelector('.day.on');
    if (el) {
      const left = el.offsetLeft - (days.clientWidth - el.offsetWidth) / 2;
      days.scrollTo({ left, behavior: smooth && !matchMedia('(prefers-reduced-motion: reduce)').matches ? 'smooth' : 'auto' });
    }
    const pv = $('#day-prev'), nx = $('#day-next');
    if (pv) pv.disabled = i === 0; if (nx) nx.disabled = i === SP.dates.length - 1;
  }
  function setIndex(k, smooth){
    if (k < 0 || k >= SP.dates.length || k === i) return;
    i = k; renderSlate(); syncStrip(smooth !== false);
  }

  // ---------------- live scores (ESPN, browser fetch) ----------------
  const LIVE = { map: {}, at: null, failed: false };
  const etDate = () => new Intl.DateTimeFormat('en-CA', { timeZone: 'America/New_York', year: 'numeric', month: '2-digit', day: '2-digit' }).format(new Date());
  const IN_PLAY = ['STATUS_IN_PROGRESS', 'STATUS_HALFTIME', 'STATUS_END_PERIOD'];
  function clockText(s){
    const t = (s && s.type) || {}, p = +s.period || 0;
    const per = p <= 4 ? 'Q' + p : (p === 5 ? 'OT' : (p - 4) + 'OT');
    if (t.name === 'STATUS_HALFTIME') return 'Halftime';
    if (t.name === 'STATUS_END_PERIOD') return 'End of ' + per;
    return p ? `${per} ${s.displayClock || ''}`.trim() : (t.shortDetail || '');
  }
  function ingest(events){
    const map = {};
    (events || []).forEach(e => {
      const s = e.status || {}, name = (s.type && s.type.name) || '';
      const comp = (e.competitions || [])[0]; if (!comp) return;
      const home = comp.competitors.find(c => c.homeAway === 'home'), away = comp.competitors.find(c => c.homeAway === 'away');
      if (!home || !away) return;
      const r = { hp: home.score, ap: away.score };
      if (IN_PLAY.indexOf(name) >= 0) { r.kind = 'live'; r.clock = clockText(s); }
      else if (name.indexOf('STATUS_FINAL') === 0) r.kind = 'final';
      else if (name === 'STATUS_POSTPONED' || name === 'STATUS_CANCELED') { r.kind = 'off'; r.label = name === 'STATUS_POSTPONED' ? 'Postponed' : 'Canceled'; }
      else return;
      map[e.id] = r;
    });
    LIVE.map = map;
  }
  async function fetchLive(){
    if (document.visibilityState !== 'visible') return;
    try {
      const r = await fetch(`https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates=${etDate().replace(/-/g, '')}&limit=50`, { cache: 'no-store' });
      if (!r.ok) throw new Error(r.status);
      const j = await r.json();
      ingest(j.events); LIVE.at = new Date().toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' }); LIVE.failed = false;
    } catch (e) { LIVE.failed = true; }
    if (SP.dates[i] === D.today) renderSlate();
  }

  // ---------------- slate ----------------
  function gameState(g){
    const lv = g.eid && LIVE.map[g.eid];
    if ('hp' in g) return { kind: 'final', hp: g.hp, ap: g.ap, ok: g.ok };
    if (lv && lv.kind === 'final') { const hp = +lv.hp, ap = +lv.ap; return { kind: 'final', hp, ap, ok: (hp > ap ? g.h : g.a) === g.pick }; }
    if (lv && lv.kind === 'live') return { kind: 'live', hp: lv.hp, ap: lv.ap, clock: lv.clock };
    if (lv && lv.kind === 'off') return { kind: 'off', label: lv.label };
    if (SP.dates[i] < (D.today || '')) return { kind: 'off', label: 'No result' };
    return { kind: 'pre' };
  }
  function teamLine(code, p, fav, st, score, other){
    const hasScore = st.kind === 'final' || st.kind === 'live';
    const sc = hasScore ? `<span class="g-sc${(+score < +other) ? ' lose' : ''}">${score}</span>` : '';
    return `<div class="g-team"><span class="g-ab">${code}</span><span class="g-nm">${nm(code)}</span>` +
      `<span class="g-pct num${fav ? ' fav' : ''}${hasScore ? '' : ' big'}" aria-label="${pct(p)} chance to win">${pct(p)}</span>${sc}</div>`;
  }
  function gameCard(g, k){
    const st = gameState(g), homeFav = g.ph >= .5;
    let top;
    if (st.kind === 'final') top = `<span>Final</span><span class="badge ${st.ok ? 'pos' : ''}">${st.ok ? 'Picked right' : 'Missed'}</span>`;
    else if (st.kind === 'live') top = `<span class="badge neg"><span class="dot"></span>Live</span><span class="clock">${st.clock}</span>`;
    else if (st.kind === 'off') top = `<span>${st.label}</span>`;
    else top = `<span class="clock">${g.t ? g.t + ' ET' : 'Time to be announced'}</span>`;
    return `<button type="button" class="game" data-k="${k}" aria-label="${nm(g.a)} at ${nm(g.h)}, open details">
      <div class="g-top">${top}</div>
      ${teamLine(g.a, 1 - g.ph, !homeFav, st, st.ap, st.hp)}
      ${teamLine(g.h, g.ph, homeFav, st, st.hp, st.ap)}
      <div class="wbar" aria-hidden="true"><i class="${homeFav ? '' : 'fav'}" style="flex:${(1 - g.ph).toFixed(3)} 1 0"></i><i class="${homeFav ? 'fav' : ''}" style="flex:${g.ph.toFixed(3)} 1 0"></i></div>
    </button>`;
  }
  function renderSlate(){
    const d = SP.dates[i], rows = SP.slates[d] || [], r = SP.running[d] || {};
    $('#slate-title').textContent = d === D.today ? 'Today' : fmt(d, { weekday: 'long', month: 'long', day: 'numeric' });
    let meta = rows.length + (rows.length === 1 ? ' game' : ' games');
    if (r.n) meta = `${r.hits} of ${r.n} picked right`;
    else if (d === D.today && LIVE.at) meta = 'Updated ' + LIVE.at;
    $('#slate-meta').textContent = meta;
    $('#slate').innerHTML = rows.length ? rows.map(gameCard).join('') :
      `<div class="empty empty-day"><h3>No games this day</h3><p>Pick another date to see its games.</p><button type="button" class="btn sm" data-act="dates">Choose a date</button></div>`;
  }

  // ---------------- sheet ----------------
  const sheet = $('#sheet'), scrim = $('#scrim'), body = $('#sheet-body'), back = $('#sheet-back');
  const stack = []; let lastFocus = null;
  function show(html){
    body.innerHTML = html;
    back.hidden = stack.length < 2;
    sheet.scrollTop = 0;
    const t = body.querySelector('h3'); if (t) { t.id = 'sheet-title'; }
  }
  function openSheet(render, push){
    if (!sheet.classList.contains('open')) { lastFocus = document.activeElement; stack.length = 0; }
    else if (!push) stack.length = 0;
    stack.push(render);
    show(render());
    sheet.removeAttribute('inert'); sheet.setAttribute('aria-hidden', 'false');
    sheet.classList.add('open'); scrim.classList.add('open');
    document.documentElement.style.overflow = 'hidden';
    setTimeout(() => $('#sheet-done').focus({ preventScroll: true }), 30);
  }
  function goBack(){ if (stack.length > 1) { stack.pop(); show(stack[stack.length - 1]()); } }
  function closeSheet(){
    if (!sheet.classList.contains('open')) return;
    sheet.classList.remove('open'); scrim.classList.remove('open');
    sheet.setAttribute('inert', ''); sheet.setAttribute('aria-hidden', 'true');
    sheet.style.transform = '';
    document.documentElement.style.overflow = '';
    stack.length = 0;
    if (lastFocus && lastFocus.isConnected) lastFocus.focus({ preventScroll: true });
  }
  // drag the bar down to dismiss (touch)
  (function(){
    let y0 = null, dy = 0;
    const bar = $('.sh-bar');
    bar.addEventListener('touchstart', e => { if (sheet.scrollTop <= 0 && innerWidth < 900) { y0 = e.touches[0].clientY; dy = 0; sheet.style.transition = 'none'; } }, { passive: true });
    bar.addEventListener('touchmove', e => { if (y0 == null) return; dy = Math.max(0, e.touches[0].clientY - y0); sheet.style.transform = `translateY(${dy}px)`; }, { passive: true });
    bar.addEventListener('touchend', () => { if (y0 == null) return; sheet.style.transition = ''; y0 = null; if (dy > 90) closeSheet(); else sheet.style.transform = ''; });
  })();

  // ---------------- game sheet ----------------
  function gameSheet(d, g){
    const st = gameState(g), homeFav = g.ph >= .5, fav = homeFav ? g.h : g.a;
    const when = st.kind === 'final' ? `Final, ${fmt(d, { weekday: 'long', month: 'short', day: 'numeric' })}` :
      st.kind === 'live' ? `Live now, ${st.clock}` :
      `${fmt(d, { weekday: 'long', month: 'short', day: 'numeric' })}${g.t ? ', ' + g.t + ' ET' : ''}`;
    const side = (c, p, f, r) => `<div class="side${f ? ' fav' : ''}${r ? ' r' : ''}"><div class="v num">${pct(p)}</div><div class="c">${nick(c)} win</div></div>`;
    let result = '';
    if (st.kind === 'final' || st.kind === 'live') {
      const lead = +st.hp >= +st.ap ? [g.h, st.hp, g.a, st.ap] : [g.a, st.ap, g.h, st.hp];
      result = `<div class="sub-h">${st.kind === 'final' ? 'Result' : 'Score'}</div><div class="list"><div class="row"><span class="grow title num">${nick(lead[0])} ${lead[1]}, ${nick(lead[2])} ${lead[3]}</span>` +
        (st.kind === 'final' ? `<span class="badge ${st.ok ? 'pos' : ''}">${st.ok ? 'Picked right' : 'Missed'}</span>` : `<span class="badge neg"><span class="dot"></span>Live</span>`) + `</div></div>`;
    }
    const why = (g.why || []).map(w => `<div class="row"><span class="grow"><span class="title">${w[1]}</span><span class="sub">${w[0]}</span></span></div>`).join('');
    const tl = c => `<button type="button" class="row" data-team="${c}" data-push="1"><span class="grow"><span class="title">${nm(c)}</span><span class="sub num">${TP[c] ? TP[c].w + dash + TP[c].l : ''}${RK[c] ? ', No. ' + RK[c].rank + ' in power rankings' : ''}</span></span>${CHEV}</button>`;
    return `<h3>${nick(g.a)} at ${nick(g.h)}<small>${when}</small></h3>
      <div class="matchup">${side(g.a, 1 - g.ph, !homeFav, false)}${side(g.h, g.ph, homeFav, true)}</div>
      <div class="wbar" aria-hidden="true"><i class="${homeFav ? '' : 'fav'}" style="flex:${(1 - g.ph).toFixed(3)} 1 0"></i><i class="${homeFav ? 'fav' : ''}" style="flex:${g.ph.toFixed(3)} 1 0"></i></div>
      ${result}
      <div class="sub-h">Why the model leans ${nick(fav)}</div>
      <div class="list why">${why || '<div class="row"><span class="grow title">No single factor stands out</span></div>'}</div>
      <div class="sub-h">Teams</div>
      <div class="list">${tl(g.a)}${tl(g.h)}</div>
      <p class="section-foot">${D.disclaimer}</p>`;
  }

  // ---------------- team sheet ----------------
  function ratingChart(all){
    const pts = all.filter(p => !D.today || p[0] <= D.today); if (pts.length < 2) return '';
    const W = 360, H = 150, L = 34, R = 8, T = 10, B = 22, pw = W - L - R, ph = H - T - B;
    const v = pts.map(p => p[1]); let lo = Math.min(...v, 1500) - 15, hi = Math.max(...v, 1500) + 15;
    const sx = k => L + k / Math.max(1, pts.length - 1) * pw, sy = e => T + (1 - (e - lo) / (hi - lo)) * ph;
    const line = pts.map((p, k) => `${k ? 'L' : 'M'}${sx(k).toFixed(1)},${sy(p[1]).toFixed(1)}`).join(' ');
    const y5 = sy(1500).toFixed(1);
    const last = pts[pts.length - 1];
    return `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Team rating over the season, now ${Math.round(last[1])}">
      <line class="grid" x1="${L}" y1="${y5}" x2="${W - R}" y2="${y5}" stroke-dasharray="3 3"/><text x="${W - R}" y="${(+y5 - 5).toFixed(1)}" text-anchor="end">League average</text>
      ${Math.abs(T + 4 - y5) > 14 ? `<text x="${L - 6}" y="${(T + 4).toFixed(1)}" text-anchor="end">${Math.round(hi)}</text>` : ''}${Math.abs(T + ph - y5) > 14 ? `<text x="${L - 6}" y="${(T + ph).toFixed(1)}" text-anchor="end">${Math.round(lo)}</text>` : ''}
      <path class="series" d="${line}"/><circle cx="${sx(pts.length - 1).toFixed(1)}" cy="${sy(last[1]).toFixed(1)}" r="3.5" style="fill:var(--brand)"/>
      <text x="${L}" y="${H - 4}">${fmt(pts[0][0], { month: 'short', day: 'numeric' })}</text><text x="${W - R}" y="${H - 4}" text-anchor="end">${fmt(last[0], { month: 'short', day: 'numeric' })}</text></svg>`;
  }
  function winsChart(code, s){
    const W = 360, L = 8, R = 8, pw = W - L - R;
    if (D.sim.focus === code && D.sim.focus_hist && D.sim.focus_hist.length) {
      const hist = D.sim.focus_hist, H = 120, T = 6, B = 22, ph = H - T - B, peak = Math.max(...hist.map(h => h.n)) || 1, bw = pw / hist.length;
      const ew = Math.round(s.exp_w), step = Math.max(1, Math.round(hist.length / 6));
      let g = '';
      hist.forEach((h, k) => {
        const bh = h.n / peak * ph;
        g += `<rect x="${(L + k * bw + .5).toFixed(1)}" y="${(T + ph - bh).toFixed(1)}" width="${Math.max(1, bw - 1).toFixed(1)}" height="${bh.toFixed(1)}" rx="1" style="fill:${h.wins === ew ? 'var(--brand)' : 'var(--label-3)'}"><title>${h.wins} wins: ${(h.n / D.sim.n_sims * 100).toFixed(1)}%</title></rect>`;
        if (k % step === 0) g += `<text x="${(L + (k + .5) * bw).toFixed(1)}" y="${H - 6}" text-anchor="middle">${h.wins}</text>`;
      });
      return `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="How many wins in ${D.sim.n_sims} simulated seasons">${g}</svg>`;
    }
    const H = 54, sx = w => L + Math.min(82, Math.max(0, w)) / 82 * pw, y = 22;
    return `<svg class="chart rng" viewBox="0 0 ${W} ${H}" role="img" aria-label="Likely between ${s.w_lo} and ${s.w_hi} wins">
      <rect x="${L}" y="${y - 3}" width="${pw}" height="6" rx="3" style="fill:var(--fill)"/>
      <rect x="${sx(s.w_lo).toFixed(1)}" y="${y - 3}" width="${(sx(s.w_hi) - sx(s.w_lo)).toFixed(1)}" height="6" rx="3" style="fill:var(--brand);opacity:.35"/>
      <circle cx="${sx(s.exp_w).toFixed(1)}" cy="${y}" r="6" style="fill:var(--brand);stroke:var(--surface);stroke-width:2"/>
      <text x="${sx(s.w_lo).toFixed(1)}" y="${y + 24}" text-anchor="middle">${s.w_lo}</text><text x="${sx(s.w_hi).toFixed(1)}" y="${y + 24}" text-anchor="middle">${s.w_hi}</text>
      <text x="${L}" y="${y - 10}">0</text><text x="${W - R}" y="${y - 10}" text-anchor="end">82 wins</text></svg>`;
  }
  function teamSheet(code){
    const t = TP[code], rk = RK[code], s = SIM[code]; if (!t) return '<h3>Team not found</h3>';
    const rec = t.rec, wl = a => `${a[0]}${dash}${a[1]}`;
    const l10w = rk ? rk.l10.split('').filter(c => c === 'W').length : null;
    const row = (k, v) => `<div class="row"><span class="grow title">${k}</span><span class="value num">${v}</span></div>`;
    const upW = {}, upL = {};
    t.upset_wins.forEach(u => upW[u.d] = 1); t.upset_losses.forEach(u => upL[u.d] = 1);
    const played = t.sched.filter(g => 'w' in g).slice().reverse(), next = t.sched.filter(g => !('w' in g));
    const opp = g => `${g.home ? 'vs' : 'at'} ${nick(g.opp)}`;
    const nextRows = next.slice(0, 5).map(g => `<div class="row"><span class="grow"><span class="title">${opp(g)}</span><span class="sub">${fmt(g.d, { weekday: 'short', month: 'short', day: 'numeric' })}${g.rest === 1 ? ', second night of a back-to-back' : ''}</span></span><span class="value num">${pct(g.p)} to win</span></div>`).join('');
    const pastRows = played.map(g => {
      const ok = (g.p >= .5) === g.w, tag = upW[g.d] ? '<span class="badge pos">Upset win</span>' : upL[g.d] ? '<span class="badge">Upset loss</span>' : '';
      return `<div class="row"><span class="grow"><span class="title">${opp(g)}</span><span class="sub">${fmt(g.d, { month: 'short', day: 'numeric' })}, given ${pct(g.p)}${ok ? '' : ', model missed'}</span></span>${tag}<span class="value num"><span class="${g.w ? 'pos' : 'neg'}">${g.w ? 'W' : 'L'}</span> ${g.pf}${dash}${g.pa}</span></div>`;
    }).join('');
    const proj = s ? `<div class="sub-h">Projected wins</div>
      <p class="body-s">Most likely ${Math.round(s.exp_w)} wins; 9 in 10 simulated seasons land between ${s.w_lo} and ${s.w_hi}.${s.top_seed >= .01 ? ` Chance of the No. 1 seed: ${pct(s.top_seed)}.` : ''}</p>${winsChart(code, s)}` : '';
    return `<h3>${t.name}<small>${t.conf}${rk ? `, No. ${rk.rank} in power rankings` : ''}</small></h3>
      <div class="stats three">
        <div class="stat"><div class="v num">${t.w}${dash}${t.l}</div><div class="c">Record</div></div>
        <div class="stat"><div class="v num">${s ? Math.round(s.exp_w) : '–'}</div><div class="c">Projected wins</div></div>
        <div class="stat"><div class="v num">${s ? pct(s.playoff) : '–'}</div><div class="c">Playoff odds</div></div>
      </div>
      <div class="sub-h">Splits</div>
      <div class="list">${rk && t.w + t.l > 10 ? row('Last 10', l10w + dash + (10 - l10w)) : ''}${row('Home', wl(rec.home))}${row('Away', wl(rec.away))}${row('Second night of a back-to-back', wl(rec.b2b))}${row('With a day or more of rest', wl(rec.rested))}${row('As the favourite', wl(rec.fav))}${row('As the underdog', wl(rec.dog))}${rk ? row('Point margin, last 10', `<span class="${rk.net10 >= 0 ? 'pos' : 'neg'}">${rk.net10 >= 0 ? '+' : '−'}${Math.abs(rk.net10).toFixed(1)}</span>`) : ''}${t.model_acc != null ? row('Model picked their games right', pct(t.model_acc)) : ''}</div>
      <div class="sub-h">Team rating</div>${ratingChart(t.elo)}
      ${proj}
      ${nextRows ? `<div class="sub-h">Coming up</div><div class="list">${nextRows}</div>` : ''}
      <div class="sub-h">Results</div><div class="list">${pastRows || '<div class="row"><span class="grow title muted">No games yet</span></div>'}</div>
      <p class="section-foot">${D.disclaimer}</p>`;
  }

  // ---------------- other sheets ----------------
  function dateSheet(){
    const opts = SP.dates.map((d, k) => `<option value="${k}"${k === i ? ' selected' : ''}>${fmt(d, { weekday: 'short', month: 'short', day: 'numeric' })}${d === D.today ? ' (today)' : ''}, ${SP.slates[d].length} games</option>`).join('');
    const t = SP.dates.indexOf(D.today);
    return `<div class="date-sheet"><h3>Choose a date</h3><select class="field" id="date-select" aria-label="Game date">${opts}</select>
      ${t >= 0 ? `<button type="button" class="btn primary block" data-act="today">Go to today</button>` : ''}</div>`;
  }
  const tpl = id => () => document.getElementById(id).innerHTML;

  // ---------------- events ----------------
  document.addEventListener('click', e => {
    const el = e.target.closest('[data-act],[data-team],[data-game],.game,.day');
    if (!el) return;
    if (el.classList.contains('day')) return setIndex(+el.dataset.i);
    if (el.classList.contains('game') && el.dataset.k != null) {
      const d = SP.dates[i], g = SP.slates[d][+el.dataset.k];
      return openSheet(() => gameSheet(d, g));
    }
    if (el.dataset.team) return openSheet(() => teamSheet(el.dataset.team), !!el.dataset.push);
    if (el.dataset.game) {
      const [d, w, l] = el.dataset.game.split('|');
      const g = (SP.slates[d] || []).find(x => (x.h === w && x.a === l) || (x.h === l && x.a === w));
      if (g) return openSheet(() => gameSheet(d, g));
      return openSheet(() => teamSheet(w));
    }
    const a = el.dataset.act;
    if (a === 'about') openSheet(tpl('about-tpl'));
    else if (a === 'rankings') openSheet(tpl('rankings-tpl'));
    else if (a === 'dates') openSheet(dateSheet);
    else if (a === 'close') closeSheet();
    else if (a === 'back') goBack();
    else if (a === 'today') { setIndex(SP.dates.indexOf(D.today), false); closeSheet(); }
    else if (a === 'prev') setIndex(i - 1);
    else if (a === 'next') setIndex(i + 1);
    else if (a === 'show-all') el.closest('.odds').classList.add('all');
    else if (a === 'conf') {
      document.querySelectorAll('[data-act="conf"]').forEach(b => { const on = b === el; b.classList.toggle('on', on); b.setAttribute('aria-selected', on); });
      document.querySelectorAll('.odds').forEach(l => l.hidden = l.dataset.conf !== el.dataset.conf);
    }
  });
  document.addEventListener('change', e => { if (e.target.id === 'date-select') { setIndex(+e.target.value, false); closeSheet(); } });
  scrim.addEventListener('click', closeSheet);
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape') return closeSheet();
    if (sheet.classList.contains('open')) return;
    const tag = e.target.tagName; if (tag === 'SELECT' || tag === 'INPUT' || tag === 'TEXTAREA') return;
    if (e.key === 'ArrowLeft') { e.preventDefault(); setIndex(i - 1); }
    else if (e.key === 'ArrowRight') { e.preventDefault(); setIndex(i + 1); }
  });

  buildStrip(); renderSlate();
  requestAnimationFrame(() => syncStrip(false));
  addEventListener('resize', () => syncStrip(false));
  if (D.live && typeof fetch === 'function') {
    fetchLive();
    setInterval(fetchLive, 60000);
    document.addEventListener('visibilitychange', () => { if (document.visibilityState === 'visible') fetchLive(); });
  }
  window.GP = { openTeam: c => openSheet(() => teamSheet(c)), closeSheet, renderSlate, ingestLive: ev => { ingest(ev); renderSlate(); },
                setDate: d => { const k = SP.dates.indexOf(d); if (k >= 0) { i = -1; setIndex(k, false); } } };
})();
"""


def _trim_bundle(sp: dict, team_pages: dict, rankings: list[dict], sim: dict) -> dict:
    """Only the fields the JS reads, floats to 3 dp; adds plain-English 'why' lines per game."""
    names = {c: t["name"] for c, t in team_pages.items()}
    slates = {}
    for d, rows in sp["slates"].items():
        out = []
        for g in rows:
            r = {k: g[k] for k in ("h", "a", "ph", "pick", "rh", "ra", "hp", "ap", "ok", "eid", "t") if k in g}
            r["why"] = why_lines(g, names)
            out.append(r)
        slates[d] = out
    teams = {}
    for code, t in team_pages.items():
        teams[code] = {"name": t["name"], "conf": t["conf"], "w": t["w"], "l": t["l"], "rec": t["rec"], "model_acc": t["model_acc"],
                       "elo": t["elo"], "upset_wins": [{"d": u["d"]} for u in t["upset_wins"]],
                       "upset_losses": [{"d": u["d"]} for u in t["upset_losses"]],
                       "sched": [{k: g[k] for k in ("d", "opp", "home", "p", "rest", "pf", "pa", "w") if k in g} for g in t["sched"]]}
    rk = [{k: r[k] for k in ("team", "rank", "l10", "net10")} for r in rankings]
    sim_t = [{k: r[k] for k in ("team", "exp_w", "w_lo", "w_hi", "playoff", "top_seed")} for r in sim["teams"]]
    return _round({"sp": {"dates": sp["dates"], "slates": slates, "running": sp["running"]}, "teams": teams, "rankings": rk,
                   "sim": {"teams": sim_t, "focus": sim["focus"], "focus_hist": sim["focus_hist"], "n_sims": sim["n_sims"]}})


def _body(bundle: dict, sim: dict, met: dict, pred: dict) -> tuple[str, str]:
    """Returns (body_html, data_blob_json)."""
    d = met["data"]
    sp, rk, ups, tiers = bundle["season_predictions"], bundle["rankings"], bundle["upsets"], bundle["tiers"]
    meta = bundle.get("meta") or {}
    live = meta.get("source") == "espn"
    upd_raw = meta.get("updated_at")
    if upd_raw:
        u = datetime.fromisoformat(upd_raw)
        updated = f"{u:%b} {u.day}, {u:%H:%M} UTC"
    else:
        updated = ""
    n_final = meta.get("n_final") or d["games_total"]
    last_played = sp.get("last_played_date") or sp["dates"][-1]
    season_hits = sp["running"][last_played]
    default_date = pred["date"] if pred.get("date") in sp["slates"] else last_played
    today = sp.get("today")
    names = {c: t["name"] for c, t in bundle["team_pages"].items()}

    blob = _trim_bundle(sp, bundle["team_pages"], rk, sim)
    blob.update({"default_date": default_date, "disclaimer": DISCLAIMER, "live": live, "today": today,
                 "updated": updated, "season": d["test_season"]})
    data_blob = json.dumps(blob, separators=(",", ":")).replace("</", "<\\/")

    hits, n = season_hits["std_hits"], season_hits["std_n"]
    summary = (f'The model picked {hits} of {n} games right this season ({hits / n:.0%}). '
               if n else "The season hasn&rsquo;t started yet. ")
    src = (f'Data from ESPN, updated {esc(updated)}.' if live else "Sample league with made-up teams, used when live data can&rsquo;t be reached.")
    all_rank = "".join(rank_row(r) for r in rk)
    top10 = "".join(rank_row(r) for r in rk[:10])
    confs = sorted({r["conf"] for r in sim["teams"]})
    seg = "".join(f'<button type="button" role="tab" data-act="conf" data-conf="{esc(c)}" class="{"on" if c == confs[0] else ""}" aria-selected="{"true" if c == confs[0] else "false"}">{esc(c)}</button>' for c in confs)
    odds = "".join(odds_list(sim, c, k == 0) for k, c in enumerate(confs))
    spots = sim["playoff_spots"]

    body = f"""
<template id="top-actions"><button type="button" class="icon-btn" data-act="dates" aria-label="Choose a date"><svg viewBox="0 0 24 24" aria-hidden="true"><rect x="3.5" y="5" width="17" height="15.5" rx="3"/><path d="M3.5 10h17M8 3v4M16 3v4"/></svg></button></template>
<main class="page wide">
  <h1 class="large-title">Games</h1>
  <p class="summary">{summary}<button type="button" class="linkbtn" data-act="about">See how</button></p>

  <div class="strip">
    <button type="button" class="icon-btn nav" id="day-prev" data-act="prev" aria-label="Previous day"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M15 5l-7 7 7 7"/></svg></button>
    <div class="days" id="days" role="tablist" aria-label="Game days"></div>
    <button type="button" class="icon-btn nav" id="day-next" data-act="next" aria-label="Next day"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M9 5l7 7-7 7"/></svg></button>
  </div>

  <div class="cols main-cols">
    <section aria-labelledby="slate-title">
      <div class="section-h slate-h"><h2 id="slate-title">Today</h2><span class="meta" id="slate-meta" aria-live="polite"></span></div>
      <div class="games" id="slate"></div>
    </section>

    <div class="side-col">
      <section class="section" aria-labelledby="rk-h">
        <div class="section-h"><h2 id="rk-h">Power rankings</h2><button type="button" class="linkbtn more" data-act="rankings">See all</button></div>
        <div class="list">{top10}</div>
      </section>

      <section class="section" aria-labelledby="po-h">
        <div class="section-h"><h2 id="po-h">Playoff odds</h2></div>
        <div class="seg" role="tablist" aria-label="Conference">{seg}</div>
        {odds}
        <p class="section-foot">Chance of finishing in the top {spots} of the conference, from {sim["n_sims"]:,} simulated seasons.</p>
      </section>

      <section class="section" aria-labelledby="up-h">
        <div class="section-h"><h2 id="up-h">Biggest upsets</h2></div>
        <div class="list">{upset_rows(ups, names)}</div>
        <p class="section-foot">The chance the model gave the winner before tip-off.</p>
      </section>
    </div>
  </div>

  <p class="footer-note">{src} <button type="button" class="linkbtn" data-act="about">About this model</button><br>{esc(DISCLAIMER)}</p>
</main>

<template id="rankings-tpl"><h3>Power rankings<small>All 30 teams by current team rating</small></h3><div class="list">{all_rank}</div>
  <p class="section-foot">Arrows show movement over the last 30 days. Dots are the last 10 games, green for a win.</p></template>
<template id="about-tpl">{about_html(met, sim, tiers, live, updated, n_final)}</template>

<div class="scrim" id="scrim"></div>
<section class="sheet" id="sheet" role="dialog" aria-modal="true" aria-labelledby="sheet-title" aria-hidden="true" inert>
  <div class="sh-bar"><div class="grabber"></div><button type="button" class="btn plain" id="sheet-back" data-act="back" hidden>Back</button><button type="button" class="btn plain done" id="sheet-done" data-act="close">Done</button></div>
  <div id="sheet-body"></div>
</section>
"""
    return body, data_blob


def _head_snippet(depth: str) -> str:
    p = SITE / "_head_snippet.html"
    s = p.read_text() if p.exists() else '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">'
    s = "\n".join(l for l in s.splitlines() if not l.strip().startswith("<!--"))
    return s.replace('href="../', f'href="{depth}')


def _page(body: str, data_blob: str, *, standalone: bool, depth: str) -> str:
    if standalone:
        head_assets = f"<style>{(SITE_ASSETS / 'app.css').read_text()}</style>"
        tail_shell = f"<script>{(SITE_ASSETS / 'app-shell.js').read_text()}</script>"
    else:
        head_assets = '<link rel="stylesheet" href="../assets/app.css">\n<script src="../assets/app-shell.js" defer></script>'
        tail_shell = ""
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
{_head_snippet(depth)}
<title>Games — NBA Projects</title>
<meta name="description" content="Win chances for every NBA game, power rankings and playoff odds.">
{head_assets}
<style>{CSS}</style>
</head>
<body data-app="predictor" data-title="Games" data-depth="{depth}">
{body}
<script id="gp-data" type="application/json">{data_blob}</script>
{tail_shell}
<script>{JS}</script>
</body></html>
"""


def render(predictions_path: Path | None = None, metrics_path: Path | None = None, out_path: Path | None = None,
           bundle: dict | None = None, sim: dict | None = None, site_dir: Path | None = None) -> Path:
    """bundle = analytics.build_all() output; sim = simulate.simulate() output. Both are computed if missing.
    Writes output/report.html (standalone) and, if site_dir is given, <site_dir>/index.html for the shared site."""
    predictions_path = Path(predictions_path) if predictions_path else OUTPUT / "predictions.json"
    metrics_path = Path(metrics_path) if metrics_path else OUTPUT / "metrics.json"
    out_path = Path(out_path) if out_path else OUTPUT / "report.html"
    OUTPUT.mkdir(exist_ok=True)
    if bundle is None:
        from analytics import build_all
        bundle = build_all()
    if sim is None:
        from simulate import simulate
        sim = simulate(feats=bundle["feats"], teams=bundle["teams"])
    pred = json.loads(predictions_path.read_text()) if predictions_path.exists() else {"date": bundle["season_predictions"]["dates"][-1]}
    met = json.loads(metrics_path.read_text())

    body, blob = _body(bundle, sim, met, pred)
    # standalone copy: hub links resolve to ../../site/ from output/
    out_path.write_text(_page(body, blob, standalone=True, depth="../../site/"))
    if site_dir is not None:
        site_dir = Path(site_dir)
        site_dir.mkdir(parents=True, exist_ok=True)
        (site_dir / "index.html").write_text(_page(body, blob, standalone=False, depth="../"))
    return out_path


if __name__ == "__main__":
    print(render())
