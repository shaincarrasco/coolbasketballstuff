"""
Render the Game Predictor page — mobile-first, shared-shell design (see ../DESIGN_BRIEF.md).

Two outputs from the same body:
  * output/report.html          standalone: shell.css + shell.js inlined, opens from file://
  * <site>/predictor/index.html references ../assets/shell.css and ../assets/shell.js
                                (python cli.py demo --site ../site/predictor)

Page order: eyebrow + h1 + lede -> 4 KPI tiles -> Slate (played days + next 7, live strip, prev/next,
select, arrow keys) -> Power rankings -> Season simulator -> Upsets + confidence
tiers (details) -> Model lab (details) -> Method & limits -> footer. Team codes
everywhere open a drawer (bottom sheet on mobile, side panel on desktop).

Data: ESPN's public scoreboard API (real NBA games) or, offline, the synthetic
sample league; the page says which. A small "Live scores" strip fetches today's
ESPN scoreboard in the browser and overlays status / score on matching cards.
Not betting advice. Nothing here is guaranteed.
"""
from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"
SITE_ASSETS = ROOT.parent / "site" / "assets"
DISCLAIMER = "Not betting advice. Nothing here is guaranteed."


def esc(s) -> str:
    return html.escape(str(s))


def tm(code: str) -> str:
    return f'<button type="button" class="tm" data-team="{esc(code)}">{esc(code)}</button>'


def _round(o, nd: int = 3):
    """Recursively round floats (trims the embedded JSON)."""
    if isinstance(o, float):
        return round(o, nd)
    if isinstance(o, dict):
        return {k: _round(v, nd) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_round(v, nd) for v in o]
    return o


# ---- SVG charts (colors are CSS variables so they follow the shell theme) -----
def histogram_svg(hist: list[dict], n_sims: int, focus: str, actual: int | None) -> str:
    W, H, L, R, T, B = 560, 200, 34, 8, 10, 28
    pw, ph = W - L - R, H - T - B
    peak = max(x["n"] for x in hist) or 1
    bw = pw / len(hist)
    p = [f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" aria-label="{esc(focus)} win-total distribution">']
    for v in (0.05, 0.10, 0.15):
        y = T + ph - v * n_sims / peak * ph
        if y > T:
            p.append(f'<line class="grid" x1="{L}" y1="{y:.1f}" x2="{W - R}" y2="{y:.1f}"/><text x="{L - 5}" y="{y + 4:.1f}" text-anchor="end">{v:.0%}</text>')
    step = max(1, len(hist) // 10)
    for i, x in enumerate(hist):
        bh = x["n"] / peak * ph
        hit = actual is not None and x["wins"] == actual
        style = "fill:var(--pos)" if hit else "fill:var(--text);fill-opacity:.55"
        p.append(f'<rect x="{L + i * bw + 1:.1f}" y="{T + ph - bh:.1f}" width="{bw - 2:.1f}" height="{bh:.1f}" rx="1.5" style="{style}"><title>{x["wins"]} wins: {x["n"] / n_sims:.1%}</title></rect>')
        if i % step == 0:
            p.append(f'<text x="{L + (i + .5) * bw:.1f}" y="{H - 10}" text-anchor="middle">{x["wins"]}</text>')
    p.append(f'<line class="axis" x1="{L}" y1="{T + ph}" x2="{W - R}" y2="{T + ph}"/></svg>')
    return "".join(p)


def walk_svg(walk: list[dict]) -> str:
    W, H, L, R, T, B = 640, 240, 40, 12, 14, 28
    pw, ph = W - L - R, H - T - B
    n = len(walk)
    lo, hi = 0.45, 0.85
    xs = [L + (i + 0.5) * pw / n for i in range(n)]
    sy = lambda v: T + (1 - (min(max(v, lo), hi) - lo) / (hi - lo)) * ph
    p = [f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" aria-label="Walk-forward accuracy by month">']
    for v in (0.5, 0.6, 0.7):
        p.append(f'<line class="grid" x1="{L}" y1="{sy(v):.1f}" x2="{W - R}" y2="{sy(v):.1f}"/><text x="{L - 6}" y="{sy(v) + 4:.1f}" text-anchor="end">{v:.0%}</text>')
    series = (("home_accuracy", "var(--faint)", 1.25, "2 4"), ("elo_accuracy", "var(--muted)", 1.5, "6 4"),
              ("hgb_accuracy", "var(--muted)", 1.5, ""), ("xgb_accuracy", "var(--text)", 2.5, ""))
    for key, color, sw, dash in series:
        d = " ".join(f"{'M' if i == 0 else 'L'}{xs[i]:.1f},{sy(w[key]):.1f}" for i, w in enumerate(walk))
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        p.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw}"{dash_attr} stroke-linejoin="round"/>')
        for i, w in enumerate(walk):
            p.append(f'<circle cx="{xs[i]:.1f}" cy="{sy(w[key]):.1f}" r="{3 if key == "xgb_accuracy" else 2.2}" fill="{color}"><title>{key.split("_")[0]} {w["month"]}: {w[key]:.1%} (n={w["n"]})</title></circle>')
            if key == "xgb_accuracy":
                p.append(f'<text x="{xs[i]:.1f}" y="{sy(w[key]) - 9:.1f}" text-anchor="middle" style="fill:var(--text);font-weight:600">{w[key]:.0%}</text>')
    for i, w in enumerate(walk):
        p.append(f'<text x="{xs[i]:.1f}" y="{H - 9}" text-anchor="middle">{w["month"][2:]}</text>')
    p.append("</svg>")
    return "".join(p)


def calibration_svg(table_xgb: list[dict], table_elo: list[dict]) -> str:
    W, H, L, B, T, R = 420, 400, 40, 40, 12, 12
    pw, ph = W - L - R, H - T - B
    sx = lambda v: L + v * pw
    sy = lambda v: T + (1 - v) * ph
    p = [f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" aria-label="Reliability diagram">']
    for v in (0, .2, .4, .6, .8, 1):
        p.append(f'<line class="grid" x1="{sx(0)}" y1="{sy(v):.1f}" x2="{sx(1)}" y2="{sy(v):.1f}"/>')
        p.append(f'<text x="{L - 6}" y="{sy(v) + 4:.1f}" text-anchor="end">{v:.1f}</text><text x="{sx(v):.1f}" y="{T + ph + 16}" text-anchor="middle">{v:.1f}</text>')
    p.append(f'<line x1="{sx(0)}" y1="{sy(0)}" x2="{sx(1)}" y2="{sy(1)}" stroke="var(--faint)" stroke-dasharray="4 4"/>')
    maxn = max([r["n"] for r in table_xgb] + [1])
    for table, color, fill in ((table_elo, "var(--muted)", "none"), (table_xgb, "var(--text)", "var(--text)")):
        pts = [(sx(r["pred_mean"]), sy(r["actual_rate"]), r["n"]) for r in table if r["n"]]
        if len(pts) > 1:
            p.append('<path d="' + " ".join(f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y, _) in enumerate(pts)) + f'" fill="none" stroke="{color}" stroke-width="1.5"/>')
        for x, y, n in pts:
            p.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{2.5 + 6 * (n / maxn) ** 0.5:.1f}" fill="{fill}" stroke="{color}" stroke-width="1.5" fill-opacity=".85"><title>n={n}</title></circle>')
    p.append(f'<text x="{L + pw / 2}" y="{H - 4}" text-anchor="middle">predicted p(home) vs actual home win rate</text></svg>')
    return "".join(p)


# ---- HTML fragments ------------------------------------------------------------
def form_dots(l10: str) -> str:
    return '<span class="form" aria-label="last 10">' + "".join(f'<i class="{"w" if c == "W" else "l"}"></i>' for c in l10) + "</span>"


def rankings_html(rankings: list[dict]) -> str:
    rows = []
    for r in rankings:
        d = r["d_elo_30"]
        de = f'<span class="{"pos" if d >= 0 else "neg"}">{d:+.0f}</span>'
        dr = r["d_rank"]
        rk = (f'<span class="rank pos">&#9650;{dr}</span>' if dr > 0 else f'<span class="rank neg">&#9660;{-dr}</span>' if dr < 0 else "")
        net = r["net10"] if r["net10"] is not None else 0.0
        rows.append(f'<tr><td class="n muted">{r["rank"]}{rk}</td><td>{tm(r["team"])}<span class="tname muted">{esc(r["name"])}</span></td>'
                    f'<td class="n"><b>{r["elo"]:.0f}</b></td><td class="n">{de}</td><td>{form_dots(r["l10"])}</td>'
                    f'<td class="n {"pos" if net >= 0 else "neg"}">{net:+.1f}</td><td class="n">{r["w"]}-{r["l"]}</td><td class="muted">{esc(r["conf"][0])}</td>'
                    f'<td class="n muted">{r["off10"]:.1f}</td><td class="n muted">{r["def10"]:.1f}</td></tr>')
    return ('<div class="tbl rank"><table><thead><tr><th class="n">#</th><th>Team</th><th class="n">Elo</th><th class="n">&Delta;30d</th>'
            '<th>L10</th><th class="n">Net10</th><th class="n">W-L</th><th>Conf</th><th class="n">Off10</th><th class="n">Def10</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


def sim_html(sim: dict) -> str:
    def conf_table(c):
        rows = []
        for r in [r for r in sim["teams"] if r["conf"] == c]:
            po = r["playoff"]
            fin = ""
            if sim["season_complete"]:
                fin = (f'<td class="n muted">{r["final_w"]}-{r["final_l"]}</td>'
                       f'<td class="{"pos" if r["made_playoffs"] else "muted"}">{"In" if r["made_playoffs"] else "out"}</td>')
            rows.append(f'<tr><td>{tm(r["team"])}</td><td class="n muted">{r["w"]}-{r["l"]}</td><td class="n">{r["exp_w"]:.1f}</td>'
                        f'<td class="n"><span class="mini"><i style="width:{po * 100:.0f}%"></i></span>{po:.0%}</td>'
                        f'<td class="n {"muted" if r["top_seed"] < .05 else ""}">{r["top_seed"]:.0%}</td>{fin}</tr>')
        extra = '<th class="n">Final</th><th>Actual</th>' if sim["season_complete"] else ""
        return (f'<div class="tbl compact"><table><thead><tr><th>{esc(c)}</th><th class="n">W-L</th><th class="n">Exp W</th>'
                f'<th class="n">Playoff</th><th class="n">#1 seed</th>{extra}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>')

    focus = next(r for r in sim["teams"] if r["team"] == sim["focus"])
    spots = sim["playoff_spots"]
    sure = [r for r in sim["teams"] if r["playoff"] >= .7]
    sure_made = sum(1 for r in sure if r["made_playoffs"]) if sim["season_complete"] else None
    sneak = [r for r in sim["teams"] if r["playoff"] < .3 and r["made_playoffs"]] if sim["season_complete"] else []
    if sim.get("live"):
        read = (f'<b>Read:</b> From {esc(sim["date"])} the simulator gives {len(sure)} teams a 70%+ chance of a top-6 seed '
                f'(direct playoff spot; the play-in is not modelled). "Exp W" is the mean of {sim["n_sims"]} simulated finishes. '
                + esc(sim.get("schedule_note") or ""))
    else:
        read = (f'<b>Read:</b> From {esc(sim["date"])} the simulator gave {len(sure)} teams a 70%+ playoff chance'
                + (f' and {sure_made} of them made it' if sure_made is not None else "")
                + (f'; {", ".join(esc(r["team"]) for r in sneak)} got in from under 30%. ' if sneak else '. ')
                + (f'Playoff-odds Brier vs the real finish: {sim["playoff_brier"]:.3f} (0 = perfect, 0.25 = coin flip).' if sim["playoff_brier"] is not None else ""))
    act = focus["final_w"]
    return f"""
    <div class="grid g2">{conf_table("East")}{conf_table("West")}</div>
    <div class="card" style="margin-top:10px">
      <div class="sec-h"><h2 style="font-size:14px">{esc(sim["focus"])} win-total distribution</h2><span class="read">{esc(focus["name"])} &middot; {focus["w"]}-{focus["l"]} on {esc(sim["date"])}</span></div>
      {histogram_svg(sim["focus_hist"], sim["n_sims"], sim["focus"], act)}
      <div class="legend"><span><i style="background:var(--text);opacity:.55"></i>simulated ({sim["n_sims"]} runs)</span>
        {f'<span><i style="background:var(--pos)"></i>actual final ({act} wins)</span>' if act is not None else f'<span class="muted">{focus["w"]}-{focus["l"]} so far</span>'}
        <span class="muted">expected {focus["exp_w"]:.1f} &middot; 5&ndash;95% {focus["w_lo"]}&ndash;{focus["w_hi"]}</span></div>
    </div>
    <p class="read-line">{read} Top {spots} per conference qualify; {sim["games_remaining"]} games were simulated from Elo with +80 home court.</p>"""


def upsets_html(ups: list[dict], tiers: list[dict], threshold: float) -> str:
    rows = []
    for u in ups[:20]:
        rows.append(f'<tr><td class="muted">{esc(u["d"][5:])}</td><td>{tm(u["winner"])} <span class="muted">{"vs" if u["home_won"] else "@"}</span> {tm(u["loser"])}</td>'
                    f'<td class="n muted">{esc(u["score"])}</td><td class="n neg">{u["p_winner"]:.0%}</td><td class="n muted">{u["elo_gap"]:+d}</td>'
                    f'<td class="muted">{esc(", ".join(u["missed"]))}</td></tr>')
    ups_t = ('<div class="tbl"><table><thead><tr><th>Date</th><th>Winner / loser</th><th class="n">Score</th><th class="n">p(winner)</th><th class="n">Elo gap</th><th>What the model missed</th></tr></thead>'
             f'<tbody>{"".join(rows)}</tbody></table></div>')
    trs = []
    for t in tiers:
        if not t["n"]:
            continue
        gap = t["actual"] - t["pred"]
        trs.append(f'<tr><td>{esc(t["tier"].replace("%-", "&ndash;"))}</td><td class="n muted">{t["n"]}</td><td class="n">{t["pred"]:.0%}</td>'
                   f'<td class="n"><b>{t["actual"]:.0%}</b></td><td class="n {"pos" if gap >= 0 else "neg"}">{gap * 100:+.1f}</td>'
                   f'<td><span class="pair"><i style="width:{t["pred"] * 100:.0f}%"></i><i class="act" style="width:{t["actual"] * 100:.0f}%"></i></span></td></tr>')
    tier_t = ('<div class="tbl compact"><table><thead><tr><th>Confidence</th><th class="n">n</th><th class="n">Pred</th><th class="n">Hit</th><th class="n">Gap</th><th>pred / hit</th></tr></thead>'
              f'<tbody>{"".join(trs)}</tbody></table></div>')
    n_total = sum(t["n"] for t in tiers)
    worst = max((t for t in tiers if t["n"]), key=lambda t: abs(t["actual"] - t["pred"]))
    return f"""
    <div>
        <div class="sub-h">Upset detector <span class="muted">{len(ups)} of {n_total} games had p(winner) &lt; {threshold:.0%} &middot; top 20</span></div>
        {ups_t}
        <p class="read-line"><b>Read:</b> "What the model missed" is pre-game context on the winner's side (rest, L10 form, streak, head-to-head). When nothing applies the honest label is <b>noise</b>: with a 12-point score SD, a 30% underdog wins 30% of the time.</p>
    </div>
    <div class="grid g2" style="align-items:start;margin-top:16px">
      <div>
        <div class="sub-h">Confidence tiers <span class="muted">pick confidence vs hit rate</span></div>
        {tier_t}
      </div>
        <p class="read-line" style="margin-top:0"><b>Read:</b> A calibrated model's hit rate matches its confidence in every tier. Largest gap here: the {esc(worst["tier"].replace("%-", "–"))} tier ({worst["actual"] - worst["pred"]:+.0%} on n={worst["n"]}); gaps inside &plusmn;5 pts on ~100-game bins are noise.</p>
    </div>"""


def lab_html(met: dict) -> str:
    m, cal, walk = met["models"], met["calibration"], met["walk_forward"]

    def mrow(name, r, best=False):
        return (f'<tr><td>{"<b>" if best else ""}{esc(name)}{"</b>" if best else ""}</td><td class="n">{r["accuracy"]:.1%}</td><td class="n">{r["log_loss"]:.3f}</td>'
                f'<td class="n">{r["brier"]:.3f}</td><td class="n">{r["roc_auc"]:.3f}</td></tr>')

    table = "".join([mrow("Home team wins", m["home_baseline"]), mrow("Elo-logit (1 feature)", m["elo_logit"]),
                     mrow("HistGB (20 features)", m["hgb"]), mrow("XGBoost raw", m["xgb_raw"]),
                     mrow(f"XGBoost + {cal['chosen']}", m["xgb"], best=True)])
    prior = met.get("prior_holdout")
    prior_table = ""
    if prior:
        pm = prior["models"]
        prior_table = ("".join([mrow("Home team wins", pm["home_baseline"]), mrow("Elo-logit (1 feature)", pm["elo_logit"]),
                                mrow("HistGB (20 features)", pm["hgb"]), mrow(f"XGBoost + {prior['calibrator']}", pm["xgb"], best=True)]))
    pooled = met.get("pooled")
    rel_xgb, rel_elo, ece_x, ece_e, rel_n = cal["reliability_xgb"], cal["reliability_elo"], cal["ece_xgb"], cal["ece_elo"], met["data"]["n_test"]
    if pooled:
        rel_xgb, rel_elo, ece_x, ece_e, rel_n = pooled["reliability_xgb"], pooled["reliability_elo"], pooled["ece_xgb"], pooled["ece_elo"], pooled["n"]
    imp = met["feature_importance"][:8]
    mx = imp[0]["gain_share"] if imp else 1
    imp_rows = "".join(f'<tr><td class="mono muted">{esc(r["feature"])}</td><td style="width:45%"><span class="bar"><i style="width:{r["gain_share"] / mx * 100:.0f}%;background:var(--muted)"></i></span></td><td class="n">{r["gain_share"]:.1%}</td></tr>' for r in imp)
    walk_rows = "".join(f'<tr><td class="muted">{w["month"]}</td><td class="n muted">{w["n"]}</td><td class="n">{w["home_accuracy"]:.1%}</td>'
                        f'<td class="n">{w["elo_accuracy"]:.1%}</td><td class="n">{w["hgb_accuracy"]:.1%}</td><td class="n"><b>{w["xgb_accuracy"]:.1%}</b></td><td class="n muted">{w["xgb_log_loss"]:.3f}</td></tr>' for w in walk)
    d = met["data"]
    xa, ea, ha = m["xgb"]["accuracy"], m["elo_logit"]["accuracy"], m["home_baseline"]["accuracy"]
    small = d.get("small_sample")
    if prior:
        pxa, pea, pha = prior["models"]["xgb"]["accuracy"], prior["models"]["elo_logit"]["accuracy"], prior["models"]["home_baseline"]["accuracy"]
        read_models = (f'<b>Read:</b> On the full {esc(prior["test_season"])} holdout (n={prior["n_test"]}) XGBoost beats "home team wins" by {(pxa - pha) * 100:.0f} points and the one-feature Elo regression by {(pxa - pea) * 100:+.1f}. '
                       + (f'The {esc(d["test_season"])} numbers above are {d["n_test"]} games in, so a ±4-point swing is noise. ' if small else '')
                       + f'ECE on the pooled {rel_n} out-of-sample games: XGB {ece_x:.3f}, Elo {ece_e:.3f}.')
        prior_block = f"""
        <div class="sub-h" style="margin-top:12px">Prior full-season holdout <span class="muted">{esc(prior["test_season"])}, n={prior["n_test"]}, trained on {esc(" + ".join(prior["train_seasons"]))}</span></div>
        <div class="tbl compact"><table><thead><tr><th>Model</th><th class="n">Acc</th><th class="n">Log loss</th><th class="n">Brier</th><th class="n">AUC</th></tr></thead><tbody>{prior_table}</tbody></table></div>"""
    else:
        read_models = f'<b>Read:</b> Every learner beats "home team wins" by about {(xa - ha) * 100:.0f} points and ties a one-feature Elo regression ({xa - ea:+.1%}). Elo carries almost all the predictable signal; the other 19 features add noise-level residual on {d["n_train"]} training games. ECE: XGB {cal["ece_xgb"]:.3f}, Elo {cal["ece_elo"]:.3f}.'
        prior_block = ""
    return f"""
    <div class="grid g2" style="align-items:start">
      <div>
        <div class="sub-h">Scoreboard <span class="muted">test season {esc(d["test_season"])} to date, n={d["n_test"]}{' · small sample' if small else ''}</span></div>
        <div class="tbl compact"><table><thead><tr><th>Model</th><th class="n">Acc</th><th class="n">Log loss</th><th class="n">Brier</th><th class="n">AUC</th></tr></thead><tbody>{table}</tbody></table></div>
        {prior_block}
        <p class="read-line">{read_models}</p>
      </div>
      <div>
        <div class="sub-h">XGBoost gain share <span class="muted">top 8 of {len(met["features"])} features</span></div>
        <div class="tbl compact"><table><tbody>{imp_rows}</tbody></table></div>
      </div>
    </div>
    <div class="grid g2" style="margin-top:14px;align-items:start">
      <div>
        <div class="sub-h">Walk-forward accuracy <span class="muted">refit each month on everything before it{" · " + esc(" + ".join(pooled["seasons"])) if pooled else ""}</span></div>
        {walk_svg(walk)}
        <div class="legend"><span><i style="background:var(--text)"></i>XGBoost</span><span><i style="background:var(--muted)"></i>HGB</span><span><i class="dash"></i>Elo-logit</span><span><i class="dot"></i>home team</span></div>
        <div class="tbl compact" style="margin-top:10px"><table><thead><tr><th>Month</th><th class="n">n</th><th class="n">Home</th><th class="n">Elo</th><th class="n">HGB</th><th class="n">XGB</th><th class="n">LL</th></tr></thead><tbody>{walk_rows}</tbody></table></div>
      </div>
      <div>
        <div class="sub-h">Reliability <span class="muted">10 bins, dot size = n{f" · pooled out-of-sample n={rel_n}" if pooled else ""}</span></div>
        {calibration_svg(rel_xgb, rel_elo)}
        <div class="legend"><span><i style="background:var(--text)"></i>XGBoost</span><span><i style="background:var(--muted)"></i>Elo-logit</span><span><i class="dash"></i>perfect</span></div>
        <p class="read-line"><b>Read:</b> Points on the diagonal mean "when the model says 70%, the home side wins 70% of the time". Calibrator chosen on a validation slice by Brier: <b>{esc(cal["chosen"])}</b>.</p>
      </div>
    </div>"""


# ---- page-specific CSS (small; shell.css carries the system) -------------------
CSS = r"""
.kpi .sub .v{color:var(--text);font-weight:600}
.grid > *,details.more .body > *{min-width:0}
.sec-h{flex-wrap:wrap}.sec-h h2{white-space:nowrap}
.desk-only{display:none}@media(min-width:900px){.desk-only{display:inline}}
.tbl .bar{display:block;min-width:70px}
@media(max-width:899px){.mini{display:none}.form{gap:2px}.form i{width:7px;height:7px}.tbl.rank td,.tbl.rank th{padding:9px 8px}}
.sub-h{font-weight:700;font-size:14px;margin:0 0 8px;display:flex;flex-wrap:wrap;gap:4px 10px;align-items:baseline}
.sub-h .muted{font-weight:500;font-size:12px}
.tm{appearance:none;border:0;background:transparent;color:var(--text);font:700 13px var(--mono);letter-spacing:.02em;padding:10px 8px;margin:-10px -6px;border-radius:6px;cursor:pointer;min-height:40px;text-decoration:underline dotted var(--faint);text-underline-offset:3px}
.tm:hover{background:var(--surface-2);text-decoration-color:var(--text)}
.tm:focus:not(:focus-visible),.btn:focus:not(:focus-visible){outline:none}
.tm:focus-visible{outline:2px solid var(--accent);outline-offset:-2px}
.tname{display:none;margin-left:8px;font-weight:500}
@media(min-width:900px){.tname{display:inline}}
.legend{display:flex;flex-wrap:wrap;gap:6px 14px;color:var(--muted);font-size:12px;margin-top:8px}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;vertical-align:-1px;margin-right:5px}
.legend i.dash{height:0;width:14px;vertical-align:3px;background:none!important;border-top:2px dashed var(--muted)}
.legend i.dot{height:0;width:14px;vertical-align:3px;background:none!important;border-top:2px dotted var(--faint)}
.mini{display:inline-block;width:40px;height:5px;border-radius:99px;background:var(--surface-2);vertical-align:2px;margin-right:6px;overflow:hidden}
.mini i{display:block;height:100%;background:var(--muted)}
.pair{display:block;width:110px}
.pair i{display:block;height:4px;border-radius:99px;background:var(--line-2);margin:2px 0}
.pair i.act{background:var(--text)}
.tbl.compact table{min-width:0}
.tbl.compact td,.tbl.compact th{padding:8px 10px}
.tbl.compact table{font-size:12.5px}

/* Slate */
.live{display:flex;align-items:center;gap:8px;flex-wrap:wrap;font-size:12px;color:var(--muted);margin:0 0 10px;padding:7px 10px;border:1px solid var(--line);border-radius:var(--r-sm);background:var(--surface)}
.live[hidden]{display:none}
.live .dot{width:7px;height:7px;border-radius:50%;background:var(--faint);flex:none}
.live.on .dot{background:var(--pos);box-shadow:0 0 0 3px color-mix(in srgb,var(--pos) 25%,transparent);animation:pulse 1.6s ease-in-out infinite}
@keyframes pulse{50%{box-shadow:0 0 0 1px color-mix(in srgb,var(--pos) 25%,transparent)}}
@media(prefers-reduced-motion:reduce){.live.on .dot{animation:none}}
.live #live-text{color:var(--text);font-weight:600}
.live #live-ts{margin-left:auto;font-family:var(--mono);font-variant-numeric:tabular-nums}
.g-fin.inplay{color:var(--text);font-weight:600}
.g-fin .q{color:var(--pos);font:600 10px var(--mono);letter-spacing:.06em;margin-left:6px}
.topbar .chip .chip-sub{display:none;font-weight:500;text-transform:none;letter-spacing:0;color:var(--faint);margin-left:6px}
@media(min-width:600px){.topbar .chip .chip-sub{display:inline}}
.slate-stats{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-top:10px}
.slate-stats .spark{margin-left:auto}
#slate-date{flex:3 1 170px;font-family:var(--mono);font-weight:600;color:var(--accent);border-color:color-mix(in srgb,var(--accent) 55%,var(--line-2))}
.controls .btn{flex:1 1 60px;justify-content:center;padding:0 10px}
.controls .btn:disabled{opacity:.35;cursor:default}
@media(min-width:900px){.controls .btn{flex:0 0 auto;min-width:96px}#slate-date{flex:0 1 240px}.slate-stats{margin-top:12px}}
.games{margin-top:12px;border:1px solid var(--line);border-radius:var(--r);background:var(--surface);overflow:hidden}
.game{display:grid;gap:6px 10px;padding:11px 12px;border-bottom:1px solid var(--line);align-items:center;
  grid-template-columns:1fr auto;grid-template-areas:"m pick" "bar bar" "p end"}
.game:last-child{border-bottom:0}
.game.head{display:none}
.g-n,.g-elo,.g-drv{display:none}
.g-m{grid-area:m;font-size:14px;display:flex;align-items:center;gap:6px;flex-wrap:wrap}
.g-m .at{color:var(--muted);font-size:12px}
.g-m .b2b{font:600 10px var(--mono);letter-spacing:.06em;color:var(--warn);border:1px solid color-mix(in srgb,var(--warn) 45%,transparent);border-radius:4px;padding:1px 4px}
.g-pick{grid-area:pick;justify-self:end}
.g-bar{grid-area:bar}
.pbar{display:flex;height:6px;border-radius:99px;overflow:hidden;background:var(--surface-2)}
.pbar i{display:block;height:100%}
.pbar i.a{background:var(--line-2)}
.pbar i.h{background:var(--muted)}
.pbar i.fav{background:var(--text)}
.g-p{grid-area:p;display:flex;justify-content:space-between;gap:12px;font-size:13px}
.g-p b{font-weight:500;color:var(--muted)}
.g-p b.fav{font-weight:700;color:var(--text)}
.g-end{grid-area:end;display:flex;align-items:center;gap:8px;justify-content:flex-end}
.g-fin{color:var(--muted);font-size:13px;white-space:nowrap}
.chip.pick{color:var(--text);border-color:var(--line-2);text-transform:none;letter-spacing:.02em;font-size:12px}
.chip.res{padding:5px 8px}
@media(min-width:900px){
  .game{grid-template-columns:28px minmax(170px,1.1fr) 90px minmax(140px,1.4fr) 52px 52px 120px minmax(150px,1.6fr) 104px 64px;grid-template-areas:none;padding:8px 12px;gap:0 12px}
  .game > *{grid-area:auto!important}
  .game.head{display:grid;font:600 11px/1 var(--mono);letter-spacing:.08em;text-transform:uppercase;color:var(--muted);padding:10px 12px}
  .g-n{display:block;color:var(--muted);font-size:12px}
  .g-elo{display:block;color:var(--muted);font-size:12px}
  .g-drv{display:block;color:var(--muted);font-size:12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .g-p,.g-end{display:contents}
  .g-p b{text-align:right;display:block}
  .g-pick{justify-self:start}
  .g-fin{text-align:right}
  .game:hover{background:var(--surface-2)}
  .game.head:hover{background:transparent}
}

/* Drawer internals */
.drawer .d-h{display:flex;align-items:flex-start;gap:10px;margin-bottom:12px}
.drawer .d-h h2{font-size:20px;letter-spacing:-.02em}
.drawer .d-h h2 small{display:block;color:var(--muted);font-size:13px;font-weight:500;margin-top:2px}
.drawer .d-h .btn{margin-left:auto}
.d-rec{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:14px}
.d-rec div{background:var(--surface-2);border-radius:var(--r-sm);padding:8px 10px}
.d-rec .l{font:600 10px/1 var(--mono);letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin-bottom:5px}
.d-rec .v{font-family:var(--mono);font-variant-numeric:tabular-nums;font-size:16px;font-weight:700}
.drawer h3{font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);font-family:var(--mono);margin:16px 0 8px;font-weight:600}
.drawer .tbl{mask-image:none;-webkit-mask-image:none}
.drawer .note{color:var(--faint);font-size:11px;margin:14px 0 0}
.kbd{font:600 11px var(--mono);color:var(--muted);border:1px solid var(--line-2);border-radius:4px;padding:1px 5px}
.hint{color:var(--faint);font-size:12px;margin-top:8px}
@media(max-width:899px){.hint .desk{display:none}}
"""


JS = r"""
(function(){
  const D = JSON.parse(document.getElementById('gp-data').textContent);
  const SP = D.sp, TP = D.teams, RK = {}; D.rankings.forEach(r => RK[r.team] = r);
  const $ = s => document.querySelector(s);
  const pct = p => Math.round(p * 100) + '%';
  const tm = c => `<button type="button" class="tm" data-team="${c}">${c}</button>`;
  let i = SP.dates.indexOf(D.default_date); if (i < 0) i = SP.dates.length - 1;

  // shell chip: the shared top bar says "Sample data" by default; say what this page actually runs on
  const chip = document.querySelector('.topbar .chip');
  if (chip && D.live) {
    chip.innerHTML = `Live data · ESPN<span class="chip-sub">· ${D.updated} · ${D.n_final} games</span>`;
    chip.title = `ESPN public API · data updated ${D.updated} · ${D.n_final} final games`;
  }

  function sparkRunning(upto){
    const W = 120, H = 28, pts = [];
    const dates = SP.dates.slice(0, upto + 1), n = dates.length;
    dates.forEach((d, k) => { const r = SP.running[d]; if (r.std_n >= 20) pts.push([k, r.std_hits / r.std_n]); });
    if (pts.length < 2) return '';
    const lo = .5, hi = .75;
    const y = v => (H - 2 - (Math.min(hi, Math.max(lo, v)) - lo) / (hi - lo) * (H - 4)).toFixed(1);
    const line = pts.map(p => `${(p[0] / Math.max(1, n - 1) * W).toFixed(1)},${y(p[1])}`).join(' ');
    const yb = y(D.baseline);
    return `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" aria-label="season-to-date accuracy"><line x1="0" y1="${yb}" x2="${W}" y2="${yb}" stroke="var(--line-2)" stroke-dasharray="3 3"/><polyline points="${line}" fill="none" stroke="var(--text)" stroke-width="1.5"/></svg>`;
  }

  function renderSlate(){
    const d = SP.dates[i], rows = SP.slates[d], r = SP.running[d];
    $('#slate-date').value = d;
    $('#slate-count').textContent = rows.length + (rows.length === 1 ? ' game' : ' games');
    const dayAcc = r.n ? r.hits / r.n : null;
    $('#st-day').className = 'chip ' + (dayAcc == null ? '' : dayAcc >= D.baseline ? 'pos' : 'neg');
    $('#st-day').textContent = r.n ? `Day ${r.hits}/${r.n} · ${pct(dayAcc)}` : (d === D.today ? 'Today' : d > (D.today || '') ? 'Upcoming' : 'Day —');
    $('#st-std').textContent = r.std_n ? `Season ${r.std_hits}/${r.std_n} · ${pct(r.std_hits / r.std_n)}` : 'Season —';
    $('#st-spark').innerHTML = sparkRunning(i);
    $('#prev').disabled = i === 0; $('#next').disabled = i === SP.dates.length - 1;
    let h = '';
    rows.forEach((g, k) => {
      const homeFav = g.ph >= .5, conf = Math.max(g.ph, 1 - g.ph);
      const flags = (g.ra === 1 ? `<span class="b2b" title="away team on a back-to-back">${g.a} B2B</span>` : '') + (g.rh === 1 ? `<span class="b2b" title="home team on a back-to-back">${g.h} B2B</span>` : '');
      const end = ('ok' in g)
        ? `<span class="g-fin num">${g.ap}–${g.hp}</span><span class="g-res"><span class="chip res ${g.ok ? 'pos' : 'neg'}">${g.ok ? 'Hit' : 'Miss'}</span></span>`
        : `<span class="g-fin muted">${g.t ? g.t + ' ET' : '—'}</span><span class="g-res"></span>`;
      h += `<div class="game" role="row"${g.eid ? ` data-espn-id="${g.eid}"` : ''}>
        <span class="g-n num">${k + 1}</span>
        <span class="g-m">${tm(g.a)}<span class="at">@</span>${tm(g.h)}${flags}</span>
        <span class="g-elo num">${g.ea} · ${g.eh}</span>
        <span class="g-bar"><span class="pbar" title="p(home) = ${pct(g.ph)}"><i class="a${homeFav ? '' : ' fav'}" style="width:${((1 - g.ph) * 100).toFixed(1)}%"></i><i class="h${homeFav ? ' fav' : ''}" style="flex:1"></i></span></span>
        <span class="g-p num"><b class="${homeFav ? '' : 'fav'}">${pct(1 - g.ph)}</b><b class="${homeFav ? 'fav' : ''}">${pct(g.ph)}</b></span>
        <span class="g-pick"><span class="chip pick">${g.pick} <span class="num">${pct(conf)}</span></span></span>
        <span class="g-drv" title="${g.drv.join(' · ')}">${g.drv.join(' · ') || '—'}</span>
        <span class="g-end">${end}</span>
      </div>`;
    });
    $('#slate-body').innerHTML = h;
    if (LIVE.events) overlayLive();
  }

  // ---- live scores: today's ESPN scoreboard, fetched in the browser (CORS *), refreshed every 60 s while visible ----
  const LIVE = { events: null, at: null, timer: null };
  const etDate = () => new Intl.DateTimeFormat('en-CA', { timeZone: 'America/New_York', year: 'numeric', month: '2-digit', day: '2-digit' }).format(new Date());
  function overlayLive(){
    const ev = LIVE.events || [];
    let inplay = 0, fin = 0, up = 0;
    ev.forEach(e => {
      const st = (e.status && e.status.type) || {}; const name = st.name || '';
      if (name === 'STATUS_IN_PROGRESS' || name === 'STATUS_HALFTIME' || name === 'STATUS_END_PERIOD') inplay++; else if (name.indexOf('STATUS_FINAL') === 0) fin++; else up++;
      const row = document.querySelector(`.game[data-espn-id="${e.id}"]`); if (!row) return;
      const comp = (e.competitions || [])[0]; if (!comp) return;
      const home = comp.competitors.find(c => c.homeAway === 'home'), away = comp.competitors.find(c => c.homeAway === 'away');
      if (!home || !away) return;
      const finEl = row.querySelector('.g-fin'), resEl = row.querySelector('.g-res');
      const score = `${away.score ?? ''}–${home.score ?? ''}`;
      if (name.indexOf('STATUS_FINAL') === 0) {
        finEl.className = 'g-fin num'; finEl.innerHTML = `${score}<span class="q" style="color:var(--muted)">FINAL</span>`;
        const winner = (+home.score > +away.score) ? home.team.abbreviation : away.team.abbreviation;
        const pick = row.querySelector('.chip.pick'); const pk = pick ? pick.textContent.trim().split(/\s/)[0] : null;
        if (pk && !resEl.querySelector('.chip')) resEl.innerHTML = `<span class="chip res ${pk === winner ? 'pos' : 'neg'}">${pk === winner ? 'Hit' : 'Miss'}</span>`;
      } else if (name === 'STATUS_IN_PROGRESS' || name === 'STATUS_HALFTIME' || name === 'STATUS_END_PERIOD') {
        finEl.className = 'g-fin num inplay'; finEl.innerHTML = `${score}<span class="q">${st.shortDetail || 'LIVE'}</span>`;
      } else if (name === 'STATUS_POSTPONED' || name === 'STATUS_CANCELED') {
        finEl.className = 'g-fin muted'; finEl.textContent = st.shortDetail || 'Postponed';
      }
    });
    const strip = $('#live'); if (!strip) return;
    strip.hidden = false; strip.classList.toggle('on', inplay > 0);
    $('#live-text').textContent = ev.length ? (inplay ? `${inplay} live` : 'No games live') + ` · ${fin} final · ${up} upcoming today` : 'No NBA games today';
    $('#live-ts').textContent = LIVE.at ? 'ESPN ' + LIVE.at : '';
  }
  async function fetchLive(){
    if (document.visibilityState !== 'visible') return;
    try {
      const r = await fetch(`https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates=${etDate().replace(/-/g, '')}&limit=50`, { cache: 'no-store' });
      if (!r.ok) throw new Error(r.status);
      const j = await r.json();
      LIVE.events = j.events || []; LIVE.at = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
      overlayLive();
    } catch (e) { /* offline or endpoint changed: degrade silently, the page is complete without it */ }
  }
  if (D.live && typeof fetch === 'function') {
    fetchLive();
    LIVE.timer = setInterval(fetchLive, 60000);
    document.addEventListener('visibilitychange', () => { if (document.visibilityState === 'visible') fetchLive(); });
  }
  $('#prev').onclick = () => { if (i > 0) { i--; renderSlate(); } };
  $('#next').onclick = () => { if (i < SP.dates.length - 1) { i++; renderSlate(); } };
  $('#slate-date').onchange = e => { i = SP.dates.indexOf(e.target.value); renderSlate(); };
  document.addEventListener('keydown', e => {
    const tag = e.target.tagName; if (tag === 'SELECT' || tag === 'INPUT') return;
    if (e.key === 'ArrowLeft') $('#prev').click(); else if (e.key === 'ArrowRight') $('#next').click(); else if (e.key === 'Escape') closeTeam();
  });

  // ---- team drawer ----
  function eloCurve(pts){
    const W = 400, H = 110, L = 36, R = 6, T = 8, B = 18, pw = W - L - R, ph = H - T - B;
    const v = pts.map(p => p[1]); const lo = Math.min(...v) - 10, hi = Math.max(...v) + 10;
    const sx = k => L + k / Math.max(1, pts.length - 1) * pw, sy = e => T + (1 - (e - lo) / (hi - lo)) * ph;
    const line = pts.map((p, k) => `${sx(k).toFixed(1)},${sy(p[1]).toFixed(1)}`).join(' ');
    const y1500 = sy(1500);
    let g = '';
    if (1500 > lo && 1500 < hi) g = `<line x1="${L}" y1="${y1500.toFixed(1)}" x2="${W - R}" y2="${y1500.toFixed(1)}" stroke="var(--line-2)" stroke-dasharray="3 3"/><text x="${L - 4}" y="${(y1500 + 3).toFixed(1)}" text-anchor="end">1500</text>`;
    const col = v[v.length - 1] >= v[0] ? 'var(--pos)' : 'var(--neg)';
    const lab = (y, t) => (g && Math.abs(y - y1500) < 12) ? '' : `<text x="${L - 4}" y="${y.toFixed(1)}" text-anchor="end">${t}</text>`;
    return `<svg class="chart" viewBox="0 0 ${W} ${H}" aria-label="Elo over the season">${g}<polyline points="${line}" fill="none" stroke="${col}" stroke-width="1.75" stroke-linejoin="round"/>
      ${lab(T + 4, Math.round(hi))}${lab(T + ph, Math.round(lo))}
      <text x="${L}" y="${H - 4}">${pts[0][0].slice(5)}</text><text x="${W - R}" y="${H - 4}" text-anchor="end">${pts[pts.length - 1][0].slice(5)}</text></svg>`;
  }
  let lastFocus = null;
  function openTeam(code, src){
    const t = TP[code], rk = RK[code]; if (!t) return;
    lastFocus = src || null;
    const rec = t.rec, wl = a => `${a[0]}-${a[1]}`;
    const ups = (list, won) => list.length ? list.map(u => `<tr><td class="muted">${u.d.slice(5)}</td><td>${u.home ? 'vs' : '@'} ${tm(u.opp)}</td><td class="n">${u.pf}-${u.pa}</td><td class="n ${won ? 'pos' : 'neg'}">${pct(u.p)}</td></tr>`).join('') : '<tr><td class="muted" colspan="4">none</td></tr>';
    const sched = t.sched.slice().reverse().map(g => `<tr><td class="muted">${g.d.slice(5)}</td><td>${g.home ? 'vs' : '@'} ${tm(g.opp)}</td><td class="n">${pct(g.p)}</td><td class="n muted">${g.rest === 1 ? '<span class="warn">B2B</span>' : g.rest + 'd'}</td>
      ${'w' in g ? `<td class="${g.w ? 'pos' : 'neg'}">${g.w ? 'W' : 'L'} <span class="muted num">${g.pf}-${g.pa}</span></td><td><span class="chip res ${(g.p >= .5) === g.w ? 'pos' : 'neg'}">${(g.p >= .5) === g.w ? 'Hit' : 'Miss'}</span></td>` : '<td colspan="2" class="muted">—</td>'}</tr>`).join('');
    const l10w = rk ? rk.l10.split('').filter(c => c === 'W').length : null;
    $('#drawer').innerHTML = `
      <div class="grab"></div>
      <div class="d-h"><h2>${code}<small>${t.name} · ${t.conf}${rk ? ` · #${rk.rank} Elo ${Math.round(rk.elo)}` : ''}</small></h2><button type="button" class="btn" id="d-close" aria-label="Close team panel">Close</button></div>
      <div class="d-rec">
        <div><div class="l">Record</div><div class="v">${t.w}-${t.l}</div></div>
        <div><div class="l">Home</div><div class="v">${wl(rec.home)}</div></div>
        <div><div class="l">Away</div><div class="v">${wl(rec.away)}</div></div>
        <div><div class="l">B2B</div><div class="v ${rec.b2b[0] >= rec.b2b[1] ? 'pos' : 'neg'}">${wl(rec.b2b)}</div></div>
        <div><div class="l">Rested</div><div class="v">${wl(rec.rested)}</div></div>
        <div><div class="l">Model acc</div><div class="v">${t.model_acc == null ? '—' : pct(t.model_acc)}</div></div>
        <div><div class="l">As fav</div><div class="v">${wl(rec.fav)}</div></div>
        <div><div class="l">As dog</div><div class="v">${wl(rec.dog)}</div></div>
        <div><div class="l">L10 · Net</div><div class="v">${rk ? `${l10w}-${10 - l10w} <span class="${rk.net10 >= 0 ? 'pos' : 'neg'}" style="font-size:13px">${(rk.net10 >= 0 ? '+' : '') + rk.net10.toFixed(1)}</span>` : '—'}</div></div>
      </div>
      <h3>Elo · season</h3>${eloCurve(t.elo)}
      <div class="grid g2"><div><h3>Biggest upset wins</h3><div class="tbl compact"><table><tbody>${ups(t.upset_wins, true)}</tbody></table></div></div>
      <div><h3>Biggest upset losses</h3><div class="tbl compact"><table><tbody>${ups(t.upset_losses, false)}</tbody></table></div></div></div>
      <h3>Schedule · newest first · p = model win prob for ${code}</h3>
      <div class="tbl compact"><table><thead><tr><th>Date</th><th>Opp</th><th class="n">p(win)</th><th class="n">Rest</th><th>Result</th><th>Model</th></tr></thead><tbody>${sched}</tbody></table></div>
      <p class="note">${D.live ? 'Data: ESPN public API · updated ' + D.updated : 'Sample data'} · ${D.disclaimer}</p>`;
    $('#drawer').classList.add('open'); $('#scrim').classList.add('open');
    $('#drawer').scrollTop = 0;
    $('#d-close').onclick = closeTeam;
    $('#d-close').focus();
  }
  function closeTeam(){ $('#drawer').classList.remove('open'); $('#scrim').classList.remove('open'); if (lastFocus && lastFocus.isConnected) lastFocus.focus(); }
  $('#scrim').onclick = closeTeam;
  document.addEventListener('click', e => { const t = e.target.closest('.tm'); if (t) openTeam(t.dataset.team, t); });
  window.GP = { openTeam, closeTeam, renderSlate, setDate: d => { i = SP.dates.indexOf(d); renderSlate(); } };
  renderSlate();
})();
"""


def _trim_bundle(sp: dict, team_pages: dict, rankings: list[dict]) -> dict:
    """Only the fields the JS reads, floats to 3 dp."""
    slates = {d: [{k: g[k] for k in ("h", "a", "ph", "eh", "ea", "pick", "drv", "rh", "ra", "hp", "ap", "ok", "eid", "t") if k in g} for g in rows]
              for d, rows in sp["slates"].items()}
    teams = {}
    for code, t in team_pages.items():
        teams[code] = {"name": t["name"], "conf": t["conf"], "w": t["w"], "l": t["l"], "rec": t["rec"], "model_acc": t["model_acc"],
                       "elo": t["elo"], "upset_wins": t["upset_wins"], "upset_losses": t["upset_losses"],
                       "sched": [{k: g[k] for k in ("d", "opp", "home", "p", "rest", "pf", "pa", "w") if k in g} for g in t["sched"]]}
    rk = [{k: r[k] for k in ("team", "rank", "elo", "l10", "net10")} for r in rankings]
    return _round({"sp": {"dates": sp["dates"], "slates": slates, "running": sp["running"]}, "teams": teams, "rankings": rk})


def _body(bundle: dict, sim: dict, met: dict, pred: dict) -> tuple[str, str]:
    """Returns (body_html, data_blob_json)."""
    m, d, cal, walk = met["models"], met["data"], met["calibration"], met["walk_forward"]
    xgbm, elom, homem = m["xgb"], m["elo_logit"], m["home_baseline"]
    sp, rk, ups, tiers = bundle["season_predictions"], bundle["rankings"], bundle["upsets"], bundle["tiers"]
    from analytics import UPSET_THRESHOLD
    meta = bundle.get("meta") or {}
    live = meta.get("source") == "espn"
    updated = (meta.get("updated_at") or "")[:16].replace("T", " ") + (" UTC" if meta.get("updated_at") else "")
    n_final = meta.get("n_final") or d["games_total"]
    prior = met.get("prior_holdout")
    small = d.get("small_sample")
    last_played = sp.get("last_played_date") or sp["dates"][-1]
    season_hits = sp["running"][last_played]
    top = rk[0]
    mover = max(rk, key=lambda r: r["d_elo_30"])
    faller = min(rk, key=lambda r: r["d_elo_30"])
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    # default slate: today (ET) if it has games, else the next game day, else the last played day
    default_date = pred["date"] if pred.get("date") in sp["slates"] else last_played
    today = sp.get("today")

    blob = _trim_bundle(sp, bundle["team_pages"], rk)
    blob.update({"default_date": default_date, "baseline": round(homem["accuracy"], 3), "disclaimer": DISCLAIMER,
                 "live": live, "today": today, "updated": updated, "n_final": n_final,
                 "last_played": last_played, "season": d["test_season"]})
    data_blob = json.dumps(blob, separators=(",", ":")).replace("</", "<\\/")
    def opt(dd):
        n = len(sp["slates"][dd])
        if dd == today:
            return f'<option value="{dd}">{dd} &middot; today</option>'
        tag = " &#9656;" if today and dd > today else ""   # small right-pointing triangle marks upcoming dates
        return f'<option value="{dd}">{dd} &middot; {n} g{tag}</option>'
    date_opts = "".join(opt(dd) for dd in sp["dates"])
    best_month = max(walk, key=lambda w: w["xgb_accuracy"])

    if live:
        lede = ("Pre-game win probabilities from Elo, rest and rolling form, trained on three real NBA seasons and scored "
                f"forward in time on every {esc(d['test_season'])} game. Real games from ESPN's public API, refreshed on each run; "
                "the point is the leak-free pipeline and honest calibration, not the picks.")
        src_line = f"Data: ESPN public API &middot; updated {esc(updated)}"
        method_data = (f"Real NBA games ({', '.join(esc(x) for x in d['seasons'])}) from ESPN's unofficial, undocumented public scoreboard API, "
                       f"regular season only (playoffs and NBA Cup final are flagged; the Cup final is kept for Elo but excluded from standings). "
                       f"Game dates are US/Eastern. The {esc(d['test_season'])} test set is {d['n_test']} games so far"
                       + (" &mdash; a small sample, so the hero number will move; the full prior-season holdout is the stable read." if small else ".")
                       + " ESPN can change or withdraw the endpoint without notice.")
        hind_label = f"{esc(d['test_season'])} hindcast"
    else:
        lede = ("Pre-game win probabilities from Elo, rest and rolling form, scored forward in time on every game of the test season. "
                "Sample league, synthetic data &mdash; the point is the leak-free pipeline and honest calibration, not the picks.")
        src_line = "Sample league &middot; synthetic data"
        method_data = ("30 fictional teams, 3 seasons &times; 1230 games simulated with drifting strength, +2.5 home court, rest effects and injuries, "
                       "tuned so a perfect oracle wins ~68% (real NBA models land at 65&ndash;68%).")
        hind_label = "Season hindcast"

    kpi_sub = (f'full {esc(prior["test_season"])} holdout <span class="v">{prior["models"]["xgb"]["accuracy"]:.1%}</span> (n={prior["n_test"]})'
               if prior else f'vs <span class="v">{homem["accuracy"]:.1%}</span> home baseline &middot; n={d["n_test"]}')
    kpi_label = f'{esc(d["test_season"])} accuracy' + (' <span class="muted" style="text-transform:none;letter-spacing:0">(n=%d)</span>' % d["n_test"] if small else '')

    body = f"""
<main class="wrap">
  <div class="eyebrow">Project 03 &middot; win probability &middot; season {esc(d["test_season"])}</div>
  <h1>Game Predictor</h1>
  <p class="lede">{lede}</p>

  <section class="grid g4" aria-label="Key numbers">
    <div class="card kpi"><div class="label">{kpi_label}</div><b class="hero">{xgbm["accuracy"]:.1%}</b><div class="sub">{kpi_sub}</div></div>
    <div class="card kpi"><div class="label">Log loss</div><b>{xgbm["log_loss"]:.3f}</b><div class="sub">Elo-logit {elom["log_loss"]:.3f} &middot; home {homem["log_loss"]:.3f}</div></div>
    <div class="card kpi"><div class="label">{hind_label}</div><b>{season_hits["std_hits"]}<span class="muted" style="font-size:18px">/{season_hits["std_n"]}</span></b><div class="sub">{len([x for x in sp["dates"] if x <= last_played])} game days &middot; frozen model{' &middot; small sample' if small else ''}</div></div>
    <div class="card kpi"><div class="label">#1 Elo team</div><b>{esc(top["team"])}</b><div class="sub">{top["elo"]:.0f} &middot; {top["w"]}-{top["l"]} &middot; {esc(top["name"])}</div></div>
  </section>

  <section class="sec" id="slate" aria-label="Slate">
    <div class="sec-h"><h2>Slate</h2><span class="read" id="slate-count"></span></div>
    <div class="live" id="live" hidden><span class="dot" aria-hidden="true"></span><span id="live-text">Live scores</span><span class="muted" id="live-ts"></span></div>
    <div class="controls">
      <button type="button" class="btn" id="prev" aria-label="Previous game day">&larr; Prev</button>
      <select id="slate-date" aria-label="Game date">{date_opts}</select>
      <button type="button" class="btn" id="next" aria-label="Next game day">Next &rarr;</button>
    </div>
    <div class="slate-stats"><span class="chip" id="st-day"></span><span class="chip" id="st-std"></span><span class="spark" id="st-spark" title="season-to-date accuracy through this date; dashed = home baseline"></span></div>
    <div class="games" role="table" aria-label="Games on the selected date">
      <div class="game head" role="row"><span class="g-n">#</span><span class="g-m">Away @ Home</span><span class="g-elo">Elo A &middot; H</span><span class="g-bar">p(away) &rarr; p(home)</span><span class="g-p"><b>Away</b><b>Home</b></span><span class="g-pick">Pick</span><span class="g-drv">Drivers</span><span class="g-end"><span class="g-fin">Final</span><span class="g-res">Result</span></span></div>
      <div id="slate-body"></div>
    </div>
    <p class="hint">Tap a team code for its page. <span class="desk"><span class="kbd">&larr;</span> <span class="kbd">&rarr;</span> change the day, <span class="kbd">esc</span> closes the panel.</span></p>
    <p class="read-line"><b>Read:</b> {"Today's and upcoming games show the model's pre-game probability; played games are a hindcast" if live else "Every slate is a hindcast"}: the model was trained on seasons {esc(" + ".join(d["train_seasons"]))} only, features use games strictly before each date, and the final score is shown as a check, never as an input. Games are sorted by confidence; "Hit" means the pick won.</p>
  </section>

  <section class="sec" id="rankings" aria-label="Power rankings">
    <div class="sec-h"><h2>Power rankings</h2><span class="read">Elo as of {esc(last_played)}</span></div>
    {rankings_html(rk)}
    <p class="read-line"><b>Read:</b> {esc(top["team"])} leads at {top["elo"]:.0f}. Biggest 30-day mover is <b>{esc(mover["team"])}</b> ({mover["d_elo_30"]:+.0f}); biggest faller <b>{esc(faller["team"])}</b> ({faller["d_elo_30"]:+.0f}). Net10 is point differential per game over the last 10 (a rating proxy; no possession data here).</p>
  </section>

  <section class="sec" id="sim" aria-label="Season simulator">
    <div class="sec-h"><h2>Season simulator</h2><span class="read">Monte Carlo from {esc(sim["date"])} &middot; {sim["n_sims"]} runs{" &middot; published schedule" if sim.get("live") else ""}</span></div>
    {sim_html(sim)}
  </section>

  <section class="sec">
    <details class="more" id="upsets"><summary>Upsets &amp; confidence tiers</summary><div class="body">{upsets_html(ups, tiers, UPSET_THRESHOLD)}</div></details>
  </section>
  <section class="sec" style="margin-top:10px">
    <details class="more" id="lab"><summary>Model lab <span class="muted desk-only" style="font-weight:500;font-size:12px;margin-left:8px">XGBoost vs Elo-logit vs HGB &middot; best month {best_month["month"]} at {best_month["xgb_accuracy"]:.0%}</span></summary><div class="body">{lab_html(met)}</div></details>
  </section>

  <section class="method" id="method">
    <b>Method &amp; limits.</b> {method_data}
    Features are built by streaming one day at a time so nothing from a game day leaks into that day's rows; a re-derivation assertion runs on every train. Train on seasons {esc(" + ".join(d["train_seasons"]))}, test on {esc(d["test_season"])}, calibrator and hyper-parameters chosen on the last 60 days of training. Elo: K=20, +80 home, margin multiplier, 25% carry-over regression.
    The simulator uses Elo probabilities (not XGBoost) with in-path updates and random tiebreaks{", top-6 = direct playoff spot (no play-in)" if sim.get("live") else ""}; off/def are points per game, not per possession; no players, injuries, travel or market lines. Three learners roughly tie a one-feature Elo regression &mdash; that is the honest result, not a bug.
    <b>{esc(DISCLAIMER)}</b>
  </section>
  <div class="foot"><span>{src_line}</span><span>Python</span><span>Pandas</span><span>scikit-learn</span><span>XGBoost</span><span>built {ts}</span></div>
</main>
<div class="scrim" id="scrim"></div>
<aside class="drawer" id="drawer" aria-label="Team panel"></aside>
"""
    return body, data_blob


def _page(body: str, data_blob: str, *, standalone: bool, depth: str) -> str:
    head_css = (f"<style>{(SITE_ASSETS / 'shell.css').read_text()}</style>" if standalone
                else '<link rel="stylesheet" href="../assets/shell.css">')
    shell_js = (f"<script>{(SITE_ASSETS / 'shell.js').read_text()}</script>" if standalone
                else '<script src="../assets/shell.js"></script>')
    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="#0b0d12"><meta name="apple-mobile-web-app-capable" content="yes"><meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<title>Game Predictor · NBA Projects</title>
{head_css}
<style>{CSS}</style>
</head>
<body data-app="predictor" data-title="Game Predictor" data-mark="GP" data-depth="{depth}">
{body}
<script id="gp-data" type="application/json">{data_blob}</script>
{shell_js}
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
