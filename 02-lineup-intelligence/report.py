"""
Build the Lineups page (NBA Projects).

Two outputs from the same builder:

* ``output/report.html`` — standalone: the shared app foundation (site/assets/app.css + app-shell.js)
  is inlined so the file opens from anywhere, no CDN, no server.
* ``<site>/index.html`` (``cli.py demo --site <dir>``) — the GitHub-Pages version that references
  ``../assets/app.css`` / ``../assets/app-shell.js`` (top bar, tab bar, toast) and the PWA head tags
  from ``site/_head_snippet.html``.

The page embeds a compact JSON payload (every team's top fives + the default team's full C(n,5) table:
bitmask, predicted net / off, feasibility, rank, top contributions); the other teams' full tables are
fetched lazily from ``data/<ABBR>.json``. The builder, swaps, pair synergy, rotation and the matchup
estimate all run in the browser from a ``Map<bitmask, row>`` lookup.

Screen (mobile-first, app-polish skill): large title + team picker -> hero (the five as initials avatars,
fit score with a plain verdict, net rating) -> matchup row (sheet, labelled as an estimate) -> roster
grouped by position with checkmarks (max 5) + a live score bar -> why this works -> best swaps ->
best pairs (full heatmap in a sheet) -> rotation -> around the league (top 10, all in a sheet) ->
about this model (sheet: metrics, honesty note, data, approximations).
"""
from __future__ import annotations

import html
import itertools
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "output")
SITE_ASSETS = os.path.normpath(os.path.join(ROOT, "..", "site", "assets"))

# Learning-curve series: the production model is the one highlighted series (brand), the rest use label tiers.
SERIES = {"xgb": "var(--brand)", "hgb": "var(--label-3)", "ridge": "var(--label-2)"}
SERIES_NAME = {"xgb": "XGBoost", "hgb": "HistGB", "ridge": "Ridge"}

# Fan-facing names for the model's lineup features ("Why this works" rows, About sheet). The sign of a
# contribution says whether the trait helps or hurts, so every label names the trait neutrally.
PLAIN_LABELS = {
    "fg3_pct_mean": "Three-point shooting", "fg3_pct_std": "Gap between best and worst shooter",
    "fg3_pct_min": "Weakest shooter", "fg3a_rate_sum": "Three-point volume", "shooters": "Number of shooters",
    "non_shooters": "Players who don't shoot threes", "ast_pct_sum": "Passing", "ast_pct_max": "Best passer",
    "usage_sum": "Shot creation", "usage_max": "Go-to scorer", "usage_std": "Balance of touches",
    "handlers": "Ball handlers", "height_mean": "Size", "height_std": "Mix of sizes", "height_min": "Smallest player",
    "height_max": "Tallest player", "bigs": "Number of bigs", "reb_pct_sum": "Rebounding",
    "dbpm_sum": "Defensive impact", "blk_pct_max": "Rim protection", "blk_pct_sum": "Shot blocking",
    "stl_pct_sum": "Ball pressure", "def_rating_proxy_mean": "Individual defense", "obpm_sum": "Offensive impact",
    "obpm_min": "Weakest offensive player", "bpm_sum": "Overall player impact", "tov_pct_sum": "Turnovers",
    "tov_pct_max": "Most turnover-prone player", "minutes_mean_log": "Minutes played this season",
    "zero_minute_players": "Players without minutes", "pair_usage_absdiff_mean": "Complementary roles",
    "pair_height_absdiff_mean": "Complementary sizes", "pair_both_handlers": "Overlapping ball handlers",
    "pair_both_nonshooters": "Non-shooters sharing the floor", "shooters_x_bigs": "Shooters around bigs",
    "shooters_x_ast_max": "Shooters with a playmaker", "handlers_x_usage_max": "Creators with a go-to scorer",
    "stl_x_blk": "Steals plus rim protection", "spacing_x_rim": "Spacing with a rim protector",
}


def nickname(team: str) -> str:
    """'Cleveland Cavaliers' -> 'Cavaliers', 'Portland Trail Blazers' -> 'Trail Blazers'."""
    for two in ("Trail Blazers",):
        if team.endswith(two):
            return two
    return team.split()[-1]


# ------------------------------------------------------------------ payload ---
TOP_EMBED = 50          # fives per team embedded in the main page (the full table is fetched lazily per team)
CONTRIB_K = 2           # top-k positive and top-k negative contributions kept per five
B64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
CONTRIB_STEP = 0.2      # contributions are quantised to 0.2 and clipped to ±6.2 (one char each) to keep team files ~150 KB


def _pack_contribs(idx_vals: list[tuple[int, float]]) -> str:
    """[(feature_idx, contribution), ...] -> 2 chars per pair: B64[idx] + B64[clip(round(v / step), -31, 31) + 32]."""
    out = []
    for j, v in idx_vals:
        q = int(np.clip(round(v / CONTRIB_STEP), -31, 31)) + 32
        out.append(B64[j] + B64[q])
    return "".join(out)


def team_abbr(tt) -> str:
    from optimize import slug
    if "team_abbr" in tt.roster.columns and isinstance(tt.roster.team_abbr.iloc[0], str):
        return str(tt.roster.team_abbr.iloc[0])
    return slug(tt.team)


def _team_payload(tt, rec: dict, full: bool) -> dict:
    """Compact, columnar table for one team.

    masks     bitmasks over roster positions, delta-encoded (sorted ascending when full) · net[k] = predicted net x 100 (int)
    off[k]    predicted off x 10 (int) · feas '0'/'1' string · con[k] packed string, 2 chars per (feature_idx, contribution)
              pair (top-2 +, top-2 −, |c| > 0.15, quantised to 0.2)
    rank is recomputed in the browser (order by net) for full tables; partial tables carry `rank` explicitly. With
    full=False only the top TOP_EMBED fives (plus the best / most-used fives) are included and the browser fetches
    data/<ABBR>.json for the rest.
    """
    from optimize import lineup_profile, best_feasible_index
    n = len(tt.roster)
    plist = []
    for i in range(n):
        p = tt.player_dict(i)
        plist.append({"id": p["player_id"], "name": p["name"], "pos": p["position"], "ht": round(p["height_in"], 1),
                      "min": int(round(p["minutes"])), "fg3": round(p["fg3_pct"], 3), "tov": round(p["tov_pct"], 1),
                      "s": p["is_shooter"], "h": p["is_handler"], "b": p["is_big"], "inj": p.get("injury_status") or ""})
    masks = (1 << tt.combos).sum(1).astype(int)
    bi = best_feasible_index(tt)
    mu = rec.get("most_used_lineup") or {}
    mu_mask = None
    if mu:
        name_to_i = {tt.roster.loc[i, "player_name"]: i for i in range(n)}
        if all(nm in name_to_i for nm in mu["players"]):
            mu_mask = int(sum(1 << name_to_i[nm] for nm in mu["players"]))
    if full:
        keep = np.argsort(masks, kind="stable")                       # mask order keeps the file small and stable
    else:
        keep = list(np.argsort(-tt.pred_net, kind="stable")[:TOP_EMBED])
        for extra in (bi, *(np.where(masks == mu_mask)[0] if mu_mask is not None else ())):
            if int(extra) not in keep:
                keep.append(int(extra))
        keep = np.array(keep, dtype=int)
    con = []
    C = tt.contribs
    for k in keep:
        pairs: list = []
        if C is not None:
            row = C[k]
            order = np.argsort(row)
            pos = [j for j in order[::-1] if row[j] > 0.15][:CONTRIB_K]
            neg = [j for j in order if row[j] < -0.15][:CONTRIB_K]
            pairs = [(int(j), float(row[j])) for j in pos + neg]
        con.append(_pack_contribs(pairs))
    mk = [int(masks[k]) for k in keep]
    deltas = [mk[0]] + [b - a for a, b in zip(mk, mk[1:])] if mk else []
    pos_of = {int(pid): i for i, pid in enumerate(tt.roster.player_id)}
    hist = {}
    for key, (mins, net) in tt.hist.items():
        if all(k in pos_of for k in key):
            extra = tt.hist_meta.get(key, {}) if hasattr(tt, "hist_meta") else {}
            hist[str(sum(1 << pos_of[k] for k in key))] = [int(round(mins)), round(net, 1), int(extra.get("games", 0)), extra.get("kind", "")]
    prof = {k: round(v, 2) for k, v in lineup_profile(tt, bi).items()}
    rot = rec.get("rotation")
    if rot and "error" not in rot:
        def unit(u, mins):
            return {"fit": u["fit_score"], "net": u["predicted_net"], "rank": u["rank_all"], "min": u["minutes"],
                    "p": [[q["player_id"], q["minutes_assigned"], q["budget"]] if mins else [q["player_id"]] for q in u["players"]]}
        rot = {"s": unit(rot["starters"], True), "b": unit(rot["bench"], True), "c": unit(rot["closing"], False),
               "overlap": rot["overlap"], "bench_up": rot["bench_outranks_starters"], "net": rot["rotation_net"],
               "fit": rot["rotation_fit"], "alloc": rot["minutes_allocated"], "left": rot["minutes_unallocated"]}
    elif rot:
        rot = {"error": rot["error"]}
    return {"abbr": team_abbr(tt), "players": plist, "n": n, "total": int(len(tt.combos)), "full": bool(full),
            "masks": deltas, "net": [int(round(float(tt.pred_net[k]) * 100)) for k in keep],
            "off": [int(round(float(tt.pred_off[k]) * 10)) if np.isfinite(tt.pred_off[k]) else 0 for k in keep],
            "feas": "".join("1" if tt.feasible[k] else "0" for k in keep), "con": con,
            **({} if full else {"rank": [int(tt.rank_all[k]) for k in keep]}),
            "hist": hist, "most_used": mu_mask, "best_mask": int(masks[bi]), "profile": prof, "rotation": rot}


def _payload(tables: dict, meta: dict, all_recs: dict, players: pd.DataFrame, source: dict, default_team: str) -> tuple[dict, dict]:
    """(main-page payload with every team light + the default team full, {abbr: full team payload})."""
    from optimize import FEATURE_LABELS, MATCHUP, league_profile
    feats = meta["feature_names"]
    teams, files = {}, {}
    for team, tt in tables.items():
        rec = all_recs.get(team, {})
        full = _team_payload(tt, rec, full=True)
        files[full["abbr"]] = full
        teams[team] = full if team == default_team else _team_payload(tt, rec, full=False)
    main = {
        "fit": {"center": round(meta["fit_center"], 4), "scale": round(meta["fit_scale"], 4)},
        "features": feats, "labels": {f: PLAIN_LABELS.get(f, FEATURE_LABELS.get(f, f)) for f in feats},
        "nick": {t: nickname(t) for t in tables},
        "teams": teams, "matchup": MATCHUP,
        "league_profile": {k: round(v, 2) for k, v in league_profile(players).items()},
        "source": {"source": source.get("source", "sample"), "label": source.get("label", "Sample league"), "updated": source.get("fetched_at") or ""},
    }
    return main, files




# ------------------------------------------------------------- static svg ---
def _curve_chart(lc: dict) -> str:
    """Learning curve (held-out rank correlation vs training lineups), sized for a phone-width sheet."""
    if not lc:
        return ""
    W, H, pad_l, pad_r, pad_b, pad_t = 360, 200, 34, 92, 26, 10
    keys = ("xgboost", "ridge", "hgb")
    skey = {"ridge": "ridge", "xgboost": "xgb", "hgb": "hgb"}
    xs = lc["n_train"]
    lo = min(min(lc[k]) for k in keys)
    hi = max(max(lc[k]) for k in keys)
    ymin = max(0.0, float(np.floor((lo - 0.03) * 20) / 20))
    ymax = min(1.0, float(np.ceil((hi + 0.03) * 20) / 20))

    def x(v):
        return pad_l + (W - pad_l - pad_r) * (v - xs[0]) / max(1, (xs[-1] - xs[0]))

    def y(v):
        return pad_t + (H - pad_b - pad_t) * (1 - (v - ymin) / max(1e-9, (ymax - ymin)))
    out = [f'<svg class="chart curve" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" role="img" '
           f'aria-label="Held-out rank correlation as training data grows">']
    ticks = np.linspace(ymin, ymax, 3)
    for v in ticks:
        out.append(f'<line class="grid" x1="{pad_l}" x2="{W - pad_r + 6}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>'
                   f'<text x="{pad_l - 6}" y="{y(v) + 4:.1f}" text-anchor="end">{v:.2f}</text>')
    for n in (xs[0], xs[-1]):
        out.append(f'<text x="{x(n):.1f}" y="{H - 6}" text-anchor="middle">{n:,} lineups</text>')
    label_y = {k: y(lc[k][-1]) for k in keys}
    for _ in range(8):
        for a, b in itertools.combinations(sorted(keys, key=label_y.get), 2):
            if abs(label_y[a] - label_y[b]) < 15:
                mid = (label_y[a] + label_y[b]) / 2
                lo_k, hi_k = (a, b) if label_y[a] <= label_y[b] else (b, a)
                label_y[lo_k], label_y[hi_k] = mid - 7.5, mid + 7.5
    for key in keys[::-1]:
        col = SERIES[skey[key]]
        pts = " ".join(f"{x(a):.1f},{y(b):.1f}" for a, b in zip(xs, lc[key]))
        w = 2.2 if key == "xgboost" else 1.6
        out.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="{w}" stroke-linejoin="round" stroke-linecap="round"/>')
        out.append(f'<circle cx="{x(xs[-1]):.1f}" cy="{y(lc[key][-1]):.1f}" r="3" fill="{col}"/>')
        out.append(f'<text class="dl" x="{x(xs[-1]) + 8:.1f}" y="{label_y[key] + 4:.1f}">{SERIES_NAME[skey[key]]} {lc[key][-1]:.2f}</text>')
    out.append("</svg>")
    return "".join(out)


# -------------------------------------------------------------------- html ---
def _ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def _nice_date(ts: str) -> str:
    """'2026-11-08 16:19 UTC' -> 'Nov 8, 2026' (falls back to the raw string)."""
    import datetime as _dt
    try:
        d = _dt.datetime.strptime(ts[:10], "%Y-%m-%d")
        return f"{d.strftime('%b')} {d.day}, {d.year}"
    except (ValueError, TypeError):
        return ts or ""


_CHEV = '<svg class="chev" viewBox="0 0 8 13" aria-hidden="true"><path d="M1.5 1.5 6.5 6.5l-5 5" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>'


def _last_name(name: str) -> str:
    """'Jaren Jackson Jr.' -> 'Jackson' (suffixes skipped)."""
    p = name.split()
    return p[-2] if len(p) > 2 and p[-1].rstrip(".") in ("Jr", "Sr", "II", "III", "IV") else p[-1]


def _board_rows(all_recs: dict, current: str, limit: int | None = None) -> str:
    """Around-the-league rows: each team's best feasible five, best first. Tapping a row loads that team."""
    rows = []
    ranked = sorted(all_recs.items(), key=lambda kv: -kv[1]["recommendations"][0]["fit_score"])
    for i, (t, o) in enumerate(ranked[:limit] if limit else ranked, 1):
        r = o["recommendations"][0]
        names = ", ".join(_last_name(p["name"]) for p in r["players"])
        rows.append(
            f'<button type="button" class="row league-row{" selected" if t == current else ""}" data-team="{html.escape(t)}">'
            f'<span class="lrank num">{i}</span>'
            f'<span class="grow"><span class="title">{html.escape(t)}</span><span class="sub">{html.escape(names)}</span></span>'
            f'<span class="lfit num">{r["fit_score"]:.0f}</span></button>')
    return "".join(rows)


def _read_asset(name: str) -> str:
    p = os.path.join(SITE_ASSETS, name)
    if not os.path.exists(p):
        raise FileNotFoundError(f"shared app asset missing: {p} (expected site/assets/{name} next to the project)")
    with open(p, encoding="utf-8") as fh:
        return fh.read()


def _head_snippet(standalone: bool) -> str:
    """PWA head tags shared by every app page. The standalone report drops the ../ links (no manifest / icons next to it)."""
    p = os.path.join(os.path.dirname(SITE_ASSETS), "_head_snippet.html")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as fh:
            lines = [ln.strip() for ln in fh if ln.strip() and not ln.strip().startswith("<!--")]
    else:
        lines = ['<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">',
                 '<meta name="color-scheme" content="light dark">']
    if standalone:
        lines = [ln for ln in lines if 'href="../' not in ln]
    return "\n".join(lines)


def _source_bits(source: dict, n_lineups: int) -> dict:
    """Plain-language data description, approximations and footer per data source."""
    src = source.get("source", "sample")
    upd = _nice_date(source.get("fetched_at") or "")
    if src == "espn":
        seasons = " and ".join(source.get("seasons", []))
        return {
            "data": (f"Box scores from ESPN's public API for the {seasons} seasons: {source.get('games_used', 0):,} games, "
                     f"the latest on {html.escape(_nice_date(str(source.get('latest_game_date') or '')))}. The model learned from "
                     f"{n_lineups:,} starting and closing fives, then scored every possible five on all 30 current rosters."),
            "approx": [
                "Lineups come from box scores, not play-by-play. Each game adds its starting five, plus its closing five "
                "(the five players with the most minutes) when that's different.",
                "A lineup's net rating is the average plus-minus of its five players, scaled to 100 possessions. Possessions are "
                "estimated from shots, free throws, turnovers and offensive rebounds.",
                "Offense and defense are split by assuming the lineup's edge over the team's game result is shared evenly.",
                "Player impact numbers are box-score estimates built from the same games, so test results are a little optimistic.",
                "Rosters are ESPN's current rosters, up to 17 players by minutes. Injury labels are ESPN's status when the page was built.",
            ],
            "foot": f"Data from ESPN, updated {html.escape(upd)}",
            "hist_none": "Never started or closed a game together",
            "updated": upd,
        }
    if src == "nba_api":
        return {
            "data": (f"Five-man lineup stints from stats.nba.com ({', '.join(source.get('seasons', []))}): {n_lineups:,} lineups "
                     "with at least 10 minutes together."),
            "approx": ["Player impact numbers are on/off estimates from the same seasons."],
            "foot": f"Data from NBA Stats, updated {html.escape(upd)}",
            "hist_none": "Never played together in the data",
            "updated": upd,
        }
    return {
        "data": "A seeded, fictional league with an invented chemistry function. Scores are for demonstration only.",
        "approx": ["Teams, players and results are synthetic."],
        "foot": "Sample data (fictional league)",
        "hist_none": "Never played together in the sample",
        "updated": "",
    }


def _about(meta: dict, sb: dict, n_combos: int, n_teams: int) -> str:
    """Body of the "About this model" sheet: what fit means, how well it predicts (with the honest
    Ridge-vs-XGBoost result), what it weighs, the data and every approximation."""
    from optimize import MATCHUP, ROTATION
    ev = meta["evaluation"]
    m = ev["metrics"]
    folds = ev["folds"]
    order = sorted(("ridge", "xgboost", "hgb"), key=lambda k: -m[k]["spearman_pooled"])
    names = {"ridge": "Ridge regression", "xgboost": "XGBoost", "hgb": "Histogram gradient boosting"}
    rows = []
    for k in order + ["mean_baseline"]:
        x = m[k]
        tag = ' <span class="badge brand">In use</span>' if k == "xgboost" else ""
        sub = (f"Best on {x.get('folds_won', 0)} of {folds} test splits" if k != "mean_baseline"
               else "Guesses the average for every lineup")
        rows.append(f'<div class="row"><span class="grow"><span class="title">{names.get(k, "Average guess")}{tag}</span>'
                    f'<span class="sub">{sub}</span></span><span class="value num">{x["spearman_pooled"]:.2f}</span></div>')
    lift = m["xgboost"]["spearman_pooled"] - m["ridge"]["spearman_pooled"]
    if lift > 0.01:
        honest = (f"XGBoost ranks lineups a little better than a simple linear model ({m['xgboost']['spearman_pooled']:.2f} vs "
                  f"{m['ridge']['spearman_pooled']:.2f}). The edge is real but modest with this much data.")
    else:
        honest = (f"A simple linear model (ridge regression) ranks lineups slightly better than XGBoost on teams neither model saw "
                  f"({m['ridge']['spearman_pooled']:.2f} vs {m['xgboost']['spearman_pooled']:.2f}). With a few thousand noisy box-score "
                  f"lineups, most of the signal is simply how good the five players are. The app keeps XGBoost because it can say "
                  f"exactly how much each trait adds or costs, which powers “Why this works.”")
    imp = meta["feature_importance"][:6]
    top = imp[0][1] if imp else 1
    imp_html = "".join(
        f'<div class="imp"><div class="imp-top"><span>{html.escape(PLAIN_LABELS.get(k, k))}</span>'
        f'<span class="num muted">{100 * v:.0f}%</span></div><div class="meter"><i style="width:{100 * v / top:.0f}%"></i></div></div>'
        for k, v in imp)
    approx = "".join(f"<li>{html.escape(a)}</li>" for a in sb["approx"])
    lc = ev.get("learning_curve") or {}
    curve = _curve_chart(lc)
    return f"""
      <h4>What fit means</h4>
      <p>Fit is a 0 to 100 score for how well five players are predicted to play together, based on their predicted net rating
      (points scored minus points allowed per 100 possessions). 50 is an average lineup, 73 is a clearly good one and 88 is elite.
      All {n_combos:,} possible fives across {n_teams} rosters are scored.</p>
      <h4>How well it predicts</h4>
      <p class="muted small">How well each model orders lineups for teams it never saw in training (rank correlation, 1.00 is perfect).</p>
      <div class="list">{''.join(rows)}</div>
      <p class="note">{html.escape(honest)}</p>
      {f'<h4>More data helps</h4><p class="muted small">Test accuracy as the number of training lineups grows.</p>{curve}' if curve else ''}
      <h4>What the model weighs most</h4>
      <div class="imps">{imp_html}</div>
      <h4>Data</h4>
      <p>{sb['data']}</p>
      <h4>Estimates and approximations</h4>
      <ul class="plain">{approx}
        <li>The matchup adjustment is a hand-set rule, not part of the model: it penalizes non-shooters against strong rim protection
        and turnover-prone players against ball pressure, and rewards a size edge, capped at {MATCHUP['clip']:.0f} points either way.</li>
        <li>The rotation splits 240 minutes by each player's share of season minutes (at most {ROTATION['budget_cap']:.0f}),
        with starters at {ROTATION['starter_minutes'][0]:.0f} to {ROTATION['starter_minutes'][1]:.0f} minutes and bench players at
        {ROTATION['bench_minutes'][0]:.0f} to {ROTATION['bench_minutes'][1]:.0f}. The closing five ignores minutes.</li>
        <li>Pair scores average every five that includes both players. They show association, not cause.</li>
        <li>Scores are point predictions with no error bars. Treat close calls as ties.</li>
      </ul>"""


_INFO_ICON = ('<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="M12 11v5.5"/>'
              '<circle cx="12" cy="7.6" r=".6" fill="currentColor"/></svg>')


def _page(meta: dict, all_recs: dict, tables: dict, data_json: str, head: str, tail: str, body_attrs: str,
          source: dict, default_team: str, data_url: str) -> str:
    ev = meta["evaluation"]
    n_combos = sum(len(t.combos) for t in tables.values())
    sb = _source_bits(source, ev["n_rows"])
    team_opts = "".join(f'<option value="{html.escape(t)}"{" selected" if t == default_team else ""}>{html.escape(t)}</option>'
                        for t in sorted(all_recs))
    about = _about(meta, sb, n_combos, len(tables))
    check = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6.5 12.5l3.6 3.6 7.4-8.2"/></svg>'

    return f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
{head}
<meta name="description" content="Pick five players from any NBA roster and see how well they fit together: a predicted lineup score, net rating, best swaps, rotation and best pairs.">
<title>Lineups | NBA Projects</title>
<style>{_CSS}</style>
</head>
<body data-app="lineup" data-title="Lineups"{body_attrs}>
<template id="top-actions"><button type="button" class="icon-btn" data-sheet="aboutSheet" aria-label="About this model">{_INFO_ICON}</button></template>
<main class="page wide" id="app" data-url="{html.escape(data_url)}" data-hist-none="{html.escape(sb['hist_none'])}">
  <div class="head">
    <h1 class="large-title">Lineups</h1>
    <label class="teampick">
      <span class="tp-name" id="teamName">{html.escape(default_team)}</span>
      <svg class="tp-chev" viewBox="0 0 12 8" aria-hidden="true"><path d="M1 1.5l5 5 5-5"/></svg>
      <select id="teamSel" aria-label="Team">{team_opts}</select>
    </label>
  </div>

  <div class="layout">
    <div class="col-a">
      <section class="hero card" id="hero" aria-label="Your lineup">
        <div class="five" id="five"></div>
        <div class="score" aria-live="polite">
          <div class="fit-v num" id="fitV">–</div>
          <div class="verdict" id="verdict">&nbsp;</div>
          <div class="rankline" id="rankline">&nbsp;</div>
        </div>
        <div class="nets">
          <div class="stat"><div class="v num" id="netV">–</div><div class="c">Net rating per 100 possessions</div></div>
          <div class="stat"><div class="v num" id="offdefV">–</div><div class="c">Points scored and allowed per 100</div></div>
        </div>
        <p class="hist" id="hist">&nbsp;</p>
        <div class="matchnote" id="matchNote" hidden></div>
        <div class="actions">
          <button type="button" class="btn sm" id="btnBest">Best five</button>
          <button type="button" class="btn sm" id="btnMost">Most used</button>
          <button type="button" class="btn sm plain" id="btnClear">Clear</button>
        </div>
      </section>

      <div class="list matchup-list">
        <button type="button" class="row" data-sheet="matchupSheet">
          <span class="grow title">Matchup</span><span class="value" id="oppVal">None</span>{_CHEV}
        </button>
      </div>

      <section class="section" id="rosterSec" aria-label="Roster">
        <div class="section-h"><h2>Roster</h2><span class="count num" id="count">0 of 5</span></div>
        <div id="roster" role="group" aria-label="Roster"></div>
      </section>
    </div>

    <div class="col-b">
      <section class="section first" id="whySec" aria-label="Why this works">
        <div class="section-h"><h2 id="whyH">Why this works</h2></div>
        <div class="list" id="why"></div>
        <p class="section-foot" id="whyFoot"></p>
      </section>

      <section class="section" id="swapSec" aria-label="Best swaps">
        <div class="section-h"><h2>Best swaps</h2></div>
        <div class="list" id="swaps"></div>
        <p class="section-foot" id="swapFoot"></p>
      </section>

      <section class="section" id="pairSec" aria-label="Best pairs">
        <div class="section-h"><h2>Best pairs</h2><button type="button" class="more linkbtn" data-sheet="heatSheet">See all</button></div>
        <div class="pairs">
          <div><h3 class="sub-h">Work best together</h3><div class="list" id="pairsBest"></div></div>
          <div><h3 class="sub-h">Struggle together</h3><div class="list" id="pairsWorst"></div></div>
        </div>
        <p class="section-foot">Compared with the team's average lineup. Tap a pair to try it.</p>
      </section>
    </div>
  </div>

  <section class="section" id="rotationSec" aria-label="Rotation">
    <div class="section-h"><h2>Rotation</h2></div>
    <div class="rot" id="rotation"></div>
  </section>

  <section class="section" id="leagueSec" aria-label="Around the league">
    <div class="section-h"><h2>Around the league</h2><button type="button" class="more linkbtn" data-sheet="leagueSheet">See all</button></div>
    <div class="list" id="leagueTop">{_board_rows(all_recs, default_team, 10)}</div>
    <p class="section-foot">Each team's best possible five. Tap a team to open it.</p>
  </section>

  <section class="section">
    <div class="list">
      <button type="button" class="row" data-sheet="aboutSheet"><span class="grow title">About this model</span>{_CHEV}</button>
    </div>
    <p class="footer-note">{sb['foot']}. Lineup scores are model estimates, not official stats.</p>
  </section>
</main>

<div class="livebar" id="livebar" aria-hidden="true">
  <button type="button" class="lb-in" id="lbBtn" tabindex="-1">
    <span class="lb-av" id="lbAv"></span>
    <span class="lb-txt"><b id="lbCount">0 of 5</b><span id="lbSub">Pick five players</span></span>
    <span class="lb-fit"><b class="num" id="lbFit">–</b><small>Fit</small></span>
  </button>
</div>

<div class="scrim" id="scrim"></div>

<div class="sheet" id="matchupSheet" role="dialog" aria-modal="true" tabindex="-1" aria-labelledby="matchH" inert>
  <div class="grabber"></div>
  <div class="sheet-h"><h3 id="matchH">Matchup</h3><button type="button" class="btn plain sm" data-close>Done</button></div>
  <p class="muted small">Pick an opponent for a rough read on how your five matches up with their best lineup.</p>
  <label class="flabel" for="oppSel">Opponent</label>
  <select id="oppSel" class="field"><option value="">None</option></select>
  <div class="opp-result" id="oppResult"></div>
  <p class="note"><span class="badge warn">Estimate</span> This uses simple hand-set rules for size, rim protection and ball pressure.
  It isn't part of the prediction model.</p>
</div>

<div class="sheet" id="heatSheet" role="dialog" aria-modal="true" tabindex="-1" aria-labelledby="heatH" inert>
  <div class="grabber"></div>
  <div class="sheet-h"><h3 id="heatH">All pairs</h3><button type="button" class="btn plain sm" data-close>Done</button></div>
  <p class="muted small" id="heatSub">Each square compares lineups with both players to the team's average. Tap one for details.</p>
  <div class="heatwrap" id="heatmap"></div>
  <div class="heatlegend"><span id="scaleLo">Worse</span><i></i><span id="scaleHi">Better</span></div>
  <div class="heatval" id="heatVal"></div>
</div>

<div class="sheet" id="leagueSheet" role="dialog" aria-modal="true" tabindex="-1" aria-labelledby="leagueH" inert>
  <div class="grabber"></div>
  <div class="sheet-h"><h3 id="leagueH">Every team's best five</h3><button type="button" class="btn plain sm" data-close>Done</button></div>
  <div class="list" id="leagueAll">{_board_rows(all_recs, default_team)}</div>
</div>

<div class="sheet" id="aboutSheet" role="dialog" aria-modal="true" tabindex="-1" aria-labelledby="aboutH" inert>
  <div class="grabber"></div>
  <div class="sheet-h"><h3 id="aboutH">About this model</h3><button type="button" class="btn plain sm" data-close>Done</button></div>
  <div class="about">{about}</div>
</div>

<template id="checkIcon">{check}</template>
<script id="data" type="application/json">{data_json}</script>
{tail}
<script>{_JS}</script>
</body></html>"""


def build_report(meta: dict, all_recs: dict, tables: dict | None = None, players: pd.DataFrame | None = None,
                 out_dir: str = OUT_DIR, site_dir: str | None = None, source: dict | None = None) -> list[str]:
    """Write output/report.html (+ output/data/<ABBR>.json) and, if site_dir is given, <site_dir>/index.html
    (+ <site_dir>/data/<ABBR>.json). Returns the HTML paths."""
    os.makedirs(out_dir, exist_ok=True)
    if players is None:
        players = pd.read_csv(os.path.join(ROOT, "data", "players.csv"))
    if tables is None:
        from optimize import score_team, team_list
        lineups = pd.read_csv(os.path.join(ROOT, "data", "lineups.csv"))
        tables = {t: score_team(t, players, lineups) for t in team_list(players)}
    if source is None:
        sp = os.path.join(ROOT, "data", "source.json")
        source = json.load(open(sp)) if os.path.exists(sp) else {"source": "sample", "label": "Sample league"}
    default_team = max(all_recs, key=lambda t: all_recs[t]["recommendations"][0]["fit_score"])
    main, files = _payload(tables, meta, all_recs, players, source, default_team)
    data_json = json.dumps(main, separators=(",", ":")).replace("</", "<\\/")
    css, js = _read_asset("app.css"), _read_asset("app-shell.js")
    paths = []

    def write_team_files(root: str) -> None:
        d = os.path.join(root, "data")
        os.makedirs(d, exist_ok=True)
        for abbr, payload in files.items():
            with open(os.path.join(d, f"{abbr}.json"), "w", encoding="utf-8") as fh:
                json.dump(payload, fh, separators=(",", ":"))
        for stale in os.listdir(d):                         # drop files of teams that no longer exist (sample -> real)
            if stale.endswith(".json") and stale[:-5] not in files:
                os.remove(os.path.join(d, stale))

    # standalone: foundation inlined; data-depth points the top-bar / tab-bar links back at the site folder
    depth = os.path.relpath(os.path.dirname(SITE_ASSETS), out_dir).replace(os.sep, "/") + "/"
    page = _page(meta, all_recs, tables, data_json, head=_head_snippet(True) + f"\n<style>{css}</style>",
                 tail=f"<script>{js.replace('</', '<\\/')}</script>", body_attrs=f' data-depth="{html.escape(depth)}"',
                 source=source, default_team=default_team, data_url="data/")
    p = os.path.join(out_dir, "report.html")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(page)
    write_team_files(out_dir)
    paths.append(p)
    for stale in ("theme.css",):
        sp = os.path.join(out_dir, stale)
        if os.path.exists(sp):
            os.remove(sp)

    if site_dir:
        os.makedirs(site_dir, exist_ok=True)
        head = (_head_snippet(False) + '\n<link rel="stylesheet" href="../assets/app.css">'
                '\n<script src="../assets/app-shell.js" defer></script>')
        page = _page(meta, all_recs, tables, data_json, head=head, tail="", body_attrs="",
                     source=source, default_team=default_team, data_url="data/")
        p = os.path.join(site_dir, "index.html")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(page)
        write_team_files(site_dir)
        paths.append(p)
    return paths


# -------------------------------------------------------------------- css ---
# Page-only styles. Tokens, chrome, lists, rows, stats, buttons, sheets, badges come from site/assets/app.css.
_CSS = r"""
body{padding-bottom:calc(var(--tabbar-h) + var(--sab) + 96px)}
@media (min-width:900px){body{padding-bottom:120px}}
.small{font-size:var(--t-foot);line-height:1.4}
.row .grow > .title,.row .grow > .sub{display:block}
.sr{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0)}

/* header: large title + team picker (native select styled as a field) */
.head{display:flex;flex-direction:column;gap:10px}
.teampick{position:relative;display:flex;align-items:center;gap:10px;height:52px;padding:0 16px;border-radius:14px;
  background:var(--surface);box-shadow:var(--shadow-1);cursor:pointer;user-select:none;transition:transform 120ms var(--ease)}
.teampick:active{transform:scale(.985)}
.teampick:focus-within{outline:2px solid var(--brand);outline-offset:2px}
.tp-name{flex:1;min-width:0;font:600 var(--t-body)/1.2 var(--font);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.tp-chev{width:12px;height:8px;flex:none;fill:none;stroke:var(--label-3);stroke-width:1.8;stroke-linecap:round;stroke-linejoin:round}
.teampick select{position:absolute;inset:0;width:100%;height:100%;opacity:0;font-size:16px;cursor:pointer;appearance:none;-webkit-appearance:none;border:0}
@media (min-width:900px){
  .head{flex-direction:row;align-items:flex-end;justify-content:space-between;gap:24px}
  .head .large-title{margin-bottom:0}
  .teampick{width:340px;flex:none}
}

/* layout */
.layout{display:block;margin-top:20px}
@media (min-width:900px){
  .layout{display:grid;grid-template-columns:minmax(0,440px) minmax(0,1fr);gap:0 40px;align-items:start}
  .col-b .section.first{margin-top:0}
}

/* hero */
.hero{padding:20px 16px 16px;text-align:center;scroll-margin-top:calc(var(--topbar-h) + var(--sat) + 12px)}
.five{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:4px;margin-bottom:18px}
.slot{display:flex;flex-direction:column;align-items:center;gap:6px;min-width:0}
.slot .nm{font:600 var(--t-foot)/1.2 var(--font);max-width:100%;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.slot .ps{font:400 var(--t-cap)/1 var(--font);color:var(--label-2)}
.av{width:36px;height:36px;border-radius:50%;flex:none;display:grid;place-items:center;background:var(--fill);color:var(--label);
  font:600 var(--t-foot)/1 var(--font);letter-spacing:.01em;user-select:none}
.av.lg{width:52px;height:52px;font-size:var(--t-callout)}
.av.open{background:none;box-shadow:inset 0 0 0 1.5px var(--fill-2);color:var(--label-3)}
.av.open svg{width:18px;height:18px;fill:none;stroke:currentColor;stroke-width:2;stroke-linecap:round}
.fit-v{font:700 64px/1 var(--font-display);letter-spacing:-.03em;color:var(--brand-text);min-height:64px}
.fit-v small{font:500 var(--t-body)/1 var(--font);color:var(--label-2);letter-spacing:0;margin-left:3px}
.fit-v.none{color:var(--label-3);font-weight:300}
.verdict{font:600 var(--t-body)/1.3 var(--font);margin-top:8px;min-height:22px;text-wrap:balance}
.rankline{font:400 var(--t-foot)/1.3 var(--font);color:var(--label-2);margin-top:3px;min-height:17px}
.nets{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:18px;padding-top:16px;position:relative;text-align:left}
.nets::before{content:"";position:absolute;top:0;left:0;right:0;height:.5px;background:var(--separator)}
.nets .stat .v{font-size:var(--t-1);font-weight:600}
.nets .stat .v .tri{font-size:.55em;vertical-align:.25em;margin-right:3px}
.nets .stat .v .faint{font-weight:500}
.hist{margin:12px 0 0;font:400 var(--t-foot)/1.4 var(--font);color:var(--label-2);text-align:left;min-height:18px}
.matchnote{margin-top:10px;display:flex;align-items:center;gap:8px;text-align:left;font:400 var(--t-foot)/1.35 var(--font);color:var(--label-2)}
.matchnote b{color:var(--label);font-weight:600}
.actions{display:flex;gap:8px;margin-top:16px}
.actions .btn{flex:1;height:44px}
.actions .btn.plain{flex:0 0 auto}
.hero.skel-on .fit-v{color:transparent;background:var(--fill);border-radius:12px;width:96px;margin:0 auto;animation:skel 1.2s linear infinite;
  background:linear-gradient(90deg,var(--fill) 0%,var(--fill-2) 50%,var(--fill) 100%);background-size:200% 100%}

.matchup-list{margin-top:12px}

/* roster */
.count{font:400 var(--t-sub)/1 var(--font);color:var(--label-2)}
.count.full{color:var(--brand-text);font-weight:600}
.sub-h{font:600 var(--t-foot)/1.2 var(--font);color:var(--label-2);margin:18px 4px 8px}
#roster .sub-h:first-child{margin-top:4px}
.prow{min-height:60px}
.prow .title{font-weight:500}
.prow .sub{display:block}
.prow.dnp .grow,.prow.dnp .av{opacity:.55}
.ck{width:26px;height:26px;border-radius:50%;flex:none;box-shadow:inset 0 0 0 1.5px var(--label-3);display:grid;place-items:center;
  transition:background-color 160ms var(--ease),box-shadow 160ms var(--ease)}
.ck svg{width:18px;height:18px;fill:none;stroke:var(--on-brand);stroke-width:2.4;stroke-linecap:round;stroke-linejoin:round;opacity:0;transform:scale(.6);
  transition:opacity 160ms var(--ease),transform 160ms var(--ease)}
.prow[aria-pressed="true"] .ck{background:var(--brand);box-shadow:none}
.prow[aria-pressed="true"] .ck svg{opacity:1;transform:none}
.prow[aria-pressed="true"] .av{background:var(--brand-tint);color:var(--brand-text)}
@media (hover:hover){a.row:hover,button.row:hover{background:var(--surface-2)}}

/* why this works: contribution rows */
.why-row{min-height:56px}
.why-row .grow{display:flex;flex-direction:column;gap:7px}
.why-row .title{white-space:normal;font-size:var(--t-callout)}
.cbar{position:relative;height:6px;border-radius:3px;background:var(--fill)}
.cbar::after{content:"";position:absolute;left:50%;top:-3px;bottom:-3px;width:1.5px;margin-left:-.75px;background:var(--label-3);border-radius:1px}
.cbar i{position:absolute;top:0;height:100%}
.cbar i.p{left:50%;background:var(--pos);border-radius:0 3px 3px 0}
.cbar i.n{right:50%;background:var(--neg);border-radius:3px 0 0 3px}
.delta{font:600 var(--t-body)/1 var(--font);font-variant-numeric:tabular-nums;white-space:nowrap;min-width:52px;text-align:right}
.empty-row{padding:16px;color:var(--label-2);font-size:var(--t-sub)}

/* swaps */
.swap .title b{font-weight:600}
.swap .title{white-space:normal}

/* pairs */
.pairs{display:grid;gap:0}
.pairs > div,.rot > div,.layout > div{min-width:0}
.pairs .sub-h{margin-top:4px}
.pairs > div + div .sub-h{margin-top:18px}
@media (min-width:600px) and (max-width:899px){.pairs{grid-template-columns:1fr 1fr;gap:16px}.pairs > div + div .sub-h{margin-top:4px}}
.duo{display:flex;flex:none;gap:3px}
.duo .av{width:30px;height:30px;font-size:var(--t-cap);box-shadow:0 0 0 2px var(--surface)}

/* rotation */
.rot{display:grid;gap:0}
.rot > div + div .sub-h{margin-top:20px}
@media (min-width:900px){.rot{grid-template-columns:repeat(3,minmax(0,1fr));gap:20px}.rot > div + div .sub-h{margin-top:4px}}
.rot .sub-h{display:flex;justify-content:space-between;align-items:baseline;margin-top:4px}
.rot .sub-h span{font-weight:400}
.rot .row{min-height:44px}
.rot .usefive{color:var(--brand-text);font-weight:500;justify-content:center}
.rot-foot{grid-column:1/-1}

/* around the league */
.league-row .lrank{width:22px;flex:none;color:var(--label-2);font:500 var(--t-sub)/1 var(--font);text-align:center}
.league-row .title{display:block}
.league-row .sub{display:block;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.league-row .lfit{font:600 var(--t-body)/1 var(--font);min-width:32px;text-align:right}
.league-row.selected .title{color:var(--brand-text);font-weight:600}
.linkbtn{appearance:none;border:0;background:none;padding:10px 0 10px 12px;margin:-10px 0;cursor:pointer;color:var(--brand-text);font:400 var(--t-sub)/1 var(--font)}
.linkbtn:active{opacity:.5}

/* live score bar: appears once the hero scrolls away */
.livebar{position:fixed;left:0;right:0;bottom:calc(var(--tabbar-h) + var(--sab) + 8px);z-index:45;padding:0 12px;pointer-events:none;
  opacity:0;transform:translateY(12px);transition:opacity var(--dur) var(--ease),transform var(--dur) var(--ease)}
.livebar.show{opacity:1;transform:none}
.livebar.show .lb-in{pointer-events:auto}
.lb-in{appearance:none;border:0;width:100%;max-width:560px;margin:0 auto;display:flex;align-items:center;gap:12px;height:60px;padding:0 16px 0 12px;
  border-radius:16px;background:var(--bar);backdrop-filter:blur(20px) saturate(180%);-webkit-backdrop-filter:blur(20px) saturate(180%);
  box-shadow:var(--shadow-2),0 0 0 .5px var(--separator);color:var(--label);text-align:left;cursor:pointer}
.lb-in:active{transform:scale(.985)}
.lb-av{display:grid;grid-template-columns:repeat(5,8px);gap:4px;flex:none;padding-left:4px}
.lb-av i{width:8px;height:8px;border-radius:50%;background:var(--fill-2);transition:background-color 160ms var(--ease)}
.lb-av i.on{background:var(--brand)}
.lb-txt{flex:1;min-width:0;display:flex;flex-direction:column;gap:2px}
.lb-txt b{font:600 var(--t-sub)/1.2 var(--font)}
.lb-txt span{font:400 var(--t-foot)/1.2 var(--font);color:var(--label-2);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.lb-fit{display:flex;flex-direction:column;align-items:flex-end;gap:2px}
.lb-fit b{font:700 var(--t-2)/1 var(--font-display);color:var(--brand-text);transition:transform 200ms var(--ease)}
.lb-fit b.bump{transform:scale(1.15)}
.lb-fit small{font:400 var(--t-cap)/1 var(--font);color:var(--label-2)}
@media (min-width:900px){.livebar{bottom:24px}}
html.sheet-open .livebar{opacity:0;pointer-events:none}
.toast{bottom:calc(var(--tabbar-h) + var(--sab) + 80px)}
@media (min-width:900px){.toast{bottom:96px}}
.hero.partial .nets,.hero.partial .hist{display:none}

/* sheets */
.sheet:focus{outline:none}
.sheet-h{display:flex;align-items:center;justify-content:space-between;gap:12px;margin:0 -8px 4px 0}
.sheet-h h3{margin:4px 0}
.sheet .btn.plain.sm{height:44px;font-weight:600;font-size:var(--t-body)}
.sheet p{margin:0 0 12px;line-height:1.45}
.sheet .note{background:var(--fill);border-radius:12px;padding:12px 14px;font-size:var(--t-sub);margin-top:12px}
.sheet .note .badge{margin-right:6px;vertical-align:1px}
.flabel{display:block;font:600 var(--t-foot)/1 var(--font);color:var(--label-2);margin:16px 4px 8px}
.opp-result{min-height:24px;margin:14px 2px 0;font-size:var(--t-sub);color:var(--label-2)}
.opp-result b{color:var(--label);font-weight:600}
.sheet .list{box-shadow:none;background:var(--surface-2)}
@media (prefers-color-scheme:dark){.sheet .list{background:var(--surface-2)}}
.about h4{font:600 var(--t-body)/1.3 var(--font);margin:24px 0 8px}
.about h4:first-child{margin-top:8px}
.about p{font-size:var(--t-sub);color:var(--label)}
.about p.muted{color:var(--label-2)}
.about .row .title .badge{margin-left:6px;vertical-align:1px}
.about .row .title{white-space:normal}
ul.plain{margin:0;padding:0 0 0 18px;font-size:var(--t-sub);line-height:1.45}
ul.plain li{margin:0 0 8px}
.imps{display:flex;flex-direction:column;gap:12px}
.imp-top{display:flex;justify-content:space-between;gap:12px;font-size:var(--t-sub);margin-bottom:6px}
.imp .meter > i{background:var(--label-3)}
.imp:first-child .meter > i{background:var(--brand)}
svg.curve{margin:4px 0 4px}
svg.curve .dl{font-size:12px;fill:var(--label-2)}

/* heatmap in a sheet: horizontal scroll */
.heatwrap{overflow-x:auto;-webkit-overflow-scrolling:touch;margin:0 calc(-1 * var(--margin));padding:0 var(--margin);overscroll-behavior-x:contain}
.heatwrap svg{display:block;max-width:none}
.heatwrap text{font:400 12px var(--font);fill:var(--label-2)}
.heat-cell{cursor:pointer}
.heat-cell.on{stroke:var(--label);stroke-width:2.5}
.heatlegend{display:flex;align-items:center;gap:8px;font-size:var(--t-foot);color:var(--label-2);margin:12px 0 0}
.heatlegend i{flex:0 0 120px;height:8px;border-radius:4px;background:linear-gradient(90deg,var(--neg),var(--surface-2),var(--pos))}
.heatval{margin-top:14px;min-height:50px;display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;gap:10px;font-size:var(--t-sub);color:var(--label-2)}
.heatval b{color:var(--label);font-weight:600}

/* loading + error */
.skrow{display:flex;align-items:center;gap:12px;min-height:56px;padding:11px 16px}
.skrow .skel{height:14px}
.err{padding:20px 16px;text-align:center}
.err p{margin:0 0 12px;color:var(--label-2);font-size:var(--t-sub)}
"""


# --------------------------------------------------------------------- js ---
_JS = r"""
(function () {
  const D = JSON.parse(document.getElementById('data').textContent);
  const $ = id => document.getElementById(id);
  const APP = $('app');
  const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
  const fit = net => 100 / (1 + Math.exp(-(net - D.fit.center) / D.fit.scale));
  const SUF = /^(Jr\.?|Sr\.?|II|III|IV)$/;
  const last = n => { const p = n.split(' '); return p.length > 2 && SUF.test(p[p.length - 1]) ? p[p.length - 2] : p[p.length - 1]; };
  const initials = n => { const p = n.split(' '); return (p[0][0] + (last(n)[0] || '')).toUpperCase(); };
  const sgn = (v, d) => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(d);
  const ord = n => n + (n % 100 >= 11 && n % 100 <= 13 ? 'th' : ['th', 'st', 'nd', 'rd'][n % 10] || 'th');
  const ht = h => { let f = Math.floor(h / 12), i = Math.round(h - f * 12); if (i === 12) { f++; i = 0; } return f + '′' + i + '″'; };
  const plural = (n, w) => `${n} ${w}${n === 1 ? '' : 's'}`;
  const toast = m => { if (window.App && window.App.toast) window.App.toast(m); };
  const haptic = () => { try { navigator.vibrate && navigator.vibrate(10); } catch (e) {} };
  const reduced = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const CHECK = $('checkIcon').innerHTML;
  const GROUP = { G: 'Guards', F: 'Forwards', C: 'Centers' }, POSN = { G: 'Guard', F: 'Forward', C: 'Center' }, PORD = { G: 0, F: 1, C: 2 };
  const DATA_URL = APP.dataset.url || 'data/';
  const HIST_NONE = APP.dataset.histNone || 'Never played together in the data';

  let team = null, T = null, lookup = null, sel = new Set(), opp = '', byId = null, loadSeq = 0, loadState = 'ok';
  const teams = Object.keys(D.teams);

  // columnar payload -> rows [mask, net, off, feasible, rank, [feature_idx, contribution, ...]]
  const B64 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_', STEP = 0.2;
  function unpack(str) { const out = []; for (let k = 0; k < str.length; k += 2) out.push(B64.indexOf(str[k]), (B64.indexOf(str[k + 1]) - 32) * STEP); return out; }
  function inflate(T) {
    if (T.combos) return T;
    const n = T.masks.length, order = Array.from({ length: n }, (_, i) => i).sort((a, b) => T.net[b] - T.net[a]);
    const rank = new Array(n);
    if (T.rank) T.rank.forEach((r, i) => rank[i] = r); else order.forEach((i, r) => rank[i] = r + 1);
    let m = 0;
    T.combos = T.masks.map((d, i) => { m += d; return [m, T.net[i] / 100, T.off[i] / 10, +T.feas[i], rank[i], unpack(T.con[i])]; });
    delete T.masks; delete T.net; delete T.off; delete T.feas; delete T.con; delete T.rank;
    return T;
  }

  function bind(name) {
    team = name; T = inflate(D.teams[name]);
    lookup = new Map(T.combos.map(r => [r[0], r]));
    byId = new Map(T.players.map((p, i) => [p.id, i]));
  }

  function setTeam(name, opts) {
    opts = opts || {};
    bind(name);
    $('teamSel').value = name; $('teamName').textContent = name;
    const os = $('oppSel');
    if (opp === name) opp = '';
    os.innerHTML = '<option value="">None</option>' + teams.filter(t => t !== name).sort().map(t => `<option value="${esc(t)}">${esc(t)}</option>`).join('');
    os.value = opp;
    maskToSel(T.best_mask);
    loadState = T.full ? 'ok' : 'loading';
    document.querySelectorAll('.league-row').forEach(r => r.classList.toggle('selected', r.dataset.team === name));
    try { history.replaceState(null, '', '#' + T.abbr); } catch (e) {}
    renderRoster(); renderPairs(); renderRotation(); update();
    if (!T.full) loadFull(name);
  }

  // The page embeds every team's top fives; the full C(n,5) table is fetched on demand (~100-250 KB per team).
  function loadFull(name) {
    const seq = ++loadSeq, abbr = D.teams[name].abbr;
    loadState = 'loading';
    fetch(DATA_URL + abbr + '.json', { cache: 'force-cache' }).then(r => { if (!r.ok) throw new Error(r.status); return r.json(); })
      .then(full => {
        D.teams[name] = full;
        if (seq !== loadSeq || team !== name) return;
        const keep = [...sel].map(i => T.players[i].id);
        bind(name); loadState = 'ok';
        sel = new Set(keep.map(id => byId.get(id)).filter(i => i !== undefined));
        renderPairs(); update();
      })
      .catch(() => {
        if (seq !== loadSeq || team !== name) return;
        loadState = 'error';
        renderPairs(); update();
      });
  }
  function maskToSel(mask) { sel = new Set(); for (let i = 0; i < T.n; i++) if (mask & (1 << i)) sel.add(i); }
  const selMask = () => [...sel].reduce((m, i) => m | (1 << i), 0);
  const selIds = () => [...sel].sort((a, b) => (PORD[T.players[a].pos] ?? 1) - (PORD[T.players[b].pos] ?? 1) || T.players[b].min - T.players[a].min);

  // ---- roster: grouped by position, checkmark when selected, max five
  const injBadge = s => !s ? '' : /out/i.test(s) ? '<span class="badge neg">Out</span>' : /day/i.test(s) ? '<span class="badge warn">Day-to-day</span>' : `<span class="badge warn">${esc(s)}</span>`;
  function prow(i) {
    const p = T.players[i], bits = [ht(p.ht)];
    if (p.min <= 0) bits.push('no minutes yet'); else if (p.fg3 > 0) bits.push(`${Math.round(p.fg3 * 100)}% from three`);
    return `<button type="button" class="row prow${p.min <= 0 ? ' dnp' : ''}" data-i="${i}" aria-pressed="${sel.has(i)}">
      <span class="av" aria-hidden="true">${initials(p.name)}</span>
      <span class="grow"><span class="title">${esc(p.name)}</span><span class="sub">${bits.join(', ')}</span></span>
      ${injBadge(p.inj)}<span class="ck" aria-hidden="true">${CHECK}</span></button>`;
  }
  function renderRoster() {
    const groups = { G: [], F: [], C: [] };
    T.players.forEach((p, i) => (groups[p.pos] || groups.F).push(i));
    $('roster').innerHTML = ['G', 'F', 'C'].filter(k => groups[k].length).map(k =>
      `<h3 class="sub-h">${GROUP[k]}</h3><div class="list">${groups[k].map(prow).join('')}</div>`).join('');
  }
  $('roster').addEventListener('click', e => { const b = e.target.closest('.prow'); if (b) toggle(+b.dataset.i); });
  function toggle(i) {
    if (sel.has(i)) sel.delete(i);
    else if (sel.size >= 5) { toast('Your five is full. Remove a player first.'); haptic(); return; }
    else { sel.add(i); if (sel.size === 5) haptic(); }
    syncRoster(); update();
  }
  function syncRoster() { document.querySelectorAll('#roster .prow').forEach(el => el.setAttribute('aria-pressed', sel.has(+el.dataset.i))); }

  function shb(ids) { const ps = ids.map(i => T.players[i]); return [ps.filter(p => p.s).length, ps.filter(p => p.h).length, ps.filter(p => p.b).length]; }
  function heightMean(ids) { return ids.reduce((a, i) => a + T.players[i].ht, 0) / ids.length; }
  function adjustment(ids) {
    if (!opp) return null;
    const O = D.teams[opp].profile, L = D.league_profile, M = D.matchup;
    const [s] = shb(ids), ns = 5 - s;
    const tov = ids.reduce((a, i) => a + T.players[i].tov, 0);
    const a = M.rim_vs_nonshooters * (O.rim - L.rim) * Math.max(0, ns - 1)
            + M.hands_vs_turnovers * (O.hands - L.hands) * (tov - L.tov)
            + M.size_edge * (heightMean(ids) - O.height);
    return Math.max(-M.clip, Math.min(M.clip, a));
  }

  function verdict(rank, total) {
    const nick = D.nick[team] || team;
    if (rank === 1) return `Best possible lineup for the ${nick}`;
    const p = rank / total * 100;
    if (p <= 50) return `Top ${p < 1 ? 1 : Math.ceil(p)}% of possible lineups for the ${nick}`;
    return `Bottom ${Math.max(1, Math.ceil((total - rank + 1) / total * 100))}% of possible lineups for the ${nick}`;
  }
  const skRows = k => Array.from({ length: k }, (_, j) => `<div class="skrow"><div class="skel" style="width:${[62, 48, 56, 40, 52][j % 5]}%"></div></div>`).join('');

  function renderFive(ids) {
    let s = '';
    for (let k = 0; k < 5; k++) {
      const i = ids[k];
      if (i === undefined) { s += `<div class="slot"><span class="av lg open" aria-hidden="true"><svg viewBox="0 0 24 24"><path d="M12 7v10M7 12h10"/></svg></span><span class="nm faint">Open</span><span class="ps">&nbsp;</span></div>`; continue; }
      const p = T.players[i];
      s += `<div class="slot"><span class="av lg" aria-hidden="true">${initials(p.name)}</span><span class="nm">${esc(last(p.name))}</span><span class="ps">${POSN[p.pos] || p.pos}</span></div>`;
    }
    $('five').innerHTML = s;
    $('five').setAttribute('aria-label', ids.length ? 'Your five: ' + ids.map(i => T.players[i].name).join(', ') : 'No players selected');
    $('lbAv').innerHTML = [0, 1, 2, 3, 4].map(k => `<i class="${k < ids.length ? 'on' : ''}"></i>`).join('');
  }

  let lastFit = null;
  function setLive(n, f, sub) {
    $('lbCount').textContent = `${n} of 5`;
    $('lbSub').textContent = sub;
    const el = $('lbFit'), txt = f === null ? '–' : f.toFixed(0);
    if (el.textContent !== txt && f !== null && lastFit !== null) { el.classList.add('bump'); setTimeout(() => el.classList.remove('bump'), 200); }
    el.textContent = txt; lastFit = f;
    const c = $('count'); c.textContent = `${n} of 5`; c.classList.toggle('full', n === 5);
  }

  function blankHero(msg, sub) {
    $('fitV').className = 'fit-v num none'; $('fitV').textContent = '–';
    $('verdict').textContent = msg; $('rankline').innerHTML = sub || '&nbsp;';
    $('netV').textContent = '–'; $('netV').className = 'v num'; $('offdefV').textContent = '–';
    $('hist').innerHTML = '&nbsp;'; $('matchNote').hidden = true; $('oppResult').innerHTML = opp ? 'Pick five players to see the matchup.' : '';
  }

  function update() {
    const ids = selIds(), n = ids.length;
    renderFive(ids);
    $('hero').classList.remove('skel-on');
    $('hero').classList.toggle('partial', n !== 5);
    $('oppVal').textContent = opp ? 'vs ' + (D.nick[opp] || opp) : 'None';
    if (n !== 5) {
      const need = 5 - n;
      const msg = n === 0 ? 'Pick five players to score a lineup' : `Pick ${plural(need, 'more player')}`;
      blankHero(msg, n === 0 ? 'Or start from the best five below' : '');
      setLive(n, null, n === 0 ? 'Pick five players' : `${need} to go`);
      $('whyH').textContent = 'Why this works';
      $('why').innerHTML = '<div class="empty-row">Pick five players to see what drives the score.</div>'; $('whyFoot').textContent = '';
      $('swaps').innerHTML = '<div class="empty-row">Swaps appear once you have five players.</div>'; $('swapFoot').textContent = '';
      return;
    }
    const r = lookup.get(selMask());
    if (!r) {
      if (loadState === 'loading') {
        blankHero(`Loading every ${D.nick[team] || team} lineup`, '&nbsp;'); $('hero').classList.add('skel-on');
        $('why').innerHTML = skRows(3); $('swaps').innerHTML = skRows(4);
      } else {
        blankHero('Not enough data for this lineup', loadState === 'error' ? 'Only the top lineups loaded. <button type="button" class="linkbtn" data-retry>Try again</button>' : '');
        $('why').innerHTML = ''; $('swaps').innerHTML = '';
      }
      setLive(5, null, 'Loading'); $('whyFoot').textContent = ''; $('swapFoot').textContent = '';
      return;
    }
    const [mask, net, off, feas, rank, cs] = r;
    const f = fit(net);
    $('fitV').className = 'fit-v num'; $('fitV').innerHTML = `${f.toFixed(0)}<small>/100</small>`;
    $('verdict').textContent = verdict(rank, T.total);
    $('rankline').textContent = `Ranked ${ord(rank)} of ${T.total.toLocaleString()}` + (feas ? '' : '. Needs a true big and only active players to count as a real option');
    $('netV').innerHTML = `<span class="tri" aria-hidden="true">${net >= 0 ? '▲' : '▼'}</span>${sgn(net, 1)}`; $('netV').className = 'v num ' + (net >= 0 ? 'pos' : 'neg');
    $('offdefV').innerHTML = `${Math.round(off)}<span class="faint">–</span>${Math.round(off - net)}`;
    const h = T.hist[String(mask)];
    if (h) {
      const g = h[2], what = h[3] === 'starting' ? `Started ${plural(g, 'game')} together` : h[3] === 'closing' ? `Closed ${plural(g, 'game')} together` : 'Played together';
      $('hist').textContent = `${what}: ${sgn(h[1], 1)} net rating in ${h[0]} minutes.`;
    } else $('hist').textContent = HIST_NONE + '.';
    setLive(5, f, verdict(rank, T.total).replace(' of possible lineups for the ' + (D.nick[team] || team), '').replace('Best possible lineup for the ' + (D.nick[team] || team), 'Best possible five'));
    const adj = adjustment(ids);
    if (adj === null) { $('matchNote').hidden = true; $('oppResult').innerHTML = ''; }
    else {
      const txt = `Against the ${esc(D.nick[opp] || opp)}: <b>${sgn(adj, 1)}</b> net rating, fit about <b>${fit(net + adj).toFixed(0)}</b>`;
      $('matchNote').hidden = false; $('matchNote').innerHTML = `<span class="badge warn">Estimate</span><span>${txt}</span>`;
      $('oppResult').innerHTML = txt + '.';
    }
    renderWhy(ids, cs, f);
    renderSwaps(ids, f);
  }

  function renderWhy(ids, cs, f) {
    const rows = [];
    for (let k = 0; k < cs.length; k += 2) rows.push([D.labels[D.features[cs[k]]], cs[k + 1]]);
    rows.sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
    const scale = Math.max(2.5, ...rows.map(r => Math.abs(r[1])));
    $('whyH').textContent = f >= 50 ? 'Why this works' : 'Why it struggles';
    $('why').innerHTML = rows.map(([lbl, v]) => `<div class="row why-row"><div class="grow"><span class="title">${esc(lbl)}</span>
      <span class="cbar" aria-hidden="true"><i class="${v > 0 ? 'p' : 'n'}" style="width:${(Math.abs(v) / scale * 50).toFixed(1)}%"></i></span></div>
      <span class="delta ${v > 0 ? 'pos' : 'neg'}">${sgn(v, 1)}</span></div>`).join('')
      || '<div class="empty-row">No single trait moves this five much either way.</div>';
    const [s, hd, b] = shb(ids);
    $('whyFoot').textContent = `${plural(s, 'shooter')}, ${plural(hd, 'ball handler')} and ${plural(b, 'big')}. Points per 100 possessions each trait adds or costs.`;
  }

  function renderSwaps(ids, baseFit) {
    if (!T.full && loadState === 'loading') { $('swaps').innerHTML = skRows(4); $('swapFoot').textContent = ''; return; }
    const out = [], m0 = selMask();
    for (const po of ids) for (let pi = 0; pi < T.n; pi++) {
      if (sel.has(pi)) continue;
      const r = lookup.get((m0 & ~(1 << po)) | (1 << pi)); if (!r) continue;
      out.push({ po, pi, d: fit(r[1]) - baseFit, f: fit(r[1]) });
    }
    out.sort((a, b) => b.d - a.d);
    const top = out.slice(0, 5);
    $('swaps').innerHTML = top.map(s => `<button type="button" class="row swap" data-po="${s.po}" data-pi="${s.pi}">
      <span class="duo" aria-hidden="true"><span class="av">${initials(T.players[s.pi].name)}</span></span>
      <span class="grow"><span class="title">Swap ${esc(last(T.players[s.po].name))} for <b>${esc(last(T.players[s.pi].name))}</b></span><span class="sub">Fit becomes ${s.f.toFixed(0)}</span></span>
      <span class="delta ${s.d >= 0 ? 'pos' : 'neg'}">${sgn(s.d, 1)}</span></button>`).join('') || '<div class="empty-row">No swaps available.</div>';
    const b = top[0];
    $('swapFoot').textContent = !b ? '' : b.d > 0.5 ? (T.full ? '' : 'Based on the top lineups only.') : 'No single swap improves this five by more than half a point.';
  }
  $('swaps').addEventListener('click', e => {
    const b = e.target.closest('.swap'); if (!b) return;
    const pi = +b.dataset.pi; sel.delete(+b.dataset.po); sel.add(pi); syncRoster(); update();
    toast(`Swapped in ${last(T.players[pi].name)}`); haptic();
  });

  // ---- pair synergy: mean fit of fives containing both players minus the team mean
  function synergy() {
    const n = T.n, sum = Array.from({ length: n }, () => new Float64Array(n)), cnt = Array.from({ length: n }, () => new Float64Array(n));
    let tot = 0;
    for (const r of T.combos) { const f = fit(r[1]); tot += f; const ids = []; for (let i = 0; i < n; i++) if (r[0] & (1 << i)) ids.push(i);
      for (let a = 0; a < 5; a++) for (let b = a + 1; b < 5; b++) { sum[ids[a]][ids[b]] += f; cnt[ids[a]][ids[b]]++; } }
    const mean = tot / T.combos.length, pairs = [];
    const S = Array.from({ length: n }, () => new Float64Array(n));
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) { const v = cnt[i][j] ? sum[i][j] / cnt[i][j] - mean : 0; S[i][j] = S[j][i] = v; pairs.push({ i, j, v }); }
    return { S, N: cnt, pairs, mean };
  }
  const pairName = (i, j) => `${esc(last(T.players[i].name))} + ${esc(last(T.players[j].name))}`;
  function pairRow(p) {
    return `<button type="button" class="row pair" data-i="${p.i}" data-j="${p.j}">
      <span class="grow title">${pairName(p.i, p.j)}</span><span class="delta ${p.v >= 0 ? 'pos' : 'neg'}">${sgn(p.v, 1)}</span></button>`;
  }
  function renderPairs() {
    if (!T.full) {
      if (loadState === 'error') {
        $('pairsBest').innerHTML = `<div class="err"><p>Couldn't load every ${esc(D.nick[team] || team)} lineup. Check your connection and try again.</p><button type="button" class="btn sm" data-retry>Try again</button></div>`;
        $('pairsWorst').innerHTML = '<div class="empty-row">Needs the full lineup table.</div>';
        $('heatmap').innerHTML = '<p class="muted small">Needs the full lineup table.</p>';
      } else {
        $('pairsBest').innerHTML = skRows(3); $('pairsWorst').innerHTML = skRows(3);
        $('heatmap').innerHTML = '<div class="skel" style="height:240px"></div>';
      }
      $('heatVal').innerHTML = '';
      return;
    }
    const { S, N, pairs, mean } = synergy();
    const sorted = [...pairs].sort((a, b) => b.v - a.v);
    $('pairsBest').innerHTML = sorted.slice(0, 3).map(pairRow).join('');
    $('pairsWorst').innerHTML = sorted.slice(-3).reverse().map(pairRow).join('');
    // full heatmap (sheet)
    const n = T.n, cell = 34, padL = 92, padT = 78, W = padL + n * cell + 8, H = padT + n * cell + 4;
    const vmax = Math.max(0.1, ...pairs.map(p => Math.abs(p.v)));
    let s = `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Pair scores for every two players">`;
    T.players.forEach((p, i) => {
      s += `<text x="${padL - 8}" y="${padT + i * cell + cell / 2 + 4}" text-anchor="end">${esc(last(p.name))}</text>`;
      s += `<text transform="translate(${padL + i * cell + cell / 2 + 4},${padT - 8}) rotate(-55)">${esc(last(p.name))}</text>`;
    });
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      const x = padL + j * cell, y = padT + i * cell;
      if (i === j) { s += `<rect x="${x + 1}" y="${y + 1}" width="${cell - 2}" height="${cell - 2}" rx="4" style="fill:var(--fill)"/>`; continue; }
      const t = Math.max(-1, Math.min(1, S[i][j] / vmax)), u = Math.round(Math.abs(t) * 100);
      s += `<rect class="heat-cell" data-i="${i}" data-j="${j}" x="${x + 1}" y="${y + 1}" width="${cell - 2}" height="${cell - 2}" rx="4" style="fill:color-mix(in srgb, var(${t >= 0 ? '--pos' : '--neg'}) ${u}%, var(--surface-2))"><title>${last(T.players[i].name)} + ${last(T.players[j].name)}: ${sgn(S[i][j], 1)}</title></rect>`;
    }
    $('heatmap').innerHTML = s + '</svg>';
    $('heatH').textContent = `All ${D.nick[team] || team} pairs`;
    $('scaleLo').textContent = `${sgn(-vmax, 1)}`; $('scaleHi').textContent = `${sgn(vmax, 1)}`;
    $('heatVal').innerHTML = `<span>Team average fit is ${mean.toFixed(0)}. Tap a square.</span>`;
    $('heatmap').onclick = e => {
      const c = e.target.closest('.heat-cell'); if (!c) return;
      const i = +c.dataset.i, j = +c.dataset.j, v = S[i][j];
      $('heatmap').querySelectorAll('.heat-cell').forEach(r => r.classList.toggle('on', (+r.dataset.i === i && +r.dataset.j === j) || (+r.dataset.i === j && +r.dataset.j === i)));
      $('heatVal').innerHTML = `<span><b>${pairName(i, j)}</b>: <b class="${v >= 0 ? 'pos' : 'neg'}">${sgn(v, 1)}</b> across ${N[Math.min(i, j)][Math.max(i, j)].toLocaleString()} lineups</span><button type="button" class="btn sm primary" data-pair="${i},${j}">Try this pair</button>`;
    };
  }
  function tryPair(i, j) {
    const want = [i, j].filter(k => !sel.has(k));
    if (!want.length) { toast('Both are already in your five'); return; }
    const order = [...sel].filter(k => k !== i && k !== j);
    while (sel.size + want.length > 5) sel.delete(order.pop());
    want.forEach(k => sel.add(k));
    syncRoster(); update(); haptic();
    toast(`Added ${want.map(k => last(T.players[k].name)).join(' and ')}`);
  }
  ['pairsBest', 'pairsWorst'].forEach(id => $(id).addEventListener('click', e => { const b = e.target.closest('.pair'); if (b) tryPair(+b.dataset.i, +b.dataset.j); }));

  // ---- rotation
  function renderRotation() {
    const R = T.rotation, el = $('rotation');
    if (!R || R.error) { el.innerHTML = `<div class="list"><div class="empty-row">${esc(R ? R.error : 'No rotation for this roster.')}</div></div>`; return; }
    const pl = id => T.players[byId.get(id)];
    const unit = (title, u, mins) => `<div><h3 class="sub-h">${title}<span>Fit ${u.fit.toFixed(0)}</span></h3><div class="list">
      ${u.p.map(q => { const p = pl(q[0]); return `<div class="row"><span class="av" aria-hidden="true">${initials(p.name)}</span><span class="grow title">${esc(p.name)}</span>${mins ? `<span class="value num">${q[1].toFixed(0)} min</span>` : `<span class="value">${POSN[p.pos] || p.pos}</span>`}</div>`; }).join('')}
      <button type="button" class="row usefive" data-load="${u.p.map(q => q[0]).join(',')}" data-name="${title}">Use this five</button></div></div>`;
    el.innerHTML = unit('Starters', R.s, true) + unit('Bench', R.b, true) + unit('Closing five', R.c, false)
      + `<p class="section-foot rot-foot">Minutes follow each player's share of the season, ${R.alloc.toFixed(0)} of 240 for these ten. The closing five ignores minutes.${R.bench_up ? ' The model actually rates the bench unit higher.' : ''}</p>`;
  }
  $('rotation').addEventListener('click', e => {
    const b = e.target.closest('[data-load]'); if (!b) return;
    sel = new Set(b.dataset.load.split(',').map(Number).map(id => byId.get(id)).filter(i => i !== undefined));
    syncRoster(); update(); toTop(); toast(`${b.dataset.name} loaded`); haptic();
  });

  function toTop() { $('hero').scrollIntoView({ behavior: reduced ? 'auto' : 'smooth', block: 'start' }); }

  // ---- sheets (bottom sheet on phones, centred modal on desktop; styles in app.css)
  let openSheet = null, lastFocus = null;
  function openS(id) {
    const s = $(id); if (!s) return;
    if (openSheet) closeS(true);
    lastFocus = document.activeElement;
    s.inert = false; s.scrollTop = 0; s.classList.add('open'); $('scrim').classList.add('open');
    document.documentElement.style.overflow = 'hidden'; document.documentElement.classList.add('sheet-open'); openSheet = s;
    setTimeout(() => s.focus({ preventScroll: true }), 60);
  }
  function closeS(quiet) {
    if (!openSheet) return;
    const s = openSheet; openSheet = null;
    s.classList.remove('open'); s.style.transform = ''; s.inert = true;
    $('scrim').classList.remove('open'); document.documentElement.style.overflow = ''; document.documentElement.classList.remove('sheet-open');
    if (!quiet && lastFocus && lastFocus.focus) lastFocus.focus({ preventScroll: true });
  }
  $('scrim').addEventListener('click', () => closeS());
  document.addEventListener('keydown', e => { if (e.key === 'Escape') closeS(); });
  // drag the sheet down to dismiss (phones)
  document.querySelectorAll('.sheet').forEach(s => {
    let y0 = null, dy = 0;
    s.addEventListener('touchstart', e => { if (s.scrollTop > 0 || e.target.closest('.heatwrap') || innerWidth >= 900) { y0 = null; return; } y0 = e.touches[0].clientY; dy = 0; }, { passive: true });
    s.addEventListener('touchmove', e => { if (y0 === null) return; dy = Math.max(0, e.touches[0].clientY - y0); s.style.transition = 'none'; s.style.transform = `translateY(${dy}px)`; }, { passive: true });
    s.addEventListener('touchend', () => { if (y0 === null) return; s.style.transition = ''; if (dy > 90) closeS(); else s.style.transform = ''; y0 = null; });
  });

  // delegated actions (the top-bar info button is cloned into the shell's header after this script runs)
  document.addEventListener('click', e => {
    const o = e.target.closest('[data-sheet]'); if (o) { e.preventDefault(); openS(o.dataset.sheet); return; }
    if (e.target.closest('[data-close]')) { closeS(); return; }
    if (e.target.closest('[data-retry]')) { loadState = 'loading'; renderPairs(); update(); loadFull(team); return; }
    const pb = e.target.closest('[data-pair]'); if (pb) { const [i, j] = pb.dataset.pair.split(',').map(Number); closeS(true); tryPair(i, j); return; }
    const lr = e.target.closest('.league-row');
    if (lr) { closeS(true); if (lr.dataset.team !== team) { setTeam(lr.dataset.team); toast(`Opened the ${D.nick[lr.dataset.team] || lr.dataset.team}`); } toTop(); }
  });

  // ---- live score bar: visible once the hero has scrolled out of view
  const bar = $('livebar');
  if ('IntersectionObserver' in window) {
    new IntersectionObserver(es => es.forEach(en => {
      const show = !en.isIntersecting && en.boundingClientRect.top < 0;
      bar.classList.toggle('show', show); bar.setAttribute('aria-hidden', String(!show)); $('lbBtn').tabIndex = show ? 0 : -1;
    }), { threshold: 0, rootMargin: '-90px 0px 0px 0px' }).observe($('hero'));
  }
  $('lbBtn').addEventListener('click', toTop);

  // ---- wiring
  $('teamSel').addEventListener('change', e => setTeam(e.target.value));
  $('oppSel').addEventListener('change', e => { opp = e.target.value; update(); });
  $('btnBest').addEventListener('click', () => { maskToSel(T.best_mask); syncRoster(); update(); toast('Best five loaded'); });
  $('btnMost').addEventListener('click', () => {
    if (T.most_used === null || T.most_used === undefined) { toast('No lineup history for this team yet'); return; }
    maskToSel(T.most_used); syncRoster(); update(); toast('Most-used five loaded');
  });
  $('btnClear').addEventListener('click', () => { sel = new Set(); syncRoster(); update(); });

  window.LineupIQ = { setTeam, toggle, tryPair, get selection() { return [...sel]; }, get team() { return team; }, get full() { return !!(T && T.full); }, get state() { return loadState; } };
  const fromHash = teams.find(t => '#' + D.teams[t].abbr === location.hash);
  setTeam(fromHash || $('teamSel').value);
})();
"""


if __name__ == "__main__":
    import sys
    with open(os.path.join(ROOT, "models", "meta.json")) as fh:
        meta = json.load(fh)
    with open(os.path.join(OUT_DIR, "recommendations.json")) as fh:
        recs = json.load(fh)
    print("\n".join(build_report(meta, recs["teams"], site_dir=sys.argv[1] if len(sys.argv) > 1 else None)))
