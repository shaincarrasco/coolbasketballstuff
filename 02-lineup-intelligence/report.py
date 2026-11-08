"""
Build the Lineup Intelligence page.

Two outputs from the same builder:

* ``output/report.html`` — standalone: the shared shell (site/assets/shell.css + shell.js)
  is inlined so the file opens from anywhere, no CDN, no server.
* ``<site>/index.html`` (``cli.py demo --site <dir>``) — the GitHub-Pages version that
  references ``../assets/shell.css`` / ``../assets/shell.js`` and renders the shared top bar.

The page embeds a compact JSON payload with EVERY C(n,5) lineup of every team
(bitmask, predicted net / off, feasibility, rank, top contributions) so the builder,
what-if swaps, pairwise synergy, rotation and the opponent heuristic all run in the
browser from a ``Map<bitmask, row>`` lookup.

Page order (mobile-first): KPI tiles for the current five -> builder (team / opponent,
roster tiles, court, "Why" bars, Read line) -> what-if swaps -> rotation -> synergy
heatmap + duos -> team board -> model lab (collapsed) -> method & limits -> foot.
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

# Model-lab series colours: neutral greys only (the accent is reserved for the hero
# number, selected tiles, active controls and the five on the court).
SERIES = {"xgb": "var(--text)", "hgb": "var(--muted)", "ridge": "var(--faint)"}
SERIES_NAME = {"xgb": "XGBoost", "hgb": "HistGB", "ridge": "Ridge"}


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
        "features": feats, "labels": {f: FEATURE_LABELS.get(f, f) for f in feats},
        "teams": teams, "matchup": MATCHUP,
        "league_profile": {k: round(v, 2) for k, v in league_profile(players).items()},
        "source": {"source": source.get("source", "sample"), "label": source.get("label", "Sample league"), "updated": source.get("fetched_at") or ""},
    }
    return main, files


# ------------------------------------------------------------- static svg ---
def _fold_chart(per_fold: list[dict]) -> str:
    W, H, pad_l, pad_b, pad_t = 720, 260, 40, 58, 12
    n = len(per_fold)
    gw = (W - pad_l - 10) / n
    bw = gw / 4.2
    vals = [f[f"{k}_spearman"] for f in per_fold for k in ("ridge", "xgb", "hgb")]
    ymin = min(0.0, float(np.floor(min(vals) * 10) / 10))
    ymax = max(0.6, float(np.ceil(max(vals) * 10) / 10))

    def y(v):
        return pad_t + (H - pad_b - pad_t) * (1 - (v - ymin) / (ymax - ymin))
    out = [f'<svg class="chart" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Spearman per held-out team">']
    for v in np.arange(ymin, ymax + 1e-9, 0.1):
        out.append(f'<line class="grid" x1="{pad_l}" x2="{W-10}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>')
        out.append(f'<text x="{pad_l-6}" y="{y(v)+4:.1f}" text-anchor="end">{v:.1f}</text>')
    for i, f in enumerate(per_fold):
        x0 = pad_l + i * gw + gw * 0.15
        for k, key in enumerate(("ridge", "xgb", "hgb")):
            v = f[f"{key}_spearman"]
            top, base = (y(v), y(0.0)) if v >= 0 else (y(0.0), y(v))
            out.append(f'<rect x="{x0 + k*bw:.1f}" y="{top:.1f}" width="{bw-2:.1f}" height="{max(0.0, base-top):.1f}" fill="{SERIES[key]}" rx="1">'
                       f'<title>{html.escape(", ".join(f["held_out_teams"]))} · {SERIES_NAME[key]}: {v:.3f}</title></rect>')
        label = f.get("label") or ", ".join(t.split()[0] for t in f["held_out_teams"])
        out.append(f'<text x="{x0 + 1.5*bw:.1f}" y="{H-pad_b+14}" text-anchor="end" transform="rotate(-28 {x0 + 1.5*bw:.1f} {H-pad_b+14})">{html.escape(label)}</text>')
    out.append(f'<line class="axis" x1="{pad_l}" x2="{W-10}" y1="{y(0.0):.1f}" y2="{y(0.0):.1f}"/>')
    lx = pad_l
    for key in ("xgb", "hgb", "ridge"):
        out.append(f'<rect x="{lx}" y="{H-13}" width="10" height="10" rx="2" fill="{SERIES[key]}"/><text x="{lx+14}" y="{H-4}">{SERIES_NAME[key]}</text>')
        lx += 84
    out.append("</svg>")
    return "".join(out)


def _curve_chart(lc: dict) -> str:
    if not lc:
        return '<div class="muted">learning curve not computed</div>'
    W, H, pad_l, pad_b, pad_t = 720, 260, 40, 36, 12
    keys = ("ridge", "xgboost", "hgb")
    skey = {"ridge": "ridge", "xgboost": "xgb", "hgb": "hgb"}
    xs = lc["n_train"]
    ymin = min(min(lc[k]) - max(lc[k + "_sd"]) for k in keys)
    ymin = max(0.0, np.floor((ymin - 0.02) * 20) / 20)
    ymax = min(1.0, float(np.ceil((max(max(lc[k]) + max(lc[k + "_sd"]) for k in keys) + 0.02) * 20) / 20))

    def x(v):
        return pad_l + (W - pad_l - 70) * (v - xs[0] * 0.8) / (xs[-1] * 1.05 - xs[0] * 0.8)

    def y(v):
        return pad_t + (H - pad_b - pad_t) * (1 - (v - ymin) / (ymax - ymin))
    out = [f'<svg class="chart" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Learning curve">']
    for v in np.arange(ymin, ymax + 1e-9, 0.05):
        out.append(f'<line class="grid" x1="{pad_l}" x2="{W-70}" y1="{y(v):.1f}" y2="{y(v):.1f}"/><text x="{pad_l-6}" y="{y(v)+4:.1f}" text-anchor="end">{v:.2f}</text>')
    for n in xs:
        out.append(f'<text x="{x(n):.1f}" y="{H-pad_b+16}" text-anchor="middle">n={n}</text>')
    out.append(f'<text x="{(pad_l+W-70)/2:.0f}" y="{H-4}" text-anchor="middle">training lineups · mean of 4 team-folds · band = ±1 sd</text>')
    label_y = {k: y(lc[k][-1]) for k in keys}
    for _ in range(6):
        for a, b in itertools.combinations(sorted(keys, key=label_y.get), 2):
            if abs(label_y[a] - label_y[b]) < 13:
                mid = (label_y[a] + label_y[b]) / 2
                label_y[a], label_y[b] = mid - 6.5, mid + 6.5
    for key in keys:
        col = SERIES[skey[key]]
        ys, sd = lc[key], lc[key + "_sd"]
        band = " ".join(f"{x(a):.1f},{y(b+s):.1f}" for a, b, s in zip(xs, ys, sd)) + " " + " ".join(f"{x(a):.1f},{y(b-s):.1f}" for a, b, s in zip(xs[::-1], ys[::-1], sd[::-1]))
        out.append(f'<polygon points="{band}" fill="{col}" opacity=".08"/>')
        pts = " ".join(f"{x(a):.1f},{y(b):.1f}" for a, b in zip(xs, ys))
        dash = ' stroke-dasharray="5 4"' if key == "ridge" else ""
        out.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2"{dash}/>')
        for a, b in zip(xs, ys):
            out.append(f'<circle cx="{x(a):.1f}" cy="{y(b):.1f}" r="3.5" fill="{col}"><title>{SERIES_NAME[skey[key]]} n={a}: {b:.3f}</title></circle>')
        out.append(f'<text x="{x(xs[-1])+8:.1f}" y="{label_y[key]+4:.1f}" style="fill:{col}">{SERIES_NAME[skey[key]]} {ys[-1]:.3f}</text>')
    out.append("</svg>")
    return "".join(out)


# -------------------------------------------------------------------- html ---
def _ordinal(n: int) -> str:
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def _board(all_recs: dict, best_team: str) -> str:
    rows = []
    for t, o in sorted(all_recs.items(), key=lambda kv: -kv[1]["recommendations"][0]["fit_score"]):
        r = o["recommendations"][0]
        mu = o.get("most_used_lineup") or {}
        names = ", ".join(p["name"].split()[-1] for p in r["players"])
        rows.append(
            f'<tr data-team="{html.escape(t)}"{" class=on" if t == best_team else ""}><td>{html.escape(t)}</td>'
            f'<td class="n"><b>{r["fit_score"]:.0f}</b></td><td class="n {"pos" if r["predicted_net"] >= 0 else "neg"}">{r["predicted_net"]:+.1f}</td>'
            f'<td class="n">{r["predicted_off"]:.1f}</td><td class="n">{r["predicted_def"]:.1f}</td>'
            f'<td class="n">{r["shooters"]} / {r["handlers"]} / {r["bigs"]}</td>'
            f'<td class="n">{mu.get("fit_score", 0):.0f}<span class="rank">#{mu.get("rank_all", "–")}</span></td>'
            f'<td class="muted">{html.escape(names)}</td></tr>')
    return ("<table><thead><tr><th>Team</th><th class=n>Fit</th><th class=n>Net</th><th class=n>Off</th><th class=n>Def</th>"
            "<th class=n>S / H / B</th><th class=n>Coach's five</th><th>Model's best five</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>")


def _read_asset(name: str) -> str:
    p = os.path.join(SITE_ASSETS, name)
    if not os.path.exists(p):
        raise FileNotFoundError(f"shared shell asset missing: {p} (expected site/assets/{name} next to the project)")
    with open(p, encoding="utf-8") as fh:
        return fh.read()


def _source_bits(source: dict, n_lineups: int) -> dict:
    """Header chip, lede, lineup wording, limits and footer text per data source."""
    src = source.get("source", "sample")
    ts = source.get("fetched_at") or ""
    if src == "espn":
        seasons = " + ".join(source.get("seasons", []))
        return {
            "chip": f'<span class="chip accent">Live data · ESPN</span><span class="muted upd">updated {html.escape(ts)}</span>',
            "lede": (f"Real NBA data from ESPN's public box scores ({seasons}, {source.get('games_used', 0):,} games, latest "
                     f"{html.escape(str(source.get('latest_game_date') or ''))}). A gradient-boosted model learns which five-man combinations "
                     f"outperform the sum of their parts from {n_lineups:,} starting- and closing-five observations, then scores every possible five "
                     f"on all 30 current rosters."),
            "lineup_word": "starting / closing fives (box-score approximation)",
            "hist_none": "never a starting or closing five in the data",
            "limits": ("<b>Data are an approximation, not stint data:</b> each team-game contributes its starting five (and its top-5-minutes "
                       "closing five when different); lineup net = Σ(the five players' +/−) / 5 per 100 estimated possessions "
                       "(0.96 · (FGA + 0.44 FTA + TOV − OREB) scaled by shared minutes), minutes = mean of the five. The off / def split assumes the "
                       "lineup's deviation from the team's game net splits evenly. BPM-style columns are proxies (Game-Score halves per 36 blended with "
                       "shrunk +/− per 36) and share games with the target, so held-out metrics are optimistic. Rosters are ESPN's current rosters "
                       "capped at 17 by minutes; DNP = no minutes this season; injury chips are ESPN's status at build time."),
            "foot": f"Data: ESPN public API · updated {html.escape(ts)}",
            "foot_tags": '<span>ESPN public API</span><span>30 NBA teams</span>',
        }
    if src == "nba_api":
        return {
            "chip": f'<span class="chip accent">Live data · NBA Stats</span><span class="muted upd">updated {html.escape(ts)}</span>',
            "lede": (f"Real 5-man stint lineups from stats.nba.com ({', '.join(source.get('seasons', []))}): {n_lineups:,} lineups with ≥ 10 minutes. "
                     "A gradient-boosted model learns which fives outperform the sum of their parts, then scores every possible five on every roster."),
            "lineup_word": "5-man stint lineups (stats.nba.com)",
            "hist_none": "never played together in the data",
            "limits": "<b>Data:</b> stats.nba.com leaguedashlineups (Advanced, ≥ 10 min); BPM columns are on/off-rating proxies.",
            "foot": f"Data: stats.nba.com via nba_api · updated {html.escape(ts)}",
            "foot_tags": '<span>stats.nba.com</span>',
        }
    return {
        "chip": '<span class="chip">Sample data · synthetic</span>',
        "lede": "A gradient-boosted model learns which five-man combinations outperform the sum of their parts in a seeded, fictional league — build any lineup and every panel updates instantly.",
        "lineup_word": "synthetic lineups",
        "hist_none": "never played together in the sample",
        "limits": "<b>Limits:</b> synthetic 8-team league with an invented synergy function; fit scores are for demonstration only.",
        "foot": "Data: synthetic sample league",
        "foot_tags": '<span>Sample league</span><span>synthetic data</span>',
    }


def _page(meta: dict, all_recs: dict, tables: dict, data_json: str, head: str, tail: str, body_attrs: str,
          source: dict, default_team: str, data_url: str) -> str:
    from optimize import MATCHUP, ROTATION, FEATURE_LABELS
    ev = meta["evaluation"]
    m = ev["metrics"]
    best_team = default_team
    n_combos = sum(len(t.combos) for t in tables.values())
    n_per = int(np.median([len(t.combos) for t in tables.values()]))
    sb = _source_bits(source, ev["n_rows"])
    team_opts = "".join(f'<option value="{html.escape(t)}"{" selected" if t == best_team else ""}>{html.escape(t)}</option>' for t in all_recs)

    def mrow(name, label, win=False):
        x = m[name]
        return (f'<tr{" class=on" if win else ""}><td>{label}</td><td class="n">{x["rmse_weighted"]:.2f}</td><td class="n">{x["r2_weighted"]:.3f}</td>'
                f'<td class="n">{x["spearman_pooled"]:.3f}</td><td class="n">{x["spearman_fold_mean"]:.3f}</td><td class="n">{x.get("folds_won", "–")}</td></tr>')
    metrics_rows = (mrow("mean_baseline", "Predict the mean") + mrow("ridge", "Ridge (standardised)") +
                    mrow("xgboost", "<b>XGBoost</b> · production", True) + mrow("hgb", "HistGradientBoosting"))
    lift = m["xgboost"]["spearman_pooled"] - m["ridge"]["spearman_pooled"]
    lift_h = m["xgboost"]["spearman_pooled"] - m["hgb"]["spearman_pooled"]
    imp = meta["feature_importance"][:10]
    top = imp[0][1] if imp else 1
    inter = {"shooters_x_bigs", "spacing_x_rim", "shooters_x_ast_max", "handlers_x_usage_max", "stl_x_blk", "pair_both_nonshooters", "pair_both_handlers"}
    inter_share = sum(v for k, v in meta["feature_importance"] if k in inter)
    if imp and imp[0][0] in ("bpm_sum", "obpm_sum", "dbpm_sum"):
        imp_read = (f"Additive talent dominates on real box-score data: {FEATURE_LABELS.get(imp[0][0], imp[0][0])} carries {100*imp[0][1]:.0f}% of the gain and "
                    f"interaction features only {100*inter_share:.0f}% combined — fit effects exist but are second-order next to who is on the floor.")
    else:
        imp_read = "Interaction features dominate — spacing only pays with a rim protector behind it, and shooters only pay when someone can find them. None of that is visible in a single player's box score."
    imp_html = "".join(
        f'<div class="imp-row"><span>{html.escape(FEATURE_LABELS.get(k, k))}</span><span class="num muted" style="text-align:right">{100*v:.1f}%</span>'
        f'<div class="bar"><i style="width:{100*v/top:.0f}%"></i></div></div>' for k, v in imp)
    mu_ranks = [o["most_used_lineup"]["rank_all"] for o in all_recs.values() if o.get("most_used_lineup") and o["most_used_lineup"].get("rank_all")]
    med_rank = int(np.median(mu_ranks)) if mu_ranks else None
    best_fit = all_recs[best_team]["recommendations"][0]["fit_score"]
    board_read = (f"{best_team} owns the league's best five at fit {best_fit:.0f}. "
                  + (f"Coaches' most-used fives sit at a median rank of {med_rank} of ~{n_per:,} — the lineups actually played are rarely the model's first choice." if med_rank else ""))
    if lift > 0.01:
        lab_read = (f"XGBoost beats the Ridge baseline by {lift:+.3f} Spearman on held-out teams — a real but modest edge with "
                    f"{ev['n_rows']} noisy lineups; HistGB lands at {lift_h:+.3f} relative to XGBoost.")
    else:
        lab_read = (f"XGBoost does not beat the Ridge baseline on held-out teams ({lift:+.3f} pooled Spearman, Ridge wins "
                    f"{m['ridge']['folds_won']} of {ev['folds']} folds): with {ev['n_rows']} noisy box-score lineups the signal is mostly additive talent, "
                    f"and a well-featured linear model is the honest benchmark. XGBoost stays the production model for its exact per-feature attributions.")
    lc = ev.get("learning_curve") or {}
    src_name = source.get("label", "Sample league")

    return f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="theme-color" content="#0b0d12">
<meta name="description" content="Lineup Intelligence — every five-man lineup of all 30 NBA rosters scored by a gradient-boosted model: builder, swaps, synergy, rotation.">
<title>Lineup Intelligence · NBA Projects</title>
{head}
<style>{_CSS}</style>
</head>
<body data-app="lineup" data-title="Lineup Intelligence" data-mark="LI" data-chip="{html.escape(source.get('chip') or 'Sample data')}"{body_attrs}>
<main class="wrap" data-url="{html.escape(data_url)}">
  <div class="eyebrow-row"><div class="eyebrow">NBA Projects · lineup optimisation</div>{sb['chip']}</div>
  <h1>Lineup Intelligence</h1>
  <p class="lede">{sb['lede']} All {n_combos:,} possible fives across {len(tables)} rosters are scored; each team's full table loads when you select it.</p>

  <section class="sec" id="current" aria-label="Current five">
    <div class="sec-h"><h2>Current five</h2><span class="read" id="roTitle">–</span></div>
    <div class="grid g4">
      <div class="card kpi"><div class="label">Fit / 100</div><b class="hero" id="roFit">–</b><div class="sub" id="roRank">select five players</div></div>
      <div class="card kpi"><div class="label">Pred net</div><b id="roNet">–</b><div class="sub" id="roHist">per 100 possessions</div></div>
      <div class="card kpi"><div class="label">Pred off / def</div><b class="sm"><span id="roOff">–</span><span class="muted"> / </span><span id="roDef">–</span></b><div class="sub">offensive / defensive rating</div></div>
      <div class="card kpi"><div class="label">Shooters / handlers / bigs</div><b id="roSHB">–</b><div class="sub" id="roFeas">3P% &gt; 36 · AST% &amp; USG &gt; 20 · ≥ 81 in or C</div></div>
    </div>
    <div id="roAdj"></div>
  </section>

  <section class="sec" id="builder" aria-label="Lineup builder">
    <div class="sec-h"><h2>Builder</h2><span class="read">tap five tiles · bitmask lookup in the browser</span></div>
    <div class="controls">
      <label class="field"><span>Team</span><select id="teamSel" aria-label="Team">{team_opts}</select></label>
      <label class="field"><span>Opponent <em>heuristic</em></span><select id="oppSel" aria-label="Opponent"><option value="">— none —</option></select></label>
    </div>
    <div class="controls btns">
      <button class="btn" id="btnBest">Model's best five</button>
      <button class="btn" id="btnMost">Coach's most-used</button>
      <button class="btn" id="btnClear">Clear</button>
    </div>
    <div class="status" id="status" aria-live="polite"></div>
    <div class="builder">
      <div><div class="tiles" id="tiles" role="group" aria-label="Roster"></div><div class="hint">height · 3P% · <span class="chip xs">S</span> shooter <span class="chip xs">H</span> handler <span class="chip xs">B</span> big · <span class="chip xs">DNP</span> no minutes this season · <span class="chip xs warn">OUT</span> injury status</div></div>
      <div id="court"></div>
    </div>
    <div class="card why">
      <div class="sec-h"><h2 class="h3">Why</h2><span class="read">SHAP-style contributions to predicted net</span></div>
      <div id="roContrib"></div>
    </div>
    <div class="read-line" id="readLine"></div>
  </section>

  <section class="sec" id="swapsSec" aria-label="What-if swaps">
    <div class="sec-h"><h2>What-if swaps</h2><span class="read">best single substitutions · tap to apply</span></div>
    <div class="swaps" id="swaps"></div>
    <div class="read-line" id="swapsRead"></div>
  </section>

  <section class="sec" id="rotationSec" aria-label="Rotation">
    <div class="sec-h"><h2>Rotation</h2><span class="read">starters · bench · closing five under a minutes budget</span></div>
    <div class="grid rotgrid" id="rotation"></div>
    <div class="read-line" id="rotRead"></div>
  </section>

  <section class="sec" id="synergy" aria-label="Pairwise synergy">
    <div class="sec-h"><h2>Pairwise synergy</h2><span class="read" id="heatTitle">mean fit with both players − team mean</span></div>
    <div class="grid heatgrid">
      <div class="card">
        <div class="heatwrap" id="heatmap"></div>
        <div class="legend"><span id="scaleLo">−</span><i></i><span id="scaleHi">+</span><span class="muted">· tap a cell</span></div>
        <div class="heatval" id="heatVal"></div>
      </div>
      <div class="card duocard">
        <div class="grid g2 duos2">
          <div><div class="label">Best duos</div><ul class="duos" id="duosBest"></ul></div>
          <div><div class="label">Worst duos</div><ul class="duos" id="duosWorst"></ul></div>
        </div>
      </div>
    </div>
    <div class="read-line" id="synRead"></div>
  </section>

  <section class="sec" id="board" aria-label="Team board">
    <div class="sec-h"><h2>Team board</h2><span class="read">best constraint-feasible five per roster · tap a row to load</span></div>
    <div class="tbl board">{_board(all_recs, best_team)}</div>
    <div class="read-line"><b>Read:</b> {html.escape(board_read)}</div>
  </section>

  <details class="more sec" id="lab">
    <summary>Model lab · GroupKFold by team ({ev['folds']} folds) · {ev['n_rows']} {sb['lineup_word']} · {ev['n_features']} features</summary>
    <div class="body">
      <div class="tbl"><table><thead><tr><th>Model</th><th class=n>RMSE (w)</th><th class=n>R² (w)</th><th class=n>Spearman</th><th class=n>Spearman / fold</th><th class=n>Folds won</th></tr></thead><tbody>{metrics_rows}</tbody></table></div>
      <div class="grid g3 lab-k">
        <div class="card kpi"><div class="label">XGB lift vs Ridge</div><b class="{'pos' if lift > 0 else 'neg'}">{lift:+.3f}</b><div class="sub">pooled Spearman</div></div>
        <div class="card kpi"><div class="label">XGB lift vs HistGB</div><b class="{'pos' if lift_h > 0 else 'neg'}">{lift_h:+.3f}</b><div class="sub">pooled Spearman</div></div>
        <div class="card kpi"><div class="label">Fit transform</div><b class="sm">σ = {meta['fit_scale']:.1f}</b><div class="sub">100 · sigmoid((net − {meta['fit_center']:.2f}) / σ)</div></div>
      </div>
      <div class="read-line"><b>Read:</b> {html.escape(lab_read)} Rank correlation is the metric that matters — the optimizer only has to <i>order</i> lineups.</div>
      <div class="grid g2" style="margin-top:14px">
        <div class="card"><div class="label">Spearman per held-out team</div>{_fold_chart(ev['per_fold'])}</div>
        <div class="card"><div class="label">Learning curve · held-out Spearman</div>{_curve_chart(lc)}</div>
      </div>
      <div class="card" style="margin-top:10px"><div class="label">What drives fit · XGBoost gain share, top {len(imp)} of {ev['n_features']}</div><div style="margin-top:8px">{imp_html}</div>
        <div class="read-line"><b>Read:</b> {html.escape(imp_read)}</div></div>
    </div>
  </details>

  <div class="method" id="method">
    <b>Method &amp; limits.</b> Features: {ev['n_features']} order-invariant lineup features (sums / spreads of shooting, creation, size, defence; shooter, handler and big counts; pairwise and interaction terms).
    Model: minutes-weighted XGBoost (depth-2, 1 500 rounds) predicts lineup net rating; a second booster predicts offensive rating and defence = off − net.
    Evaluated with GroupKFold by team, so every number in the lab is on rosters the model never saw. Fit = 100 · sigmoid((net − μ) / σ) with μ, σ from the minutes-weighted history (50 = average, 73 = +1σ, 88 = +2σ).
    Contributions are XGBoost <span class="num">pred_contribs</span> (exact SHAP values for trees).
    <b>Opponent mode is a heuristic, not learned:</b> adj = {MATCHUP['rim_vs_nonshooters']} · (opp rim BLK% − league) · max(0, our non-shooters − 1) {MATCHUP['hands_vs_turnovers']:+} · (opp STL% − league) · (our TOV% − league) {MATCHUP['size_edge']:+} · (our avg height − opp avg height), clipped to ±{MATCHUP['clip']:.0f}; "opp" is the opponent's best feasible five, "league" the mean of all {len(tables)} teams' best fives. Read it as a direction to think in, not a prediction.
    <b>Rotation is a heuristic too:</b> minutes budgets are season-minute share × 240 capped at {ROTATION['budget_cap']:.0f}; starters {ROTATION['starter_minutes'][0]:.0f}–{ROTATION['starter_minutes'][1]:.0f} min, bench {ROTATION['bench_minutes'][0]:.0f}–{ROTATION['bench_minutes'][1]:.0f}; leftover minutes go to the rest of the roster.
    {sb['limits']}
    <b>Limits:</b> {ev['n_rows']} lineups; no uncertainty on predictions; synergy is marginal (averaged over every five containing the pair), not causal; fit scores are calibrated point predictions, not probabilities. CLI: <span class="num">python cli.py recommend --team X --vs Y --rotation</span>.
  </div>
  <div class="foot"><span>{sb['foot']}</span>{sb['foot_tags']}<span>Python</span><span>Pandas</span><span>scikit-learn</span><span>XGBoost</span></div>
</main>
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
    css, js = _read_asset("shell.css"), _read_asset("shell.js")
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

    # standalone: shell inlined; data-depth points the top-bar links back at the site folder
    depth = os.path.relpath(os.path.dirname(SITE_ASSETS), out_dir).replace(os.sep, "/") + "/"
    page = _page(meta, all_recs, tables, data_json, head=f"<style>{css}</style>",
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
        head = ('<link rel="stylesheet" href="../assets/shell.css">'
                '<link rel="manifest" href="../manifest.webmanifest">'
                '<link rel="apple-touch-icon" href="../icons/apple-touch-icon.png">'
                '<link rel="icon" type="image/png" sizes="192x192" href="../icons/icon-192.png">'
                '<meta name="apple-mobile-web-app-capable" content="yes">'
                '<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">')
        page = _page(meta, all_recs, tables, data_json, head=head, tail='<script src="../assets/shell.js"></script>', body_attrs="",
                     source=source, default_team=default_team, data_url="data/")
        p = os.path.join(site_dir, "index.html")
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(page)
        write_team_files(site_dir)
        paths.append(p)
    return paths


# -------------------------------------------------------------------- css ---
_CSS = r"""
.h3{font-size:14px}
.eyebrow-row{display:flex;flex-wrap:wrap;align-items:center;gap:8px 12px}
.eyebrow-row .upd{font:500 11px var(--mono);letter-spacing:.04em}
.chip.xs.warn{color:var(--warn);border-color:color-mix(in srgb,var(--warn) 45%,transparent)}
.tile.out{border-style:dashed}
.sec-h{flex-wrap:wrap;row-gap:2px}
.sec-h h2{white-space:nowrap}
.hint{color:var(--muted);font-size:12px;margin-top:8px;display:flex;flex-wrap:wrap;align-items:center;gap:4px 6px}
@media(max-width:479px){.topbar .brand > span:not(.mark){display:none}}
.field{display:flex;flex-direction:column;gap:5px;font:600 11px/1 var(--mono);letter-spacing:.1em;text-transform:uppercase;color:var(--muted)}
.field em{font-style:normal;color:var(--warn)}
.field select{width:100%}
.controls.btns{margin-top:8px}
.controls.btns .btn{justify-content:center}
.status{min-height:18px;color:var(--warn);font-size:12px;margin-top:6px}
.builder{display:grid;gap:12px;margin-top:4px}
@media(min-width:900px){.builder{grid-template-columns:minmax(0,1.2fr) minmax(0,1fr);align-items:start}}
.tiles{display:grid;gap:8px;grid-template-columns:repeat(2,minmax(0,1fr))}
@media(min-width:560px){.tiles{grid-template-columns:repeat(3,minmax(0,1fr))}}
.tile{appearance:none;font:inherit;color:var(--text);text-align:left;width:100%;display:flex;flex-direction:column;gap:7px;min-height:76px;padding:10px 11px;-webkit-tap-highlight-color:transparent;transition:border-color .12s}
.tile .top{display:flex;justify-content:space-between;align-items:center;gap:6px}
.tile .nm{font-weight:600;font-size:13px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.tile .st{font:12px var(--mono);color:var(--muted);font-variant-numeric:tabular-nums;display:flex;justify-content:space-between;align-items:center;gap:6px;white-space:nowrap}
.tile .roles{display:inline-flex;gap:3px}
.tile.dnp{opacity:.55}
.tile.on .nm{color:var(--accent)}
.chip.xs{padding:3px 6px;font-size:10px;letter-spacing:.04em;gap:3px}
.court{width:100%;height:auto;display:block;border-radius:var(--r);border:1px solid var(--line);background:var(--surface)}
.court text{font-family:var(--sans)}
.kpi b.sm{font-size:18px;white-space:nowrap}
@media(min-width:700px){.kpi b.sm{font-size:22px}}
.why{margin-top:12px}
.why .sec-h{margin-bottom:4px}
.why-row{display:grid;grid-template-columns:minmax(0,1fr) 48px;gap:3px 10px;align-items:center;padding:7px 0;border-bottom:1px solid var(--line)}
.why-row:last-child{border-bottom:0;padding-bottom:0}
.why-row .lbl{font-size:13px}
.why-row .val{text-align:right;font-family:var(--mono);font-size:13px;font-variant-numeric:tabular-nums}
.why-row .bar{grid-column:1/-1;position:relative;height:6px}
.why-row .bar::after{content:"";position:absolute;left:50%;top:-2px;bottom:-2px;width:1px;background:var(--line-2)}
.why-row .bar i{position:absolute;top:0;height:100%}
.why-row .bar i.p{left:50%;background:var(--pos)}
.why-row .bar i.n{right:50%;background:var(--neg)}
.adj{margin-top:10px;display:flex;flex-wrap:wrap;align-items:center;gap:8px;font-size:13px;color:var(--muted)}
.adj b{color:var(--text)}
.swaps{display:grid;gap:8px}
@media(min-width:700px){.swaps{grid-template-columns:1fr 1fr}}
@media(min-width:1100px){.swaps{grid-template-columns:repeat(3,1fr)}}
.swap{appearance:none;width:100%;min-height:48px;display:flex;align-items:center;gap:10px;padding:8px 12px;background:var(--surface);border:1px solid var(--line);border-radius:var(--r-sm);color:var(--text);font:14px/1.3 var(--sans);text-align:left;cursor:pointer}
.swap:hover{border-color:var(--line-2)}
.swap .who{flex:1;min-width:0}
.swap .who b{font-weight:600}
.swap .who i{font-style:normal;color:var(--muted);margin:0 4px}
.swap .to{font:12px var(--mono);color:var(--muted);white-space:nowrap}
.unit .label{font:600 11px/1 var(--mono);letter-spacing:.1em;text-transform:uppercase;color:var(--muted)}
.unit .top{display:flex;align-items:baseline;gap:10px;margin-top:8px}
.unit .top b{font:700 26px/1 var(--mono);letter-spacing:-.02em;font-variant-numeric:tabular-nums}
.unit .top span{color:var(--muted);font-size:12px}
.unit ul{list-style:none;margin:8px 0 10px;padding:0}
.unit li{display:flex;justify-content:space-between;align-items:center;gap:8px;padding:6px 0;border-bottom:1px solid var(--line);font-size:13px}
.unit li:last-child{border-bottom:0}
.unit li .mins{font:12px var(--mono);color:var(--muted);font-variant-numeric:tabular-nums}
.unit .btn{width:100%;justify-content:center}
.rotgrid{grid-template-columns:minmax(0,1fr)}
@media(min-width:700px){.rotgrid{grid-template-columns:repeat(3,minmax(0,1fr))}}
.unit li .mins{white-space:nowrap}
.rot-note{grid-column:1/-1;color:var(--muted);font-size:12px}
.heatgrid{grid-template-columns:minmax(0,1fr)}
.heatgrid > .card{min-width:0}
@media(min-width:900px){.heatgrid{grid-template-columns:minmax(0,1.5fr) minmax(0,1fr)}}
.heatwrap{overflow-x:auto;-webkit-overflow-scrolling:touch;mask-image:linear-gradient(90deg,#000 calc(100% - 20px),transparent);-webkit-mask-image:linear-gradient(90deg,#000 calc(100% - 20px),transparent)}
.heatwrap svg{display:block;min-width:660px;width:100%;height:auto}
@media(min-width:900px){.heatwrap{mask-image:none;-webkit-mask-image:none}}
.heatwrap text{font:12px var(--sans);fill:var(--muted)}
.heat-cell{cursor:pointer}
.heat-cell.on{stroke:var(--text);stroke-width:2.5}
.legend{display:flex;align-items:center;gap:8px;font:11px var(--mono);color:var(--muted);margin-top:10px}
.legend i{flex:0 0 110px;height:8px;border-radius:99px;background:linear-gradient(90deg,var(--neg),var(--surface-2),var(--pos))}
.heatval{margin-top:10px;min-height:40px;display:flex;flex-wrap:wrap;align-items:center;gap:8px 12px;font-size:13px;color:var(--muted)}
.heatval b{color:var(--text)}
.duos2{gap:14px}
.duocard .label{font:600 11px/1 var(--mono);letter-spacing:.1em;text-transform:uppercase;color:var(--muted);margin-bottom:4px}
.duos{list-style:none;margin:0;padding:0}
.duos li{display:flex;justify-content:space-between;gap:8px;padding:8px 0;border-bottom:1px solid var(--line);font-size:13px}
.duos li:last-child{border-bottom:0}
.duos li b{font-family:var(--mono);font-weight:600;font-variant-numeric:tabular-nums}
.board td{padding:12px}
.board tbody tr{cursor:pointer}
.board tr.on td:first-child{color:var(--accent);font-weight:600}
table tr.on td{background:color-mix(in srgb,var(--text) 4%,transparent)}
details.more.sec summary{min-height:48px}
details.more .body > .tbl{margin-top:2px}
.lab-k{margin:12px 0}
.imp-row{display:grid;grid-template-columns:minmax(0,1fr) 52px;gap:4px 10px;align-items:center;padding:6px 0;font-size:13px}
.imp-row .bar{grid-column:1/-1}
.imp-row .bar i{background:var(--muted)}
.card .label{font:600 11px/1 var(--mono);letter-spacing:.1em;text-transform:uppercase;color:var(--muted)}
.card > svg.chart{margin-top:10px}
"""


# --------------------------------------------------------------------- js ---
_JS = r"""
(function () {
  const D = JSON.parse(document.getElementById('data').textContent);
  const $ = id => document.getElementById(id);
  const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
  const fit = net => 100 / (1 + Math.exp(-(net - D.fit.center) / D.fit.scale));
  const last = n => n.split(' ').slice(-1)[0];
  const initials = n => { const p = n.split(' '); return (p[0][0] + p[p.length - 1][0]).toUpperCase(); };
  const sgn = (v, d) => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(d);
  const ord = n => n + (n % 100 >= 11 && n % 100 <= 13 ? 'th' : ['th', 'st', 'nd', 'rd'][n % 10] || 'th');
  const ht = h => { let f = Math.floor(h / 12), i = Math.round(h - f * 12); if (i === 12) { f++; i = 0; } return f + '′' + i + '″'; };
  const pct = v => (v * 100).toFixed(1) + '%';
  const css = getComputedStyle(document.body);
  const hex = v => { v = v.trim().replace('#', ''); return [0, 2, 4].map(k => parseInt(v.slice(k, k + 2), 16)); };
  const POS = hex(css.getPropertyValue('--pos') || '#3ddc84'), NEG = hex(css.getPropertyValue('--neg') || '#ff5c7a'), MID = hex(css.getPropertyValue('--surface-2') || '#181c25');

  let team = null, T = null, lookup = null, sel = new Set(), opp = '', byId = null, heatSel = null, loadSeq = 0;
  const teams = Object.keys(D.teams);
  const DATA_URL = document.querySelector('main.wrap').dataset.url || 'data/';
  const REAL = D.source && D.source.source !== 'sample';

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

  function setTeam(name, keepSel) {
    bind(name);
    $('teamSel').value = name;
    const os = $('oppSel');
    os.innerHTML = '<option value="">— none —</option>' + teams.filter(t => t !== name).map(t => `<option value="${esc(t)}">${esc(t)}</option>`).join('');
    if (opp === name) opp = '';
    os.value = opp;
    if (!keepSel) maskToSel(T.best_mask);
    heatSel = null;
    document.querySelectorAll('.board tbody tr').forEach(tr => tr.classList.toggle('on', tr.dataset.team === name));
    renderTiles(); renderHeat(); renderRotation(); update();
    if (!T.full) loadFull(name);
  }

  // The main page embeds every team's top fives; the full C(n,5) table is fetched on demand (~100-250 KB per team).
  function loadFull(name) {
    const seq = ++loadSeq, abbr = D.teams[name].abbr;
    $('status').textContent = `loading the full ${D.teams[name].total.toLocaleString()}-lineup table for ${name}…`;
    fetch(DATA_URL + abbr + '.json', { cache: 'force-cache' }).then(r => { if (!r.ok) throw new Error(r.status); return r.json(); })
      .then(full => {
        D.teams[name] = full;
        if (seq !== loadSeq || team !== name) return;
        bind(name); $('status').textContent = '';
        renderHeat(); update();
      })
      .catch(() => {
        if (seq !== loadSeq || team !== name) return;
        $('status').textContent = `Full table for ${name} could not be loaded (open the page over http, not file://) — showing the top ${T.combos.length} fives only.`;
      });
  }
  function maskToSel(mask) { sel = new Set(); for (let i = 0; i < T.n; i++) if (mask & (1 << i)) sel.add(i); }
  const selMask = () => [...sel].reduce((m, i) => m | (1 << i), 0);

  function roles(p) { return (p.s ? '<span class="chip xs" title="shooter">S</span>' : '') + (p.h ? '<span class="chip xs" title="handler">H</span>' : '') + (p.b ? '<span class="chip xs" title="big">B</span>' : ''); }
  const injShort = s => /out/i.test(s) ? 'OUT' : /day/i.test(s) ? 'DTD' : /quest/i.test(s) ? 'GTD' : /doubt/i.test(s) ? 'DBT' : s.slice(0, 3).toUpperCase();
  function injChip(p) { return p.inj ? `<span class="chip xs warn" title="${esc(p.inj)}">${injShort(p.inj)}</span>` : ''; }
  function renderTiles() {
    $('tiles').innerHTML = T.players.map((p, i) => `
      <button type="button" class="tile${sel.has(i) ? ' on' : ''}${p.min <= 0 ? ' dnp' : ''}${/out/i.test(p.inj || '') ? ' out' : ''}" data-i="${i}" aria-pressed="${sel.has(i)}" title="${esc(p.name)} · ${p.pos} · ${ht(p.ht)} · 3P ${pct(p.fg3)} · ${p.min} min${p.inj ? ' · ' + esc(p.inj) : ''}">
        <span class="top"><span class="nm">${esc(p.name)}</span><span class="roles">${injChip(p)}<span class="chip xs">${p.pos}</span></span></span>
        <span class="st"><span>${ht(p.ht)} · ${pct(p.fg3)}</span><span class="roles">${p.min <= 0 ? '<span class="chip xs">DNP</span>' : roles(p)}</span></span>
      </button>`).join('');
    $('tiles').querySelectorAll('.tile').forEach(el => el.addEventListener('click', () => toggle(+el.dataset.i)));
  }
  function toggle(i) {
    if (sel.has(i)) sel.delete(i);
    else if (sel.size >= 5) { flash('Five already selected — remove one first.'); return; }
    else sel.add(i);
    syncTiles(); update();
  }
  function syncTiles() { $('tiles').querySelectorAll('.tile').forEach(el => { const on = sel.has(+el.dataset.i); el.classList.toggle('on', on); el.setAttribute('aria-pressed', on); }); }
  let flashT; function flash(msg) { $('status').textContent = msg; clearTimeout(flashT); flashT = setTimeout(() => $('status').textContent = '', 2200); }
  function scrollTop() { window.scrollTo({ top: $('current').getBoundingClientRect().top + window.scrollY - 64, behavior: 'smooth' }); }

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

  function update() {
    const ids = [...sel].sort((a, b) => a - b), n = ids.length;
    renderCourt(ids);
    if (n !== 5) {
      const need = 5 - n;
      $('roTitle').textContent = need === 5 ? 'no players selected' : ids.map(i => last(T.players[i].name)).join(' · ');
      $('roFit').textContent = '–'; $('roRank').textContent = `select ${need} more player${need === 1 ? '' : 's'}`;
      $('roNet').textContent = '–'; $('roOff').textContent = '–'; $('roDef').textContent = '–'; $('roNet').className = '';
      $('roSHB').textContent = n ? shb(ids).join(' / ') : '–'; $('roFeas').textContent = 'shooters · handlers · bigs';
      $('roContrib').innerHTML = '<div class="muted" style="font-size:13px">Pick five players to see what drives the prediction.</div>';
      $('roHist').textContent = 'per 100 possessions'; $('roAdj').innerHTML = ''; $('swaps').innerHTML = '';
      $('readLine').innerHTML = `<b>Read:</b> Select ${need} more player${need === 1 ? '' : 's'} to score the five.`;
      $('swapsRead').innerHTML = '';
      return;
    }
    const r = lookup.get(selMask());
    if (!r) {
      $('roTitle').textContent = ids.map(i => last(T.players[i].name)).join(' · ');
      $('roFit').textContent = '–'; $('roRank').textContent = T.full ? 'lineup not in table' : 'not in the embedded top fives — full table still loading';
      $('roNet').textContent = '–'; $('roOff').textContent = '–'; $('roDef').textContent = '–'; $('roNet').className = '';
      $('roContrib').innerHTML = ''; $('roAdj').innerHTML = ''; $('swaps').innerHTML = ''; $('swapsRead').innerHTML = '';
      $('readLine').innerHTML = `<b>Read:</b> ${T.full ? 'This five is not in the scored table.' : 'Waiting for the full table of this team to load.'}`;
      return;
    }
    const [mask, net, off, feas, rank, cs] = r;
    const f = fit(net);
    $('roTitle').textContent = ids.map(i => last(T.players[i].name)).join(' · ');
    $('roFit').textContent = f.toFixed(0);
    $('roRank').innerHTML = `rank <span class="num" style="color:var(--text)">${rank}</span> of ${T.total.toLocaleString()}`;
    $('roNet').textContent = sgn(net, 1); $('roNet').className = net >= 0 ? 'pos' : 'neg';
    $('roOff').textContent = off.toFixed(1); $('roDef').textContent = (off - net).toFixed(1);
    $('roSHB').textContent = shb(ids).join(' / ');
    $('roFeas').innerHTML = feas ? 'constraint-feasible' : '<span class="warn">violates constraints · no big or DNP player</span>';
    const h = T.hist[String(mask)];
    $('roHist').textContent = h ? `observed ${sgn(h[1], 1)} over ${h[0]} min${h[3] ? ` as a ${h[3]} five` : ' together'}${h[2] ? ` · ${h[2]} game${h[2] === 1 ? '' : 's'}` : ''}`
      : (REAL ? 'never a starting or closing five in the data' : 'never played together in the sample');
    const adj = adjustment(ids);
    $('roAdj').innerHTML = adj === null ? '' : `<div class="adj"><span class="chip">heuristic</span><span>vs <b>${esc(opp)}</b>: ${sgn(adj, 1)} net → adjusted net <b>${sgn(net + adj, 1)}</b>, adjusted fit <b>${fit(net + adj).toFixed(0)}</b></span></div>`;
    let ch = '';
    for (let k = 0; k < cs.length; k += 2) {
      const v = cs[k + 1], w = Math.min(50, Math.abs(v) / 5 * 50);
      ch += `<div class="why-row"><span class="lbl">${esc(D.labels[D.features[cs[k]]])}</span><span class="val ${v > 0 ? 'pos' : 'neg'}">${sgn(v, 1)}</span><div class="bar"><i class="${v > 0 ? 'p' : 'n'}" style="width:${w}%"></i></div></div>`;
    }
    $('roContrib').innerHTML = ch || '<div class="muted" style="font-size:13px">No single feature moves this five by more than ±0.15.</div>';
    const pos = [], neg = [];
    for (let k = 0; k < cs.length; k += 2) (cs[k + 1] > 0 ? pos : neg).push(D.labels[D.features[cs[k]]]);
    let read = `This five is the model's ${ord(rank)}-ranked lineup of ${T.total.toLocaleString()}`;
    read += pos.length ? `; ${pos[0]} is the main driver` : '';
    read += neg.length ? `${pos.length ? ', ' : '; '}${neg[0]} is the main drag` : '';
    read += '.';
    if (!feas) read += ' It breaks the ≥ 1 big / no-DNP constraint, so the optimizer would never pick it.';
    if (adj !== null) read += ` Against ${opp} the heuristic moves it ${sgn(adj, 1)} to fit ${fit(net + adj).toFixed(0)}.`;
    $('readLine').innerHTML = `<b>Read:</b> ${esc(read)}`;
    renderSwaps(ids, net, f);
  }

  function renderSwaps(ids, baseNet, baseFit) {
    const out = [];
    for (const po of ids) for (let pi = 0; pi < T.n; pi++) {
      if (sel.has(pi)) continue;
      const m = (selMask() & ~(1 << po)) | (1 << pi);
      const r = lookup.get(m); if (!r) continue;
      out.push({ po, pi, d: fit(r[1]) - baseFit, f: fit(r[1]) });
    }
    out.sort((a, b) => b.d - a.d);
    const top = out.slice(0, 6);
    $('swaps').innerHTML = top.map(s => `<button type="button" class="swap" data-po="${s.po}" data-pi="${s.pi}"><span class="who"><b>${esc(last(T.players[s.po].name))}</b><i>→</i><b>${esc(last(T.players[s.pi].name))}</b></span><span class="chip ${s.d >= 0 ? 'pos' : 'neg'}">${sgn(s.d, 1)} fit</span><span class="to">→ ${s.f.toFixed(0)}</span></button>`).join('') || '<div class="muted">no swaps</div>';
    $('swaps').querySelectorAll('.swap').forEach(b => b.addEventListener('click', () => { sel.delete(+b.dataset.po); sel.add(+b.dataset.pi); syncTiles(); update(); scrollTop(); }));
    const b = top[0];
    $('swapsRead').innerHTML = !b ? '' : b.d > 0.5
      ? `<b>Read:</b> Best single move: ${esc(last(T.players[b.po].name))} out, ${esc(last(T.players[b.pi].name))} in (${sgn(b.d, 1)} fit → ${b.f.toFixed(0)}).`
      : `<b>Read:</b> No single substitution improves this five by more than half a point — it is a local optimum.`;
  }

  function renderCourt(ids) {
    const W = 600, H = 400;
    const spots = [[300, 72], [112, 158], [488, 158], [182, 282], [418, 282]];
    const ps = ids.map(i => T.players[i]).sort((a, b) => a.ht - b.ht);
    let s = `<svg class="court" viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Half court with the selected five placed by size">`;
    s += `<g fill="none" stroke="var(--line-2)" stroke-width="2" stroke-linecap="round">
      <line x1="20" y1="372" x2="580" y2="372"/><path d="M50 372V262A260 260 0 0 1 550 262V372"/>
      <rect x="225" y="182" width="150" height="190"/><circle cx="300" cy="182" r="60"/><path d="M240 20A60 60 0 0 0 360 20"/><line x1="20" y1="20" x2="580" y2="20"/>
      <line x1="270" y1="344" x2="330" y2="344" stroke="var(--muted)"/><circle cx="300" cy="333" r="9" stroke="var(--muted)"/></g>`;
    ps.forEach((p, k) => {
      const [x, y] = spots[k];
      s += `<circle cx="${x}" cy="${y}" r="23" fill="var(--accent)"/>`;
      s += `<text x="${x}" y="${y + 5}" text-anchor="middle" font-size="13" font-weight="700" fill="var(--accent-ink)" style="font-family:var(--mono)">${initials(p.name)}</text>`;
      s += `<text x="${x}" y="${y + 42}" text-anchor="middle" font-size="12" font-weight="600" fill="var(--text)" style="paint-order:stroke;stroke:var(--surface);stroke-width:4px">${esc(p.name)}</text>`;
      s += `<text x="${x}" y="${y + 57}" text-anchor="middle" font-size="11" fill="var(--muted)" style="paint-order:stroke;stroke:var(--surface);stroke-width:3px;font-family:var(--mono)">${p.pos} · ${ht(p.ht)} · ${pct(p.fg3)}</text>`;
    });
    for (let k = ps.length; k < 5; k++) { const [x, y] = spots[k]; s += `<circle cx="${x}" cy="${y}" r="23" fill="none" stroke="var(--faint)" stroke-dasharray="4 4"/><text x="${x}" y="${y + 4}" text-anchor="middle" font-size="12" fill="var(--faint)">?</text>`; }
    $('court').innerHTML = s + '</svg>';
  }

  // ---- synergy heatmap
  function synergy() {
    const n = T.n, sum = Array.from({ length: n }, () => new Float64Array(n)), cnt = Array.from({ length: n }, () => new Float64Array(n));
    let tot = 0;
    for (const r of T.combos) { const f = fit(r[1]); tot += f; const ids = []; for (let i = 0; i < n; i++) if (r[0] & (1 << i)) ids.push(i);
      for (let a = 0; a < 5; a++) for (let b = a + 1; b < 5; b++) { sum[ids[a]][ids[b]] += f; cnt[ids[a]][ids[b]]++; } }
    const mean = tot / T.combos.length, pairs = [];
    const S = Array.from({ length: n }, () => new Float64Array(n)), N = cnt;
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) { const v = cnt[i][j] ? sum[i][j] / cnt[i][j] - mean : 0; S[i][j] = S[j][i] = v; pairs.push({ i, j, v }); }
    return { S, N, pairs, mean };
  }
  function col(v, vmax) {
    const t = Math.max(-1, Math.min(1, v / vmax)), c = t >= 0 ? POS : NEG, u = Math.abs(t);
    return `rgb(${MID.map((m, k) => Math.round(m + (c[k] - m) * u)).join(',')})`;
  }
  function renderHeat() {
    if (!T.full) {
      $('heatmap').innerHTML = `<div class="muted" style="padding:24px 0;font-size:13px">Pairwise synergy needs every five of the roster — loading the full table for ${esc(team)}…</div>`;
      $('heatTitle').textContent = `${team}`; $('heatVal').innerHTML = ''; $('duosBest').innerHTML = ''; $('duosWorst').innerHTML = ''; $('synRead').innerHTML = '';
      return;
    }
    const { S, N, pairs, mean } = synergy();
    const n = T.n, cell = 40, padL = 96, padT = 84, W = padL + n * cell + 56, H = padT + n * cell + 6;
    const vmax = Math.max(0.1, ...pairs.map(p => Math.abs(p.v)));
    let s = `<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Pairwise synergy heatmap">`;
    T.players.forEach((p, i) => {
      s += `<text x="${padL - 8}" y="${padT + i * cell + cell / 2 + 4}" text-anchor="end">${esc(last(p.name))}</text>`;
      s += `<text transform="translate(${padL + i * cell + cell / 2 + 4},${padT - 8}) rotate(-50)">${esc(last(p.name))}</text>`;
    });
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
      const x = padL + j * cell, y = padT + i * cell;
      if (i === j) { s += `<rect x="${x + 1}" y="${y + 1}" width="${cell - 2}" height="${cell - 2}" rx="3" fill="var(--bg)"/>`; continue; }
      s += `<rect class="heat-cell" data-i="${i}" data-j="${j}" x="${x + 1}" y="${y + 1}" width="${cell - 2}" height="${cell - 2}" rx="3" fill="${col(S[i][j], vmax)}"><title>${esc(last(T.players[i].name))} + ${esc(last(T.players[j].name))}: ${sgn(S[i][j], 1)} fit vs team mean</title></rect>`;
    }
    $('heatmap').innerHTML = s + '</svg>';
    $('heatTitle').textContent = `${team} · team mean fit ${mean.toFixed(1)}`;
    $('scaleLo').textContent = `−${vmax.toFixed(1)}`; $('scaleHi').textContent = `+${vmax.toFixed(1)}`;
    const showCell = (i, j) => {
      heatSel = [i, j];
      $('heatmap').querySelectorAll('.heat-cell').forEach(r => r.classList.toggle('on', (+r.dataset.i === i && +r.dataset.j === j) || (+r.dataset.i === j && +r.dataset.j === i)));
      $('heatVal').innerHTML = `<span><b>${esc(last(T.players[i].name))} + ${esc(last(T.players[j].name))}</b> · <b class="${S[i][j] >= 0 ? 'pos' : 'neg'}">${sgn(S[i][j], 1)}</b> fit vs team mean over ${N[Math.min(i, j)][Math.max(i, j)]} fives</span><button type="button" class="btn" id="loadDuo">Load duo in builder</button>`;
      $('loadDuo').addEventListener('click', () => { if (!sel.has(i) && sel.size >= 5) sel.delete([...sel][0]); sel.add(i); if (!sel.has(j) && sel.size >= 5) sel.delete([...sel].find(x => x !== i)); sel.add(j); syncTiles(); update(); scrollTop(); });
    };
    $('heatmap').querySelectorAll('.heat-cell').forEach(r => r.addEventListener('click', () => showCell(+r.dataset.i, +r.dataset.j)));
    $('heatVal').innerHTML = '<span class="muted">Tap a cell for the pair’s value.</span>';
    const sorted = [...pairs].sort((a, b) => b.v - a.v);
    const li = p => `<li><span>${esc(last(T.players[p.i].name))} + ${esc(last(T.players[p.j].name))}</span><b class="${p.v >= 0 ? 'pos' : 'neg'}">${sgn(p.v, 1)}</b></li>`;
    $('duosBest').innerHTML = sorted.slice(0, 5).map(li).join('');
    $('duosWorst').innerHTML = sorted.slice(-5).reverse().map(li).join('');
    const b = sorted[0], w = sorted[sorted.length - 1];
    $('synRead').innerHTML = `<b>Read:</b> ${esc(last(T.players[b.i].name))} + ${esc(last(T.players[b.j].name))} lift the fives they share by ${sgn(b.v, 1)} fit; ${esc(last(T.players[w.i].name))} + ${esc(last(T.players[w.j].name))} drag theirs by ${sgn(w.v, 1)}. Averages over every five containing the pair, not a causal effect.`;
  }

  // ---- rotation
  function renderRotation() {
    const R = T.rotation, el = $('rotation');
    if (!R || R.error) { el.innerHTML = `<div class="card unit"><div class="label">Rotation</div><div class="muted" style="margin-top:8px">${esc(R ? R.error : 'not computed')}</div></div>`; $('rotRead').innerHTML = ''; return; }
    const pl = id => T.players[byId.get(id)];
    const unit = (title, u, mins) => `<div class="card unit"><div class="label">${title}</div>
      <div class="top"><b>${u.fit.toFixed(0)}</b><span>fit · net ${sgn(u.net, 1)} · rank ${u.rank}</span></div>
      <ul>${u.p.map(q => `<li><span>${esc(pl(q[0]).name)} <span class="chip xs">${pl(q[0]).pos}</span></span>${mins ? `<span class="mins">${q[1].toFixed(0)} <span class="faint">/ ${q[2].toFixed(0)}</span></span>` : ''}</li>`).join('')}</ul>
      <button type="button" class="btn" data-load="${u.p.map(q => q[0]).join(',')}">Load in builder</button></div>`;
    el.innerHTML = unit(`Starters · ${R.s.min.toFixed(0)} min`, R.s, true) + unit(`Bench · ${R.b.min.toFixed(0)} min${R.overlap ? ` · overlap ${R.overlap}` : ''}`, R.b, true) + unit('Closing five · minutes ignored', R.c, false)
      + `<div class="rot-note">Minutes shown as assigned / budget. Rotation net ${sgn(R.net, 1)} (minutes-weighted, fit ${R.fit}) · ${R.alloc.toFixed(0)} of 240 min allocated, ${R.left.toFixed(0)} left for the rest of the roster.</div>`;
    el.querySelectorAll('button[data-load]').forEach(b => b.addEventListener('click', () => { const ids = b.dataset.load.split(',').map(Number); sel = new Set(ids.map(id => byId.get(id)).filter(i => i !== undefined)); syncTiles(); update(); scrollTop(); }));
    const same = R.s.p.map(q => q[0]).sort().join() === R.c.p.map(q => q[0]).sort().join();
    $('rotRead').innerHTML = `<b>Read:</b> Starters carry ${R.s.min.toFixed(0)} min at fit ${R.s.fit.toFixed(0)}, the bench ${R.b.min.toFixed(0)} min at fit ${R.b.fit.toFixed(0)}; ${same ? 'the closing five is the starting five' : 'the closing five differs from the starters once minutes are ignored'}.${R.bench_up ? ' The model rates the bench unit higher — the starters are forced by the minutes budget.' : ''}`;
  }

  // ---- wiring
  // the shared shell hard-codes a "Sample data" chip in the top bar; relabel it with this page's real source
  const topChip = document.querySelector('.topbar .chip');
  if (topChip && REAL) { topChip.textContent = D.source.label === 'ESPN' ? 'Live · ESPN' : 'Live · ' + D.source.label; topChip.classList.add('accent'); topChip.title = 'updated ' + D.source.updated; }
  $('teamSel').addEventListener('change', e => setTeam(e.target.value));
  $('oppSel').addEventListener('change', e => { opp = e.target.value; update(); });
  $('btnBest').addEventListener('click', () => { maskToSel(T.best_mask); syncTiles(); update(); });
  $('btnMost').addEventListener('click', () => { if (T.most_used === null || T.most_used === undefined) { flash('No history for this roster.'); return; } maskToSel(T.most_used); syncTiles(); update(); });
  $('btnClear').addEventListener('click', () => { sel = new Set(); syncTiles(); update(); });
  document.querySelectorAll('.board tbody tr[data-team]').forEach(tr => tr.addEventListener('click', () => { setTeam(tr.dataset.team); scrollTop(); }));
  if (window.matchMedia && window.matchMedia('(min-width:900px)').matches) $('lab').open = true;
  window.LineupIQ = { setTeam, toggle, get selection() { return [...sel]; }, get team() { return team; }, get full() { return !!(T && T.full); } };
  setTeam($('teamSel').value);
})();
"""


if __name__ == "__main__":
    import sys
    with open(os.path.join(ROOT, "models", "meta.json")) as fh:
        meta = json.load(fh)
    with open(os.path.join(OUT_DIR, "recommendations.json")) as fh:
        recs = json.load(fh)
    print("\n".join(build_report(meta, recs["teams"], site_dir=sys.argv[1] if len(sys.argv) > 1 else None)))
