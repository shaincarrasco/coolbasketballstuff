"""
Lineup optimizer: score every five on a roster, then pick, swap, pair and rotate.

Core object: `TeamTable` = every C(n, 5) combination of a roster scored by the
trained XGBoost model (predicted net / off / def, 0-100 fit score, SHAP-style
`pred_contribs` attributions) plus the constraint-feasibility mask. Everything
else in this module is a view over that table:

* recommend()        top-N feasible fives with explanations (+ optional trade /
                     opponent-aware re-ranking)
* best_swaps()       the best single substitutions for a given five
* pair_synergy()     pairwise synergy: mean fit of lineups containing both players
                     minus the team mean, for all 78 pairs on a 13-man roster
* build_rotation()   starting five + bench five (minimal overlap) under a minutes
                     budget, plus the closing five that ignores minutes
* matchup_adjust()   documented HEURISTIC adjustment vs an opponent's best lineup
                     (not learned -- clearly labelled as such everywhere)
* export_team()      output/all_lineups_<team>.csv and output/synergy_<team>.csv

Rosters above MAX_EXHAUSTIVE players fall back to a beam search for the top
lineups only (the pairwise / builder features need the full table and are
skipped with a note).
"""
from __future__ import annotations

import itertools
import json
import os
import re
from dataclasses import dataclass, asdict, field

import numpy as np
import pandas as pd
import xgboost as xgb

from features import features_for_combos, feature_names, lineup_features, _flags
from model import fit_score, load, load_off, MODEL_DIR, ROOT

MAX_EXHAUSTIVE = 17
BEAM = 400
OUT_DIR = os.path.join(ROOT, "output")

# Human-readable labels for explanations
FEATURE_LABELS = {
    "shooters": "shooters on floor", "non_shooters": "non-shooters on floor",
    "spacing_x_rim": "spacing x rim protection", "shooters_x_ast_max": "shooters x best passer",
    "shooters_x_bigs": "shooters x bigs", "pair_both_nonshooters": "non-shooter pairs",
    "pair_both_handlers": "handler pairs", "blk_pct_max": "best rim protector BLK%",
    "height_max": "tallest player", "height_mean": "average height", "height_std": "size variety",
    "height_min": "shortest player", "bigs": "bigs on floor", "handlers": "ball handlers",
    "handlers_x_usage_max": "handlers x top usage", "usage_sum": "total usage", "usage_max": "top usage",
    "usage_std": "usage spread", "ast_pct_sum": "total AST%", "ast_pct_max": "best passer AST%",
    "tov_pct_sum": "total TOV%", "tov_pct_max": "worst TOV%", "stl_pct_sum": "total STL%",
    "stl_x_blk": "steals x rim protection", "reb_pct_sum": "total REB%", "dbpm_sum": "defensive BPM sum",
    "obpm_sum": "offensive BPM sum", "obpm_min": "weakest offensive player", "bpm_sum": "total BPM",
    "fg3_pct_mean": "avg 3P%", "fg3_pct_std": "3P% spread", "fg3_pct_min": "worst 3P%",
    "fg3a_rate_sum": "3PA rate sum", "def_rating_proxy_mean": "avg def rating",
    "pair_usage_absdiff_mean": "usage complementarity", "pair_height_absdiff_mean": "height complementarity",
    "minutes_mean_log": "experience (minutes)", "zero_minute_players": "zero-minute players",
    "blk_pct_sum": "total BLK%",
}

# --- Opponent-aware heuristic (NOT learned) -----------------------------------
# Adjustment to a lineup's predicted net rating when facing the opponent's best five.
# Each term is a plain product of "our" lineup trait and "their" trait relative to
# the league's best-lineup average; coefficients were chosen by hand to keep the
# total within a few points. See README "Opponent-aware mode".
MATCHUP = {
    "rim_vs_nonshooters": -0.6,     # their best rim protector (BLK% over league) x our non-shooters beyond the first
    "hands_vs_turnovers": -0.08,    # their total STL% over league x our total TOV% over league
    "size_edge": 0.5,               # our avg height - their avg height (inches)
    "clip": 4.0,                    # |adjustment| cap
}
ROTATION = {
    "team_minutes": 240.0, "budget_cap": 36.0, "starter_min_budget": 18.0, "bench_min_budget": 8.0,
    "starter_minutes": (30.0, 34.0), "bench_minutes": (14.0, 18.0), "starter_candidates": 60,
}


@dataclass
class Constraints:
    min_bigs: int = 1
    max_zero_minute: int = 0
    must_include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()


@dataclass
class LineupRecommendation:
    rank: int
    players: list[dict]
    predicted_net: float
    predicted_off: float
    predicted_def: float
    fit_score: float
    rank_all: int                # rank among ALL C(n,5) fives of the roster (1 = best)
    shooters: int
    handlers: int
    bigs: int
    height_mean: float
    minutes_together: float      # observed minutes in historical data (0 if never played)
    observed_net: float | None   # observed net rating if lineup exists in history
    reasons_for: list[dict]
    reasons_against: list[dict]
    summary: str
    matchup_adjustment: float | None = None   # only with --vs
    adjusted_net: float | None = None
    adjusted_fit: float | None = None


@dataclass
class TeamTable:
    team: str
    roster: pd.DataFrame                 # flagged, positional index 0..n-1
    combos: np.ndarray                   # (n_combos, 5) positional indices, each row sorted
    X: pd.DataFrame                      # features per combo
    pred_net: np.ndarray
    pred_off: np.ndarray
    pred_def: np.ndarray
    contribs: np.ndarray | None          # (n_combos, n_features) pred_contribs (bias dropped)
    feasible: np.ndarray                 # bool mask under Constraints
    rank_all: np.ndarray                 # 1-based rank by pred_net among all combos
    hist: dict = field(default_factory=dict)  # sorted player_id tuple -> (minutes, observed net)
    search: str = "exhaustive"
    added_players: list[str] = field(default_factory=list)

    @property
    def fit(self) -> np.ndarray:
        return fit_score(self.pred_net)

    def index_of(self, combo) -> int | None:
        key = tuple(sorted(int(i) for i in combo))
        return self._lookup.get(key)

    def __post_init__(self):
        self._lookup = {tuple(int(v) for v in row): i for i, row in enumerate(self.combos)}

    def player_dict(self, i: int) -> dict:
        p = self.roster.loc[i]
        return {
            "player_id": int(p.player_id), "name": p.player_name, "position": p.position,
            "height_in": float(p.height_in), "minutes": float(p.minutes), "fg3_pct": float(p.fg3_pct),
            "usage_pct": float(p.usage_pct), "ast_pct": float(p.ast_pct), "blk_pct": float(p.blk_pct),
            "stl_pct": float(p.stl_pct), "tov_pct": float(p.tov_pct), "reb_pct": float(p.reb_pct),
            "obpm": float(p.obpm), "dbpm": float(p.dbpm),
            "is_shooter": int(p.is_shooter), "is_handler": int(p.is_handler), "is_big": int(p.is_big),
        }


# ------------------------------------------------------------------ helpers ---
def slug(team: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", team.lower()).strip("_")


def resolve_team(team: str, players: pd.DataFrame) -> str:
    teams = players.team.unique().tolist()
    match = [t for t in teams if team.lower() in t.lower()]
    if len(match) != 1:
        exact = [t for t in teams if t.lower() == team.lower()]
        if len(exact) == 1:
            return exact[0]
        raise ValueError(f"Team '{team}' not found or ambiguous. Options: {', '.join(teams)}")
    return match[0]


def find_player(players: pd.DataFrame, pattern: str, exclude_team: str | None = None) -> pd.Series:
    pool = players if exclude_team is None else players[players.team != exclude_team]
    hit = pool[pool.player_name.str.contains(pattern, case=False, regex=False)]
    if len(hit) != 1:
        opts = ", ".join(f"{r.player_name} ({r.team})" for _, r in hit.head(8).iterrows()) or "none"
        raise ValueError(f"Player '{pattern}' not found or ambiguous ({len(hit)} matches: {opts}).")
    return hit.iloc[0]


def _match_names(roster: pd.DataFrame, patterns: tuple[str, ...]) -> set[int]:
    out: set[int] = set()
    for pat in patterns:
        hit = roster[roster.player_name.str.contains(pat, case=False, regex=False)]
        if hit.empty:
            raise ValueError(f"No player on roster matches '{pat}'. Roster: {', '.join(roster.player_name)}")
        out.update(hit.index.tolist())
    return out


def _feasible_mask(roster: pd.DataFrame, combos: np.ndarray, c: Constraints, must: set[int], excl: set[int]) -> np.ndarray:
    big = roster.is_big.to_numpy()[combos].sum(1)
    zero = (roster.minutes.to_numpy() <= 0)[combos].sum(1)
    ok = (big >= c.min_bigs) & (zero <= c.max_zero_minute)
    if excl:
        ok &= ~np.isin(combos, list(excl)).any(1)
    for m in must:
        ok &= (combos == m).any(1)
    return ok


def _beam_combos(roster: pd.DataFrame, booster, c: Constraints, must: set[int], excl: set[int]) -> np.ndarray:
    """Beam search for large rosters: grow partial lineups, score partials by random completions."""
    n = len(roster)
    pool = [i for i in range(n) if i not in excl]
    beam: list[tuple[int, ...]] = [tuple(sorted(must))] if must else [()]
    rng = np.random.default_rng(0)
    while len(beam[0]) < 5:
        expanded = {tuple(sorted(part + (i,))) for part in beam for i in pool if i not in part}
        exp = list(expanded)
        if len(exp[0]) == 5:
            arr = np.array(exp)
            return arr[_feasible_mask(roster, arr, c, must, excl)]
        scores = []
        for part in exp:
            rest = [i for i in pool if i not in part]
            fills = [tuple(sorted(part + tuple(rng.choice(rest, 5 - len(part), replace=False)))) for _ in range(6)]
            scores.append(float(booster.predict(features_for_combos(roster, fills)).mean()))
        order = np.argsort(scores)[::-1][:BEAM]
        beam = [exp[i] for i in order]
    return np.array(beam)


# --------------------------------------------------------------- scoring ---
def score_team(team: str, players: pd.DataFrame, lineups: pd.DataFrame | None = None,
               constraints: Constraints | None = None, add_players: tuple[str, ...] = (),
               with_contribs: bool = True, model_dir: str = MODEL_DIR) -> TeamTable:
    """Score every five on a roster (optionally after adding players from other teams)."""
    c = constraints or Constraints()
    booster, _ridge, _meta = load(model_dir)
    off_model = load_off(model_dir)
    team_name = resolve_team(team, players)
    roster = players[players.team == team_name].copy()
    added = []
    for pat in add_players:
        row = find_player(players, pat, exclude_team=team_name).copy()
        row["team"] = team_name
        added.append(f"{row.player_name} (from {find_player(players, pat, exclude_team=team_name).team})")
        roster = pd.concat([roster, row.to_frame().T], ignore_index=True)
    roster = _flags(roster.reset_index(drop=True))
    for col in ("height_in", "minutes", "usage_pct", "fg3_pct", "fg3a_rate", "ast_pct", "reb_pct",
                "stl_pct", "blk_pct", "tov_pct", "obpm", "dbpm", "def_rating_proxy"):
        roster[col] = roster[col].astype(float)
    roster["player_id"] = roster["player_id"].astype(int)
    must = _match_names(roster, c.must_include)
    excl = _match_names(roster, c.exclude)

    n = len(roster)
    if n <= MAX_EXHAUSTIVE:
        combos = np.array(list(itertools.combinations(range(n), 5)), dtype=int)
        search = "exhaustive"
    else:
        combos = _beam_combos(roster, booster, c, must, excl)
        search = f"beam({BEAM})"
    X = features_for_combos(roster, combos)
    pred_net = booster.predict(X).astype(float)
    if off_model is not None:
        pred_off = off_model.predict(X).astype(float)
    else:
        pred_off = np.full_like(pred_net, np.nan)
    pred_def = pred_off - pred_net
    contribs = None
    if with_contribs:
        dm = xgb.DMatrix(X, feature_names=list(X.columns))
        contribs = booster.get_booster().predict(dm, pred_contribs=True)[:, :-1]
    feasible = _feasible_mask(roster, combos, c, must, excl)
    order = np.argsort(-pred_net, kind="stable")
    rank_all = np.empty(len(pred_net), dtype=int)
    rank_all[order] = np.arange(1, len(pred_net) + 1)

    hist = {}
    if lineups is not None:
        for _, r in lineups[lineups.team == team_name].iterrows():
            hist[tuple(sorted((int(r.p1), int(r.p2), int(r.p3), int(r.p4), int(r.p5))))] = (float(r.minutes), float(r.net_rating))

    return TeamTable(team=team_name, roster=roster, combos=combos, X=X, pred_net=pred_net, pred_off=pred_off,
                     pred_def=pred_def, contribs=contribs, feasible=feasible, rank_all=rank_all, hist=hist,
                     search=search, added_players=added)


def _explain_row(tt: TeamTable, i: int, k: int = 4) -> tuple[list[dict], list[dict]]:
    if tt.contribs is None:
        return [], []
    row = tt.contribs[i]
    names = list(tt.X.columns)
    order = np.argsort(row)
    f = tt.X.iloc[i]

    def entry(j):
        return {"feature": names[j], "label": FEATURE_LABELS.get(names[j], names[j]),
                "value": round(float(f.iloc[j]), 2), "contribution": round(float(row[j]), 2)}
    pros = [entry(j) for j in order[::-1] if row[j] > 0.15][:k]
    cons = [entry(j) for j in order if row[j] < -0.15][:k]
    return pros, cons


def _summary(f: pd.Series, pros: list[dict], cons: list[dict]) -> str:
    bits = [f"{int(f['shooters'])} shooters, {int(f['handlers'])} handlers, {int(f['bigs'])} bigs, avg {f['height_mean']:.1f}in"]
    if pros:
        bits.append("helped by " + ", ".join(p["label"] for p in pros[:2]))
    if cons:
        bits.append("hurt by " + ", ".join(c["label"] for c in cons[:2]))
    return "; ".join(bits)


def _recommendation(tt: TeamTable, i: int, rank: int) -> LineupRecommendation:
    combo = tt.combos[i]
    f = tt.X.iloc[i]
    key = tuple(sorted(int(tt.roster.loc[j, "player_id"]) for j in combo))
    mins, onet = tt.hist.get(key, (0.0, None))
    pros, cons = _explain_row(tt, i)
    players = sorted((tt.player_dict(int(j)) for j in combo), key=lambda p: p["height_in"])
    return LineupRecommendation(
        rank=rank, players=players,
        predicted_net=round(float(tt.pred_net[i]), 2),
        predicted_off=round(float(tt.pred_off[i]), 2) if np.isfinite(tt.pred_off[i]) else None,
        predicted_def=round(float(tt.pred_def[i]), 2) if np.isfinite(tt.pred_def[i]) else None,
        fit_score=round(float(fit_score(tt.pred_net[i])), 1),
        rank_all=int(tt.rank_all[i]),
        shooters=int(f["shooters"]), handlers=int(f["handlers"]), bigs=int(f["bigs"]),
        height_mean=round(float(f["height_mean"]), 1),
        minutes_together=mins, observed_net=onet,
        reasons_for=pros, reasons_against=cons, summary=_summary(f, pros, cons),
    )


# ------------------------------------------------------------- matchups ---
def lineup_profile(tt: TeamTable, i: int) -> dict:
    f = tt.X.iloc[i]
    return {"rim": float(f["blk_pct_max"]), "hands": float(f["stl_pct_sum"]), "height": float(f["height_mean"]),
            "shooters": int(f["shooters"]), "tov": float(f["tov_pct_sum"]), "def_proxy": float(f["def_rating_proxy_mean"])}


def best_feasible_index(tt: TeamTable) -> int:
    idx = np.where(tt.feasible)[0]
    if len(idx) == 0:
        raise ValueError("No lineup satisfies the constraints.")
    return int(idx[np.argmax(tt.pred_net[idx])])


def league_profile(players: pd.DataFrame, constraints: Constraints | None = None, model_dir: str = MODEL_DIR) -> dict:
    """Average matchup profile of every team's best feasible five (the heuristic's baseline)."""
    profs = []
    for t in players.team.unique():
        tt = score_team(t, players, None, constraints, with_contribs=False, model_dir=model_dir)
        profs.append(lineup_profile(tt, best_feasible_index(tt)))
    return {k: float(np.mean([p[k] for p in profs])) for k in profs[0]}


def matchup_adjust(X: pd.DataFrame, opp: dict, league: dict) -> np.ndarray:
    """Heuristic net-rating adjustment of OUR lineups (rows of X) vs the opponent's best five."""
    non_shooters = X["non_shooters"].to_numpy()
    adj = (MATCHUP["rim_vs_nonshooters"] * (opp["rim"] - league["rim"]) * np.maximum(0, non_shooters - 1)
           + MATCHUP["hands_vs_turnovers"] * (opp["hands"] - league["hands"]) * (X["tov_pct_sum"].to_numpy() - league["tov"])
           + MATCHUP["size_edge"] * (X["height_mean"].to_numpy() - opp["height"]))
    return np.clip(adj, -MATCHUP["clip"], MATCHUP["clip"])


# ---------------------------------------------------------------- views ---
def best_swaps(tt: TeamTable, combo, k: int = 5, feasible_only: bool = False) -> list[dict]:
    """Best single substitutions (one out, one in) for a given five, by delta fit."""
    base_i = tt.index_of(combo)
    if base_i is None:
        return []
    base_fit = float(fit_score(tt.pred_net[base_i]))
    five = set(int(x) for x in combo)
    out = []
    for p_out in five:
        for p_in in range(len(tt.roster)):
            if p_in in five:
                continue
            j = tt.index_of((five - {p_out}) | {p_in})
            if j is None or (feasible_only and not tt.feasible[j]):
                continue
            out.append({
                "out": tt.roster.loc[p_out, "player_name"], "in": tt.roster.loc[p_in, "player_name"],
                "out_idx": p_out, "in_idx": p_in,
                "fit": round(float(fit_score(tt.pred_net[j])), 1),
                "delta_fit": round(float(fit_score(tt.pred_net[j])) - base_fit, 1),
                "delta_net": round(float(tt.pred_net[j] - tt.pred_net[base_i]), 2),
            })
    out.sort(key=lambda s: -s["delta_fit"])
    return out[:k]


def pair_synergy(tt: TeamTable) -> tuple[pd.DataFrame, float]:
    """Mean fit of all lineups containing both players minus the team mean over all lineups."""
    fit = tt.fit
    team_mean = float(fit.mean())
    n = len(tt.roster)
    rows = []
    for i, j in itertools.combinations(range(n), 2):
        m = (tt.combos == i).any(1) & (tt.combos == j).any(1)
        rows.append({"i": i, "j": j, "player_a": tt.roster.loc[i, "player_name"], "player_b": tt.roster.loc[j, "player_name"],
                     "n_lineups": int(m.sum()), "mean_fit": float(fit[m].mean()), "synergy": float(fit[m].mean() - team_mean)})
    return pd.DataFrame(rows), team_mean


def minutes_budget(roster: pd.DataFrame) -> np.ndarray:
    """Per-player minutes budget: share of the team's season minutes x 240, capped (heuristic)."""
    m = roster.minutes.to_numpy(dtype=float)
    share = m / m.sum() * ROTATION["team_minutes"]
    return np.where(m <= 0, 0.0, np.minimum(share, ROTATION["budget_cap"]))


def build_rotation(tt: TeamTable) -> dict:
    """Starting five + bench five with minimal overlap under a minutes budget, plus a closing five.

    * budget_i      = min(36, 240 x season-minute share)  (0 for zero-minute players)
    * starters      must have budget >= 18; each plays min(budget, 34), at least 30 if budget allows
    * bench         must have budget >= 8; each plays min(budget, 18), at least 14 if budget allows
    * objective     minutes-weighted predicted net of the two units
                    (sum_starters mins x net_S + sum_bench mins x net_B) / total
    * overlap       0 if the roster allows two disjoint eligible fives, otherwise the smallest
                    overlap that makes it possible (shared players split their minutes)
    * closing five  best feasible lineup by predicted net, budgets ignored
    """
    r = tt.roster
    budget = minutes_budget(r)
    s_lo, s_hi = ROTATION["starter_minutes"]
    b_lo, b_hi = ROTATION["bench_minutes"]
    starter_ok = budget >= ROTATION["starter_min_budget"]
    bench_ok = budget >= ROTATION["bench_min_budget"]
    feas = tt.feasible
    s_cands = np.where(feas & starter_ok[tt.combos].all(1))[0]
    b_cands = np.where(feas & bench_ok[tt.combos].all(1))[0]
    if len(s_cands) == 0 or len(b_cands) == 0:
        return {"error": "not enough eligible players for a two-unit rotation", "budget": budget.round(1).tolist()}
    s_cands = s_cands[np.argsort(-tt.pred_net[s_cands])][:ROTATION["starter_candidates"]]

    def unit_minutes(idx, lo, hi):
        return np.clip(budget[idx], lo, hi) if (budget[idx] >= lo).all() else np.minimum(budget[idx], hi)

    best = None
    for overlap in (0, 1, 2):
        for si in s_cands:
            S = tt.combos[si]
            ov = np.isin(tt.combos[b_cands], S).sum(1)
            pool = b_cands[ov == overlap]
            if len(pool) == 0:
                continue
            bi = pool[np.argmax(tt.pred_net[pool])]
            B = tt.combos[bi]
            ms = unit_minutes(S, s_lo, s_hi)
            mb = unit_minutes(B, b_lo, b_hi)
            shared = set(S) & set(B)
            for p in shared:                       # a shared player cannot exceed his budget across both units
                k_s, k_b = list(S).index(p), list(B).index(p)
                tot = ms[k_s] + mb[k_b]
                if tot > budget[p]:
                    scale = budget[p] / tot
                    ms[k_s] *= scale
                    mb[k_b] *= scale
            total = ms.sum() + mb.sum()
            obj = (ms.sum() * tt.pred_net[si] + mb.sum() * tt.pred_net[bi]) / total
            if best is None or obj > best["objective"]:
                best = {"objective": float(obj), "si": int(si), "bi": int(bi), "ms": ms, "mb": mb, "overlap": overlap}
        if best is not None:
            break
    if best is None:
        return {"error": "no bench five could be paired with any starting five", "budget": budget.round(1).tolist()}

    ci = best_feasible_index(tt)
    def unit(i, mins):
        combo = tt.combos[i]
        return {
            "players": [{**tt.player_dict(int(p)), "minutes_assigned": round(float(m), 1), "budget": round(float(budget[p]), 1)}
                        for p, m in sorted(zip(combo, mins), key=lambda pm: -pm[1])],
            "predicted_net": round(float(tt.pred_net[i]), 2), "fit_score": round(float(fit_score(tt.pred_net[i])), 1),
            "rank_all": int(tt.rank_all[i]), "minutes": round(float(np.sum(mins)), 1),
        }
    allocated = float(best["ms"].sum() + best["mb"].sum())
    return {
        "starters": unit(best["si"], best["ms"]),
        "bench": unit(best["bi"], best["mb"]),
        "closing": {**unit(ci, np.zeros(5)), "minutes": None},
        "overlap": best["overlap"],
        "bench_outranks_starters": bool(tt.pred_net[best["bi"]] > tt.pred_net[best["si"]]),
        "rotation_net": round(best["objective"], 2),
        "rotation_fit": round(float(fit_score(best["objective"])), 1),
        "minutes_allocated": round(allocated, 1),
        "minutes_unallocated": round(ROTATION["team_minutes"] - allocated, 1),
        "budget": {tt.roster.loc[i, "player_name"]: round(float(b), 1) for i, b in enumerate(budget)},
        "note": "Heuristic minutes budget (season-minute share x 240, capped at 36). Starters 30-34, bench 14-18 where budgets allow; leftover minutes go to the rest of the roster.",
    }


# ----------------------------------------------------------- recommend ---
def recommend(team: str, players: pd.DataFrame, lineups: pd.DataFrame | None = None,
              constraints: Constraints | None = None, top_n: int = 10, model_dir: str = MODEL_DIR,
              add_players: tuple[str, ...] = (), vs: str | None = None, with_rotation: bool = False,
              with_swaps: bool = True) -> dict:
    c = constraints or Constraints()
    _b, _r, meta = load(model_dir)
    tt = score_team(team, players, lineups, c, add_players=add_players, model_dir=model_dir)
    feas_idx = np.where(tt.feasible)[0]
    if len(feas_idx) == 0:
        raise ValueError("No lineup satisfies the constraints.")

    matchup = None
    score = tt.pred_net.copy()
    if vs:
        opp_tt = score_team(vs, players, None, c, with_contribs=False, model_dir=model_dir)
        if opp_tt.team == tt.team:
            raise ValueError("--vs must name a different team")
        oi = best_feasible_index(opp_tt)
        opp = lineup_profile(opp_tt, oi)
        lg = league_profile(players, c, model_dir)
        adj = matchup_adjust(tt.X, opp, lg)
        score = tt.pred_net + adj
        matchup = {
            "opponent": opp_tt.team,
            "opponent_best_five": [opp_tt.roster.loc[int(p), "player_name"] for p in opp_tt.combos[oi]],
            "opponent_profile": opp, "league_profile": lg, "coefficients": MATCHUP,
            "formula": "adj = -0.6*(opp_rim-lg_rim)*max(0,non_shooters-1) - 0.08*(opp_stl-lg_stl)*(tov_sum-lg_tov) + 0.5*(height_mean-opp_height); clipped to +-4",
            "label": "HEURISTIC adjustment (hand-set coefficients, not learned from data)",
        }

    order = feas_idx[np.argsort(-score[feas_idx], kind="stable")][:top_n]
    recs = []
    for rank, i in enumerate(order, start=1):
        rec = _recommendation(tt, int(i), rank)
        if vs:
            rec.matchup_adjustment = round(float(score[i] - tt.pred_net[i]), 2)
            rec.adjusted_net = round(float(score[i]), 2)
            rec.adjusted_fit = round(float(fit_score(score[i])), 1)
        recs.append(rec)

    # Compare with the lineup the coach actually used most (if history available)
    most_used = None
    if tt.hist:
        key, (mins, onet) = max(tt.hist.items(), key=lambda kv: kv[1][0])
        pos = {int(pid): i for i, pid in enumerate(tt.roster.player_id)}
        if all(k in pos for k in key):
            j = tt.index_of([pos[k] for k in key])
            pn = float(tt.pred_net[j]) if j is not None else float(_b.predict(pd.DataFrame([lineup_features(tt.roster.loc[[pos[k] for k in key]])])[feature_names()])[0])
            most_used = {"players": [tt.roster.loc[pos[k], "player_name"] for k in key], "minutes": mins, "observed_net": onet,
                         "predicted_net": round(pn, 2), "fit_score": round(float(fit_score(pn)), 1),
                         "rank_all": int(tt.rank_all[j]) if j is not None else None}

    out = {
        "team": tt.team,
        "roster_size": int(len(tt.roster)),
        "added_players": tt.added_players,
        "combos_total": int(len(tt.combos)),
        "combos_evaluated": int(tt.feasible.sum()),
        "search": tt.search,
        "constraints": asdict(c),
        "fit_score_transform": f"100 * sigmoid((net_pred - {meta['fit_center']}) / {meta['fit_scale']})",
        "most_used_lineup": most_used,
        "matchup": matchup,
        "recommendations": [asdict(r) for r in recs],
        "note": "Sample data. Fit scores are for demonstration only.",
    }
    if with_swaps and len(order):
        out["swaps_for_top"] = best_swaps(tt, tt.combos[int(order[0])])
    if with_rotation:
        out["rotation"] = build_rotation(tt)
    return out


def recommend_all(players: pd.DataFrame, lineups: pd.DataFrame, constraints: Constraints | None = None, top_n: int = 10) -> dict:
    return {t: recommend(t, players, lineups, constraints, top_n, with_rotation=True) for t in players.team.unique()}


# -------------------------------------------------------------- exports ---
def export_team(tt: TeamTable, out_dir: str = OUT_DIR) -> tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    names = tt.roster.player_name.to_numpy()
    fit = tt.fit
    df = pd.DataFrame({
        "rank_all": tt.rank_all, "feasible": tt.feasible.astype(int),
        **{f"p{k+1}": names[tt.combos[:, k]] for k in range(5)},
        "predicted_net": tt.pred_net.round(2), "predicted_off": tt.pred_off.round(2), "predicted_def": tt.pred_def.round(2),
        "fit_score": fit.round(1),
        "shooters": tt.X.shooters.astype(int), "handlers": tt.X.handlers.astype(int), "bigs": tt.X.bigs.astype(int),
        "height_mean": tt.X.height_mean.round(1),
    }).sort_values("rank_all")
    p1 = os.path.join(out_dir, f"all_lineups_{slug(tt.team)}.csv")
    df.to_csv(p1, index=False)
    syn, team_mean = pair_synergy(tt)
    syn = syn.drop(columns=["i", "j"]).assign(team_mean_fit=round(team_mean, 2)).sort_values("synergy", ascending=False)
    p2 = os.path.join(out_dir, f"synergy_{slug(tt.team)}.csv")
    syn.round(3).to_csv(p2, index=False)
    return p1, p2


if __name__ == "__main__":
    import sys
    players = pd.read_csv(os.path.join(ROOT, "data", "players.csv"))
    lineups = pd.read_csv(os.path.join(ROOT, "data", "lineups.csv"))
    team = sys.argv[1] if len(sys.argv) > 1 else players.team.iloc[0]
    out = recommend(team, players, lineups, with_rotation=True)
    print(json.dumps(out["recommendations"][0], indent=2))
    print(json.dumps(out["swaps_for_top"], indent=2))
    print(json.dumps({k: v for k, v in out["rotation"].items() if k != "budget"}, indent=2)[:3000])
