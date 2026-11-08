"""
Monte Carlo rest-of-season simulator (SAMPLE DATA).

From a chosen date D inside the final season:
  * standings  = results of games played before D
  * Elo        = each team's pre-game Elo as of D (from the leak-free feature stream)
  * remaining  = every game on or after D, simulated N times with Elo win
                 probabilities (home advantage included); Elo is updated inside
                 each simulated path (K=20, no margin multiplier) so hot streaks
                 compound the way they do in the real thing.

Vectorised over simulations: one numpy loop over the ~remaining games with
arrays of shape (N,) -> N=2000 takes well under a second.

Outputs: expected wins, playoff odds (top-6 per conference, random tiebreak),
top-seed odds, win-total distribution per team, plus the *actual* final record
when the season has already been played in the data (hindcast check).
Not betting advice. Nothing here is guaranteed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from features import ELO_HOME_ADV, ELO_K, build_features, load_games

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"
PLAYOFF_SPOTS = 6


def default_sim_date(feats: pd.DataFrame, season: str) -> pd.Timestamp:
    """Live data (unplayed games present): today. Complete season (hindcast): a date
    ~55% of the way through the season's game days (roughly the All-Star break)."""
    f = feats[feats["season"] == season]
    if f["home_pts"].isna().any():
        from predict import today_et
        return max(today_et(), f["date"].min())
    days = sorted(f["date"].unique())
    return pd.Timestamp(days[int(len(days) * 0.55)])


def simulate(date: str | None = None, n_sims: int = 2000, seed: int = 42, feats: pd.DataFrame | None = None,
             teams: pd.DataFrame | None = None, focus: str | None = None, games_path: Path | None = None) -> dict:
    if feats is None:
        games = load_games(Path(games_path) if games_path else ROOT / "data" / "games.csv")
        feats = build_features(games)
    if teams is None:
        from analytics import load_teams
        teams = load_teams()
    season = sorted(feats["season"].unique())[-1]
    f = feats[feats["season"] == season].sort_values(["date", "game_id"])
    d0 = pd.to_datetime(date) if date else default_sim_date(feats, season)
    if not (f["date"].min() <= d0 <= f["date"].max()):
        raise SystemExit(f"--date must be inside season {season} ({f['date'].min().date()} .. {f['date'].max().date()})")
    live = bool(f["home_pts"].isna().any())   # in-progress season: simulate the published remaining schedule
    cup = f["cup_final"].fillna(False).astype(bool) if "cup_final" in f.columns else pd.Series(False, index=f.index)

    codes = list(teams["code"])
    idx = {t: i for i, t in enumerate(codes)}
    conf = teams["conference"].to_numpy()
    T = len(codes)

    # --- standings & Elo as of d0 --------------------------------------------
    if live:
        before = f[f["home_pts"].notna()]                       # everything final so far (incl. today's early tips)
        remaining = f[f["home_pts"].isna() & (f["date"] >= d0)]  # published schedule, not yet played
    else:
        before = f[(f["date"] < d0)].dropna(subset=["home_pts"])
        remaining = f[f["date"] >= d0]
    wins = np.zeros(T); losses = np.zeros(T)
    for _, g in before[~cup.loc[before.index]].iterrows():       # NBA Cup final does not count in the standings
        hw = g["home_pts"] > g["away_pts"]
        wins[idx[g["home_team"]]] += hw; losses[idx[g["home_team"]]] += not hw
        wins[idx[g["away_team"]]] += not hw; losses[idx[g["away_team"]]] += hw
    elo0 = np.full(T, 1500.0)
    seen = set()
    for _, g in remaining.iterrows():  # pre-game Elo of each team's first remaining game == Elo as of d0
        for side, col in (("home_team", "home_elo"), ("away_team", "away_elo")):
            if g[side] not in seen:
                elo0[idx[g[side]]] = float(g[col]); seen.add(g[side])
    for t in codes:  # teams with no remaining games keep their last known Elo
        if t not in seen:
            tg = f[(f["home_team"] == t) | (f["away_team"] == t)].iloc[-1]
            elo0[idx[t]] = float(tg["home_elo"] if tg["home_team"] == t else tg["away_elo"])

    # --- vectorised Monte Carlo ----------------------------------------------
    rng = np.random.default_rng(seed)
    elo = np.tile(elo0, (n_sims, 1))                    # (N, T)
    W = np.tile(wins, (n_sims, 1))
    h_idx = remaining["home_team"].map(idx).to_numpy()
    a_idx = remaining["away_team"].map(idx).to_numpy()
    for h, a in zip(h_idx, a_idx):
        p = 1.0 / (1.0 + 10 ** (-((elo[:, h] + ELO_HOME_ADV) - elo[:, a]) / 400.0))
        hw = rng.random(n_sims) < p
        W[:, h] += hw; W[:, a] += ~hw
        delta = ELO_K * (hw - p)
        elo[:, h] += delta; elo[:, a] -= delta

    # --- seeding per conference (random tiebreak) ----------------------------
    playoff = np.zeros(T); seed1 = np.zeros(T); seed_mean = np.zeros(T)
    jitter = rng.random(W.shape) * 1e-3
    for c in np.unique(conf):
        m = np.where(conf == c)[0]
        order = np.argsort(-(W[:, m] + jitter[:, m]), axis=1)          # (N, 15) team positions by rank
        ranks = np.empty_like(order)
        np.put_along_axis(ranks, order, np.arange(len(m))[None, :].repeat(n_sims, 0), axis=1)
        playoff[m] = (ranks < PLAYOFF_SPOTS).mean(axis=0)
        seed1[m] = (ranks == 0).mean(axis=0)
        seed_mean[m] = ranks.mean(axis=0) + 1

    # --- actual outcome (hindcast) -------------------------------------------
    played_all = f.dropna(subset=["home_pts"])
    final_w = np.zeros(T); final_l = np.zeros(T)
    for _, g in played_all.iterrows():
        hw = g["home_pts"] > g["away_pts"]
        final_w[idx[g["home_team"]]] += hw; final_l[idx[g["home_team"]]] += not hw
        final_w[idx[g["away_team"]]] += not hw; final_l[idx[g["away_team"]]] += hw
    season_complete = (not live) and bool(remaining["home_pts"].notna().all()) and len(remaining) > 0
    games_per_team = pd.concat([f["home_team"], f["away_team"]]).value_counts()
    schedule_note = None
    if live:
        missing = int(82 * T // 2 - len(f[~cup]))
        schedule_note = (f"Published schedule as of {d0.date()}: {len(remaining)} unplayed games. "
                         + (f"{missing} games are not on the schedule yet (NBA Cup knockout rounds and the extra "
                            "regular-season games for non-finalists are fixed in December); expected wins are "
                            "pro-rated to 82 games below." if missing > 0 else ""))
    actual_playoff = np.zeros(T, dtype=bool)
    if season_complete:
        for c in np.unique(conf):
            m = np.where(conf == c)[0]
            top = m[np.argsort(-final_w[m], kind="stable")[:PLAYOFF_SPOTS]]
            actual_playoff[top] = True

    names = dict(teams[["code", "name"]].values)
    rows = []
    gp = wins + losses
    # live: a team whose schedule still misses games (Cup-dependent) gets its simulated total pro-rated to 82
    sched_g = np.array([int(games_per_team.get(t, 0)) for t in codes], dtype=float)
    if live:
        sched_g -= np.array([int(cup[(f["home_team"] == t) | (f["away_team"] == t)].sum()) for t in codes])
    scale = np.where(live & (sched_g > 0), 82.0 / np.maximum(sched_g, 1), 1.0)
    W_adj = W * scale[None, :]
    for i, t in enumerate(codes):
        rows.append({
            "team": t, "name": names[t], "conf": conf[i], "elo": round(float(elo0[i]), 1),
            "w": int(wins[i]), "l": int(losses[i]), "gp": int(gp[i]), "sched_g": int(sched_g[i]),
            "exp_w": round(float(W_adj[:, i].mean()), 1), "sd_w": round(float(W_adj[:, i].std()), 1),
            "w_lo": int(np.percentile(W_adj[:, i], 5)), "w_hi": int(np.percentile(W_adj[:, i], 95)),
            "playoff": round(float(playoff[i]), 4), "top_seed": round(float(seed1[i]), 4), "seed_mean": round(float(seed_mean[i]), 2),
            "final_w": int(final_w[i]) if season_complete else None, "final_l": int(final_l[i]) if season_complete else None,
            "made_playoffs": bool(actual_playoff[i]) if season_complete else None,
        })
    rows.sort(key=lambda r: (r["conf"], -r["exp_w"]))
    focus = focus or max(rows, key=lambda r: r["sd_w"] * (0.5 - abs(0.5 - r["playoff"])) + r["exp_w"] * 1e-3)["team"]
    fw = np.rint(W_adj[:, idx[focus]]).astype(int)
    lo, hi = int(fw.min()), int(fw.max())
    hist = [{"wins": w, "n": int((fw == w).sum())} for w in range(lo, hi + 1)]
    brier = (float(np.mean((playoff - actual_playoff) ** 2)) if season_complete else None)

    out = {
        "season": season, "date": str(d0.date()), "n_sims": n_sims, "seed": seed, "sample_data": "cup_final" not in f.columns,
        "live": live, "schedule_note": schedule_note,
        "games_played": int(len(before)), "games_remaining": int(len(remaining)), "playoff_spots": PLAYOFF_SPOTS,
        "method": "Elo win probabilities with in-path Elo updates (K=20, +80 home), random tiebreaks",
        "season_complete": season_complete, "playoff_brier": brier,
        "focus": focus, "focus_hist": hist, "teams": rows,
        "disclaimer": "Not betting advice. Nothing here is guaranteed.",
    }
    OUTPUT.mkdir(exist_ok=True)
    (OUTPUT / "simulation.json").write_text(json.dumps(out, indent=2))
    pd.DataFrame(rows).to_csv(OUTPUT / "simulation.csv", index=False)
    return out


def print_sim(out: dict) -> None:
    print(f"\nSEASON SIMULATOR  from {out['date']}  ({out['games_played']} played, {out['games_remaining']} remaining, "
          f"N={out['n_sims']})   [{'SAMPLE DATA' if out.get('sample_data') else 'ESPN DATA'}]")
    if out.get("schedule_note"):
        print(out["schedule_note"])
    for c in ("East", "West"):
        print(f"\n-- {c} --   top-{out['playoff_spots']} make the playoffs")
        print(f"{'team':<5}{'W-L':<8}{'elo':>6}{'expW':>7}{'5-95%':>9}{'playoff':>9}{'#1 seed':>9}" + ("   final" if out["season_complete"] else ""))
        for r in [r for r in out["teams"] if r["conf"] == c]:
            fin = ""
            if out["season_complete"]:
                fin = f"   {r['final_w']}-{r['final_l']} {'IN' if r['made_playoffs'] else 'out'}"
            print(f"{r['team']:<5}{str(r['w']) + '-' + str(r['l']):<8}{r['elo']:>6.0f}{r['exp_w']:>7.1f}"
                  f"{str(r['w_lo']) + '-' + str(r['w_hi']):>9}{r['playoff']:>9.1%}{r['top_seed']:>9.1%}{fin}")
    h = out["focus_hist"]
    peak = max(x["n"] for x in h)
    print(f"\n{out['focus']} win-total distribution:")
    for x in h:
        if x["n"]:
            print(f"{x['wins']:>4} {'#' * int(40 * x['n'] / peak):<40} {x['n'] / out['n_sims']:.1%}")
    if out["playoff_brier"] is not None:
        print(f"\nhindcast: playoff-odds Brier vs what actually happened = {out['playoff_brier']:.3f}")
    print(out["disclaimer"])


if __name__ == "__main__":
    import sys
    print_sim(simulate(sys.argv[1] if len(sys.argv) > 1 else None))
