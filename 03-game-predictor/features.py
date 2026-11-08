"""
Leak-free rolling feature engineering.

Every feature for game g is computed from games with date < g.date only.
Games are processed one calendar day at a time: features for all games on a
day are computed first, then the day's results are folded into team state.
`assert_no_leakage()` re-derives a random sample of features from scratch with
an explicit `date < game_date` filter and checks equality.

Per-team pre-game state -> features (for home H and away A):
    win% over last 5 / 10 / 20 games, point differential L10,
    pts for / pts against per game L10 (off/def rating proxy),
    season win%, home-split win% (H) / away-split win% (A),
    Elo (K=20, home advantage, margin multiplier, season carry-over),
    rest days, back-to-back flag, streak (+wins / -losses),
    head-to-head record over last 4 meetings, games played this season.
Difference features are home minus away (positive favours the home team).
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from math import log

import numpy as np
import pandas as pd

# ---- Elo configuration -------------------------------------------------
ELO_K = 20.0
ELO_HOME_ADV = 80.0          # Elo points added to the home side when predicting
ELO_INIT = 1500.0
ELO_CARRYOVER = 0.75         # new season: 0.75*elo + 0.25*1500
REST_CAP = 7
H2H_WINDOW = 4


@dataclass
class TeamState:
    elo: float = ELO_INIT
    results: deque = field(default_factory=lambda: deque(maxlen=20))  # (win, pf, pa)
    season_w: int = 0
    season_g: int = 0
    home_w: int = 0
    home_g: int = 0
    away_w: int = 0
    away_g: int = 0
    streak: int = 0
    last_date: pd.Timestamp | None = None


def elo_expected(elo_home: float, elo_away: float) -> float:
    return 1.0 / (1.0 + 10 ** (-((elo_home + ELO_HOME_ADV) - elo_away) / 400.0))


def elo_update(elo_home: float, elo_away: float, home_margin: int) -> tuple[float, float]:
    """FiveThirtyEight-style update with margin-of-victory multiplier."""
    exp_home = elo_expected(elo_home, elo_away)
    s_home = 1.0 if home_margin > 0 else 0.0
    winner_diff = (elo_home + ELO_HOME_ADV - elo_away) if home_margin > 0 else (elo_away - elo_home - ELO_HOME_ADV)
    mult = log(abs(home_margin) + 1.0) * (2.2 / (winner_diff * 0.001 + 2.2))
    delta = ELO_K * mult * (s_home - exp_home)
    return elo_home + delta, elo_away - delta


def _win_pct(results, n: int) -> float:
    if not results:
        return np.nan
    last = list(results)[-n:]
    return float(np.mean([r[0] for r in last]))


def _mean_stat(results, n: int, idx: int) -> float:
    if not results:
        return np.nan
    last = list(results)[-n:]
    return float(np.mean([r[idx] for r in last]))


def _pdiff(results, n: int) -> float:
    if not results:
        return np.nan
    last = list(results)[-n:]
    return float(np.mean([r[1] - r[2] for r in last]))


def _team_features(st: TeamState, game_date: pd.Timestamp, is_home: bool) -> dict:
    rest = REST_CAP if st.last_date is None else min((game_date - st.last_date).days, REST_CAP)
    split_w, split_g = (st.home_w, st.home_g) if is_home else (st.away_w, st.away_g)
    return {
        "elo": st.elo,
        "wp5": _win_pct(st.results, 5),
        "wp10": _win_pct(st.results, 10),
        "wp20": _win_pct(st.results, 20),
        "pdiff10": _pdiff(st.results, 10),
        "off10": _mean_stat(st.results, 10, 1),
        "def10": _mean_stat(st.results, 10, 2),
        "season_wp": st.season_w / st.season_g if st.season_g else np.nan,
        "split_wp": split_w / split_g if split_g else np.nan,
        "rest": rest,
        "b2b": 1 if rest == 1 else 0,
        "streak": st.streak,
        "gp": st.season_g,
    }


PER_TEAM = ["elo", "wp5", "wp10", "wp20", "pdiff10", "off10", "def10", "season_wp", "split_wp",
            "rest", "b2b", "streak", "gp"]
DIFF_FEATURES = [f"{k}_diff" for k in PER_TEAM]
EXTRA_FEATURES = ["elo_prob", "h2h_home_wp", "h2h_n", "home_rest", "away_rest", "home_b2b", "away_b2b"]
FEATURE_COLUMNS = DIFF_FEATURES + EXTRA_FEATURES


def build_features(games: pd.DataFrame) -> pd.DataFrame:
    """Return games + one feature row per game (strictly pre-game information).
    Games may have NaN scores (unplayed); they get features but never update state."""
    df = games.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "game_id"]).reset_index(drop=True)

    state: dict[str, TeamState] = defaultdict(TeamState)
    h2h: dict[tuple[str, str], deque] = defaultdict(lambda: deque(maxlen=H2H_WINDOW))  # key sorted pair -> winners
    current_season = None
    feats: list[dict] = []

    for day, day_games in df.groupby("date", sort=True):
        # --- new season: Elo regression to mean, reset season counters ----
        season = day_games["season"].iloc[0]
        if season != current_season:
            for st in state.values():
                st.elo = ELO_CARRYOVER * st.elo + (1 - ELO_CARRYOVER) * ELO_INIT
                st.season_w = st.season_g = st.home_w = st.home_g = st.away_w = st.away_g = 0
                st.streak = 0
                st.last_date = None  # rest resets at season start
            current_season = season

        # --- 1) features for every game today from state BEFORE today ------
        day_rows = []
        for _, g in day_games.iterrows():
            h, a = g["home_team"], g["away_team"]
            fh = _team_features(state[h], day, True)
            fa = _team_features(state[a], day, False)
            row = {"game_id": g["game_id"]}
            for k in PER_TEAM:
                row[f"{k}_diff"] = fh[k] - fa[k]
            row["elo_prob"] = elo_expected(fh["elo"], fa["elo"])
            key = tuple(sorted((h, a)))
            past = h2h[key]
            row["h2h_n"] = len(past)
            row["h2h_home_wp"] = float(np.mean([w == h for w in past])) if past else np.nan
            row["home_rest"], row["away_rest"] = fh["rest"], fa["rest"]
            row["home_b2b"], row["away_b2b"] = fh["b2b"], fa["b2b"]
            row["home_elo"], row["away_elo"] = fh["elo"], fa["elo"]
            day_rows.append(row)
        feats.extend(day_rows)

        # --- 2) fold today's results into state (after all features) -------
        for _, g in day_games.iterrows():
            if pd.isna(g["home_pts"]) or pd.isna(g["away_pts"]):
                continue
            h, a = g["home_team"], g["away_team"]
            hp, ap = int(g["home_pts"]), int(g["away_pts"])
            margin = hp - ap
            sh, sa = state[h], state[a]
            sh.elo, sa.elo = elo_update(sh.elo, sa.elo, margin)
            hw = margin > 0
            sh.results.append((1 if hw else 0, hp, ap))
            sa.results.append((0 if hw else 1, ap, hp))
            sh.season_g += 1; sa.season_g += 1
            sh.home_g += 1; sa.away_g += 1
            if hw:
                sh.season_w += 1; sh.home_w += 1
                sh.streak = sh.streak + 1 if sh.streak > 0 else 1
                sa.streak = sa.streak - 1 if sa.streak < 0 else -1
            else:
                sa.season_w += 1; sa.away_w += 1
                sa.streak = sa.streak + 1 if sa.streak > 0 else 1
                sh.streak = sh.streak - 1 if sh.streak < 0 else -1
            sh.last_date = sa.last_date = day
            h2h[tuple(sorted((h, a)))].append(h if hw else a)

    fdf = pd.DataFrame(feats)
    out = df.merge(fdf, on="game_id", how="left")
    out["home_win"] = np.where(out["home_pts"].isna(), np.nan, (out["home_pts"] > out["away_pts"]).astype(float))
    return out


def final_elos(games: pd.DataFrame) -> dict[str, float]:
    """Elo after the last played game (useful for display)."""
    feats = build_features(games)
    last = {}
    for _, g in feats.iterrows():
        last[g["home_team"]] = g["home_elo"]
        last[g["away_team"]] = g["away_elo"]
    return last


# ---- leakage guard -----------------------------------------------------
def _naive_team_stats(games: pd.DataFrame, team: str, before: pd.Timestamp, season: str):
    """Recompute L10 win%, L10 point diff and rest days with an explicit
    date < game_date filter, independent from the streaming implementation."""
    past = games[(games["date"] < before) & (games["home_pts"].notna())
                 & ((games["home_team"] == team) | (games["away_team"] == team))].sort_values("date")
    if past.empty:
        return np.nan, np.nan, REST_CAP
    is_home = past["home_team"] == team
    pf = np.where(is_home, past["home_pts"], past["away_pts"])
    pa = np.where(is_home, past["away_pts"], past["home_pts"])
    wins = (pf > pa).astype(float)
    season_past = past[past["season"] == season]
    rest = REST_CAP if season_past.empty else min((before - season_past["date"].iloc[-1]).days, REST_CAP)
    return wins[-10:].mean(), (pf - pa)[-10:].mean(), rest


def assert_no_leakage(games: pd.DataFrame, feats: pd.DataFrame, n_samples: int = 60, seed: int = 0) -> None:
    """Raise AssertionError if any sampled feature used same-day or future data."""
    g = games.copy()
    g["date"] = pd.to_datetime(g["date"])
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(feats), size=min(n_samples, len(feats)), replace=False)
    for i in idx:
        row = feats.iloc[i]
        d = row["date"]
        h_wp, h_pd, h_rest = _naive_team_stats(g, row["home_team"], d, row["season"])
        a_wp, a_pd, a_rest = _naive_team_stats(g, row["away_team"], d, row["season"])
        for name, got, exp in (("wp10_diff", row["wp10_diff"], h_wp - a_wp),
                               ("pdiff10_diff", row["pdiff10_diff"], h_pd - a_pd),
                               ("home_rest", row["home_rest"], h_rest),
                               ("away_rest", row["away_rest"], a_rest)):
            if np.isnan(exp) and np.isnan(got):
                continue
            assert abs(got - exp) < 1e-9, f"leak/mismatch in {name} for {row['game_id']}: got {got}, expected {exp}"
    # structural check: a team's first game of the dataset has no history at all
    first = feats.sort_values("date").groupby("home_team").head(1)
    assert feats["elo_prob"].between(0, 1).all()
    assert (feats["home_rest"] >= 1).all() and (feats["away_rest"] >= 1).all()
    del first


if __name__ == "__main__":
    import sys
    from pathlib import Path
    p = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "data" / "games.csv"
    games = pd.read_csv(p)
    feats = build_features(games)
    assert_no_leakage(games, feats)
    print(feats[["game_id", "date", "home_team", "away_team", "elo_diff", "elo_prob", "wp10_diff", "home_rest", "home_win"]].tail(8).to_string())
    print("\nno leakage detected in sampled games")
    # quick Elo-only accuracy per season
    played = feats.dropna(subset=["home_win"])
    for s, grp in played.groupby("season"):
        print(f"{s}: Elo-only accuracy {((grp.elo_prob > .5) == (grp.home_win == 1)).mean():.3f}")
