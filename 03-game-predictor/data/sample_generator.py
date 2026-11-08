"""
Offline sample-data generator for the NBA Game Predictor.

stats.nba.com is not reachable from the build machine, so the demo runs on a
simulated league instead. The simulation is deliberately "honest": there is
hidden team strength that drifts over the season (random walk), home-court
advantage, rest / back-to-back effects, temporary injury dips, and a large
game-level noise term, so that even a perfect-knowledge predictor only gets
~68% of games right (the same regime as real NBA models at ~65-68%).

All 30 teams are fictional. No real NBA names, cities or codes are used.

Output schema (data/games.csv):
    game_id, date, season, home_team, away_team, home_pts, away_pts
"""
from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

# 30 fictional franchises: (3-letter code, name)
TEAMS = [
    ("AUR", "Aurora Comets"), ("BRK", "Brackenridge Wolves"), ("CAS", "Cascadia Rooks"),
    ("DUN", "Dunmore Foundry"), ("EMB", "Ember Bay Phoenix"), ("FAL", "Falcon Ridge Talons"),
    ("GLD", "Glasswater Herons"), ("HLW", "Hollowbrook Stags"), ("IRN", "Ironvale Hammers"),
    ("JUN", "Juniper City Jays"), ("KES", "Kestrel Point Kites"), ("LUM", "Lumen Harbor Lights"),
    ("MRW", "Marrowfield Bison"), ("NOR", "Northgate Sentinels"), ("OBS", "Obsidian Falls Ravens"),
    ("PNE", "Pinecrest Lumberjacks"), ("QRY", "Quarry Hill Miners"), ("RVN", "Riverton Otters"),
    ("SLT", "Saltmarsh Gulls"), ("TMB", "Timberline Elk"), ("UMB", "Umberfield Suns"),
    ("VLE", "Vale Harbor Mariners"), ("WYR", "Wyreford Dragons"), ("XEN", "Xenia Springs Owls"),
    ("YEL", "Yellowstone Pass Grizzlies"), ("ZEP", "Zephyr Coast Breakers"), ("COP", "Copperline Express"),
    ("HAV", "Havenport Anchors"), ("MST", "Mistwood Shadows"), ("SKY", "Skyreach Condors"),
]
CODES = [c for c, _ in TEAMS]

# ---- simulation constants (points) -------------------------------------
STRENGTH_SD_INIT = 5.0        # spread of team quality at season start (like NBA SRS)
STRENGTH_DRIFT_SD = 0.12      # daily random-walk step of hidden strength
SEASON_CARRYOVER = 0.6        # strength_new = 0.6*old + noise between seasons
HOME_ADV = 2.5
B2B_PENALTY = -1.6            # second night of a back-to-back
LONG_REST_BONUS = 0.5         # 3+ days of rest
INJURY_RATE_PER_SEASON = 3.5  # Poisson mean injury events per team per season
GAME_NOISE_SD = 12.0          # single-game margin noise (NBA ~12-13)
TOTAL_MEAN, TOTAL_SD = 226.0, 14.0

GAMES_PER_TEAM = 82
SEASON_DAYS = 168             # late October -> mid April


def _balanced_schedule(rng: np.random.Generator) -> list[tuple[str, str]]:
    """Return a list of (home, away) pairs: every pair meets home-and-home (58
    games/team) plus 24 extra games/team drawn from random perfect matchings,
    with home games balanced to 41 per team."""
    n = len(CODES)
    games: list[tuple[str, str]] = []
    home_count = {c: 0 for c in CODES}
    for i in range(n):
        for j in range(i + 1, n):
            games.append((CODES[i], CODES[j]))
            games.append((CODES[j], CODES[i]))
            home_count[CODES[i]] += 1
            home_count[CODES[j]] += 1
    extra_rounds = GAMES_PER_TEAM - 2 * (n - 1)  # 24
    for _ in range(extra_rounds):
        order = rng.permutation(n)
        for k in range(0, n, 2):
            a, b = CODES[order[k]], CODES[order[k + 1]]
            # give home court to whoever has fewer home games so far
            if home_count[a] < home_count[b] or (home_count[a] == home_count[b] and rng.random() < 0.5):
                home, away = a, b
            else:
                home, away = b, a
            games.append((home, away))
            home_count[home] += 1
    assert len(games) == n * GAMES_PER_TEAM // 2 == 1230
    return games


def _assign_dates(games, start: date, rng: np.random.Generator) -> list[tuple[date, str, str]]:
    """Greedy day-by-day scheduler. ~7.3 games/day, no team twice a day, and a
    mild preference against back-to-backs so B2Bs land near ~15-20% of games."""
    remaining = list(rng.permutation(len(games)))
    last_played: dict[str, date | None] = {c: None for c in CODES}
    scheduled = []
    day = 0
    while remaining:
        d = start + timedelta(days=day)
        days_left = max(SEASON_DAYS - day, 1)
        target = len(remaining) / days_left
        n_today = int(np.clip(rng.poisson(target * 1.05), 3, 15)) if days_left > 1 else len(remaining)
        busy: set[str] = set()
        picked = []

        def rest_of(t: str) -> int:
            lp = last_played[t]
            return 9 if lp is None else (d - lp).days

        # Prefer games whose teams are most rested; this is what pushes the
        # back-to-back rate down toward realistic (~15-20%) levels.
        order = sorted(remaining, key=lambda i: -(min(rest_of(games[i][0]), rest_of(games[i][1])) + rng.random()))
        for idx in order:
            if len(picked) >= n_today:
                break
            h, a = games[idx]
            if h in busy or a in busy:
                continue
            if min(rest_of(h), rest_of(a)) == 1 and rng.random() < 0.75:  # mostly dodge B2Bs
                continue
            picked.append(idx)
            busy.update((h, a))
        for idx in picked:
            h, a = games[idx]
            scheduled.append((d, h, a))
            last_played[h] = last_played[a] = d
            remaining.remove(idx)
        day += 1
        if day > SEASON_DAYS + 30:  # safety valve; place whatever is left
            for idx in remaining:
                h, a = games[idx]
                scheduled.append((start + timedelta(days=day), h, a))
            break
    scheduled.sort()
    return scheduled


def simulate(seasons: int = 3, seed: int = 42, first_year: int = 2023) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    strength = {c: rng.normal(0, STRENGTH_SD_INIT) for c in CODES}
    rows = []
    gid = 1
    for s in range(seasons):
        season_label = f"{first_year + s}-{str(first_year + s + 1)[-2:]}"
        start = date(first_year + s, 10, 22)
        if s > 0:  # off-season churn, regression to the mean
            for c in CODES:
                strength[c] = SEASON_CARRYOVER * strength[c] + rng.normal(0, STRENGTH_SD_INIT * 0.8)
        # injuries: list of (start_day, end_day, dip) per team
        injuries: dict[str, list[tuple[int, int, float]]] = {c: [] for c in CODES}
        for c in CODES:
            for _ in range(rng.poisson(INJURY_RATE_PER_SEASON)):
                d0 = int(rng.integers(0, SEASON_DAYS))
                injuries[c].append((d0, d0 + int(rng.integers(5, 35)), float(rng.uniform(1.5, 6.5))))

        schedule = _assign_dates(_balanced_schedule(rng), start, rng)
        last_game: dict[str, date | None] = {c: None for c in CODES}
        cur_day = None
        for d, home, away in schedule:
            # advance the hidden strength random walk once per calendar day
            if cur_day is None:
                cur_day = d
            while cur_day < d:
                for c in CODES:
                    strength[c] += rng.normal(0, STRENGTH_DRIFT_SD)
                cur_day += timedelta(days=1)
            day_idx = (d - start).days

            def eff(team: str) -> float:
                st = strength[team]
                for a, b, dip in injuries[team]:
                    if a <= day_idx <= b:
                        st -= dip
                lg = last_game[team]
                rest = (d - lg).days if lg else 5
                if rest == 1:
                    st += B2B_PENALTY
                elif rest >= 3:
                    st += LONG_REST_BONUS
                return st

            margin = eff(home) - eff(away) + HOME_ADV + rng.normal(0, GAME_NOISE_SD)
            total = rng.normal(TOTAL_MEAN, TOTAL_SD)
            home_pts = int(round((total + margin) / 2))
            away_pts = int(round((total - margin) / 2))
            if home_pts == away_pts:  # overtime: someone wins
                if rng.random() < 0.5 + HOME_ADV / 100:
                    home_pts += int(rng.integers(1, 6))
                else:
                    away_pts += int(rng.integers(1, 6))
            rows.append({
                "game_id": f"S{s + 1}G{gid:05d}",
                "date": d.isoformat(),
                "season": season_label,
                "home_team": home,
                "away_team": away,
                "home_pts": home_pts,
                "away_pts": away_pts,
            })
            gid += 1
            last_game[home] = last_game[away] = d
    df = pd.DataFrame(rows)
    df = df.sort_values(["date", "game_id"]).reset_index(drop=True)
    return df


def generate(out_path: Path | None = None, seed: int = 42, seasons: int = 3) -> Path:
    out_path = Path(out_path) if out_path else HERE / "games.csv"
    df = simulate(seasons=seasons, seed=seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    teams = pd.DataFrame(TEAMS, columns=["code", "name"])
    # two fictional conferences of 15 (first half East, second half West) for the season simulator
    teams["conference"] = ["East" if i < len(TEAMS) // 2 else "West" for i in range(len(TEAMS))]
    teams.to_csv(out_path.parent / "teams.csv", index=False)
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate the simulated league dataset (sample data).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seasons", type=int, default=3)
    ap.add_argument("--out", default=str(HERE / "games.csv"))
    args = ap.parse_args()
    p = generate(Path(args.out), seed=args.seed, seasons=args.seasons)
    df = pd.read_csv(p)
    print(f"wrote {p} ({len(df)} games, {df['season'].nunique()} seasons)")
    print(f"home win rate: {(df.home_pts > df.away_pts).mean():.3f}")
