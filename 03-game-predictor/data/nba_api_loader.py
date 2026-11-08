"""
Optional real-data path via `nba_api` (stats.nba.com).

Builds the exact same games.csv schema as sample_generator.py:
    game_id, date, season, home_team, away_team, home_pts, away_pts

This is NOT used by the default demo: stats.nba.com is frequently blocked from
cloud machines and rate-limits aggressively. Every network call is wrapped in
try/except with a timeout and falls back to the sample generator with a clear
message, so the pipeline always degrades gracefully.

Usage:
    python data/nba_api_loader.py --seasons 2022-23 2023-24 2024-25
    python cli.py generate --real --seasons 2022-23 2023-24 2024-25
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
TIMEOUT_SECONDS = 20
FALLBACK_MSG = "stats.nba.com unreachable — using sample data"


def _fetch_season_log(season: str, timeout: int = TIMEOUT_SECONDS) -> pd.DataFrame:
    """One row per team-game from LeagueGameLog (regular season)."""
    from nba_api.stats.endpoints import leaguegamelog  # imported lazily on purpose

    log = leaguegamelog.LeagueGameLog(
        season=season,
        season_type_all_star="Regular Season",
        player_or_team_abbreviation="T",
        timeout=timeout,
    )
    return log.get_data_frames()[0]


def _fetch_season_finder(season: str, timeout: int = TIMEOUT_SECONDS) -> pd.DataFrame:
    """Alternative endpoint: LeagueGameFinder, same team-game shape."""
    from nba_api.stats.endpoints import leaguegamefinder

    finder = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        season_type_nullable="Regular Season",
        league_id_nullable="00",
        timeout=timeout,
    )
    return finder.get_data_frames()[0]


def team_games_to_schema(team_rows: pd.DataFrame, season: str) -> pd.DataFrame:
    """Collapse the two team rows per GAME_ID into one game row.
    MATCHUP looks like 'BOS vs. NYK' (home) or 'BOS @ NYK' (away)."""
    df = team_rows.copy()
    df["is_home"] = df["MATCHUP"].str.contains(" vs. ", regex=False)
    home = df[df.is_home][["GAME_ID", "GAME_DATE", "TEAM_ABBREVIATION", "PTS"]].rename(
        columns={"TEAM_ABBREVIATION": "home_team", "PTS": "home_pts"})
    away = df[~df.is_home][["GAME_ID", "TEAM_ABBREVIATION", "PTS"]].rename(
        columns={"TEAM_ABBREVIATION": "away_team", "PTS": "away_pts"})
    games = home.merge(away, on="GAME_ID", how="inner")
    games["date"] = pd.to_datetime(games["GAME_DATE"]).dt.strftime("%Y-%m-%d")
    games["season"] = season
    games = games.rename(columns={"GAME_ID": "game_id"})
    games = games[["game_id", "date", "season", "home_team", "away_team", "home_pts", "away_pts"]]
    games = games.dropna().astype({"home_pts": int, "away_pts": int})
    return games.sort_values(["date", "game_id"]).reset_index(drop=True)


def load_real_games(seasons: list[str], out_path: Path | None = None, pause: float = 1.0) -> Path | None:
    """Try to download `seasons` from stats.nba.com. Returns the written csv
    path on success, or None (after printing FALLBACK_MSG) on any failure."""
    out_path = Path(out_path) if out_path else HERE / "games.csv"
    try:
        import nba_api  # noqa: F401
    except ImportError:
        print("nba_api is not installed (pip install nba_api). " + FALLBACK_MSG, file=sys.stderr)
        return None

    frames = []
    for season in seasons:
        rows = None
        for fetch in (_fetch_season_log, _fetch_season_finder):
            try:
                rows = fetch(season)
                if rows is not None and len(rows) > 0:
                    break
            except Exception as exc:  # network, timeout, JSON, schema ... anything
                print(f"  [{fetch.__name__}] {season}: {type(exc).__name__}: {str(exc)[:120]}", file=sys.stderr)
                rows = None
        if rows is None or len(rows) == 0:
            print(FALLBACK_MSG, file=sys.stderr)
            return None
        frames.append(team_games_to_schema(rows, season))
        print(f"  fetched {season}: {len(frames[-1])} games")
        time.sleep(pause)  # be polite to the API

    games = pd.concat(frames, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    games.to_csv(out_path, index=False)
    print(f"wrote {out_path} ({len(games)} real games)")
    return out_path


def load_or_fallback(seasons: list[str], out_path: Path | None = None, seed: int = 42) -> Path:
    """Real data if reachable, otherwise the seeded simulation."""
    p = load_real_games(seasons, out_path)
    if p is not None:
        return p
    from sample_generator import generate  # same directory
    return generate(out_path, seed=seed, seasons=len(seasons))


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    ap = argparse.ArgumentParser(description="Download real NBA games via nba_api (falls back to sample data).")
    ap.add_argument("--seasons", nargs="+", default=["2022-23", "2023-24", "2024-25"])
    ap.add_argument("--out", default=str(HERE / "games.csv"))
    args = ap.parse_args()
    load_or_fallback(args.seasons, Path(args.out))
