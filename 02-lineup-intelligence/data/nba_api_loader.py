"""
Optional real-data path via `nba_api` (stats.nba.com).

NOT used by the default demo. stats.nba.com was unreachable from the build
machine, so everything in this repo was developed against data/sample_generator.py.
This module is the adapter you would switch to with network access:

    python cli.py generate --real --season 2024-25
    # or from Python:
    from data.nba_api_loader import load_real_data
    players, lineups = load_real_data("2024-25")   # returns None, None on failure

Endpoints
---------
* leaguedashlineups     (GroupQuantity=5, MeasureType=Advanced)
      -> GROUP_ID "-id1-id2-id3-id4-id5-", TEAM_ABBREVIATION, MIN, OFF_RATING,
         DEF_RATING, NET_RATING
* leaguedashplayerstats (MeasureType=Advanced)  -> USG_PCT, AST_PCT, REB_PCT,
         TS_PCT, PACE, MIN, GP ...
* leaguedashplayerstats (MeasureType=Base)      -> FG3M/FG3A -> 3P%, 3PA rate,
         STL, BLK, TOV
* commonplayerinfo / leaguedashplayerbiostats   -> PLAYER_HEIGHT_INCHES, POSITION

Caveats (read before pointing this at the live site)
----------------------------------------------------
* stats.nba.com throttles aggressively and blocks many cloud IP ranges.
  nba_api already sends browser-like headers; keep its defaults and ALWAYS
  pass a `timeout`. Add `time.sleep(0.6-1.0)` between calls. Cache responses
  to disk (we write data/players.csv + data/lineups.csv) so you only pull once.
* Lineup tables are noisy. A season of 5-man data has thousands of rows, most
  with < 20 minutes. The model weights by minutes, but you should still filter
  (e.g. MIN >= 10) and expect weaker fit than on the synthetic set.
* BPM-style columns are not provided by stats.nba.com. We proxy obpm/dbpm with
  (OFF_RATING - league OFF_RATING) and (league DEF_RATING - DEF_RATING) from the
  player on/off advanced table, scaled down. Swap in real BPM from another
  source if you have it.
* Player IDs in GROUP_ID are stable across endpoints, which is what we join on.
"""
from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np
import pandas as pd

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
TIMEOUT = 20          # seconds per request
PAUSE = 0.8           # seconds between requests (rate-limit courtesy)
UNREACHABLE_MSG = "stats.nba.com unreachable — using sample data (run `python cli.py generate`)."


def _fail(reason: str) -> tuple[None, None]:
    print(f"[nba_api_loader] {reason}")
    print(f"[nba_api_loader] {UNREACHABLE_MSG}")
    return None, None


def load_real_data(season: str = "2024-25", min_lineup_minutes: float = 10.0,
                   out_dir: str = DATA_DIR, save: bool = True) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Pull 5-man lineups and player stats. Returns (players, lineups) or (None, None)."""
    try:
        from nba_api.stats.endpoints import leaguedashlineups, leaguedashplayerstats, leaguedashplayerbiostats
    except ImportError:
        return _fail("nba_api is not installed (pip install nba_api).")

    try:
        print(f"[nba_api_loader] requesting 5-man lineups for {season} (timeout {TIMEOUT}s)...")
        lu = leaguedashlineups.LeagueDashLineups(
            season=season, group_quantity=5, measure_type_detailed_defense="Advanced",
            per_mode_detailed="Totals", timeout=TIMEOUT,
        ).get_data_frames()[0]
        time.sleep(PAUSE)

        print("[nba_api_loader] requesting player advanced stats...")
        adv = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season, measure_type_detailed_defense="Advanced", per_mode_detailed="Totals", timeout=TIMEOUT,
        ).get_data_frames()[0]
        time.sleep(PAUSE)

        print("[nba_api_loader] requesting player base stats...")
        base = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season, measure_type_detailed_defense="Base", per_mode_detailed="Totals", timeout=TIMEOUT,
        ).get_data_frames()[0]
        time.sleep(PAUSE)

        print("[nba_api_loader] requesting player bio stats (height/position)...")
        bio = leaguedashplayerbiostats.LeagueDashPlayerBioStats(season=season, timeout=TIMEOUT).get_data_frames()[0]
    except Exception as exc:  # requests.ReadTimeout, ConnectionError, JSONDecodeError (HTML block page) ...
        return _fail(f"request failed: {type(exc).__name__}: {str(exc)[:120]}")

    try:
        players = _build_players(adv, base, bio)
        lineups = _build_lineups(lu, players, min_lineup_minutes)
    except Exception as exc:
        return _fail(f"could not parse response tables: {type(exc).__name__}: {exc}")

    if save:
        os.makedirs(out_dir, exist_ok=True)
        players.to_csv(os.path.join(out_dir, "players.csv"), index=False)
        lineups.to_csv(os.path.join(out_dir, "lineups.csv"), index=False)
        print(f"[nba_api_loader] saved {len(players)} players, {len(lineups)} lineups -> {out_dir}")
    return players, lineups


def _build_players(adv: pd.DataFrame, base: pd.DataFrame, bio: pd.DataFrame) -> pd.DataFrame:
    a = adv.set_index("PLAYER_ID")
    b = base.set_index("PLAYER_ID")
    h = bio.set_index("PLAYER_ID")
    df = pd.DataFrame(index=a.index)
    df["player_id"] = a.index
    df["player_name"] = a["PLAYER_NAME"]
    df["team"] = a["TEAM_ABBREVIATION"]
    df["team_abbr"] = a["TEAM_ABBREVIATION"]
    df["position"] = h.reindex(a.index)["PLAYER_POSITION"].fillna("F") if "PLAYER_POSITION" in h else "F"
    df["height_in"] = pd.to_numeric(h.reindex(a.index)["PLAYER_HEIGHT_INCHES"], errors="coerce").fillna(78.0)
    df["minutes"] = a["MIN"].astype(float)
    df["usage_pct"] = a["USG_PCT"].astype(float) * 100
    fga = b.reindex(a.index)["FGA"].replace(0, np.nan)
    df["fg3_pct"] = (b.reindex(a.index)["FG3M"] / b.reindex(a.index)["FG3A"].replace(0, np.nan)).fillna(0.0)
    df["fg3a_rate"] = (b.reindex(a.index)["FG3A"] / fga).fillna(0.0)
    df["ast_pct"] = a["AST_PCT"].astype(float) * 100
    df["reb_pct"] = a["REB_PCT"].astype(float) * 100
    poss = (a["MIN"] * a["PACE"] / 48.0).replace(0, np.nan)
    df["stl_pct"] = (100 * b.reindex(a.index)["STL"] / poss).fillna(0.0)
    df["blk_pct"] = (100 * b.reindex(a.index)["BLK"] / poss).fillna(0.0)
    df["tov_pct"] = a["TM_TOV_PCT"].astype(float) * 100 if "TM_TOV_PCT" in a else 12.0
    # BPM proxies from on-court ratings (see module docstring caveats)
    w = a["MIN"].clip(lower=1)
    lg_off = np.average(a["OFF_RATING"], weights=w)
    lg_def = np.average(a["DEF_RATING"], weights=w)
    df["obpm"] = ((a["OFF_RATING"] - lg_off) * 0.4).round(2)
    df["dbpm"] = ((lg_def - a["DEF_RATING"]) * 0.4).round(2)
    df["def_rating_proxy"] = a["DEF_RATING"].astype(float)
    df = df[df.minutes > 0].reset_index(drop=True)
    return df


def _build_lineups(lu: pd.DataFrame, players: pd.DataFrame, min_minutes: float) -> pd.DataFrame:
    known = set(players.player_id)
    rows = []
    for i, r in lu.iterrows():
        ids = [int(x) for x in str(r["GROUP_ID"]).strip("-").split("-") if x]
        if len(ids) != 5 or not set(ids) <= known or float(r["MIN"]) < min_minutes:
            continue
        ids = sorted(ids)
        rows.append({
            "lineup_id": i + 1, "team": r["TEAM_ABBREVIATION"],
            "p1": ids[0], "p2": ids[1], "p3": ids[2], "p4": ids[3], "p5": ids[4],
            "minutes": float(r["MIN"]), "off_rating": float(r["OFF_RATING"]),
            "def_rating": float(r["DEF_RATING"]), "net_rating": float(r["NET_RATING"]),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    p, l = load_real_data()
    if p is None:
        raise SystemExit(1)
    print(p.head(), l.head(), sep="\n")
