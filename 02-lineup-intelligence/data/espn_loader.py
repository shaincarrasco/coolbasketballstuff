"""
Real NBA data from ESPN's free public API (no key, CORS *).

    python cli.py generate --source espn           # (default `auto` tries nba_api first)
    python -m data.espn_loader                     # refresh the cache + rebuild the CSVs

Endpoints
---------
* scoreboard  https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates=YYYYMM&limit=1000
              one call per month (the YYYYMMDD-YYYYMMDD range form returns 400); events carry id, date,
              season.year / season.type (1 pre, 2 regular, 3 playoffs, 5 play-in), competitors, status.
* summary     .../summary?event=<id>  -> boxscore.players[team].statistics[0] with labels
              ['MIN','PTS','FG','3PT','FT','REB','AST','TO','STL','BLK','OREB','DREB','PF','+/-'],
              athletes[] (athlete.id / displayName / position, starter, didNotPlay, stats[]),
              boxscore.teams[].statistics (team totals incl. FGA, FTA, TOV, OREB), injuries[].
* teams       .../teams?limit=50  (30 teams: id, abbreviation, displayName)
* roster      .../teams/<id>/roster -> athletes[] with height (inches), weight, age,
              position.abbreviation, injuries[0].status, status.type (active / two-way ...).

Politeness: <= 3 requests / s (token bucket), retries with exponential backoff, a UA header.

Incremental cache (committed, under data/cache/)
------------------------------------------------
* espn_games.csv           one row per final game we have parsed (game_id, date, season, type, teams, scores)
* espn_boxscores.csv.gz    one row per player per game (box-score line + starter flag)
* espn_team_totals.csv.gz  one row per team per game (FGA, FTA, TOV, OREB, PTS, REB, 3PA, MIN)
* espn_rosters.json        current rosters (refetched every run; 30 cheap calls)

A daily run only fetches game summaries whose id is not in espn_games.csv.

What becomes "players.csv" and "lineups.csv" (and how it maps to features.py)
--------------------------------------------------------------------------------
PLAYERS — one row per player who appears on a current roster OR in any cached box
score (the latter are needed to featurise historical lineups; they get active=0 and
are hidden from the builder). Season aggregates pool the current season and the
previous one (sums of box-score counts; two seasons for stability early in the year):

    feature column       ESPN derivation (per-player sums over games, team context from team totals)
    ------------------   ----------------------------------------------------------------------------
    minutes              MIN this season only (rotation budget, DNP flag; 0 = has not played this season)
    usage_pct            100 * sum[(FGA + .44 FTA + TOV) * TmMIN/5] / sum[MIN * (TmFGA + .44 TmFTA + TmTOV)]
    fg3_pct, fg3a_rate   3PM / 3PA, 3PA / FGA
    ast_pct              100 * sum AST / sum[(MIN / (TmMIN/5)) * TmFGM - FGM]
    reb_pct              100 * sum[REB * TmMIN/5] / sum[MIN * (TmREB + OppREB)]
    stl_pct              100 * sum[STL * TmMIN/5] / sum[MIN * OppPoss]        OppPoss = .96 (FGA + .44 FTA + TOV - OREB)
    blk_pct              100 * sum[BLK * TmMIN/5] / sum[MIN * (OppFGA - Opp3PA)]
    tov_pct              100 * TOV / (FGA + .44 FTA + TOV)
    obpm, dbpm           BPM is not on ESPN. Proxies: the offensive / defensive halves of Hollinger's
                         Game Score per 36 min (off: PTS + .4 FGM - .7 FGA - .4 (FTA - FTM) + .7 OREB + .7 AST - TOV;
                         def: .3 DREB + STL + .7 BLK - .4 PF), centred on the minutes-weighted league mean and
                         scaled to sd 2.5, averaged 50/50 with the player's shrunk +/- per 36
                         (pm36 * MIN / (MIN + 300)) / 2.  They are talent proxies, not BPM.
    def_rating_proxy     minutes-weighted mean of the team's game defensive rating (OppPTS / OppPoss * 100)
                         over the games the player appeared in (team context, not on/off)
    height_in, position  roster endpoint (fallback: box-score position, median height by position)
    injury_status        roster endpoint injuries[0].status ('Out', 'Day-To-Day', ...) else ''
    active               1 = on the current ESPN roster (capped at 17 per team by minutes so the
                         optimizer's exhaustive C(n,5) table fits the page), else 0

LINEUPS — NOT stint data. Two documented approximations per team-game:
    kind=starting  the five flagged `starter` in the box score
    kind=closing   the five with the most minutes (skipped when identical to the starters)
    minutes        mean of the five players' minutes
    net_rating     (sum of the five players' +/-) / 5, scaled to per-100 possessions with
                   lineup_poss = game_poss * minutes / game_minutes,
                   game_poss = mean over both teams of .96 (FGA + .44 FTA + TOV - OREB)
    off_rating     team game ORtg + (lineup net - team game net) / 2   (even split, rough)
    def_rating     off_rating - net_rating
Identical fives are aggregated across games (minutes-weighted mean net, total minutes,
games played). Rows with < MIN_LINEUP_MINUTES total minutes are dropped.
"""
from __future__ import annotations

import datetime as dt
import gzip
import io
import json
import os
import sys
import time
from typing import Iterable

import numpy as np
import pandas as pd
import requests

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(DATA_DIR, "cache")
BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
UA = "nba-projects-lineup-intelligence/1.0 (+https://github.com; polite: <=3 req/s)"
RATE_PER_S = 3.0
TIMEOUT = 20
MIN_LINEUP_MINUTES = 20.0
MAX_ROSTER = 17
SEASON_TYPES = {2, 3, 5}          # regular season, playoffs, play-in (no preseason)
NBA_TEAM_IDS = set(range(1, 31))  # ESPN ids of the 30 franchises; All-Star / Rising Stars sides use other ids

GAMES_CSV = os.path.join(CACHE_DIR, "espn_games.csv")
BOX_CSV = os.path.join(CACHE_DIR, "espn_boxscores.csv.gz")
TOTALS_CSV = os.path.join(CACHE_DIR, "espn_team_totals.csv.gz")
ROSTERS_JSON = os.path.join(CACHE_DIR, "espn_rosters.json")
SKIP_CSV = os.path.join(CACHE_DIR, "espn_skip.csv")      # game ids whose summary 404s (exhibitions), never retried
SOURCE_JSON = os.path.join(DATA_DIR, "source.json")

BOX_COLS = ["game_id", "date", "season", "season_type", "team_id", "team_abbr", "opp_abbr", "home",
            "player_id", "player_name", "position", "starter", "dnp", "min", "pts", "fgm", "fga", "fg3m", "fg3a",
            "ftm", "fta", "reb", "oreb", "dreb", "ast", "tov", "stl", "blk", "pf", "pm"]
TOT_COLS = ["game_id", "team_id", "team_abbr", "team_name", "opp_id", "opp_abbr", "home", "pts", "opp_pts",
            "fgm", "fga", "fg3a", "fta", "ftm", "reb", "oreb", "tov", "min"]


# ------------------------------------------------------------------ http ---
class _Client:
    def __init__(self, rate: float = RATE_PER_S):
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": UA, "Accept": "application/json"})
        self.min_gap = 1.0 / rate
        self.last = 0.0
        self.calls = 0

    def get(self, url: str, params: dict | None = None, retries: int = 4) -> dict:
        for attempt in range(retries):
            wait = self.last + self.min_gap - time.monotonic()
            if wait > 0:
                time.sleep(wait)
            self.last = time.monotonic()
            self.calls += 1
            try:
                r = self.s.get(url, params=params, timeout=TIMEOUT)
                if r.status_code == 200:
                    return r.json()
                if r.status_code in (400, 404):
                    raise RuntimeError(f"HTTP {r.status_code} for {r.url}")
            except (requests.RequestException, ValueError) as exc:
                if attempt == retries - 1:
                    raise RuntimeError(f"{type(exc).__name__}: {str(exc)[:120]}") from exc
            time.sleep(1.5 * 2 ** attempt)
        raise RuntimeError(f"gave up on {url}")


# --------------------------------------------------------------- helpers ---
def _f(x, default=0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _split(s: str) -> tuple[float, float]:
    """'7-16' -> (7, 16)."""
    try:
        a, b = str(s).split("-")
        return float(a), float(b)
    except ValueError:
        return 0.0, 0.0


def _season_months(season_year: int) -> list[str]:
    """ESPN season.year Y covers Oct Y-1 .. Jun Y."""
    return [f"{season_year - 1}{m:02d}" for m in (10, 11, 12)] + [f"{season_year}{m:02d}" for m in (1, 2, 3, 4, 5, 6)]


def current_season_year(client: _Client) -> int:
    d = client.get(f"{BASE}/scoreboard", {"limit": 5})
    return int(d["leagues"][0]["season"]["year"])


# ------------------------------------------------------------- schedule ---
def list_final_games(client: _Client, season_year: int, today: dt.date | None = None) -> pd.DataFrame:
    """Every FINAL game (regular season / play-in / playoffs) of one season from the monthly scoreboards."""
    today = today or dt.date.today()
    rows = []
    for ym in _season_months(season_year):
        if dt.date(int(ym[:4]), int(ym[4:]), 1) > today:
            break
        try:
            d = client.get(f"{BASE}/scoreboard", {"dates": ym, "limit": 1000})
        except RuntimeError as exc:
            print(f"[espn] scoreboard {ym}: {exc}")
            continue
        for e in d.get("events", []):
            st = e.get("status", {}).get("type", {})
            if st.get("name") != "STATUS_FINAL" or int(e["season"]["type"]) not in SEASON_TYPES:
                continue
            if int(e["season"]["year"]) != season_year:
                continue
            comp = e["competitions"][0]
            home = next(c for c in comp["competitors"] if c["homeAway"] == "home")
            away = next(c for c in comp["competitors"] if c["homeAway"] == "away")
            if int(home["team"]["id"]) not in NBA_TEAM_IDS or int(away["team"]["id"]) not in NBA_TEAM_IDS:
                continue                                   # All-Star weekend etc.
            rows.append({"game_id": int(e["id"]), "date": e["date"][:10], "season": season_year, "season_type": int(e["season"]["type"]),
                         "home_id": int(home["team"]["id"]), "home_abbr": home["team"]["abbreviation"], "home_score": _f(home.get("score")),
                         "away_id": int(away["team"]["id"]), "away_abbr": away["team"]["abbreviation"], "away_score": _f(away.get("score"))})
    return pd.DataFrame(rows).drop_duplicates("game_id") if rows else pd.DataFrame(columns=["game_id", "date", "season", "season_type"])


# --------------------------------------------------------------- summary ---
def parse_summary(d: dict, game: dict) -> tuple[list[dict], list[dict], list[dict]]:
    """-> (player rows, team-total rows, injury rows) for one game summary."""
    bs = d.get("boxscore", {})
    teams = bs.get("teams", [])
    if len(teams) != 2 or len(bs.get("players", [])) != 2:
        return [], [], []
    tot = {}
    for t in teams:
        st = {s["name"]: s.get("displayValue", "") for s in t.get("statistics", [])}
        fgm, fga = _split(st.get("fieldGoalsMade-fieldGoalsAttempted", "0-0"))
        f3m, f3a = _split(st.get("threePointFieldGoalsMade-threePointFieldGoalsAttempted", "0-0"))
        ftm, fta = _split(st.get("freeThrowsMade-freeThrowsAttempted", "0-0"))
        tid = int(t["team"]["id"])
        tot[tid] = {"game_id": game["game_id"], "team_id": tid, "team_abbr": t["team"]["abbreviation"], "team_name": t["team"].get("displayName", ""),
                    "home": int(t.get("homeAway") == "home"), "fgm": fgm, "fga": fga, "fg3a": f3a, "ftm": ftm, "fta": fta,
                    "reb": _f(st.get("totalRebounds")), "oreb": _f(st.get("offensiveRebounds")),
                    "tov": _f(st.get("totalTurnovers") or st.get("turnovers")), "pts": 0.0, "opp_pts": 0.0, "opp_id": 0, "opp_abbr": "", "min": 0.0}
    ids = list(tot)
    for a, b in ((ids[0], ids[1]), (ids[1], ids[0])):
        tot[a]["opp_id"], tot[a]["opp_abbr"] = b, tot[b]["team_abbr"]
    players = []
    for t in bs["players"]:
        tid = int(t["team"]["id"])
        if tid not in tot or not t.get("statistics"):
            continue
        stat = t["statistics"][0]
        labels = stat.get("labels", [])
        idx = {lab: i for i, lab in enumerate(labels)}
        need = ("MIN", "PTS", "FG", "3PT", "FT", "REB", "AST", "TO", "STL", "BLK", "OREB", "DREB", "PF", "+/-")
        if any(k not in idx for k in need):
            continue
        tmin = tpts = 0.0
        for a in stat.get("athletes", []):
            ath = a.get("athlete", {})
            s = a.get("stats") or []
            dnp = bool(a.get("didNotPlay")) or len(s) < len(labels)
            g = (lambda k: s[idx[k]]) if not dnp else (lambda k: "0")
            fgm, fga = _split(g("FG")); f3m, f3a = _split(g("3PT")); ftm, fta = _split(g("FT"))
            mins = _f(g("MIN"))
            row = {"game_id": game["game_id"], "date": game["date"], "season": game["season"], "season_type": game["season_type"],
                   "team_id": tid, "team_abbr": tot[tid]["team_abbr"], "opp_abbr": tot[tid]["opp_abbr"], "home": tot[tid]["home"],
                   "player_id": int(ath.get("id", 0)), "player_name": ath.get("displayName", ""),
                   "position": (ath.get("position") or {}).get("abbreviation", ""), "starter": int(bool(a.get("starter"))), "dnp": int(dnp),
                   "min": mins, "pts": _f(g("PTS")), "fgm": fgm, "fga": fga, "fg3m": f3m, "fg3a": f3a, "ftm": ftm, "fta": fta,
                   "reb": _f(g("REB")), "oreb": _f(g("OREB")), "dreb": _f(g("DREB")), "ast": _f(g("AST")), "tov": _f(g("TO")),
                   "stl": _f(g("STL")), "blk": _f(g("BLK")), "pf": _f(g("PF")), "pm": _f(str(g("+/-")).replace("+", ""))}
            if row["player_id"]:
                players.append(row)
            tmin += mins
            tpts += row["pts"]
        tot[tid]["min"] = tmin
        tot[tid]["pts"] = tpts
    for a, b in ((ids[0], ids[1]), (ids[1], ids[0])):
        tot[a]["opp_pts"] = tot[b]["pts"]
    injuries = []
    for t in d.get("injuries", []) or []:
        for i in t.get("injuries", []) or []:
            injuries.append({"team_abbr": t.get("team", {}).get("abbreviation", ""), "player_id": int(i.get("athlete", {}).get("id", 0) or 0),
                             "status": i.get("status", ""), "date": i.get("date", "")})
    return players, list(tot.values()), injuries


# ----------------------------------------------------------------- cache ---
def _read_csv(path: str, cols: list[str]) -> pd.DataFrame:
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame(columns=cols)


def _write_gz(df: pd.DataFrame, path: str) -> None:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=9) as fh:
        fh.write(buf.getvalue())


def refresh_cache(client: _Client, seasons: Iterable[int], prev_games_per_team: int | None = None,
                  max_games: int | None = None, verbose: bool = True) -> dict:
    """Fetch every final game of `seasons` that is not cached yet. Returns counts."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    games = _read_csv(GAMES_CSV, ["game_id", "date", "season", "season_type"])
    have = set(games.game_id.astype(int)) if len(games) else set()
    skip = _read_csv(SKIP_CSV, ["game_id"])
    have |= set(skip.game_id.astype(int)) if len(skip) else set()
    seasons = sorted(set(seasons))
    sched = pd.concat([list_final_games(client, y) for y in seasons], ignore_index=True)
    if len(sched) == 0:
        return {"fetched": 0, "cached": len(have), "schedule": 0}
    if prev_games_per_team and len(seasons) > 1:
        # keep every game of the latest season, but only each team's last N games of earlier seasons
        cur = seasons[-1]
        keep = set(sched[sched.season == cur].game_id)
        old = sched[sched.season < cur].sort_values("date")
        for abbr in pd.unique(pd.concat([old.home_abbr, old.away_abbr])):
            g = old[(old.home_abbr == abbr) | (old.away_abbr == abbr)].tail(prev_games_per_team)
            keep |= set(g.game_id)
        sched = sched[sched.game_id.isin(keep)]
    todo = sched[~sched.game_id.isin(have)].sort_values("date")
    if max_games is not None:
        todo = todo.tail(max_games)
    if verbose:
        print(f"[espn] schedule: {len(sched)} final games in seasons {seasons}; cached {len(have)}; fetching {len(todo)} "
              f"(~{len(todo) / RATE_PER_S / 60:.1f} min at {RATE_PER_S:.0f} req/s)")
    new_box, new_tot, new_games, failed = [], [], [], 0
    t0 = time.time()
    for k, (_, g) in enumerate(todo.iterrows(), start=1):
        game = {"game_id": int(g.game_id), "date": g.date, "season": int(g.season), "season_type": int(g.season_type)}
        try:
            d = client.get(f"{BASE}/summary", {"event": game["game_id"]})
            pl, tt, _inj = parse_summary(d, game)
        except RuntimeError as exc:
            failed += 1
            if verbose:
                print(f"[espn]   game {game['game_id']} failed: {exc}")
            if "HTTP 404" in str(exc):
                pd.concat([_read_csv(SKIP_CSV, ["game_id"]), pd.DataFrame([game])], ignore_index=True).to_csv(SKIP_CSV, index=False)
            continue
        if not pl or len(tt) != 2:
            failed += 1
            continue
        new_box.extend(pl); new_tot.extend(tt)
        new_games.append({**game, "home_abbr": g.home_abbr, "away_abbr": g.away_abbr, "home_score": g.home_score, "away_score": g.away_score})
        if verbose and (k % 100 == 0 or k == len(todo)):
            print(f"[espn]   {k}/{len(todo)} summaries ({time.time() - t0:.0f}s)")
        if k % 200 == 0:                      # checkpoint so an interrupted run keeps its progress
            _append(new_box, new_tot, new_games); new_box, new_tot, new_games = [], [], []
    _append(new_box, new_tot, new_games)
    games = _read_csv(GAMES_CSV, ["game_id"])
    return {"fetched": int(len(todo) - failed), "failed": failed, "cached": int(len(games)), "schedule": int(len(sched)),
            "seconds": round(time.time() - t0, 1)}


def _append(box_rows: list[dict], tot_rows: list[dict], game_rows: list[dict]) -> None:
    if not game_rows:
        return
    box = _read_csv(BOX_CSV, BOX_COLS)
    tot = _read_csv(TOTALS_CSV, TOT_COLS)
    games = _read_csv(GAMES_CSV, ["game_id"])
    box = pd.concat([box, pd.DataFrame(box_rows)[BOX_COLS]], ignore_index=True)
    tot = pd.concat([tot, pd.DataFrame(tot_rows)[TOT_COLS]], ignore_index=True)
    games = pd.concat([games, pd.DataFrame(game_rows)], ignore_index=True).drop_duplicates("game_id")
    _write_gz(box, BOX_CSV)
    _write_gz(tot, TOTALS_CSV)
    games.sort_values(["date", "game_id"]).to_csv(GAMES_CSV, index=False)


def fetch_rosters(client: _Client, verbose: bool = True) -> dict:
    """{team_id: {team: {...}, players: [...]}} for all 30 teams; also saved to the cache."""
    teams = client.get(f"{BASE}/teams", {"limit": 50})["sports"][0]["leagues"][0]["teams"]
    out = {}
    for t in teams:
        tm = t["team"]
        try:
            d = client.get(f"{BASE}/teams/{tm['id']}/roster")
        except RuntimeError as exc:
            print(f"[espn] roster {tm['abbreviation']}: {exc}")
            continue
        players = []
        for a in d.get("athletes", []):
            inj = (a.get("injuries") or [{}])[0]
            players.append({"player_id": int(a["id"]), "player_name": a.get("displayName", ""), "height_in": _f(a.get("height"), np.nan),
                            "weight_lb": _f(a.get("weight"), np.nan), "age": a.get("age"),
                            "position": (a.get("position") or {}).get("abbreviation", ""), "jersey": a.get("jersey", ""),
                            "injury_status": inj.get("status", "") or "", "status": (a.get("status") or {}).get("type", "")})
        out[str(tm["id"])] = {"team": {"id": int(tm["id"]), "abbr": tm["abbreviation"], "name": tm["displayName"],
                                       "record": tm.get("recordSummary", "")}, "players": players}
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(ROSTERS_JSON, "w") as fh:
        json.dump({"fetched_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%MZ"), "teams": out}, fh)
    if verbose:
        print(f"[espn] rosters: {len(out)} teams, {sum(len(v['players']) for v in out.values())} players")
    return out


# --------------------------------------------------------------- players ---
def build_players(box: pd.DataFrame, tot: pd.DataFrame, rosters: dict, seasons: list[int]) -> pd.DataFrame:
    cur = max(seasons)
    box = box[box.season.isin(seasons) & (box.dnp == 0) & (box["min"] > 0) & box.team_id.isin(NBA_TEAM_IDS)].copy()
    tot = tot[tot.team_id.isin(NBA_TEAM_IDS) & tot.opp_id.isin(NBA_TEAM_IDS)]
    box = box[box.game_id.isin(tot.game_id)]
    t = tot.set_index(["game_id", "team_id"])
    opp = tot.set_index(["game_id", "opp_id"])          # opponent totals keyed by (game, the team they faced)
    key = pd.MultiIndex.from_arrays([box.game_id, box.team_id])
    for c in ("fga", "fta", "tov", "fgm", "reb", "min"):
        box[f"tm_{c}"] = t[c].reindex(key).to_numpy()
    for c in ("fga", "fta", "tov", "oreb", "reb", "fg3a", "pts"):
        box[f"op_{c}"] = opp[c].reindex(key).to_numpy()
    box["tm_min5"] = box.tm_min / 5.0
    box["op_poss"] = 0.96 * (box.op_fga + 0.44 * box.op_fta + box.op_tov - box.op_oreb)
    box["tm_drtg"] = 100 * box.op_pts / box.op_poss.clip(lower=60)
    # per-row pieces that sum into the rate stats
    box["usg_num"] = (box.fga + 0.44 * box.fta + box.tov) * box.tm_min5
    box["usg_den"] = box["min"] * (box.tm_fga + 0.44 * box.tm_fta + box.tm_tov)
    box["ast_den"] = (box["min"] / box.tm_min5) * box.tm_fgm - box.fgm
    box["reb_num"] = box.reb * box.tm_min5
    box["reb_den"] = box["min"] * (box.tm_reb + box.op_reb)
    box["stl_num"] = box.stl * box.tm_min5
    box["stl_den"] = box["min"] * box.op_poss
    box["blk_num"] = box.blk * box.tm_min5
    box["blk_den"] = box["min"] * (box.op_fga - box.op_fg3a)
    box["drtg_w"] = box.tm_drtg * box["min"]
    box["gs_off"] = box.pts + 0.4 * box.fgm - 0.7 * box.fga - 0.4 * (box.fta - box.ftm) + 0.7 * box.oreb + 0.7 * box.ast - box.tov
    box["gs_def"] = 0.3 * box.dreb + box.stl + 0.7 * box.blk - 0.4 * box.pf
    sums = box.groupby("player_id")[["min", "pts", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "reb", "oreb", "dreb", "ast", "tov",
                                     "stl", "blk", "pf", "pm", "usg_num", "usg_den", "ast_den", "reb_num", "reb_den", "stl_num",
                                     "stl_den", "blk_num", "blk_den", "drtg_w", "gs_off", "gs_def"]].sum()
    games = box.groupby("player_id").size().rename("games")
    cur_min = box[box.season == cur].groupby("player_id")["min"].sum().rename("minutes_cur")
    cur_games = box[box.season == cur].groupby("player_id").size().rename("games_cur")
    last_row = box.sort_values("date").groupby("player_id").tail(1).set_index("player_id")
    df = pd.DataFrame(index=sums.index)
    df["player_name"] = last_row.player_name
    df["box_position"] = last_row.position
    df["last_team_abbr"] = last_row.team_abbr
    df["last_team_id"] = last_row.team_id
    df["games"] = games
    df["minutes_pooled"] = sums["min"]
    df["minutes"] = cur_min.reindex(df.index).fillna(0.0)
    df["games_cur"] = cur_games.reindex(df.index).fillna(0).astype(int)
    m = sums["min"].clip(lower=1)
    df["usage_pct"] = 100 * sums.usg_num / sums.usg_den.replace(0, np.nan)
    df["fg3_pct"] = sums.fg3m / sums.fg3a.replace(0, np.nan)
    df["fg3a_rate"] = sums.fg3a / sums.fga.replace(0, np.nan)
    df["ast_pct"] = 100 * sums.ast / sums.ast_den.clip(lower=1)
    df["reb_pct"] = 100 * sums.reb_num / sums.reb_den.replace(0, np.nan)
    df["stl_pct"] = 100 * sums.stl_num / sums.stl_den.replace(0, np.nan)
    df["blk_pct"] = 100 * sums.blk_num / sums.blk_den.replace(0, np.nan)
    df["tov_pct"] = 100 * sums.tov / (sums.fga + 0.44 * sums.fta + sums.tov).replace(0, np.nan)
    df["def_rating_proxy"] = sums.drtg_w / m
    df["pm36"] = 36 * sums.pm / m
    df["pm36_shrunk"] = df.pm36 * sums["min"] / (sums["min"] + 300.0)
    df["pts36"] = 36 * sums.pts / m
    df["ast36"] = 36 * sums.ast / m
    df["reb36"] = 36 * sums.reb / m
    gs_off36, gs_def36 = 36 * sums.gs_off / m, 36 * sums.gs_def / m
    w = sums["min"]

    def z(x):
        mu = np.average(x, weights=w)
        sd = np.sqrt(np.average((x - mu) ** 2, weights=w)) or 1.0
        return (x - mu) / sd * 2.5
    df["obpm"] = 0.5 * z(gs_off36) + 0.5 * df.pm36_shrunk / 2
    df["dbpm"] = 0.5 * z(gs_def36) + 0.5 * df.pm36_shrunk / 2
    # low-minute players: shrink the noisy rates toward league medians (weighted by minutes / (minutes + 150))
    rate_cols = ["usage_pct", "fg3_pct", "fg3a_rate", "ast_pct", "reb_pct", "stl_pct", "blk_pct", "tov_pct"]
    med = {c: float(np.average(df[c].fillna(df[c].median()), weights=w)) for c in rate_cols}
    shrink = (sums["min"] / (sums["min"] + 150.0)).to_numpy()
    for c in rate_cols:
        v = df[c].fillna(med[c]).to_numpy()
        df[c] = med[c] + (v - med[c]) * shrink
    df["def_rating_proxy"] = df.def_rating_proxy.fillna(df.def_rating_proxy.median())

    # Roster join: current team, height, position, injury status, active flag (capped at MAX_ROSTER by minutes)
    ros_rows = []
    for tid, v in rosters.items():
        for p in v["players"]:
            ros_rows.append({**p, "team_id": int(tid), "team_abbr": v["team"]["abbr"], "team": v["team"]["name"]})
    ros = pd.DataFrame(ros_rows).drop_duplicates("player_id").set_index("player_id")
    team_names = {int(tid): v["team"]["name"] for tid, v in rosters.items()}
    all_ids = sorted(set(df.index) | set(ros.index))
    out = df.reindex(all_ids)
    on_roster = out.index.isin(ros.index)
    out["player_name"] = ros.player_name.reindex(out.index).where(on_roster, out.player_name)
    out["team_id"] = np.where(on_roster, ros.team_id.reindex(out.index), out.last_team_id)
    out["team_id"] = pd.to_numeric(out.team_id, errors="coerce").fillna(0).astype(int)
    out["team_abbr"] = np.where(on_roster, ros.team_abbr.reindex(out.index), out.last_team_abbr)
    out["team"] = [team_names.get(int(t), str(a)) for t, a in zip(out.team_id, out.team_abbr)]
    out["position"] = ros.position.reindex(out.index).where(on_roster, out.box_position).fillna("F").replace("", "F")
    out["height_in"] = pd.to_numeric(ros.height_in.reindex(out.index), errors="coerce")
    pos_med = out.groupby("position").height_in.median()
    fallback = {"PG": 74.5, "SG": 76.5, "G": 75.5, "SF": 78.5, "F": 79.0, "PF": 80.5, "C": 83.0}
    out["height_in"] = out.height_in.fillna(out.position.map(pos_med)).fillna(out.position.map(fallback)).fillna(79.0)
    out["age"] = ros.age.reindex(out.index)
    out["injury_status"] = ros.injury_status.reindex(out.index).fillna("")
    out["roster_status"] = ros.status.reindex(out.index).fillna("")
    out["on_roster"] = on_roster.astype(int)
    # players on a roster with no box score at all (rookies / two-way who have not played): league-median rates, 0 minutes
    for c in rate_cols:
        out[c] = out[c].fillna(med[c])
    for c in ("obpm", "dbpm", "pm36", "pm36_shrunk", "pts36", "ast36", "reb36"):
        out[c] = out[c].fillna(0.0)
    for c in ("minutes", "minutes_pooled", "games", "games_cur"):
        out[c] = out[c].fillna(0)
    out["def_rating_proxy"] = out.def_rating_proxy.fillna(df.def_rating_proxy.median())
    # active = on roster, capped at MAX_ROSTER per team by (current minutes, pooled minutes)
    out["active"] = 0
    for tid, grp in out[out.on_roster == 1].groupby("team_id"):
        keep = grp.sort_values(["minutes", "minutes_pooled"], ascending=False).index[:MAX_ROSTER]
        out.loc[keep, "active"] = 1
    out = out.reset_index().rename(columns={"index": "player_id"})
    out["player_id"] = out.player_id.astype(int)
    cols = ["player_id", "player_name", "team", "team_abbr", "team_id", "position", "height_in", "age", "minutes", "games_cur",
            "minutes_pooled", "games", "usage_pct", "fg3_pct", "fg3a_rate", "ast_pct", "reb_pct", "stl_pct", "blk_pct", "tov_pct",
            "obpm", "dbpm", "def_rating_proxy", "pm36", "pts36", "ast36", "reb36", "injury_status", "roster_status", "on_roster", "active"]
    out = out[cols].sort_values(["team", "minutes"], ascending=[True, False]).reset_index(drop=True)
    for c in ("height_in", "usage_pct", "ast_pct", "reb_pct", "tov_pct", "def_rating_proxy", "pm36", "pts36", "ast36", "reb36"):
        out[c] = out[c].astype(float).round(1)
    for c in ("fg3_pct", "fg3a_rate"):
        out[c] = out[c].astype(float).round(3)
    for c in ("stl_pct", "blk_pct", "obpm", "dbpm"):
        out[c] = out[c].astype(float).round(2)
    out["minutes"] = out.minutes.astype(float).round(0)
    return out


# --------------------------------------------------------------- lineups ---
def build_lineups(box: pd.DataFrame, tot: pd.DataFrame, players: pd.DataFrame, seasons: list[int],
                  min_minutes: float = MIN_LINEUP_MINUTES) -> pd.DataFrame:
    box = box[box.season.isin(seasons) & (box.dnp == 0) & box.team_id.isin(NBA_TEAM_IDS)].copy()
    known = set(players.player_id)
    tot = tot[tot.team_id.isin(NBA_TEAM_IDS) & tot.opp_id.isin(NBA_TEAM_IDS)].copy()
    tot["poss"] = 0.96 * (tot.fga + 0.44 * tot.fta + tot.tov - tot.oreb)
    game_poss = tot.groupby("game_id").poss.mean()
    t = tot.set_index(["game_id", "team_id"])
    rows = []
    for (gid, tid), g in box.groupby(["game_id", "team_id"], sort=False):
        try:
            tr = t.loc[(gid, tid)]
        except KeyError:
            continue
        game_min = max(tr["min"] / 5.0, 48.0)
        poss = float(game_poss.get(gid, np.nan))
        if not np.isfinite(poss) or poss < 60:
            poss = 100.0 * game_min / 48.0
        team_net = 100 * (tr.pts - tr.opp_pts) / poss
        team_off = 100 * tr.pts / poss
        starters = g[g.starter == 1]
        if len(starters) > 5:
            starters = starters.nlargest(5, "min")
        closing = g.nlargest(5, "min")
        units = []
        if len(starters) == 5:
            units.append(("starting", starters))
        if len(closing) == 5 and (len(starters) != 5 or set(closing.player_id) != set(starters.player_id)):
            units.append(("closing", closing))
        for kind, u in units:
            pids = sorted(int(x) for x in u.player_id)
            if not set(pids) <= known:
                continue
            mins = float(u["min"].mean())
            if mins <= 0:
                continue
            lineup_poss = poss * mins / game_min
            net = float(u.pm.sum()) / 5.0 / lineup_poss * 100
            off = team_off + (net - team_net) / 2
            rows.append({"team_id": int(tid), "team_abbr": tr.team_abbr, "team": tr.team_name, "game_id": gid, "season": int(g.season.iloc[0]),
                         "kind": kind, "p1": pids[0], "p2": pids[1], "p3": pids[2], "p4": pids[3], "p5": pids[4],
                         "minutes": mins, "net": net, "off": off})
    if not rows:
        return pd.DataFrame(columns=["lineup_id", "team", "p1", "p2", "p3", "p4", "p5", "minutes", "off_rating", "def_rating", "net_rating"])
    df = pd.DataFrame(rows)
    df["w_net"] = df.net * df.minutes
    df["w_off"] = df.off * df.minutes
    df["is_start"] = (df.kind == "starting").astype(int)
    agg = df.groupby(["team_id", "team_abbr", "team", "p1", "p2", "p3", "p4", "p5"], as_index=False).agg(
        minutes=("minutes", "sum"), games=("game_id", "nunique"), w_net=("w_net", "sum"), w_off=("w_off", "sum"),
        starts=("is_start", "sum"), seasons=("season", lambda s: "/".join(str(x) for x in sorted(set(s)))))
    agg["net_rating"] = agg.w_net / agg.minutes
    agg["off_rating"] = agg.w_off / agg.minutes
    agg["def_rating"] = agg.off_rating - agg.net_rating
    agg["kind"] = np.where(agg.starts > 0, "starting", "closing")
    agg = agg[agg.minutes >= min_minutes].copy()
    # centre the league so minutes-weighted off == def (every possession is scored by one team and allowed by the other)
    shift = np.average(agg.off_rating - agg.def_rating, weights=agg.minutes) / 2.0
    agg["off_rating"] -= shift
    agg["def_rating"] += shift
    agg = agg.sort_values(["team", "minutes"], ascending=[True, False]).reset_index(drop=True)
    agg.insert(0, "lineup_id", np.arange(1, len(agg) + 1))
    cols = ["lineup_id", "team", "team_abbr", "p1", "p2", "p3", "p4", "p5", "minutes", "off_rating", "def_rating", "net_rating",
            "games", "kind", "seasons"]
    out = agg[cols]
    for c in ("minutes", "off_rating", "def_rating", "net_rating"):
        out[c] = out[c].astype(float).round(1)
    return out


# ------------------------------------------------------------------ main ---
def load_espn_data(seasons: list[int] | None = None, prev_games_per_team: int | None = 40, refresh: bool = True,
                   out_dir: str = DATA_DIR, save: bool = True, verbose: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Refresh the cache (if `refresh`), build players.csv + lineups.csv, write source.json. Returns (players, lineups, source)."""
    client = _Client()
    t0 = time.time()
    if seasons is None:
        cur = current_season_year(client)
        seasons = [cur - 1, cur]
    stats = {"fetched": 0}
    if refresh:
        stats = refresh_cache(client, seasons, prev_games_per_team=prev_games_per_team, verbose=verbose)
        rosters = fetch_rosters(client, verbose=verbose)
    else:
        with open(ROSTERS_JSON) as fh:
            rosters = json.load(fh)["teams"]
    box = pd.read_csv(BOX_CSV)
    tot = pd.read_csv(TOTALS_CSV)
    games = pd.read_csv(GAMES_CSV)
    players = build_players(box, tot, rosters, seasons)
    lineups = build_lineups(box, tot, players, seasons)
    cur = max(seasons)
    used = games[games.season.isin(seasons) & games.game_id.isin(tot[tot.team_id.isin(NBA_TEAM_IDS) & tot.opp_id.isin(NBA_TEAM_IDS)].game_id)]
    source = {
        "source": "espn", "label": "ESPN", "chip": "Live data · ESPN",
        "fetched_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "seasons": [f"{y - 1}-{str(y)[2:]}" for y in seasons], "current_season": f"{cur - 1}-{str(cur)[2:]}",
        "games_used": int(len(used)), "games_current_season": int((used.season == cur).sum()),
        "games_cached": int(len(games)), "new_games_fetched": int(stats.get("fetched", 0)), "requests": client.calls,
        "latest_game_date": str(used.date.max()) if len(used) else None,
        "prev_games_per_team": prev_games_per_team,
        "players": int(len(players)), "players_active": int(players.active.sum()), "teams": int(players[players.active == 1].team.nunique()),
        "lineups": int(len(lineups)), "lineups_starting": int((lineups.kind == "starting").sum()), "lineups_closing": int((lineups.kind == "closing").sum()),
        "lineup_definition": "starting five / closing five (top-5 minutes) per game; net = sum(+/-)/5 per 100 est. possessions; NOT stint-level data",
        "seconds": round(time.time() - t0, 1),
    }
    if save:
        os.makedirs(out_dir, exist_ok=True)
        players.to_csv(os.path.join(out_dir, "players.csv"), index=False)
        lineups.to_csv(os.path.join(out_dir, "lineups.csv"), index=False)
        with open(os.path.join(out_dir, "source.json"), "w") as fh:
            json.dump(source, fh, indent=2)
    if verbose:
        print(f"[espn] {source['games_used']} games ({source['games_current_season']} this season, {source['current_season']}) -> "
              f"{len(players)} players ({source['players_active']} on {source['teams']} rosters), {len(lineups)} lineups "
              f"({source['lineups_starting']} starting / {source['lineups_closing']} closing) in {source['seconds']}s, {client.calls} requests")
    return players, lineups, source


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Refresh the ESPN cache and rebuild data/players.csv + data/lineups.csv")
    ap.add_argument("--seasons", type=int, nargs="*", help="ESPN season years, e.g. 2026 2027 (default: previous + current)")
    ap.add_argument("--prev-games", type=int, default=40, help="per-team cap on earlier-season games (0 = all)")
    ap.add_argument("--no-refresh", action="store_true", help="rebuild from the cache without any network calls")
    a = ap.parse_args()
    load_espn_data(a.seasons or None, prev_games_per_team=a.prev_games or None, refresh=not a.no_refresh)
