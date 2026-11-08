"""
Real NBA data from ESPN's free public scoreboard API (no key).

Endpoints (unofficial, undocumented, may change without notice):
    scoreboard  https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates=YYYYMMDD&limit=50
                also accepts a whole month: dates=YYYYMM&limit=500 (verified; the
                YYYYMMDD-YYYYMMDD range form returns 400 for the NBA, so months are
                used for backfill and single days for the incremental refresh)
    standings   https://site.api.espn.com/apis/v2/sports/basketball/nba/standings
                (East / West children -> conference per team)

Politeness: <= 3 requests/s (min 0.35 s between calls), User-Agent header,
4 retries with exponential backoff on 429 / 5xx / network errors.

Dates: ESPN gives the UTC tip-off; `date` here is the US/Eastern local date so
rest-day / back-to-back logic matches how the NBA schedule is read.

Files written (all relative to data/):
    cache/espn_games.csv   every event fetched (incremental cache, committed)
    cache/espn_meta.json   last refresh timestamp, request count, counts
    games.csv              pipeline schema: game_id, date, season, home_team,
                           away_team, home_pts, away_pts  + status, neutral,
                           home_name, away_name, season_type, tip_utc, espn_id
                           (regular season + playoffs; playoffs are flagged by
                           season_type != 2 and excluded from training by default,
                           see features.load_games)
    cache/espn_scheduled.csv  unplayed regular-season games in the next 7 days
    teams.csv              code, name, conference, espn_id (conference from standings)

Incremental refresh: only dates after the last FINAL game in the cache are
fetched (re-fetching the last 3 days to catch late finals), through today + 7
days; the rest of the current season's published schedule is refreshed at most
once a day. First run from 2023-10-24: ~40 month requests, ~45 s.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

HERE = Path(__file__).resolve().parent
CACHE_DIR = HERE / "cache"
CACHE = CACHE_DIR / "espn_games.csv"
META = CACHE_DIR / "espn_meta.json"
SCHEDULED = CACHE_DIR / "espn_scheduled.csv"
BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
STANDINGS = "https://site.api.espn.com/apis/v2/sports/basketball/nba/standings"
UA = "nba-projects-game-predictor/1.0 (learning project; polite client)"
ET = ZoneInfo("America/New_York")
DEFAULT_START = date(2023, 10, 24)   # opening night 2023-24
MIN_INTERVAL = 0.35                  # seconds between requests (<= 3 req/s)
RELOOK_DAYS = 3                      # re-fetch the last N days to catch late finals
SCHEDULED_DAYS = 7
TIMEOUT = 20

# static fallback for the offline path (standings endpoint unreachable); refreshed from standings when online
FALLBACK_CONF = {"East": "ATL BKN BOS CHA CHI CLE DET IND MIA MIL NY ORL PHI TOR WSH".split(),
                 "West": "DAL DEN GS HOU LAC LAL MEM MIN NO OKC PHX POR SA SAC UTAH".split()}

CACHE_COLS = ["espn_id", "date", "tip_utc", "season", "season_year", "season_type", "home_team", "away_team",
              "home_pts", "away_pts", "status", "neutral", "home_name", "away_name", "home_id", "away_id", "fetched_at"]
GAMES_COLS = ["game_id", "date", "season", "home_team", "away_team", "home_pts", "away_pts",
              "status", "neutral", "cup_final", "home_name", "away_name", "season_type", "tip_utc", "espn_id"]


class EspnError(RuntimeError):
    pass


# ---- HTTP ---------------------------------------------------------------
class _Client:
    def __init__(self):
        self.last = 0.0
        self.requests = 0

    def get(self, url: str, retries: int = 4) -> dict:
        delay = 1.0
        for attempt in range(retries + 1):
            wait = MIN_INTERVAL - (time.monotonic() - self.last)
            if wait > 0:
                time.sleep(wait)
            self.last = time.monotonic()
            self.requests += 1
            try:
                req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
                with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                    return json.loads(r.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                if exc.code in (429, 500, 502, 503, 504) and attempt < retries:
                    time.sleep(delay); delay *= 2
                    continue
                raise EspnError(f"HTTP {exc.code} for {url}") from exc
            except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
                if attempt < retries:
                    time.sleep(delay); delay *= 2
                    continue
                raise EspnError(f"{type(exc).__name__}: {exc} for {url}") from exc
        raise EspnError(f"gave up on {url}")


def season_label(year: int) -> str:
    """ESPN season.year is the year the season ends: 2026 -> '2025-26'."""
    return f"{year - 1}-{str(year)[-2:]}"


def et_date(iso_utc: str) -> date:
    dt = datetime.strptime(iso_utc, "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc)
    return dt.astimezone(ET).date()


def today_et() -> date:
    return datetime.now(timezone.utc).astimezone(ET).date()


def parse_events(events: list[dict], fetched_at: str) -> list[dict]:
    rows = []
    for e in events:
        try:
            comp = e["competitions"][0]
            comps = {c["homeAway"]: c for c in comp["competitors"]}
            h, a = comps["home"], comps["away"]
            if not h["team"].get("abbreviation") or not a["team"].get("abbreviation"):
                continue  # TBD opponents (preseason / exhibition placeholders)
        except (KeyError, IndexError):
            continue
        status = (comp.get("status") or e.get("status") or {}).get("type", {}).get("name", "")
        final = status.startswith("STATUS_FINAL")

        def score(c):
            s = c.get("score")
            try:
                return int(float(s)) if final and s not in (None, "") else None
            except (TypeError, ValueError):
                return None

        rows.append({
            "espn_id": str(e["id"]), "date": et_date(e["date"]).isoformat(), "tip_utc": e["date"],
            "season": season_label(int(e["season"]["year"])), "season_year": int(e["season"]["year"]),
            "season_type": int(e["season"].get("type", 0)),
            "home_team": h["team"]["abbreviation"], "away_team": a["team"]["abbreviation"],
            "home_pts": score(h), "away_pts": score(a), "status": status,
            "neutral": bool(comp.get("neutralSite", False)),
            "home_name": h["team"].get("displayName", h["team"]["abbreviation"]),
            "away_name": a["team"].get("displayName", a["team"]["abbreviation"]),
            "home_id": str(h["team"].get("id", "")), "away_id": str(a["team"].get("id", "")),
            "fetched_at": fetched_at,
        })
    return rows


def _months(start: date, end: date) -> list[str]:
    out, y, m = [], start.year, start.month
    while (y, m) <= (end.year, end.month):
        out.append(f"{y}{m:02d}")
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def season_end(d: date) -> date:
    """Last day of the regular-season window that contains d (regular season ends mid-April;
    use April 30 as the fetch horizon of the published schedule)."""
    return date(d.year + 1, 4, 30) if d.month >= 7 else date(d.year, 4, 30)


# ---- main entry ----------------------------------------------------------
def fetch_games(start: date | str = DEFAULT_START, end: date | str | None = None, cache_path: Path = CACHE,
                scheduled_days: int = SCHEDULED_DAYS, force: bool = False, log=print) -> dict:
    """Refresh the cache incrementally and (re)write games.csv, teams.csv and
    cache/espn_scheduled.csv. Returns a summary dict. Raises EspnError if ESPN
    cannot be reached at all (caller decides whether to fall back)."""
    start = pd.Timestamp(start).date()
    today = today_et()
    end = pd.Timestamp(end).date() if end else max(season_end(today), today + timedelta(days=scheduled_days))
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    cli = _Client()

    cached = pd.read_csv(cache_path, dtype={"espn_id": str, "home_id": str, "away_id": str}) \
        if cache_path.exists() and not force else pd.DataFrame(columns=CACHE_COLS)
    finals = cached[cached["status"].astype(str).str.startswith("STATUS_FINAL")] if len(cached) else cached
    last_final = pd.Timestamp(finals["date"].max()).date() if len(finals) else None

    rows: list[dict] = []
    fetched_dates: set[str] = set()
    if last_final is None:
        # cold start: whole months from start to horizon
        for ym in _months(start, end):
            d = cli.get(f"{BASE}/scoreboard?dates={ym}&limit=500")
            rows += parse_events(d.get("events", []), now_iso)
            log(f"  [espn] {ym[:4]}-{ym[4:]}: {len(d.get('events', []))} events")
        refreshed_schedule = True
    else:
        # incremental: single days from last final - 3 through today + scheduled_days
        d0 = max(start, last_final - timedelta(days=RELOOK_DAYS))
        d1 = min(end, today + timedelta(days=scheduled_days))
        days = [d0 + timedelta(days=i) for i in range((d1 - d0).days + 1)]
        if len(days) > 20:  # long gap: months are cheaper
            for ym in _months(d0, d1):
                d = cli.get(f"{BASE}/scoreboard?dates={ym}&limit=500")
                rows += parse_events(d.get("events", []), now_iso)
        else:
            for day in days:
                d = cli.get(f"{BASE}/scoreboard?dates={day:%Y%m%d}&limit=50")
                rows += parse_events(d.get("events", []), now_iso)
        fetched_dates = {r["date"] for r in rows}
        log(f"  [espn] refreshed {d0} .. {d1}: {len(rows)} events in {cli.requests} requests")
        # rest of the published schedule: at most once a day
        future = cached[(cached["date"] > d1.isoformat())]
        stale = True
        if len(future):
            fa = pd.to_datetime(future["fetched_at"], utc=True, errors="coerce").max()
            stale = pd.isna(fa) or (pd.Timestamp.now(tz="UTC") - fa) > pd.Timedelta(hours=24)
        refreshed_schedule = False
        if stale and d1 < end:
            n0 = cli.requests
            for ym in _months(d1 + timedelta(days=1), end):
                d = cli.get(f"{BASE}/scoreboard?dates={ym}&limit=500")
                rows += parse_events(d.get("events", []), now_iso)
            refreshed_schedule = True
            log(f"  [espn] refreshed published schedule through {end}: {cli.requests - n0} requests")

    new = pd.DataFrame(rows, columns=CACHE_COLS)
    if last_final is None:
        merged = new
    else:
        # the fetched window supersedes everything cached inside it (catches postponements / reschedules)
        if refreshed_schedule:
            keep = cached[cached["date"] < d0.isoformat()]
        else:
            keep = cached[(cached["date"] < d0.isoformat()) | (cached["date"] > d1.isoformat())]
        merged = pd.concat([keep, new], ignore_index=True)
    merged = merged.drop_duplicates("espn_id", keep="last")
    merged = merged[merged["date"] >= start.isoformat()].sort_values(["date", "tip_utc", "espn_id"]).reset_index(drop=True)
    merged.to_csv(cache_path, index=False)

    teams = _teams_from_standings(cli, merged, log)
    summary = _write_outputs(merged, teams, today, scheduled_days, log)
    summary.update({"source": "espn", "updated_at": now_iso, "requests": cli.requests, "cache": str(cache_path),
                    "start": start.isoformat(), "end": end.isoformat(), "incremental": last_final is not None})
    META.write_text(json.dumps(summary, indent=2))
    (HERE / "meta.json").write_text(json.dumps(summary, indent=2))
    return summary


def _write_outputs(merged: pd.DataFrame, teams: pd.DataFrame, today: date, scheduled_days: int, log=print) -> dict:
    """games.csv (pipeline schema), teams.csv and cache/espn_scheduled.csv from the cache frame."""
    teams.to_csv(HERE / "teams.csv", index=False)
    g = merged[merged["season_type"].isin([2, 3])].copy()             # regular season + playoffs (flagged)
    g = g[~g["status"].isin(["STATUS_POSTPONED", "STATUS_CANCELED", "STATUS_CANCELLED"])]
    g = g[g["home_team"].isin(teams["code"]) & g["away_team"].isin(teams["code"])]  # drops All-Star etc.
    g["game_id"] = "E" + g["espn_id"].astype(str)
    is_final = g["status"].str.startswith("STATUS_FINAL")
    g.loc[~is_final, ["home_pts", "away_pts"]] = float("nan")
    # NBA Cup final: a neutral-site game in December that is the only game of its day.
    # It is a real game (kept for Elo / training) but does not count in the standings.
    per_day = g.groupby("date")["espn_id"].transform("size")
    g["cup_final"] = (g["season_type"] == 2) & g["neutral"].astype(bool) & (pd.to_datetime(g["date"]).dt.month == 12) & (per_day == 1)
    g = g[GAMES_COLS].sort_values(["date", "tip_utc", "game_id"]).reset_index(drop=True)
    g.to_csv(HERE / "games.csv", index=False)

    sched = g[(g["season_type"] == 2) & g["home_pts"].isna() & (g["date"] >= today.isoformat())
              & (g["date"] <= (today + timedelta(days=scheduled_days)).isoformat())]
    sched.to_csv(SCHEDULED, index=False)

    reg = g[g["season_type"] == 2]
    summary = {
        "today_et": today.isoformat(),
        "n_cached_events": int(len(merged)), "n_games": int(len(g)), "n_regular": int(len(reg)),
        "n_final": int(reg["home_pts"].notna().sum()), "n_scheduled_7d": int(len(sched)),
        "n_scheduled_total": int(reg["home_pts"].isna().sum()),
        "last_final": str(reg.loc[reg["home_pts"].notna(), "date"].max()),
        "seasons": {s: int(n) for s, n in reg.groupby("season").size().items()},
    }
    log(f"  [espn] games.csv: {summary['n_regular']} regular-season games ({summary['n_final']} final, "
        f"{summary['n_scheduled_total']} scheduled), last final {summary['last_final']}")
    return summary


def from_cache_offline(cache_path: Path = CACHE, log=print) -> dict | None:
    """ESPN unreachable: rebuild games.csv / teams.csv from the committed cache (stale but real)."""
    if not Path(cache_path).exists():
        return None
    merged = pd.read_csv(cache_path, dtype={"espn_id": str, "home_id": str, "away_id": str})
    if len(merged) < 500:
        return None
    teams_path = HERE / "teams.csv"
    teams = pd.read_csv(teams_path) if teams_path.exists() else pd.DataFrame()
    if "espn_id" not in teams.columns or (teams.get("conference") == "?").any():
        teams = _teams_from_standings(None, merged, log)   # offline: names from the cache, static conference map
    prev = json.loads(META.read_text()) if META.exists() else {}
    summary = _write_outputs(merged, teams, today_et(), SCHEDULED_DAYS, log)
    summary.update({"source": "espn", "offline": True, "updated_at": prev.get("updated_at"), "requests": 0,
                    "cache": str(cache_path), "incremental": True})
    (HERE / "meta.json").write_text(json.dumps(summary, indent=2))
    return summary


def _teams_from_standings(cli: _Client | None, games: pd.DataFrame, log=print) -> pd.DataFrame:
    names: dict[str, tuple[str, str]] = {}
    for side in ("home", "away"):
        for code, name, tid in games[[f"{side}_team", f"{side}_name", f"{side}_id"]].drop_duplicates(f"{side}_team").itertuples(index=False):
            names.setdefault(code, (name, tid))
    conf: dict[str, str] = {}
    try:
        if cli is None:
            raise EspnError("offline")
        s = cli.get(STANDINGS)
        for child in s.get("children", []):
            cname = "East" if "East" in child.get("name", "") else "West" if "West" in child.get("name", "") else child.get("abbreviation", "?")
            for entry in (child.get("standings") or {}).get("entries", []):
                t = entry["team"]
                conf[t["abbreviation"]] = cname
                names.setdefault(t["abbreviation"], (t.get("displayName", t["abbreviation"]), str(t.get("id", ""))))
    except (EspnError, KeyError, TypeError) as exc:
        log(f"  [espn] standings unavailable ({exc}); keeping cached conferences if any")
        old = HERE / "teams.csv"
        if old.exists():
            prev = pd.read_csv(old)
            if "espn_id" in prev.columns and not (prev["conference"] == "?").any():
                conf = dict(prev[["code", "conference"]].values)
    if not conf:
        conf = {c: side for side, codes in FALLBACK_CONF.items() for c in codes if c in names}
    codes = sorted(c for c in conf) if conf else sorted(names)
    # Only current NBA franchises: those in the standings (30). Fallback: the 30 most frequent codes.
    if not conf:
        freq = pd.concat([games["home_team"], games["away_team"]]).value_counts()
        codes = sorted(freq.index[:30])
    return pd.DataFrame([{"code": c, "name": names.get(c, (c, ""))[0], "conference": conf.get(c, "?"),
                          "espn_id": names.get(c, (c, ""))[1]} for c in codes])


def load_or_fallback(seed: int = 42, log=print, **kw) -> tuple[Path, str]:
    """ESPN if reachable, otherwise the seeded simulation. Returns (games_path, source)."""
    try:
        fetch_games(log=log, **kw)
        return HERE / "games.csv", "espn"
    except EspnError as exc:
        cached = from_cache_offline(log=log)
        if cached:
            log(f"[generate] ESPN unreachable ({exc}) -> using the CACHED ESPN data (last final {cached['last_final']}, "
                f"fetched {cached.get('updated_at')})")
            return HERE / "games.csv", "espn"
        log(f"[generate] ESPN unreachable ({exc}) -> falling back to SAMPLE data")
        sys.path.insert(0, str(HERE))
        from sample_generator import generate
        p = generate(HERE / "games.csv", seed=seed, seasons=3)
        return p, "sample"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Fetch real NBA games from ESPN's public scoreboard API into data/games.csv")
    ap.add_argument("--start", default=DEFAULT_START.isoformat())
    ap.add_argument("--end", default=None)
    ap.add_argument("--force", action="store_true", help="ignore the cache and backfill everything")
    args = ap.parse_args()
    print(json.dumps(fetch_games(args.start, args.end, force=args.force), indent=2))
