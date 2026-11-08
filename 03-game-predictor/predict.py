"""
Predict a slate of games for a given date.

Features for the slate are built from games strictly before that date, so
running this on a historical date is an honest hindcast (actual results are
shown next to the prediction when they exist, purely as a sanity check).

Key drivers come from XGBoost's `pred_contribs=True` (SHAP-style log-odds
contributions per feature), rendered as short human-readable strings like
"Elo +85", "rest +1 day", "L10 form +.3".
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import xgboost as xgb

from features import build_features, load_games
from model import load_model

ROOT = Path(__file__).resolve().parent
MIN_SLATE = 6
ET = ZoneInfo("America/New_York")


def today_et() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc).astimezone(ET).date())


def read_meta() -> dict:
    p = ROOT / "data" / "meta.json"
    return json.loads(p.read_text()) if p.exists() else {"source": "sample"}


def default_date(games: pd.DataFrame) -> pd.Timestamp:
    """Real data: today's slate (US/Eastern), else the next day with games (from
    data/cache/espn_scheduled.csv when present), else the last played date.
    Sample data: the last day of the final season with a full slate (>= MIN_SLATE games)."""
    d = pd.to_datetime(games["date"])
    unplayed = games["home_pts"].isna() | games["away_pts"].isna()
    if unplayed.any():
        today = today_et()
        sched_path = ROOT / "data" / "cache" / "espn_scheduled.csv"
        upcoming = pd.to_datetime(pd.read_csv(sched_path)["date"]) if sched_path.exists() else d[unplayed]
        upcoming = upcoming[upcoming >= today]
        if (d == today).any():
            return today
        if len(upcoming):
            return upcoming.min()
        return d[unplayed].min() if (d[unplayed] >= today).any() else d[~unplayed].max()
    last_season = games.loc[d == d.max(), "season"].iloc[0]
    counts = games[games["season"] == last_season].groupby("date").size()
    full = counts[counts >= MIN_SLATE]
    return pd.to_datetime(full.index.max() if len(full) else counts.index.max())


def tip_et(tip_utc) -> str | None:
    """'2025-11-08T00:00Z' -> '7:00 PM' (US/Eastern)."""
    if not isinstance(tip_utc, str):
        return None
    try:
        t = datetime.strptime(tip_utc, "%Y-%m-%dT%H:%MZ").replace(tzinfo=timezone.utc).astimezone(ET)
    except ValueError:
        return None
    return t.strftime("%-I:%M %p")


# ---- driver formatting -------------------------------------------------
def _fmt_driver(feature: str, value: float, contrib: float) -> str:
    arrow = "↑" if contrib > 0 else "↓"   # direction of push for the HOME team
    v = 0.0 if pd.isna(value) else float(value)
    sgn = "+" if v >= 0 else "-"
    a = abs(v)
    text = {
        "elo_diff": f"Elo {sgn}{a:.0f}",
        "elo_prob": f"Elo p(home) {v:.0%}",
        "rest_diff": f"rest {sgn}{a:.0f} day" + ("" if a == 1 else "s"),
        "home_rest": f"home rest {a:.0f}d",
        "away_rest": f"away rest {a:.0f}d",
        "b2b_diff": ("home on B2B" if v > 0 else "away on B2B") if v != 0 else "no B2B",
        "home_b2b": "home on B2B" if v else "home rested",
        "away_b2b": "away on B2B" if v else "away rested",
        "wp5_diff": f"L5 form {sgn}{a:.1f}".replace("0.", "."),
        "wp10_diff": f"L10 form {sgn}{a:.1f}".replace("0.", "."),
        "wp20_diff": f"L20 form {sgn}{a:.1f}".replace("0.", "."),
        "pdiff10_diff": f"L10 margin {sgn}{a:.1f}",
        "off10_diff": f"L10 offence {sgn}{a:.1f} pts",
        "def10_diff": f"L10 defence {'-' if v >= 0 else '+'}{a:.1f} pts",
        "season_wp_diff": f"season win% {sgn}{a:.2f}".replace("0.", "."),
        "split_wp_diff": f"home/away split {sgn}{a:.2f}".replace("0.", "."),
        "streak_diff": f"streak {sgn}{a:.0f}",
        "gp_diff": f"games played {sgn}{a:.0f}",
        "h2h_home_wp": f"H2H {v:.0%}",
        "h2h_n": f"H2H meetings {a:.0f}",
    }.get(feature, f"{feature} {v:+.2f}")
    return f"{text} {arrow}"


def predict_date(date: str | None = None, games_path: Path | None = None, top_drivers: int = 3) -> dict:
    games_path = Path(games_path) if games_path else ROOT / "data" / "games.csv"
    games = load_games(games_path)
    teams_path = games_path.parent / "teams.csv"
    names = dict(pd.read_csv(teams_path)[["code", "name"]].values) if teams_path.exists() else {}
    meta = read_meta()

    target = pd.to_datetime(date) if date else default_date(games)
    feats = build_features(games)
    slate = feats[feats["date"] == target].copy()
    if slate.empty:
        raise SystemExit(f"no games on {target.date()} in {games_path}")

    booster, calibrator, cal_name, cols = load_model()
    dm = xgb.DMatrix(slate[cols], feature_names=cols)
    raw = booster.predict(dm)
    p_home = calibrator.predict(raw)
    contribs = booster.predict(dm, pred_contribs=True)  # (n, n_features + 1) with bias last

    # season hindcast: every played game of the current season scored by the same frozen model
    season = sorted(feats["season"].unique())[-1]
    cur = feats[(feats["season"] == season) & feats["home_win"].notna()]
    p_cur = calibrator.predict(booster.predict(xgb.DMatrix(cur[cols], feature_names=cols)))
    hind_hits = int(((p_cur >= 0.5) == (cur["home_win"] == 1)).sum())

    rows = []
    for i, (_, g) in enumerate(slate.iterrows()):
        c = contribs[i][:-1]
        order = np.argsort(-np.abs(c))[:top_drivers]
        drivers = [_fmt_driver(cols[j], g[cols[j]], c[j]) for j in order if abs(c[j]) > 1e-4]
        ph = float(np.clip(p_home[i], 0.01, 0.99))
        pick = g["home_team"] if ph >= 0.5 else g["away_team"]
        row = {
            "game_id": g["game_id"],
            "date": str(target.date()),
            "home_team": g["home_team"],
            "away_team": g["away_team"],
            "home_name": names.get(g["home_team"], g["home_team"]),
            "away_name": names.get(g["away_team"], g["away_team"]),
            "p_home": round(ph, 4),
            "p_away": round(1 - ph, 4),
            "pick": pick,
            "confidence": round(max(ph, 1 - ph), 4),
            "elo_home": round(float(g["home_elo"]), 1),
            "elo_away": round(float(g["away_elo"]), 1),
            "elo_prob_home": round(float(g["elo_prob"]), 4),
            "drivers": drivers,
            "contributions": {cols[j]: round(float(c[j]), 4) for j in range(len(cols))},
        }
        if "espn_id" in g and not pd.isna(g["espn_id"]):
            row["espn_id"] = str(int(g["espn_id"]))
            row["tip_et"] = tip_et(g.get("tip_utc"))
            row["status"] = g.get("status")
        if not pd.isna(g["home_pts"]):
            row["actual"] = {"home_pts": int(g["home_pts"]), "away_pts": int(g["away_pts"]),
                             "winner": g["home_team"] if g["home_pts"] > g["away_pts"] else g["away_team"]}
            row["correct"] = row["actual"]["winner"] == pick
        rows.append(row)
    rows.sort(key=lambda r: -r["confidence"])

    out = {
        "date": str(target.date()),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": meta.get("source", "sample"),
        "data_updated_at": meta.get("updated_at"),
        "sample_data": meta.get("source", "sample") != "espn",
        "disclaimer": "Not betting advice. Nothing here is guaranteed.",
        "model": {"type": "xgboost", "calibrator": cal_name, "n_features": len(cols)},
        "n_games": len(rows),
        "slate_accuracy": (float(np.mean([r["correct"] for r in rows])) if all("correct" in r for r in rows) else None),
        "season_hindcast": {"season": season, "n": int(len(cur)), "hits": hind_hits,
                            "accuracy": round(hind_hits / len(cur), 4) if len(cur) else None,
                            "small_sample": bool(len(cur) < 200)},
        "games": rows,
    }
    (ROOT / "output").mkdir(exist_ok=True)
    (ROOT / "output" / "predictions.json").write_text(json.dumps(out, indent=2))
    return out


def print_slate(out: dict) -> None:
    tag = "SAMPLE DATA" if out.get("sample_data", True) else f"ESPN data, updated {out.get('data_updated_at', '')[:16]}Z"
    print(f"\nSLATE {out['date']}  ({out['n_games']} games)  [{tag}]  model=xgb+{out['model']['calibrator']}")
    print(f"{'pick':<5}{'home':<6}{'away':<6}{'p(home)':>9}{'conf':>7}  drivers")
    for r in out["games"]:
        res = f"   tip {r['tip_et']} ET" if r.get("tip_et") and "actual" not in r else ""
        if "actual" in r:
            res = f"   actual {r['actual']['home_pts']}-{r['actual']['away_pts']} {'OK' if r['correct'] else 'X'}"
        print(f"{r['pick']:<5}{r['home_team']:<6}{r['away_team']:<6}{r['p_home']:>9.1%}{r['confidence']:>7.0%}  "
              + ", ".join(r["drivers"]) + res)
    if out["slate_accuracy"] is not None:
        print(f"hindcast accuracy on this slate: {out['slate_accuracy']:.0%}")
    h = out.get("season_hindcast")
    if h and h["n"]:
        print(f"season {h['season']} hindcast: {h['hits']}/{h['n']} = {h['accuracy']:.1%}" + ("  (small sample)" if h["small_sample"] else ""))
    print(out["disclaimer"])


if __name__ == "__main__":
    import sys
    print_slate(predict_date(sys.argv[1] if len(sys.argv) > 1 else None))
