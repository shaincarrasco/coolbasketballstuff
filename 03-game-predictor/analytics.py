"""
Season-wide analytics on top of the trained model (all SAMPLE DATA).

  season_predictions()  every test-season game scored by the frozen XGBoost
                        model (trained on seasons 1-2 only) -> per-date slates
                        with hindcast hit/miss and running accuracy
  elo_history()         per-team pre-game Elo series (for sparklines / curves)
  power_rankings()      Elo table as of a date: rank change vs 30 days earlier,
                        L10 form string, off/def proxies, Elo sparkline points
  team_pages()          per-team drill-down: schedule + model prob per game,
                        rest/B2B record, home/away split, biggest upsets
  upsets() / tiers()    games the model gave the winner < 35 %, and a
                        predicted-probability -> actual-win-rate tier table

Everything is computed from `features.build_features()` output, i.e. strictly
pre-game information, plus the model's probabilities. Not betting advice.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from features import build_features
from model import load_model
from predict import _fmt_driver

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output"
UPSET_THRESHOLD = 0.35
TIER_EDGES = [0.0, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 1.0001]


def load_teams(path: Path | None = None) -> pd.DataFrame:
    path = Path(path) if path else ROOT / "data" / "teams.csv"
    t = pd.read_csv(path)
    if "conference" not in t.columns:  # older teams.csv: assign two conferences of 15
        t["conference"] = ["East" if i < len(t) // 2 else "West" for i in range(len(t))]
        t.to_csv(path, index=False)
    return t


def scored_features(games_path: Path | None = None) -> tuple[pd.DataFrame, str, list[str]]:
    """Feature frame for every game with model probability `p_home` (frozen model)
    and SHAP-style contributions for the test season. Returns (feats, test_season, cols)."""
    games_path = Path(games_path) if games_path else ROOT / "data" / "games.csv"
    games = pd.read_csv(games_path)
    feats = build_features(games)
    booster, calibrator, _, cols = load_model()
    dm = xgb.DMatrix(feats[cols], feature_names=cols)
    feats["p_home"] = np.clip(calibrator.predict(booster.predict(dm)), 0.01, 0.99)
    test_season = sorted(feats["season"].unique())[-1]
    is_test = feats["season"] == test_season
    contribs = booster.predict(xgb.DMatrix(feats.loc[is_test, cols], feature_names=cols), pred_contribs=True)[:, :-1]
    feats["drivers"] = None
    drv = []
    for (_, g), c in zip(feats[is_test].iterrows(), contribs):
        order = np.argsort(-np.abs(c))[:3]
        drv.append([_fmt_driver(cols[j], g[cols[j]], c[j]) for j in order if abs(c[j]) > 1e-4])
    feats.loc[is_test, "drivers"] = pd.Series(drv, index=feats.index[is_test], dtype=object)
    return feats, test_season, cols


# ---- 1) slates for the whole test season ---------------------------------
def season_predictions(feats: pd.DataFrame, test_season: str) -> dict:
    te = feats[(feats["season"] == test_season)].sort_values(["date", "game_id"])
    slates: dict[str, list[dict]] = {}
    running: dict[str, dict] = {}
    hits = n = 0
    for day, grp in te.groupby("date", sort=True):
        key = str(day.date())
        rows = []
        for _, g in grp.iterrows():
            ph = float(g["p_home"])
            pick = g["home_team"] if ph >= 0.5 else g["away_team"]
            row = {"id": g["game_id"], "h": g["home_team"], "a": g["away_team"], "ph": round(ph, 3),
                   "eh": round(float(g["home_elo"])), "ea": round(float(g["away_elo"])),
                   "ep": round(float(g["elo_prob"]), 3), "pick": pick,
                   "drv": g["drivers"] if isinstance(g["drivers"], list) else [],
                   "rh": int(g["home_rest"]), "ra": int(g["away_rest"])}
            if not pd.isna(g["home_pts"]):
                row["hp"], row["ap"] = int(g["home_pts"]), int(g["away_pts"])
                row["w"] = g["home_team"] if g["home_pts"] > g["away_pts"] else g["away_team"]
                row["ok"] = row["w"] == pick
                hits += row["ok"]; n += 1
            rows.append(row)
        rows.sort(key=lambda r: -max(r["ph"], 1 - r["ph"]))
        slates[key] = rows
        day_hits = sum(1 for r in rows if r.get("ok"))
        day_n = sum(1 for r in rows if "ok" in r)
        running[key] = {"n": day_n, "hits": day_hits, "std_n": n, "std_hits": hits}
    return {"season": test_season, "dates": list(slates.keys()), "slates": slates, "running": running}


# ---- 2) Elo history + rankings --------------------------------------------
def elo_history(feats: pd.DataFrame, season: str | None = None) -> dict[str, list[tuple[pd.Timestamp, float]]]:
    """Per team: [(date, pre-game Elo), ..., (last_date, post-game Elo)]."""
    f = feats if season is None else feats[feats["season"] == season]
    hist: dict[str, list] = {}
    for _, g in f.sort_values(["date", "game_id"]).iterrows():
        hist.setdefault(g["home_team"], []).append((g["date"], float(g["home_elo"])))
        hist.setdefault(g["away_team"], []).append((g["date"], float(g["away_elo"])))
    # append each team's post-game Elo after its last played game
    from features import elo_update
    played = f.dropna(subset=["home_pts"]).sort_values(["date", "game_id"])
    for t in hist:
        tg = played[(played["home_team"] == t) | (played["away_team"] == t)]
        if tg.empty:
            continue
        tg = tg.iloc[-1]
        eh, ea = elo_update(float(tg["home_elo"]), float(tg["away_elo"]), int(tg["home_pts"] - tg["away_pts"]))
        hist[t].append((tg["date"], eh if tg["home_team"] == t else ea))
    return hist


def _elo_at(h: list[tuple[pd.Timestamp, float]], when: pd.Timestamp) -> float:
    v = h[0][1]
    for d, e in h:
        if d <= when:
            v = e
        else:
            break
    return v


def power_rankings(feats: pd.DataFrame, teams: pd.DataFrame, as_of: pd.Timestamp | None = None,
                   season: str | None = None) -> list[dict]:
    season = season or sorted(feats["season"].unique())[-1]
    f = feats[feats["season"] == season]
    played = f.dropna(subset=["home_pts"])
    as_of = pd.to_datetime(as_of) if as_of is not None else played["date"].max()
    played = played[played["date"] <= as_of]
    hist = elo_history(feats[feats["date"] <= as_of], season)
    names = dict(teams[["code", "name"]].values)
    conf = dict(teams[["code", "conference"]].values)
    rows = []
    for t in teams["code"]:
        tg = played[(played["home_team"] == t) | (played["away_team"] == t)].sort_values("date")
        is_home = tg["home_team"] == t
        pf = np.where(is_home, tg["home_pts"], tg["away_pts"]).astype(float)
        pa = np.where(is_home, tg["away_pts"], tg["home_pts"]).astype(float)
        wins = pf > pa
        h = hist.get(t, [(as_of, 1500.0)])
        elo_now = h[-1][1]
        elo_30 = _elo_at(h, as_of - pd.Timedelta(days=30))
        spark = [round(e) for _, e in h]
        # thin the sparkline to <= 60 points
        if len(spark) > 60:
            idx = np.linspace(0, len(spark) - 1, 60).round().astype(int)
            spark = [spark[i] for i in idx]
        rows.append({
            "team": t, "name": names.get(t, t), "conf": conf.get(t, "?"),
            "elo": round(elo_now, 1), "elo_30": round(elo_30, 1), "d_elo_30": round(elo_now - elo_30, 1),
            "w": int(wins.sum()), "l": int((~wins).sum()),
            "l10": "".join("W" if w else "L" for w in wins[-10:]),
            "off10": round(float(pf[-10:].mean()), 1) if len(pf) else None,
            "def10": round(float(pa[-10:].mean()), 1) if len(pa) else None,
            "net10": round(float((pf[-10:] - pa[-10:]).mean()), 1) if len(pf) else None,
            "off": round(float(pf.mean()), 1) if len(pf) else None,
            "def": round(float(pa.mean()), 1) if len(pa) else None,
            "spark": spark,
        })
    rows.sort(key=lambda r: -r["elo"])
    for i, r in enumerate(rows):
        r["rank"] = i + 1
    rank_30 = {r["team"]: i + 1 for i, r in enumerate(sorted(rows, key=lambda r: -r["elo_30"]))}
    for r in rows:
        r["rank_30"] = rank_30[r["team"]]
        r["d_rank"] = r["rank_30"] - r["rank"]  # positive = moved up
    return rows


# ---- 3) team pages ---------------------------------------------------------
def team_pages(feats: pd.DataFrame, teams: pd.DataFrame, season: str) -> dict[str, dict]:
    f = feats[feats["season"] == season].sort_values(["date", "game_id"])
    names = dict(teams[["code", "name"]].values)
    conf = dict(teams[["code", "conference"]].values)
    pages = {}
    for t in teams["code"]:
        tg = f[(f["home_team"] == t) | (f["away_team"] == t)]
        sched, elo_curve = [], []
        rec = {"home": [0, 0], "away": [0, 0], "b2b": [0, 0], "rested": [0, 0], "fav": [0, 0], "dog": [0, 0]}
        for _, g in tg.iterrows():
            home = g["home_team"] == t
            opp = g["away_team"] if home else g["home_team"]
            p = float(g["p_home"]) if home else 1 - float(g["p_home"])
            rest = int(g["home_rest"] if home else g["away_rest"])
            elo_pre = float(g["home_elo"] if home else g["away_elo"])
            elo_curve.append([str(g["date"].date()), round(elo_pre)])
            row = {"d": str(g["date"].date()), "opp": opp, "home": bool(home), "p": round(p, 3), "rest": rest}
            if not pd.isna(g["home_pts"]):
                pf = int(g["home_pts"] if home else g["away_pts"]); pa = int(g["away_pts"] if home else g["home_pts"])
                w = pf > pa
                row.update({"pf": pf, "pa": pa, "w": bool(w)})
                key = "home" if home else "away"
                rec[key][0 if w else 1] += 1
                rec["b2b" if rest == 1 else "rested"][0 if w else 1] += 1
                rec["fav" if p >= 0.5 else "dog"][0 if w else 1] += 1
            sched.append(row)
        played = [r for r in sched if "w" in r]
        upset_wins = sorted([r for r in played if r["w"] and r["p"] < 0.5], key=lambda r: r["p"])[:3]
        upset_losses = sorted([r for r in played if not r["w"] and r["p"] >= 0.5], key=lambda r: -r["p"])[:3]
        pages[t] = {
            "name": names.get(t, t), "conf": conf.get(t, "?"),
            "w": sum(r["w"] for r in played), "l": sum(not r["w"] for r in played),
            "rec": rec, "sched": sched, "elo": elo_curve,
            "upset_wins": upset_wins, "upset_losses": upset_losses,
            "model_acc": round(float(np.mean([(r["p"] >= 0.5) == r["w"] for r in played])), 3) if played else None,
        }
    return pages


# ---- 4) upsets + confidence tiers -----------------------------------------
def upsets(feats: pd.DataFrame, season: str, threshold: float = UPSET_THRESHOLD) -> list[dict]:
    te = feats[(feats["season"] == season)].dropna(subset=["home_pts"])
    out = []
    for _, g in te.iterrows():
        home_won = g["home_pts"] > g["away_pts"]
        p_winner = float(g["p_home"]) if home_won else 1 - float(g["p_home"])
        if p_winner < threshold:
            winner = g["home_team"] if home_won else g["away_team"]
            loser = g["away_team"] if home_won else g["home_team"]
            # "what the model missed": pre-game context of the winner, phrased from the winner's side
            sign = 1 if home_won else -1
            missed = []
            if g["home_pts"] - g["away_pts"] != 0:
                missed.append(f"margin {abs(int(g['home_pts'] - g['away_pts']))}")
            wr, lr = (g["home_rest"], g["away_rest"]) if home_won else (g["away_rest"], g["home_rest"])
            if lr == 1 and wr > 1:
                missed.append("favourite on B2B")
            elif lr == 1:
                missed.append("both on B2B")
            wp10 = sign * float(g["wp10_diff"]) if not pd.isna(g["wp10_diff"]) else 0.0
            if wp10 > 0.1:
                missed.append(f"winner hotter L10 ({wp10:+.1f})")
            streak = sign * float(g["streak_diff"]) if not pd.isna(g["streak_diff"]) else 0.0
            if streak >= 3:
                missed.append(f"winner streak edge {streak:+.0f}")
            h2h = g["h2h_home_wp"] if not pd.isna(g["h2h_home_wp"]) else None
            if h2h is not None:
                hw = h2h if home_won else 1 - h2h
                if hw >= 0.75 and g["h2h_n"] >= 2:
                    missed.append(f"H2H favoured winner {hw:.0%}")
            if not missed[1:]:
                missed.append("no pre-game signal: noise")
            out.append({"d": str(g["date"].date()), "winner": winner, "loser": loser,
                        "score": f"{int(g['home_pts'])}-{int(g['away_pts'])}", "home_won": bool(home_won),
                        "p_winner": round(p_winner, 3), "elo_gap": round(float(g["home_elo"] - g["away_elo"]) * -sign),
                        "missed": missed, "drv": g["drivers"] if isinstance(g["drivers"], list) else []})
    out.sort(key=lambda r: r["p_winner"])
    return out


def tiers(feats: pd.DataFrame, season: str, edges: list[float] = TIER_EDGES) -> list[dict]:
    te = feats[(feats["season"] == season)].dropna(subset=["home_pts"])
    conf = np.maximum(te["p_home"], 1 - te["p_home"]).to_numpy()
    hit = ((te["p_home"] >= 0.5) == (te["home_pts"] > te["away_pts"])).to_numpy()
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi)
        rows.append({"tier": f"{max(lo, .5):.0%}-{min(hi, 1):.0%}", "lo": lo, "hi": min(hi, 1.0), "n": int(m.sum()),
                     "pred": round(float(conf[m].mean()), 3) if m.any() else None,
                     "actual": round(float(hit[m].mean()), 3) if m.any() else None})
    return rows


# ---- bundle ----------------------------------------------------------------
def build_all(games_path: Path | None = None, as_of: str | None = None) -> dict:
    """Compute everything the report / CLI needs once; also writes CSV exports."""
    teams = load_teams()
    feats, test_season, cols = scored_features(games_path)
    sp = season_predictions(feats, test_season)
    rk = power_rankings(feats, teams, as_of=as_of, season=test_season)
    tp = team_pages(feats, teams, test_season)
    up = upsets(feats, test_season)
    tr = tiers(feats, test_season)
    OUTPUT.mkdir(exist_ok=True)
    pd.DataFrame([{k: v for k, v in r.items() if k != "spark"} for r in rk]).to_csv(OUTPUT / "rankings.csv", index=False)
    pd.DataFrame([{"date": d, **{k: v for k, v in g.items() if k != "drv"}, "drivers": " | ".join(g["drv"])}
                  for d in sp["dates"] for g in sp["slates"][d]]).to_csv(OUTPUT / "season_predictions.csv", index=False)
    pd.DataFrame([{**{k: v for k, v in r.items() if k not in ("missed", "drv")}, "missed": " | ".join(r["missed"])}
                  for r in up]).to_csv(OUTPUT / "upsets.csv", index=False)
    pd.DataFrame(tr).to_csv(OUTPUT / "confidence_tiers.csv", index=False)
    return {"season": test_season, "feats": feats, "teams": teams, "season_predictions": sp,
            "rankings": rk, "team_pages": tp, "upsets": up, "tiers": tr}


def print_rankings(rows: list[dict], as_of: str | None = None) -> None:
    print(f"\nPOWER RANKINGS (Elo){'  as of ' + as_of if as_of else ''}   [SAMPLE DATA]")
    print(f"{'#':>3} {'team':<5}{'conf':<6}{'elo':>7}{'30d':>7}{'rk30':>5}  {'W-L':<7}{'L10':<11}{'off':>6}{'def':>6}{'net':>6}")
    for r in rows:
        arrow = f"{r['d_rank']:+d}" if r["d_rank"] else "="
        print(f"{r['rank']:>3} {r['team']:<5}{r['conf']:<6}{r['elo']:>7.0f}{r['d_elo_30']:>+7.0f}{arrow:>5}  "
              f"{str(r['w']) + '-' + str(r['l']):<7}{r['l10']:<11}{r['off10'] or 0:>6.1f}{r['def10'] or 0:>6.1f}{r['net10'] or 0:>+6.1f}")


def print_team(code: str, page: dict, rank_row: dict | None) -> None:
    print(f"\n{code} · {page['name']}  ({page['conf']})  {page['w']}-{page['l']}"
          + (f"  Elo {rank_row['elo']:.0f} (#{rank_row['rank']})" if rank_row else "") + "   [SAMPLE DATA]")
    r = page["rec"]
    print(f"home {r['home'][0]}-{r['home'][1]}  away {r['away'][0]}-{r['away'][1]}  "
          f"B2B {r['b2b'][0]}-{r['b2b'][1]}  rested {r['rested'][0]}-{r['rested'][1]}  "
          f"as favourite {r['fav'][0]}-{r['fav'][1]}  as underdog {r['dog'][0]}-{r['dog'][1]}  model acc {page['model_acc']:.0%}")
    print("biggest upset wins :", ", ".join(f"{u['d']} {'vs' if u['home'] else '@'} {u['opp']} {u['pf']}-{u['pa']} (p={u['p']:.0%})" for u in page["upset_wins"]) or "-")
    print("biggest upset losses:", ", ".join(f"{u['d']} {'vs' if u['home'] else '@'} {u['opp']} {u['pf']}-{u['pa']} (p={u['p']:.0%})" for u in page["upset_losses"]) or "-")
    print(f"\n{'date':<12}{'':<3}{'opp':<5}{'p(win)':>8}{'rest':>5}  result")
    for g in page["sched"][-15:]:
        res = f"{'W' if g['w'] else 'L'} {g['pf']}-{g['pa']}" if "w" in g else "-"
        print(f"{g['d']:<12}{'vs' if g['home'] else '@':<3}{g['opp']:<5}{g['p']:>8.0%}{g['rest']:>5}  {res}")
    print("(last 15 games)  Not betting advice. Nothing here is guaranteed.")


if __name__ == "__main__":
    b = build_all()
    print_rankings(b["rankings"])
    print(f"upsets (<{UPSET_THRESHOLD:.0%}): {len(b['upsets'])}; tiers: {json.dumps(b['tiers'])}")
