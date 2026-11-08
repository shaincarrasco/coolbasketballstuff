"""
Five IQ — Lineup Intelligence CLI.

    python cli.py generate [--seed 5150] [--per-team 70] [--real --season 2024-25]
    python cli.py train
    python cli.py recommend --team "Harbor City" [--top 10] [--min-bigs 1] [--max-zero-minute 0]
                            [--must "Okonkwo"] [--exclude "Thorne"]
                            [--add-player "Xiong"]        # trade scenario: pull a player from another team
                            [--vs "Ironvale"]             # opponent-aware re-rank (heuristic, labelled)
                            [--rotation] [--json]
    python cli.py demo [--site ../site/lineup]   # generate -> train -> recommend (all teams) -> exports -> report (+ site page)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR = os.path.join(ROOT, "output")


def _load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    pp, lp = os.path.join(DATA_DIR, "players.csv"), os.path.join(DATA_DIR, "lineups.csv")
    if not (os.path.exists(pp) and os.path.exists(lp)):
        print("No data found. Run `python cli.py generate` first.")
        sys.exit(1)
    return pd.read_csv(pp), pd.read_csv(lp)


def cmd_generate(args) -> None:
    if getattr(args, "real", False):
        from data.nba_api_loader import load_real_data
        players, lineups = load_real_data(season=args.season)
        if players is not None:
            print(f"Real data: {len(players)} players, {len(lineups)} lineups -> data/")
            return
        print("Falling back to the synthetic sample generator.")
    from data.sample_generator import generate
    players, lineups = generate(seed=args.seed, per_team=args.per_team)
    print(f"Sample data: {len(players)} players across {players.team.nunique()} fictional teams, "
          f"{len(lineups)} five-man lineups -> data/players.csv, data/lineups.csv")


def cmd_train(args) -> dict:
    from model import train
    players, lineups = _load_data()
    return train(players, lineups)


def _constraints(args):
    from optimize import Constraints
    return Constraints(min_bigs=args.min_bigs, max_zero_minute=args.max_zero_minute,
                       must_include=tuple(args.must or ()), exclude=tuple(args.exclude or ()))


def _last(name: str) -> str:
    return name.split()[-1]


def _print_recs(out: dict) -> None:
    print(f"\n{out['team']} — {out['combos_evaluated']} feasible of {out['combos_total']} fives ({out['search']}), constraints {out['constraints']}")
    if out.get("added_players"):
        print(f"Trade scenario: added {', '.join(out['added_players'])} -> {out['roster_size']}-man roster")
    mu = out.get("most_used_lineup")
    if mu:
        print(f"Coach's most-used five: {', '.join(mu['players'])}\n   observed {mu['observed_net']:+.1f} in {mu['minutes']:.0f} min · model fit {mu['fit_score']} · rank {mu['rank_all']}/{out['combos_total']}")
    m = out.get("matchup")
    if m:
        print(f"\n[HEURISTIC] vs {m['opponent']} — their best five: {', '.join(_last(n) for n in m['opponent_best_five'])}")
        print(f"   opp rim BLK% {m['opponent_profile']['rim']:.1f} (lg {m['league_profile']['rim']:.1f}) · opp STL% {m['opponent_profile']['hands']:.1f} (lg {m['league_profile']['hands']:.1f}) · opp height {m['opponent_profile']['height']:.1f}")
        print(f"   {m['formula']}")
        print(f"\n{'#':<3}{'Fit':>5} {'Adj':>5} {'Net':>6} {'adj':>5}  {'S/H/B':<6} {'Ht':>5}  Lineup")
    else:
        print(f"\n{'#':<3}{'Fit':>5} {'Net':>6} {'Off':>6} {'Def':>6}  {'S/H/B':<6} {'Ht':>5} {'rank':>5}  Lineup")
    for r in out["recommendations"]:
        names = ", ".join(_last(p["name"]) for p in r["players"])
        shb = f"{r['shooters']}/{r['handlers']}/{r['bigs']}"
        if m:
            print(f"{r['rank']:<3}{r['fit_score']:>5.0f} {r['adjusted_fit']:>5.0f} {r['predicted_net']:>+6.1f} {r['matchup_adjustment']:>+5.1f}  {shb:<6} {r['height_mean']:>5.1f}  {names}")
        else:
            print(f"{r['rank']:<3}{r['fit_score']:>5.0f} {r['predicted_net']:>+6.1f} {r['predicted_off']:>6.1f} {r['predicted_def']:>6.1f}  {shb:<6} {r['height_mean']:>5.1f} {r['rank_all']:>5}  {names}")
        print(f"      {r['summary']}")
    if out.get("swaps_for_top"):
        print("\nWhat-if swaps for #1:")
        for s in out["swaps_for_top"]:
            print(f"   swap {_last(s['out'])} for {_last(s['in'])}: {s['delta_fit']:+.1f} fit (net {s['delta_net']:+.1f})")
    rot = out.get("rotation")
    if rot:
        if "error" in rot:
            print(f"\nRotation: {rot['error']}")
        else:
            print(f"\nRotation (minutes budget heuristic, overlap {rot['overlap']}, {rot['minutes_allocated']:.0f}/240 min allocated, rotation net {rot['rotation_net']:+.1f}):")
            for key in ("starters", "bench", "closing"):
                u = rot[key]
                mins = "" if key == "closing" else " · " + ", ".join(f"{_last(p['name'])} {p['minutes_assigned']:.0f}" for p in u["players"])
                print(f"   {key:<9} fit {u['fit_score']:>5.1f} net {u['predicted_net']:>+5.1f}{mins}" if mins else
                      f"   {key:<9} fit {u['fit_score']:>5.1f} net {u['predicted_net']:>+5.1f} · {', '.join(_last(p['name']) for p in u['players'])}")
            if rot.get("bench_outranks_starters"):
                print("   note: the model rates the bench unit higher; the starters are forced by the minutes budget")
    print(f"\n{out['note']}")


def _build_pages(meta: dict, all_recs: dict, tables: dict, players: pd.DataFrame, site: str | None) -> list[str]:
    from report import build_report
    return build_report(meta, all_recs, tables, players, site_dir=site)


def cmd_recommend(args) -> dict:
    from optimize import recommend
    players, lineups = _load_data()
    out = recommend(args.team, players, lineups, _constraints(args), top_n=args.top,
                    add_players=tuple(args.add_player or ()), vs=args.vs, with_rotation=args.rotation)
    if args.json:
        print(json.dumps(out, indent=2))
    else:
        _print_recs(out)
    if args.site:
        # the page covers every team, so rebuild all recommendations + tables (a few seconds)
        from optimize import recommend_all, score_team
        with open(os.path.join(ROOT, "models", "meta.json")) as fh:
            meta = json.load(fh)
        cons = _constraints(args)
        all_recs = recommend_all(players, lineups, cons, top_n=args.top)
        tables = {t: score_team(t, players, lineups, cons) for t in players.team.unique()}
        for p in _build_pages(meta, all_recs, tables, players, args.site):
            print(f"wrote {p}")
    return out


def cmd_demo(args) -> None:
    t0 = time.time()
    print("== 1/5 generate ==")
    cmd_generate(args)
    print("\n== 2/5 train ==")
    meta = cmd_train(args)
    print("\n== 3/5 recommend + rotations (all teams) ==")
    from optimize import recommend_all, score_team, export_team
    players, lineups = _load_data()
    cons = _constraints(args)
    all_recs = recommend_all(players, lineups, cons, top_n=args.top)
    os.makedirs(OUT_DIR, exist_ok=True)
    payload = {
        "note": "Sample data. Fictional teams and players. Fit scores are for demonstration only.",
        "fit_score_transform": f"100 * sigmoid((net_pred - {meta['fit_center']:.3f}) / {meta['fit_scale']:.3f})",
        "metrics": meta["evaluation"]["metrics"],
        "teams": all_recs,
    }
    rec_path = os.path.join(OUT_DIR, "recommendations.json")
    with open(rec_path, "w") as fh:
        json.dump(payload, fh, indent=2)
    best_team = max(all_recs, key=lambda t: all_recs[t]["recommendations"][0]["fit_score"])
    for t in all_recs:
        r = all_recs[t]["recommendations"][0]
        print(f"  {t:<26} best fit {r['fit_score']:>5.1f}  net {r['predicted_net']:>+6.1f}  "
              f"{', '.join(_last(p['name']) for p in r['players'])}")
    _print_recs(all_recs[best_team])

    print("\n== 4/5 exports (every five + pairwise synergy per team) ==")
    tables = {}
    for t in players.team.unique():
        tt = score_team(t, players, lineups, cons)
        tables[t] = tt
        p1, p2 = export_team(tt, OUT_DIR)
        print(f"  wrote {os.path.relpath(p1, ROOT)}  {os.path.relpath(p2, ROOT)}")

    print("\n== 5/5 report ==")
    paths = _build_pages(meta, all_recs, tables, players, args.site)
    print(f"wrote {rec_path}\n" + "\n".join(f"wrote {p}" for p in paths) + f"\ndone in {time.time() - t0:.1f}s")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="cli.py", description="Five IQ — Lineup Intelligence")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_gen(p):
        p.add_argument("--seed", type=int, default=5150)
        p.add_argument("--per-team", type=int, default=70, help="historical lineups per team (sample data)")
        p.add_argument("--real", action="store_true", help="try nba_api (stats.nba.com) before falling back to sample data")
        p.add_argument("--season", default="2024-25")

    def add_rec(p, need_team: bool):
        if need_team:
            p.add_argument("--team", required=True, help="team name or substring, e.g. 'Harbor City'")
            p.add_argument("--add-player", action="append", help="trade scenario: add a player from another team (name substring, repeatable)")
            p.add_argument("--vs", help="opponent team: re-rank with the documented matchup HEURISTIC")
            p.add_argument("--rotation", action="store_true", help="also build starters + bench + closing five under a minutes budget")
        p.add_argument("--top", type=int, default=10)
        p.add_argument("--min-bigs", type=int, default=1)
        p.add_argument("--max-zero-minute", type=int, default=0, help="max players with 0 season minutes")
        p.add_argument("--must", action="append", help="player name substring that must be included (repeatable)")
        p.add_argument("--exclude", action="append", help="player name substring to exclude (repeatable)")
        p.add_argument("--json", action="store_true")
        p.add_argument("--site", metavar="DIR", help="also write DIR/index.html for the static site (uses ../assets/shell.css + shell.js)")

    add_gen(sub.add_parser("generate", help="create data/players.csv and data/lineups.csv"))
    sub.add_parser("train", help="train Ridge + XGBoost + HistGradientBoosting, write models/")
    add_rec(sub.add_parser("recommend", help="best fives for one team"), need_team=True)
    d = sub.add_parser("demo", help="generate -> train -> recommend all -> exports -> report")
    add_gen(d)
    add_rec(d, need_team=False)

    args = ap.parse_args(argv)
    try:
        {"generate": cmd_generate, "train": cmd_train, "recommend": cmd_recommend, "demo": cmd_demo}[args.cmd](args)
    except ValueError as exc:
        print(f"error: {exc}")
        sys.exit(2)


if __name__ == "__main__":
    main()
