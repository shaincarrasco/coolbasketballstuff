"""
NBA Game Predictor — command line.

    python cli.py generate [--seed 42] [--seasons 3] [--real --real-seasons 2022-23 2023-24 2024-25]
    python cli.py train
    python cli.py predict [--date YYYY-MM-DD]
    python cli.py report [--site DIR]
    python cli.py simulate [--date YYYY-MM-DD] [--sims 2000] [--team CODE]
    python cli.py rankings [--date YYYY-MM-DD]
    python cli.py team CODE
    python cli.py demo [--site DIR]   # generate -> train -> predict -> simulate -> report
                                      # --site also writes DIR/index.html for the shared static site

Learning project on simulated data. Not betting advice. Nothing here is guaranteed.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data"))


def cmd_generate(args) -> None:
    if args.real:
        from nba_api_loader import load_or_fallback
        p = load_or_fallback(args.real_seasons, ROOT / "data" / "games.csv", seed=args.seed)
    else:
        from sample_generator import generate
        p = generate(ROOT / "data" / "games.csv", seed=args.seed, seasons=args.seasons)
    import pandas as pd
    df = pd.read_csv(p)
    print(f"[generate] {p} -> {len(df)} games, seasons {sorted(df.season.unique())}, "
          f"home win rate {(df.home_pts > df.away_pts).mean():.3f}")


def cmd_train(args) -> None:
    from model import train
    train(ROOT / "data" / "games.csv")


def cmd_predict(args) -> None:
    from predict import predict_date, print_slate
    print_slate(predict_date(args.date))


def cmd_report(args) -> None:
    from report import render
    print(f"[report] {render(site_dir=args.site)}")
    if args.site:
        print(f"[report] site page -> {Path(args.site) / 'index.html'}")


def cmd_simulate(args) -> None:
    from simulate import print_sim, simulate
    print_sim(simulate(args.date, n_sims=args.sims, seed=args.seed, focus=args.team))


def cmd_rankings(args) -> None:
    from analytics import build_all, print_rankings
    print_rankings(build_all(as_of=args.date)["rankings"], args.date)
    print("exported output/rankings.csv  -  SAMPLE DATA, not betting advice")


def cmd_team(args) -> None:
    from analytics import build_all, print_team
    b = build_all()
    code = args.code.upper()
    if code not in b["team_pages"]:
        raise SystemExit(f"unknown team {code}; known: {', '.join(sorted(b['team_pages']))}")
    print_team(code, b["team_pages"][code], next((r for r in b["rankings"] if r["team"] == code), None))


def cmd_demo(args) -> None:
    t0 = time.time()
    if not (ROOT / "data" / "games.csv").exists() or args.regenerate:
        cmd_generate(argparse.Namespace(real=False, seed=args.seed, seasons=3, real_seasons=[]))
    else:
        print("[generate] data/games.csv exists, reusing (use --regenerate to rebuild)")
    cmd_train(args)
    cmd_predict(argparse.Namespace(date=args.date))
    from analytics import build_all
    from report import render
    from simulate import print_sim, simulate
    bundle = build_all()
    sim = simulate(args.sim_date, n_sims=args.sims, feats=bundle["feats"], teams=bundle["teams"])
    print_sim(sim)
    print(f"[report] {render(bundle=bundle, sim=sim, site_dir=args.site)}")
    if args.site:
        print(f"[report] site page -> {Path(args.site) / 'index.html'}")
    print(f"\n[demo] done in {time.time() - t0:.1f}s. Open output/report.html. "
          "SAMPLE DATA — not betting advice. Nothing here is guaranteed.")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="cli.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="write data/games.csv (simulated by default)")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--seasons", type=int, default=3)
    g.add_argument("--real", action="store_true", help="try stats.nba.com via nba_api, fall back to sample data")
    g.add_argument("--real-seasons", nargs="+", default=["2022-23", "2023-24", "2024-25"])
    g.set_defaults(func=cmd_generate)

    t = sub.add_parser("train", help="train XGBoost + baselines, write models/ and metrics.json")
    t.set_defaults(func=cmd_train)

    p = sub.add_parser("predict", help="win probabilities for one date's slate")
    p.add_argument("--date", default=None, help="YYYY-MM-DD (default: last full slate of the final season)")
    p.set_defaults(func=cmd_predict)

    r = sub.add_parser("report", help="render output/report.html from predictions + metrics")
    r.add_argument("--site", default=None, metavar="DIR", help="also write DIR/index.html referencing ../assets/shell.css|js")
    r.set_defaults(func=cmd_report)

    sm = sub.add_parser("simulate", help="Monte Carlo rest-of-season from a date (expected wins, playoff odds)")
    sm.add_argument("--date", default=None, help="YYYY-MM-DD inside the final season (default ~55%% through)")
    sm.add_argument("--sims", type=int, default=2000)
    sm.add_argument("--seed", type=int, default=42)
    sm.add_argument("--team", default=None, help="team code for the win-total histogram")
    sm.set_defaults(func=cmd_simulate)

    rk = sub.add_parser("rankings", help="Elo power rankings with 30-day change, L10 form, off/def proxies")
    rk.add_argument("--date", default=None, help="as-of date (default: end of data)")
    rk.set_defaults(func=cmd_rankings)

    tp = sub.add_parser("team", help="team drill-down: schedule, splits, rest/B2B record, upsets")
    tp.add_argument("code")
    tp.set_defaults(func=cmd_team)

    d = sub.add_parser("demo", help="generate -> train -> predict -> simulate -> report")
    d.add_argument("--seed", type=int, default=42)
    d.add_argument("--date", default=None)
    d.add_argument("--sim-date", default=None)
    d.add_argument("--sims", type=int, default=2000)
    d.add_argument("--regenerate", action="store_true", help="rebuild games.csv even if it exists")
    d.add_argument("--site", default=None, metavar="DIR", help="also write DIR/index.html for the shared site (e.g. ../site/predictor)")
    d.set_defaults(func=cmd_demo)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
