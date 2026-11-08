"""
NBA Game Predictor — command line.

    python cli.py generate [--source espn|sample] [--seed 42] [--seasons 3] [--force]
                                      # espn (default): real NBA games from ESPN's public API, incremental cache
                                      # sample: the seeded synthetic league (offline fallback)
                                      # --real: legacy nba_api path (stats.nba.com), kept for reference
    python cli.py train
    python cli.py predict [--date YYYY-MM-DD]
    python cli.py report [--site DIR]
    python cli.py simulate [--date YYYY-MM-DD] [--sims 2000] [--team CODE]
    python cli.py rankings [--date YYYY-MM-DD]
    python cli.py team CODE
    python cli.py demo [--site DIR]   # generate -> train -> predict -> simulate -> report
                                      # --site also writes DIR/index.html for the shared static site

Learning project. Real data from ESPN's unofficial public API (sample league as fallback).
Not betting advice. Nothing here is guaranteed.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data"))


BANNER = "=" * 78


def _write_sample_meta(n: int) -> None:
    import json
    from datetime import datetime, timezone
    (ROOT / "data" / "meta.json").write_text(json.dumps({
        "source": "sample", "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "n_games": n}, indent=2))


def cmd_generate(args) -> str:
    """Returns the data source actually used: 'espn' or 'sample'."""
    source = "sample"
    if getattr(args, "real", False):  # legacy nba_api path
        from nba_api_loader import load_or_fallback
        p = load_or_fallback(args.real_seasons, ROOT / "data" / "games.csv", seed=args.seed)
        source = "nba_api"
    elif args.source == "espn":
        from espn_loader import load_or_fallback
        p, source = load_or_fallback(seed=args.seed, force=getattr(args, "force", False))
        if source == "sample":
            print(BANNER + "\n  ESPN API not reachable -> running on the SAMPLE league (synthetic data)\n" + BANNER)
    else:
        from sample_generator import generate
        p = generate(ROOT / "data" / "games.csv", seed=args.seed, seasons=args.seasons)
    if source != "espn":
        import pandas as pd
        _write_sample_meta(len(pd.read_csv(p)))
        # sample league has no cache of upcoming games
        sched = ROOT / "data" / "cache" / "espn_scheduled.csv"
        if sched.exists():
            sched.unlink()
    import pandas as pd
    df = pd.read_csv(p)
    played = df.dropna(subset=["home_pts"])
    print(f"[generate] {p} -> {len(df)} games ({len(played)} played), seasons {sorted(df.season.unique())}, "
          f"home win rate {(played.home_pts > played.away_pts).mean():.3f}, source={source}")
    return source


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
    print("exported output/rankings.csv  -  not betting advice")


def cmd_team(args) -> None:
    from analytics import build_all, print_team
    b = build_all()
    code = args.code.upper()
    if code not in b["team_pages"]:
        raise SystemExit(f"unknown team {code}; known: {', '.join(sorted(b['team_pages']))}")
    print_team(code, b["team_pages"][code], next((r for r in b["rankings"] if r["team"] == code), None))


def cmd_demo(args) -> None:
    t0 = time.time()
    if args.source == "espn":
        print(BANNER + "\n  Data: ESPN public scoreboard API (unofficial). Refreshing the incremental cache ...\n" + BANNER)
        source = cmd_generate(argparse.Namespace(source="espn", seed=args.seed, seasons=3, force=args.regenerate))
    elif not (ROOT / "data" / "games.csv").exists() or args.regenerate:
        source = cmd_generate(argparse.Namespace(source="sample", seed=args.seed, seasons=3))
    else:
        print("[generate] data/games.csv exists, reusing (use --regenerate to rebuild)")
        source = "sample"
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
    tag = "REAL NBA DATA via ESPN" if source == "espn" else "SAMPLE DATA"
    print(f"\n[demo] done in {time.time() - t0:.1f}s. Open output/report.html. "
          f"{tag} — not betting advice. Nothing here is guaranteed.")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="cli.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="write data/games.csv + data/teams.csv (ESPN by default, sample league as fallback)")
    g.add_argument("--source", choices=["espn", "sample"], default="espn", help="espn (default) or sample")
    g.add_argument("--seed", type=int, default=42, help="sample league seed")
    g.add_argument("--seasons", type=int, default=3, help="sample league seasons")
    g.add_argument("--force", action="store_true", help="espn: ignore the cache and backfill from 2023-10-24")
    g.add_argument("--real", action="store_true", help="LEGACY: stats.nba.com via nba_api (often blocked), falls back to sample")
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
    d.add_argument("--source", choices=["espn", "sample"], default="espn", help="espn (default; falls back to sample if offline) or sample")
    d.add_argument("--regenerate", action="store_true", help="espn: ignore the cache; sample: rebuild games.csv")
    d.add_argument("--site", default=None, metavar="DIR", help="also write DIR/index.html for the shared site (e.g. ../site/predictor)")
    d.set_defaults(func=cmd_demo)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
