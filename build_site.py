#!/usr/bin/env python3
"""Build the static GitHub Pages site in ./site.

What it does (re-runnable, idempotent):
  1. For each project, runs ``python cli.py demo --site site/<app>`` with cwd = the
     project folder and NO_OPEN=1, so each app writes ``site/<app>/index.html``.
     A failing demo is reported per app and does NOT fail the build.
  2. Injects the PWA head tags from ``site/_head_snippet.html`` into each
     ``site/<app>/index.html`` <head> if the page has no ``rel="manifest"`` yet.
     The snippet uses ``../`` paths, so it is valid as-is for any page one level
     below the site root. App authors may paste it into their templates instead.
  3. Reads headline KPIs from the projects' output JSON
     (01 results.json, 02 recommendations.json, 03 metrics.json; missing -> "—")
     and renders ``site/index.template.html`` -> ``site/index.html`` by replacing
     ``{{TOKEN}}`` placeholders.
  4. Regenerates icons if missing, deletes ``site/**/_preview_*.png``, writes
     ``site/.nojekyll`` and prints a summary.

Exit code is non-zero only if the hub itself cannot be built.

Usage:  python build_site.py [--skip-demos]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
SITE = os.path.join(ROOT, "site")
DASH = "—"

APPS = [
    # (site folder, project folder, KPI json relative to project)
    ("jumpshot", "01-jumpshot-analyzer", "output/results.json"),
    ("lineup", "02-lineup-intelligence", "output/recommendations.json"),
    ("predictor", "03-game-predictor", "output/metrics.json"),
]


# --------------------------------------------------------------------------- helpers
def load_json(path: str):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        print(f"  ! could not read {os.path.relpath(path, ROOT)}: {e}")
        return None


def fmt(v, nd=1, suffix=""):
    """Format a number to nd decimals; None/garbage -> em dash."""
    try:
        if v is None:
            return DASH
        return f"{float(v):.{nd}f}{suffix}"
    except (TypeError, ValueError):
        return DASH


def fmt_int(v):
    try:
        return f"{int(round(float(v))):,}"
    except (TypeError, ValueError):
        return DASH


def html_escape(s) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def short_ref(name) -> str:
    """'Reference C — quick release' -> 'Ref C'"""
    if not name:
        return DASH
    m = re.match(r"Reference\s+(\w+)", str(name))
    return f"Ref {m.group(1)}" if m else str(name)


# --------------------------------------------------------------------------- KPIs
def kpis_jumpshot(d) -> dict:
    out = dict.fromkeys(["JS_SCORE", "JS_CLOSEST", "JS_RELEASE", "JS_FRAMES", "JS_FPS"], DASH)
    if not isinstance(d, dict):
        return out
    user = d.get("user") or {}
    summ = user.get("summary") or {}
    out["JS_SCORE"] = fmt(d.get("headline_score"), 1)
    out["JS_CLOSEST"] = short_ref(d.get("closest"))
    rel = summ.get("release_time_from_onset_s", summ.get("release_time_s"))
    out["JS_RELEASE"] = fmt(rel, 2, " s")
    out["JS_FRAMES"] = fmt_int(user.get("n_frames"))
    out["JS_FPS"] = fmt_int(user.get("fps"))
    return out


def kpis_lineup(d) -> dict:
    keys = ["LU_TOP_FIT", "LU_TOP_TEAM", "LU_COMBOS", "LU_TEAMS", "LU_SPEARMAN", "LU_MODEL"]
    out = dict.fromkeys(keys, DASH)
    if not isinstance(d, dict):
        return out
    teams = d.get("teams") or {}
    best_fit, best_team, combos = None, None, 0
    for name, t in teams.items():
        combos += int(t.get("combos_evaluated") or 0)
        recs = t.get("recommendations") or []
        if recs:
            fs = recs[0].get("fit_score")
            if fs is not None and (best_fit is None or fs > best_fit):
                best_fit, best_team = fs, name
    out["LU_TOP_FIT"] = fmt(best_fit, 1)
    out["LU_TOP_TEAM"] = html_escape(best_team) if best_team else DASH
    out["LU_COMBOS"] = fmt_int(combos) if combos else DASH
    out["LU_TEAMS"] = fmt_int(len(teams)) if teams else DASH
    metrics = d.get("metrics") or {}
    # Prefer the model that won the most folds; skip the mean baseline.
    cands = {k: v for k, v in metrics.items() if k != "mean_baseline" and isinstance(v, dict)}
    if cands:
        best = max(cands, key=lambda k: (cands[k].get("folds_won") or 0, cands[k].get("spearman_pooled") or 0))
        out["LU_SPEARMAN"] = fmt(cands[best].get("spearman_pooled"), 2)
        out["LU_MODEL"] = {"xgboost": "XGBoost", "hgb": "HistGB", "ridge": "Ridge"}.get(best, best)
    return out


def kpis_predictor(d) -> dict:
    keys = ["GP_ACC", "GP_N_TEST", "GP_LOGLOSS", "GP_BASE_LOGLOSS", "GP_ECE", "GP_CAL"]
    out = dict.fromkeys(keys, DASH)
    if not isinstance(d, dict):
        return out
    models = d.get("models") or {}
    main = models.get("xgb") or models.get("xgb_raw") or models.get("elo_logit") or {}
    base = models.get("home_baseline") or {}
    out["GP_ACC"] = fmt((main.get("accuracy") or 0) * 100, 1, "%") if main.get("accuracy") is not None else DASH
    out["GP_N_TEST"] = fmt_int((d.get("data") or {}).get("n_test", main.get("n")))
    out["GP_LOGLOSS"] = fmt(main.get("log_loss"), 3)
    out["GP_BASE_LOGLOSS"] = fmt(base.get("log_loss"), 3)
    cal = d.get("calibration") or {}
    out["GP_ECE"] = fmt(cal.get("ece_xgb"), 3)
    out["GP_CAL"] = html_escape(cal.get("chosen") or DASH)
    return out


# --------------------------------------------------------------------------- steps
def run_demo(app: str, project: str) -> bool:
    proj = os.path.join(ROOT, project)
    cli = os.path.join(proj, "cli.py")
    if not os.path.exists(cli):
        print(f"[{app}] skip: {project}/cli.py not found")
        return False
    out_dir = os.path.join(SITE, app)
    os.makedirs(out_dir, exist_ok=True)
    env = dict(os.environ, NO_OPEN="1", PYTHONUNBUFFERED="1")
    cmd = [sys.executable, "cli.py", "demo", "--site", os.path.relpath(out_dir, proj)]
    print(f"[{app}] $ (cd {project} && {' '.join(cmd[1:])})")
    try:
        r = subprocess.run(cmd, cwd=proj, env=env, capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        print(f"[{app}] FAILED: demo timed out after 900 s")
        return False
    if r.returncode != 0:
        tail = (r.stderr or r.stdout or "").strip().splitlines()[-6:]
        print(f"[{app}] FAILED (exit {r.returncode}). Last lines:")
        for ln in tail:
            print("     " + ln)
        return False
    page = os.path.join(out_dir, *PAGE_OF.get(app, "index.html").split("/"))
    ok = os.path.exists(page)
    print(f"[{app}] demo ok" + ("" if ok else f" but site/{app}/{PAGE_OF.get(app, 'index.html')} was not written"))
    return ok


# Page each app's demo generates. Jumpshot's site/jumpshot/index.html is the hand-built
# on-device upload app (not generated), so its demo writes the static report to sample/.
PAGE_OF = {"jumpshot": "sample/index.html"}


def inject_head(app: str, snippet: str, rel: str = "index.html") -> str:
    page = os.path.join(SITE, app, *rel.split("/"))
    extra = rel.count("/")
    if extra:
        snippet = snippet.replace('"../', '"' + "../" * (extra + 1))
    if not os.path.exists(page):
        return "missing"
    with open(page, encoding="utf-8") as f:
        html = f.read()
    if 'rel="manifest"' in html or "rel='manifest'" in html:
        return "present"
    # Replace an existing viewport tag (snippet has its own, with viewport-fit).
    html2 = re.sub(r"<meta\s+name=[\"']viewport[\"'][^>]*>\s*", "", html, count=1, flags=re.I)
    m = re.search(r"<head[^>]*>", html2, flags=re.I)
    if not m:
        return "no <head>"
    html2 = html2[: m.end()] + "\n" + snippet.strip() + "\n" + html2[m.end():]
    with open(page, "w", encoding="utf-8") as f:
        f.write(html2)
    return "injected"


def build_hub(tokens: dict) -> None:
    tpl_path = os.path.join(SITE, "index.template.html")
    with open(tpl_path, encoding="utf-8") as f:
        tpl = f.read()
    tokens = dict(tokens, BUILT=dt.date.today().isoformat())
    html = re.sub(r"\{\{(\w+)\}\}", lambda m: str(tokens.get(m.group(1), DASH)), tpl)
    leftover = re.findall(r"\{\{\w+\}\}", html)
    if leftover:
        raise RuntimeError(f"unresolved placeholders: {leftover}")
    with open(os.path.join(SITE, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)


def ensure_icons() -> None:
    need = ["icon-192.png", "icon-512.png", "icon-maskable-192.png", "icon-maskable-512.png", "apple-touch-icon.png"]
    icons = os.path.join(SITE, "icons")
    if all(os.path.exists(os.path.join(icons, n)) for n in need):
        return
    gen = os.path.join(icons, "make_icons.py")
    if os.path.exists(gen):
        subprocess.run([sys.executable, gen], check=False)


def clean_previews() -> int:
    n = 0
    for dirpath, _, files in os.walk(SITE):
        for fn in files:
            if fn.startswith(("_preview_", "_ui_", "_hub_")) and fn.endswith(".png"):
                os.remove(os.path.join(dirpath, fn))
                n += 1
    return n


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-demos", action="store_true", help="do not run the per-app demos; only rebuild the hub")
    args = ap.parse_args()

    os.makedirs(SITE, exist_ok=True)
    summary = {}

    # 1. demos
    for app, project, _ in APPS:
        summary[app] = {"demo": "skipped" if args.skip_demos else ("ok" if run_demo(app, project) else "FAILED")}

    # 2. head snippet
    snip_path = os.path.join(SITE, "_head_snippet.html")
    snippet = open(snip_path, encoding="utf-8").read() if os.path.exists(snip_path) else ""
    for app, _, _ in APPS:
        summary[app]["head"] = inject_head(app, snippet, PAGE_OF.get(app, "index.html")) if snippet else "no snippet"

    # 3. KPIs + hub
    tokens = {}
    tokens.update(kpis_jumpshot(load_json(os.path.join(ROOT, APPS[0][1], APPS[0][2]))))
    tokens.update(kpis_lineup(load_json(os.path.join(ROOT, APPS[1][1], APPS[1][2]))))
    tokens.update(kpis_predictor(load_json(os.path.join(ROOT, APPS[2][1], APPS[2][2]))))
    try:
        ensure_icons()
        build_hub(tokens)
    except Exception as e:  # the only fatal path
        print(f"HUB FAILED: {e}")
        return 1

    # 4. housekeeping
    removed = clean_previews()
    open(os.path.join(SITE, ".nojekyll"), "a").close()

    print("\n== build summary ==")
    for app, s in summary.items():
        page = os.path.exists(os.path.join(SITE, app, "index.html"))
        print(f"  {app:10s} demo={s['demo']:8s} head={s['head']:9s} page={'yes' if page else 'NO'}")
    print(f"  hub        site/index.html written; KPIs: " +
          ", ".join(f"{k}={v}" for k, v in tokens.items() if k in ("JS_SCORE", "LU_TOP_FIT", "GP_ACC")))
    print(f"  removed {removed} _preview_*.png; .nojekyll present")
    failed = [a for a, s in summary.items() if s["demo"] == "FAILED"]
    if failed:
        print(f"  note: demos failed for {', '.join(failed)} (hub still built; exit 0)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
