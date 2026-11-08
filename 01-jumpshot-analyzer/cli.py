#!/usr/bin/env python
"""
cli.py — AI Jumpshot Analyzer command line.

    python cli.py build-references demo/ref_*.mp4 [--references references/] [--names "A" "B" ...]
    python cli.py analyze demo/user_shot.mp4 --references references/ [--output output/]
    python cli.py batch my_clips/ --references references/   # every mp4 -> output/batch.csv + report section
    python cli.py analyze --webcam          # how to record a clip (no camera is opened)
    python cli.py demo                      # generate synthetic clips, build library, analyze
    python cli.py demo --site ../site/jumpshot   # also emit sample/index.html (+ video) and app/*.json
    python cli.py export-web --out ../site/jumpshot/app   # only the web-app JSON (+ parity fixtures)

Every analyze/batch run appends to output/history.jsonl (progress tracking) and writes
output/results.csv (per-feature values) next to results.json.

Pose backend flags (all commands):
    --pose-model PATH            MediaPipe pose_landmarker .task file (tasks API)
    --landmarks-from-synthetic   allow using <clip>.landmarks.npz sidecars written by demo_data.py
                                 when no MediaPipe pose model can run in this environment
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from pose import extract_landmarks, mediapipe_backend, find_pose_model  # noqa: E402
from features import shot_from_pose, save_shot, Shot  # noqa: E402
from model import FeatureStats, train_encoder, save_model, load_model  # noqa: E402
from compare import load_reference_library, compare_shot, results_for_json, FEEDBACK_SPEC  # noqa: E402
from drills import coaching_plan  # noqa: E402

ENCODER_FILE = "_encoder.pt"
HISTORY_FILE = "history.jsonl"
BATCH_FILE = "batch.csv"

WEBCAM_HOWTO = """\
[webcam] This build does not open a camera. Record a clip, then run `cli.py analyze <clip.mp4>`.

  How to record a usable jumpshot clip
  ------------------------------------
  * Phone or webcam on a tripod / chair, lens at roughly hip-to-chest height.
  * Side-on or 3/4 view, 3-5 m away, whole body visible from feet to fingertips
    through the entire motion (stance -> dip -> release -> follow-through).
  * Landscape or portrait both work; >= 30 fps (60 fps if available), steady light,
    plain background, one shooter in frame, no one between camera and shooter.
  * One shot per clip, trimmed to ~2 s around the shot (0.5 s before the dip).
  * Shooting hand is auto-detected; pass --side left|right to force it.

  Quick capture commands (optional):
    ffmpeg -f v4l2 -framerate 30 -video_size 1280x720 -i /dev/video0 -t 3 shot.mp4      # Linux
    ffmpeg -f avfoundation -framerate 30 -i "0" -t 3 shot.mp4                             # macOS
    ffmpeg -f dshow -framerate 30 -i video="Integrated Camera" -t 3 shot.mp4              # Windows

  Then: python cli.py analyze shot.mp4 --references references/ --pose-model models/pose_landmarker_lite.task
"""


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_") or "shot"


def _pretty_name(video_path: str) -> str:
    base = os.path.splitext(os.path.basename(video_path))[0]
    sidecar_meta = None
    try:
        import numpy as np
        from pose import sidecar_path
        sp = sidecar_path(video_path)
        if os.path.isfile(sp):
            d = np.load(sp, allow_pickle=False)
            sidecar_meta = json.loads(str(d["meta"])) if "meta" in d else None
    except Exception:
        sidecar_meta = None
    if sidecar_meta and sidecar_meta.get("name"):
        return sidecar_meta["name"]
    return base.replace("_", " ").title()


def _extract(video: str, args, name: str) -> Shot:
    seq = extract_landmarks(video, model_path=args.pose_model, allow_synthetic=args.landmarks_from_synthetic,
                            prefer_synthetic=getattr(args, "prefer_synthetic", False))
    shot = shot_from_pose(seq, name, side=None if args.side == "auto" else args.side)
    ev = shot.events
    print(f"[features] {name}: side={shot.side} frames={shot.per_frame.shape[0]} gap-filled={shot.missing_frames} "
          f"onset=f{ev.onset} set=f{ev.set_point} release=f{ev.release} ft_end=f{ev.follow_through_end} "
          f"hold={ev.hold_s:.2f}s release_height={shot.summary['release_height']:+.2f} torso")
    return shot


# ------------------------------------------------------- history / export

def _history_entry(results: dict, label: str, timestamp: str | None = None) -> dict:
    best = results["references"][0]
    deltas = {fb["key"]: round(float(fb["delta"]), 4) for fb in best["feedback"]}
    sev = {fb["key"]: fb["severity"] for fb in best["feedback"]}
    return dict(timestamp=timestamp or dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
                clip=os.path.basename(results["user"].get("video", "") or results["user"]["name"]),
                name=results["user"]["name"], label=label, score=float(results["headline_score"]),
                closest=results["closest"], dtw_score=float(best["dtw_score"]),
                learned_score=best["learned_score"], deltas=deltas, severities=sev,
                confidence=results["user"].get("quality", {}).get("confidence", "ok"))


def append_history(out_dir: str, entry: dict) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, HISTORY_FILE)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=float) + "\n")
    return path


def load_history(out_dir: str) -> list[dict]:
    path = os.path.join(out_dir, HISTORY_FILE)
    if not os.path.isfile(path):
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    out.sort(key=lambda e: e.get("timestamp", ""))
    return out


def write_results_csv(out_dir: str, results: dict, user: Shot) -> tuple[str, str]:
    """results.csv (summary metrics: user / closest ref / delta / severity) and
    features_per_frame.csv (raw per-frame features of the user clip)."""
    best = results["references"][0]
    p1 = os.path.join(out_dir, "results.csv")
    with open(p1, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "label", "unit", "user", "reference", "delta", "severity", "reference_name"])
        for fb in best["feedback"]:
            w.writerow([fb["key"], fb["label"], fb["unit"], f"{fb['user']:.4f}", f"{fb['reference']:.4f}",
                        f"{fb['delta']:.4f}", fb["severity"], best["name"]])
        # extra summary keys not in the feedback spec
        covered = {spec[0] for spec in FEEDBACK_SPEC}
        for k, v in results["user"]["summary"].items():
            if k not in covered:
                w.writerow([k, k, "", f"{float(v):.4f}", f"{float(best['summary'].get(k, float('nan'))):.4f}", "", "", best["name"]])
    p2 = os.path.join(out_dir, "features_per_frame.csv")
    from features import FEATURE_NAMES
    ev = user.events
    marks = {ev.onset: "onset", ev.set_point: "set_point", ev.release: "release",
             ev.follow_through: "follow_through", ev.follow_through_end: "follow_through_end"}
    with open(p2, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "t_s", "event"] + FEATURE_NAMES)
        for i, row in enumerate(user.per_frame):
            w.writerow([i, f"{i / user.fps:.3f}", marks.get(i, "")] + [f"{float(v):.4f}" for v in row])
    return p1, p2


def _load_or_train(args, refs):
    enc_path = os.path.join(args.references, ENCODER_FILE)
    if os.path.isfile(enc_path) and not getattr(args, "no_model", False):
        return load_model(enc_path)
    stats = FeatureStats.fit([s.seq for s in refs])
    if getattr(args, "no_model", False):
        print("[model] --no-model: DTW only")
        return None, stats, {}
    print("[model] no saved encoder; training now")
    model, info = train_encoder(refs, stats, steps=args.steps, seed=args.seed)
    save_model(enc_path, model, stats, info)
    return model, stats, info


def batch_analyze(videos: list[str], refs, model, stats, args, out_dir: str, label: str = "batch",
                  timestamps: list[str] | None = None, names: list[str] | None = None) -> list[dict]:
    """Analyze each clip (features + scoring only, no video overlay), append to history,
    and return comparison rows. Clips that fail are reported as rows with an error."""
    rows = []
    for k, v in enumerate(videos):
        name = (names[k] if names else None) or _pretty_name(v)
        try:
            shot = _extract(v, args, name)
            res = compare_shot(shot, refs, model, stats)
        except Exception as ex:  # keep going through the folder
            print(f"[batch] {os.path.basename(v)}: FAILED ({ex})")
            rows.append(dict(clip=os.path.basename(v), name=name, error=str(ex)))
            continue
        entry = _history_entry(res, label, timestamps[k] if timestamps else None)
        append_history(out_dir, entry)
        best = res["references"][0]
        n_major = sum(fb["severity"] == "major" for fb in best["feedback"])
        n_minor = sum(fb["severity"] == "minor" for fb in best["feedback"])
        top = [fb["label"] for fb in sorted(best["feedback"], key=lambda f: -abs(f["delta"]) /
                                           {"deg": 5, "s": 0.06, "torso": 0.06}[f["unit"]]) if fb["severity"] != "ok"][:2]
        rows.append(dict(clip=os.path.basename(v), name=name, timestamp=entry["timestamp"], score=res["headline_score"],
                         closest=res["closest"], dtw_score=best["dtw_score"], learned_score=best["learned_score"],
                         side=shot.side, frames=int(shot.per_frame.shape[0]), confidence=shot.quality.get("confidence", "ok"),
                         n_major=n_major, n_minor=n_minor, top_findings="; ".join(top),
                         **{f"d_{fb['key']}": round(fb["delta"], 4) for fb in best["feedback"]}))
        print(f"[batch] {os.path.basename(v):<28} {res['headline_score']:5.1f}  closest={res['closest']}  "
              f"major={n_major} minor={n_minor} conf={shot.quality.get('confidence', 'ok')}")
    return rows


def write_batch_csv(out_dir: str, rows: list[dict]) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, BATCH_FILE)
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


def load_batch_csv(out_dir: str) -> list[dict]:
    path = os.path.join(out_dir, BATCH_FILE)
    if not os.path.isfile(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------- commands

def cmd_build_references(args) -> int:
    videos = []
    for pat in args.videos:
        hits = sorted(glob.glob(pat))
        videos += hits if hits else [pat]
    if not videos:
        print("no reference videos given", file=sys.stderr)
        return 2
    os.makedirs(args.references, exist_ok=True)
    names = args.names or [_pretty_name(v) for v in videos]
    if len(names) != len(videos):
        print("--names must match the number of videos", file=sys.stderr)
        return 2
    shots = []
    for v, n in zip(videos, names):
        shot = _extract(v, args, n)
        out = os.path.join(args.references, _slug(n) + ".npz")
        save_shot(out, shot)
        print(f"[library] saved {out}")
        shots.append(shot)
    if not args.no_train:
        stats = FeatureStats.fit([s.seq for s in shots])
        model, info = train_encoder(shots, stats, steps=args.steps, seed=args.seed)
        save_model(os.path.join(args.references, ENCODER_FILE), model, stats, info)
        print(f"[library] encoder saved to {os.path.join(args.references, ENCODER_FILE)}")
    return 0


def cmd_analyze(args) -> int:
    t0 = time.time()
    if getattr(args, "webcam", False):
        print(WEBCAM_HOWTO)
        return 0
    if not args.video:
        print("analyze: a video path is required (or --webcam for recording instructions)", file=sys.stderr)
        return 2
    refs = load_reference_library(args.references)
    print(f"[library] {len(refs)} references: " + ", ".join(r.name for r in refs))
    model, stats, info = _load_or_train(args, refs)
    user = _extract(args.video, args, args.name or _pretty_name(args.video))
    for fl in user.quality.get("flags", []):
        print(f"   [qual {fl['severity']:5}] {fl['message']}")
    results = compare_shot(user, refs, model, stats)
    best_name = results["closest"]
    best = next(r for r in refs if r.name == best_name)
    print(f"[score] mechanics {results['headline_score']:.1f}/100 ({results['score_kind']})  "
          f"style match {results['style_match']:.1f} (closest: {best_name})")
    for m in results["mechanics"]:
        v = "—" if m["value"] is None else f"{m['value']:.2f}"
        print(f"   [{m['status']:10}] {m['label']:<38} {v:>8} {m['unit']:<5} band {m['band']}  {m['points']:.1f}/{m['maxPoints']}")
    for r in results["references"]:
        print(f"           {r['name']:<32} {r['score']:5.1f}  (dtw {r['dtw_score']:5.1f}, learned {r['learned_score']})")
    for fb in results["references"][0]["feedback"]:
        if fb["severity"] != "ok":
            print(f"   [{fb['severity']:5}] {fb['message']}")

    os.makedirs(args.output, exist_ok=True)
    plan = coaching_plan(results["feedback"])
    results["coaching_plan"] = plan
    entry = _history_entry(results, getattr(args, "label", None) or "analyze")
    append_history(args.output, entry)
    history = load_history(args.output)
    batch_rows = load_batch_csv(args.output)

    from overlay import render_side_by_side, key_frames
    from report import write_report
    video_out = os.path.join(args.output, "side_by_side.mp4")
    render_side_by_side(args.video, user, best, results, video_out)
    frames = key_frames(args.video, user, best)
    site_dir = getattr(args, "site", None)
    report_path = write_report(results, user, best, frames, args.output, model_info=info, refs=refs,
                               history=history, batch_rows=batch_rows, plan=plan, site_dir=site_dir)
    print(f"[report] wrote {report_path}" + (f" and {os.path.join(site_dir, 'sample', 'index.html')}" if site_dir else ""))
    if site_dir:
        import export_web
        export_web.export(os.path.join(site_dir, "app"), references=args.references)
    csv_summary, csv_frames = write_results_csv(args.output, results, user)
    results_json = results_for_json(results)
    results_json["model"] = {k: v for k, v in (info or {}).items() if k != "loss_history"}
    results_json["pose_backend"] = dict(mediapipe=mediapipe_backend(), model_file=find_pose_model(args.pose_model),
                                        landmarks_source=user.source)
    results_json["outputs"] = dict(report=report_path, video=video_out, results_csv=csv_summary,
                                   features_csv=csv_frames, history=os.path.join(args.output, HISTORY_FILE))
    with open(os.path.join(args.output, "results.json"), "w") as f:
        json.dump(results_json, f, indent=2, default=float)
    print(f"[done] {time.time() - t0:.1f}s -> {args.output}/report.html, side_by_side.mp4, results.json, "
          f"results.csv, history.jsonl ({len(history)} sessions)")
    return 0


def cmd_batch(args) -> int:
    t0 = time.time()
    folder = args.folder
    videos = sorted(p for p in glob.glob(os.path.join(folder, "*")) if p.lower().endswith((".mp4", ".mov", ".avi", ".mkv")))
    if not videos:
        print(f"batch: no video files in {folder}", file=sys.stderr)
        return 2
    refs = load_reference_library(args.references)
    model, stats, info = _load_or_train(args, refs)
    rows = batch_analyze(videos, refs, model, stats, args, args.output, label=args.label or "batch")
    path = write_batch_csv(args.output, rows)
    ok = [r for r in rows if "score" in r]
    print(f"[batch] {len(ok)}/{len(rows)} clips scored -> {path}")
    if ok and not args.no_report:
        best_row = max(ok, key=lambda r: r["score"])
        pick = next(v for v in videos if os.path.basename(v) == best_row["clip"])
        print(f"[batch] rendering full report for the top clip: {best_row['clip']}")
        args.video, args.name, args.label = pick, best_row["name"], "batch-report"
        return cmd_analyze(args)
    print(f"[done] {time.time() - t0:.1f}s")
    return 0


def cmd_export_web(args) -> int:
    import export_web
    export_web.export(args.out, references=args.references, fixtures=not args.no_fixtures)
    return 0


def cmd_demo(args) -> int:
    t0 = time.time()
    import demo_data
    demo_dir = os.path.join(HERE, "demo")
    manifest = demo_data.generate(demo_dir)
    # The demo clips are drawn cartoons: their exact joint positions (sidecars) are the ground truth,
    # and MediaPipe drops ~40 % of frames on the drawn figure. So the demo uses the sidecars unless
    # --detect-synthetic asks for the real detector (when it can run).
    backend = mediapipe_backend()
    can_run_mp = backend == "solutions" or (backend == "tasks" and find_pose_model(args.pose_model) is not None)
    args.landmarks_from_synthetic = True
    if getattr(args, "detect_synthetic", False) and can_run_mp:
        print("[demo] --detect-synthetic: running MediaPipe on the synthetic clips")
    else:
        args.prefer_synthetic = True
        print("[demo] using the synthetic clips' exact landmark sidecars"
              + ("" if can_run_mp else " (MediaPipe pose model not runnable here)")
              + "; pass --detect-synthetic to run MediaPipe on the drawn clips instead.")
    args.videos = [r["video"] for r in manifest["references"]]
    args.names = [r["name"] for r in manifest["references"]]
    rc = cmd_build_references(args)
    if rc:
        return rc
    # earlier practice sessions -> history.jsonl (back-dated, labelled demo) + batch.csv
    sessions = demo_data.generate_sessions(os.path.join(demo_dir, "sessions"))
    hist_path = os.path.join(args.output, HISTORY_FILE)
    if os.path.isfile(hist_path):
        os.remove(hist_path)   # demo always starts a fresh, reproducible history
    refs = load_reference_library(args.references)
    model, stats, _ = _load_or_train(args, refs)
    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    stamps = [(now - dt.timedelta(days=d)).isoformat() for d in (28, 21, 14, 7)]
    rows = batch_analyze([s_["video"] for s_ in sessions], refs, model, stats, args, args.output, label="demo",
                         timestamps=stamps, names=[s_["name"] for s_ in sessions])
    write_batch_csv(args.output, rows)
    args.video = manifest["user"]["video"]
    args.name = manifest["user"]["name"]
    args.label = "demo"
    rc = cmd_analyze(args)
    print(f"[demo] total {time.time() - t0:.1f}s")
    return rc


# -------------------------------------------------------------------- main

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cli.py", description="AI Jumpshot Analyzer")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--references", default=os.path.join(HERE, "references"), help="reference library dir")
    common.add_argument("--pose-model", default=None, help="MediaPipe pose_landmarker .task file")
    common.add_argument("--landmarks-from-synthetic", action="store_true",
                        help="allow <clip>.landmarks.npz sidecars (synthetic demo clips) when MediaPipe cannot run")
    common.add_argument("--side", choices=["auto", "right", "left"], default="auto", help="shooting hand")
    common.add_argument("--steps", type=int, default=250, help="encoder training steps")
    common.add_argument("--seed", type=int, default=0)
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-references", parents=[common], help="extract features from reference clips, train encoder")
    b.add_argument("videos", nargs="+", help="reference videos (globs ok)")
    b.add_argument("--names", nargs="*", default=None, help="display names, one per video")
    b.add_argument("--no-train", action="store_true", help="skip encoder training")
    b.set_defaults(func=cmd_build_references)

    a = sub.add_parser("analyze", parents=[common], help="analyze a user shot against the library")
    a.add_argument("video", nargs="?", default=None)
    a.add_argument("--output", default=os.path.join(HERE, "output"))
    a.add_argument("--name", default=None, help="label for the user shot")
    a.add_argument("--label", default=None, help="session label stored in history.jsonl")
    a.add_argument("--no-model", action="store_true", help="DTW only, skip the learned embedding")
    a.add_argument("--webcam", action="store_true", help="print how to record a clip (no camera is opened)")
    a.add_argument("--site", default=None, metavar="DIR", help="also write the static sample report to DIR/sample/index.html (+ video, linking ../../assets/shell.{css,js}) and export the web-app data (library/drills/encoder/demo JSON) to DIR/app/")
    a.set_defaults(func=cmd_analyze)

    bt = sub.add_parser("batch", parents=[common], help="analyze every video in a folder -> output/batch.csv")
    bt.add_argument("folder")
    bt.add_argument("--output", default=os.path.join(HERE, "output"))
    bt.add_argument("--label", default=None, help="session label stored in history.jsonl")
    bt.add_argument("--no-model", action="store_true")
    bt.add_argument("--no-report", action="store_true", help="only write batch.csv; skip the full report")
    bt.add_argument("--site", default=None, metavar="DIR", help="also write DIR/sample/index.html + DIR/app/*.json for the static site")
    bt.set_defaults(func=cmd_batch, name=None)

    d = sub.add_parser("demo", parents=[common], help="generate synthetic data, build library, analyze")
    d.add_argument("--output", default=os.path.join(HERE, "output"))
    d.add_argument("--no-train", action="store_true")
    d.add_argument("--no-model", action="store_true")
    d.add_argument("--site", default=None, metavar="DIR", help="also write DIR/sample/index.html (+ video) and DIR/app/*.json for the static site (never touches DIR/index.html)")
    d.add_argument("--detect-synthetic", action="store_true",
                   help="run MediaPipe on the drawn demo clips instead of using their exact landmark sidecars")
    d.set_defaults(func=cmd_demo, name=None)

    ew = sub.add_parser("export-web", parents=[common], help="export library/drills/encoder/demo JSON (+ parity fixtures) for the web app")
    ew.add_argument("--out", required=True, help="output dir, e.g. ../site/jumpshot/app")
    ew.add_argument("--no-fixtures", action="store_true", help="skip tests/fixtures/*.json")
    ew.set_defaults(func=cmd_export_web)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
