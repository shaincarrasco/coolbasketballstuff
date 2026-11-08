"""
export_web.py — export everything the in-browser engine (site/jumpshot/app/engine.mjs) needs so
that scoring a clip on a phone gives the same numbers as this Python pipeline.

    python cli.py export-web --out ../site/jumpshot/app        (also run by `demo/analyze --site DIR`)

Writes into OUT/:
    library.json   reference library: per reference name, events, summary, 60 x F time-normalised
                   features (seq60), canonical landmarks over onset -> follow-through end resampled
                   to 60 frames (the "ghost"), plus every constant the JS port needs (feature list,
                   feedback tolerances, z-score stats, DTW band / score mapping, score weights,
                   event-detector / smoothing / visibility constants, radar axes).
    drills.json    drills.DRILLS + GENERIC_MAINTENANCE (finding -> drill table).
    encoder.json   ShotEncoder weights (eval-mode state_dict as nested arrays, 6 significant
                   figures; decoder head omitted, it is training-only) + architecture description.
    demo.json      the demo user clip as web-app frames ({t, lm: 132 numbers}) + clip meta, so
                   "Try a sample" runs through exactly the same JS path as an uploaded video.
    tests/fixtures/*.json   Python's results on the demo user clip, each demo session clip, one
                   reference clip and a gap-injected copy of the user clip (input frames included):
                   events, summary, per-frame features, seq60, comparisons, score, embeddings.

Inputs are rounded *before* the Python fixture is computed, so Python and JS see identical numbers.
"""
from __future__ import annotations

import datetime as dt
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from pose import LM, MIN_VIS, MIRROR_PAIRS, NUM_LANDMARKS, PoseSequence, load_landmarks  # noqa: E402
from features import (FEATURE_NAMES, FEATURE_UNITS, N_FRAMES, SIDES, MIN_GOOD_FRAMES, BODY_LANDMARKS,  # noqa: E402
                      canonical_landmarks, pair_mid, shot_from_pose)
from compare import (FEEDBACK_SPEC, RADAR_AXES, W_DTW, W_LEARNED, compare_shot,  # noqa: E402
                     load_reference_library)
from drills import DRILLS, GENERIC_MAINTENANCE, coaching_plan  # noqa: E402
import features as _F  # noqa: E402
import mechanics as _M  # noqa: E402

FORMAT_VERSION = 1
FEATURE_LABELS = {"elbow_angle": "Elbow angle", "knee_angle": "Knee angle", "hip_angle": "Hip angle",
                  "trunk_lean": "Trunk lean", "wrist_height": "Wrist height", "wrist_forward": "Wrist forward",
                  "elbow_tuck": "Elbow tuck", "wrist_angle": "Wrist angle", "hip_rise": "Hip rise"}
SHORT = {"elbow_angle_set": "Elbow @ set", "elbow_tuck_set": "Elbow tuck @ set", "knee_angle_min": "Knee bend (min)",
         "hip_angle_set": "Hip alignment @ set", "trunk_lean_release": "Trunk lean @ release",
         "release_height": "Release height", "release_forward": "Release forward", "release_time_s": "Load → release",
         "follow_through_hold_s": "Follow-through hold", "follow_through_wrist_angle": "Wrist snap",
         "jump_height": "Jump height", "elbow_angle_release": "Elbow @ release"}


# ----------------------------------------------------------------------------- helpers

def _r(x, nd=4):
    """Round floats (recursively) for compact JSON; NaN/inf -> None."""
    if isinstance(x, dict):
        return {k: _r(v, nd) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_r(v, nd) for v in x]
    if isinstance(x, np.ndarray):
        return _r(x.tolist(), nd)
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        x = float(x)
        if not np.isfinite(x):
            return None
        return round(x, nd)
    return x


def _sig(a: np.ndarray, sig: int = 6):
    """float32 array -> nested lists of floats with `sig` significant figures."""
    flat = [float(f"{float(v):.{sig}g}") for v in np.asarray(a, np.float32).ravel()]
    return np.array(flat, dtype=np.float64).reshape(np.asarray(a).shape).tolist()


def _write(path: str, obj) -> int:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, separators=(",", ":"), allow_nan=False)
    return os.path.getsize(path)


def frames_from_seq(seq: PoseSequence, nd: int = 6) -> list:
    """PoseSequence -> web-app frames [{t, lm:[x,y,z,vis]*33} | null] (normalised image coords)."""
    out = []
    for i in range(seq.n_frames):
        lm = seq.landmarks[i]
        vis = np.nan_to_num(seq.visibility[i], nan=0.0)
        if np.isnan(lm[:, 0]).all():
            out.append(None)
            continue
        row = []
        for k in range(NUM_LANDMARKS):
            x, y, z = (float(v) for v in lm[k])
            v = float(vis[k])
            if not np.isfinite(x) or not np.isfinite(y):
                x, y, z, v = 0.0, 0.0, 0.0, 0.0
            row += [round(x, nd), round(y, nd), round(z if np.isfinite(z) else 0.0, nd), round(v, 4)]
        out.append(dict(t=round(i / seq.fps, 6), lm=row))
    return out


def seq_from_frames(frames: list, meta: dict, source: str = "web-frames") -> PoseSequence:
    """Inverse of frames_from_seq (uniform frame rate assumed — Python indexes frames by fps)."""
    T = len(frames)
    lms = np.full((T, NUM_LANDMARKS, 3), np.nan, np.float32)
    vis = np.zeros((T, NUM_LANDMARKS), np.float32)
    for i, fr in enumerate(frames):
        if fr is None:
            continue
        a = np.asarray(fr["lm"], np.float64).reshape(NUM_LANDMARKS, 4)
        lms[i] = a[:, :3]
        vis[i] = a[:, 3]
    return PoseSequence(lms, vis, float(meta["fps"]), int(meta["width"]), int(meta["height"]), source)


def clip_meta(seq: PoseSequence, name: str) -> dict:
    return dict(name=name, fps=float(seq.fps), width=int(seq.width), height=int(seq.height),
                duration=round(seq.n_frames / seq.fps, 6), source=seq.source)


def ghost_frames(shot) -> dict:
    """Reference landmarks for the ghost overlay: canonical (right-handed), facing +1, relative to
    the stance anchor (hip-mid x / ankle-mid y over the first 0.15 s) in torso units (y down),
    onset -> follow-through end resampled to N_FRAMES (same linear resampling as features.resample)."""
    ev = shot.events
    px, side, facing, mirrored = canonical_landmarks(shot.landmarks_px, shot.width, shot.side, shot.facing)
    first = px[: max(2, int(shot.fps * 0.15))]
    hip = pair_mid(first[:, LM.LEFT_HIP], first[:, LM.RIGHT_HIP])
    ank = pair_mid(first[:, LM.LEFT_ANKLE], first[:, LM.RIGHT_ANKLE])
    cx, floor = float(np.nanmedian(hip[:, 0])), float(np.nanmedian(ank[:, 1]))
    torso = float(shot.torso_px)
    rel = px.copy()
    rel[..., 0] = facing * (rel[..., 0] - cx) / torso
    rel[..., 1] = (rel[..., 1] - floor) / torso
    start, end = ev.onset, max(ev.follow_through_end, ev.onset + 1)
    seg = rel[start:end + 1].reshape(end - start + 1, -1)
    t_src = np.linspace(0, 1, seg.shape[0])
    t_dst = np.linspace(0, 1, N_FRAMES)
    res = np.stack([np.interp(t_dst, t_src, seg[:, c]) for c in range(seg.shape[1])], axis=1)
    span = max(1, end - start)
    phase = lambda f: (f - start) / span * (N_FRAMES - 1)
    return dict(frames=_r(res, 4), events=dict(onset=0.0, set=round(phase(ev.set_point), 3),
                                               release=round(phase(ev.release), 3), ftEnd=float(N_FRAMES - 1)),
                mirrored=bool(mirrored))


# ----------------------------------------------------------------------------- exports

def build_library(refs, stats) -> dict:
    tol = {k: (ok, mj) for k, _, _, ok, mj, _ in FEEDBACK_SPEC}
    metric_of = {feat: key for feat, key, _ in RADAR_AXES}
    feats = [dict(key=n, label=FEATURE_LABELS[n], unit=FEATURE_UNITS[n], tolerance=tol[metric_of[n]][0],
                  higherIsBetter=None, metric=metric_of[n]) for n in FEATURE_NAMES]
    metrics = [dict(key=k, label=l, short=SHORT.get(k, l), unit=u, tolOk=ok, tolMajor=mj, bigger=w[0], smaller=w[1])
               for k, l, u, ok, mj, w in FEEDBACK_SPEC]
    lib_refs = []
    for r in refs:
        ev = r.events
        lib_refs.append(dict(
            name=r.name, side=r.side, facing=float(r.facing), fps=float(r.fps), nFrames=int(r.per_frame.shape[0]),
            torsoPx=round(float(r.torso_px), 3), width=int(r.width), height=int(r.height),
            events=dict(onset=ev.onset, set=ev.set_point, peak=ev.peak_wrist_velocity, release=ev.release,
                        followThrough=ev.follow_through, ftEnd=ev.follow_through_end, holdS=round(ev.hold_s, 6),
                        releaseFound=bool(ev.release_found)),
            summary=_r(r.summary, 6), seq60=_r(r.seq, 4), ghost=ghost_frames(r)))
    keys = [k for k, *_ in FEEDBACK_SPEC]
    lib_mean = {k: float(np.mean([r.summary[k] for r in refs])) for k in refs[0].summary} if refs else {}
    return dict(
        format=FORMAT_VERSION, generated=dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        source="01-jumpshot-analyzer/export_web.py",
        note="All reference shooters are procedurally generated (synthetic); no real player data.",
        featureNames=FEATURE_NAMES, features=feats, metrics=metrics, metricOrder=keys,
        radarAxes=[dict(feature=f, key=k, label=l, unit=FEATURE_UNITS[f]) for f, k, l in RADAR_AXES],
        stats=dict(mean=_r(stats.mean, 6), std=_r(stats.std, 6)),
        libraryMean=_r(lib_mean, 6),
        dtw=dict(bandDivisor=4, scale=2.2, power=1.5, cost="euclidean on z-scored features",
                 distance="D[n,m] / path length", score="100*exp(-(d/scale)^power)"),
        learned=dict(score="100*((cos+1)/2)^2"),
        weights=dict(dtw=W_DTW, learned=W_LEARNED),
        scoring=dict(headline="mechanics", note="score = evidence-based mechanics score (0-100, unmeasured checks "
                     "re-normalised away); styleMatch = 0.6*DTW + 0.4*learned similarity to the closest reference."),
        mechanics=dict(
            checks=[dict(key=k, label=l, unit=u, band=list(b), ideal=i, tolerance=t, points=p, source=src, above=w[0], below=w[1])
                    for k, l, u, b, i, t, p, src, w in _M.MECHANICS],
            sources=_M.SOURCES,
            constants=dict(VIEW_FLAG=_M.VIEW_FLAG, VIEW_FRONTAL=_M.VIEW_FRONTAL, MIN_JUMP_TORSO=_M.MIN_JUMP_TORSO,
                           DIP_WINDOW_S=_M.DIP_WINDOW_S, JUMP_BEFORE_S=_M.JUMP_BEFORE_S, JUMP_AFTER_S=_M.JUMP_AFTER_S,
                           MIN_MEASURED_POINTS=_M.MIN_MEASURED_POINTS),
            note="Coaching ranges from published jump-shot kinematics (3-D lab data of skilled shooters) applied to "
                 "2-D phone video; not laws. Linear falloff to 0 points at range +/- tolerance."),
        constants=dict(
            N_FRAMES=N_FRAMES, MIN_VIS=MIN_VIS, MIN_GOOD_FRAMES=MIN_GOOD_FRAMES, MIN_POSE_FRAMES=8,
            BODY_LANDMARKS=BODY_LANDMARKS,
            smoothing=dict(method="savgol", window_s=0.2, polyorder=2, min_window=5, mode="interp"),
            events=dict(version=2, load_torso=_F.LOAD_TORSO, warmup_s=_F.WARMUP_S, set_window_s=_F.SET_WINDOW_S,
                        flex_lookback_s=_F.FLEX_LOOKBACK_S, min_flex_deg=_F.MIN_FLEX_DEG, min_ext_deg=_F.MIN_EXT_DEG,
                        onset_lookback_s=_F.ONSET_LOOKBACK_S, reach_s=_F.REACH_S, cut_torso=_F.CUT_TORSO,
                        glitch_ratio=list(_F.GLITCH_RATIO),
                        top_frac=0.05, min_top=2, release_window_s=0.6, release_deg=_F.RELEASE_DEG,
                        release_within_max_deg=4.0, release_found_deg=_F.RELEASE_FOUND_DEG, onset_baseline_s=0.15,
                        onset_knee_deg=4.0, onset_wrist_torso=0.12, ft_height_frac=0.65, ft_elbow_deg=140.0,
                        ft_min_s=0.2, hip_baseline_s=0.2),
            quality=dict(body_vis_major=70, body_vis_minor=85, arm_vis_minor=85, gaps_frac=0.1),
            mirrorPairs=MIRROR_PAIRS),
        references=lib_refs,
    )


def build_drills() -> dict:
    return dict(format=FORMAT_VERSION,
                note="Generic form-shooting drills (drills.py); not medical or professional coaching advice.",
                drills={k: {d: dict(drill=v[0], prescription=v[1], cue=v[2]) for d, v in dd.items()} for k, dd in DRILLS.items()},
                generic=dict(drill=GENERIC_MAINTENANCE[0], prescription=GENERIC_MAINTENANCE[1], cue=GENERIC_MAINTENANCE[2]),
                maxItems=6)


def build_encoder(model, stats, info: dict) -> dict:
    sd = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()
          if not k.startswith("decoder") and not k.endswith("num_batches_tracked")}
    bn_eps = [m.eps for m in model.net if m.__class__.__name__ == "BatchNorm1d"]
    arch = dict(
        input="(60, F) features z-scored with stats.mean/std, transposed to (F, 60)",
        layers=[
            dict(op="conv1d", w="net.0.weight", b="net.0.bias", stride=1, padding=2),
            dict(op="batchnorm", prefix="net.1", eps=bn_eps[0]), dict(op="gelu"),
            dict(op="conv1d", w="net.3.weight", b="net.3.bias", stride=2, padding=2),
            dict(op="batchnorm", prefix="net.4", eps=bn_eps[1]), dict(op="gelu"),
            dict(op="conv1d", w="net.6.weight", b="net.6.bias", stride=2, padding=1),
            dict(op="batchnorm", prefix="net.7", eps=bn_eps[2]), dict(op="gelu"),
            dict(op="conv1d", w="net.9.weight", b="net.9.bias", stride=1, padding=1),
            dict(op="gelu"),
            dict(op="pool", how="concat(mean over time, max over time)"),
            dict(op="linear", w="head.0.weight", b="head.0.bias"), dict(op="gelu"),
            dict(op="linear", w="head.2.weight", b="head.2.bias"),
            dict(op="l2normalize", eps=1e-12),
        ],
        gelu="exact (erf)", embedDim=int(model.head[2].out_features), nFrames=N_FRAMES, nFeatures=len(FEATURE_NAMES))
    return dict(format=FORMAT_VERSION, arch=arch, stats=dict(mean=_r(stats.mean, 6), std=_r(stats.std, 6)),
                info={k: (round(v, 4) if isinstance(v, float) else v) for k, v in (info or {}).items() if k != "loss_history"},
                state={k: _sig(v) for k, v in sd.items()})


def _round_seq(seq: PoseSequence, nd: int = 6) -> PoseSequence:
    """Round a sequence exactly as frames_from_seq does, so the Python fixture sees the JSON numbers."""
    frames = frames_from_seq(seq, nd)
    meta = dict(fps=seq.fps, width=seq.width, height=seq.height)
    out = seq_from_frames(frames, meta, seq.source)
    out.video_path = seq.video_path
    return out, frames


def python_result(seq: PoseSequence, name: str, refs, model, stats) -> dict:
    from model import embed
    shot = shot_from_pose(seq, name)
    res = compare_shot(shot, refs, model, stats)
    best = res["references"][0]
    plan = coaching_plan(res["feedback"])
    ev = shot.events
    emb = embed(model, shot.seq, stats) if model is not None else None
    return dict(
        hand=shot.side, facing="right" if shot.facing > 0 else "left", mirrored=shot.side == "left",
        events=dict(onset=ev.onset, set=ev.set_point, peak=ev.peak_wrist_velocity, release=ev.release,
                    followThrough=ev.follow_through, ftEnd=ev.follow_through_end, holdS=ev.hold_s,
                    releaseFound=ev.release_found, peakExtension=ev.peak_extension, setMethod=ev.set_method),
        torsoPx=float(shot.torso_px),
        summary=_r(shot.summary, 6),
        perFrame={n: _r(shot.per_frame[:, j], 4) for j, n in enumerate(FEATURE_NAMES)},
        seq60=_r(shot.seq, 4),
        comparisons=[dict(name=r["name"], score=r["score"], dtw=r["dtw_score"], learned=r["learned_score"],
                          dtwDistance=r["dtw_distance"]) for r in res["references"]],
        closest=res["closest"], score=float(res["headline_score"]), scoreKind=res["score_kind"],
        mechanicsScore=res["mechanics_score"], styleMatch=float(res["style_match"]),
        mechanics=[dict(key=m["key"], value=_r(m["value"], 6), status=m["status"], points=m["points"],
                        maxPoints=m["maxPoints"], text=m["text"]) for m in res["mechanics"]],
        feedback=[dict(key=f["key"], kind=f.get("kind", "style"), severity=f["severity"], delta=_r(f["delta"], 6),
                       user=_r(f["user"], 6), ref=_r(f["reference"], 6), text=f["message"]) for f in res["feedback"]],
        plan=[dict(key=p["key"], drill=p["drill"], severity=p["severity"]) for p in plan],
        radar=dict(user=res["radar"]["user"], ref=res["radar"]["closest"], mean=res["radar"]["library_mean"]),
        quality=dict(level=shot.quality["confidence"], codes=[f["code"] for f in shot.quality["flags"]],
                     bodyVisibilityPct=shot.quality["body_visibility_pct"]),
        embedding=None if emb is None else _r(emb, 6),
    )


def _inject_gaps(seq: PoseSequence) -> PoseSequence:
    """Copy with dropped frames (no person) and low-visibility joints, to test gap filling parity."""
    lms, vis = seq.landmarks.copy(), seq.visibility.copy()
    for i in (5, 19, 20, 21, 47):
        lms[i] = np.nan
        vis[i] = 0.0
    vis[30:33, LM.RIGHT_WRIST] = 0.3          # wrist briefly occluded by the ball
    vis[10:14, LM.RIGHT_KNEE] = 0.2
    return PoseSequence(lms, vis, seq.fps, seq.width, seq.height, seq.source + "+gaps", seq.video_path)


def export(out_dir: str, references: str | None = None, demo_dir: str | None = None, fixtures: bool = True,
           verbose: bool = True) -> dict:
    from model import FeatureStats, load_model
    references = references or os.path.join(HERE, "references")
    demo_dir = demo_dir or os.path.join(HERE, "demo")
    refs = load_reference_library(references)
    enc_path = os.path.join(references, "_encoder.pt")
    if os.path.isfile(enc_path):
        model, stats, info = load_model(enc_path)
    else:
        model, stats, info = None, FeatureStats.fit([s.seq for s in refs]), {}
    sizes = {}
    sizes["library.json"] = _write(os.path.join(out_dir, "library.json"), build_library(refs, stats))
    sizes["drills.json"] = _write(os.path.join(out_dir, "drills.json"), build_drills())
    enc_file = os.path.join(out_dir, "encoder.json")
    if model is not None:
        sizes["encoder.json"] = _write(enc_file, build_encoder(model, stats, info))
    elif os.path.isfile(enc_file):
        os.remove(enc_file)                    # stale weights would not match this library

    user_side = os.path.join(demo_dir, "user_shot.landmarks.npz")
    if os.path.isfile(user_side):
        useq = load_landmarks(user_side, os.path.join(demo_dir, "user_shot.mp4"))
        name = "User shot (sample)"
        useq_r, uframes = _round_seq(useq)
        sizes["demo.json"] = _write(os.path.join(out_dir, "demo.json"), dict(
            format=FORMAT_VERSION, meta=clip_meta(useq_r, name),
            note="Procedurally generated sample shooter (synthetic landmarks, no real footage).", frames=uframes))
        if fixtures:
            fx_dir = os.path.join(out_dir, "tests", "fixtures")
            os.makedirs(fx_dir, exist_ok=True)
            for old in glob.glob(os.path.join(fx_dir, "*.json")):
                os.remove(old)
            clips = [("user_shot", useq, name)]
            clips.append(("user_shot_gaps", _inject_gaps(useq), "User shot (gaps)"))
            for p in sorted(glob.glob(os.path.join(demo_dir, "sessions", "session_*.landmarks.npz"))):
                slug = os.path.basename(p).split(".")[0]
                clips.append((slug, load_landmarks(p), slug.replace("_", " ").title()))
            ref_a = os.path.join(demo_dir, "ref_a_high_release.landmarks.npz")
            if os.path.isfile(ref_a):
                clips.append(("ref_a_as_user", load_landmarks(ref_a), "Reference A as user"))
            index = []
            for slug, seq, nm in clips:
                seq_r, frames = _round_seq(seq)
                exp = python_result(seq_r, nm, refs, model, stats)
                fx = dict(format=FORMAT_VERSION, name=nm, slug=slug, meta=clip_meta(seq_r, nm),
                          frames=frames, expected=exp)
                _write(os.path.join(fx_dir, f"{slug}.json"), fx)
                index.append(dict(slug=slug, score=exp["score"], closest=exp["closest"]))
            ref_emb = {}
            if model is not None:
                from model import embed
                ref_emb = {r.name: _r(embed(model, r.seq, stats), 6) for r in refs}
            _write(os.path.join(fx_dir, "_index.json"), dict(clips=index, refEmbeddings=ref_emb))
            sizes["tests/fixtures"] = len(index)
    if verbose:
        print("[export-web] " + ", ".join(f"{k} {v / 1024:.0f} KB" if k.endswith(".json") else f"{k}: {v}"
                                          for k, v in sizes.items()) + f" -> {out_dir}")
    return sizes


def export_clip_fixtures(frames_json: str, out_dir: str, windows: list[tuple[int, int, str]] | None = None,
                         references: str | None = None, rate_step: int = 1) -> list[dict]:
    """Python fixtures for arbitrary web-app frames (e.g. a real phone clip run through pose.py) — for
    local parity checks only (point JUMPSHOT_EXTRA_FIXTURES at out_dir). windows: (start, end, slug)
    inclusive frame ranges; rate_step > 1 keeps every n-th frame (fps / n)."""
    from model import FeatureStats, load_model
    references = references or os.path.join(HERE, "references")
    refs = load_reference_library(references)
    enc = os.path.join(references, "_encoder.pt")
    model, stats, _ = load_model(enc) if os.path.isfile(enc) else (None, FeatureStats.fit([s.seq for s in refs]), {})
    d = json.load(open(frames_json))
    frames, meta = d["frames"], dict(d["meta"])
    windows = windows or [(0, len(frames) - 1, "clip")]
    os.makedirs(out_dir, exist_ok=True)
    out = []
    for lo, hi, slug in windows:
        fr = frames[lo:hi + 1][::rate_step]
        m = dict(meta, fps=meta["fps"] / rate_step)
        fr = [None if f is None else dict(f, t=round(i / m["fps"], 6)) for i, f in enumerate(fr)]
        m["duration"] = round(len(fr) / m["fps"], 6)
        seq = seq_from_frames(fr, m, "web-frames")
        try:
            exp = python_result(seq, slug, refs, model, stats)
        except ValueError as ex:
            exp = dict(error=str(ex))
        _write(os.path.join(out_dir, f"{slug}.json"), dict(format=FORMAT_VERSION, name=slug, slug=slug, meta=m, frames=fr, expected=exp))
        out.append(dict(slug=slug, score=exp.get("score"), closest=exp.get("closest")))
    _write(os.path.join(out_dir, "_index.json"), dict(clips=out, refEmbeddings={}))
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--references", default=None)
    ap.add_argument("--no-fixtures", action="store_true")
    a = ap.parse_args()
    export(a.out, references=a.references, fixtures=not a.no_fixtures)
