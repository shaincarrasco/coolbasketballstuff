"""
compare.py — score a user shot against a reference library and produce feedback.

Form similarity (0-100) per reference blends:
    * DTW score      — dynamic time warping on z-scored per-frame features (robust baseline)
    * learned score  — cosine similarity of ShotEncoder embeddings
    final = W_DTW * dtw + W_LEARNED * learned

Per-feature feedback compares the user's summary metrics with the reference's,
in measured units (degrees, seconds, torso-lengths) with direction words.
"""
from __future__ import annotations

import glob
import os

import numpy as np

from features import FEATURE_NAMES, FEATURE_UNITS, Shot, load_shot
from model import FeatureStats, ShotEncoder, cosine_score, dtw_score, dtw_similarity, embed

W_DTW = 0.6
W_LEARNED = 0.4

# (summary key, label, unit, tolerance for "ok", tolerance for "major", bigger/smaller wording)
FEEDBACK_SPEC = [
    ("elbow_angle_set", "Elbow angle at set point", "deg", 5, 12, ("wider (less loaded)", "tighter (more loaded)")),
    ("elbow_tuck_set", "Elbow in front of shoulder at set point", "torso", 0.06, 0.15, ("further out", "more tucked")),
    ("knee_angle_min", "Deepest knee bend", "deg", 6, 15, ("shallower", "deeper")),
    ("hip_angle_set", "Shoulder-hip-knee alignment at set point", "deg", 6, 15, ("more upright", "more folded")),
    ("trunk_lean_release", "Trunk lean at release", "deg", 3, 8, ("more forward", "more upright")),
    ("release_height", "Release height (wrist above head)", "torso", 0.06, 0.15, ("higher", "lower")),
    ("release_forward", "Release point in front of shoulder", "torso", 0.08, 0.2, ("further forward", "further back")),
    ("release_time_s", "Load-to-release time", "s", 0.06, 0.15, ("later (slower)", "earlier (quicker)")),
    ("follow_through_hold_s", "Follow-through hold", "s", 0.08, 0.2, ("longer", "shorter")),
    ("follow_through_wrist_angle", "Wrist snap (elbow-wrist-finger angle)", "deg", 8, 20, ("flatter (less snap)", "sharper (more snap)")),
    ("jump_height", "Jump height (hip rise)", "torso", 0.08, 0.2, ("higher", "lower")),
    ("elbow_angle_release", "Elbow extension at release", "deg", 5, 12, ("more extended", "less extended")),
]


def _fmt(v: float, unit: str) -> str:
    if unit == "deg":
        return f"{v:.0f}°"
    if unit == "s":
        return f"{v:.2f}s"
    if unit == "torso":
        return f"{v:.2f} torso"
    return f"{v:.2f}"


def _fmt_delta(v: float, unit: str) -> str:
    if unit == "deg":
        return f"{abs(v):.0f}°"
    if unit == "s":
        return f"{abs(v):.2f}s"
    if unit == "torso":
        return f"{abs(v):.2f} torso-lengths (~{abs(v) * 50:.0f} cm for a 50 cm torso)"
    return f"{abs(v):.2f}"


def feature_feedback(user: Shot, ref: Shot) -> list[dict]:
    items = []
    for key, label, unit, tol_ok, tol_major, (bigger, smaller) in FEEDBACK_SPEC:
        u = float(user.summary[key])
        r = float(ref.summary[key])
        d = u - r
        if abs(d) <= tol_ok:
            sev = "ok"
            msg = f"{label}: {_fmt(u, unit)} vs {_fmt(r, unit)} — within {_fmt_delta(tol_ok, unit).split(' (')[0]} of reference."
        else:
            sev = "major" if abs(d) >= tol_major else "minor"
            word = bigger if d > 0 else smaller
            msg = f"{label}: {_fmt(u, unit)} vs {_fmt(r, unit)} — {_fmt_delta(d, unit)} {word} than reference."
        items.append(dict(key=key, label=label, unit=unit, user=u, reference=r, delta=d, severity=sev, message=msg))
    return items


# Radar chart axes: one representative value per per-frame feature (all in the Shot summary).
RADAR_AXES = [
    ("elbow_angle", "elbow_angle_set", "Elbow @ set"),
    ("knee_angle", "knee_angle_min", "Knee min"),
    ("hip_angle", "hip_angle_set", "Hip align"),
    ("trunk_lean", "trunk_lean_release", "Trunk lean"),
    ("wrist_height", "release_height", "Release height"),
    ("wrist_forward", "release_forward", "Release fwd"),
    ("elbow_tuck", "elbow_tuck_set", "Elbow tuck"),
    ("wrist_angle", "follow_through_wrist_angle", "Wrist snap"),
    ("hip_rise", "jump_height", "Jump height"),
]


def radar_profile(user: Shot, refs: list[Shot], best: Shot) -> dict:
    """Min-max normalised (0..1) radar values for user / closest reference / library mean.
    The range per axis spans the library and the user with a 10 % margin, so a value on
    the rim means 'highest of everything seen', not 'best'."""
    axes, u_raw, b_raw, m_raw, lo, hi = [], [], [], [], [], []
    for feat, key, label in RADAR_AXES:
        vals = [float(r.summary[key]) for r in refs]
        uv = float(user.summary[key])
        bv = float(best.summary[key])
        mv = float(np.mean(vals))
        a, b = min(vals + [uv]), max(vals + [uv])
        pad = max((b - a) * 0.1, 1e-6)
        axes.append(dict(feature=feat, key=key, label=label, unit=FEATURE_UNITS[feat]))
        u_raw.append(uv); b_raw.append(bv); m_raw.append(mv); lo.append(a - pad); hi.append(b + pad)

    def norm(vs):
        return [round((v - l) / (h - l), 4) for v, l, h in zip(vs, lo, hi)]

    return dict(axes=axes, user=norm(u_raw), closest=norm(b_raw), library_mean=norm(m_raw),
                raw=dict(user=u_raw, closest=b_raw, library_mean=m_raw), lo=lo, hi=hi)


def load_reference_library(ref_dir: str) -> list[Shot]:
    paths = sorted(glob.glob(os.path.join(ref_dir, "*.npz")))
    paths = [p for p in paths if not os.path.basename(p).startswith("_")]
    if not paths:
        raise FileNotFoundError(f"no reference .npz files in {ref_dir}; run `cli.py build-references` first")
    return [load_shot(p) for p in paths]


def compare_shot(user: Shot, refs: list[Shot], model: ShotEncoder | None, stats: FeatureStats) -> dict:
    zu = embed(model, user.seq, stats) if model is not None else None
    per_ref = []
    for ref in refs:
        d, path = dtw_similarity(stats.apply(user.seq), stats.apply(ref.seq))
        s_dtw = dtw_score(d)
        if zu is not None:
            s_learned = cosine_score(zu, embed(model, ref.seq, stats))
            final = W_DTW * s_dtw + W_LEARNED * s_learned
        else:
            s_learned = None
            final = s_dtw
        # per-feature DTW contribution (which features drive the distance)
        ua, ra = stats.apply(user.seq), stats.apply(ref.seq)
        contrib = np.zeros(len(FEATURE_NAMES))
        for i, j in path:
            contrib += np.abs(ua[i] - ra[j])
        contrib /= len(path)
        per_ref.append(dict(
            name=ref.name, score=round(final, 1), dtw_score=round(s_dtw, 1),
            learned_score=None if s_learned is None else round(s_learned, 1),
            dtw_distance=round(d, 3), dtw_path=path,
            feature_distance={n: round(float(c), 3) for n, c in zip(FEATURE_NAMES, contrib)},
            feedback=feature_feedback(user, ref),
            summary=ref.summary, events=ref.events.to_dict(),
        ))
    per_ref.sort(key=lambda r: -r["score"])
    best = per_ref[0]
    best_shot = next(r for r in refs if r.name == best["name"])
    from mechanics import mechanics_feedback
    style_fb = [dict(fb, kind="style") for fb in best["feedback"]]
    mech_score = user.mechanics_score
    return dict(
        user=dict(name=user.name, side=user.side, facing=user.facing, fps=user.fps, events=user.events.to_dict(),
                  summary=user.summary, n_frames=int(user.per_frame.shape[0]), missing_frames=user.missing_frames,
                  source=user.source, quality=user.quality, video=user.video_path),
        closest=best["name"],
        # headline = evidence-based mechanics score; similarity to the references is the style match
        headline_score=mech_score if mech_score is not None else best["score"],
        score_kind="mechanics" if mech_score is not None else "style",
        mechanics_score=mech_score, style_match=best["score"], mechanics=user.mechanics,
        feedback=mechanics_feedback(user.mechanics) + style_fb,
        weights=dict(dtw=W_DTW, learned=W_LEARNED),
        references=per_ref,
        radar=radar_profile(user, refs, best_shot),
    )


def results_for_json(results: dict) -> dict:
    """Strip bulky fields (DTW paths) for results.json."""
    out = dict(results)
    out["references"] = [{k: v for k, v in r.items() if k != "dtw_path"} for r in results["references"]]
    return out
