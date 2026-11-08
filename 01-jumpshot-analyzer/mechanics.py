"""
mechanics.py — evidence-based mechanics score (the headline score, 0-100).

Each check measures one kinematic quantity of the shot and compares it with a coaching range taken
from published jump-shot kinematics (or, where marked, a coaching convention). Inside the range =
full points; outside, points fall off linearly to 0 at range ± tolerance. A value that cannot be
measured reliably from this clip (joint hidden, camera too frontal, no jump, clip cut) is
'unmeasured' and its points are removed from the total (re-normalised), never scored as 0.

These are conservative coaching ranges, not laws: published values come from small samples of
skilled shooters filmed in 3-D; here every angle is a 2-D projection from one phone camera.
Similarity to the (synthetic) reference shooters is reported separately as the style match.
Mirrored line-by-line in site/jumpshot/app/engine.mjs (parity-tested).
"""
from __future__ import annotations

import numpy as np

from features import FEATURE_NAMES, SIDES, LM

SOURCES = [
    dict(id="crosbi2019",
         cite="Kinematic analysis of jump shot with backward take-off (CROSBI 683397, 2019)",
         note="Model jump shot: elbow 72–88° at the set position, knee flexion 102–115°, elbow ~164° at "
              "release, release at or just before the peak of the jump."),
    dict(id="frontiers2026",
         cite="Frontiers in Sports and Active Living (2026), doi:10.3389/fspor.2026.1732293",
         url="https://doi.org/10.3389/fspor.2026.1732293",
         note="Proficient 3-point shooters show greater hip, knee and ankle flexion."),
    dict(id="taborda",
         cite="Taborda et al.",
         note="Elbow 126.7 ± 9.4° and shoulder flexion 129.8 ± 9.2° at release."),
    dict(id="franca2021",
         cite="França et al. (2021), Acta of Bioengineering and Biomechanics 24(1)",
         note="Release at or just before the peak of the jump; the jump stays close to vertical."),
    dict(id="coaching",
         cite="Coaching convention (not a published norm)",
         note="Hold the follow-through about 0.3 s or longer; keep the trunk and hips stacked through the jump."),
]

# key, label, unit, band (lo, hi), ideal, tolerance, max points, sources, (word above, word below)
MECHANICS = [
    ("set_elbow", "Elbow angle at set point", "deg", (70, 100), 85, 20, 20, ["crosbi2019"],
     ("more open (under-loaded)", "more closed (over-folded)")),
    ("dip_knee", "Deepest knee bend in the dip", "deg", (100, 130), 115, 20, 15, ["crosbi2019", "frontiers2026"],
     ("straighter (shallow dip)", "deeper")),
    ("release_elbow", "Elbow extension at release", "deg", (125, 180), 160, 20, 20, ["taborda", "crosbi2019"],
     ("more extended", "less extended")),
    ("release_shoulder", "Shoulder flexion at release", "deg", (120, 150), 130, 20, 15, ["taborda"],
     ("higher (arm behind the head)", "lower (pushed out in front)")),
    ("release_timing", "Release vs jump peak", "s", (-0.12, 0.03), -0.03, 0.12, 15, ["crosbi2019", "franca2021"],
     ("later (after the peak, on the way down)", "earlier (well before the peak)")),
    ("follow_through", "Follow-through hold", "s", (0.3, 3.0), 0.5, 0.2, 10, ["coaching"],
     ("longer", "shorter")),
    ("balance_lean", "Trunk lean change, set to release", "deg", (0, 10), 0, 10, 2, ["coaching", "franca2021"],
     ("more", "less")),
    ("balance_drift", "Horizontal hip drift during the jump", "torso", (0, 0.4), 0, 0.4, 3, ["coaching", "franca2021"],
     ("more", "less")),
]
ANGLE_KEYS = {"set_elbow", "release_elbow", "release_shoulder", "balance_lean"}
VIEW_FLAG = 0.45          # shoulder width / torso length above this -> off-side camera flag
VIEW_FRONTAL = 0.75       # ... above this -> arm/trunk angles unmeasured
MIN_JUMP_TORSO = 0.05
DIP_WINDOW_S = 0.8
JUMP_BEFORE_S, JUMP_AFTER_S = 0.3, 0.4
MIN_MEASURED_POINTS = 40


def _fmt(v: float, unit: str) -> str:
    if unit == "deg":
        return f"{v:.0f}°"
    if unit == "s":
        return f"{v:+.2f} s" if v < 0 else f"{v:.2f} s"
    if unit == "torso":
        return f"{v:.2f} torso"
    return f"{v:.2f}"


def _fmt_range(lo: float, hi: float, unit: str) -> str:
    u = {"deg": "°", "s": " s", "torso": " torso"}.get(unit, "")
    return f"{lo:g}–{hi:g}{u}"


def _fmt_dist(d: float, unit: str) -> str:
    if unit == "deg":
        return f"{d:.0f}°"
    if unit == "s":
        return f"{d:.2f} s"
    if unit == "torso":
        return f"{d:.2f} torso"
    return f"{d:.2f}"


def score_item(spec, value, reason=None, frame=None) -> dict:
    key, label, unit, (lo, hi), ideal, tol, pts, src, (above, below) = spec
    item = dict(key=key, label=label, unit=unit, band=[lo, hi], ideal=ideal, tolerance=tol, maxPoints=pts,
                source=src, frame=frame)
    if value is None or not np.isfinite(value):
        item.update(value=None, status="unmeasured", points=0.0,
                    reason=reason or "not measurable on this clip",
                    text=f"{label}: not measured — {reason or 'not measurable on this clip'}.")
        return item
    value = float(value)
    dist = lo - value if value < lo else (value - hi if value > hi else 0.0)
    p = pts * max(0.0, 1.0 - dist / tol)
    status = "good" if dist == 0 else ("close" if dist <= 0.5 * tol else "off")
    rng = _fmt_range(lo, hi, unit)
    if dist == 0:
        text = f"{label}: {_fmt(value, unit)} — inside the {rng} range."
    else:
        text = f"{label}: {_fmt(value, unit)} — {_fmt_dist(dist, unit)} {above if value > hi else below} than the {rng} range."
    item.update(value=value, status=status, points=round(p, 2), edge=float(hi if value > hi else lo),
                delta=float(value - hi if value > hi else (value - lo if value < lo else 0.0)), text=text)
    return item


def _observed(obs: np.ndarray, joints, i: int) -> bool:
    """Every joint observed (visibility mask, not gap-filled) in at least half of frames i-1..i+1."""
    T = obs.shape[0]
    a, b = max(0, i - 1), min(T - 1, i + 1)
    n = b - a + 1
    for j in joints:
        if 2 * int(obs[a:b + 1, j].sum()) < n:
            return False
    return True


def mechanics_profile(feats: np.ndarray, ex: dict, ev, seq, side: str, quality: dict, torso: float = 1.0) -> dict:
    """Mechanics checks for one shot. seq = the visibility-/glitch-masked PoseSequence (NaN = not
    observed). Appends the off-side-view flag to `quality` (in place). Returns {items, score}."""
    T = feats.shape[0]
    fps = float(ev.fps)
    f = {n: feats[:, i] for i, n in enumerate(FEATURE_NAMES)}
    s = SIDES[side]
    obs = ~np.isnan(seq.landmarks[:, :, 0])
    rel, sp = ev.release, ev.set_point
    codes = {fl["code"] for fl in quality.get("flags", [])}
    spec = {m[0]: m for m in MECHANICS}

    sw = ex["shoulder_width"][ev.onset:ev.follow_through_end + 1]
    sw = sw[np.isfinite(sw)]
    view = float(np.median(sw)) if sw.size else 0.0
    if view > VIEW_FLAG:
        quality.setdefault("flags", []).append(dict(
            code="off_side_view", severity="minor",
            message=f"Camera is off the side (shoulders {view * 100:.0f} % of torso length apart) - joint angles "
                    f"are 2-D projections; film side-on for the most accurate numbers."))
        if quality.get("confidence") == "ok":
            quality["confidence"] = "minor"
    frontal = view > VIEW_FRONTAL
    arm = [s["shoulder"], s["elbow"], s["wrist"]]
    items = []

    def angle_item(key, value, joints, frame, why_hidden):
        if frontal and key in ANGLE_KEYS:
            return score_item(spec[key], None, "camera too frontal to measure this angle", frame)
        if not _observed(obs, joints, frame):
            return score_item(spec[key], None, why_hidden, frame)
        return score_item(spec[key], value, None, frame)

    items.append(angle_item("set_elbow", f["elbow_angle"][sp], arm, sp, "shooting arm hidden at the set point"))
    lo = max(0, rel - int(round(fps * DIP_WINDOW_S)))
    kmin = int(lo + np.argmin(f["knee_angle"][lo:rel + 1]))
    items.append(angle_item("dip_knee", f["knee_angle"][kmin], [s["hip"], s["knee"], s["ankle"]], kmin,
                            "leg hidden at the bottom of the dip"))
    items.append(angle_item("release_elbow", f["elbow_angle"][rel], arm, rel, "shooting arm hidden at release"))
    items.append(angle_item("release_shoulder", ex["shoulder_flex"][rel], [s["shoulder"], s["elbow"], s["hip"]], rel,
                            "shoulder or hip hidden at release"))
    # release timing vs the peak of the jump (hip height)
    hr = f["hip_rise"]
    a, b = max(0, rel - int(round(fps * JUMP_BEFORE_S))), min(T - 1, rel + int(round(fps * JUMP_AFTER_S)))
    pk = int(a + np.argmax(hr[a:b + 1]))
    rise = float(hr[pk] - hr[min(sp, pk):pk + 1].min())
    if rise < MIN_JUMP_TORSO:
        items.append(score_item(spec["release_timing"], None, "no clear jump (set shot?)", rel))
    elif pk == T - 1:
        items.append(score_item(spec["release_timing"], None, "clip ends before the peak of the jump", rel))
    elif not _observed(obs, [s["hip"]], pk):
        items.append(score_item(spec["release_timing"], None, "hips hidden at the peak of the jump", rel))
    else:
        # sub-frame: interpolated release crossing vs parabolic peak of hip height
        pk_sub = float(pk)
        if 0 < pk < T - 1:
            y0, y1, y2 = float(hr[pk - 1]), float(hr[pk]), float(hr[pk + 1])
            den = y0 - 2.0 * y1 + y2
            if den < 0:
                pk_sub = pk + 0.5 * (y0 - y2) / den
        rel_sub = ev.release_sub if ev.release_sub >= 0 else float(rel)
        items.append(score_item(spec["release_timing"], (rel_sub - pk_sub) / fps, None, rel))
    if "clip_cut" in codes and ev.hold_s < spec["follow_through"][3][0]:
        items.append(score_item(spec["follow_through"], None, "clip ends during the follow-through", rel))
    else:
        items.append(score_item(spec["follow_through"], ev.hold_s, None, rel))
    trunk = [LM.LEFT_SHOULDER if side == "left" else LM.RIGHT_SHOULDER, s["hip"]]
    if not _observed(obs, trunk, sp):
        items.append(score_item(spec["balance_lean"], None, "trunk hidden at the set point", rel))
    else:
        items.append(angle_item("balance_lean", abs(float(f["trunk_lean"][rel]) - float(f["trunk_lean"][sp])),
                                trunk, rel, "trunk hidden at release"))
    e = min(T - 1, rel + int(round(fps * JUMP_AFTER_S)))
    drift = abs(float(ex["hip_x"][e]) - float(ex["hip_x"][sp])) / max(float(torso), 1e-6)
    if not (_observed(obs, [s["hip"]], sp) and _observed(obs, [s["hip"]], e)):
        items.append(score_item(spec["balance_drift"], None, "hips hidden during the jump", e))
    else:
        items.append(score_item(spec["balance_drift"], drift, None, e))

    measured = [it for it in items if it["status"] != "unmeasured"]
    tot = sum(it["maxPoints"] for it in measured)
    score = round(100.0 * sum(it["points"] for it in measured) / tot, 1) if tot >= MIN_MEASURED_POINTS else None
    return dict(items=items, score=score, view=view)


def mechanics_feedback(items: list[dict]) -> list[dict]:
    """Feedback entries (biggest point losses first) in the same shape as compare.feature_feedback."""
    out = []
    for it in items:
        if it["status"] == "unmeasured":
            continue
        sev = {"good": "ok", "close": "minor", "off": "major"}[it["status"]]
        out.append(dict(key=it["key"], label=it["label"], unit=it["unit"], user=it["value"], reference=it["edge"],
                        delta=it["delta"], severity=sev, message=it["text"], kind="mechanics",
                        points_lost=round(it["maxPoints"] - it["points"], 2)))
    out.sort(key=lambda d: -d["points_lost"])
    return out
