"""
drills.py — map measured findings to concrete, generic form-shooting drills.

Each feedback item from compare.feature_feedback has a `key` (summary metric) and a
signed `delta` (user - reference). DRILLS maps (key, direction) -> a drill with a
prescription (sets x reps / seconds) and a one-line cue. Drills are deliberately
generic and sensible; this is not medical or professional coaching advice.
"""
from __future__ import annotations

# key -> {"high": drill when user value > reference, "low": drill when user value < reference}
# drill = (name, prescription, cue)
DRILLS: dict[str, dict[str, tuple[str, str, str]]] = {
    "elbow_angle_set": {
        "high": ("L-shape set-point holds", "3 x 10 holds, 3 s each",
                 "Form the elbow into an L (about 90 deg) at the set point before every rep; check in a mirror."),
        "low": ("Set-point ball lifts", "3 x 10 lifts from waist to set",
                "Load less: the forearm should be roughly vertical at the set point, not folded onto the biceps."),
    },
    "elbow_tuck_set": {
        "high": ("Wall elbow-track drill", "3 x 12 reps, 1 m from a wall",
                 "Stand side-on to a wall so the elbow brushes it on the way up: keeps the elbow under the ball."),
        "low": ("Elbow-under-ball form shooting", "3 x 15 from 1 m",
                "Let the elbow sit slightly in front of the shoulder so the forearm can extend toward the rim."),
    },
    "knee_angle_min": {
        "high": ("Dip-and-go form shots", "3 x 15 from the free-throw line",
                 "Sit into a quarter squat before the lift; the legs start the shot, the arm finishes it."),
        "low": ("Chair-height dip reps", "3 x 12 reps",
                "Dip only to the top of a chair seat: a shallower, quicker load keeps the rhythm one-motion."),
    },
    "hip_angle_set": {
        "high": ("Hip-hinge load", "3 x 10 slow reps, no ball",
                 "Hinge slightly at the hips at the set point so the chest stays over the feet."),
        "low": ("Tall-chest form shooting", "3 x 15 from 1.5 m",
                "Load with the knees, not the trunk: keep the chest up and the eyes on the rim."),
    },
    "trunk_lean_release": {
        "high": ("Heel-tap balance shots", "3 x 10 from 2 m",
                 "Finish with the hips under the shoulders; land where you jumped from."),
        "low": ("Toe-line lean-in shots", "3 x 10 from 2 m",
                "Allow a few degrees of forward lean at release so the shot carries, instead of fading."),
    },
    "release_height": {
        "high": ("Natural-release form shots", "3 x 15 from 1 m",
                 "Release as the arm reaches full extension, not after an extra reach; keep the wrist relaxed."),
        "low": ("High-release one-hand form shooting", "3 x 20 from 1 m (3 ft)",
                "Finish with the elbow above the eyebrow and the wrist above the head before the ball leaves."),
    },
    "release_forward": {
        "high": ("Set-point under the eye", "3 x 15 from 1.5 m",
                 "Start the ball closer to the forehead so the release stays over the head, not pushed out."),
        "low": ("Reach-to-the-rim finish", "3 x 15 from 1.5 m",
                "Extend toward the rim on release; the fingers should finish pointing at the target."),
    },
    "release_time_s": {
        "high": ("One-motion form shooting", "3 x 20 from 1 m (3 ft)",
                 "No pause at the set point: dip, lift and release in one continuous motion."),
        "low": ("Count-two rhythm shots", "3 x 15 from the free-throw line",
                "Say 'one' at the dip and 'two' at release so the load is not rushed."),
    },
    "follow_through_hold_s": {
        "high": ("Relaxed-finish reps", "3 x 15 from 2 m",
                 "Hold the follow-through until the ball hits the rim, then relax; no rigid over-hold."),
        "low": ("Freeze the follow-through", "3 x 15, freeze 2 s each",
                "Hold the finish (hand in the rim) for a full two seconds before dropping the arm."),
    },
    "follow_through_wrist_angle": {
        "high": ("Cookie-jar wrist snaps", "3 x 20 flicks from a chair",
                 "Reach the hand into the cookie jar: a full wrist snap puts backspin on the ball."),
        "low": ("Soft-wrist form shots", "3 x 15 from 1 m",
                "Snap through the ball without forcing the fingers down; let the follow-through be loose."),
    },
    "jump_height": {
        "high": ("Rhythm shots, minimal hop", "3 x 15 from the elbow",
                 "Shoot on the way up, releasing before the peak of a small hop; save the legs for range."),
        "low": ("Leg-drive form shots", "3 x 15 from the free-throw line",
                "Drive through the balls of the feet so the legs supply the power and the arm stays soft."),
    },
    "elbow_angle_release": {
        "high": ("Soft-lockout finishes", "3 x 15 from 1.5 m",
                 "Extend fully but keep the follow-through loose; a hard lockout stiffens the wrist."),
        "low": ("Full-extension form shooting", "3 x 20 from 1 m",
                "Reach the elbow to full extension at release; the arm should look like a straight line to the rim."),
    },
}

# mechanics checks (mechanics.py) -> drills; "high" = above the coaching range, "low" = below it
DRILLS.update({
    "set_elbow": {
        "high": DRILLS["elbow_angle_set"]["high"],
        "low": DRILLS["elbow_angle_set"]["low"],
    },
    "dip_knee": {
        "high": DRILLS["knee_angle_min"]["high"],
        "low": DRILLS["knee_angle_min"]["low"],
    },
    "release_elbow": {
        "high": DRILLS["elbow_angle_release"]["high"],
        "low": DRILLS["elbow_angle_release"]["low"],
    },
    "release_shoulder": {
        "high": ("Set-point under the eye", "3 x 15 from 1.5 m",
                 "Keep the ball in front of the forehead, not behind the head, so the arm finishes up and out toward the rim."),
        "low": ("High-release one-hand form shooting", "3 x 20 from 1 m (3 ft)",
                "Finish with the elbow above the eyebrow: the upper arm should point up toward the rim, not straight ahead."),
    },
    "release_timing": {
        "high": ("Rhythm shots, minimal hop", "3 x 15 from the elbow",
                 "Release on the way up or at the top of the jump, never on the way down."),
        "low": ("Jump-then-release timing reps", "3 x 10 from the free-throw line",
                "Let the legs finish the jump: the ball leaves the hand as you reach the top, not early on the way up."),
    },
    "follow_through": {
        "high": DRILLS["follow_through_hold_s"]["high"],
        "low": DRILLS["follow_through_hold_s"]["low"],
    },
    "balance_lean": {
        "high": DRILLS["trunk_lean_release"]["high"],
        "low": DRILLS["trunk_lean_release"]["high"],
    },
    "balance_drift": {
        "high": ("Land-on-the-spot shots", "3 x 10 on a tape cross",
                 "Jump straight up and land on the same spot: the jump goes up, not forward or sideways."),
        "low": ("Land-on-the-spot shots", "3 x 10 on a tape cross",
                "Jump straight up and land on the same spot: the jump goes up, not forward or sideways."),
    },
})

GENERIC_MAINTENANCE = ("Form shooting warm-up", "50 makes from 1 m, one hand",
                       "Nothing major flagged - keep the daily close-range form routine to lock the pattern in.")


def drill_for(key: str, delta: float) -> tuple[str, str, str] | None:
    d = DRILLS.get(key)
    if not d:
        return None
    return d["high"] if delta > 0 else d["low"]


def coaching_plan(feedback: list[dict], max_items: int = 6) -> list[dict]:
    """Ordered drill plan from feedback items: mechanics findings first (biggest point losses), then
    style-match findings (majors first, then by |delta|). A drill is listed once."""
    order = {"major": 0, "minor": 1, "ok": 2}
    mech = [fb for fb in feedback if fb["severity"] != "ok" and fb.get("kind") == "mechanics"]
    mech.sort(key=lambda fb: -fb.get("points_lost", 0.0))
    style = [fb for fb in feedback if fb["severity"] != "ok" and fb.get("kind") != "mechanics"]
    style.sort(key=lambda fb: (order[fb["severity"]], -abs(fb["delta"])))
    plan, seen = [], set()
    for fb in mech + style:
        if len(plan) >= max_items:
            break
        d = drill_for(fb["key"], fb["delta"])
        if not d or d[0] in seen:
            continue
        seen.add(d[0])
        name, rx, cue = d
        plan.append(dict(key=fb["key"], kind=fb.get("kind", "style"), finding=fb["label"], severity=fb["severity"], delta=fb["delta"],
                         unit=fb["unit"], drill=name, prescription=rx, cue=cue))
    if not plan:
        name, rx, cue = GENERIC_MAINTENANCE
        plan.append(dict(key="maintain", finding="Form within tolerance", severity="ok", delta=0.0, unit="",
                         drill=name, prescription=rx, cue=cue))
    return plan
