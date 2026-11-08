"""
features.py — shot-mechanics features from 33-landmark pose sequences.

Per-frame features (all body-size invariant: angles in degrees, lengths in
torso-lengths, where torso = mean shoulder-midpoint -> hip-midpoint distance):

    elbow_angle    shoulder-elbow-wrist angle of the shooting arm (deg)
    knee_angle     hip-knee-ankle angle, shooting side (deg); 180 = straight
    hip_angle      shoulder-hip-knee angle (deg); 180 = fully extended / aligned
    trunk_lean     angle of the hip->shoulder vector from vertical (deg, + = forward)
    wrist_height   (nose_y - wrist_y) / torso; + = wrist above head
    wrist_forward  horizontal wrist offset from the shoulder along the facing
                   direction, / torso (+ = in front)
    elbow_tuck     horizontal elbow offset from the shoulder along the facing
                   direction, / torso (larger = elbow pushed out in front)
    wrist_angle    elbow-wrist-index angle (deg); drops after release = snap
    hip_rise       (hip_y_start - hip_y) / torso; jump-height proxy

Per-shot events: motion onset, set point (deepest dip), peak upward wrist
velocity, release, follow-through, follow-through end. Each shot is time-
normalised to N_FRAMES=60 samples between onset and follow-through end.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np
from scipy.signal import savgol_filter

from pose import LM, MIRROR_PAIRS, PoseSequence, fill_missing, mask_low_visibility

FEATURE_NAMES = ["elbow_angle", "knee_angle", "hip_angle", "trunk_lean", "wrist_height",
                 "wrist_forward", "elbow_tuck", "wrist_angle", "hip_rise"]
FEATURE_UNITS = {"elbow_angle": "deg", "knee_angle": "deg", "hip_angle": "deg", "trunk_lean": "deg",
                 "wrist_height": "torso", "wrist_forward": "torso", "elbow_tuck": "torso",
                 "wrist_angle": "deg", "hip_rise": "torso"}
N_FRAMES = 60

SIDES = {
    "right": dict(shoulder=LM.RIGHT_SHOULDER, elbow=LM.RIGHT_ELBOW, wrist=LM.RIGHT_WRIST, index=LM.RIGHT_INDEX,
                  hip=LM.RIGHT_HIP, knee=LM.RIGHT_KNEE, ankle=LM.RIGHT_ANKLE),
    "left": dict(shoulder=LM.LEFT_SHOULDER, elbow=LM.LEFT_ELBOW, wrist=LM.LEFT_WRIST, index=LM.LEFT_INDEX,
                 hip=LM.LEFT_HIP, knee=LM.LEFT_KNEE, ankle=LM.LEFT_ANKLE),
}


# ------------------------------------------------------------------ helpers

def angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Angle at b formed by a-b-c, vectorised over the leading axis. Inputs (T, 2|3)."""
    v1 = a - b
    v2 = c - b
    cosang = np.sum(v1 * v2, axis=-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))


def to_pixels(seq: PoseSequence) -> np.ndarray:
    """(T, 33, 2) pixel coordinates (x right, y down) with gaps filled."""
    filled, _ = fill_missing(seq.landmarks)
    px = filled[:, :, :2].copy()
    px[:, :, 0] *= seq.width
    px[:, :, 1] *= seq.height
    return px


def smooth(x: np.ndarray, fps: float) -> np.ndarray:
    """Savitzky-Golay smoothing along axis 0 (window ~ 0.2 s, odd, >= 5). Columns that are
    entirely NaN (a landmark never observed) stay NaN and do not affect the others."""
    T = x.shape[0]
    win = int(round(fps * 0.2)) | 1
    win = max(5, min(win, T - (1 - T % 2)))
    if T < 5 or win < 5:
        return x
    bad = np.isnan(x).any(axis=0)
    if bad.any():
        x = np.where(bad[None], 0.0, x).astype(x.dtype)
        out = savgol_filter(x, win, 2, axis=0, mode="interp")
        out[:, bad] = np.nan
        return out
    return savgol_filter(x, win, 2, axis=0, mode="interp")


def pair_mid(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Midpoint of two (T, 2) landmark tracks; falls back to the one that exists if the other
    was never observed (e.g. the far shoulder hidden for the whole side-on clip)."""
    if np.isnan(a).all():
        return b.copy()
    if np.isnan(b).all():
        return a.copy()
    return (a + b) / 2


def detect_shooting_side(px: np.ndarray) -> str:
    """The shooting arm is the wrist that gets highest (smallest y) at any point."""
    rw, lw = px[:, LM.RIGHT_WRIST, 1], px[:, LM.LEFT_WRIST, 1]
    if np.isnan(lw).all():
        return "right"
    if np.isnan(rw).all():
        return "left"
    return "right" if np.nanmin(rw) <= np.nanmin(lw) else "left"


def mirror_landmarks(px: np.ndarray, width: float) -> np.ndarray:
    """Mirror a (T, 33, 2) pixel landmark array left<->right: flips x about the frame centre
    and swaps the left/right landmark indices so a left-handed shooter becomes a canonical
    right-handed one. Used for mirror normalisation (see canonical_landmarks)."""
    out = px.copy()
    out[:, :, 0] = width - out[:, :, 0]
    for a, b in MIRROR_PAIRS:
        out[:, [a, b]] = out[:, [b, a]]
    return out


def canonical_landmarks(px: np.ndarray, width: float, side: str, facing: float) -> tuple[np.ndarray, str, float, bool]:
    """Mirror-normalise handedness: a left-handed shooter is mirrored into a right-handed one.
    Returns (landmarks, side, facing, mirrored) after normalisation (a mirror flips facing too).
    Features are already side/facing invariant (they use the detected shooting side and the facing
    sign); this is for overlays and the scrubber so a left-handed user and a right-handed
    reference can be drawn on top of each other. Facing is matched separately by the caller."""
    if side == "left":
        return mirror_landmarks(px, width), "right", -facing, True
    return px, side, facing, False


def facing_direction(px: np.ndarray, side: str) -> float:
    """+1 if the shooter faces image-right, -1 if image-left (from nose vs ear / shoulder-wrist)."""
    nose = px[:, LM.NOSE, 0]
    le, re_ = px[:, LM.LEFT_EAR, 0], px[:, LM.RIGHT_EAR, 0]
    ears = pair_mid(le[:, None], re_[:, None])[:, 0]
    d = np.nanmedian(nose - ears) if not np.isnan(ears).all() else np.nan
    if not np.isfinite(d) or abs(d) < 1e-3:
        s = SIDES[side]
        d = np.nanmedian(px[:, s["wrist"], 0] - px[:, s["shoulder"], 0])
    return 1.0 if d >= 0 else -1.0


# ------------------------------------------------------------- per frame

def per_frame_features(px: np.ndarray, side: str, fps: float, extras: bool = False):
    """Returns (features (T, F), torso_px, facing) — plus a dict of extra per-frame series when
    extras=True (used by the event detector and the mechanics score). Coordinates are smoothed first."""
    s = SIDES[side]
    p = smooth(px, fps)
    sh_mid = pair_mid(p[:, LM.LEFT_SHOULDER], p[:, LM.RIGHT_SHOULDER])
    hip_mid = pair_mid(p[:, LM.LEFT_HIP], p[:, LM.RIGHT_HIP])
    torso = float(np.nanmedian(np.linalg.norm(sh_mid - hip_mid, axis=1)))
    torso = max(torso, 1e-3)
    facing = facing_direction(px, side)

    shoulder, elbow, wrist, index = p[:, s["shoulder"]], p[:, s["elbow"]], p[:, s["wrist"]], p[:, s["index"]]
    hip, knee, ankle = p[:, s["hip"]], p[:, s["knee"]], p[:, s["ankle"]]
    nose = p[:, LM.NOSE]

    elbow_angle = angle_deg(shoulder, elbow, wrist)
    knee_angle = angle_deg(hip, knee, ankle)
    hip_angle = angle_deg(shoulder, hip, knee)
    trunk_vec = sh_mid - hip_mid                      # image coords: y down
    trunk_lean = np.degrees(np.arctan2(facing * trunk_vec[:, 0], -trunk_vec[:, 1]))
    wrist_height = (nose[:, 1] - wrist[:, 1]) / torso
    wrist_forward = facing * (wrist[:, 0] - shoulder[:, 0]) / torso
    elbow_tuck = facing * (elbow[:, 0] - shoulder[:, 0]) / torso
    wrist_angle = angle_deg(elbow, wrist, index)
    hip_y0 = float(np.nanmedian(hip_mid[: max(3, int(fps * 0.2)), 1]))
    hip_rise = (hip_y0 - hip_mid[:, 1]) / torso

    feats = np.stack([elbow_angle, knee_angle, hip_angle, trunk_lean, wrist_height,
                      wrist_forward, elbow_tuck, wrist_angle, hip_rise], axis=1).astype(np.float32)
    if not extras:
        return feats, torso, facing
    # shoulder flexion: angle between the trunk's downward axis (shoulder-mid -> hip-mid) and the upper arm
    shoulder_flex = angle_deg(hip_mid - sh_mid + shoulder, shoulder, elbow)
    lsh, rsh = p[:, LM.LEFT_SHOULDER, 0], p[:, LM.RIGHT_SHOULDER, 0]
    sw = np.abs(lsh - rsh) / torso
    ex = dict(wrist_above_shoulder=((shoulder[:, 1] - wrist[:, 1]) / torso).astype(np.float32),
              shoulder_flex=shoulder_flex.astype(np.float32),
              hip_x=hip_mid[:, 0].astype(np.float32), hip_y=hip_mid[:, 1].astype(np.float32),
              shoulder_width=sw.astype(np.float32))
    return feats, torso, facing, ex


# ---------------------------------------------------------------- events

@dataclass
class ShotEvents:
    onset: int
    set_point: int
    peak_wrist_velocity: int
    release: int
    follow_through: int
    follow_through_end: int
    hold_s: float
    fps: float
    release_found: bool = True       # False when the lock-out / velocity-decay criterion never fired
    peak_extension: int = -1         # frame of peak elbow extension velocity (the release anchor)
    set_method: str = "legacy"       # 'elbow_min' | 'knee_min' | 'legacy' (v1 fallback)
    baseline_start: int = 0          # first frame of the stance baseline (onset / hip-rise reference)
    release_sub: float = -1.0        # sub-frame release position (threshold crossing), for timing

    def to_dict(self) -> dict:
        return asdict(self)


# Event-detector v2 constants (real phone footage). The detector is anchored on the release:
#   1) extension = peak elbow angular velocity over frames where the ball is *loaded* (shooting wrist
#      above the shoulder or within LOAD_TORSO of it) and the wrist is rising, after the warm-up —
#      never the dribble / walk-in, never the arm coming back down — and that reaches an extended arm
#      (>= RELEASE_FOUND_DEG) with the wrist above the head within REACH_S (not the waist-to-set lift)
#   2) release   = first frame after that peak with the elbow >= RELEASE_DEG (or arm max - 4)
#   3) set point = last elbow-flexion minimum in the loaded zone within SET_WINDOW_S before release
#      (a real minimum: flexed >= MIN_FLEX_DEG within FLEX_LOOKBACK_S before it, extends >= MIN_EXT_DEG
#      after it); if there is none (one continuous extension), the deepest knee bend in that window
#   4) onset     = first departure from the stance pose, searched from ONSET_LOOKBACK_S before set
# The v1 path (wrist top -> deepest knee before it) is kept as a fallback when no loaded frame exists.
LOAD_TORSO = 0.3
WARMUP_S = 0.2
SET_WINDOW_S = 0.8
FLEX_LOOKBACK_S = 0.3
MIN_FLEX_DEG = 8.0
MIN_EXT_DEG = 20.0
ONSET_LOOKBACK_S = 1.0
REACH_S = 0.4


RELEASE_DEG = 150.0        # v1 used 160: on real footage the ball leaves the hand at ~150 deg (2-D)
RELEASE_FOUND_DEG = 140.0  # the arm must extend at least this far after the extension peak


def _release_from(elbow: np.ndarray, peak_e: int, window_end: int, T: int) -> tuple[int, bool, float]:
    """Release = first frame after the extension peak where the elbow reaches RELEASE_DEG, or within
    4 deg of the arm's own maximum in the extension window when it never gets there."""
    arm_max = float(elbow[peak_e:window_end + 1].max())
    thr = min(RELEASE_DEG, arm_max - 4.0)
    rel = peak_e
    for i in range(peak_e, T):
        if elbow[i] >= thr:
            rel = i
            break
    rel = int(min(max(rel, 1), T - 1))
    # sub-frame release time: linear interpolation of the threshold crossing (for timing metrics)
    sub = float(rel)
    a, b = float(elbow[rel - 1]), float(elbow[rel])
    if rel > peak_e and a < thr <= b and b > a:
        sub = rel - 1 + (thr - a) / (b - a)
    return rel, bool(arm_max >= RELEASE_FOUND_DEG), sub


CUT_TORSO = 0.8           # hip-mid moving more than this (torso lengths) in one frame = scene cut


def scene_cuts(px: np.ndarray) -> np.ndarray:
    """Frames that start a new scene (an edited clip / camera jump): the hip midpoint of the gap-filled
    pixel landmarks moves more than CUT_TORSO torso lengths from the previous frame."""
    T = px.shape[0]
    if T < 2:
        return np.zeros(0, np.int64)
    sh = pair_mid(px[:, LM.LEFT_SHOULDER], px[:, LM.RIGHT_SHOULDER])
    hp = pair_mid(px[:, LM.LEFT_HIP], px[:, LM.RIGHT_HIP])
    torso = float(np.nanmedian(np.linalg.norm(sh - hp, axis=1)))
    if not np.isfinite(torso) or torso <= 1e-6:
        return np.zeros(0, np.int64)
    step = np.linalg.norm(np.diff(hp, axis=0), axis=1) / torso
    return (np.flatnonzero(step > CUT_TORSO) + 1).astype(np.int64)


def detect_events(feats: np.ndarray, fps: float, wrist_above_shoulder: np.ndarray | None = None,
                  cuts: np.ndarray | None = None) -> ShotEvents:
    T = feats.shape[0]
    f = {n: feats[:, i] for i, n in enumerate(FEATURE_NAMES)}
    elbow, knee, wh = f["elbow_angle"], f["knee_angle"], f["wrist_height"]
    # wrist vertical velocity in torso/s (+ = upward, since wrist_height grows upward)
    v = np.gradient(wh) * fps
    ve = np.gradient(elbow) * fps
    warm = min(max(T - 3, 0), int(round(fps * WARMUP_S)))
    loaded = None
    if wrist_above_shoulder is not None:
        loaded = np.asarray(wrist_above_shoulder) >= -LOAD_TORSO
        loaded[:warm] = False
    method = "legacy"
    # the shooting extension: ball loaded AND the wrist moving up (excludes the arm coming back down)
    rising = loaded & (v > 0) if loaded is not None else None
    if rising is not None and rising.any():
        # ... that leads to a shot: the arm reaches RELEASE_FOUND_DEG with the wrist above the head
        # within REACH_S (excludes the lift from the waist to the set point)
        reach_n = int(round(fps * REACH_S))
        high = (elbow >= RELEASE_FOUND_DEG) & (wh > 0)
        nxt = np.full(T + 1, T, dtype=np.int64)            # next frame index >= i where `high`
        for i in range(T - 1, -1, -1):
            nxt[i] = i if high[i] else nxt[i + 1]
        good = rising & (nxt[:T] - np.arange(T) <= reach_n)
        cand = np.flatnonzero(good if good.any() else rising)
        peak_e = int(cand[np.argmax(ve[cand])])
        window_end = min(T - 1, peak_e + int(round(fps * 0.6)))
        rel, release_found, rel_sub = _release_from(elbow, peak_e, window_end, T)
        scene0 = int(max([c for c in (cuts if cuts is not None else []) if c <= peak_e], default=0))
        lo = max(warm, scene0, rel - int(round(fps * SET_WINDOW_S)))
        lb = max(1, int(round(fps * FLEX_LOOKBACK_S)))
        set_pt = -1
        for m in range(min(peak_e, T - 2), lo - 1, -1):
            if m < 1 or not loaded[m]:
                continue
            if not (elbow[m] <= elbow[m - 1] and elbow[m] <= elbow[m + 1]):
                continue
            flex_before = float(elbow[max(0, m - lb):m].max()) - float(elbow[m])
            ext_after = float(elbow[m:rel + 1].max()) - float(elbow[m])
            if flex_before >= MIN_FLEX_DEG and ext_after >= MIN_EXT_DEG:
                set_pt = m
                method = "elbow_min"
                break
        if set_pt < 0:
            hi = max(lo + 1, peak_e)
            set_pt = int(lo + np.argmin(knee[lo:hi]))
            method = "knee_min"
        s0 = max(0, scene0, set_pt - int(round(fps * ONSET_LOOKBACK_S)))
    else:
        # v1: dip (deepest knee bend) -> elbow extension -> wrist at its top
        top = int(np.argmax(wh >= wh.max() - 0.05 * (wh.max() - wh.min() + 1e-6)))
        top = max(top, 2)
        set_pt = int(np.argmin(knee[:top]))
        peak_e = set_pt + int(np.argmax(ve[set_pt:top + 1]))
        window_end = min(T - 1, max(top, peak_e + int(round(fps * 0.6))))
        rel, release_found, rel_sub = _release_from(elbow, peak_e, window_end, T)
        s0 = 0
    # peak upward wrist velocity within the set-point -> release window
    peak_v = int(set_pt + np.argmax(v[set_pt:rel + 1])) if rel >= set_pt else int(rel)
    # onset = first frame where knee bend or wrist movement departs from the start pose
    nb = max(2, int(fps * 0.15))
    start_knee = float(np.median(knee[s0:s0 + nb]))
    start_wh = float(np.median(wh[s0:s0 + nb]))
    onset = s0
    for i in range(s0, set_pt):
        if (start_knee - knee[i]) > 4 or abs(wh[i] - start_wh) > 0.12:
            onset = i
            break
    onset = max(s0, onset - 1) if onset > s0 else s0
    # follow-through: frames after release where wrist stays high and elbow extended
    rel_h = wh[rel]
    ft_end = rel
    for i in range(rel, T):
        if wh[i] >= 0.65 * rel_h and elbow[i] >= 140:
            ft_end = i
        else:
            break
    hold_s = (ft_end - rel) / fps
    ft = int(min(T - 1, rel + max(2, (ft_end - rel) // 2)))
    ft_end = int(min(T - 1, max(ft_end, rel + int(fps * 0.2))))
    return ShotEvents(onset=int(onset), set_point=int(set_pt), peak_wrist_velocity=peak_v, release=int(rel),
                      follow_through=ft, follow_through_end=ft_end, hold_s=float(hold_s), fps=float(fps),
                      release_found=bool(release_found), peak_extension=int(peak_e), set_method=method,
                      baseline_start=int(s0), release_sub=float(rel_sub))


# ------------------------------------------------------- time normalisation

def rebaseline_hip_rise(feats: np.ndarray, ex: dict, ev: ShotEvents, torso: float, fps: float) -> None:
    """hip_rise is measured from the stance baseline that starts at ev.baseline_start (frame 0 for a
    trimmed clip, so v1 results are unchanged); recomputed in place when the baseline moved."""
    if ev.baseline_start <= 0:
        return
    hy = ex["hip_y"]
    s0 = ev.baseline_start
    y0 = float(np.nanmedian(hy[s0:s0 + max(3, int(fps * 0.2))]))
    feats[:, FEATURE_NAMES.index("hip_rise")] = ((y0 - hy) / torso).astype(np.float32)


def resample(feats: np.ndarray, start: int, end: int, n: int = N_FRAMES) -> np.ndarray:
    """Linearly resample feats[start:end+1] to n samples."""
    end = max(end, start + 1)
    seg = feats[start:end + 1]
    t_src = np.linspace(0, 1, seg.shape[0])
    t_dst = np.linspace(0, 1, n)
    return np.stack([np.interp(t_dst, t_src, seg[:, j]) for j in range(seg.shape[1])], axis=1).astype(np.float32)


def event_phase(ev: ShotEvents, frame: int) -> float:
    """Map a raw frame index to normalised time in [0, 1] of the shot window."""
    return float(np.clip((frame - ev.onset) / max(1, ev.follow_through_end - ev.onset), 0, 1))


# ------------------------------------------------------------- per shot

@dataclass
class Shot:
    name: str
    side: str
    facing: float
    fps: float
    torso_px: float
    per_frame: np.ndarray            # (T, F)
    seq: np.ndarray                  # (N_FRAMES, F) time-normalised
    events: ShotEvents
    summary: dict = field(default_factory=dict)
    landmarks_px: np.ndarray | None = None   # (T, 33, 2) pixel coords, filled
    width: int = 0
    height: int = 0
    video_path: str = ""
    source: str = ""
    missing_frames: int = 0
    quality: dict = field(default_factory=dict)
    mechanics: list = field(default_factory=list)     # mechanics.mechanics_profile items
    mechanics_score: float | None = None               # 0..100 headline score (None if unmeasured)
    extras: dict = field(default_factory=dict)        # extra per-frame series (not saved)

    def to_npz_dict(self) -> dict:
        import json
        return dict(name=self.name, side=self.side, facing=self.facing, fps=self.fps, torso_px=self.torso_px,
                    per_frame=self.per_frame, seq=self.seq, events=json.dumps(self.events.to_dict()),
                    summary=json.dumps(self.summary), landmarks_px=self.landmarks_px, width=self.width,
                    height=self.height, video_path=self.video_path, source=self.source,
                    missing_frames=self.missing_frames, feature_names=np.array(FEATURE_NAMES),
                    quality=json.dumps(self.quality))


def summarize(feats: np.ndarray, ev: ShotEvents, fps: float) -> dict:
    f = {n: feats[:, i] for i, n in enumerate(FEATURE_NAMES)}
    rel, sp, ft = ev.release, ev.set_point, ev.follow_through
    return {
        "elbow_angle_set": float(f["elbow_angle"][sp]),
        "elbow_angle_release": float(f["elbow_angle"][rel]),
        "elbow_tuck_set": float(f["elbow_tuck"][sp]),
        "knee_angle_min": float(f["knee_angle"][ev.onset:rel + 1].min()),
        "knee_angle_set": float(f["knee_angle"][sp]),
        "hip_angle_set": float(f["hip_angle"][sp]),
        "trunk_lean_release": float(f["trunk_lean"][rel]),
        "release_height": float(f["wrist_height"][rel]),
        "release_forward": float(f["wrist_forward"][rel]),
        "release_time_s": float((rel - sp) / fps),          # load-to-release
        "release_time_from_onset_s": float((rel - ev.onset) / fps),
        "peak_wrist_velocity_torso_per_s": float((np.gradient(f["wrist_height"]) * fps)[ev.peak_wrist_velocity]),
        "follow_through_wrist_angle": float(f["wrist_angle"][ft]),
        "follow_through_hold_s": float(ev.hold_s),
        "jump_height": float(np.max(f["hip_rise"][ev.onset:ev.follow_through_end + 1])),
        "shot_duration_s": float((ev.follow_through_end - ev.onset) / fps),
    }


MIN_GOOD_FRAMES = 24          # < 0.8 s at 30 fps: too short to hold stance -> follow-through
BODY_LANDMARKS = list(range(11, 33))   # shoulders .. feet (face landmarks excluded)


def assess_quality(seq: PoseSequence, ev: ShotEvents, side: str, side_forced: bool) -> dict:
    """Confidence / quality flags shown in the report. Each flag: (code, severity, message)."""
    vis = np.nan_to_num(seq.visibility[:, BODY_LANDMARKS], nan=0.0)
    body_vis_pct = float((vis >= 0.5).mean() * 100.0)
    s = SIDES[side]
    arm_vis = np.nan_to_num(seq.visibility[:, [s["shoulder"], s["elbow"], s["wrist"]]], nan=0.0)
    arm_vis_pct = float((arm_vis >= 0.5).mean() * 100.0)
    missing = int(seq.missing_frames.sum())
    n = seq.n_frames
    flags: list[dict] = []

    def flag(code, sev, msg):
        flags.append(dict(code=code, severity=sev, message=msg))

    if n < MIN_GOOD_FRAMES:
        flag("few_frames", "major", f"Only {n} frames with pose ({n / seq.fps:.1f} s) - trim to ~2 s around the shot at >= 30 fps.")
    if body_vis_pct < 70:
        flag("low_visibility", "major", f"Body landmark visibility {body_vis_pct:.0f} % - the whole body should be in frame.")
    elif body_vis_pct < 85:
        flag("low_visibility", "minor", f"Body landmark visibility {body_vis_pct:.0f} % - some joints were guessed/interpolated.")
    if arm_vis_pct < 85:
        flag("arm_occluded", "minor", f"Shooting-arm visibility {arm_vis_pct:.0f} % - ball/defender may be covering the wrist.")
    if missing > 0.1 * n:
        flag("gaps", "minor", f"{missing} of {n} frames had no pose and were gap-filled.")
    if not ev.release_found:
        flag("release_not_found", "major", "Release not found: the shooting arm never reached ~140 deg extension; the release frame is a guess.")
    warm = int(round(seq.fps * WARMUP_S))
    if ev.set_point <= ev.onset or ev.release - ev.set_point < 2 or (ev.set_method != "legacy" and ev.set_point <= warm):
        flag("events_degenerate", "major", "Set point / release are too close together - event detection is unreliable on this clip.")
    if ev.follow_through_end >= n - 1:
        flag("clip_cut", "minor", "Clip ends during the follow-through; hold time is a lower bound.")
    worst = "ok"
    for f in flags:
        if f["severity"] == "major":
            worst = "major"
        elif f["severity"] == "minor" and worst == "ok":
            worst = "minor"
    return dict(body_visibility_pct=round(body_vis_pct, 1), arm_visibility_pct=round(arm_vis_pct, 1),
                n_frames=int(n), missing_frames=missing, release_found=bool(ev.release_found),
                side=side, side_detected=not side_forced, confidence=worst, flags=flags)


GLITCH_RATIO = (0.6, 1.6)     # leg-bone / torso length, relative to its clip median


def glitch_frames(seq: PoseSequence) -> np.ndarray:
    """Frames whose skeleton is implausible: any visible thigh or shank whose length relative to the
    torso is outside GLITCH_RATIO x its clip median (MediaPipe locking onto the wrong limbs, typically
    in the first frames of a clip). Scale-invariant, so walking toward the camera is fine."""
    T = seq.n_frames
    px = seq.landmarks[:, :, :2].astype(np.float64) * np.array([seq.width, seq.height], np.float64)
    sh = np.nanmean(np.stack([px[:, LM.LEFT_SHOULDER], px[:, LM.RIGHT_SHOULDER]]), axis=0) if T else px[:, 0]
    hp = np.nanmean(np.stack([px[:, LM.LEFT_HIP], px[:, LM.RIGHT_HIP]]), axis=0) if T else px[:, 0]
    torso = np.linalg.norm(sh - hp, axis=1)
    bad = np.zeros(T, bool)
    for a, b in ((LM.LEFT_HIP, LM.LEFT_KNEE), (LM.LEFT_KNEE, LM.LEFT_ANKLE),
                 (LM.RIGHT_HIP, LM.RIGHT_KNEE), (LM.RIGHT_KNEE, LM.RIGHT_ANKLE)):
        r = np.linalg.norm(px[:, a] - px[:, b], axis=1) / np.where(torso > 1e-6, torso, np.nan)
        ok = np.isfinite(r)
        if ok.sum() < 8:
            continue
        rel = r / np.median(r[ok])
        bad |= ok & ((rel < GLITCH_RATIO[0]) | (rel > GLITCH_RATIO[1]))
    return bad


def missing_required(px: np.ndarray, side: str) -> list[str]:
    """Landmarks the features need that were never observed in the clip (after the visibility mask)."""
    s = SIDES[side]
    never = lambda i: bool(np.isnan(px[:, i, 0]).all())
    out = [k for k in ("shoulder", "elbow", "wrist", "index", "hip", "knee", "ankle") if never(s[k])]
    if never(LM.NOSE):
        out.append("nose")
    return out


def shot_from_pose(seq: PoseSequence, name: str, side: str | None = None) -> Shot:
    seq = mask_low_visibility(seq)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        glitch = glitch_frames(seq)
    if glitch.any():
        lms = seq.landmarks.copy()
        lms[glitch] = np.nan
        seq = PoseSequence(lms, seq.visibility, seq.fps, seq.width, seq.height, seq.source, seq.video_path, seq.meta)
    n_pose = int((~seq.missing_frames).sum())
    if n_pose < 8:
        raise ValueError(f"{name}: too few frames with pose ({n_pose})")
    px = to_pixels(seq)
    side_forced = side is not None
    side = side or detect_shooting_side(px)
    miss = missing_required(px, side)
    if miss:
        raise ValueError(f"{name}: {side}-side landmarks never visible: {', '.join(miss)}")
    feats, torso, facing, ex = per_frame_features(px, side, seq.fps, extras=True)
    ev = detect_events(feats, seq.fps, ex["wrist_above_shoulder"], scene_cuts(px))
    rebaseline_hip_rise(feats, ex, ev, torso, seq.fps)
    norm = resample(feats, ev.onset, ev.follow_through_end)
    summ = summarize(feats, ev, seq.fps)
    quality = assess_quality(seq, ev, side, side_forced)
    from mechanics import mechanics_profile
    mech = mechanics_profile(feats, ex, ev, seq, side, quality, torso)
    return Shot(name=name, side=side, facing=facing, fps=seq.fps, torso_px=torso, per_frame=feats, seq=norm,
                events=ev, summary=summ, landmarks_px=px, width=seq.width, height=seq.height,
                video_path=seq.video_path, source=seq.source, missing_frames=int(seq.missing_frames.sum()),
                quality=quality, mechanics=mech["items"], mechanics_score=mech["score"], extras=ex)


def save_shot(path: str, shot: Shot) -> None:
    np.savez_compressed(path, **shot.to_npz_dict())


def load_shot(path: str) -> Shot:
    import json
    d = np.load(path, allow_pickle=False)
    ev = ShotEvents(**json.loads(str(d["events"])))
    return Shot(name=str(d["name"]), side=str(d["side"]), facing=float(d["facing"]), fps=float(d["fps"]),
                torso_px=float(d["torso_px"]), per_frame=d["per_frame"], seq=d["seq"], events=ev,
                summary=json.loads(str(d["summary"])), landmarks_px=d["landmarks_px"], width=int(d["width"]),
                height=int(d["height"]), video_path=str(d["video_path"]), source=str(d["source"]),
                missing_frames=int(d["missing_frames"]))
