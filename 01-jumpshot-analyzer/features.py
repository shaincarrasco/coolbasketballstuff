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

from pose import LM, MIRROR_PAIRS, PoseSequence, fill_missing

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
    """Savitzky-Golay smoothing along axis 0 (window ~ 0.2 s, odd, >= 5)."""
    T = x.shape[0]
    win = int(round(fps * 0.2)) | 1
    win = max(5, min(win, T - (1 - T % 2)))
    if T < 5 or win < 5:
        return x
    return savgol_filter(x, win, 2, axis=0, mode="interp")


def detect_shooting_side(px: np.ndarray) -> str:
    """The shooting arm is the wrist that gets highest (smallest y) at any point."""
    r = np.nanmin(px[:, LM.RIGHT_WRIST, 1])
    l = np.nanmin(px[:, LM.LEFT_WRIST, 1])
    return "right" if r <= l else "left"


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
    ears = np.nanmean(px[:, [LM.LEFT_EAR, LM.RIGHT_EAR], 0], axis=1)
    d = np.nanmedian(nose - ears)
    if abs(d) < 1e-3:
        s = SIDES[side]
        d = np.nanmedian(px[:, s["wrist"], 0] - px[:, s["shoulder"], 0])
    return 1.0 if d >= 0 else -1.0


# ------------------------------------------------------------- per frame

def per_frame_features(px: np.ndarray, side: str, fps: float) -> tuple[np.ndarray, float, float]:
    """Returns (features (T, F), torso_px, facing). Coordinates are smoothed first."""
    s = SIDES[side]
    p = smooth(px, fps)
    sh_mid = (p[:, LM.LEFT_SHOULDER] + p[:, LM.RIGHT_SHOULDER]) / 2
    hip_mid = (p[:, LM.LEFT_HIP] + p[:, LM.RIGHT_HIP]) / 2
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
    return feats, torso, facing


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

    def to_dict(self) -> dict:
        return asdict(self)


def detect_events(feats: np.ndarray, fps: float) -> ShotEvents:
    T = feats.shape[0]
    f = {n: feats[:, i] for i, n in enumerate(FEATURE_NAMES)}
    # wrist vertical velocity in torso/s (+ = upward, since wrist_height grows upward)
    v = np.gradient(f["wrist_height"]) * fps
    # Ordering used to anchor the shot: dip (deepest knee bend) -> elbow extension -> wrist at its top.
    #   1) wrist top  = first frame where the wrist is within 5 % of its highest point
    #   2) set point  = deepest knee bend before the wrist top (the loaded position)
    #   3) extension  = peak elbow angular velocity between set point and wrist top
    #      (peak wrist velocity alone also fires on the lift from waist to set point, and so does the
    #       elbow for an under-loaded shooter, hence the window)
    ve = np.gradient(f["elbow_angle"]) * fps
    wh = f["wrist_height"]
    top = int(np.argmax(wh >= wh.max() - 0.05 * (wh.max() - wh.min() + 1e-6)))
    top = max(top, 2)
    set_pt = int(np.argmin(f["knee_angle"][:top]))
    peak_e = set_pt + int(np.argmax(ve[set_pt:top + 1]))
    # release = first frame after the extension peak where the elbow is (nearly) locked out:
    # >= 160 deg, or within 4 deg of the arm's own maximum in the extension window when it never
    # gets there (velocity-decay criteria were too noise-sensitive). release_found is False when
    # the arm never reached 150 deg at all (no real extension -> the fallback frame is a guess).
    window_end = min(T - 1, max(top, peak_e + int(round(fps * 0.6))))
    arm_max = float(f["elbow_angle"][peak_e:window_end + 1].max())
    thr = min(160.0, arm_max - 4.0)
    rel = peak_e
    for i in range(peak_e, T):
        if f["elbow_angle"][i] >= thr:
            rel = i
            break
    rel = int(min(max(rel, 1), T - 1))
    release_found = bool(f["elbow_angle"][rel] >= 150.0)
    # peak upward wrist velocity within the set-point -> release window
    peak_v = int(set_pt + np.argmax(v[set_pt:rel + 1]))
    # onset = first frame where knee bend or wrist movement departs from the start pose
    start_knee = float(np.median(f["knee_angle"][: max(2, int(fps * 0.15))]))
    start_wh = float(np.median(f["wrist_height"][: max(2, int(fps * 0.15))]))
    onset = 0
    for i in range(set_pt):
        if (start_knee - f["knee_angle"][i]) > 4 or abs(f["wrist_height"][i] - start_wh) > 0.12:
            onset = i
            break
    onset = max(0, onset - 1)
    # follow-through: frames after release where wrist stays high and elbow extended
    rel_h = f["wrist_height"][rel]
    ft_end = rel
    for i in range(rel, T):
        if f["wrist_height"][i] >= 0.65 * rel_h and f["elbow_angle"][i] >= 140:
            ft_end = i
        else:
            break
    hold_s = (ft_end - rel) / fps
    ft = int(min(T - 1, rel + max(2, (ft_end - rel) // 2)))
    ft_end = int(min(T - 1, max(ft_end, rel + int(fps * 0.2))))
    return ShotEvents(onset=int(onset), set_point=set_pt, peak_wrist_velocity=peak_v, release=rel,
                      follow_through=ft, follow_through_end=ft_end, hold_s=float(hold_s), fps=float(fps),
                      release_found=bool(release_found))


# ------------------------------------------------------- time normalisation

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
        "knee_angle_min": float(f["knee_angle"].min()),
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
        "jump_height": float(np.max(f["hip_rise"])),
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
        flag("release_not_found", "major", "Release not found: the shooting arm never reached ~150 deg extension; the release frame is a guess.")
    if ev.set_point <= ev.onset or ev.release - ev.set_point < 2:
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


def shot_from_pose(seq: PoseSequence, name: str, side: str | None = None) -> Shot:
    if seq.n_frames < 8:
        raise ValueError(f"{name}: too few frames with pose ({seq.n_frames})")
    px = to_pixels(seq)
    side_forced = side is not None
    side = side or detect_shooting_side(px)
    feats, torso, facing = per_frame_features(px, side, seq.fps)
    ev = detect_events(feats, seq.fps)
    norm = resample(feats, ev.onset, ev.follow_through_end)
    summ = summarize(feats, ev, seq.fps)
    quality = assess_quality(seq, ev, side, side_forced)
    return Shot(name=name, side=side, facing=facing, fps=seq.fps, torso_px=torso, per_frame=feats, seq=norm,
                events=ev, summary=summ, landmarks_px=px, width=seq.width, height=seq.height,
                video_path=seq.video_path, source=seq.source, missing_frames=int(seq.missing_frames.sum()),
                quality=quality)


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
