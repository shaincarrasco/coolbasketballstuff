"""
demo_data.py — procedural synthetic jumpshot clips.

No real basketball footage is available (and none is downloaded), so this module
animates a 2-D side-view humanoid shooter with OpenCV. Mechanics are
parameterised (elbow set angle, knee bend, release height, timing, follow-through)
so we can generate several "reference shooters" with distinct, known mechanics and
a "user" shot with deliberate flaws.

For every MP4 it also writes a sidecar `<clip>.landmarks.npz` containing the exact
joint positions in MediaPipe's 33-landmark layout (normalised image coordinates,
plus visibility). `pose.py` can consume that sidecar via `--landmarks-from-synthetic`
when no MediaPipe pose model file is available in the environment.

Nothing here is modelled on, or named after, a real player.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, asdict

import cv2
import numpy as np

from pose import NUM_LANDMARKS, LM, save_landmarks

# ---------------------------------------------------------------- parameters

@dataclass
class ShotParams:
    name: str
    slug: str
    # mechanics
    elbow_set_deg: float = 90.0      # elbow angle at set point (deeper = more loaded)
    knee_min_deg: float = 135.0      # minimum knee angle at bottom of dip (smaller = deeper)
    hip_min_deg: float = 150.0       # shoulder-hip-knee angle at dip (trunk fold)
    release_shoulder_deg: float = 168.0  # upper-arm elevation at release (180 = straight up)
    release_elbow_deg: float = 172.0 # elbow extension at release
    set_shoulder_deg: float = 95.0   # upper-arm elevation at set point
    jump_height_m: float = 0.25
    t_dip: float = 0.75              # time of deepest dip (s)
    t_release: float = 1.25          # time of release (s)
    hold_s: float = 0.45             # follow-through hold after release (s)
    trunk_lean_deg: float = 4.0      # forward lean at release (deg from vertical)
    wrist_snap_deg: float = 70.0     # wrist flexion after release
    duration_s: float = 2.0
    fps: int = 30
    noise_px: float = 1.2            # simulated landmark jitter in sidecar
    seed: int = 0

    def to_json(self) -> dict:
        return asdict(self)


DEFAULT_REFERENCES = [
    ShotParams(
        name="Reference A — high release", slug="ref_a_high_release",
        elbow_set_deg=88, knee_min_deg=140, hip_min_deg=158,
        release_shoulder_deg=178, release_elbow_deg=176, set_shoulder_deg=105,
        jump_height_m=0.30, t_dip=0.78, t_release=1.30, hold_s=0.50, trunk_lean_deg=2.0, seed=1,
    ),
    ShotParams(
        name="Reference B — deep knee bend", slug="ref_b_deep_knee_bend",
        elbow_set_deg=84, knee_min_deg=112, hip_min_deg=138,
        release_shoulder_deg=165, release_elbow_deg=172, set_shoulder_deg=92,
        jump_height_m=0.38, t_dip=0.85, t_release=1.40, hold_s=0.42, trunk_lean_deg=7.0, seed=2,
    ),
    ShotParams(
        name="Reference C — quick release", slug="ref_c_quick_release",
        elbow_set_deg=96, knee_min_deg=145, hip_min_deg=160,
        release_shoulder_deg=160, release_elbow_deg=170, set_shoulder_deg=90,
        jump_height_m=0.18, t_dip=0.58, t_release=0.92, hold_s=0.35, trunk_lean_deg=5.0, seed=3,
    ),
]

DEFAULT_USER = ShotParams(
    name="User shot", slug="user_shot",
    # flaws relative to Reference A: flared/under-loaded elbow, shallow dip,
    # lower + slightly late release, short follow-through.
    elbow_set_deg=104, knee_min_deg=150, hip_min_deg=160,
    release_shoulder_deg=158, release_elbow_deg=165, set_shoulder_deg=98,
    jump_height_m=0.17, t_dip=0.82, t_release=1.42, hold_s=0.22, trunk_lean_deg=9.0,
    noise_px=1.8, seed=7,
)

# Earlier "practice sessions" for the progress panel: the same shooter, mechanics drifting
# from clearly worse toward the current clip (DEFAULT_USER). Labelled as demo in history.jsonl.
DEMO_SESSIONS = [
    ShotParams(name="Session 1 (demo)", slug="session_01", elbow_set_deg=112, knee_min_deg=156, hip_min_deg=163,
               release_shoulder_deg=150, release_elbow_deg=160, set_shoulder_deg=96, jump_height_m=0.13,
               t_dip=0.86, t_release=1.52, hold_s=0.15, trunk_lean_deg=12.0, noise_px=2.0, seed=11),
    ShotParams(name="Session 2 (demo)", slug="session_02", elbow_set_deg=108, knee_min_deg=152, hip_min_deg=162,
               release_shoulder_deg=154, release_elbow_deg=162, set_shoulder_deg=97, jump_height_m=0.15,
               t_dip=0.84, t_release=1.48, hold_s=0.18, trunk_lean_deg=10.0, noise_px=1.9, seed=12),
    ShotParams(name="Session 3 (demo)", slug="session_03", elbow_set_deg=110, knee_min_deg=149, hip_min_deg=160,
               release_shoulder_deg=152, release_elbow_deg=164, set_shoulder_deg=99, jump_height_m=0.16,
               t_dip=0.80, t_release=1.40, hold_s=0.20, trunk_lean_deg=11.0, noise_px=1.8, seed=13),
    ShotParams(name="Session 4 (demo)", slug="session_04", elbow_set_deg=105, knee_min_deg=151, hip_min_deg=161,
               release_shoulder_deg=157, release_elbow_deg=165, set_shoulder_deg=98, jump_height_m=0.17,
               t_dip=0.82, t_release=1.44, hold_s=0.21, trunk_lean_deg=9.5, noise_px=1.8, seed=14),
]

# ---------------------------------------------------------------- kinematics

SEG = dict(torso=0.55, neck=0.07, head_r=0.11, upper_arm=0.30, forearm=0.27,
           hand=0.17, thigh=0.46, shank=0.44, foot=0.22, shoulder_w=0.20, hip_w=0.16)

W, H = 480, 640
PPM = 215.0            # pixels per metre
GROUND_Y = 600
CX = 215


def _smoothstep(a: float, b: float, t: float) -> float:
    if t <= a:
        return 0.0
    if t >= b:
        return 1.0
    u = (t - a) / (b - a)
    return u * u * (3 - 2 * u)


def _keyframe(t: float, keys: list[tuple[float, float]]) -> float:
    """Smooth piecewise interpolation through (time, value) keyframes."""
    if t <= keys[0][0]:
        return keys[0][1]
    for (t0, v0), (t1, v1) in zip(keys[:-1], keys[1:]):
        if t <= t1:
            return v0 + (v1 - v0) * _smoothstep(t0, t1, t)
    return keys[-1][1]


def pose_at(p: ShotParams, t: float) -> dict[str, np.ndarray]:
    """Return 2-D joint positions in metres (x forward, y up), ground at y=0."""
    t_land = p.t_release + 0.32
    t_end = p.duration_s
    t_ft_end = min(p.t_release + p.hold_s, t_end - 0.05)
    t_rise = p.t_dip + 0.55 * (p.t_release - p.t_dip)     # mid-extension

    knee = _keyframe(t, [(0.0, 172), (0.25, 170), (p.t_dip, p.knee_min_deg), (t_rise, 165),
                         (p.t_release - 0.05, 178), (t_land, 150), (t_land + 0.25, 168), (t_end, 172)])
    hip = _keyframe(t, [(0.0, 172), (0.25, 170), (p.t_dip, p.hip_min_deg), (t_rise, 168),
                        (p.t_release - 0.05, 180 - p.trunk_lean_deg), (t_land, 158),
                        (t_land + 0.25, 170), (t_end, 172)])
    # shooting arm
    sh_r = _keyframe(t, [(0.0, 25), (0.3, 35), (p.t_dip, p.set_shoulder_deg),
                         (p.t_release, p.release_shoulder_deg), (t_ft_end, p.release_shoulder_deg - 3),
                         (t_ft_end + 0.25, 60), (t_end, 35)])
    el_r = _keyframe(t, [(0.0, 65), (0.3, 70), (p.t_dip, p.elbow_set_deg), (t_rise, p.elbow_set_deg + 25),
                         (p.t_release, p.release_elbow_deg), (t_ft_end, p.release_elbow_deg - 6),
                         (t_ft_end + 0.25, 110), (t_end, 80)])
    wr_r = _keyframe(t, [(0.0, 10), (p.t_dip, -55), (p.t_release - 0.03, -50),
                         (p.t_release + 0.08, p.wrist_snap_deg), (t_ft_end, p.wrist_snap_deg),
                         (t_ft_end + 0.3, 10)])
    # guide arm: mirrors loosely, lower, and peels away after release
    sh_l = _keyframe(t, [(0.0, 20), (p.t_dip, p.set_shoulder_deg - 15), (p.t_release, p.release_shoulder_deg - 35),
                         (t_ft_end, 100), (t_end, 25)])
    el_l = _keyframe(t, [(0.0, 70), (p.t_dip, p.elbow_set_deg + 10), (p.t_release, 150), (t_ft_end, 140), (t_end, 85)])
    wr_l = _keyframe(t, [(0.0, 0), (p.t_dip, -30), (p.t_release, 10), (t_end, 0)])

    # ---- legs: ankle on ground, hip above ankle, symmetric knee bend
    bend = math.radians(180 - knee)
    half = bend / 2
    ankle = np.array([0.0, 0.0])
    knee_pt = ankle + np.array([SEG["shank"] * math.sin(half), SEG["shank"] * math.cos(half)])
    hip_pt = knee_pt + np.array([-SEG["thigh"] * math.sin(half), SEG["thigh"] * math.cos(half)])

    # airborne offset (parabola between take-off and landing)
    t_off = p.t_release - 0.12
    jump = 0.0
    if t_off < t < t_land:
        u = (t - t_off) / (t_land - t_off)
        jump = p.jump_height_m * 4 * u * (1 - u)
    lift = np.array([0.0, jump])

    # ---- trunk from hip angle: direction of torso relative to thigh direction
    thigh_dir = hip_pt - knee_pt
    thigh_ang = math.atan2(thigh_dir[1], thigh_dir[0])          # ~ +90deg = up
    torso_ang = thigh_ang - math.radians(180 - hip)             # fold forward (toward +x)
    torso_dir = np.array([math.cos(torso_ang), math.sin(torso_ang)])
    shoulder_pt = hip_pt + SEG["torso"] * torso_dir
    neck_pt = shoulder_pt + SEG["neck"] * torso_dir
    head_c = neck_pt + SEG["head_r"] * torso_dir

    def arm(shoulder_deg, elbow_deg, wrist_deg):
        # upper arm elevation measured from the downward torso axis, rotating forward
        down = -torso_dir
        a = math.radians(shoulder_deg)
        # clockwise rotation of the "down" vector swings the arm forward (+x) then up
        c, s = math.cos(a), math.sin(a)
        ua_dir = np.array([down[0] * c - down[1] * s, down[0] * s + down[1] * c])
        elbow = shoulder_pt + SEG["upper_arm"] * ua_dir
        b = math.radians(180 - elbow_deg)
        c, s = math.cos(b), math.sin(b)
        fa_dir = np.array([ua_dir[0] * c - ua_dir[1] * s, ua_dir[0] * s + ua_dir[1] * c])
        wrist = elbow + SEG["forearm"] * fa_dir
        w = math.radians(wrist_deg)
        c, s = math.cos(w), math.sin(w)
        hd_dir = np.array([fa_dir[0] * c - fa_dir[1] * s, fa_dir[0] * s + fa_dir[1] * c])
        hand_tip = wrist + SEG["hand"] * hd_dir
        thumb = wrist + 0.5 * SEG["hand"] * np.array([-hd_dir[1], hd_dir[0]]) * 0.6 + 0.4 * SEG["hand"] * hd_dir
        return elbow, wrist, hand_tip, thumb

    el_r_pt, wr_r_pt, tip_r, th_r = arm(sh_r, el_r, wr_r)
    el_l_pt, wr_l_pt, tip_l, th_l = arm(sh_l, el_l, wr_l)

    heel = ankle + np.array([-0.06, 0.0])
    toe = ankle + np.array([SEG["foot"] - 0.06, 0.0])
    joints = dict(
        hip=hip_pt, knee=knee_pt, ankle=ankle, heel=heel, toe=toe,
        shoulder=shoulder_pt, neck=neck_pt, head=head_c,
        elbow_r=el_r_pt, wrist_r=wr_r_pt, tip_r=tip_r, thumb_r=th_r,
        elbow_l=el_l_pt, wrist_l=wr_l_pt, tip_l=tip_l, thumb_l=th_l,
        torso_dir=torso_dir,
    )
    for k in list(joints.keys()):
        if k != "torso_dir":
            joints[k] = joints[k] + lift
    joints["released"] = t >= p.t_release
    joints["t"] = t
    return joints


def to_px(pt: np.ndarray) -> tuple[int, int]:
    return int(round(CX + pt[0] * PPM)), int(round(GROUND_Y - pt[1] * PPM))


# ------------------------------------------------------------ landmark export

def joints_to_landmarks(j: dict[str, np.ndarray], rng: np.random.Generator, noise_px: float) -> tuple[np.ndarray, np.ndarray]:
    """Map the synthetic joints to MediaPipe's 33-landmark layout (normalised)."""
    lm = np.full((NUM_LANDMARKS, 3), np.nan, dtype=np.float32)
    vis = np.zeros(NUM_LANDMARKS, dtype=np.float32)
    td = j["torso_dir"]
    fwd = np.array([td[1], -td[0]])  # perpendicular, pointing forward (+x-ish)
    if fwd[0] < 0:
        fwd = -fwd
    hr = SEG["head_r"]
    head = j["head"]
    # camera looks at the shooter's right side: right-side joints are near (z<0),
    # left-side joints are far (z>0) and slightly offset in x.
    zr, zl = -0.09, 0.09
    far_dx = np.array([-0.015, 0.0])

    def put(idx, pt, z, v=1.0):
        lm[idx] = (pt[0], pt[1], z)
        vis[idx] = v

    put(LM.NOSE, head + 0.95 * hr * fwd, 0.0)
    put(LM.RIGHT_EYE, head + 0.55 * hr * fwd + 0.35 * hr * td, zr * 0.4)
    put(LM.RIGHT_EYE_INNER, head + 0.65 * hr * fwd + 0.33 * hr * td, zr * 0.3)
    put(LM.RIGHT_EYE_OUTER, head + 0.45 * hr * fwd + 0.37 * hr * td, zr * 0.5)
    put(LM.LEFT_EYE, head + 0.55 * hr * fwd + 0.35 * hr * td, zl * 0.4, 0.5)
    put(LM.LEFT_EYE_INNER, head + 0.65 * hr * fwd + 0.33 * hr * td, zl * 0.3, 0.5)
    put(LM.LEFT_EYE_OUTER, head + 0.45 * hr * fwd + 0.37 * hr * td, zl * 0.5, 0.5)
    put(LM.RIGHT_EAR, head - 0.1 * hr * fwd + 0.1 * hr * td, zr * 0.8)
    put(LM.LEFT_EAR, head - 0.1 * hr * fwd + 0.1 * hr * td, zl * 0.8, 0.4)
    put(LM.MOUTH_RIGHT, head + 0.8 * hr * fwd - 0.35 * hr * td, zr * 0.2)
    put(LM.MOUTH_LEFT, head + 0.8 * hr * fwd - 0.35 * hr * td, zl * 0.2, 0.6)

    sh = j["shoulder"]
    put(LM.RIGHT_SHOULDER, sh, zr)
    put(LM.LEFT_SHOULDER, sh + far_dx, zl, 0.7)
    put(LM.RIGHT_ELBOW, j["elbow_r"], zr)
    put(LM.LEFT_ELBOW, j["elbow_l"] + far_dx, zl, 0.6)
    put(LM.RIGHT_WRIST, j["wrist_r"], zr)
    put(LM.LEFT_WRIST, j["wrist_l"] + far_dx, zl, 0.6)
    put(LM.RIGHT_INDEX, j["tip_r"], zr)
    put(LM.RIGHT_PINKY, j["tip_r"] * 0.9 + j["wrist_r"] * 0.1, zr * 1.1)
    put(LM.RIGHT_THUMB, j["thumb_r"], zr * 0.8)
    put(LM.LEFT_INDEX, j["tip_l"] + far_dx, zl, 0.5)
    put(LM.LEFT_PINKY, j["tip_l"] * 0.9 + j["wrist_l"] * 0.1 + far_dx, zl * 1.1, 0.5)
    put(LM.LEFT_THUMB, j["thumb_l"] + far_dx, zl * 0.8, 0.5)

    put(LM.RIGHT_HIP, j["hip"], zr * 0.8)
    put(LM.LEFT_HIP, j["hip"] + far_dx, zl * 0.8, 0.7)
    put(LM.RIGHT_KNEE, j["knee"], zr * 0.8)
    put(LM.LEFT_KNEE, j["knee"] + far_dx, zl * 0.8, 0.6)
    put(LM.RIGHT_ANKLE, j["ankle"], zr * 0.8)
    put(LM.LEFT_ANKLE, j["ankle"] + far_dx, zl * 0.8, 0.6)
    put(LM.RIGHT_HEEL, j["heel"], zr * 0.8)
    put(LM.LEFT_HEEL, j["heel"] + far_dx, zl * 0.8, 0.5)
    put(LM.RIGHT_FOOT_INDEX, j["toe"], zr * 0.8)
    put(LM.LEFT_FOOT_INDEX, j["toe"] + far_dx, zl * 0.8, 0.5)

    # metres -> normalised image coordinates (+ detector-like jitter)
    px = CX + lm[:, 0] * PPM + rng.normal(0, noise_px, NUM_LANDMARKS)
    py = GROUND_Y - lm[:, 1] * PPM + rng.normal(0, noise_px, NUM_LANDMARKS)
    out = np.stack([px / W, py / H, lm[:, 2] * PPM / W], axis=1).astype(np.float32)
    return out, vis


# ------------------------------------------------------------------ drawing

COL_SKIN = (120, 160, 225)      # BGR
COL_SKIN_FAR = (85, 115, 165)
COL_JERSEY = (60, 60, 200)
COL_JERSEY_FAR = (40, 40, 140)
COL_SHORTS = (40, 40, 40)
COL_SHOES = (230, 230, 230)
COL_BALL = (30, 110, 235)


def _court(frame_idx: int) -> np.ndarray:
    img = np.zeros((H, W, 3), np.uint8)
    # wall gradient
    for y in range(H):
        v = int(18 + 22 * (y / H))
        img[y, :] = (v, v - 4, v - 6)
    # floor
    cv2.rectangle(img, (0, GROUND_Y), (W, H), (48, 70, 95), -1)
    cv2.line(img, (0, GROUND_Y), (W, GROUND_Y), (90, 120, 150), 2)
    # a few floor boards
    for x in range(0, W, 48):
        cv2.line(img, (x, GROUND_Y + 2), (x - 12, H), (60, 85, 110), 1)
    # faint three-point arc on the floor (decoration only)
    cv2.ellipse(img, (W + 40, GROUND_Y - 2), (260, 14), 0, 180, 360, (110, 150, 180), 2)
    return img


def _limb(img, a, b, thick, col):
    cv2.line(img, to_px(a), to_px(b), col, thick, cv2.LINE_AA)
    cv2.circle(img, to_px(a), thick // 2, col, -1, cv2.LINE_AA)
    cv2.circle(img, to_px(b), thick // 2, col, -1, cv2.LINE_AA)


def draw_shooter(img: np.ndarray, j: dict[str, np.ndarray], p: ShotParams, ball_pos: np.ndarray | None):
    far = np.array([-0.015, 0.0])
    # far leg + arm
    _limb(img, j["hip"] + far, j["knee"] + far, 26, COL_SHORTS)
    _limb(img, j["knee"] + far, j["ankle"] + far, 20, COL_SKIN_FAR)
    _limb(img, j["heel"] + far, j["toe"] + far, 14, (160, 160, 160))
    _limb(img, j["shoulder"] + far, j["elbow_l"] + far, 18, COL_SKIN_FAR)
    _limb(img, j["elbow_l"] + far, j["wrist_l"] + far, 15, COL_SKIN_FAR)
    _limb(img, j["wrist_l"] + far, j["tip_l"] + far, 12, COL_SKIN_FAR)
    # torso (jersey)
    td = j["torso_dir"]
    side = np.array([td[1], -td[0]])
    hip, sh = j["hip"], j["shoulder"]
    poly = np.array([to_px(hip - 0.11 * side), to_px(hip + 0.11 * side),
                     to_px(sh + 0.14 * side), to_px(sh - 0.14 * side)], np.int32)
    cv2.fillPoly(img, [poly], COL_JERSEY, cv2.LINE_AA)
    cv2.circle(img, to_px(sh), 16, COL_JERSEY, -1, cv2.LINE_AA)
    # shorts
    poly2 = np.array([to_px(hip - 0.12 * side), to_px(hip + 0.12 * side),
                      to_px(hip + 0.10 * side - 0.18 * td), to_px(hip - 0.10 * side - 0.18 * td)], np.int32)
    cv2.fillPoly(img, [poly2], COL_SHORTS, cv2.LINE_AA)
    # neck + head
    _limb(img, j["shoulder"], j["neck"], 12, COL_SKIN)
    cv2.circle(img, to_px(j["head"]), int(SEG["head_r"] * PPM), COL_SKIN, -1, cv2.LINE_AA)
    # face hint (eye) so the head has orientation
    eye = j["head"] + 0.55 * SEG["head_r"] * side * (1 if side[0] > 0 else -1) + 0.3 * SEG["head_r"] * td
    cv2.circle(img, to_px(eye), 3, (30, 30, 30), -1, cv2.LINE_AA)
    # near leg
    _limb(img, j["hip"], j["knee"], 28, COL_SHORTS)
    _limb(img, j["knee"], j["ankle"], 22, COL_SKIN)
    _limb(img, j["heel"], j["toe"], 15, COL_SHOES)
    # near (shooting) arm
    _limb(img, j["shoulder"], j["elbow_r"], 20, COL_SKIN)
    _limb(img, j["elbow_r"], j["wrist_r"], 16, COL_SKIN)
    _limb(img, j["wrist_r"], j["tip_r"], 13, COL_SKIN)
    # ball
    if ball_pos is not None:
        cv2.circle(img, to_px(ball_pos), 24, COL_BALL, -1, cv2.LINE_AA)
        cv2.circle(img, to_px(ball_pos), 24, (10, 40, 90), 2, cv2.LINE_AA)
        cv2.line(img, (to_px(ball_pos)[0] - 24, to_px(ball_pos)[1]), (to_px(ball_pos)[0] + 24, to_px(ball_pos)[1]), (10, 40, 90), 2)


def ball_position(p: ShotParams, j: dict[str, np.ndarray], t: float) -> np.ndarray | None:
    if t < p.t_release:
        # ball rests on the shooting hand
        return j["wrist_r"] + 0.55 * (j["tip_r"] - j["wrist_r"]) + np.array([0.02, 0.10])
    # flight: parabola up and to the right, leaves the frame
    rel = pose_at(p, p.t_release)
    start = rel["wrist_r"] + 0.55 * (rel["tip_r"] - rel["wrist_r"]) + np.array([0.02, 0.10])
    dt = t - p.t_release
    pos = start + np.array([3.2 * dt, 5.0 * dt - 0.5 * 9.81 * dt * dt])
    if pos[0] * PPM + CX > W + 60:
        return None
    return pos


# ------------------------------------------------------------------- render

def render_clip(p: ShotParams, out_mp4: str) -> str:
    rng = np.random.default_rng(p.seed)
    n = int(round(p.duration_s * p.fps))
    writer = cv2.VideoWriter(out_mp4, cv2.VideoWriter_fourcc(*"mp4v"), p.fps, (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"could not open VideoWriter for {out_mp4}")
    lms = np.zeros((n, NUM_LANDMARKS, 3), np.float32)
    viss = np.zeros((n, NUM_LANDMARKS), np.float32)
    for i in range(n):
        t = i / p.fps
        j = pose_at(p, t)
        img = _court(i)
        draw_shooter(img, j, p, ball_position(p, j, t))
        writer.write(img)
        lms[i], viss[i] = joints_to_landmarks(j, rng, p.noise_px)
    writer.release()
    sidecar = save_landmarks(out_mp4, lms, viss, fps=p.fps, width=W, height=H,
                             source="synthetic", meta=p.to_json())
    return sidecar


def generate(demo_dir: str, references: list[ShotParams] | None = None, user: ShotParams | None = None) -> dict:
    references = references or DEFAULT_REFERENCES
    user = user or DEFAULT_USER
    os.makedirs(demo_dir, exist_ok=True)
    manifest = {"references": [], "user": None, "note": "All clips are procedurally generated; no real footage."}
    for p in references:
        path = os.path.join(demo_dir, f"{p.slug}.mp4")
        side = render_clip(p, path)
        manifest["references"].append({"name": p.name, "video": path, "landmarks": side, "params": p.to_json()})
        print(f"[demo_data] wrote {path} (+ sidecar)")
    upath = os.path.join(demo_dir, f"{user.slug}.mp4")
    side = render_clip(user, upath)
    manifest["user"] = {"name": user.name, "video": upath, "landmarks": side, "params": user.to_json()}
    print(f"[demo_data] wrote {upath} (+ sidecar)")
    with open(os.path.join(demo_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def generate_sessions(session_dir: str, sessions: list[ShotParams] | None = None) -> list[dict]:
    """Render the earlier practice-session clips (+ sidecars) used by the demo progress panel."""
    sessions = sessions or DEMO_SESSIONS
    os.makedirs(session_dir, exist_ok=True)
    out = []
    for p in sessions:
        path = os.path.join(session_dir, f"{p.slug}.mp4")
        side = render_clip(p, path)
        out.append({"name": p.name, "video": path, "landmarks": side, "params": p.to_json()})
    print(f"[demo_data] wrote {len(out)} session clips to {session_dir}")
    return out


if __name__ == "__main__":
    import sys
    generate(sys.argv[1] if len(sys.argv) > 1 else "demo")
