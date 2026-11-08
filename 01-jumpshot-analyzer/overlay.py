"""
overlay.py — neon skeleton overlays and the side-by-side comparison video.

Left panel : the user's video with the skeleton drawn on top
             (hot-white joints, amber bones, hot-orange shooting arm).
Right panel: the reference, time-aligned to the user's shot phases — the
             reference video with its skeleton if the clip is still on disk,
             otherwise a stick figure on a dark background.
Feature readouts (elbow / knee angles, wrist height, phase) are drawn on frame.
"""
from __future__ import annotations

import os
import shutil
import subprocess

import cv2
import numpy as np

from features import FEATURE_NAMES, Shot
from pose import BONES, LEFT_ARM_BONES, RIGHT_ARM_BONES, LM

# BGR "Film Room" palette (matches output/theme.css written by report.py):
# charcoal background, amber bones, hot-white joints, hot-orange shooting arm.
CYAN = (245, 245, 255)        # joints: hot white (name kept for call sites)
LIME = (26, 140, 255)         # bones: amber #ff8c1a
MAGENTA = (26, 77, 255)       # shooting arm: hot orange #ff4d1a
AMBER = (26, 140, 255)
WHITE = (236, 236, 245)
MUTED = (150, 150, 160)
BG = (16, 15, 14)
PANEL = (26, 24, 22)

PANEL_H = 640
VIDEO_H = 470          # video area inside a panel; the rest holds header + readouts
HEADER_H = 64
FOOTER_H = 76
GAP = 12
FONT = cv2.FONT_HERSHEY_SIMPLEX
FI = {n: i for i, n in enumerate(FEATURE_NAMES)}


# ----------------------------------------------------------------- drawing

def _glow_line(img, a, b, color, thickness):
    dark = tuple(int(c * 0.35) for c in color)
    cv2.line(img, a, b, dark, thickness + 6, cv2.LINE_AA)
    cv2.line(img, a, b, color, thickness, cv2.LINE_AA)


def draw_skeleton(img: np.ndarray, pts: np.ndarray, side: str, scale: float = 1.0, alpha: float = 1.0) -> np.ndarray:
    """pts: (33, 2) pixel coordinates already in `img` space. Returns img (drawn in place / blended)."""
    layer = img if alpha >= 1.0 else img.copy()
    arm = RIGHT_ARM_BONES if side == "right" else LEFT_ARM_BONES
    arm_joints = {LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW, LM.RIGHT_WRIST, LM.RIGHT_INDEX, LM.RIGHT_PINKY, LM.RIGHT_THUMB} \
        if side == "right" else {LM.LEFT_SHOULDER, LM.LEFT_ELBOW, LM.LEFT_WRIST, LM.LEFT_INDEX, LM.LEFT_PINKY, LM.LEFT_THUMB}
    t = max(2, int(round(3 * scale)))
    for a, b in BONES:
        pa, pb = pts[a], pts[b]
        if not (np.isfinite(pa).all() and np.isfinite(pb).all()):
            continue
        col = MAGENTA if (a, b) in arm else LIME
        if a < 11 or b < 11:
            col = tuple(int(c * 0.6) for c in LIME)
        _glow_line(layer, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), col, t + (1 if col == MAGENTA else 0))
    r = max(3, int(round(4 * scale)))
    for i, p in enumerate(pts):
        if not np.isfinite(p).all() or i < 11:
            continue
        col = MAGENTA if i in arm_joints else CYAN
        cv2.circle(layer, (int(p[0]), int(p[1])), r + 3, tuple(int(c * 0.3) for c in col), -1, cv2.LINE_AA)
        cv2.circle(layer, (int(p[0]), int(p[1])), r, col, -1, cv2.LINE_AA)
        cv2.circle(layer, (int(p[0]), int(p[1])), max(1, r - 2), WHITE, -1, cv2.LINE_AA)
    if alpha < 1.0:
        cv2.addWeighted(layer, alpha, img, 1 - alpha, 0, img)
    return img


def _ascii(s: str) -> str:
    # Hershey fonts only cover ASCII
    return s.replace("\u2014", "-").replace("\u2013", "-").replace("\u00b0", " deg").encode("ascii", "replace").decode()


def text(img, s, org, scale=0.55, color=WHITE, thick=1, mono=False):
    s = _ascii(s)
    font = cv2.FONT_HERSHEY_DUPLEX if mono else FONT
    cv2.putText(img, s, org, font, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, s, org, font, scale, color, thick, cv2.LINE_AA)


def phase_label(shot: Shot, frame: int) -> str:
    ev = shot.events
    if frame < ev.onset:
        return "STANCE"
    if frame < ev.set_point:
        return "LOAD / DIP"
    if frame < ev.release:
        return "RISE" if frame > ev.set_point + 2 else "SET POINT"
    if frame <= ev.release + 1:
        return "RELEASE"
    if frame <= ev.follow_through_end:
        return "FOLLOW-THROUGH"
    return "LANDING"


def readout_lines(shot: Shot, frame: int) -> list[tuple[str, tuple]]:
    f = shot.per_frame[min(frame, shot.per_frame.shape[0] - 1)]
    return [
        (f"elbow {f[FI['elbow_angle']]:5.0f} deg", LIME),
        (f"knee  {f[FI['knee_angle']]:5.0f} deg", LIME),
        (f"hip   {f[FI['hip_angle']]:5.0f} deg", LIME),
        (f"wrist {f[FI['wrist_height']]:+5.2f} torso above head", CYAN),
        (f"jump  {f[FI['hip_rise']]:+5.2f} torso", CYAN),
    ]


# -------------------------------------------------------------- alignment

def align_frames(user: Shot, ref: Shot) -> np.ndarray:
    """For each user frame, the reference frame index at the same shot phase (piecewise-linear
    through onset / set point / release / follow-through end)."""
    ue, re = user.events, ref.events
    uk = np.array([0, ue.onset, ue.set_point, ue.release, ue.follow_through_end, user.per_frame.shape[0] - 1], float)
    rk = np.array([0, re.onset, re.set_point, re.release, re.follow_through_end, ref.per_frame.shape[0] - 1], float)
    # enforce monotonic knots
    uk = np.maximum.accumulate(uk)
    rk = np.maximum.accumulate(rk)
    idx = np.interp(np.arange(user.per_frame.shape[0]), uk, rk)
    return np.clip(np.round(idx).astype(int), 0, ref.per_frame.shape[0] - 1)


# ------------------------------------------------------------------ panels

def _fit(frame: np.ndarray, pts: np.ndarray, target_h: int) -> tuple[np.ndarray, np.ndarray, float]:
    s = target_h / frame.shape[0]
    out = cv2.resize(frame, (int(round(frame.shape[1] * s)), target_h), interpolation=cv2.INTER_AREA)
    return out, pts * s, s


def stick_panel(shot: Shot, frame: int, width: int, height: int) -> np.ndarray:
    """Reference stick figure on a dark background, scaled to fit the panel."""
    img = np.full((height, width, 3), PANEL, np.uint8)
    # subtle grid
    for x in range(0, width, 40):
        cv2.line(img, (x, 0), (x, height), (40, 28, 22), 1)
    for y in range(0, height, 40):
        cv2.line(img, (0, y), (width, y), (40, 28, 22), 1)
    pts_all = shot.landmarks_px
    mn = np.nanmin(pts_all.reshape(-1, 2), axis=0)
    mx = np.nanmax(pts_all.reshape(-1, 2), axis=0)
    span = np.maximum(mx - mn, 1)
    usable_h = height - 190 if height > 400 else height - 40
    s = min((width - 60) / span[0], usable_h / span[1])
    off = np.array([(width - span[0] * s) / 2, 40 + (usable_h - span[1] * s) / 2])
    pts = (pts_all[min(frame, pts_all.shape[0] - 1)] - mn) * s + off
    # floor line under the lowest point
    floor_y = int((mx[1] - mn[1]) * s + off[1]) + 6
    cv2.line(img, (20, floor_y), (width - 20, floor_y), MUTED, 1, cv2.LINE_AA)
    draw_skeleton(img, pts, shot.side, scale=1.0)
    return img


def user_panel_from_frame(frame_bgr: np.ndarray, shot: Shot, frame_idx: int, height: int,
                          video_h: int | None = None) -> np.ndarray:
    """Video frame (with skeleton) letterboxed into a dark panel of the given height."""
    pts = shot.landmarks_px[min(frame_idx, shot.landmarks_px.shape[0] - 1)]
    video_h = video_h or height
    out, pts_s, s = _fit(frame_bgr, pts, video_h)
    draw_skeleton(out, pts_s, shot.side, scale=max(0.7, s))
    if video_h >= height:
        return out
    panel = np.full((height, out.shape[1], 3), PANEL, np.uint8)
    y0 = 32
    panel[y0:y0 + video_h] = out
    return panel


def _read_all_frames(path: str) -> list[np.ndarray]:
    frames = []
    if path and os.path.isfile(path):
        cap = cv2.VideoCapture(path)
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        cap.release()
    return frames


def _panel_header(img, title, color, sub=""):
    cv2.rectangle(img, (0, 0), (img.shape[1], 30), (0, 0, 0), -1)
    text(img, title, (10, 21), 0.6, color, 1, mono=True)
    if sub:
        text(img, sub, (img.shape[1] - 10 - int(len(sub) * 11), 21), 0.5, MUTED, 1)


def _draw_readouts(img, shot, frame_idx, color_phase):
    lines = readout_lines(shot, frame_idx)
    y0 = img.shape[0] - 14 - 20 * len(lines)
    cv2.rectangle(img, (0, y0 - 26), (img.shape[1], img.shape[0]), (0, 0, 0), -1)
    text(img, phase_label(shot, frame_idx), (10, y0 - 7), 0.55, color_phase, 1, mono=True)
    for i, (s, col) in enumerate(lines):
        text(img, s, (10, y0 + 15 + 20 * i), 0.48, col, 1)


# ------------------------------------------------------------------ video

def _ffmpeg_h264(src: str, dst: str) -> bool:
    ff = shutil.which("ffmpeg")
    if not ff:
        return False
    try:
        subprocess.run([ff, "-y", "-loglevel", "error", "-i", src, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "23", "-movflags", "+faststart", dst], check=True, timeout=240)
        return True
    except Exception:
        return False


def render_side_by_side(user_video: str, user: Shot, ref: Shot, results: dict, out_path: str,
                        verbose: bool = True) -> str:
    user_frames = _read_all_frames(user_video)
    if not user_frames:
        raise FileNotFoundError(f"cannot read user video {user_video}")
    ref_frames = _read_all_frames(ref.video_path)
    n = min(len(user_frames), user.per_frame.shape[0])
    amap = align_frames(user, ref)
    # panel geometry
    u0, _, _ = _fit(user_frames[0], user.landmarks_px[0], VIDEO_H)
    uw = u0.shape[1]
    if ref_frames:
        r0, _, _ = _fit(ref_frames[0], ref.landmarks_px[0], VIDEO_H)
        rw = r0.shape[1]
    else:
        rw = uw
    W = uw + rw + GAP * 3
    H = HEADER_H + PANEL_H + FOOTER_H
    W += W % 2
    H += H % 2
    tmp = out_path + ".tmp.mp4"
    writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), user.fps, (W, H))
    if not writer.isOpened():
        raise RuntimeError("could not open VideoWriter")
    best = results["references"][0]
    score = best["score"]
    for i in range(n):
        canvas = np.full((H, W, 3), BG, np.uint8)
        # header
        text(canvas, "AI JUMPSHOT ANALYZER", (GAP, 26), 0.7, CYAN, 1, mono=True)
        text(canvas, f"form similarity {score:.0f}/100  vs  {ref.name}", (GAP, 50), 0.5, WHITE, 1)
        text(canvas, f"frame {i:03d}  t={i / user.fps:5.2f}s", (W - 230, 26), 0.5, MUTED, 1)
        # left: user
        up = user_panel_from_frame(user_frames[i], user, i, PANEL_H, VIDEO_H)
        _panel_header(up, "USER", CYAN, f"{user.side} hand")
        _draw_readouts(up, user, i, CYAN)
        if i == user.events.release:
            cv2.rectangle(up, (2, 2), (up.shape[1] - 3, up.shape[0] - 3), MAGENTA, 4)
        canvas[HEADER_H:HEADER_H + PANEL_H, GAP:GAP + uw] = up
        # right: reference
        j = int(amap[i])
        if ref_frames:
            rp = user_panel_from_frame(ref_frames[min(j, len(ref_frames) - 1)], ref, j, PANEL_H, VIDEO_H)
        else:
            rp = stick_panel(ref, j, rw, PANEL_H)
        _panel_header(rp, "REFERENCE", LIME, f"{ref.side} hand")
        _draw_readouts(rp, ref, j, LIME)
        if j == ref.events.release:
            cv2.rectangle(rp, (2, 2), (rp.shape[1] - 3, rp.shape[0] - 3), MAGENTA, 4)
        x0 = GAP * 2 + uw
        canvas[HEADER_H:HEADER_H + PANEL_H, x0:x0 + rp.shape[1]] = rp
        # footer: live deltas
        fu = user.per_frame[i]
        fr = ref.per_frame[j]
        d_elbow = fu[FI["elbow_angle"]] - fr[FI["elbow_angle"]]
        d_knee = fu[FI["knee_angle"]] - fr[FI["knee_angle"]]
        d_wh = fu[FI["wrist_height"]] - fr[FI["wrist_height"]]
        y = HEADER_H + PANEL_H + 30
        text(canvas, f"delta elbow {d_elbow:+.0f} deg   delta knee {d_knee:+.0f} deg   delta wrist height {d_wh:+.2f} torso",
             (GAP, y), 0.52, AMBER, 1)
        text(canvas, "white joints / amber bones / orange shooting arm  -  reference time-warped to user phases",
             (GAP, y + 26), 0.42, MUTED, 1)
        writer.write(canvas)
    writer.release()
    if _ffmpeg_h264(tmp, out_path):
        os.remove(tmp)
    else:
        os.replace(tmp, out_path)
    if verbose:
        print(f"[overlay] wrote {out_path} ({n} frames, {W}x{H})")
    return out_path


# -------------------------------------------------------------- key frames

def key_frames(user_video: str, user: Shot, ref: Shot, height: int = 360) -> list[dict]:
    """Set point / release / follow-through stills for user and reference (BGR images)."""
    user_frames = _read_all_frames(user_video)
    ref_frames = _read_all_frames(ref.video_path)
    out = []
    for label, ukey in [("Set point", "set_point"), ("Release", "release"), ("Follow-through", "follow_through")]:
        ui = getattr(user.events, ukey)
        ri = getattr(ref.events, ukey)
        ui = min(ui, len(user_frames) - 1)
        uimg = user_panel_from_frame(user_frames[ui], user, ui, height)
        if ref_frames:
            rimg = user_panel_from_frame(ref_frames[min(ri, len(ref_frames) - 1)], ref, ri, height)
        else:
            rimg = stick_panel(ref, ri, uimg.shape[1], height)
        out.append(dict(label=label, user_frame=ui, ref_frame=ri, user_img=uimg, ref_img=rimg,
                        user_t=ui / user.fps, ref_t=ri / ref.fps))
    return out
