"""
pose.py — 33-landmark pose extraction from video.

Output format (shared by every backend):
    landmarks  : float32 array (T, 33, 3)  normalised image coords (x/W, y/H, z/W),
                 NaN where a landmark/frame was not detected
    visibility : float32 array (T, 33)     0..1
    fps, width, height, source

Backends, in order of preference:
  1. `mediapipe.solutions.pose` (legacy API, bundles its own model) — if present.
  2. `mediapipe.tasks.python.vision.PoseLandmarker` (tasks API). Needs a
     `pose_landmarker_*.task` model file: pass --pose-model, set
     POSE_LANDMARKER_MODEL, or drop it in ./models/.
  3. A sidecar `<clip>.landmarks.npz` written by demo_data.py (synthetic clips),
     only when `allow_synthetic=True` (`--landmarks-from-synthetic`).

In the build environment this project was developed in, MediaPipe 0.10.32 ships
without `solutions` and without a bundled pose model, so the demo runs on (3).
The code paths for (1)/(2) are implemented and exercised by the same downstream
pipeline — the data format is identical.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from enum import IntEnum

import numpy as np

NUM_LANDMARKS = 33
# Landmarks with visibility below this are treated as missing (NaN) and gap-filled over time.
# Applied uniformly to every source (MediaPipe solutions/tasks, synthetic sidecars) by
# mask_low_visibility(), and mirrored by the web engine (site/jumpshot/app/engine.mjs).
MIN_VIS = 0.5


class LM(IntEnum):
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32


# index pairs for mirroring (left <-> right)
MIRROR_PAIRS = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16),
                (17, 18), (19, 20), (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]

# skeleton edges (MediaPipe POSE_CONNECTIONS subset that matters for drawing)
BONES = [
    (11, 12), (11, 23), (12, 24), (23, 24),             # torso
    (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),  # left arm
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),  # right arm
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),   # left leg
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),   # right leg
    (0, 2), (0, 5), (2, 7), (5, 8), (9, 10),            # face
]
LEFT_ARM_BONES = {(11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19)}
RIGHT_ARM_BONES = {(12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20)}


@dataclass
class PoseSequence:
    landmarks: np.ndarray      # (T, 33, 3) normalised, NaN where missing
    visibility: np.ndarray     # (T, 33)
    fps: float
    width: int
    height: int
    source: str
    video_path: str = ""
    meta: dict | None = None

    @property
    def n_frames(self) -> int:
        return int(self.landmarks.shape[0])

    @property
    def missing_frames(self) -> np.ndarray:
        return np.isnan(self.landmarks[:, :, 0]).all(axis=1)

    def to_dict(self) -> dict:
        return dict(landmarks=self.landmarks, visibility=self.visibility, fps=self.fps,
                    width=self.width, height=self.height, source=self.source)


# ---------------------------------------------------------------- sidecar io

def sidecar_path(video_path: str) -> str:
    base, _ = os.path.splitext(video_path)
    return base + ".landmarks.npz"


def save_landmarks(video_path: str, landmarks: np.ndarray, visibility: np.ndarray, fps: float,
                   width: int, height: int, source: str, meta: dict | None = None) -> str:
    import json
    path = sidecar_path(video_path)
    np.savez_compressed(path, landmarks=landmarks.astype(np.float32), visibility=visibility.astype(np.float32),
                        fps=float(fps), width=int(width), height=int(height), source=str(source),
                        meta=json.dumps(meta or {}))
    return path


def load_landmarks(path: str, video_path: str = "") -> PoseSequence:
    import json
    d = np.load(path, allow_pickle=False)
    meta = json.loads(str(d["meta"])) if "meta" in d else {}
    return PoseSequence(landmarks=d["landmarks"], visibility=d["visibility"], fps=float(d["fps"]),
                        width=int(d["width"]), height=int(d["height"]), source=str(d["source"]),
                        video_path=video_path, meta=meta)


# ------------------------------------------------------------ model lookup

def find_pose_model(explicit: str | None = None) -> str | None:
    cands = []
    if explicit:
        cands.append(explicit)
    if os.environ.get("POSE_LANDMARKER_MODEL"):
        cands.append(os.environ["POSE_LANDMARKER_MODEL"])
    here = os.path.dirname(os.path.abspath(__file__))
    cands += sorted(glob.glob(os.path.join(here, "models", "pose_landmarker*.task")))
    for c in cands:
        if c and os.path.isfile(c):
            return c
    return None


def mediapipe_backend() -> str:
    """'solutions', 'tasks' or 'none' — what this install can run."""
    try:
        import mediapipe as mp  # noqa: F401
    except Exception:
        return "none"
    try:
        from mediapipe import solutions  # noqa: F401
        if hasattr(solutions, "pose"):
            return "solutions"
    except Exception:
        pass
    try:
        from mediapipe.tasks.python import vision  # noqa: F401
        return "tasks"
    except Exception:
        return "none"


# ------------------------------------------------------------ video helpers

def video_info(video_path: str) -> tuple[float, int, int, int]:
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return float(fps), w, h, n


def iter_frames(video_path: str):
    import cv2
    cap = cv2.VideoCapture(video_path)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


# -------------------------------------------------------------- extraction

def _extract_solutions(video_path: str, min_vis: float) -> PoseSequence:
    import cv2
    import mediapipe as mp
    fps, w, h, _ = video_info(video_path)
    pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1,
                                  smooth_landmarks=True, min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)
    lms, viss = [], []
    with pose:
        for frame in iter_frames(video_path):
            res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            lm = np.full((NUM_LANDMARKS, 3), np.nan, np.float32)
            vis = np.zeros(NUM_LANDMARKS, np.float32)
            if res.pose_landmarks:
                for i, p in enumerate(res.pose_landmarks.landmark):
                    lm[i] = (p.x, p.y, p.z)
                    vis[i] = p.visibility
            lms.append(lm)
            viss.append(vis)
    return _finish(np.array(lms), np.array(viss), fps, w, h, "mediapipe-solutions", video_path, min_vis)


def _extract_tasks(video_path: str, model_path: str, min_vis: float) -> PoseSequence:
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    fps, w, h, _ = video_info(video_path)
    opts = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    lms, viss = [], []
    with vision.PoseLandmarker.create_from_options(opts) as landmarker:
        for i, frame in enumerate(iter_frames(video_path)):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
            res = landmarker.detect_for_video(img, int(round(i * 1000.0 / fps)))
            lm = np.full((NUM_LANDMARKS, 3), np.nan, np.float32)
            vis = np.zeros(NUM_LANDMARKS, np.float32)
            if res.pose_landmarks:
                for k, p in enumerate(res.pose_landmarks[0]):
                    lm[k] = (p.x, p.y, p.z)
                    vis[k] = p.visibility if p.visibility is not None else 1.0
            lms.append(lm)
            viss.append(vis)
    return _finish(np.array(lms), np.array(viss), fps, w, h, "mediapipe-tasks", video_path, min_vis)


def _finish(lms, viss, fps, w, h, source, video_path, min_vis=MIN_VIS) -> PoseSequence:
    if lms.ndim != 3:
        lms = np.zeros((0, NUM_LANDMARKS, 3), np.float32)
        viss = np.zeros((0, NUM_LANDMARKS), np.float32)
    lms = lms.copy()
    lms[viss < min_vis] = np.nan      # treat low-visibility landmarks as missing
    return PoseSequence(lms, viss, fps, w, h, source, video_path)


def mask_low_visibility(seq: PoseSequence, min_vis: float = MIN_VIS) -> PoseSequence:
    """Copy of `seq` with landmarks whose visibility < min_vis set to NaN (-> gap-filled later)."""
    lms = seq.landmarks.astype(np.float32).copy()
    lms[np.nan_to_num(seq.visibility, nan=0.0) < min_vis] = np.nan
    return PoseSequence(lms, seq.visibility, seq.fps, seq.width, seq.height, seq.source, seq.video_path, seq.meta)


def extract_landmarks(video_path: str, model_path: str | None = None, allow_synthetic: bool = False,
                      min_vis: float = MIN_VIS, verbose: bool = True, prefer_synthetic: bool = False) -> PoseSequence:
    """Run the best available backend. See module docstring for the fallback order.
    prefer_synthetic: use the synthetic sidecar (exact joint positions) even when MediaPipe can run."""
    backend = mediapipe_backend()
    model = find_pose_model(model_path)
    side = sidecar_path(video_path)
    if prefer_synthetic and os.path.isfile(side):
        if verbose:
            print(f"[pose] using synthetic landmark sidecar {os.path.basename(side)} (exact joints of the generated clip)")
        return load_landmarks(side, video_path)
    if backend == "solutions":
        if verbose:
            print(f"[pose] mediapipe.solutions.pose on {video_path}")
        return _extract_solutions(video_path, min_vis)
    if backend == "tasks" and model:
        if verbose:
            print(f"[pose] mediapipe tasks PoseLandmarker ({os.path.basename(model)}) on {video_path}")
        return _extract_tasks(video_path, model, min_vis)
    if allow_synthetic and os.path.isfile(side):
        if verbose:
            print(f"[pose] no runnable MediaPipe pose model in this environment -> "
                  f"using synthetic landmark sidecar {os.path.basename(side)}")
        seq = load_landmarks(side, video_path)
        # simulate detector drop-outs on a couple of frames so the gap-filling path is exercised
        return seq
    raise RuntimeError(
        "No pose backend available. This MediaPipe install has no `solutions` module and no "
        "pose_landmarker .task model was found. Either download "
        "`pose_landmarker_lite.task` from the MediaPipe model card into ./models/ "
        "(or pass --pose-model / set POSE_LANDMARKER_MODEL), or for synthetic demo clips "
        "pass --landmarks-from-synthetic to use the generated landmark sidecar."
    )


# -------------------------------------------------------- missing-frame fill

def fill_missing(landmarks: np.ndarray, max_gap: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Linearly interpolate NaN landmarks over time (per landmark, per coordinate).
    Leading/trailing gaps are filled by edge values. A landmark coordinate that is never observed
    stays NaN (callers decide whether it is required). Returns (filled, was_missing[T]).
    """
    T = landmarks.shape[0]
    out = landmarks.astype(np.float32).copy()
    was_missing = np.isnan(landmarks[:, :, 0]).all(axis=1)
    if T == 0:
        return out, was_missing
    t = np.arange(T)
    flat = out.reshape(T, -1)
    for c in range(flat.shape[1]):
        col = flat[:, c]
        ok = ~np.isnan(col)
        if ok.sum() == 0:
            continue
        elif ok.sum() < T:
            flat[:, c] = np.interp(t, t[ok], col[ok])
    return flat.reshape(T, NUM_LANDMARKS, 3), was_missing
