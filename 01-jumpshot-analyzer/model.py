"""
model.py — learned shot embedding + DTW baseline.

ShotEncoder: a small 1-D convolutional network that maps a time-normalised
(N_FRAMES x F) feature sequence to a 64-d unit vector. It is trained with a
contrastive (NT-Xent / SimCLR-style) objective on augmented views of the shots
in the reference library: two augmented views of the same shot are positives,
views of other shots are negatives. Augmentations are applied at the *landmark*
level (time-warp, speed change, coordinate jitter, scale/translate, left-right
mirror) and features are recomputed, so the encoder learns invariance to the
nuisances a phone video introduces rather than to the mechanics themselves.

dtw_similarity: classic dynamic time warping on z-scored feature sequences, so
the comparison never depends on the learned model alone.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F_

from features import FEATURE_NAMES, N_FRAMES, Shot, per_frame_features, detect_events, resample, rebaseline_hip_rise
from pose import MIRROR_PAIRS

EMBED_DIM = 64

# The model is tiny; on small CPUs multi-threaded BLAS is pure overhead (measured 60x slower
# with 2 threads than 1 on a 2-core box). Pin to a single thread unless the user overrides.
import os as _os
if not _os.environ.get("TORCH_NUM_THREADS"):
    torch.set_num_threads(1)


# ----------------------------------------------------------------- encoder

class ShotEncoder(nn.Module):
    def __init__(self, n_features: int = len(FEATURE_NAMES), hidden: int = 48, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_features, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden), nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2, stride=2),
            nn.BatchNorm1d(hidden), nn.GELU(),
            nn.Conv1d(hidden, hidden * 2, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm1d(hidden * 2), nn.GELU(),
            nn.Conv1d(hidden * 2, hidden * 2, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.head = nn.Sequential(nn.Linear(hidden * 2 * 2, hidden * 2), nn.GELU(), nn.Linear(hidden * 2, embed_dim))
        # auxiliary decoder: predict the coarse (6-bin) feature profile from the embedding so the
        # embedding keeps encoding *magnitudes*, not just "which library shot is this".
        self.n_bins = 6
        self.decoder = nn.Linear(embed_dim, n_features * self.n_bins)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) -> (B, F, T)
        h = self.net(x.transpose(1, 2))
        pooled = torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1)   # mean + max pooling over time
        z = self.head(pooled)
        return F_.normalize(z, dim=1)

    def reconstruct(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    @staticmethod
    def coarse_profile(x: torch.Tensor, n_bins: int = 6) -> torch.Tensor:
        B, T, F = x.shape
        return x[:, : (T // n_bins) * n_bins].reshape(B, n_bins, T // n_bins, F).mean(2).reshape(B, -1)


# ---------------------------------------------------------- normalisation

class FeatureStats:
    """Per-feature mean/std for z-scoring (fitted on the reference library)."""

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean = mean.astype(np.float32)
        self.std = np.maximum(std.astype(np.float32), 1e-3)

    @classmethod
    def fit(cls, seqs: list[np.ndarray]) -> "FeatureStats":
        allf = np.concatenate(seqs, axis=0)
        std = allf.std(axis=0)
        # floor the std so near-constant features don't blow up; use sensible physical scales
        floor = np.array([6, 6, 6, 6, 0.08, 0.08, 0.06, 6, 0.08], np.float32)[: allf.shape[1]]
        return cls(allf.mean(axis=0), np.maximum(std, floor))

    def apply(self, seq: np.ndarray) -> np.ndarray:
        return ((seq - self.mean) / self.std).astype(np.float32)

    def to_dict(self) -> dict:
        return dict(mean=self.mean, std=self.std)


# ------------------------------------------------------------ augmentation

def augment_landmarks(px: np.ndarray, fps: float, rng: np.random.Generator, side: str) -> tuple[np.ndarray, str]:
    """
    Landmark-level augmentation of a (T, 33, 2) pixel sequence. Returns new sequence and
    (possibly swapped) shooting side.
    """
    T = px.shape[0]
    out = px.copy()
    # 1) non-linear time warp: resample along a random monotonic time map
    if rng.random() < 0.9:
        knots = 6
        w = rng.gamma(9.0, 1.0, knots)
        cum = np.concatenate([[0], np.cumsum(w)])
        cum = cum / cum[-1]
        src_t = np.interp(np.linspace(0, 1, T), np.linspace(0, 1, knots + 1), cum) * (T - 1)
        flat = out.reshape(T, -1)
        out = np.stack([np.interp(src_t, np.arange(T), flat[:, c]) for c in range(flat.shape[1])], axis=1).reshape(T, 33, 2)
    # 2) global speed change (crop or pad in time)
    if rng.random() < 0.6:
        speed = rng.uniform(0.8, 1.25)
        n_new = max(12, int(round(T / speed)))
        flat = out.reshape(T, -1)
        src = np.linspace(0, T - 1, n_new)
        out = np.stack([np.interp(src, np.arange(T), flat[:, c]) for c in range(flat.shape[1])], axis=1).reshape(n_new, 33, 2)
    T = out.shape[0]
    # 3) scale + translate (camera distance / framing)
    s = rng.uniform(0.75, 1.3)
    centre = np.nanmean(out.reshape(-1, 2), axis=0)      # NaN = landmark never observed (stays NaN)
    out = (out - centre) * s + centre + rng.normal(0, 25, 2)
    # 4) small in-plane rotation (camera roll)
    ang = np.radians(rng.normal(0, 1.5))
    c, sn = np.cos(ang), np.sin(ang)
    rel = out - centre
    out = np.stack([rel[..., 0] * c - rel[..., 1] * sn, rel[..., 0] * sn + rel[..., 1] * c], axis=-1) + centre
    # 5) per-landmark jitter + slow drift (detector noise)
    torso = np.nanmean(np.linalg.norm(np.nanmean(out[:, 11:13], 1) - np.nanmean(out[:, 23:25], 1), axis=1))
    out = out + rng.normal(0, 0.012 * torso, out.shape)
    out = out + np.cumsum(rng.normal(0, 0.002 * torso, (T, 1, 2)), axis=0)
    # 6) mirror (left-handed shooter)
    if rng.random() < 0.5:
        out[:, :, 0] = 2 * centre[0] - out[:, :, 0]
        for a, b in MIRROR_PAIRS:
            out[:, [a, b]] = out[:, [b, a]]
        side = "left" if side == "right" else "right"
    return out.astype(np.float32), side


def shot_view(px: np.ndarray, fps: float, side: str) -> np.ndarray:
    """Landmarks -> time-normalised (N_FRAMES, F) features (used for augmented views)."""
    feats, torso, _, ex = per_frame_features(px, side, fps, extras=True)
    ev = detect_events(feats, fps, ex["wrist_above_shoulder"])
    rebaseline_hip_rise(feats, ex, ev, torso, fps)
    return resample(feats, ev.onset, ev.follow_through_end)


def build_view_pool(shots: list[Shot], views_per_shot: int, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    pool = []
    for s in shots:
        views = [s.seq]
        for _ in range(views_per_shot - 1):
            px, side = augment_landmarks(s.landmarks_px, s.fps, rng, s.side)
            try:
                views.append(shot_view(px, s.fps, side))
            except Exception:
                views.append(s.seq)
        pool.append(np.stack(views))
    return pool


# ------------------------------------------------------------------ train

def nt_xent(z: torch.Tensor, labels: torch.Tensor, temperature: float = 0.25) -> torch.Tensor:
    """Supervised-contrastive loss: positives share a label (same source shot)."""
    sim = z @ z.t() / temperature
    n = z.shape[0]
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, -1e9)
    pos = (labels[:, None] == labels[None, :]) & ~eye
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    loss = -(log_prob * pos).sum(1) / pos.sum(1).clamp(min=1)
    return loss.mean()


def train_encoder(shots: list[Shot], stats: FeatureStats, steps: int = 250, views_per_shot: int = 24,
                  batch_views: int = 6, seed: int = 0, recon_weight: float = 1.0,
                  verbose: bool = True) -> tuple[ShotEncoder, dict]:
    torch.manual_seed(seed)
    t0 = time.time()
    pool = build_view_pool(shots, views_per_shot, seed)
    pool_t = [torch.from_numpy(stats.apply(v)) for v in pool]
    model = ShotEncoder()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    rng = np.random.default_rng(seed)
    history = []
    model.train()
    for step in range(steps):
        xs, ys = [], []
        for k, views in enumerate(pool_t):
            idx = rng.choice(views.shape[0], size=min(batch_views, views.shape[0]), replace=False)
            xs.append(views[idx])
            ys.append(torch.full((len(idx),), k))
        x = torch.cat(xs)
        y = torch.cat(ys)
        # feature-level jitter on top of landmark-level augmentation
        x = x + 0.05 * torch.randn_like(x)
        z = model(x)
        loss_c = nt_xent(z, y)
        loss_r = F_.mse_loss(model.reconstruct(z), ShotEncoder.coarse_profile(x, model.n_bins))
        loss = loss_c + recon_weight * loss_r
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        history.append(float(loss.detach()))
        if verbose and (step % 50 == 0 or step == steps - 1):
            print(f"[model] step {step:4d}  loss {float(loss.detach()):.4f} (contrastive {float(loss_c.detach()):.3f}, recon {float(loss_r.detach()):.3f})")
    model.eval()
    info = dict(steps=steps, final_loss=history[-1], train_seconds=time.time() - t0,
                n_shots=len(shots), views_per_shot=views_per_shot, loss_history=history)
    if verbose:
        print(f"[model] trained in {info['train_seconds']:.1f}s, final loss {info['final_loss']:.4f}")
    return model, info


@torch.no_grad()
def embed(model: ShotEncoder, seq: np.ndarray, stats: FeatureStats) -> np.ndarray:
    model.eval()
    x = torch.from_numpy(stats.apply(seq))[None]
    return model(x)[0].cpu().numpy()


def save_model(path: str, model: ShotEncoder, stats: FeatureStats, info: dict | None = None) -> None:
    torch.save(dict(state_dict=model.state_dict(), stats=stats.to_dict(), info=info or {},
                    feature_names=FEATURE_NAMES, n_frames=N_FRAMES), path)


def load_model(path: str) -> tuple[ShotEncoder, FeatureStats, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = ShotEncoder(n_features=len(ck["feature_names"]))
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, FeatureStats(ck["stats"]["mean"], ck["stats"]["std"]), ck.get("info", {})


# -------------------------------------------------------------------- DTW

def dtw_similarity(a: np.ndarray, b: np.ndarray, band: int | None = None) -> tuple[float, list[tuple[int, int]]]:
    """
    Dynamic time warping between two (T, F) z-scored sequences.
    Returns (normalised distance = total path cost / path length, warping path).
    """
    n, m = a.shape[0], b.shape[0]
    band = band if band is not None else max(n, m) // 4
    cost = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        lo, hi = max(1, i - band), min(m, i + band)
        for j in range(lo, hi + 1):
            D[i, j] = cost[i - 1, j - 1] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    # backtrack
    i, j = n, m
    path = [(i - 1, j - 1)]
    while i > 1 or j > 1:
        steps = [(D[i - 1, j - 1], i - 1, j - 1), (D[i - 1, j], i - 1, j), (D[i, j - 1], i, j - 1)]
        _, i, j = min(steps, key=lambda s: s[0])
        path.append((i - 1, j - 1))
    path.reverse()
    return float(D[n, m] / len(path)), path


def dtw_score(distance: float, scale: float = 2.2, power: float = 1.5) -> float:
    """
    Map a normalised DTW distance (z-units per step) to 0..100.
    Calibrated on the synthetic library: augmented copies of the same shot land at ~80-95,
    shots with clearly different mechanics at ~35-55.
    """
    return float(100.0 * np.exp(-((distance / scale) ** power)))


def cosine_score(z1: np.ndarray, z2: np.ndarray) -> float:
    """Map cosine similarity of unit embeddings to 0..100 (sharpened around high similarity)."""
    c = float(np.clip(np.dot(z1, z2), -1, 1))
    return float(100.0 * ((c + 1) / 2) ** 2)
