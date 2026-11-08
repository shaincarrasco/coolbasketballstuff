"""
report.py — mobile-first HTML report for the Jumpshot Analyzer (see ../DESIGN_BRIEF.md).

Two outputs from the same renderer:

* ``output/report.html`` — standalone: the shared shell stylesheet (site/assets/shell.css) is
  embedded inline, so the file works on its own from ``file://``. No top bar (that is rendered
  by shell.js on the site build).
* ``<site>/index.html`` (``--site <dir>``) — links ``../assets/shell.css`` + ``../assets/shell.js``
  with relative paths and copies ``side_by_side.mp4`` next to it. ``<body data-app="jumpshot">``
  lets shell.js draw the top bar / app switcher and pick the amber accent.

Visual rules (from the brief): dark surfaces, one accent (amber) used only for the hero score,
active controls, the selected tile and the user skeleton; the reference ghost is ``--muted``;
good / bad deltas are ``--pos`` / ``--neg`` against the per-metric tolerances in compare.py.
Every section carries a plain-English ``.read-line`` computed from the data. No external
requests: all JS and JSON are inline, key frames are base64 PNGs.
"""
from __future__ import annotations

import base64
import datetime as dt
import html
import json
import math
import os
import shutil

import cv2
import numpy as np

from compare import FEEDBACK_SPEC
from features import FEATURE_NAMES, FEATURE_UNITS, canonical_landmarks

HERE = os.path.dirname(os.path.abspath(__file__))
SHELL_CSS = os.path.normpath(os.path.join(HERE, "..", "site", "assets", "shell.css"))
TOL_OK = {spec[0]: spec[3] for spec in FEEDBACK_SPEC}
TOL_MAJOR = {spec[0]: spec[4] for spec in FEEDBACK_SPEC}
LABELS = {spec[0]: spec[1] for spec in FEEDBACK_SPEC}
WORDS = {spec[0]: spec[5] for spec in FEEDBACK_SPEC}           # (user bigger, user smaller)
SHORT = {"elbow_angle_set": "Elbow @ set", "elbow_tuck_set": "Elbow tuck @ set", "knee_angle_min": "Knee bend (min)",
         "hip_angle_set": "Hip alignment @ set", "trunk_lean_release": "Trunk lean @ release", "release_height": "Release height",
         "release_forward": "Release forward", "release_time_s": "Load → release", "follow_through_hold_s": "Follow-through hold",
         "follow_through_wrist_angle": "Wrist snap", "jump_height": "Jump height", "elbow_angle_release": "Elbow @ release"}

# Colours used inside SVG/canvas (same values as shell.css tokens — SVG fill cannot read vars in
# every renderer, and the canvas needs literal strings).
C_ACCENT, C_MUTED, C_FAINT, C_LINE, C_TEXT, C_SURF2 = "#ffa030", "#8a93a6", "#5b6478", "#242a36", "#e6e9f0", "#181c25"
C_POS, C_NEG, C_WARN = "#3ddc84", "#ff5c7a", "#ffb020"

# --------------------------------------------------------------------------- page css

PAGE_CSS = """/* Jumpshot Analyzer — page-specific additions on top of shell.css */
.sec-h{flex-wrap:wrap}.sec-h h2{white-space:nowrap}
.kpi .sub .d{font-family:var(--mono);font-variant-numeric:tabular-nums}
.kpi .name{font-size:18px;font-weight:700;line-height:1.15;letter-spacing:-.01em}
.flagrow{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}
.chip.warn{color:var(--warn);border-color:color-mix(in srgb,var(--warn) 45%,transparent)}
.chip.pos,.chip.neg,.chip.warn,.chip.accent{font-size:12px}
/* readability floor: 12px everywhere */
.eyebrow,.kpi .label,th,.chip,.foot,.rank,.chart text,.lbl{font-size:12px}
/* scrubber */
.scrub{padding:0;overflow:hidden}
.scrub .stage{position:relative;background:var(--surface-2);border-bottom:1px solid var(--line)}
.scrub canvas{display:block;width:100%;height:auto}
.scrub .stage .tag{position:absolute;top:10px;font:600 12px var(--mono);letter-spacing:.06em;text-transform:uppercase}
.scrub .stage .tag.u{left:12px;color:var(--accent)}.scrub .stage .tag.r{right:12px;color:var(--muted)}
.scrub .panel{padding:12px 14px 14px;display:flex;flex-direction:column;gap:12px;min-width:0}
.scrub .tline{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.scrub .tline .t{font:700 22px var(--mono);font-variant-numeric:tabular-nums;letter-spacing:-.01em}
.scrub .tline .f{color:var(--muted);font:500 12px var(--mono)}
.scrub .seg button{height:40px;min-width:44px;padding:0 12px}
.scrub .btn{min-width:44px;justify-content:center}
.scrub .btn.on{border-color:var(--accent);color:var(--accent)}
.scrub .row{display:flex;flex-wrap:wrap;gap:8px;align-items:center}
.scrub .lbl{font:600 12px var(--mono);letter-spacing:.06em;text-transform:uppercase;color:var(--muted);margin-right:2px}
.timeline{position:relative;padding-top:22px}
.timeline input[type=range]{height:40px;margin:0;display:block}
.timeline .marks{position:absolute;left:0;right:0;top:0;height:20px;pointer-events:none}
.timeline .mk{position:absolute;top:0;transform:translateX(-50%);font:600 12px var(--mono);color:var(--muted);white-space:nowrap}
.timeline .mk::after{content:"";display:block;width:1px;height:6px;margin:1px auto 0;background:var(--faint)}
.timeline .mk.rel{color:var(--text)}
.ro-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:6px}
.ro{background:var(--surface-2);border:1px solid var(--line);border-radius:var(--r-sm);padding:8px 9px;min-width:0}
.ro .k{font-size:12px;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.ro .v{font:600 15px var(--mono);font-variant-numeric:tabular-nums;margin-top:2px}
.ro .d{font:500 12px var(--mono);font-variant-numeric:tabular-nums;color:var(--muted)}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:var(--muted)}
.legend i{display:inline-block;width:16px;height:3px;border-radius:2px;vertical-align:middle;margin-right:6px}
.legend i.dash{height:0;border-top:2px dashed var(--muted);background:none}
@media(min-width:900px){.scrub{display:grid;grid-template-columns:minmax(0,1.15fr) minmax(320px,1fr)}
  .scrub .stage{border-bottom:0;border-right:1px solid var(--line)}}
/* radar + plan + sessions */
.radar svg{max-width:440px;margin:0 auto}
.mech td:first-child{white-space:normal;min-width:120px}
.plan{list-style:none;margin:0;padding:0;display:grid;gap:8px;counter-reset:d}
.plan li{display:grid;grid-template-columns:28px 1fr;gap:10px;padding:12px 14px;background:var(--surface);border:1px solid var(--line);border-radius:var(--r)}
.plan li::before{counter-increment:d;content:counter(d);font:700 18px var(--mono);color:var(--muted);line-height:1.2}
.plan .dn{font-weight:700;font-size:15px}
.plan .rx{font:500 13px var(--mono);color:var(--text);margin:2px 0 6px}
.plan .why{font-size:13px;color:var(--muted);display:flex;flex-wrap:wrap;gap:6px 8px;align-items:center}
.trend{display:grid;grid-template-columns:1fr 1fr;gap:10px 14px;margin-top:12px}
@media(min-width:900px){.sess{display:grid;grid-template-columns:1fr 1fr;gap:20px;align-items:start}.sess .trend{margin-top:0}.plan{grid-template-columns:1fr 1fr}}
.trend ul{list-style:none;margin:4px 0 0;padding:0;font-size:13px}
.trend li{display:flex;justify-content:space-between;gap:8px;padding:5px 0;border-bottom:1px solid var(--line)}
.trend li:last-child{border-bottom:0}
.spark{display:block;width:100%;height:auto;max-width:560px}
.frames{display:grid;grid-template-columns:1fr;gap:10px}
@media(min-width:700px){.frames{grid-template-columns:repeat(3,1fr)}}
.frame{background:var(--surface-2);border:1px solid var(--line);border-radius:var(--r-sm);overflow:hidden}
.frame .pair{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:var(--line)}
.frame img{width:100%;display:block}
.frame .cap{padding:8px 10px;display:flex;justify-content:space-between;gap:8px;font-size:12px;color:var(--muted)}
video{width:100%;border:1px solid var(--line);border-radius:var(--r-sm);background:#000;display:block}
.curves{display:grid;grid-template-columns:1fr 1fr;gap:8px}
@media(min-width:700px){.curves{grid-template-columns:repeat(3,1fr)}}
.curve{background:var(--surface-2);border:1px solid var(--line);border-radius:var(--r-sm);padding:8px 10px}
.curve .k{display:flex;justify-content:space-between;gap:6px;font-size:12px;color:var(--muted)}
.curve .k span:first-child{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.curve .k span:last-child{white-space:nowrap}
.curve svg{display:block;width:100%;height:auto}
.refs{display:grid;gap:8px}
.refs .tile{display:grid;grid-template-columns:1fr auto;gap:4px 10px;align-items:center;cursor:default}
.refs .tile .nm{font-weight:600;font-size:13px}
.refs .tile .sc{font:700 20px var(--mono);font-variant-numeric:tabular-nums}
.refs .tile.on .sc{color:var(--accent)}
.refs .tile .bar{grid-column:1/-1}
.refs .tile .meta{grid-column:1/-1;font-size:12px;color:var(--muted)}
.stack{display:grid;gap:12px}
.card h3{margin:0 0 8px;font-size:14px;font-weight:700}
.note{font-size:12px;color:var(--muted);margin:8px 0 0}
.standalone body{padding-top:0}
"""

# --------------------------------------------------------------------------- helpers


def _b64_png(img_bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img_bgr)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def _fmt(v: float, unit: str, signed: bool = False) -> str:
    sign = "+" if (signed and v > 0) else ""
    if unit == "deg":
        return f"{sign}{v:.0f}°"
    if unit == "s":
        return f"{sign}{v:.2f} s"
    if unit == "torso":
        return f"{sign}{v:.2f} torso"
    return f"{sign}{v:.2f}"


def _sev_class(sev: str) -> str:
    return {"ok": "pos", "minor": "warn", "major": "neg"}.get(sev, "")


def _delta_class(key: str, d: float) -> str:
    a = abs(d)
    if a <= TOL_OK[key]:
        return "pos"
    return "neg" if a >= TOL_MAJOR[key] else "warn"


def _short(name: str) -> str:
    """'Reference C — quick release' -> 'Reference C'."""
    return name.split(" — ")[0].split(" - ")[0].strip()


def _gap_units(fb: dict) -> float:
    return abs(float(fb["delta"])) / TOL_OK[fb["key"]]


def _phrase(fb: dict) -> str:
    """'7° wider (less loaded)' from a feedback item."""
    bigger, smaller = WORDS[fb["key"]]
    return f"{_fmt(abs(fb['delta']), fb['unit'])} {bigger if fb['delta'] > 0 else smaller}"


def _read(text: str) -> str:
    return f'<p class="read-line"><b>Read:</b> {text}</p>'


# --------------------------------------------------------------------------- svg atoms


def _curve_svg(u: np.ndarray, r: np.ndarray, w: int = 160, h: int = 56) -> str:
    lo = float(min(u.min(), r.min()))
    hi = float(max(u.max(), r.max()))
    span = (hi - lo) or 1.0

    def path(a, col, extra=""):
        pts = " ".join(f"{i / (len(a) - 1) * w:.1f},{h - 4 - (v - lo) / span * (h - 8):.1f}" for i, v in enumerate(a))
        return f'<polyline fill="none" stroke="{col}" stroke-width="1.75" stroke-linejoin="round" {extra} points="{pts}"/>'

    return (f'<svg viewBox="0 0 {w} {h}" preserveAspectRatio="none" aria-hidden="true">'
            f'{path(r, C_MUTED, "stroke-dasharray=\'4 3\'")}{path(u, C_ACCENT)}</svg>')


def _radar_svg(radar: dict, best_name: str) -> str:
    n = len(radar["axes"])
    W, H = 440, 300
    cx, cy, R = W / 2, H / 2 + 4, 100
    ang = [(-math.pi / 2 + 2 * math.pi * i / n) for i in range(n)]

    def pt(i, r):
        return cx + math.cos(ang[i]) * r, cy + math.sin(ang[i]) * r

    def ring(k):
        return " ".join(f"{pt(i, R * k)[0]:.1f},{pt(i, R * k)[1]:.1f}" for i in range(n))

    rings = "".join(f'<polygon points="{ring(k)}" fill="none" stroke="{C_LINE}" stroke-width="1"/>' for k in (0.25, 0.5, 0.75, 1.0))
    spokes = "".join(f'<line x1="{cx}" y1="{cy}" x2="{pt(i, R)[0]:.1f}" y2="{pt(i, R)[1]:.1f}" stroke="{C_LINE}"/>' for i in range(n))

    def poly(vals, stroke, fill, extra="", width=2):
        pts = " ".join(f"{pt(i, R * max(0.02, min(1, v)))[0]:.1f},{pt(i, R * max(0.02, min(1, v)))[1]:.1f}" for i, v in enumerate(vals))
        return f'<polygon points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="{width}" stroke-linejoin="round" {extra}/>'

    dots = "".join(
        f'<circle cx="{pt(i, R * max(0.02, min(1, v)))[0]:.1f}" cy="{pt(i, R * max(0.02, min(1, v)))[1]:.1f}" r="3.5" fill="{C_ACCENT}">'
        f'<title>{html.escape(a["label"])}: you {_fmt(radar["raw"]["user"][i], a["unit"])} · {html.escape(_short(best_name))} '
        f'{_fmt(radar["raw"]["closest"][i], a["unit"])} · library mean {_fmt(radar["raw"]["library_mean"][i], a["unit"])}</title></circle>'
        for i, (v, a) in enumerate(zip(radar["user"], radar["axes"])))
    labels = []
    for i, a in enumerate(radar["axes"]):
        x, y = pt(i, R + 18)
        c = math.cos(ang[i])
        anchor = "middle" if abs(c) < 0.25 else ("start" if c > 0 else "end")
        labels.append(f'<text x="{x:.1f}" y="{y + 4:.1f}" text-anchor="{anchor}">{html.escape(a["label"])}</text>')
    return (f'<svg class="chart" viewBox="0 0 {W} {H}" role="img" aria-label="radar chart of mechanics features">{rings}{spokes}'
            f'{poly(radar["library_mean"], C_FAINT, "none", "stroke-dasharray=\'3 4\'", 1.25)}'
            f'{poly(radar["closest"], C_MUTED, "rgba(138,147,166,.10)", "", 1.5)}'
            f'{poly(radar["user"], C_ACCENT, "rgba(255,160,48,.16)")}{dots}{"".join(labels)}</svg>')


def _sessions_svg(history: list[dict], w: int = 360, h: int = 150) -> str:
    if not history:
        return ""
    scores = [float(e["score"]) for e in history]
    n = len(scores)
    pad_l, pad_r, pad_t, pad_b = 36, 26, 20, 24
    lo, hi = max(0.0, min(scores) - 8), min(100.0, max(scores) + 8)
    span = (hi - lo) or 1.0

    def X(i):
        return pad_l + (i / max(1, n - 1)) * (w - pad_l - pad_r) if n > 1 else w / 2

    def Y(s):
        return pad_t + (1 - (s - lo) / span) * (h - pad_t - pad_b)

    pts = " ".join(f"{X(i):.1f},{Y(s):.1f}" for i, s in enumerate(scores))
    area = f"{X(0):.1f},{h - pad_b} " + pts + f" {X(n - 1):.1f},{h - pad_b}"
    grid = "".join(f'<line class="grid" x1="{pad_l}" x2="{w - pad_r}" y1="{Y(v):.1f}" y2="{Y(v):.1f}"/>'
                   f'<text x="{pad_l - 6}" y="{Y(v) + 4:.1f}" text-anchor="end">{v:.0f}</text>' for v in (lo, hi))
    dots, xl = [], []
    for i, e in enumerate(history):
        last = i == n - 1
        dots.append(f'<circle cx="{X(i):.1f}" cy="{Y(scores[i]):.1f}" r="{5 if last else 3.5}" fill="{C_ACCENT if last else C_SURF2}" '
                    f'stroke="{C_ACCENT}" stroke-width="2"><title>{html.escape(e.get("name", ""))} · {html.escape(str(e.get("timestamp", ""))[:10])} · {scores[i]:.1f}</title></circle>')
        dots.append(f'<text x="{X(i):.1f}" y="{Y(scores[i]) - 10:.1f}" text-anchor="middle" style="fill:{C_TEXT if last else C_MUTED};font-weight:600">{scores[i]:.0f}</text>')
        if n <= 6 or i % max(1, n // 5) == 0 or last:
            xl.append(f'<text x="{X(i):.1f}" y="{h - 8}" text-anchor="middle">{html.escape(str(e.get("timestamp", ""))[5:10])}</text>')
    return (f'<svg class="chart spark" viewBox="0 0 {w} {h}" role="img" aria-label="form score over sessions">{grid}'
            f'<polygon points="{area}" fill="rgba(255,160,48,.10)"/>'
            f'<polyline points="{pts}" fill="none" stroke="{C_ACCENT}" stroke-width="2" stroke-linejoin="round"/>'
            f'{"".join(dots)}{"".join(xl)}</svg>')


def _trends(history: list[dict]) -> tuple[list[tuple], list[tuple], list[tuple], list[tuple]]:
    """(improving, worsening, best_now, worst_now) as (label, value) lists — deltas normalised by tolerance."""
    if not history:
        return [], [], [], []
    first, last = history[0].get("deltas", {}), history[-1].get("deltas", {})
    keys = [k for k in last if k in TOL_OK]
    change = {k: abs(first.get(k, last[k])) / TOL_OK[k] - abs(last[k]) / TOL_OK[k] for k in keys}  # + = closer to ref
    now = {k: abs(last[k]) / TOL_OK[k] for k in keys}
    L = {k: SHORT.get(k, LABELS[k]) for k in keys}
    improving = sorted(((L[k], v) for k, v in change.items() if v > 0.05), key=lambda t: -t[1])[:3]
    worsening = sorted(((L[k], v) for k, v in change.items() if v < -0.05), key=lambda t: t[1])[:3]
    best_now = sorted(((L[k], v) for k, v in now.items()), key=lambda t: t[1])[:3]
    worst_now = sorted(((L[k], v) for k, v in now.items()), key=lambda t: -t[1])[:3]
    return improving, worsening, best_now, worst_now


# --------------------------------------------------------------------------- scrubber data


def _scrub_payload(user_shot, ref_shot) -> dict:
    from features import mirror_landmarks
    from overlay import align_frames
    u_px, u_side, u_facing, u_mir = canonical_landmarks(user_shot.landmarks_px, user_shot.width, user_shot.side, user_shot.facing)
    r_px, r_side, r_facing, r_mir = canonical_landmarks(ref_shot.landmarks_px, ref_shot.width, ref_shot.side, ref_shot.facing)
    if r_facing != u_facing:
        r_px = mirror_landmarks(r_px, ref_shot.width)
        r_side = "left" if r_side == "right" else "right"
        r_mir = not r_mir
    align = align_frames(user_shot, ref_shot)

    def pack(shot, px, side):
        ev = shot.events
        first = px[: max(2, int(shot.fps * 0.15))]
        hip = (first[:, 23] + first[:, 24]) / 2
        ank = (first[:, 27] + first[:, 28]) / 2
        return dict(name=shot.name, fps=shot.fps, side=side,
                    torso=float(shot.torso_px), cx=float(np.nanmedian(hip[:, 0])), floor=float(np.nanmedian(ank[:, 1])),
                    lm=np.round(np.nan_to_num(px, nan=-1.0), 1).tolist(),
                    feats=np.round(shot.per_frame, 2).tolist(),
                    events=dict(onset=ev.onset, set_point=ev.set_point, peak=ev.peak_wrist_velocity, release=ev.release,
                                follow_through=ev.follow_through, follow_through_end=ev.follow_through_end))

    return dict(features=FEATURE_NAMES, units=[FEATURE_UNITS[n] for n in FEATURE_NAMES],
                user=dict(pack(user_shot, u_px, u_side), mirrored=u_mir), ref=dict(pack(ref_shot, r_px, r_side), mirrored=r_mir),
                align=align.tolist())


SCRUB_JS = r"""
(function(){
  const D = window.SHOT_DATA;
  const css = getComputedStyle(document.body);
  const tok = n => (css.getPropertyValue(n) || '').trim();
  const COL = { accent: tok('--accent') || '#ffa030', muted: tok('--muted') || '#8a93a6', text: tok('--text') || '#e6e9f0',
                line: tok('--line-2') || '#2e3545', bg: tok('--surface-2') || '#181c25' };
  const BONES = [[11,12],[11,23],[12,24],[23,24],[11,13],[13,15],[15,17],[15,19],[15,21],[17,19],[12,14],[14,16],[16,18],[16,20],[16,22],[18,20],
                 [23,25],[25,27],[27,29],[27,31],[29,31],[24,26],[26,28],[28,30],[28,32],[30,32],[0,2],[0,5],[2,7],[5,8],[9,10]];
  const ARM_R = new Set(["12-14","14-16","16-18","16-20","16-22","18-20"]);
  const ARM_L = new Set(["11-13","13-15","15-17","15-19","15-21","17-19"]);
  const cv = document.getElementById('scrub-canvas'), ctx = cv.getContext('2d');
  const stage = cv.parentElement;
  const slider = document.getElementById('scrub-range');
  const tEl = document.getElementById('scrub-time'), fEl = document.getElementById('scrub-frame'), phEl = document.getElementById('scrub-phase');
  const ro = document.getElementById('scrub-readouts');
  const playBtn = document.getElementById('scrub-play'), ghostBtn = document.getElementById('scrub-ghost');
  const T = D.user.lm.length;
  let frame = Math.min(D.user.events.release, T-1), playing = false, speed = 0.5, ghost = true, last = 0, acc = 0;
  let W = 300, H = 300, FLOOR_Y = 260, SCALE = 50;
  slider.max = T - 1; slider.value = frame;

  function size(){
    const w = Math.max(200, stage.clientWidth), h = Math.round(Math.min(560, Math.max(300, w * (w < 600 ? 1.0 : 0.82))));
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    W = w; H = h; cv.style.height = h + 'px'; cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    FLOOR_Y = H - 26; SCALE = (H - 48) / 4.4;               // ~4.4 torso-lengths from floor to fingertips at release
    render();
  }
  function xf(shot){ return p => { const s = SCALE / shot.torso; return [W/2 + (p[0]-shot.cx)*s, FLOOR_Y + (p[1]-shot.floor)*s]; }; }
  const xfU = xf(D.user), xfR = xf(D.ref);

  function pad2(n){ return (n<10?'0':'')+n; }
  function phase(f){ const e=D.user.events;
    if (f < e.onset) return 'stance'; if (f < e.set_point) return 'load'; if (f < e.release) return 'lift';
    if (f === e.release) return 'release'; if (f <= e.follow_through_end) return 'follow-through'; return 'recovery'; }
  function fmt(v,u){ return u==='deg' ? v.toFixed(0)+'°' : u==='s' ? v.toFixed(2)+' s' : v.toFixed(2); }

  function drawFloor(){
    ctx.fillStyle = COL.bg; ctx.fillRect(0,0,W,H);
    ctx.strokeStyle = COL.line; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(0, FLOOR_Y + 8.5); ctx.lineTo(W, FLOOR_Y + 8.5); ctx.stroke();
    ctx.setLineDash([2,5]); ctx.globalAlpha = .6;
    for (let k = 1; k <= 4; k++){ const y = FLOOR_Y + 8.5 - k * SCALE; ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke(); }
    ctx.setLineDash([]); ctx.globalAlpha = 1;
    ctx.fillStyle = COL.muted; ctx.font = '12px ui-monospace, Menlo, monospace'; ctx.textAlign = 'left';
    ctx.fillText('1 torso', 8, FLOOR_Y + 8.5 - SCALE - 4);
  }
  function drawSkel(pts, map, side, st){
    const arm = side==='left' ? ARM_L : ARM_R;
    const P = pts.map(p => (p[0] < 0 ? null : map(p)));
    ctx.lineCap = 'round'; ctx.lineJoin = 'round'; ctx.setLineDash(st.dash || []);
    for (const [a,b] of BONES){ if (!P[a] || !P[b]) continue; const isArm = arm.has(a+'-'+b), face = a<11||b<11;
      ctx.strokeStyle = st.col; ctx.globalAlpha = (face ? .5 : 1) * st.alpha; ctx.lineWidth = isArm ? st.w + 1.5 : st.w;
      ctx.beginPath(); ctx.moveTo(P[a][0],P[a][1]); ctx.lineTo(P[b][0],P[b][1]); ctx.stroke(); }
    ctx.setLineDash([]);
    if (st.joints){ for (let i=11;i<33;i++){ if(!P[i]) continue; ctx.globalAlpha = st.alpha;
      ctx.fillStyle = st.col; ctx.beginPath(); ctx.arc(P[i][0],P[i][1], 3.2, 0, 7); ctx.fill();
      ctx.fillStyle = COL.bg; ctx.beginPath(); ctx.arc(P[i][0],P[i][1], 1.3, 0, 7); ctx.fill(); } }
    if (P[0] && P[7] && P[8]){ const ex=(P[7][0]+P[8][0])/2, ey=(P[7][1]+P[8][1])/2; const r=Math.hypot(P[0][0]-ex,P[0][1]-ey)+3;
      ctx.globalAlpha = st.alpha; ctx.strokeStyle = st.col; ctx.lineWidth = st.w; ctx.setLineDash(st.dash||[]);
      ctx.beginPath(); ctx.arc(ex,ey,r,0,7); ctx.stroke(); ctx.setLineDash([]); }
    ctx.globalAlpha = 1;
  }
  function render(){
    drawFloor();
    const rf = D.align[frame];
    if (ghost) drawSkel(D.ref.lm[rf], xfR, D.ref.side, {col: COL.muted, alpha: .7, w: 2.5, dash: [6,5], joints: false});
    drawSkel(D.user.lm[frame], xfU, D.user.side, {col: COL.accent, alpha: 1, w: 3.5, joints: true});
    tEl.textContent = (frame / D.user.fps).toFixed(2) + ' s';
    fEl.textContent = 'f' + pad2(frame) + ' / ' + pad2(T-1) + ' · ref f' + pad2(rf);
    phEl.textContent = phase(frame);
    let h = '';
    for (let i=0;i<D.features.length;i++){ const u = D.user.feats[frame][i], r = D.ref.feats[rf][i], d = u-r, un = D.units[i];
      const tol = un==='deg' ? 5 : 0.06; const cls = Math.abs(d) <= tol ? 'pos' : (Math.abs(d) >= tol * 2.5 ? 'neg' : 'warn');
      h += '<div class="ro"><div class="k" title="'+D.features[i]+'">'+D.features[i].replace('_',' ')+'</div><div class="v">'+fmt(u,un)+'</div>'
         + '<div class="d">ref '+fmt(r,un)+' <span class="'+cls+'">'+(d>0?'+':'')+fmt(d,un)+'</span></div></div>'; }
    ro.innerHTML = h;
    if (+slider.value !== frame) slider.value = frame;
    document.querySelectorAll('[data-jump]').forEach(x => x.classList.toggle('on', D.user.events[x.dataset.jump] === frame));
  }
  function setFrame(f){ frame = Math.max(0, Math.min(T-1, f|0)); render(); }
  function pause(){ playing = false; playBtn.textContent = 'Play'; playBtn.classList.remove('primary'); }
  function play(){ playing = true; playBtn.textContent = 'Pause'; playBtn.classList.add('primary'); last = performance.now(); acc = 0; requestAnimationFrame(tick); }
  function tick(now){ if (!playing) return; acc += (now - last) * speed / 1000; last = now; const step = 1/D.user.fps;
    while (acc >= step){ acc -= step; frame = (frame + 1) % T; } render(); requestAnimationFrame(tick); }
  slider.addEventListener('input', e => { pause(); setFrame(+e.target.value); });
  playBtn.addEventListener('click', () => playing ? pause() : play());
  document.getElementById('scrub-prev').addEventListener('click', () => { pause(); setFrame(frame-1); });
  document.getElementById('scrub-next').addEventListener('click', () => { pause(); setFrame(frame+1); });
  ghostBtn.addEventListener('click', () => { ghost = !ghost; ghostBtn.classList.toggle('on', ghost); ghostBtn.setAttribute('aria-pressed', ghost); render(); });
  document.querySelectorAll('[data-jump]').forEach(b => b.addEventListener('click', () => { pause(); setFrame(D.user.events[b.dataset.jump]); }));
  document.querySelectorAll('[data-speed]').forEach(b => b.addEventListener('click', () => { speed = +b.dataset.speed;
    document.querySelectorAll('[data-speed]').forEach(x => x.classList.toggle('on', x===b)); }));
  document.addEventListener('keydown', e => { const t = document.activeElement.tagName;
    if (['INPUT','TEXTAREA','VIDEO','SELECT'].includes(t) && document.activeElement !== slider) return;
    if (e.key === 'ArrowRight'){ pause(); setFrame(frame + (e.shiftKey?5:1)); e.preventDefault(); }
    else if (e.key === 'ArrowLeft'){ pause(); setFrame(frame - (e.shiftKey?5:1)); e.preventDefault(); }
    else if (e.key === 'Home'){ pause(); setFrame(0); } else if (e.key === 'End'){ pause(); setFrame(T-1); }
    else if (e.key === ' ' && t !== 'BUTTON' && t !== 'SUMMARY'){ playing ? pause() : play(); e.preventDefault(); } });
  if ('ResizeObserver' in window) new ResizeObserver(size).observe(stage); else window.addEventListener('resize', size);
  window.scrubSetFrame = setFrame;   // for tests / console
  size();

  // progressive disclosure: details that are collapsed on mobile open on desktop
  const mq = window.matchMedia('(min-width: 900px)');
  function sync(){ document.querySelectorAll('details[data-desktop-open]').forEach(d => { d.open = mq.matches; }); }
  sync(); if (mq.addEventListener) mq.addEventListener('change', sync);
  // the shell fades the right edge of .tbl as a scroll hint; drop it when nothing overflows
  function tbl(){ document.querySelectorAll('.tbl').forEach(t => { const o = t.scrollWidth > t.clientWidth + 1; t.style.maskImage = t.style.webkitMaskImage = o ? '' : 'none'; }); }
  tbl(); window.addEventListener('resize', tbl); document.querySelectorAll('details').forEach(d => d.addEventListener('toggle', tbl));
})();
"""


# --------------------------------------------------------------------------- page


def _render(results: dict, user_shot, ref_shot, frames: list[dict], *, video_name: str, model_info: dict,
            history: list[dict], batch_rows: list[dict], plan: list[dict], site: bool) -> str:
    best = results["references"][0]
    others = results["references"][1:]
    score = float(results["headline_score"])
    u = results["user"]
    e = html.escape
    ev = u["events"]
    s = u["summary"]
    q = u.get("quality", {}) or {}
    mi = model_info or {}
    fps = float(u["fps"])
    T = int(u["n_frames"])
    fb_by_key = {fb["key"]: fb for fb in best["feedback"]}
    ranked = sorted(best["feedback"], key=_gap_units, reverse=True)
    worst = ranked[0]
    n_major = sum(fb["severity"] == "major" for fb in best["feedback"])
    n_minor = sum(fb["severity"] == "minor" for fb in best["feedback"])
    n_ok = len(best["feedback"]) - n_major - n_minor
    best_short = _short(best["name"])
    best_tag = best["name"].split(" — ")[1] if " — " in best["name"] else ""

    # ---- scrubber payload
    payload = _scrub_payload(user_shot, ref_shot)
    data_json = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    marks = [("onset", "onset", ""), ("set", "set_point", ""), ("release", "release", "rel"), ("end", "follow_through_end", "")]
    mark_html = "".join(f'<span class="mk {cls}" style="left:{ev[key] / max(1, T - 1) * 100:.1f}%">{lbl}</span>' for lbl, key, cls in marks)

    # ---- quality
    conf = q.get("confidence", "ok")
    flags = q.get("flags", [])
    conf_cls = {"ok": "pos", "low": "warn", "minor": "warn", "major": "neg"}.get(conf, "warn")
    flag_chips = "".join(f'<span class="chip {_sev_class(fl["severity"])}" title="{e(fl["message"])}">{e(fl["code"].replace("_", " "))}</span>' for fl in flags)
    flag_list = "".join(f'<li>{e(fl["message"])}</li>' for fl in flags) or \
        "<li>No quality flags: whole body visible, enough frames, release detected at full arm extension.</li>"

    # ---- KPI tiles
    rt, rh = fb_by_key["release_time_s"], fb_by_key["release_height"]

    def tile_sub(fb):
        return (f'<span class="d {_delta_class(fb["key"], fb["delta"])}">{_fmt(fb["delta"], fb["unit"], signed=True)}</span> '
                f'vs {e(best_short)} ({_fmt(fb["reference"], fb["unit"])})')

    runner = f"next {e(_short(others[0]['name']))} at {others[0]['score']:.0f}" if others else "only reference"
    kpis = f"""
    <div class="grid g4">
      <div class="card kpi"><div class="label">Form score</div><b class="hero">{score:.0f}<span style="font-size:16px;color:var(--muted)">/100</span></b>
        <div class="sub">vs {e(best_short)} · DTW {best['dtw_score']:.0f} · learned {best['learned_score'] if best['learned_score'] is not None else '—'}</div></div>
      <div class="card kpi"><div class="label">Closest reference</div><div class="name">{e(best_short)}</div>
        <div class="sub">{e(best_tag) + ' · ' if best_tag else ''}{runner}</div></div>
      <div class="card kpi"><div class="label">Load → release</div><b>{s['release_time_s']:.2f}<span style="font-size:14px;color:var(--muted)"> s</span></b>
        <div class="sub">{tile_sub(rt)}</div></div>
      <div class="card kpi"><div class="label">Release height</div><b>{s['release_height']:+.2f}<span style="font-size:14px;color:var(--muted)"> torso</span></b>
        <div class="sub">{tile_sub(rh)}</div></div>
    </div>
    <div class="flagrow"><span class="chip {conf_cls}">confidence {e(conf)}</span><span class="chip">hand {e(u['side'])}{'' if q.get('side_detected', True) else ' (forced)'}</span>
      <span class="chip">{T} f · {fps:.0f} fps</span>{flag_chips}</div>
    {_read(f"Your form matches <b>{e(best_short)}</b> at <b>{score:.0f}/100</b>. The biggest gap is {e(worst['label'].lower())}: "
           f"{e(_phrase(worst))} than the reference; {n_major} major and {n_minor} minor finding{'s' if n_major + n_minor != 1 else ''}, {n_ok} within tolerance.")}"""

    # ---- scrubber
    rel_u = float(payload["user"]["feats"][ev["release"]][FEATURE_NAMES.index("wrist_height")])
    rel_r = float(payload["ref"]["feats"][payload["align"][ev["release"]]][FEATURE_NAMES.index("wrist_height")])
    mirror_note = ""
    if payload["user"]["mirrored"] or payload["ref"]["mirrored"]:
        who = " and ".join(n for n, m in (("user", payload["user"]["mirrored"]), ("reference", payload["ref"]["mirrored"])) if m)
        mirror_note = f" The {who} skeleton is mirrored so both are right-handed and face the same way."
    scrub = f"""
    <div class="card scrub">
      <div class="stage"><canvas id="scrub-canvas" width="360" height="360" aria-label="your skeleton with the time-warped reference ghost"></canvas>
        <span class="tag u">You</span><span class="tag r">{e(best_short)} ghost</span></div>
      <div class="panel">
        <div class="tline"><span class="t" id="scrub-time">0.00 s</span><span class="f" id="scrub-frame">f00</span><span class="chip accent" id="scrub-phase">—</span></div>
        <div class="timeline"><div class="marks">{mark_html}</div>
          <input type="range" id="scrub-range" min="0" max="{T - 1}" value="{ev['release']}" step="1" aria-label="frame"></div>
        <div class="row">
          <button class="btn" id="scrub-prev" aria-label="previous frame">‹</button>
          <button class="btn" id="scrub-play" style="min-width:84px">Play</button>
          <button class="btn" id="scrub-next" aria-label="next frame">›</button>
          <div class="seg" role="group" aria-label="speed"><button data-speed="0.25">¼×</button><button class="on" data-speed="0.5">½×</button><button data-speed="1">1×</button></div>
          <button class="btn on" id="scrub-ghost" aria-pressed="true">Ghost</button>
        </div>
        <div class="row"><span class="lbl">Jump</span>
          <div class="seg" role="group" aria-label="jump to event"><button data-jump="onset">Onset</button><button data-jump="set_point">Set</button>
            <button class="on" data-jump="release">Release</button><button data-jump="follow_through">Follow</button></div></div>
        <div class="ro-grid" id="scrub-readouts"></div>
        <div class="legend"><span><i style="background:var(--accent)"></i>you</span><span><i class="dash"></i>{e(best_short)} (time-warped)</span>
          <span>← → step · shift ×5 · space play</span></div>
      </div>
    </div>
    {_read(f"At release (f{ev['release']}, {ev['release'] / fps:.2f} s) your wrist is <b>{rel_u:.2f} torso</b> above the head vs {rel_r:.2f} for {e(best_short)}; "
           f"both figures are scaled to the same torso length and share a floor line, so the ghost is a true overlay.{e(mirror_note)}")}"""

    # ---- mechanics: radar + table
    rows = []
    for fb in best["feedback"]:
        rows.append(f"""<tr><td title="{e(fb['label'])}">{e(SHORT.get(fb['key'], fb['label']))}</td>
            <td class="n {_delta_class(fb['key'], fb['delta'])}">{_fmt(fb['delta'], fb['unit'], signed=True)}</td>
            <td><span class="chip {_sev_class(fb['severity'])}">{fb['severity']}</span></td>
            <td class="n">{_fmt(fb['user'], fb['unit'])}</td><td class="n">{_fmt(fb['reference'], fb['unit'])}</td></tr>""")
    second = ranked[1]
    mech = f"""
    <div class="grid g2">
      <div class="card radar">{_radar_svg(results['radar'], best['name'])}
        <div class="legend" style="margin-top:6px"><span><i style="background:var(--accent)"></i>you</span><span><i style="background:var(--muted)"></i>{e(best_short)}</span>
          <span><i class="dash" style="border-color:var(--faint)"></i>library mean</span></div>
        <p class="note">Axes are min–max normalised over library + you; the rim is the highest value seen, not "best". Tap a dot for raw values.</p></div>
      <div class="tbl mech"><table>
        <thead><tr><th>Metric</th><th class="n">Δ you − ref</th><th>Status</th><th class="n">You</th><th class="n">Ref</th></tr></thead>
        <tbody>{''.join(rows)}</tbody></table></div>
    </div>
    {_read(f"Two gaps dominate: {e(worst['label'].lower())} ({e(_phrase(worst))}) and {e(second['label'].lower())} ({e(_phrase(second))}). "
           f"Green deltas are inside the per-metric tolerance; red ones are at least the 'major' threshold.")}"""

    # ---- coaching plan
    plan_items = "".join(f"""<li><div>
        <div class="dn">{e(p['drill'])}</div><div class="rx">{e(p['prescription'])}</div>
        <div class="why"><span class="chip {_sev_class(p['severity'])}">{e(p['finding'])}{(' ' + _fmt(p['delta'], p['unit'], signed=True)) if p['unit'] else ''}</span>
          <span>{e(p['cue'])}</span></div></div></li>""" for p in plan)
    plan_read = (f"Start with <b>{e(plan[0]['drill'])}</b> — it targets the largest gap ({e(plan[0]['finding'].lower())}). "
                 f"{len(plan)} drill{'s' if len(plan) != 1 else ''} ordered major → minor; drills are generic form-shooting work, not professional coaching."
                 if plan else "No drills: every metric is within tolerance of the closest reference.")
    coaching = f'<ol class="plan">{plan_items}</ol>{_read(plan_read)}'

    # ---- sessions
    improving, worsening, best_now, worst_now = _trends(history)

    def tl(items, cls, arrow):
        if not items:
            return '<li class="muted">nothing to report yet</li>'
        return "".join(f'<li><span>{e(lbl)}</span><span class="num {cls}">{arrow} {abs(v):.1f}×</span></li>' for lbl, v in items)

    sess_rows = "".join(f"""<tr><td class="num">{e(str(h.get('timestamp', ''))[:10])}</td>
        <td class="n"><b>{float(h.get('score', 0)):.1f}</b></td><td>{e(h.get('name', ''))}</td><td>{e(_short(h.get('closest', '')))}</td>
        <td class="n">{sum(v == 'major' for v in h.get('severities', {}).values())} / {sum(v == 'minor' for v in h.get('severities', {}).values())}</td>
        <td><span class="chip {({'ok': 'pos'}).get(h.get('confidence', 'ok'), 'warn')}">{e(h.get('confidence', 'ok'))}</span></td>
        <td><span class="chip">{e(h.get('label', ''))}</span></td></tr>"""
                        for h in reversed(history[-12:]))
    if len(history) >= 2:
        d_score = float(history[-1]["score"]) - float(history[0]["score"])
        d_cls = "pos" if d_score >= 0 else "neg"
        sess_read = (f"<b>{len(history)} sessions</b>: score {float(history[0]['score']):.0f} → {float(history[-1]['score']):.0f} "
                     f"(<span class=\"{d_cls}\">{d_score:+.0f}</span>), best {max(float(h['score']) for h in history):.0f}. "
                     + (f"Most improved: {e(improving[0][0].lower())}." if improving else "")
                     + (f" Watch: {e(worsening[0][0].lower())} is drifting away from the reference." if worsening else ""))
    else:
        sess_read = f"{len(history)} session recorded — run analyze again to start a trend."
    sessions = f"""
      <div class="stack">
        <div class="card sess"><div><h3>Form score by session</h3>{_sessions_svg(history)}</div>
          <div class="trend">
            <div><div class="label lbl">Most improved</div><ul>{tl(improving, 'pos', '▲')}</ul></div>
            <div><div class="label lbl">Trending worse</div><ul>{tl(worsening, 'neg', '▼')}</ul></div>
            <div><div class="label lbl">Best now</div><ul>{tl(best_now, 'pos', '●')}</ul></div>
            <div><div class="label lbl">Worst now</div><ul>{tl(worst_now, 'neg', '●')}</ul></div>
          </div>
          <p class="note">Values are |you − reference| in tolerance units (1.0× = the "ok" threshold). Trend = first session vs latest.</p></div>
        <div class="tbl"><table>
          <thead><tr><th>Date</th><th class="n">Score</th><th>Clip</th><th>Closest</th><th class="n">Maj / min</th><th>Conf.</th><th>Label</th></tr></thead>
          <tbody>{sess_rows}</tbody></table></div>
        {_read(sess_read)}
      </div>"""

    # ---- film: video + key frames
    frame_cards = "".join(f"""
        <div class="frame"><div class="pair"><img src="{_b64_png(fr_['user_img'])}" alt="you at {e(fr_['label'])}" loading="lazy">
            <img src="{_b64_png(fr_['ref_img'])}" alt="reference at {e(fr_['label'])}" loading="lazy"></div>
          <div class="cap"><span>{e(fr_['label'])}</span><span class="num">you f{fr_['user_frame']} · ref f{fr_['ref_frame']}</span></div></div>""" for fr_ in frames)
    film = f"""
      <div class="stack">
        <video playsinline muted controls preload="metadata" loop src="{e(video_name)}"></video>
        <div class="frames">{frame_cards}</div>
        {_read(f"Left is you with the skeleton, right is {e(best_short)} time-warped to your shot phases, so the same phase shows on both sides.")}
      </div>"""

    # ---- evidence: references, curves, quality
    ref_tiles = "".join(f"""<div class="tile {'on' if r is best else ''}"><span class="nm">{e(r['name'])}</span><span class="sc">{r['score']:.0f}</span>
        <div class="bar"><i style="width:{r['score']:.0f}%;background:{'var(--text)' if r is best else 'var(--faint)'}"></i></div>
        <span class="meta">DTW {r['dtw_score']:.0f} · learned {'—' if r['learned_score'] is None else f"{r['learned_score']:.0f}"} · distance {r['dtw_distance']:.2f}</span></div>"""
                        for r in results["references"])
    curves = "".join(f"""<div class="curve"><div class="k"><span>{e(name.replace('_', ' '))}</span><span class="num">DTW {best['feature_distance'][name]:.2f}</span></div>
        {_curve_svg(user_shot.seq[:, i], ref_shot.seq[:, i])}</div>""" for i, name in enumerate(FEATURE_NAMES))
    spread = (results["references"][0]["score"] - results["references"][-1]["score"]) if len(results["references"]) > 1 else 0
    evidence = f"""
      <div class="stack">
        <div class="card"><h3>All references</h3><div class="refs">{ref_tiles}</div>
          <p class="note">Score = {results['weights']['dtw']:.0%} DTW on z-scored feature sequences + {results['weights']['learned']:.0%} cosine similarity of learned embeddings.</p></div>
        <div class="card"><h3>Time-normalised feature curves</h3><div class="curves">{curves}</div>
          <p class="note">60 samples onset → follow-through; amber you, dashed grey {e(best_short)}.</p></div>
        <div class="card"><h3>Pose quality</h3>
          <div class="grid g4">
            <div class="kpi" style="padding:0"><div class="label">Body visible</div><b style="font-size:20px">{q.get('body_visibility_pct', 0):.0f}%</b></div>
            <div class="kpi" style="padding:0"><div class="label">Arm visible</div><b style="font-size:20px">{q.get('arm_visibility_pct', 0):.0f}%</b></div>
            <div class="kpi" style="padding:0"><div class="label">Frames w/ pose</div><b style="font-size:20px">{q.get('n_frames', T)}<span class="muted" style="font-size:13px">/{T}</span></b></div>
            <div class="kpi" style="padding:0"><div class="label">Release found</div><b style="font-size:20px" class="{'pos' if q.get('release_found', True) else 'neg'}">{'yes' if q.get('release_found', True) else 'no'}</b></div>
          </div>
          <ul style="margin:10px 0 0;padding-left:18px;font-size:13px;color:var(--muted)">{flag_list}</ul></div>
        {_read(f"{e(best_short)} wins by {spread:.0f} points over the lowest-ranked reference; the curves show where in the motion the match breaks down.")}
      </div>"""

    # ---- batch
    batch = ""
    if batch_rows:
        brows = []
        scored = [r for r in batch_rows if not r.get("error")]
        for r in sorted(batch_rows, key=lambda r: -float(r.get("score") or -1)):
            if r.get("error"):
                brows.append(f'<tr><td class="num">{e(r["clip"])}</td><td colspan="6" class="neg">failed: {e(r["error"])}</td></tr>')
                continue
            sc = float(r["score"])
            brows.append(f"""<tr><td>{e(r.get('name', '') or r['clip'])}<div class="muted" style="font-size:12px">{e(r['clip'])}</div></td>
              <td class="n"><b class="num">{sc:.1f}</b><div class="bar" style="margin-top:5px;width:72px"><i style="width:{sc:.0f}%;background:var(--faint)"></i></div></td>
              <td>{e(_short(r.get('closest', '')))}</td><td class="n">{e(str(r.get('n_major', '')))} / {e(str(r.get('n_minor', '')))}</td>
              <td><span class="chip {({'ok': 'pos'}).get(r.get('confidence', 'ok'), 'warn')}">{e(r.get('confidence', 'ok'))}</span></td>
              <td>{e(str(r.get('side', '')))} · {e(str(r.get('frames', '')))} f</td>
              <td style="white-space:normal;min-width:220px;font-size:12px;color:var(--muted)">{e(r.get('top_findings', ''))}</td></tr>""")
        if scored:
            top = max(scored, key=lambda r: float(r["score"]))
            low = min(scored, key=lambda r: float(r["score"]))
            b_read = (f"{len(scored)} of {len(batch_rows)} clips scored; best <b>{e(top.get('name', '') or top['clip'])}</b> at {float(top['score']):.0f}, "
                      f"lowest {e(low.get('name', '') or low['clip'])} at {float(low['score']):.0f}. Source: batch.csv.")
        else:
            b_read = "No clip in the batch could be scored."
        batch = f"""
    <section class="sec">
      <details class="more" data-desktop-open><summary><span>Batch comparison <span class="muted" style="font-weight:500">· {len(batch_rows)} clips</span></span></summary>
        <div class="body"><div class="tbl"><table>
          <thead><tr><th>Clip</th><th class="n">Score</th><th>Closest</th><th class="n">Maj / min</th><th>Conf.</th><th>Hand · frames</th><th>Top findings</th></tr></thead>
          <tbody>{''.join(brows)}</tbody></table></div>{_read(b_read)}</div></details>
    </section>"""

    # ---- method & limits
    source_note = ("Landmarks came from the synthetic sidecar (no MediaPipe pose model in this environment)."
                   if "synthetic" in str(u.get("source", "")) else f"Landmarks from {e(str(u.get('source', '')))}.")
    method = f"""
    <div class="method"><b>Method &amp; limits.</b> MediaPipe-layout pose landmarks (33 per frame) → 9 body-size-normalised mechanics features per frame
      (lengths in torso-lengths, shoulder-mid to hip-mid; 1 torso ≈ 50 cm for an adult) → shot events (onset f{ev['onset']}, set point f{ev['set_point']},
      peak wrist velocity f{ev['peak_wrist_velocity']}, release f{ev['release']}, follow-through end f{ev['follow_through_end']}) → similarity =
      {results['weights']['dtw']:.0%} DTW on z-scored sequences + {results['weights']['learned']:.0%} cosine of a contrastive 1-D conv embedding
      ({mi.get('steps', '?')} steps, {mi.get('train_seconds', 0):.1f} s CPU, final loss {mi.get('final_loss', float('nan')):.3f}). Tolerances per metric come from
      compare.py and are heuristics, not biomechanical norms. {e(source_note)} All demo clips are procedurally generated: no real footage, no real
      player data, no real player names. 2-D pose only — depth, ball flight and outcome are not measured. Not medical or professional coaching advice.
      Generated {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}.</div>
    <div class="foot"><span>Sample data</span><span>Synthetic clips</span><span>Python</span><span>OpenCV</span><span>MediaPipe</span><span>PyTorch</span></div>"""

    # ---- assemble
    if site:
        head_css = '<link rel="stylesheet" href="../assets/shell.css">'
        shell_js = '<script src="../assets/shell.js"></script>'
        pwa = ('<link rel="manifest" href="../manifest.webmanifest">'
               '<meta name="apple-mobile-web-app-capable" content="yes">'
               '<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">'
               '<link rel="apple-touch-icon" href="../icons/apple-touch-icon.png">')
        html_cls = ""
    else:
        with open(SHELL_CSS, encoding="utf-8") as f:
            head_css = f"<style>{f.read()}</style>"
        shell_js = ""
        pwa = ""
        html_cls = ' class="standalone"'

    return f"""<!DOCTYPE html>
<html lang="en"{html_cls}><head><meta charset="utf-8"><title>Jumpshot Analyzer — NBA Projects</title>
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="#0b0d12">{pwa}
{head_css}
<style>{PAGE_CSS}</style></head>
<body data-app="jumpshot" data-title="Jumpshot Analyzer" data-mark="JS">
<main class="wrap">
  <div class="eyebrow">NBA Projects · 01 · Computer vision</div>
  <h1>Jumpshot Analyzer</h1>
  <p class="lede">One shot's mechanics, measured from body pose and scored against a reference library — the biggest gap and what to drill, first.</p>

  {kpis}

  <section class="sec"><div class="sec-h"><h2>Frame scrubber</h2><span class="read">you vs the time-warped reference</span></div>{scrub}</section>

  <section class="sec"><div class="sec-h"><h2>Mechanics</h2><span class="read">vs {e(best_short)}</span></div>{mech}</section>

  <section class="sec"><div class="sec-h"><h2>Coaching plan</h2><span class="read">{len(plan)} drills</span></div>{coaching}</section>

  <section class="sec">
    <details class="more" data-desktop-open><summary><span>Sessions <span class="muted" style="font-weight:500">· {len(history)} runs</span></span></summary>
      <div class="body">{sessions}</div></details>
  </section>

  <section class="sec">
    <details class="more"><summary><span>Film <span class="muted" style="font-weight:500">· side-by-side video · key frames</span></span></summary>
      <div class="body">{film}</div></details>
  </section>

  <section class="sec">
    <details class="more"><summary><span>Evidence <span class="muted" style="font-weight:500">· references · curves · pose quality</span></span></summary>
      <div class="body">{evidence}</div></details>
  </section>
  {batch}
  {method}
</main>
<script>window.SHOT_DATA = {data_json};</script>
<script>{SCRUB_JS}</script>
{shell_js}
</body></html>"""


def write_report(results: dict, user_shot, ref_shot, frames: list[dict], out_dir: str,
                 video_name: str = "side_by_side.mp4", model_info: dict | None = None, refs=None,
                 history: list[dict] | None = None, batch_rows: list[dict] | None = None,
                 plan: list[dict] | None = None, site_dir: str | None = None) -> str:
    """Write out_dir/report.html (standalone). If site_dir is given, also write site_dir/index.html
    linked to ../assets/shell.{css,js} and copy the side-by-side video next to it."""
    os.makedirs(out_dir, exist_ok=True)
    if plan is None:
        from drills import coaching_plan
        plan = coaching_plan(results["references"][0]["feedback"])
    kw = dict(video_name=video_name, model_info=model_info or {}, history=history or [], batch_rows=batch_rows or [], plan=plan)
    path = os.path.join(out_dir, "report.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(_render(results, user_shot, ref_shot, frames, site=False, **kw))
    stale = os.path.join(out_dir, "theme.css")       # written by the previous ("Film Room") version
    if os.path.isfile(stale):
        os.remove(stale)
    if site_dir:
        os.makedirs(site_dir, exist_ok=True)
        src_video = os.path.join(out_dir, video_name)
        if os.path.isfile(src_video):
            shutil.copyfile(src_video, os.path.join(site_dir, video_name))
        with open(os.path.join(site_dir, "index.html"), "w", encoding="utf-8") as f:
            f.write(_render(results, user_shot, ref_shot, frames, site=True, **kw))
    return path
