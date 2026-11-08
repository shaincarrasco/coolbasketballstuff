// engine.mjs — Jumpshot Analyzer analysis engine (pure JS, no DOM; browser + Node >= 18).
//
// Faithful port of the Python pipeline in 01-jumpshot-analyzer/:
//   features.py  visibility mask, gap fill, Savitzky-Golay smoothing, torso normalisation, the 9
//                per-frame features, shooting-hand / facing detection, mirror canonicalisation,
//                shot events (dip -> set -> extension -> release -> follow-through), quality flags,
//                60-frame resampling, per-shot summary
//   model.py     z-scoring, DTW (Sakoe-Chiba band = max(n,m)//4) + score mapping, ShotEncoder forward
//                pass (conv1d / batchnorm / GELU / mean+max pool / linear / L2 normalise), cosine score
//   compare.py   0.6 * DTW + 0.4 * learned, per-feature feedback with the same tolerances, radar
//   drills.py    coaching plan
// Constants come from library.json (exported by export_web.py) where they are data; the algorithm
// constants are copied verbatim from features.py. Parity with Python is checked by tests/parity.test.mjs.
//
// Input frames: [{t, lm: Float32Array|Array(132) = 33 x (x, y, z, visibility)} | {t, lm:null} | null],
// x, y normalised image coords (0..1, y down). They are converted to pixels with meta.width/height
// before any angle or length is computed (same as Python's to_pixels), so non-square video is handled.
// Time-based quantities use t: clips with a steady frame interval are analysed frame-by-frame exactly
// like Python; jittery / variable frame timing is first resampled onto a uniform time grid.

export const ENGINE_VERSION = '2.0.0';

// ------------------------------------------------------------------ constants (verbatim from Python)

const NUM_LM = 33;
const LM = {
  NOSE: 0, LEFT_EAR: 7, RIGHT_EAR: 8, LEFT_SHOULDER: 11, RIGHT_SHOULDER: 12, LEFT_ELBOW: 13, RIGHT_ELBOW: 14,
  LEFT_WRIST: 15, RIGHT_WRIST: 16, LEFT_INDEX: 19, RIGHT_INDEX: 20, LEFT_HIP: 23, RIGHT_HIP: 24,
  LEFT_KNEE: 25, RIGHT_KNEE: 26, LEFT_ANKLE: 27, RIGHT_ANKLE: 28,
};
const MIRROR_PAIRS = [[1, 4], [2, 5], [3, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16],
  [17, 18], [19, 20], [21, 22], [23, 24], [25, 26], [27, 28], [29, 30], [31, 32]];
const MIRROR_OF = (() => { const m = Array.from({ length: NUM_LM }, (_, i) => i); for (const [a, b] of MIRROR_PAIRS) { m[a] = b; m[b] = a; } return m; })();
const SIDES = {
  right: { shoulder: 12, elbow: 14, wrist: 16, index: 20, hip: 24, knee: 26, ankle: 28 },
  left: { shoulder: 11, elbow: 13, wrist: 15, index: 19, hip: 23, knee: 25, ankle: 27 },
};
const FEATURE_NAMES = ['elbow_angle', 'knee_angle', 'hip_angle', 'trunk_lean', 'wrist_height',
  'wrist_forward', 'elbow_tuck', 'wrist_angle', 'hip_rise'];
const N_FRAMES = 60;
const MIN_VIS = 0.5;
const MIN_GOOD_FRAMES = 24;
const MIN_POSE_FRAMES = 8;
const BODY_LANDMARKS = Array.from({ length: 22 }, (_, i) => 11 + i);
const W_DTW = 0.6, W_LEARNED = 0.4;            // compare.py (overridden by library.weights if present)
const AUTO_WINDOW_MIN_S = 3.5;                 // longer clips without opts.window -> detectShots window

/** Ordered per-frame features. tolerance = "ok" tolerance of the summary metric that represents the
 *  feature on the radar (compare.RADAR_AXES / FEEDBACK_SPEC); higherIsBetter is null for all: the
 *  score measures similarity to the reference, not "more is better". */
export const FEATURES = [
  { key: 'elbow_angle', label: 'Elbow angle', unit: 'deg', tolerance: 5, higherIsBetter: null, metric: 'elbow_angle_set' },
  { key: 'knee_angle', label: 'Knee angle', unit: 'deg', tolerance: 6, higherIsBetter: null, metric: 'knee_angle_min' },
  { key: 'hip_angle', label: 'Hip angle', unit: 'deg', tolerance: 6, higherIsBetter: null, metric: 'hip_angle_set' },
  { key: 'trunk_lean', label: 'Trunk lean', unit: 'deg', tolerance: 3, higherIsBetter: null, metric: 'trunk_lean_release' },
  { key: 'wrist_height', label: 'Wrist height', unit: 'torso', tolerance: 0.06, higherIsBetter: null, metric: 'release_height' },
  { key: 'wrist_forward', label: 'Wrist forward', unit: 'torso', tolerance: 0.08, higherIsBetter: null, metric: 'release_forward' },
  { key: 'elbow_tuck', label: 'Elbow tuck', unit: 'torso', tolerance: 0.06, higherIsBetter: null, metric: 'elbow_tuck_set' },
  { key: 'wrist_angle', label: 'Wrist angle', unit: 'deg', tolerance: 8, higherIsBetter: null, metric: 'follow_through_wrist_angle' },
  { key: 'hip_rise', label: 'Hip rise', unit: 'torso', tolerance: 0.08, higherIsBetter: null, metric: 'jump_height' },
];

// compare.FEEDBACK_SPEC (fallback if library.json lacks `metrics`)
const FEEDBACK_SPEC = [
  ['elbow_angle_set', 'Elbow angle at set point', 'deg', 5, 12, ['wider (less loaded)', 'tighter (more loaded)']],
  ['elbow_tuck_set', 'Elbow in front of shoulder at set point', 'torso', 0.06, 0.15, ['further out', 'more tucked']],
  ['knee_angle_min', 'Deepest knee bend', 'deg', 6, 15, ['shallower', 'deeper']],
  ['hip_angle_set', 'Shoulder-hip-knee alignment at set point', 'deg', 6, 15, ['more upright', 'more folded']],
  ['trunk_lean_release', 'Trunk lean at release', 'deg', 3, 8, ['more forward', 'more upright']],
  ['release_height', 'Release height (wrist above head)', 'torso', 0.06, 0.15, ['higher', 'lower']],
  ['release_forward', 'Release point in front of shoulder', 'torso', 0.08, 0.2, ['further forward', 'further back']],
  ['release_time_s', 'Load-to-release time', 's', 0.06, 0.15, ['later (slower)', 'earlier (quicker)']],
  ['follow_through_hold_s', 'Follow-through hold', 's', 0.08, 0.2, ['longer', 'shorter']],
  ['follow_through_wrist_angle', 'Wrist snap (elbow-wrist-finger angle)', 'deg', 8, 20, ['flatter (less snap)', 'sharper (more snap)']],
  ['jump_height', 'Jump height (hip rise)', 'torso', 0.08, 0.2, ['higher', 'lower']],
  ['elbow_angle_release', 'Elbow extension at release', 'deg', 5, 12, ['more extended', 'less extended']],
].map(([key, label, unit, tolOk, tolMajor, w]) => ({ key, label, unit, tolOk, tolMajor, bigger: w[0], smaller: w[1] }));

const RADAR_AXES = [
  ['elbow_angle', 'elbow_angle_set', 'Elbow @ set'], ['knee_angle', 'knee_angle_min', 'Knee min'],
  ['hip_angle', 'hip_angle_set', 'Hip align'], ['trunk_lean', 'trunk_lean_release', 'Trunk lean'],
  ['wrist_height', 'release_height', 'Release height'], ['wrist_forward', 'release_forward', 'Release fwd'],
  ['elbow_tuck', 'elbow_tuck_set', 'Elbow tuck'], ['wrist_angle', 'follow_through_wrist_angle', 'Wrist snap'],
  ['hip_rise', 'jump_height', 'Jump height'],
].map(([feature, key, label]) => ({ feature, key, label, unit: FEATURES.find(f => f.key === feature).unit }));

const JOINT_WORDS = { shoulder: 'shoulder', elbow: 'elbow', wrist: 'wrist', index: 'hand', hip: 'hip', knee: 'knee', ankle: 'ankle', nose: 'head' };

// ------------------------------------------------------------------ numeric helpers (numpy semantics)

const f32 = Math.fround;
const isNum = v => typeof v === 'number' && Number.isFinite(v);

/** Python's round(): half to even. */
function pyRound(x) {
  const r = Math.round(x);
  if (Math.abs(x % 1) === 0.5) return 2 * Math.round(x / 2);
  return r;
}
const pyInt = x => Math.trunc(x);

function nanMedian(a, lo = 0, hi = a.length) {
  const v = [];
  for (let i = lo; i < hi; i++) if (!Number.isNaN(a[i])) v.push(a[i]);
  if (!v.length) return NaN;
  v.sort((p, q) => p - q);
  const m = v.length >> 1;
  return v.length % 2 ? v[m] : (v[m - 1] + v[m]) / 2;
}
function nanMin(a) { let m = Infinity, any = false; for (const v of a) if (!Number.isNaN(v)) { any = true; if (v < m) m = v; } return any ? m : NaN; }
function argmax(a, lo = 0, hi = a.length) { let bi = lo, bv = -Infinity; for (let i = lo; i < hi; i++) if (a[i] > bv) { bv = a[i]; bi = i; } return bi; }
function argmin(a, lo = 0, hi = a.length) { let bi = lo, bv = Infinity; for (let i = lo; i < hi; i++) if (a[i] < bv) { bv = a[i]; bi = i; } return bi; }
function gradient(a) {
  const n = a.length, g = new Float64Array(n);
  if (n < 2) return g;
  g[0] = a[1] - a[0]; g[n - 1] = a[n - 1] - a[n - 2];
  for (let i = 1; i < n - 1; i++) g[i] = (a[i + 1] - a[i - 1]) / 2;
  return g;
}
/** np.interp for increasing xp. */
function interp1(x, xp, fp) {
  const n = xp.length;
  if (x <= xp[0]) return fp[0];
  if (x >= xp[n - 1]) return fp[n - 1];
  let lo = 0, hi = n - 1;
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (xp[m] <= x) lo = m; else hi = m; }
  const d = xp[hi] - xp[lo];
  return d === 0 ? fp[hi] : fp[lo] + (fp[hi] - fp[lo]) * (x - xp[lo]) / d;
}
function linspace01(n) { const out = new Float64Array(n); if (n === 1) { out[0] = 0; return out; } const step = 1 / (n - 1); for (let i = 0; i < n; i++) out[i] = i * step; out[n - 1] = 1; return out; }
function angleDeg(ax, ay, bx, by, cx, cy) {
  const v1x = ax - bx, v1y = ay - by, v2x = cx - bx, v2y = cy - by;
  const c = (v1x * v2x + v1y * v2y) / (Math.hypot(v1x, v1y) * Math.hypot(v2x, v2y) + 1e-9);
  return Math.acos(Math.min(1, Math.max(-1, c))) * 180 / Math.PI;
}
function erf(x) {   // Abramowitz & Stegun 7.1.26, |error| < 1.5e-7
  const s = x < 0 ? -1 : 1; x = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * x);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return s * y;
}
const gelu = x => 0.5 * x * (1 + erf(x / Math.SQRT2));

// ------------------------------------------------------------------ Savitzky-Golay (scipy, mode="interp")

const _sgCache = new Map();
function savgolWeights(win, order = 2) {
  const key = win + ':' + order;
  if (_sgCache.has(key)) return _sgCache.get(key);
  const h = (win - 1) / 2, k = order + 1;
  // least squares fit of a polynomial in u = j - h: coef = (A^T A)^-1 A^T y; value(p) = [1,u_p,u_p^2] coef
  const ATA = Array.from({ length: k }, () => new Float64Array(k));
  for (let j = 0; j < win; j++) { const u = j - h; for (let r = 0; r < k; r++) for (let c = 0; c < k; c++) ATA[r][c] += u ** (r + c); }
  const inv = invert(ATA);
  const weightsAt = p => {
    const up = p - h, row = new Float64Array(k);
    for (let c = 0; c < k; c++) { let s = 0; for (let r = 0; r < k; r++) s += up ** r * inv[r][c]; row[c] = s; }
    const w = new Float64Array(win);
    for (let j = 0; j < win; j++) { const u = j - h; let s = 0; for (let c = 0; c < k; c++) s += row[c] * u ** c; w[j] = s; }
    return w;
  };
  const res = { win, h, center: weightsAt(h), at: Array.from({ length: win }, (_, p) => weightsAt(p)) };
  _sgCache.set(key, res);
  return res;
}
function invert(M) {
  const n = M.length, A = M.map((r, i) => { const o = new Float64Array(2 * n); o.set(r); o[n + i] = 1; return o; });
  for (let c = 0; c < n; c++) {
    let p = c; for (let r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r;
    [A[c], A[p]] = [A[p], A[c]];
    const d = A[c][c]; for (let j = 0; j < 2 * n; j++) A[c][j] /= d;
    for (let r = 0; r < n; r++) if (r !== c) { const f = A[r][c]; if (f) for (let j = 0; j < 2 * n; j++) A[r][j] -= f * A[c][j]; }
  }
  return A.map(r => r.slice(n));
}
function savgolWindow(T, fps) {
  let win = pyRound(fps * 0.2) | 1;
  win = Math.max(5, Math.min(win, T - (1 - T % 2)));
  return (T < 5 || win < 5) ? 0 : win;
}
/** features.smooth for one column (float32 output like scipy on float32 input). */
function smoothCol(x, win) {
  const T = x.length, out = new Float64Array(T);
  if (!win) { out.set(x); return out; }
  for (let i = 0; i < T; i++) if (Number.isNaN(x[i])) { out.fill(NaN); return out; }
  const W = savgolWeights(win), h = W.h;
  for (let i = h; i < T - h; i++) { let s = 0; const w = W.center; for (let j = 0; j < win; j++) s += w[j] * x[i - h + j]; out[i] = f32(s); }
  for (let p = 0; p < h; p++) { let s = 0; const w = W.at[p]; for (let j = 0; j < win; j++) s += w[j] * x[j]; out[p] = f32(s); }
  for (let p = win - h; p < win; p++) { let s = 0; const w = W.at[p]; const o = T - win; for (let j = 0; j < win; j++) s += w[j] * x[o + j]; out[o + p] = f32(s); }
  return out;
}

// ------------------------------------------------------------------ input -> uniform grid of pixel landmarks

function frameLm(fr) { return fr && fr.lm && fr.lm.length >= NUM_LM * 4 ? fr.lm : null; }

/** Times of frames[lo..hi] (missing t interpolated / extrapolated). */
function frameTimes(frames, lo, hi, fpsHint) {
  const n = hi - lo + 1, t = new Float64Array(n).fill(NaN);
  for (let i = 0; i < n; i++) { const fr = frames[lo + i]; if (fr && isNum(fr.t)) t[i] = fr.t; }
  const known = new Uint8Array(n); let any = false;
  for (let i = 0; i < n; i++) if (!Number.isNaN(t[i])) { known[i] = 1; any = true; }
  const dt = 1 / (fpsHint || 30);
  if (!any) { for (let i = 0; i < n; i++) t[i] = (lo + i) * dt; return t; }
  const nextKnown = new Int32Array(n).fill(-1);
  for (let i = n - 1, nk = -1; i >= 0; i--) { nextKnown[i] = nk; if (known[i]) nk = i; }
  let a = -1;
  for (let i = 0; i < n; i++) {
    if (known[i]) { a = i; continue; }
    const b = nextKnown[i];
    if (a >= 0 && b >= 0) t[i] = t[a] + (t[b] - t[a]) * (i - a) / (b - a);
    else if (a >= 0) t[i] = t[a] + (i - a) * dt;
    else t[i] = t[b] - (b - i) * dt;
  }
  return t;
}

function quantile(a, q) { const v = Array.from(a).filter(Number.isFinite).sort((x, y) => x - y); if (!v.length) return NaN; const p = (v.length - 1) * q, i = Math.floor(p); return v[i] + (v[Math.min(i + 1, v.length - 1)] - v[i]) * (p - i); }

/** Decide the analysis grid for frames[lo..hi] (uniform in time, so frame index == t * fps):
 *   'direct'    the input frames themselves (steady frame interval: the Python-identical path)
 *   'slotted'   frames sit on a 1/fps lattice but some are absent (dropped / skipped frames): each
 *               frame goes to its slot, absent slots are gap-filled like Python's missing frames
 *   'resampled' irregular timing: landmarks linearly resampled onto a grid at the median frame rate
 *  pos[i] = fractional grid position of input frame lo+i; nearest[g] = input frame (relative) nearest
 *  to grid index g. */
function buildGrid(frames, meta, lo, hi) {
  const metaFps = isNum(meta.fps) && meta.fps > 0 ? meta.fps : 0;
  const tIn = frameTimes(frames, lo, hi, metaFps);
  const n = tIn.length;
  const dts = []; for (let i = 1; i < n; i++) if (tIn[i] > tIn[i - 1]) dts.push(tIn[i] - tIn[i - 1]);
  const q20 = dts.length ? quantile(dts, 0.2) : (metaFps ? 1 / metaFps : 1 / 30);
  let fps = metaFps && q20 > 0.6 / metaFps ? metaFps : 1 / q20;     // meta.fps unless frames come faster
  fps = Math.min(240, Math.max(5, fps));
  const pos = new Float64Array(n);
  let mode = 'direct', slotted = true;
  for (let i = 0; i < n; i++) {
    const p = (tIn[i] - tIn[0]) * fps, s = Math.round(p);
    if (Math.abs(p - s) > 0.25 || (i > 0 && s <= pos[i - 1])) { slotted = false; break; }
    pos[i] = s;
  }
  let N = n;
  if (slotted && pos[n - 1] !== n - 1) {
    if (pos[n - 1] + 1 <= 4 * n) { mode = 'slotted'; N = pos[n - 1] + 1; } else slotted = false;
  }
  let tGrid;
  if (!slotted) {
    mode = 'resampled';
    fps = Math.min(240, Math.max(5, 1 / (dts.length ? nanMedian(Float64Array.from(dts)) : 1 / 30)));
    N = Math.max(1, Math.floor((tIn[n - 1] - tIn[0]) * fps + 1e-6) + 1);
    for (let i = 0; i < n; i++) pos[i] = (tIn[i] - tIn[0]) * fps;
  }
  if (mode === 'direct') { tGrid = tIn; for (let i = 0; i < n; i++) pos[i] = i; }
  else { tGrid = new Float64Array(N); for (let k = 0; k < N; k++) tGrid[k] = tIn[0] + k / fps; }
  const nearest = new Int32Array(N);
  for (let g = 0, i = 0; g < N; g++) { while (i + 1 < n && Math.abs(pos[i + 1] - g) <= Math.abs(pos[i] - g)) i++; nearest[g] = i; }
  return { mode, uniform: mode === 'direct', fps, tIn, tGrid, n: N, pos, nearest };
}

/** Masked pixel landmarks on the grid: X[l], Y[l], V[l] Float64Array(n); NaN = missing / vis < MIN_VIS. */
function gridLandmarks(frames, lo, grid, W, H) {
  const n = grid.n;
  const X = [], Y = [], V = [];
  for (let l = 0; l < NUM_LM; l++) { X.push(new Float64Array(n).fill(NaN)); Y.push(new Float64Array(n).fill(NaN)); V.push(new Float64Array(n)); }
  const read = (i, l) => {           // input frame i (relative), landmark l -> [x, y, vis] or null
    const lm = frameLm(frames[lo + i]);
    if (!lm) return null;
    const x = +lm[4 * l], y = +lm[4 * l + 1], v = lm[4 * l + 3] == null ? 1 : +lm[4 * l + 3];
    return [x, y, isNum(v) ? v : 0];
  };
  const put = (k, l, x, y, v) => {
    V[l][k] = v;
    if (v >= MIN_VIS && isNum(x) && isNum(y)) { X[l][k] = f32(f32(x) * W); Y[l][k] = f32(f32(y) * H); }
  };
  if (grid.mode !== 'resampled') {
    for (let i = 0; i < grid.tIn.length; i++) { const k = grid.pos[i]; for (let l = 0; l < NUM_LM; l++) { const r = read(i, l); if (r) put(k, l, r[0], r[1], r[2]); } }
    return { X, Y, V };
  }
  const tIn = grid.tIn, half = 0.5 / grid.fps;
  let i = 0;
  for (let k = 0; k < n; k++) {
    const g = grid.tGrid[k];
    while (i < tIn.length - 2 && tIn[i + 1] <= g) i++;
    const ta = tIn[i], tb = tIn[Math.min(i + 1, tIn.length - 1)];
    const u = tb > ta ? Math.min(1, Math.max(0, (g - ta) / (tb - ta))) : 0;
    for (let l = 0; l < NUM_LM; l++) {
      const a = read(i, l), b = read(Math.min(i + 1, tIn.length - 1), l);
      const okA = a && a[2] >= MIN_VIS && isNum(a[0]), okB = b && b[2] >= MIN_VIS && isNum(b[0]);
      if (okA && okB) put(k, l, a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u, a[2] + (b[2] - a[2]) * u);
      else if (okA && Math.abs(g - ta) <= half) put(k, l, a[0], a[1], a[2]);
      else if (okB && Math.abs(tb - g) <= half) put(k, l, b[0], b[1], b[2]);
      else V[l][k] = Math.min(a ? a[2] : 0, b ? b[2] : 0);
    }
  }
  return { X, Y, V };
}

/** pose.fill_missing for one column over axis `t` (linear; edges = edge values; never-seen stays NaN). */
function fillCol(col, t) {
  const n = col.length, xs = [], ys = [];
  for (let i = 0; i < n; i++) if (!Number.isNaN(col[i])) { xs.push(t ? t[i] : i); ys.push(col[i]); }
  const out = Float64Array.from(col);
  if (!xs.length || xs.length === n) return out;
  for (let i = 0; i < n; i++) if (Number.isNaN(col[i])) out[i] = f32(interp1(t ? t[i] : i, xs, ys));
  return out;
}

function pairMid(ax, bx) {
  const aNa = ax.every(Number.isNaN), bNa = bx.every(Number.isNaN);
  if (aNa) return Float64Array.from(bx);
  if (bNa) return Float64Array.from(ax);
  return ax.map((v, i) => (v + bx[i]) / 2);
}

// ------------------------------------------------------------------ features.py port

function detectShootingSide(Y) {
  const rw = Y[LM.RIGHT_WRIST], lw = Y[LM.LEFT_WRIST];
  if (lw.every(Number.isNaN)) return 'right';
  if (rw.every(Number.isNaN)) return 'left';
  return nanMin(rw) <= nanMin(lw) ? 'right' : 'left';
}

function facingDirection(X, side) {
  const nose = X[LM.NOSE];
  const ears = pairMid(X[LM.LEFT_EAR], X[LM.RIGHT_EAR]);
  let d = ears.every(Number.isNaN) ? NaN : nanMedian(nose.map((v, i) => v - ears[i]));
  if (!Number.isFinite(d) || Math.abs(d) < 1e-3) {
    const s = SIDES[side];
    d = nanMedian(X[s.wrist].map((v, i) => v - X[s.shoulder][i]));
  }
  return d >= 0 ? 1 : -1;
}

function missingRequired(X, side) {
  const s = SIDES[side], out = [];
  for (const k of ['shoulder', 'elbow', 'wrist', 'index', 'hip', 'knee', 'ankle']) if (X[s[k]].every(Number.isNaN)) out.push(k);
  if (X[LM.NOSE].every(Number.isNaN)) out.push('nose');
  return out;
}

/** per_frame_features: returns {feats: Float32Array[F] (T each), torso, facing}. */
function perFrameFeatures(X, Y, side, fps) {
  const T = X[0].length, s = SIDES[side], win = savgolWindow(T, fps);
  const need = new Set([LM.NOSE, LM.LEFT_SHOULDER, LM.RIGHT_SHOULDER, LM.LEFT_HIP, LM.RIGHT_HIP, ...Object.values(s)]);
  const PX = {}, PY = {};
  for (const l of need) { PX[l] = smoothCol(X[l], win); PY[l] = smoothCol(Y[l], win); }
  const shX = pairMid(PX[LM.LEFT_SHOULDER], PX[LM.RIGHT_SHOULDER]), shY = pairMid(PY[LM.LEFT_SHOULDER], PY[LM.RIGHT_SHOULDER]);
  const hpX = pairMid(PX[LM.LEFT_HIP], PX[LM.RIGHT_HIP]), hpY = pairMid(PY[LM.LEFT_HIP], PY[LM.RIGHT_HIP]);
  let torso = nanMedian(shX.map((v, i) => Math.hypot(v - hpX[i], shY[i] - hpY[i])));
  torso = Math.max(torso, 1e-3);
  const facing = facingDirection(X, side);
  const F = FEATURE_NAMES.map(() => new Float32Array(T));
  const hipY0 = nanMedian(hpY, 0, Math.min(T, Math.max(3, pyInt(fps * 0.2))));
  const [sx, sy, ex, ey, wx, wy, ix, iy] = [PX[s.shoulder], PY[s.shoulder], PX[s.elbow], PY[s.elbow], PX[s.wrist], PY[s.wrist], PX[s.index], PY[s.index]];
  const [hx, hy, kx, ky, ax, ay] = [PX[s.hip], PY[s.hip], PX[s.knee], PY[s.knee], PX[s.ankle], PY[s.ankle]];
  const ny = PY[LM.NOSE];
  for (let i = 0; i < T; i++) {
    const tvx = shX[i] - hpX[i], tvy = shY[i] - hpY[i];
    F[0][i] = angleDeg(sx[i], sy[i], ex[i], ey[i], wx[i], wy[i]);
    F[1][i] = angleDeg(hx[i], hy[i], kx[i], ky[i], ax[i], ay[i]);
    F[2][i] = angleDeg(sx[i], sy[i], hx[i], hy[i], kx[i], ky[i]);
    F[3][i] = Math.atan2(facing * tvx, -tvy) * 180 / Math.PI;
    F[4][i] = (ny[i] - wy[i]) / torso;
    F[5][i] = facing * (wx[i] - sx[i]) / torso;
    F[6][i] = facing * (ex[i] - sx[i]) / torso;
    F[7][i] = angleDeg(ex[i], ey[i], wx[i], wy[i], ix[i], iy[i]);
    F[8][i] = (hipY0 - hpY[i]) / torso;
  }
  // extras (features.per_frame_features(..., extras=True)): event detector + mechanics inputs
  const n = T, was = new Float32Array(n), sflex = new Float32Array(n), hipXs = new Float32Array(n), hipYs = new Float32Array(n), sw = new Float32Array(n);
  const lsx = PX[LM.LEFT_SHOULDER], rsx = PX[LM.RIGHT_SHOULDER];
  for (let i = 0; i < n; i++) {
    was[i] = (sy[i] - wy[i]) / torso;
    sflex[i] = angleDeg(hpX[i] - shX[i] + sx[i], hpY[i] - shY[i] + sy[i], sx[i], sy[i], ex[i], ey[i]);
    hipXs[i] = hpX[i]; hipYs[i] = hpY[i];
    sw[i] = Math.abs(lsx[i] - rsx[i]) / torso;
  }
  return { feats: F, torso, facing, ex: { wrist_above_shoulder: was, shoulder_flex: sflex, hip_x: hipXs, hip_y: hipYs, shoulder_width: sw } };
}

// event detector v2 constants (features.py)
const EV = { LOAD_TORSO: 0.3, WARMUP_S: 0.2, SET_WINDOW_S: 0.8, FLEX_LOOKBACK_S: 0.3, MIN_FLEX_DEG: 8.0, MIN_EXT_DEG: 20.0,
  ONSET_LOOKBACK_S: 1.0, REACH_S: 0.4, RELEASE_DEG: 150.0, RELEASE_FOUND_DEG: 140.0, CUT_TORSO: 0.8 };

function releaseFrom(elbow, peakE, windowEnd, T) {
  let armMax = -Infinity; for (let i = peakE; i <= windowEnd; i++) armMax = Math.max(armMax, elbow[i]);
  const thr = Math.min(EV.RELEASE_DEG, armMax - 4.0);
  let rel = peakE;
  for (let i = peakE; i < T; i++) if (elbow[i] >= thr) { rel = i; break; }
  rel = Math.min(Math.max(rel, 1), T - 1);
  let sub = rel; const a = elbow[rel - 1], b = elbow[rel];
  if (rel > peakE && a < thr && thr <= b && b > a) sub = rel - 1 + (thr - a) / (b - a);
  return { rel, releaseFound: armMax >= EV.RELEASE_FOUND_DEG, sub };
}

/** features.scene_cuts on gap-filled pixel landmarks: frames that start a new scene. */
function sceneCuts(X, Y) {
  const T = X[0].length; if (T < 2) return [];
  const shX = pairMid(X[LM.LEFT_SHOULDER], X[LM.RIGHT_SHOULDER]), shY = pairMid(Y[LM.LEFT_SHOULDER], Y[LM.RIGHT_SHOULDER]);
  const hpX = pairMid(X[LM.LEFT_HIP], X[LM.RIGHT_HIP]), hpY = pairMid(Y[LM.LEFT_HIP], Y[LM.RIGHT_HIP]);
  const torso = nanMedian(shX.map((v, i) => Math.hypot(v - hpX[i], shY[i] - hpY[i])));
  if (!Number.isFinite(torso) || torso <= 1e-6) return [];
  const cuts = [];
  for (let i = 1; i < T; i++) if (Math.hypot(hpX[i] - hpX[i - 1], hpY[i] - hpY[i - 1]) / torso > EV.CUT_TORSO) cuts.push(i);
  return cuts;
}

/** features.detect_events v2 (anchored on the release; v1 path as fallback). */
function detectEvents(F, fps, was = null, cuts = null) {
  const T = F[0].length;
  const elbow = F[0], knee = F[1], wh = F[4];
  const v = gradient(wh).map(x => x * fps);
  const ve = gradient(elbow).map(x => x * fps);
  const warm = Math.min(Math.max(T - 3, 0), pyInt(pyRound(fps * EV.WARMUP_S)));
  let rising = null;
  if (was) {
    rising = new Uint8Array(T);
    for (let i = warm; i < T; i++) rising[i] = (was[i] >= -EV.LOAD_TORSO && v[i] > 0) ? 1 : 0;
  }
  let method = 'legacy', setPt, peakE, rel, releaseFound, s0, relSub;
  if (rising && rising.some(x => x)) {
    const reachN = pyInt(pyRound(fps * EV.REACH_S));
    const nxt = new Int32Array(T + 1); nxt[T] = T;
    for (let i = T - 1; i >= 0; i--) nxt[i] = (elbow[i] >= EV.RELEASE_FOUND_DEG && wh[i] > 0) ? i : nxt[i + 1];
    const good = []; const all = [];
    for (let i = 0; i < T; i++) if (rising[i]) { all.push(i); if (nxt[i] - i <= reachN) good.push(i); }
    const cand = good.length ? good : all;
    peakE = cand[0]; for (const i of cand) if (ve[i] > ve[peakE]) peakE = i;
    const windowEnd = Math.min(T - 1, peakE + pyInt(pyRound(fps * 0.6)));
    ({ rel, releaseFound, sub: relSub } = releaseFrom(elbow, peakE, windowEnd, T));
    let scene0 = 0; for (const c of cuts || []) if (c <= peakE && c > scene0) scene0 = c;
    const lo = Math.max(warm, scene0, rel - pyInt(pyRound(fps * EV.SET_WINDOW_S)));
    const lb = Math.max(1, pyInt(pyRound(fps * EV.FLEX_LOOKBACK_S)));
    setPt = -1;
    for (let m = Math.min(peakE, T - 2); m >= lo; m--) {
      if (m < 1 || !(was[m] >= -EV.LOAD_TORSO)) continue;
      if (!(elbow[m] <= elbow[m - 1] && elbow[m] <= elbow[m + 1])) continue;
      let mb = -Infinity; for (let k = Math.max(0, m - lb); k < m; k++) mb = Math.max(mb, elbow[k]);
      let ma = -Infinity; for (let k = m; k <= rel; k++) ma = Math.max(ma, elbow[k]);
      if (mb - elbow[m] >= EV.MIN_FLEX_DEG && ma - elbow[m] >= EV.MIN_EXT_DEG) { setPt = m; method = 'elbow_min'; break; }
    }
    if (setPt < 0) { const hi = Math.max(lo + 1, peakE); setPt = argmin(knee, lo, Math.min(hi, T)); method = 'knee_min'; }
    s0 = Math.max(0, scene0, setPt - pyInt(pyRound(fps * EV.ONSET_LOOKBACK_S)));
  } else {
    let whMax = -Infinity, whMin = Infinity; for (const x of wh) { if (x > whMax) whMax = x; if (x < whMin) whMin = x; }
    const thrTop = whMax - 0.05 * (whMax - whMin + 1e-6);
    let top = 0; for (let i = 0; i < T; i++) if (wh[i] >= thrTop) { top = i; break; }
    top = Math.max(top, 2);
    setPt = argmin(knee, 0, Math.min(top, T));
    peakE = argmax(ve, setPt, Math.min(top + 1, T));
    const windowEnd = Math.min(T - 1, Math.max(top, peakE + pyInt(pyRound(fps * 0.6))));
    ({ rel, releaseFound, sub: relSub } = releaseFrom(elbow, peakE, windowEnd, T));
    s0 = 0;
  }
  const peakV = rel >= setPt ? argmax(v, setPt, rel + 1) : rel;
  const nb = Math.max(2, pyInt(fps * 0.15));
  const startKnee = nanMedian(knee, s0, Math.min(s0 + nb, T)), startWh = nanMedian(wh, s0, Math.min(s0 + nb, T));
  let onset = s0;
  for (let i = s0; i < setPt; i++) if ((startKnee - knee[i]) > 4 || Math.abs(wh[i] - startWh) > 0.12) { onset = i; break; }
  onset = onset > s0 ? Math.max(s0, onset - 1) : s0;
  const relH = wh[rel];
  let ftEnd = rel;
  for (let i = rel; i < T; i++) { if (wh[i] >= 0.65 * relH && elbow[i] >= 140) ftEnd = i; else break; }
  const holdS = (ftEnd - rel) / fps;
  const ft = Math.min(T - 1, rel + Math.max(2, Math.floor((ftEnd - rel) / 2)));
  ftEnd = Math.min(T - 1, Math.max(ftEnd, rel + pyInt(fps * 0.2)));
  return { onset, set: setPt, peak: peakV, release: rel, followThrough: ft, ftEnd, holdS, releaseFound, peakExtension: peakE, setMethod: method, baselineStart: s0, releaseSub: relSub, warm };
}

/** features.rebaseline_hip_rise (in place). */
function rebaselineHipRise(F, ex, ev, torso, fps) {
  if (ev.baselineStart <= 0) return;
  const hy = ex.hip_y, s0 = ev.baselineStart;
  const y0 = nanMedian(hy, s0, Math.min(hy.length, s0 + Math.max(3, pyInt(fps * 0.2))));
  for (let i = 0; i < hy.length; i++) F[8][i] = (y0 - hy[i]) / torso;
}

/** features.glitch_frames: implausible skeletons (leg bone / torso ratio vs its clip median). */
function glitchFrames(Xm, Ym) {
  const T = Xm[0].length, bad = new Uint8Array(T);
  const nm = (a, b, i) => Number.isNaN(a[i]) ? b[i] : Number.isNaN(b[i]) ? a[i] : (a[i] + b[i]) / 2;
  const torso = new Float64Array(T);
  for (let i = 0; i < T; i++) {
    const sx = nm(Xm[11], Xm[12], i), sy = nm(Ym[11], Ym[12], i), hx = nm(Xm[23], Xm[24], i), hy = nm(Ym[23], Ym[24], i);
    torso[i] = Math.hypot(sx - hx, sy - hy);
  }
  for (const [a, b] of [[23, 25], [25, 27], [24, 26], [26, 28]]) {
    const r = new Float64Array(T).fill(NaN); const okv = [];
    for (let i = 0; i < T; i++) { const t = torso[i] > 1e-6 ? torso[i] : NaN; const v = Math.hypot(Xm[a][i] - Xm[b][i], Ym[a][i] - Ym[b][i]) / t; if (Number.isFinite(v)) { r[i] = v; okv.push(v); } }
    if (okv.length < 8) continue;
    const med = nanMedian(Float64Array.from(okv));
    for (let i = 0; i < T; i++) if (Number.isFinite(r[i])) { const q = r[i] / med; if (q < 0.6 || q > 1.6) bad[i] = 1; }
  }
  return bad;
}

/** features.resample: feats[start..end] -> n x F (Float32Array rows). */
function resampleRows(F, start, end, n = N_FRAMES) {
  end = Math.max(end, start + 1);
  const L = end - start + 1, ts = linspace01(L), td = linspace01(n);
  const rows = Array.from({ length: n }, () => new Float32Array(F.length));
  for (let j = 0; j < F.length; j++) {
    const seg = new Float64Array(L); for (let i = 0; i < L; i++) seg[i] = F[j][Math.min(start + i, F[j].length - 1)];
    for (let k = 0; k < n; k++) rows[k][j] = interp1(td[k], ts, seg);
  }
  return rows;
}

function summarize(F, ev, fps) {
  const f = Object.fromEntries(FEATURE_NAMES.map((n, j) => [n, F[j]]));
  const { release: rel, set: sp, followThrough: ft } = ev;
  let kmin = Infinity; for (let i = ev.onset; i <= rel; i++) kmin = Math.min(kmin, f.knee_angle[i]);
  let jmax = -Infinity; for (let i = ev.onset; i <= ev.ftEnd; i++) jmax = Math.max(jmax, f.hip_rise[i]);
  return {
    elbow_angle_set: f.elbow_angle[sp], elbow_angle_release: f.elbow_angle[rel], elbow_tuck_set: f.elbow_tuck[sp],
    knee_angle_min: kmin, knee_angle_set: f.knee_angle[sp], hip_angle_set: f.hip_angle[sp],
    trunk_lean_release: f.trunk_lean[rel], release_height: f.wrist_height[rel], release_forward: f.wrist_forward[rel],
    release_time_s: (rel - sp) / fps, release_time_from_onset_s: (rel - ev.onset) / fps,
    peak_wrist_velocity_torso_per_s: gradient(f.wrist_height)[ev.peak] * fps,
    follow_through_wrist_angle: f.wrist_angle[ft], follow_through_hold_s: ev.holdS, jump_height: jmax,
    shot_duration_s: (ev.ftEnd - ev.onset) / fps,
  };
}

/** features.assess_quality. V = raw visibility on the grid, missing = frames with no usable landmark. */
function assessQuality(V, missingMask, n, fps, ev, side) {
  let ok = 0, tot = 0;
  for (const l of BODY_LANDMARKS) for (let i = 0; i < n; i++) { tot++; if (V[l][i] >= 0.5) ok++; }
  const bodyVis = tot ? ok / tot * 100 : 0;
  const s = SIDES[side || 'right'];
  let aok = 0, atot = 0;
  for (const l of [s.shoulder, s.elbow, s.wrist]) for (let i = 0; i < n; i++) { atot++; if (V[l][i] >= 0.5) aok++; }
  const armVis = atot ? aok / atot * 100 : 0;
  const missing = missingMask.reduce((a, b) => a + (b ? 1 : 0), 0);
  const flags = [];
  const flag = (code, severity, text) => flags.push({ code, severity, text });
  if (n < MIN_GOOD_FRAMES) flag('few_frames', 'major', `Only ${n} frames with pose (${(n / fps).toFixed(1)} s) - trim to ~2 s around the shot at >= 30 fps.`);
  if (bodyVis < 70) flag('low_visibility', 'major', `Body landmark visibility ${bodyVis.toFixed(0)} % - the whole body should be in frame.`);
  else if (bodyVis < 85) flag('low_visibility', 'minor', `Body landmark visibility ${bodyVis.toFixed(0)} % - some joints were guessed/interpolated.`);
  if (armVis < 85) flag('arm_occluded', 'minor', `Shooting-arm visibility ${armVis.toFixed(0)} % - ball/defender may be covering the wrist.`);
  if (missing > 0.1 * n) flag('gaps', 'minor', `${missing} of ${n} frames had no pose and were gap-filled.`);
  if (ev) {
    if (!ev.releaseFound) flag('release_not_found', 'major', 'Release not found: the shooting arm never reached ~140 deg extension; the release frame is a guess.');
    if (ev.set <= ev.onset || ev.release - ev.set < 2 || (ev.setMethod !== 'legacy' && ev.set <= pyInt(pyRound(fps * EV.WARMUP_S)))) flag('events_degenerate', 'major', 'Set point / release are too close together - event detection is unreliable on this clip.');
    if (ev.ftEnd >= n - 1) flag('clip_cut', 'minor', 'Clip ends during the follow-through; hold time is a lower bound.');
  }
  let level = 'ok';
  for (const f of flags) { if (f.severity === 'major') level = 'major'; else if (f.severity === 'minor' && level === 'ok') level = 'minor'; }
  return { level, flags, visibility: Math.round(bodyVis * 10) / 1000, armVisibility: Math.round(armVis * 10) / 1000, frames: n, missingFrames: missing };
}

// ------------------------------------------------------------------ model.py port: DTW, encoder

function zscore(rows, stats) {
  return rows.map(r => { const o = new Float32Array(r.length); for (let j = 0; j < r.length; j++) o[j] = (r[j] - stats.mean[j]) / stats.std[j]; return o; });
}

/** model.dtw_similarity: (distance, path). */
function dtwSimilarity(a, b, band = null) {
  const n = a.length, m = b.length;
  band = band ?? Math.floor(Math.max(n, m) / 4);
  const F = a[0].length;
  const cost = Array.from({ length: n }, (_, i) => { const r = new Float64Array(m); for (let j = 0; j < m; j++) { let s = 0; for (let k = 0; k < F; k++) { const d = a[i][k] - b[j][k]; s += d * d; } r[j] = Math.sqrt(s); } return r; });
  const D = Array.from({ length: n + 1 }, () => new Float64Array(m + 1).fill(Infinity));
  D[0][0] = 0;
  for (let i = 1; i <= n; i++) {
    const lo = Math.max(1, i - band), hi = Math.min(m, i + band);
    for (let j = lo; j <= hi; j++) D[i][j] = cost[i - 1][j - 1] + Math.min(D[i - 1][j], D[i][j - 1], D[i - 1][j - 1]);
  }
  let i = n, j = m;
  const path = [[i - 1, j - 1]];
  while (i > 1 || j > 1) {
    const steps = [[D[i - 1][j - 1], i - 1, j - 1], [D[i - 1][j], i - 1, j], [D[i][j - 1], i, j - 1]];
    let best = steps[0]; for (const s of steps) if (s[0] < best[0]) best = s;
    i = best[1]; j = best[2];
    path.push([i - 1, j - 1]);
  }
  path.reverse();
  return { distance: D[n][m] / path.length, path };
}
const dtwScore = (d, scale = 2.2, power = 1.5) => 100 * Math.exp(-((d / scale) ** power));
const cosineScore = (z1, z2) => { let c = 0; for (let i = 0; i < z1.length; i++) c += z1[i] * z2[i]; c = Math.min(1, Math.max(-1, c)); return 100 * ((c + 1) / 2) ** 2; };

function conv1d(x, w, b, stride, pad) {      // x: [C][L], w: [O][C][K]
  const C = x.length, L = x[0].length, O = w.length, K = w[0][0].length;
  const Lout = Math.floor((L + 2 * pad - K) / stride) + 1;
  const out = Array.from({ length: O }, () => new Float64Array(Lout));
  for (let o = 0; o < O; o++) {
    const wo = w[o], ob = out[o];
    for (let t = 0; t < Lout; t++) {
      let s = b[o];
      const base = t * stride - pad;
      for (let c = 0; c < C; c++) { const xc = x[c], wc = wo[c]; for (let k = 0; k < K; k++) { const p = base + k; if (p >= 0 && p < L) s += wc[k] * xc[p]; } }
      ob[t] = s;
    }
  }
  return out;
}
function batchnorm(x, st, prefix, eps) {
  const g = st[prefix + '.weight'], bb = st[prefix + '.bias'], rm = st[prefix + '.running_mean'], rv = st[prefix + '.running_var'];
  return x.map((row, c) => { const sc = g[c] / Math.sqrt(rv[c] + eps); return row.map(v => (v - rm[c]) * sc + bb[c]); });
}
function linear(x, w, b) { return w.map((row, o) => { let s = b[o]; for (let i = 0; i < row.length; i++) s += row[i] * x[i]; return s; }); }

/** ShotEncoder.forward for one z-scored (60 x F) sequence -> unit Float64Array(embedDim). */
function encode(enc, zrows) {
  const st = enc.state;
  let h = zrows[0].length ? Array.from({ length: zrows[0].length }, (_, c) => Float64Array.from(zrows, r => r[c])) : [];
  for (const L of enc.arch.layers) {
    if (L.op === 'conv1d') h = conv1d(h, st[L.w], st[L.b], L.stride, L.padding);
    else if (L.op === 'batchnorm') h = batchnorm(h, st, L.prefix, L.eps);
    else if (L.op === 'gelu') h = Array.isArray(h) && h[0] && h[0].length !== undefined ? h.map(r => r.map(gelu)) : h.map(gelu);
    else if (L.op === 'pool') {
      const mean = h.map(r => r.reduce((a, b) => a + b, 0) / r.length);
      const mx = h.map(r => r.reduce((a, b) => Math.max(a, b), -Infinity));
      h = Float64Array.from([...mean, ...mx]);
    } else if (L.op === 'linear') h = Float64Array.from(linear(h, st[L.w], st[L.b]));
    else if (L.op === 'l2normalize') { const nrm = Math.max(Math.hypot(...h), L.eps ?? 1e-12); h = h.map(v => v / nrm); }
    else throw new Error('encoder: unknown op ' + L.op);
  }
  return h;
}

// ------------------------------------------------------------------ compare.py / drills.py port

function fmtVal(v, unit) {
  if (unit === 'deg') return `${v.toFixed(0)}°`;
  if (unit === 's') return `${v.toFixed(2)}s`;
  if (unit === 'torso') return `${v.toFixed(2)} torso`;
  return v.toFixed(2);
}
function fmtDelta(v, unit) {
  const a = Math.abs(v);
  if (unit === 'deg') return `${a.toFixed(0)}°`;
  if (unit === 's') return `${a.toFixed(2)}s`;
  if (unit === 'torso') return `${a.toFixed(2)} torso-lengths (~${(a * 50).toFixed(0)} cm for a 50 cm torso)`;
  return a.toFixed(2);
}
function featureFeedback(userSum, refSum, metrics) {
  return metrics.map(m => {
    const u = +userSum[m.key], r = +refSum[m.key], d = u - r;
    let severity, text;
    if (Math.abs(d) <= m.tolOk) {
      severity = 'ok';
      text = `${m.label}: ${fmtVal(u, m.unit)} vs ${fmtVal(r, m.unit)} — within ${fmtDelta(m.tolOk, m.unit).split(' (')[0]} of reference.`;
    } else {
      severity = Math.abs(d) >= m.tolMajor ? 'major' : 'minor';
      text = `${m.label}: ${fmtVal(u, m.unit)} vs ${fmtVal(r, m.unit)} — ${fmtDelta(d, m.unit)} ${d > 0 ? m.bigger : m.smaller} than reference.`;
    }
    return { key: m.key, label: m.label, short: m.short || m.label, user: u, ref: r, delta: d, unit: m.unit, severity, text,
      tolOk: m.tolOk, tolMajor: m.tolMajor, gap: Math.abs(d) / m.tolOk, kind: 'style' };
  });
}
function coachingPlan(feedback, drills, maxItems = 6) {
  const order = { major: 0, minor: 1, ok: 2 };
  const mech = feedback.filter(f => f.severity !== 'ok' && f.kind === 'mechanics').sort((a, b) => (b.pointsLost || 0) - (a.pointsLost || 0));
  const style = feedback.filter(f => f.severity !== 'ok' && f.kind !== 'mechanics').sort((a, b) => (order[a.severity] - order[b.severity]) || (Math.abs(b.delta) - Math.abs(a.delta)));
  const plan = [], seen = new Set();
  for (const fb of [...mech, ...style]) {
    if (plan.length >= maxItems) break;
    const d = drills && drills.drills && drills.drills[fb.key];
    if (!d) continue;
    const dr = fb.delta > 0 ? d.high : d.low;
    if (seen.has(dr.drill)) continue;
    seen.add(dr.drill);
    plan.push({ key: fb.key, kind: fb.kind || 'style', finding: fb.label, severity: fb.severity, delta: fb.delta, unit: fb.unit, drill: dr.drill, prescription: dr.prescription, cue: dr.cue });
  }
  if (!plan.length) {
    const g = (drills && drills.generic) || { drill: 'Form shooting warm-up', prescription: '50 makes from 1 m, one hand', cue: 'Nothing major flagged - keep the daily close-range form routine to lock the pattern in.' };
    plan.push({ key: 'maintain', finding: 'Form within tolerance', severity: 'ok', delta: 0, unit: '', drill: g.drill, prescription: g.prescription, cue: g.cue });
  }
  return plan;
}
function radarProfile(userSum, refs, best, axes) {
  const lo = [], hi = [], u = [], b = [], m = [];
  for (const ax of axes) {
    const vals = refs.map(r => +r.summary[ax.key]);
    const uv = +userSum[ax.key], bv = +best.summary[ax.key], mv = vals.reduce((p, q) => p + q, 0) / vals.length;
    const all = [...vals, uv], a = Math.min(...all), c = Math.max(...all), pad = Math.max((c - a) * 0.1, 1e-6);
    u.push(uv); b.push(bv); m.push(mv); lo.push(a - pad); hi.push(c + pad);
  }
  const norm = vs => vs.map((v, i) => Math.round((v - lo[i]) / (hi[i] - lo[i]) * 1e4) / 1e4);
  return { axes: axes.map(a => a.label), axisInfo: axes.map(a => ({ ...a })), user: norm(u), ref: norm(b), mean: norm(m),
    raw: { user: u, ref: b, mean: m }, lo, hi };
}
const round1 = x => Number(x.toFixed(1));

// ------------------------------------------------------------------ mechanics.py port (headline score)

const MECHANICS_SPEC = [   // fallback copy of mechanics.MECHANICS (library.json `mechanics` wins)
  ['set_elbow', 'Elbow angle at set point', 'deg', [70, 100], 85, 20, 20, ['crosbi2019'], ['more open (under-loaded)', 'more closed (over-folded)']],
  ['dip_knee', 'Deepest knee bend in the dip', 'deg', [100, 130], 115, 20, 15, ['crosbi2019', 'frontiers2026'], ['straighter (shallow dip)', 'deeper']],
  ['release_elbow', 'Elbow extension at release', 'deg', [125, 180], 160, 20, 20, ['taborda', 'crosbi2019'], ['more extended', 'less extended']],
  ['release_shoulder', 'Shoulder flexion at release', 'deg', [120, 150], 130, 20, 15, ['taborda'], ['higher (arm behind the head)', 'lower (pushed out in front)']],
  ['release_timing', 'Release vs jump peak', 's', [-0.12, 0.03], -0.03, 0.12, 15, ['crosbi2019', 'franca2021'], ['later (after the peak, on the way down)', 'earlier (well before the peak)']],
  ['follow_through', 'Follow-through hold', 's', [0.3, 3.0], 0.5, 0.2, 10, ['coaching'], ['longer', 'shorter']],
  ['balance_lean', 'Trunk lean change, set to release', 'deg', [0, 10], 0, 10, 2, ['coaching', 'franca2021'], ['more', 'less']],
  ['balance_drift', 'Horizontal hip drift during the jump', 'torso', [0, 0.4], 0, 0.4, 3, ['coaching', 'franca2021'], ['more', 'less']],
].map(([key, label, unit, band, ideal, tolerance, points, source, words]) => ({ key, label, unit, band, ideal, tolerance, points, source, above: words[0], below: words[1] }));
const MECH = { ANGLE_KEYS: new Set(['set_elbow', 'release_elbow', 'release_shoulder', 'balance_lean']), VIEW_FLAG: 0.45, VIEW_FRONTAL: 0.75,
  MIN_JUMP_TORSO: 0.05, DIP_WINDOW_S: 0.8, JUMP_BEFORE_S: 0.3, JUMP_AFTER_S: 0.4, MIN_MEASURED_POINTS: 40 };

function mFmt(v, unit) {
  if (unit === 'deg') return `${v.toFixed(0)}°`;
  if (unit === 's') return `${v.toFixed(2)} s`;
  if (unit === 'torso') return `${v.toFixed(2)} torso`;
  return v.toFixed(2);
}
const mRange = (lo, hi, unit) => `${String(lo)}–${String(hi)}${{ deg: '°', s: ' s', torso: ' torso' }[unit] || ''}`;

/** mechanics.score_item */
function scoreItem(sp, value, reason = null, frame = null) {
  const item = { key: sp.key, label: sp.label, unit: sp.unit, band: [sp.band[0], sp.band[1]], ideal: sp.ideal, tolerance: sp.tolerance, maxPoints: sp.points, source: sp.source, frame };
  if (value == null || !Number.isFinite(value)) {
    const why = reason || 'not measurable on this clip';
    return { ...item, value: null, status: 'unmeasured', points: 0, reason: why, text: `${sp.label}: not measured — ${why}.` };
  }
  const [lo, hi] = sp.band;
  const dist = value < lo ? lo - value : (value > hi ? value - hi : 0);
  const p = sp.points * Math.max(0, 1 - dist / sp.tolerance);
  const status = dist === 0 ? 'good' : (dist <= 0.5 * sp.tolerance ? 'close' : 'off');
  const rng = mRange(lo, hi, sp.unit);
  const text = dist === 0 ? `${sp.label}: ${mFmt(value, sp.unit)} — inside the ${rng} range.`
    : `${sp.label}: ${mFmt(value, sp.unit)} — ${fmtDistM(dist, sp.unit)} ${value > hi ? sp.above : sp.below} than the ${rng} range.`;
  return { ...item, value, status, points: Math.round(p * 100) / 100, edge: value > hi ? hi : lo, delta: value > hi ? value - hi : (value < lo ? value - lo : 0), text };
}
function fmtDistM(d, unit) { return unit === 'deg' ? `${d.toFixed(0)}°` : unit === 's' ? `${d.toFixed(2)} s` : unit === 'torso' ? `${d.toFixed(2)} torso` : d.toFixed(2); }

function observedNear(Xm, joints, i) {
  const T = Xm[0].length, a = Math.max(0, i - 1), b = Math.min(T - 1, i + 1), n = b - a + 1;
  for (const j of joints) { let c = 0; for (let k = a; k <= b; k++) if (!Number.isNaN(Xm[j][k])) c++; if (2 * c < n) return false; }
  return true;
}

/** mechanics.mechanics_profile. Appends the off-side-view flag to quality (in place). */
function mechanicsProfile(F, ex, ev, Xm, side, quality, torso, fps, specList) {
  const T = F[0].length, s = SIDES[side];
  const spec = Object.fromEntries(specList.map(m => [m.key, m]));
  const rel = ev.release, sp = ev.set;
  const codes = new Set(quality.flags.map(f => f.code));
  const swv = []; for (let i = ev.onset; i <= ev.ftEnd; i++) if (Number.isFinite(ex.shoulder_width[i])) swv.push(ex.shoulder_width[i]);
  const view = swv.length ? nanMedian(Float64Array.from(swv)) : 0;
  if (view > MECH.VIEW_FLAG) {
    quality.flags.push({ code: 'off_side_view', severity: 'minor', text: `Camera is off the side (shoulders ${(view * 100).toFixed(0)} % of torso length apart) - joint angles are 2-D projections; film side-on for the most accurate numbers.` });
    if (quality.level === 'ok') quality.level = 'minor';
  }
  const frontal = view > MECH.VIEW_FRONTAL;
  const arm = [s.shoulder, s.elbow, s.wrist];
  const items = [];
  const angleItem = (key, value, joints, frame, why) => {
    if (frontal && MECH.ANGLE_KEYS.has(key)) return scoreItem(spec[key], null, 'camera too frontal to measure this angle', frame);
    if (!observedNear(Xm, joints, frame)) return scoreItem(spec[key], null, why, frame);
    return scoreItem(spec[key], value, null, frame);
  };
  items.push(angleItem('set_elbow', F[0][sp], arm, sp, 'shooting arm hidden at the set point'));
  const lo = Math.max(0, rel - pyInt(pyRound(fps * MECH.DIP_WINDOW_S)));
  const kmin = argmin(F[1], lo, rel + 1);
  items.push(angleItem('dip_knee', F[1][kmin], [s.hip, s.knee, s.ankle], kmin, 'leg hidden at the bottom of the dip'));
  items.push(angleItem('release_elbow', F[0][rel], arm, rel, 'shooting arm hidden at release'));
  items.push(angleItem('release_shoulder', ex.shoulder_flex[rel], [s.shoulder, s.elbow, s.hip], rel, 'shoulder or hip hidden at release'));
  const hr = F[8];
  const a = Math.max(0, rel - pyInt(pyRound(fps * MECH.JUMP_BEFORE_S))), b = Math.min(T - 1, rel + pyInt(pyRound(fps * MECH.JUMP_AFTER_S)));
  const pk = argmax(hr, a, b + 1);
  let mn = Infinity; for (let i = Math.min(sp, pk); i <= pk; i++) mn = Math.min(mn, hr[i]);
  const rise = hr[pk] - mn;
  if (rise < MECH.MIN_JUMP_TORSO) items.push(scoreItem(spec.release_timing, null, 'no clear jump (set shot?)', rel));
  else if (pk === T - 1) items.push(scoreItem(spec.release_timing, null, 'clip ends before the peak of the jump', rel));
  else if (!observedNear(Xm, [s.hip], pk)) items.push(scoreItem(spec.release_timing, null, 'hips hidden at the peak of the jump', rel));
  else {
    let pkSub = pk;
    if (pk > 0 && pk < T - 1) { const y0 = hr[pk - 1], y1 = hr[pk], y2 = hr[pk + 1], den = y0 - 2 * y1 + y2; if (den < 0) pkSub = pk + 0.5 * (y0 - y2) / den; }
    const relSub = ev.releaseSub >= 0 ? ev.releaseSub : rel;
    items.push(scoreItem(spec.release_timing, (relSub - pkSub) / fps, null, rel));
  }
  if (codes.has('clip_cut') && ev.holdS < spec.follow_through.band[0]) items.push(scoreItem(spec.follow_through, null, 'clip ends during the follow-through', rel));
  else items.push(scoreItem(spec.follow_through, ev.holdS, null, rel));
  const trunk = [side === 'left' ? LM.LEFT_SHOULDER : LM.RIGHT_SHOULDER, s.hip];
  if (!observedNear(Xm, trunk, sp)) items.push(scoreItem(spec.balance_lean, null, 'trunk hidden at the set point', rel));
  else items.push(angleItem('balance_lean', Math.abs(F[3][rel] - F[3][sp]), trunk, rel, 'trunk hidden at release'));
  const e = Math.min(T - 1, rel + pyInt(pyRound(fps * MECH.JUMP_AFTER_S)));
  const drift = Math.abs(ex.hip_x[e] - ex.hip_x[sp]) / Math.max(torso, 1e-6);
  if (!(observedNear(Xm, [s.hip], sp) && observedNear(Xm, [s.hip], e))) items.push(scoreItem(spec.balance_drift, null, 'hips hidden during the jump', e));
  else items.push(scoreItem(spec.balance_drift, drift, null, e));
  const measured = items.filter(it => it.status !== 'unmeasured');
  const tot = measured.reduce((q, it) => q + it.maxPoints, 0);
  const score = tot >= MECH.MIN_MEASURED_POINTS ? round1(100 * measured.reduce((q, it) => q + it.points, 0) / tot) : null;
  return { items, score, view };
}

/** mechanics.mechanics_feedback */
function mechanicsFeedback(items) {
  const out = [];
  for (const it of items) {
    if (it.status === 'unmeasured') continue;
    const severity = { good: 'ok', close: 'minor', off: 'major' }[it.status];
    out.push({ key: it.key, label: it.label, short: it.label, user: it.value, ref: it.edge, delta: it.delta, unit: it.unit, severity, text: it.text,
      kind: 'mechanics', pointsLost: Math.round((it.maxPoints - it.points) * 100) / 100, band: it.band });
  }
  return out.sort((p, q) => q.pointsLost - p.pointsLost);
}


// ------------------------------------------------------------------ assets

/** Prepare already-parsed JSON assets (used by loadAssets; handy in tests / workers). */
export function prepareAssets({ library, drills = null, encoder = null }) {
  if (!library || !Array.isArray(library.references) || !library.references.length) throw new Error('library.json has no references');
  const stats = { mean: Float64Array.from(library.stats.mean), std: Float64Array.from(library.stats.std) };
  const metrics = library.metrics && library.metrics.length ? library.metrics : FEEDBACK_SPEC;
  const axes = library.radarAxes && library.radarAxes.length ? library.radarAxes : RADAR_AXES;
  const weights = library.weights || { dtw: W_DTW, learned: W_LEARNED };
  const toRows = a => a.map(r => Float32Array.from(r));
  const refs = library.references.map(r => ({
    name: r.name, side: r.side, facing: r.facing, fps: r.fps, events: r.events, summary: r.summary,
    seq60: toRows(r.seq60), ghost: r.ghost ? { frames: r.ghost.frames.map(f => Float64Array.from(f, v => v == null ? NaN : v)), events: r.ghost.events } : null,
  }));
  for (const r of refs) r.z = zscore(r.seq60, stats);
  let enc = null;
  if (encoder && encoder.state && encoder.arch) {
    try {
      enc = { arch: encoder.arch, state: encoder.state };
      const es = encoder.stats || library.stats;
      enc.stats = { mean: Float64Array.from(es.mean), std: Float64Array.from(es.std) };
      for (const r of refs) r.emb = encode(enc, zscore(r.seq60, enc.stats));
      if (!refs.every(r => r.emb.every(Number.isFinite))) throw new Error('non-finite embedding');
    } catch (e) { enc = null; for (const r of refs) delete r.emb; }
  }
  const mechSpec = library.mechanics && library.mechanics.checks && library.mechanics.checks.length ? library.mechanics.checks : MECHANICS_SPEC;
  return { library, drills, encoder: enc, refs, stats, metrics, axes, weights, learned: !!enc, mechSpec, sources: (library.mechanics && library.mechanics.sources) || [] };
}

async function readJson(base, name, optional) {
  const url = base ? new URL(name, base.endsWith('/') ? base : base + '/') : null;
  try {
    if (url && url.protocol === 'file:') {
      const fs = await import('node:fs/promises');
      return JSON.parse(await fs.readFile(url, 'utf8'));
    }
    const res = await fetch(url ? url.href : name);
    if (!res.ok) throw new Error(`${name}: HTTP ${res.status}`);
    return await res.json();
  } catch (e) {
    if (optional) return null;
    throw e;
  }
}

/** Fetch library.json, drills.json and (optional) encoder.json from baseUrl (http(s) or file: URL). */
export async function loadAssets(baseUrl) {
  let base = baseUrl;
  if (base == null) base = typeof location !== 'undefined' ? new URL('./', import.meta.url).href : new URL('./', import.meta.url).href;
  if (typeof location !== 'undefined' && !/^[a-z]+:/i.test(base)) base = new URL(base, location.href).href;
  const [library, drills, encoder] = await Promise.all([readJson(base, 'library.json', false), readJson(base, 'drills.json', true), readJson(base, 'encoder.json', true)]);
  return prepareAssets({ library, drills, encoder });
}

// ------------------------------------------------------------------ core pipeline on one window

function fail(reason, code, quality, extra = {}) {
  return { ok: false, reason, code, quality: quality || { level: 'major', flags: [{ code, severity: 'major', text: reason }], visibility: 0, frames: 0 }, ...extra };
}

/** Run the Python-equivalent pipeline on frames[lo..hi]. Returns internal result in grid space. */
function analyzeWindow(frames, meta, lo, hi, sideOpt) {
  const W = meta.width, H = meta.height;
  const grid = buildGrid(frames, meta, lo, hi);
  const { fps, n } = grid;
  const { X: Xm, Y: Ym, V } = gridLandmarks(frames, lo, grid, W, H);
  const glitch = glitchFrames(Xm, Ym);
  for (let i = 0; i < n; i++) if (glitch[i]) for (let l = 0; l < NUM_LM; l++) { Xm[l][i] = NaN; Ym[l][i] = NaN; }
  const missingMask = Array.from({ length: n }, (_, i) => { for (let l = 0; l < NUM_LM; l++) if (!Number.isNaN(Xm[l][i])) return false; return true; });
  const nPose = missingMask.filter(m => !m).length;
  const baseQ = () => assessQuality(V, missingMask, n, fps, null, sideOpt === 'left' ? 'left' : 'right');
  if (nPose < MIN_POSE_FRAMES) return { fail: fail(nPose === 0 ? "Couldn't find a person in this clip — make sure your whole body is visible and well lit." : `Only found you in ${nPose} frame${nPose === 1 ? '' : 's'} — keep your whole body in frame for the whole shot (about 2 seconds).`, 'no_pose', baseQ()) };
  // the grid is uniform in time (index = t * fps), so index-based filling == time-based filling
  const X = Xm.map(c => fillCol(c, null)), Y = Ym.map(c => fillCol(c, null));
  const side = sideOpt === 'left' || sideOpt === 'right' ? sideOpt : detectShootingSide(Y);
  const miss = missingRequired(X, side);
  if (miss.length) {
    const words = [...new Set(miss.map(k => (k === 'nose' ? 'head' : `${side} ${JOINT_WORDS[k]}`)))].join(', ');
    return { fail: fail(`Couldn't see your ${words} — keep your whole body, from your feet to your shooting hand and head, in frame.`, 'landmarks_missing', assessQuality(V, missingMask, n, fps, null, side)) };
  }
  const { feats, torso, facing, ex } = perFrameFeatures(X, Y, side, fps);
  const ev = detectEvents(feats, fps, ex.wrist_above_shoulder, sceneCuts(X, Y));
  rebaselineHipRise(feats, ex, ev, torso, fps);
  const quality = assessQuality(V, missingMask, n, fps, ev, side);
  return { grid, fps, n, X, Y, Xm, Ym, Vis: V, missingMask, side, feats, torso, facing, ex, ev, quality, W, H };
}

// ------------------------------------------------------------------ public API

/**
 * Analyze one shot. frames: [{t, lm}|null], meta: {fps, width, height, duration}.
 * opts: {side: 'auto'|'left'|'right', window: {start, end}|null (inclusive frame indices),
 *        autoWindow: true (clips > 3.5 s without a window are cut to the best detectShots window)}.
 * Never throws: on bad input returns {ok:false, reason, code, quality}.
 */
export function analyze(frames, meta, assets, opts = {}) {
  try {
    return analyzeImpl(frames, meta || {}, assets, opts || {});
  } catch (e) {
    return fail(`Analysis failed: ${e && e.message ? e.message : e}`, 'internal_error', null);
  }
}

function analyzeImpl(frames, meta, assets, opts) {
  if (!Array.isArray(frames) || frames.length === 0) return fail('No frames to analyze — the video could not be read.', 'no_frames', null);
  if (!assets || !assets.refs) return fail('Reference library not loaded.', 'no_assets', null);
  if (!(isNum(meta.width) && meta.width > 0 && isNum(meta.height) && meta.height > 0)) return fail('Missing video dimensions (meta.width / meta.height) — needed to measure angles correctly.', 'no_dimensions', null);
  const N = frames.length;
  let lo = 0, hi = N - 1, windowSrc = 'full';
  if (opts.window && isNum(opts.window.start) && isNum(opts.window.end)) {
    lo = Math.max(0, Math.min(N - 1, Math.floor(opts.window.start)));
    hi = Math.max(lo, Math.min(N - 1, Math.floor(opts.window.end)));
    windowSrc = 'given';
  } else if (opts.window == null && opts.autoWindow !== false) {
    const tt = frameTimes(frames, 0, N - 1, meta.fps);
    if (tt[N - 1] - tt[0] > AUTO_WINDOW_MIN_S) {
      const shots = detectShots(frames, meta);
      if (shots.length) {
        const best = shots.reduce((a, b) => (b.confidence > a.confidence ? b : a));
        lo = best.start; hi = best.end; windowSrc = 'auto';
      }
    }
  }
  const sideOpt = opts.side === 'left' || opts.side === 'right' ? opts.side : 'auto';
  const w = analyzeWindow(frames, meta, lo, hi, sideOpt);
  if (w.fail) return { ...w.fail, window: { start: lo, end: hi, source: windowSrc } };
  const { grid, fps, n, feats, ev, quality, side, facing, torso } = w;

  // partial shots -> not analyzable
  const codes = new Set(quality.flags.map(f => f.code));
  if (opts.debug) opts.debug.window = { lo, hi, ev, feats, fps, mode: grid.mode };
  if (codes.has('release_not_found')) return fail("Couldn't find a release — make sure the whole shot, from dip to follow-through, is in the clip.", 'release_not_found', quality, { window: { start: lo, end: hi, source: windowSrc } });
  if (codes.has('events_degenerate')) {
    const reason = (ev.set <= ev.onset || ev.set <= ev.warm) ? "Couldn't find the dip before the release — start recording about a second before you start your shot." : "Couldn't separate the set point from the release — make sure the whole shot, from dip to follow-through, is in the clip.";
    return fail(reason, 'events_degenerate', quality, { window: { start: lo, end: hi, source: windowSrc } });
  }

  const seq60 = resampleRows(feats, ev.onset, ev.ftEnd);
  const summary = summarize(feats, ev, fps);
  for (const k of Object.keys(summary)) if (!Number.isFinite(summary[k])) return fail('Pose data too noisy to measure this shot — try better light and a steadier camera.', 'nan_features', quality);

  // ---- scoring (compare.compare_shot)
  const wts = assets.weights || { dtw: W_DTW, learned: W_LEARNED };
  const zu = zscore(seq60, assets.stats);
  const embU = assets.encoder ? encode(assets.encoder, zscore(seq60, assets.encoder.stats)) : null;
  const comps = assets.refs.map(ref => {
    const { distance, path } = dtwSimilarity(zu, ref.z);
    const sDtw = dtwScore(distance);
    let sL = null, fin = sDtw;
    if (embU && ref.emb) { sL = cosineScore(embU, ref.emb); fin = wts.dtw * sDtw + wts.learned * sL; }
    const contrib = new Float64Array(FEATURE_NAMES.length);
    for (const [i, j] of path) for (let k = 0; k < contrib.length; k++) contrib[k] += Math.abs(zu[i][k] - ref.z[j][k]);
    const featureDistance = Object.fromEntries(FEATURE_NAMES.map((nm, k) => [nm, Math.round(contrib[k] / path.length * 1000) / 1000]));
    return { name: ref.name, score: round1(fin), dtw: round1(sDtw), learned: sL == null ? null : round1(sL), dtwDistance: Math.round(distance * 1000) / 1000, featureDistance, _ref: ref, _path: path };
  });
  comps.sort((a, b) => b.score - a.score);
  const best = comps[0], bestRef = best._ref;
  const mech = mechanicsProfile(feats, w.ex, ev, w.Xm, side, quality, torso, fps, assets.mechSpec || MECHANICS_SPEC);
  const styleFeedback = featureFeedback(summary, bestRef.summary, assets.metrics);
  const feedback = [...mechanicsFeedback(mech.items), ...styleFeedback];
  const plan = coachingPlan(feedback, assets.drills);
  const radar = radarProfile(summary, assets.refs, bestRef, assets.axes);

  // ---- map grid -> input frames
  const tIn = grid.tIn, nIn = N;
  const toInput = g => lo + grid.nearest[Math.max(0, Math.min(grid.n - 1, g))];   // grid index -> input frame index
  const gridPos = iAbs => grid.pos[iAbs - lo];                                     // input frame -> (fractional) grid index
  const events = { onset: toInput(ev.onset), set: toInput(ev.set), release: toInput(ev.release), ftEnd: toInput(ev.ftEnd),
    peak: toInput(ev.peak), followThrough: toInput(ev.followThrough), holdS: ev.holdS, releaseFound: ev.releaseFound,
    peakExtension: toInput(ev.peakExtension), setMethod: ev.setMethod,
    t: { onset: grid.tGrid[ev.onset], set: grid.tGrid[ev.set], release: grid.tGrid[ev.release], ftEnd: grid.tGrid[ev.ftEnd] } };
  const perFrame = {};
  FEATURE_NAMES.forEach((nm, j) => {
    const arr = new Float32Array(nIn).fill(NaN);
    for (let i = lo; i <= hi; i++) {
      if (grid.mode !== 'resampled') arr[i] = feats[j][grid.pos[i - lo]];
      else { const g = gridPos(i), a = Math.max(0, Math.min(n - 1, Math.floor(g))), b = Math.min(n - 1, a + 1), u = Math.max(0, Math.min(1, g - a)); arr[i] = feats[j][a] + (feats[j][b] - feats[j][a]) * u; }
    }
    perFrame[nm] = arr;
  });

  // ---- landmarks for drawing (input frames, gap-filled over t, normalised image coords, 132 layout)
  const { landmarks, canonical } = drawLandmarks(frames, lo, hi, tIn, w.W, w.H, side);
  const mirrored = side === 'left';
  const ghost = buildGhost(bestRef, { lo, hi, tIn, events, ev, gridPos, w, side, facing, torso });

  return {
    ok: true, engine: ENGINE_VERSION,
    quality, hand: side, facing: facing > 0 ? 'right' : 'left', mirrored,
    window: { start: lo, end: hi, source: windowSrc }, fps, timing: grid.mode, torsoPx: torso,
    events, perFrame, summary, seq60, canonical, landmarks,
    mechanics: mech.items.map(it => ({ ...it, frame: it.frame == null ? null : toInput(it.frame) })),
    mechanicsScore: mech.score, styleMatch: best.score, scoreKind: mech.score != null ? 'mechanics' : 'style', view: { shoulderWidth: mech.view },
    comparisons: comps.map(({ _ref, _path, ...c }) => c),
    closest: { name: best.name, score: best.score, ghost: ghost.frames, warp: ghost.warp, start: events.onset, end: events.ftEnd },
    feedback, plan, radar, score: mech.score != null ? mech.score : best.score, learned: !!embU, embedding: embU ? Float32Array.from(embU) : null,
  };
}

/** Gap-filled (over t) landmarks of frames[lo..hi] as normalised 132-arrays; canonical = mirrored
 *  into a right-handed shooter when the shooting hand is left (x -> 1 - x, left/right swapped). */
function drawLandmarks(frames, lo, hi, tIn, W, H, side) {
  const n = hi - lo + 1, N = frames.length;
  const cols = [];
  for (let l = 0; l < NUM_LM; l++) {
    const cx = new Float64Array(n).fill(NaN), cy = new Float64Array(n).fill(NaN);
    for (let i = 0; i < n; i++) {
      const lm = frameLm(frames[lo + i]); if (!lm) continue;
      const v = lm[4 * l + 3] == null ? 1 : +lm[4 * l + 3];
      if (v >= MIN_VIS && isNum(+lm[4 * l]) && isNum(+lm[4 * l + 1])) { cx[i] = +lm[4 * l]; cy[i] = +lm[4 * l + 1]; }
    }
    cols.push([fillCol(cx, tIn), fillCol(cy, tIn)]);
  }
  const landmarks = new Array(N).fill(null), canonical = new Array(N).fill(null);
  const mir = side === 'left';
  for (let i = 0; i < n; i++) {
    const a = new Float32Array(NUM_LM * 4), c = new Float32Array(NUM_LM * 4);
    for (let l = 0; l < NUM_LM; l++) {
      const x = cols[l][0][i], y = cols[l][1][i], ok = !Number.isNaN(x);
      a[4 * l] = ok ? x : 0; a[4 * l + 1] = ok ? y : 0; a[4 * l + 3] = ok ? 1 : 0;
      const d = mir ? MIRROR_OF[l] : l;
      c[4 * d] = ok ? (mir ? 1 - x : x) : 0; c[4 * d + 1] = ok ? y : 0; c[4 * d + 3] = ok ? 1 : 0;
    }
    landmarks[lo + i] = a; canonical[lo + i] = c;
  }
  return { landmarks, canonical };
}

/** Closest reference's landmarks time-warped onto user frames onset..ftEnd (piecewise-linear through
 *  onset / set / release / follow-through end, like overlay.align_frames), scaled to the user's torso,
 *  anchored at the user's stance (hip-mid x, ankle-mid y) and turned to the user's facing and hand.
 *  Output in the user's normalised image coords (132 layout), so it can be drawn over the video. */
function buildGhost(ref, ctx) {
  const { events, ev, gridPos, w, side, facing, torso } = ctx;
  if (!ref.ghost) return { frames: [], warp: new Int32Array(0) };
  const nb = Math.max(2, pyInt(w.fps * 0.15));
  const hipX = pairMid(w.X[LM.LEFT_HIP].slice(0, nb), w.X[LM.RIGHT_HIP].slice(0, nb));
  const ankY = pairMid(w.Y[LM.LEFT_ANKLE].slice(0, nb), w.Y[LM.RIGHT_ANKLE].slice(0, nb));
  const cx = nanMedian(hipX), floor = nanMedian(ankY);
  const ge = ref.ghost.events, G = ref.ghost.frames;
  const uk = [ev.onset, ev.set, ev.release, ev.ftEnd], rk = [0, ge.set, ge.release, G.length - 1];
  for (let i = 1; i < 4; i++) { uk[i] = Math.max(uk[i], uk[i - 1]); rk[i] = Math.max(rk[i], rk[i - 1]); }
  const frames = [], warp = new Int32Array(events.ftEnd - events.onset + 1);
  for (let iAbs = events.onset, k = 0; iAbs <= events.ftEnd; iAbs++, k++) {
    const g = gridPos(iAbs);
    const r = Math.max(0, Math.min(G.length - 1, Math.round(interp1(g, uk, rk))));
    warp[k] = r;
    const src = G[r], out = new Float32Array(NUM_LM * 4);
    for (let l = 0; l < NUM_LM; l++) {
      const d = side === 'left' ? MIRROR_OF[l] : l;
      const xn = src[2 * l], yn = src[2 * l + 1];
      if (Number.isNaN(xn) || Number.isNaN(yn)) continue;
      out[4 * d] = (cx + facing * xn * torso) / w.W;
      out[4 * d + 1] = (floor + yn * torso) / w.H;
      out[4 * d + 3] = 1;
    }
    frames.push(out);
  }
  return { frames, warp };
}

// ------------------------------------------------------------------ shot detection in long clips

/**
 * Find candidate shots in a longer clip: the wrist rises above the head with the elbow extending,
 * preceded by a lower ball position. Returns frame-index windows padded ~0.8 s before the lift and
 * ~0.6 s after the arm comes down, merged when overlapping, sorted by time:
 *   [{start, end, releaseGuess, confidence (0..1), tStart, tEnd, tRelease, hand}]
 */
export function detectShots(frames, meta = {}) {
  try { return detectShotsImpl(frames, meta || {}); } catch (e) { return []; }
}

function detectShotsImpl(frames, meta) {
  if (!Array.isArray(frames) || frames.length < 5) return [];
  const N = frames.length;
  const W = isNum(meta.width) && meta.width > 0 ? meta.width : 1, H = isNum(meta.height) && meta.height > 0 ? meta.height : 1;
  const t = frameTimes(frames, 0, N - 1, meta.fps);
  const col = (l, k) => {
    const out = new Float64Array(N).fill(NaN);
    for (let i = 0; i < N; i++) { const lm = frameLm(frames[i]); if (!lm) continue; const v = lm[4 * l + 3] == null ? 1 : +lm[4 * l + 3]; if (v >= MIN_VIS) out[i] = +lm[4 * l + k] * (k ? H : W); }
    return shortFill(out, t, 0.4);
  };
  const P = {};
  for (const l of [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]) P[l] = [col(l, 0), col(l, 1)];
  const mid = (a, b, k) => P[a][k].map((v, i) => Number.isNaN(v) ? P[b][k][i] : Number.isNaN(P[b][k][i]) ? v : (v + P[b][k][i]) / 2);
  const shX = mid(11, 12, 0), shY = mid(11, 12, 1), hpX = mid(23, 24, 0), hpY = mid(23, 24, 1);
  const torsoArr = shX.map((v, i) => Math.hypot(v - hpX[i], shY[i] - hpY[i]));
  const torso = nanMedian(torsoArr);
  if (!Number.isFinite(torso) || torso <= 0) return [];
  const smoothT = (a, win) => {      // centred moving average over +-win seconds
    const out = new Float64Array(N); let j0 = 0;
    for (let i = 0; i < N; i++) {
      while (j0 < N && t[j0] < t[i] - win) j0++;
      let s = 0, c = 0;
      for (let j = j0; j < N && t[j] <= t[i] + win; j++) if (!Number.isNaN(a[j])) { s += a[j]; c++; }
      out[i] = c && !Number.isNaN(a[i]) ? s / c : NaN;
    }
    return out;
  };
  const arms = { right: SIDES.right, left: SIDES.left };
  const A = {};
  for (const [hand, s] of Object.entries(arms)) {
    const [wx, wy] = P[s.wrist], [ex, ey] = P[s.elbow], [sx, sy] = P[s.shoulder];
    const headY = P[0][1].map((v, i) => Number.isNaN(v) ? shY[i] - 0.45 * torso : v);
    const wh = smoothT(wy.map((v, i) => (headY[i] - v) / torso), 0.05);
    const el = smoothT(wx.map((_, i) => angleDeg(sx[i], sy[i], ex[i], ey[i], wx[i], wy[i])), 0.05);
    const kn = s.knee ? P[s.knee] : null;
    const knee = kn ? smoothT(kn[0].map((_, i) => angleDeg(P[s.hip][0][i], P[s.hip][1][i], kn[0][i], kn[1][i], P[s.ankle][0][i], P[s.ankle][1][i])), 0.05) : null;
    A[hand] = { wh, el, knee };
  }
  const HIGH = 0.15, LOW = -0.3, PRE_S = 1.5;
  const cands = [];
  for (const [hand, a] of Object.entries(A)) {
    let i = 0;
    while (i < N) {
      if (!(a.wh[i] > HIGH)) { i++; continue; }
      let j = i; while (j + 1 < N && (a.wh[j + 1] > HIGH || (Number.isNaN(a.wh[j + 1]) && t[j + 1] - t[j] < 0.15))) j++;
      const s0 = i, s1 = j; i = j + 1;
      if (s1 - s0 < 1) continue;
      let peakWh = -Infinity, peakEl = -Infinity;
      for (let k = s0; k <= s1; k++) { if (a.wh[k] > peakWh) peakWh = a.wh[k]; if (a.el[k] > peakEl) peakEl = a.el[k]; }
      // lift start: lowest wrist within PRE_S before the segment
      let lift = s0, minWh = Infinity;
      for (let k = s0; k >= 0 && t[s0] - t[k] <= PRE_S; k--) if (a.wh[k] < minWh) { minWh = a.wh[k]; lift = k; }
      let rel = -1; const thr = Math.min(160, peakEl - 4);
      for (let k = Math.max(0, lift); k <= s1; k++) if (a.el[k] >= thr && a.wh[k] > 0) { rel = k; break; }
      if (rel < 0) rel = argmax(a.wh, s0, s1 + 1);
      let kneeDip = 0;
      if (a.knee) { let kmin = Infinity, kmax = -Infinity; for (let k = Math.max(0, lift - 10); k <= rel; k++) { if (a.knee[k] < kmin) kmin = a.knee[k]; if (a.knee[k] > kmax) kmax = a.knee[k]; } if (Number.isFinite(kmin) && Number.isFinite(kmax)) kneeDip = kmax - kmin; }
      const cExt = clamp01((peakEl - 130) / 35), cHigh = clamp01((peakWh - HIGH) / 0.45), cLift = clamp01((HIGH - minWh) / 0.8);
      const cDip = clamp01(kneeDip / 15);
      let vis = 0, cnt = 0; for (let k = lift; k <= s1; k++) { cnt++; if (!Number.isNaN(a.wh[k]) && !Number.isNaN(a.el[k])) vis++; }
      const conf = (0.35 * cExt + 0.2 * cHigh + 0.25 * cLift + 0.2 * cDip) * (cnt ? vis / cnt : 0);
      if (peakEl < 120 || minWh > LOW + 0.25) continue;     // arm never extended or ball never came from below
      cands.push({ hand, lift, s1, rel, conf });
    }
  }
  if (!cands.length) return [];
  const idxAt = (tt, dir) => { if (dir < 0) { let k = 0; while (k + 1 < N && t[k + 1] <= tt) k++; return k; } let k = N - 1; while (k - 1 >= 0 && t[k - 1] >= tt) k--; return k; };
  // scene cuts (edited clips / camera jumps): the hip jumps > CUT_TORSO torso lengths between frames
  const cutsD = []; let prev = -1;
  for (let i = 0; i < N; i++) {
    if (Number.isNaN(hpX[i]) || Number.isNaN(hpY[i])) continue;
    if (prev >= 0 && i - prev <= 3 && Math.hypot(hpX[i] - hpX[prev], hpY[i] - hpY[prev]) / torso > EV.CUT_TORSO * (i - prev)) cutsD.push(i);
    prev = i;
  }
  const clampToScene = w => {
    for (const c of cutsD) { if (c <= w.releaseGuess && c > w.start) w.start = c; if (c > w.releaseGuess && c - 1 < w.end) w.end = c - 1; }
    return w;
  };
  let wins = cands.map(c => clampToScene({ start: idxAt(t[c.lift] - 0.8, -1), end: idxAt(t[c.s1] + 0.6, 1), releaseGuess: c.rel, confidence: c.conf, hand: c.hand, lift: c.lift, s1: c.s1 }));
  wins.sort((p, q) => p.releaseGuess - q.releaseGuess);
  const merged = [];
  for (const wv of wins) {
    const last = merged[merged.length - 1];
    if (last && wv.start <= last.end) {
      if (t[wv.releaseGuess] - t[last.releaseGuess] < 1.0) {            // same shot seen twice (both arms / split segment)
        last.end = Math.max(last.end, wv.end); last.start = Math.min(last.start, wv.start); last.s1 = Math.max(last.s1, wv.s1);
        if (wv.confidence > last.confidence) { last.releaseGuess = wv.releaseGuess; last.hand = wv.hand; last.confidence = wv.confidence; }
        continue;
      }
      // two shots close together: split the padding between them
      const cut = Math.max(last.releaseGuess + 1, Math.min(wv.releaseGuess - 1, Math.round((last.s1 + wv.lift) / 2)));
      last.end = Math.min(last.end, cut); wv.start = Math.max(wv.start, cut + 1);
    }
    merged.push({ ...wv });
  }
  return merged.map(m => ({ start: m.start, end: m.end, releaseGuess: m.releaseGuess, confidence: Math.round(m.confidence * 100) / 100,
    tStart: t[m.start], tEnd: t[m.end], tRelease: t[m.releaseGuess], hand: m.hand }));
}
const clamp01 = x => Number.isFinite(x) ? Math.max(0, Math.min(1, x)) : 0;
/** Fill NaN gaps no longer than maxGap seconds (linear over t); longer gaps stay NaN. */
function shortFill(a, t, maxGap) {
  const out = Float64Array.from(a), n = a.length;
  let i = 0;
  while (i < n) {
    if (!Number.isNaN(out[i])) { i++; continue; }
    let j = i; while (j < n && Number.isNaN(out[j])) j++;
    const L = i - 1, R = j;
    if (L >= 0 && R < n && t[R] - t[L] <= maxGap) for (let k = i; k < j; k++) out[k] = out[L] + (out[R] - out[L]) * (t[k] - t[L]) / (t[R] - t[L]);
    i = j;
  }
  return out;
}

// ------------------------------------------------------------------ history

/**
 * results: array (oldest first) of analyze() results or saved entries {score, deltas:{key:delta}, timestamp?}.
 * -> {scores, trend:{change, slope, direction}, mostImproved:[{key,label,value}], trendingWorse:[...],
 *     best:[...], worst:[...]} — feature changes are |delta| / ok-tolerance (report._trends).
 */
export function summarizeHistory(results, metrics = FEEDBACK_SPEC) {
  const entries = (results || []).filter(r => r && (r.ok !== false) && isNum(r.score)).map(r => ({
    score: r.score, timestamp: r.timestamp ?? null,
    deltas: r.deltas || Object.fromEntries((r.feedback || []).map(f => [f.key, f.delta])),
  }));
  const scores = entries.map(e => e.score);
  const n = scores.length;
  let slope = 0;
  if (n >= 2) {
    const mx = (n - 1) / 2, my = scores.reduce((a, b) => a + b, 0) / n;
    let sxy = 0, sxx = 0; scores.forEach((s, i) => { sxy += (i - mx) * (s - my); sxx += (i - mx) ** 2; });
    slope = sxx ? sxy / sxx : 0;
  }
  const change = n >= 2 ? scores[n - 1] - scores[0] : 0;
  const direction = n < 2 ? 'flat' : change > 1 ? 'up' : change < -1 ? 'down' : 'flat';
  const out = { scores, trend: { change: Math.round(change * 10) / 10, slope: Math.round(slope * 100) / 100, direction }, mostImproved: [], trendingWorse: [], best: [], worst: [] };
  if (!n) return out;
  const tol = Object.fromEntries(metrics.map(m => [m.key, m]));
  const first = entries[0].deltas, last = entries[n - 1].deltas;
  const keys = Object.keys(last).filter(k => tol[k]);
  const lab = k => tol[k].short || tol[k].label;
  const change_ = Object.fromEntries(keys.map(k => [k, Math.abs(first[k] ?? last[k]) / tol[k].tolOk - Math.abs(last[k]) / tol[k].tolOk]));
  const now = Object.fromEntries(keys.map(k => [k, Math.abs(last[k]) / tol[k].tolOk]));
  const item = (k, v) => ({ key: k, label: lab(k), value: Math.round(v * 100) / 100 });
  out.mostImproved = keys.filter(k => change_[k] > 0.05).sort((a, b) => change_[b] - change_[a]).slice(0, 3).map(k => item(k, change_[k]));
  out.trendingWorse = keys.filter(k => change_[k] < -0.05).sort((a, b) => change_[a] - change_[b]).slice(0, 3).map(k => item(k, change_[k]));
  out.best = [...keys].sort((a, b) => now[a] - now[b]).slice(0, 3).map(k => item(k, now[k]));
  out.worst = [...keys].sort((a, b) => now[b] - now[a]).slice(0, 3).map(k => item(k, now[k]));
  return out;
}

// test hooks (internal; not part of the UI contract)
export const _internals = { savgolWeights, smoothCol, dtwSimilarity, encode, zscore, pyRound, buildGrid, detectEvents };
