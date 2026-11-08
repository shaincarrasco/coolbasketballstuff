// pose.mjs — on-device MediaPipe PoseLandmarker wrapper: video -> frames (see CONTRACT.md).
// Nothing leaves the device: the model + wasm are same-origin files under ../vendor/.
//
//   ensureLandmarker({ model:'lite'|'full', onProgress, signal }) -> { landmarker, delegate, fellBack }
//   extractPose(video, { start, end, landmarker, nativeFps, onFrame, signal }) -> { frames, meta, stats }
//   grabFrame(video, t, maxSide) -> HTMLCanvasElement (seek + draw; video must not be playing)
//   PoseError(code, message)  codes: model | simd | codec | aborted | gpu | extract

const VENDOR = new URL('../vendor/', import.meta.url);
const FILES = {
  bundle: new URL('vision_bundle.mjs', VENDOR).href,
  loader: new URL('wasm/vision_wasm_internal.js', VENDOR).href,
  wasm: new URL('wasm/vision_wasm_internal.wasm', VENDOR).href,
  lite: new URL('models/pose_landmarker_lite.task', VENDOR).href,
  full: new URL('models/pose_landmarker_full.task', VENDOR).href,
};
// Fallback sizes when the server omits Content-Length (bytes, tasks-vision 1.2.2).
const KNOWN = { wasm: 14603354, lite: 5777746, full: 9398198 };

// Privacy guard: tasks-vision 1.2.2 ships a usage logger that POSTs to odml.pa.googleapis.com every 60 s.
// This app promises that nothing leaves the device, so cross-origin fetches from this page are answered
// locally with a 403 (the logger then disables itself) and never reach the network.
(function guardFetch() {
  if (typeof window === 'undefined' || !window.fetch || window.fetch.__jsGuard) return;
  const orig = window.fetch.bind(window);
  const guarded = function (input, init) {
    let url;
    try { url = new URL(typeof input === 'string' ? input : input && input.url, location.href); } catch (e) { return orig(input, init); }
    if (/^https?:$/.test(url.protocol) && url.origin !== location.origin) {
      if (!guarded.warned) { guarded.warned = true; console.info('[jumpshot] blocked off-device request to', url.host); }
      return Promise.resolve(new Response(null, { status: 403, statusText: 'Blocked: on-device only' }));
    }
    return orig(input, init);
  };
  guarded.__jsGuard = true;
  window.fetch = guarded;
})();

// MediaPipe's wasm routes glog INFO/WARNING lines (e.g. "INFO: Created TensorFlow Lite XNNPACK delegate")
// through console.error/warn. They are not errors; demote them to console.debug.
(function quietGlog() {
  if (typeof console === 'undefined' || console.__jsQuiet) return;
  const re = /^(INFO|WARNING|VERBOSE)[: ]|^[IW]\d{4} \d\d:\d\d|Graph successfully started|OpenGL error checking is disabled|GL version:/;
  for (const k of ['error', 'warn', 'log', 'info']) {
    const orig = console[k].bind(console);
    console[k] = (...a) => (typeof a[0] === 'string' && re.test(a[0]) ? console.debug(...a) : orig(...a));
  }
  console.__jsQuiet = true;
})();

export class PoseError extends Error {
  constructor(code, message, cause) { super(message); this.code = code; this.cause = cause; }
}

let visionMod = null;            // the imported bundle
let wasmUrl = null;              // blob: URL of the wasm bytes (downloaded once, with progress)
const modelBytes = {};           // model -> Uint8Array
const cache = {};                // `${model}` -> { landmarker, delegate }
let tsCursor = 0;                // detectForVideo timestamps must increase across the landmarker's life

export function simdSupported() {
  try {
    // Minimal module using a v128 instruction (from wasm-feature-detect).
    return WebAssembly.validate(new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11]));
  } catch (e) { return false; }
}

async function fetchBytes(url, known, onBytes, signal) {
  let res;
  try { res = await fetch(url, { signal }); }
  catch (e) { if (signal && signal.aborted) throw new PoseError('aborted', 'Cancelled'); throw new PoseError('model', 'Network error while downloading the AI model.', e); }
  if (!res.ok) throw new PoseError('model', 'The AI model could not be downloaded (HTTP ' + res.status + ').');
  const total = +res.headers.get('content-length') || known || 0;
  if (!res.body || !res.body.getReader) {
    const buf = new Uint8Array(await res.arrayBuffer()); onBytes(buf.length, total || buf.length); return buf;
  }
  const reader = res.body.getReader();
  const chunks = []; let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value); got += value.length; onBytes(got, Math.max(total, got));
  }
  const out = new Uint8Array(got); let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.length; }
  return out;
}

/** Download (with byte progress) and create the PoseLandmarker. GPU first, CPU fallback. */
export async function ensureLandmarker({ model = 'lite', onProgress = () => {}, signal } = {}) {
  if (cache[model]) return { ...cache[model], fellBack: false, cached: true };
  if (!simdSupported()) throw new PoseError('simd', 'This browser does not support WebAssembly SIMD, which the pose model needs.');

  const need = [];
  if (!wasmUrl) need.push('wasm');
  if (!modelBytes[model]) need.push(model);
  const prog = { wasm: [0, KNOWN.wasm], [model]: [0, KNOWN[model]] };
  const report = () => {
    let l = 0, t = 0; for (const k of need) { l += prog[k][0]; t += prog[k][1]; }
    onProgress({ phase: 'download', loaded: l, total: t });
  };
  if (need.length) report();
  const jobs = [];
  if (!visionMod) jobs.push(import(FILES.bundle).then(m => { visionMod = m; }).catch(e => { throw new PoseError('model', 'The AI library failed to load.', e); }));
  if (need.includes('wasm')) jobs.push(fetchBytes(FILES.wasm, KNOWN.wasm, (l, t) => { prog.wasm = [l, t]; report(); }, signal)
    .then(b => { wasmUrl = URL.createObjectURL(new Blob([b], { type: 'application/wasm' })); }));
  if (need.includes(model)) jobs.push(fetchBytes(FILES[model], KNOWN[model], (l, t) => { prog[model] = [l, t]; report(); }, signal)
    .then(b => { modelBytes[model] = b; }));
  await Promise.all(jobs);
  if (signal && signal.aborted) throw new PoseError('aborted', 'Cancelled');

  onProgress({ phase: 'init' });
  const fileset = { wasmLoaderPath: FILES.loader, wasmBinaryPath: wasmUrl };
  const make = (delegate) => visionMod.PoseLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetBuffer: modelBytes[model].slice(), delegate },
    runningMode: 'VIDEO', numPoses: 1,
    minPoseDetectionConfidence: 0.5, minPosePresenceConfidence: 0.5, minTrackingConfidence: 0.5,
    outputSegmentationMasks: false,
  });
  let landmarker = null, delegate = 'GPU', fellBack = false, gpuErr = null;
  if (hasWebGL2()) {
    try { landmarker = await make('GPU'); } catch (e) { gpuErr = e; }
  } else gpuErr = new Error('WebGL2 unavailable');
  if (!landmarker) {
    delegate = 'CPU'; fellBack = true;
    try { landmarker = await make('CPU'); }
    catch (e) { throw new PoseError('model', 'The pose model could not start on this device.', e); }
  }
  cache[model] = { landmarker, delegate, gpuErr };
  return { landmarker, delegate, fellBack, gpuErr };
}

/** Replace a cached GPU landmarker with a CPU one (used when GPU detection throws mid-run). */
export async function downgradeToCPU(model) {
  const c = cache[model];
  try { c && c.landmarker && c.landmarker.close(); } catch (e) {}
  delete cache[model];
  const fileset = { wasmLoaderPath: FILES.loader, wasmBinaryPath: wasmUrl };
  const landmarker = await visionMod.PoseLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetBuffer: modelBytes[model].slice(), delegate: 'CPU' },
    runningMode: 'VIDEO', numPoses: 1, outputSegmentationMasks: false,
  });
  cache[model] = { landmarker, delegate: 'CPU' };
  return cache[model];
}

function hasWebGL2() {
  try {
    const c = document.createElement('canvas'), gl = c.getContext('webgl2'); if (!gl) return false;
    // Software rasterizers (SwiftShader, llvmpipe) make the GPU delegate ~10x slower than wasm SIMD on CPU.
    const ext = gl.getExtension('WEBGL_debug_renderer_info');
    const r = ext ? String(gl.getParameter(ext.UNMASKED_RENDERER_WEBGL)) : '';
    const lose = gl.getExtension('WEBGL_lose_context'); if (lose) lose.loseContext();
    if (/swiftshader|llvmpipe|software|basic render/i.test(r)) { softwareGL = r; return false; }
    return true;
  } catch (e) { return false; }
}
let softwareGL = null;
export const gpuInfo = () => ({ softwareGL });

function toF32(lms) {
  if (!lms || !lms.length) return null;
  const a = new Float32Array(33 * 4);
  for (let i = 0; i < 33 && i < lms.length; i++) {
    const p = lms[i];
    a[i * 4] = p.x; a[i * 4 + 1] = p.y; a[i * 4 + 2] = p.z || 0; a[i * 4 + 3] = p.visibility == null ? 1 : p.visibility;
  }
  return a;
}

const once = (el, ev, ms = 8000) => new Promise((res, rej) => {
  let to;
  const ok = () => { clearTimeout(to); el.removeEventListener(ev, ok); el.removeEventListener('error', bad); res(); };
  const bad = () => { clearTimeout(to); el.removeEventListener(ev, ok); el.removeEventListener('error', bad); rej(new PoseError('codec', 'The video could not be decoded.')); };
  el.addEventListener(ev, ok); el.addEventListener('error', bad);
  to = setTimeout(() => { el.removeEventListener(ev, ok); el.removeEventListener('error', bad); rej(new PoseError('extract', 'Timed out waiting for the video (' + ev + ').')); }, ms);
});

async function seekTo(video, t) {
  if (Math.abs(video.currentTime - t) < 1e-4 && video.readyState >= 2) return;
  const p = once(video, 'seeked');
  video.currentTime = t;
  await p;
}

/**
 * Run pose detection over [start, end] seconds of `video`.
 * Primary path: requestVideoFrameCallback while playing at an adaptive playbackRate <= 1
 * (every decoded frame gets an exact mediaTime). Fallback: seek-by-seek at 30 fps.
 * Output is resampled onto a uniform time grid so the engine sees a constant fps.
 */
export async function extractPose(video, opts) {
  const { start, end, onFrame = () => {}, signal, maxSide = 640, model = 'lite' } = opts;
  let landmarker = opts.landmarker;
  const W0 = video.videoWidth, H0 = video.videoHeight;
  if (!W0 || !H0) throw new PoseError('codec', 'The video has no picture this browser can decode.');
  const s = Math.min(1, maxSide / Math.max(W0, H0));
  const w = Math.max(2, Math.round(W0 * s)), h = Math.max(2, Math.round(H0 * s));
  const canvas = document.createElement('canvas'); canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext('2d', { alpha: false, desynchronized: true });
  const dur = Math.max(0.05, end - start);
  const nativeFps = opts.nativeFps || 30;
  const raw = [];                      // { t (rel s), lm }
  const t0 = performance.now();
  let runBase = tsCursor + 1000;       // ms; keep timestamps strictly increasing across runs
  let detectMs = 30, gpuFailed = false;
  const stats = { path: 'rvfc', detectMs: 0, delegateSwitched: false };

  const detect = async (tRel) => {
    let ts = Math.floor(runBase + tRel * 1000);
    if (ts <= tsCursor) ts = tsCursor + 1;
    tsCursor = ts;
    const a = performance.now();
    let res;
    try { res = landmarker.detectForVideo(canvas, ts); }
    catch (e) {
      if (gpuFailed) throw new PoseError('extract', 'Pose detection failed on this frame.', e);
      gpuFailed = true; stats.delegateSwitched = true;
      landmarker = (await downgradeToCPU(model)).landmarker;
      runBase = tsCursor + 1000; tsCursor = runBase;
      res = landmarker.detectForVideo(canvas, tsCursor);
    }
    const d = performance.now() - a;
    detectMs = detectMs * 0.8 + d * 0.2;
    const lm = toF32(res && res.landmarks && res.landmarks[0]);
    return lm;
  };
  const progress = (tRel, lm) => {
    const frac = Math.min(1, Math.max(0, tRel / dur));
    const el = (performance.now() - t0) / 1000;
    const eta = frac > 0.04 ? el / frac * (1 - frac) : null;
    onFrame({ frac, done: raw.length, expected: Math.round(dur * Math.min(60, nativeFps)), eta, canvas, lm, t: tRel, detectMs });
  };

  const useRVFC = 'requestVideoFrameCallback' in HTMLVideoElement.prototype && !opts.forceSeek;
  if (useRVFC) {
    try { await runRVFC(); }
    catch (e) {
      if (e && e.code === 'aborted') throw e;
      if (e && e.code === 'fallback') { stats.path = 'seek'; stats.fallback = e.message + (e.cause ? ': ' + (e.cause.message || e.cause) : ''); console.info('[jumpshot] frame callbacks unavailable, using seek mode —', stats.fallback); raw.length = 0; await runSeek(); }
      else throw e;
    }
  } else { stats.path = 'seek'; await runSeek(); }
  if (signal && signal.aborted) throw new PoseError('aborted', 'Cancelled');
  stats.detectMs = detectMs;
  stats.rawFrames = raw.length;
  stats.wall = (performance.now() - t0) / 1000;

  // ---- resample onto a uniform grid
  const intervals = [];
  for (let i = 1; i < raw.length; i++) { const d = raw[i].t - raw[i - 1].t; if (d > 1e-4) intervals.push(d); }
  intervals.sort((a, b) => a - b);
  const minI = intervals.length ? intervals[Math.floor(intervals.length * 0.1)] : 1 / 30;
  let fps = Math.min(60, 1 / minI, nativeFps * 1.05);
  const effective = raw.length / dur;
  fps = Math.min(fps, Math.max(15, effective * 1.15));
  const STD = [15, 24, 25, 30, 50, 60];
  fps = STD.reduce((b, v) => (v <= fps + 0.5 ? v : b), 15);
  const n = Math.max(1, Math.floor(dur * fps + 1e-6) + 1);
  const frames = new Array(n);
  let j = 0;
  for (let k = 0; k < n; k++) {
    const g = k / fps;
    while (j + 1 < raw.length && raw[j + 1].t <= g) j++;
    const a = raw[j], b = raw[j + 1];
    let lm = null;
    if (a && b && a.t <= g && b.t >= g && a.lm && b.lm && (b.t - a.t) <= 2.5 / fps) {
      const u = (g - a.t) / Math.max(1e-6, b.t - a.t);
      lm = new Float32Array(132);
      for (let q = 0; q < 132; q++) lm[q] = a.lm[q] + (b.lm[q] - a.lm[q]) * u;
    } else {
      const near = [a, b].filter(Boolean).sort((x, y) => Math.abs(x.t - g) - Math.abs(y.t - g))[0];
      if (near && near.lm && Math.abs(near.t - g) <= 0.75 / fps) lm = near.lm;
    }
    frames[k] = { t: +g.toFixed(4), lm };
  }
  raw.length = 0;
  return { frames, meta: { fps, width: W0, height: H0, duration: dur }, stats };

  // ---------------------------------------------------------------------------
  function runRVFC() {
    return new Promise(async (resolve, reject) => {
      let finished = false, lastMt = -1, rate = Math.min(1, Math.max(0.15, 550 / (nativeFps * 45))), lastFrameWall = performance.now();
      const cleanup = () => {
        finished = true; clearInterval(watch);
        video.removeEventListener('ended', onEnded); video.removeEventListener('error', onErr);
        try { video.pause(); } catch (e) {}
        if (signal) signal.removeEventListener('abort', onAbort);
      };
      const done = () => { if (finished) return; cleanup(); resolve(); };
      const onEnded = () => done();
      const onErr = () => { if (finished) return; cleanup(); reject(new PoseError('codec', 'The video stopped decoding.')); };
      const onAbort = () => { if (finished) return; cleanup(); reject(new PoseError('aborted', 'Cancelled')); };
      video.addEventListener('ended', onEnded); video.addEventListener('error', onErr);
      if (signal) signal.addEventListener('abort', onAbort);
      const watch = setInterval(() => {
        // No frame callbacks for 4 s (background tab, throttled decoder…) -> seek fallback.
        if (!finished && performance.now() - lastFrameWall > Math.max(4000, detectMs * 8)) {
          if (document.hidden) { lastFrameWall = performance.now(); return; }
          cleanup(); reject(Object.assign(new Error('rvfc stalled'), { code: 'fallback' }));
        }
      }, 500);
      const onVF = async (now, md) => {
        if (finished) return;
        lastFrameWall = performance.now();
        const mt = md.mediaTime;
        if (mt >= end - 0.5 / nativeFps) { done(); return; }
        if (mt + 1e-3 >= start && mt > lastMt + 1e-4) {
          lastMt = mt;
          ctx.drawImage(video, 0, 0, w, h);
          const tRel = mt - start;
          let lm;
          try { lm = await detect(tRel); } catch (e) { cleanup(); reject(e); return; }
          lastFrameWall = performance.now();
          raw.push({ t: tRel, lm });
          progress(tRel, lm);
          // keep detection ahead of playback: frames/s arriving * cost <= ~85 % of wall time
          const want = Math.max(0.07, Math.min(1, 550 / (Math.min(nativeFps, 60) * Math.max(4, detectMs + 6))));
          if (Math.abs(want - rate) / rate > 0.15) { rate = want; try { video.playbackRate = rate; } catch (e) {} }
        }
        if (!finished) video.requestVideoFrameCallback(onVF);
      };
      try {
        video.muted = true; video.playsInline = true; video.loop = false;
        video.pause();
        await seekTo(video, start);
        video.playbackRate = rate;
        video.requestVideoFrameCallback(onVF);
        await video.play();
      } catch (e) {
        if (finished) return;
        cleanup();
        if (e && e.code === 'codec') reject(e);
        else reject(Object.assign(new Error('play failed'), { code: 'fallback', cause: e }));
      }
    });
  }

  async function runSeek() {
    const step = 1 / 30;
    video.pause();
    for (let t = 0; t <= dur + 1e-6; t += step) {
      if (signal && signal.aborted) throw new PoseError('aborted', 'Cancelled');
      await seekTo(video, start + t);
      ctx.drawImage(video, 0, 0, w, h);
      const lm = await detect(t);
      raw.push({ t, lm });
      progress(t, lm);
      if (raw.length % 4 === 0) await new Promise(r => setTimeout(r, 0));   // let the UI paint
    }
  }
}

/** Seek a (paused, dedicated) video to t and return a canvas of that frame, long side <= maxSide. */
export async function grabFrame(video, t, maxSide = 480) {
  await seekTo(video, Math.max(0, Math.min(t, (video.duration || t) - 1e-3)));
  if (video.readyState < 2) await once(video, 'loadeddata', 4000).catch(() => {});
  const s = Math.min(1, maxSide / Math.max(video.videoWidth, video.videoHeight));
  const c = document.createElement('canvas');
  c.width = Math.max(2, Math.round(video.videoWidth * s)); c.height = Math.max(2, Math.round(video.videoHeight * s));
  c.getContext('2d').drawImage(video, 0, 0, c.width, c.height);
  return c;
}

export const MODEL_SIZES = KNOWN;
