// Jumpshot Analyzer — in-browser app. Video never leaves the device.
// Views: home · record · clip · processing · shots · results · history · compare · error  (hash router)
import { ensureLandmarker, extractPose, grabFrame, PoseError, MODEL_SIZES } from './pose.mjs';

const APP_BASE = new URL('./', import.meta.url);
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
const esc = s => String(s == null ? '' : s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const num = v => (v == null || v === '' ? NaN : +v);
const fin = v => Number.isFinite(num(v));
const RM = matchMedia('(prefers-reduced-motion: reduce)');
const COARSE = matchMedia('(pointer: coarse)').matches || /Android|iPhone|iPad|iPod/i.test(navigator.userAgent);
const css = getComputedStyle(document.body);
const tok = (n, d) => (css.getPropertyValue(n) || '').trim() || d;
const COL = { accent: tok('--accent', '#ffa030'), muted: tok('--muted', '#8a93a6'), text: tok('--text', '#e6e9f0'), line: tok('--line-2', '#2e3545'),
  bg: tok('--surface-2', '#181c25'), pos: tok('--pos', '#3ddc84'), neg: tok('--neg', '#ff5c7a'), warn: tok('--warn', '#ffb020'), faint: tok('--faint', '#5b6478'), ink: '#0b0d12' };
const MAX_WINDOW = 15;

// ------------------------------------------------------------------ state & prefs
const prefs = Object.assign({ side: 'auto', model: 'lite' }, safeJSON(localStorage.getItem('jumpshot.prefs')));
function savePrefs() { try { localStorage.setItem('jumpshot.prefs', JSON.stringify(prefs)); } catch (e) {} }
function safeJSONArr(s) { try { const a = JSON.parse(s); return Array.isArray(a) ? a : []; } catch (e) { return []; } }
function safeJSON(s) { try { return JSON.parse(s) || {}; } catch (e) { return {}; } }

const state = {
  file: null, url: null, source: null,             // the clip in memory (object URL, never uploaded)
  vmeta: null, trim: [0, 0], fpsEst: null,
  busy: false, abort: null,
  frames: null, pmeta: null, shots: null, trimStart: 0,
  session: null,                                   // current session (saved in IndexedDB)
  sessionVideo: null,                              // { id, url, trimStart } when the session's video is in memory
  error: null, compare: [],
};
window.__jumpshot = { state, prefs };

// ------------------------------------------------------------------ engine (lazy)
let engineP = null, assetsP = null;
function getEngine() {
  return engineP || (engineP = import('./engine.mjs').catch(e => { engineP = null; throw e; }));
}
function getAssets() {
  return assetsP || (assetsP = getEngine().then(e => e.loadAssets(APP_BASE.href)).catch(e => { assetsP = null; throw e; }));
}

// ------------------------------------------------------------------ utilities
function toast(msg, ms = 2600) {
  const t = $('#toast'); t.textContent = msg; t.classList.add('show');
  clearTimeout(toast._t); toast._t = setTimeout(() => t.classList.remove('show'), ms);
}
const buzz = p => { try { navigator.vibrate && navigator.vibrate(p); } catch (e) {} };
let wakeLock = null;
async function keepAwake(on) {
  try {
    if (on && 'wakeLock' in navigator && !wakeLock) { wakeLock = await navigator.wakeLock.request('screen'); wakeLock.addEventListener('release', () => { wakeLock = null; }); }
    if (!on && wakeLock) { await wakeLock.release(); wakeLock = null; }
  } catch (e) { wakeLock = null; }
}
document.addEventListener('visibilitychange', () => { if (state.busy && document.visibilityState === 'visible') keepAwake(true); });
window.addEventListener('beforeunload', e => { if (state.busy) { e.preventDefault(); e.returnValue = ''; } });

const fmtS = (v, d = 2) => (fin(v) ? num(v).toFixed(d) + ' s' : '—');
const fmtDate = (iso, opts) => { try { return new Date(iso).toLocaleDateString(undefined, opts || { month: 'short', day: 'numeric', year: 'numeric' }); } catch (e) { return String(iso).slice(0, 10); } };
const fmtTime = iso => { try { return new Date(iso).toLocaleTimeString(undefined, { hour: 'numeric', minute: '2-digit' }); } catch (e) { return ''; } };
const fmtBytes = b => (b > 1e6 ? (b / 1e6).toFixed(1) + ' MB' : Math.round(b / 1e3) + ' KB');
function fmtVal(v, unit, signed) {
  if (!fin(v)) return '—';
  v = num(v); const sg = signed && v > 0 ? '+' : '';
  if (unit === 'deg') return sg + v.toFixed(0) + '°';
  if (unit === 's') return sg + v.toFixed(2) + ' s';
  if (unit === 'torso') return sg + v.toFixed(2) + ' torso';
  if (unit === '%') return sg + v.toFixed(0) + '%';
  return sg + v.toFixed(2);
}
const fmtNum = (v, unit) => { if (!fin(v)) return '—'; v = num(v); return unit === 'deg' ? v.toFixed(0) + '°' : unit === 's' ? v.toFixed(2) : v.toFixed(2); };
const splitName = n => { const s = String(n || 'Reference').split(/\s+[—–-]\s+/); return { title: s[0], style: s.slice(1).join(' — ') }; };
const mean = a => a.reduce((x, y) => x + y, 0) / (a.length || 1);
const sd = a => { if (a.length < 2) return 0; const m = mean(a); return Math.sqrt(a.reduce((x, y) => x + (y - m) * (y - m), 0) / (a.length - 1)); };

// summary metric lookups (tolerant to key naming between engine versions)
const MKEYS = {
  load: ['release_time_s', 'load_to_release_s', 'load_release_s'], relh: ['release_height'], elbow: ['elbow_angle_set', 'elbow_at_set'],
  knee: ['knee_angle_min', 'knee_min'], jump: ['jump_height'], hold: ['follow_through_hold_s', 'hold_s'], elbowRel: ['elbow_angle_release'],
  fwd: ['release_forward'], lean: ['trunk_lean_release'],
};
const MUNIT = { load: 's', relh: 'torso', elbow: 'deg', knee: 'deg', jump: 'torso', hold: 's', elbowRel: 'deg', fwd: 'torso', lean: 'deg' };
const MLABEL = { load: 'Load → release', relh: 'Release height', elbow: 'Elbow @ set', knee: 'Knee bend (min)', jump: 'Jump height', hold: 'Follow-through hold', elbowRel: 'Elbow @ release', fwd: 'Release forward', lean: 'Trunk lean @ release' };
function metric(r, k) { const s = (r && r.summary) || {}; for (const key of MKEYS[k]) if (fin(s[key])) return num(s[key]); return NaN; }
function fbOf(r, k) { return ((r && r.feedback) || []).find(f => MKEYS[k].includes(f.key)); }
const fbRef = f => num(f.ref != null ? f.ref : f.reference);
const fbText = f => f.text || f.message || '';
const sevRank = s => (s === 'major' ? 2 : s === 'minor' ? 1 : 0);
const sevCls = s => (s === 'major' ? 'neg' : s === 'minor' ? 'warn' : s === 'ok' ? 'pos' : '');

// ---- engine v2 scores: mechanics score (primary) + style match (secondary).
// Results saved before engine v2 have neither `mechanics` nor `styleMatch`: mechanics renders "—",
// style match falls back to closest.score (the old headline was exactly that similarity score).
const hasMech = r => !!(r && Array.isArray(r.mechanics) && r.mechanics.length);
function mechScore(r) { if (!r) return NaN; if (fin(r.mechanicsScore)) return num(r.mechanicsScore); return r.scoreKind === 'mechanics' && fin(r.score) ? num(r.score) : NaN; }
function styleScore(r) { if (!r) return NaN; if (fin(r.styleMatch)) return num(r.styleMatch); if (r.closest && fin(r.closest.score)) return num(r.closest.score); return !hasMech(r) && !r.scoreKind && fin(r.score) ? num(r.score) : NaN; }
const sessResults = s => (s ? (Array.isArray(s.results) && s.results.length ? s.results : s.result ? [s.result] : []) : []);
function sessScores(s) {
  const rs = sessResults(s); const m = rs.map(mechScore).filter(Number.isFinite), st = rs.map(styleScore).filter(Number.isFinite);
  return { mech: m.length ? mean(m) : NaN, style: st.length ? mean(st) : NaN, n: rs.length };
}
const f1 = v => (fin(v) ? num(v).toFixed(1) : '—');
const MSTATUS = { good: 'pos', close: 'warn', off: 'neg', unmeasured: 'muted' };
const MSTATUS_TXT = { good: 'in range', close: 'close', off: 'off', unmeasured: 'not measured' };
const gainOf = m => (m && m.status !== 'unmeasured' && fin(m.points) && fin(m.maxPoints) ? Math.max(0, num(m.maxPoints) - num(m.points)) : 0);
const fmtPts = v => (fin(v) ? (Math.abs(num(v) - Math.round(num(v))) < 0.05 ? String(Math.round(num(v))) : num(v).toFixed(1)) : '—');
function topFixes(r, n = 2) { return (hasMech(r) ? r.mechanics : []).filter(m => (m.status === 'off' || m.status === 'close') && gainOf(m) > 0).sort((a, b) => gainOf(b) - gainOf(a)).slice(0, n); }
const MINUS = '−';
function fmtMech(v, unit, key) {
  if (!fin(v)) return '—';
  v = num(v);
  if (unit === 'deg') return v.toFixed(0) + '°';
  if (unit === 's') return (key === 'release_timing' ? (v > 0.005 ? '+' : v < -0.005 ? MINUS : '') + Math.abs(v).toFixed(2) : v.toFixed(2)) + ' s';
  if (unit === 'torso') return v.toFixed(2) + ' torso';
  return v.toFixed(2);
}
const tn = x => String(parseFloat(num(x).toFixed(2)));
const fmtIdeal = (v, unit, key) => (!fin(v) ? '—' : unit === 'deg' ? num(v).toFixed(0) + '°' : unit === 's' ? (key === 'release_timing' && num(v) < 0 ? MINUS : '') + tn(Math.abs(num(v))) + ' s' : unit === 'torso' ? tn(v) + ' torso' : tn(v));
function fmtRange(band, unit) {
  if (!band || band.length < 2) return '—';
  const [lo, hi] = band.map(num);
  if (unit === 'deg') return `${lo.toFixed(0)}–${hi.toFixed(0)}°`;
  if (unit === 's') { if (lo >= 0) return `${tn(lo)}–${tn(hi)} s`; const f = x => (x < 0 ? MINUS : x > 0 ? '+' : '') + tn(Math.abs(x)); return `${f(lo)} to ${f(hi)} s`; }
  if (unit === 'torso') return `${tn(lo)}–${tn(hi)} torso`;
  return `${lo}–${hi}`;
}
/** Plain-English instruction that moves a check back into its range. */
function fixPhrase(m) {
  const v = num(m.value), hi = num((m.band || [])[1]), up = v > hi, id = num(m.ideal);
  const d = x => Math.round(x) + '°';
  switch (m.key) {
    case 'set_elbow': return `get your elbow to ~${d(id)} at the set point${up ? ' (fold it a little more)' : ' (open it up a little)'}`;
    case 'dip_knee': return up ? `sink deeper in the dip — bend your knees to ~${d(id)}` : `ease off the dip — bend your knees to about ${d(id)}, not deeper`;
    case 'release_elbow': return up ? `finish with a softer elbow at release (~${d(id)})` : `extend your elbow more at release — toward ~${d(id)}`;
    case 'release_shoulder': return up ? `keep the ball in front at release — shoulder at ~${d(id)}, not behind your head` : `finish higher at release — lift your arm to ~${d(id)} at the shoulder`;
    case 'release_timing': return up ? 'let the ball go at the top of your jump, not on the way down' : 'let the ball go later — right at the top of your jump';
    case 'follow_through': return up ? 'relax your follow-through a little sooner' : 'hold your follow-through longer (about half a second)';
    case 'balance_lean': return 'keep your trunk angle steady from the set point to release';
    case 'balance_drift': return 'jump straight up — keep your hips over your feet';
    default: return `bring ${String(m.label || m.key).toLowerCase()} into the ${fmtRange(m.band, m.unit)} range`;
  }
}
function mechSources(assets) {
  const list = (assets && assets.sources && assets.sources.length ? assets.sources : (assets && assets.library && assets.library.mechanics && assets.library.mechanics.sources)) || [];
  return { list, n: Object.fromEntries(list.map((s, i) => [s.id, i + 1])) };
}
const supRefs = (ids, S) => { const ns = (ids || []).map(id => S.n[id]).filter(Boolean).sort((a, b) => a - b); return ns.length ? `<sup class="srcs-sup">${ns.map(n => `<button type="button" class="src-ref" data-src="${n}" aria-label="Source ${n}">${n}</button>`).join('')}</sup>` : ''; };

// landmarks: Float32Array(33*k) flat, plain flat array, or nested [[x,y,(z),(v)]...]
function P(lm, i) {
  if (!lm) return null;
  if (typeof lm[0] === 'object' && lm[0] !== null) { const p = lm[i]; if (!p) return null; const x = num(p[0] != null ? p[0] : p.x), y = num(p[1] != null ? p[1] : p.y); const v = p.length > 3 ? num(p[3]) : (p.visibility != null ? num(p.visibility) : 1); return Number.isFinite(x) && Number.isFinite(y) ? [x, y, Number.isFinite(v) ? v : 1] : null; }
  const k = Math.round(lm.length / 33) || 4; const x = num(lm[i * k]), y = num(lm[i * k + 1]); const v = k >= 4 ? num(lm[i * k + 3]) : 1;
  return Number.isFinite(x) && Number.isFinite(y) ? [x, y, Number.isFinite(v) ? v : 1] : null;
}

// ------------------------------------------------------------------ IndexedDB (no video stored)
const DB = {
  _db: null,
  open() {
    if (this._db) return Promise.resolve(this._db);
    return new Promise((res, rej) => {
      const rq = indexedDB.open('jumpshot', 1);
      rq.onupgradeneeded = () => { const db = rq.result; if (!db.objectStoreNames.contains('sessions')) db.createObjectStore('sessions', { keyPath: 'id' }).createIndex('created', 'created'); };
      rq.onsuccess = () => { this._db = rq.result; res(this._db); };
      rq.onerror = () => rej(rq.error);
    });
  },
  async tx(mode, fn) {
    const db = await this.open();
    return new Promise((res, rej) => { const t = db.transaction('sessions', mode); const st = t.objectStore('sessions'); let out; Promise.resolve(fn(st)).then(v => { out = v; }); t.oncomplete = () => res(out); t.onerror = () => rej(t.error); t.onabort = () => rej(t.error); });
  },
  put(s) { return this.tx('readwrite', st => { st.put(s); }); },
  del(id) { return this.tx('readwrite', st => { st.delete(id); }); },
  get(id) { return this.tx('readonly', st => new Promise(r => { const q = st.get(id); q.onsuccess = () => r(q.result || null); })); },
  all() { return this.tx('readonly', st => new Promise(r => { const q = st.getAll(); q.onsuccess = () => r((q.result || []).sort((a, b) => (a.created < b.created ? -1 : 1))); })); },
  clear() { return this.tx('readwrite', st => { st.clear(); }); },
};
window.__jumpshot.DB = DB;
async function refreshCount() {
  try { const n = (await DB.all()).length; $('#hist-count').textContent = n ? String(n) : ''; return n; } catch (e) { return 0; }
}

// ------------------------------------------------------------------ router
const VIEWS = ['home', 'record', 'clip', 'processing', 'shots', 'results', 'history', 'compare', 'error'];
let current = null, leaveHooks = [];
function go(h, replace) { if (replace) history.replaceState(null, '', h); else if (location.hash !== h) { location.hash = h; return; } route(); }
function onLeave(fn) { leaveHooks.push(fn); }
async function route() {
  const h = location.hash.replace(/^#\/?/, '');
  const [name, a, b] = h.split('/');
  let view = name || 'home';
  if (view === 'session') view = 'results';
  if (!VIEWS.includes(view)) view = 'home';
  // guards
  if (view === 'clip' && !state.file) return go('#/', true);
  if (view === 'processing' && !state.busy) return go(state.session ? '#/results' : '#/', true);
  if (view === 'shots' && !state.shots) return go('#/', true);
  if (view === 'results' && name === 'results' && !state.session) return go('#/history', true);
  if (view === 'error' && !state.error) return go('#/', true);
  if (current === 'processing' && view !== 'processing' && state.busy) { state.abort && state.abort.abort(); toast('Analysis cancelled'); }
  leaveHooks.splice(0).forEach(f => { try { f(); } catch (e) {} });
  current = view;
  for (const v of VIEWS) { const el = $(`[data-view="${v}"]`); if (el) el.hidden = v !== view; }
  $$('.subnav a[data-nav]').forEach(x => x.classList.toggle('on', x.dataset.nav === (['history', 'compare'].includes(view) || name === 'session' ? 'history' : 'home')));
  $('#method').hidden = !['home', 'results'].includes(view);
  if (!(view === 'results' && a)) window.scrollTo(0, 0);
  try {
    if (view === 'home') enterHome();
    else if (view === 'record') enterRecord();
    else if (view === 'clip') enterClip();
    else if (view === 'shots') renderShots();
    else if (view === 'results') {
      if (name === 'session') { const s = await DB.get(a); if (!s) { toast('Session not found'); return go('#/history', true); } state.session = s; }
      renderResults(state.session, b != null ? +b : state.session.primary || 0);
      window.scrollTo(0, 0);
    }
    else if (view === 'history') renderHistory();
    else if (view === 'compare') renderCompare(a, b);
    else if (view === 'error') renderError();
  } catch (e) { console.error(e); showError(errUnknown(e)); }
}
window.addEventListener('hashchange', route);
document.addEventListener('click', e => { const b = e.target.closest('[data-back]'); if (b) { e.preventDefault(); history.length > 1 ? history.back() : go('#/'); } });

// ------------------------------------------------------------------ home
function enterHome() {
  refreshCount();
  DB.all().then(list => {
    const box = $('#home-last'); const s = list[list.length - 1];
    if (!s) { box.hidden = true; return; }
    box.hidden = false;
    const prev = list[list.length - 2];
    const cur = sessScores(s), pv = prev ? sessScores(prev) : null;
    const d = pv && fin(cur.mech) && fin(pv.mech) ? cur.mech - pv.mech : null;
    box.innerHTML = `<div class="label">Your last session</div>
      <a class="last" href="#/session/${esc(s.id)}" style="text-decoration:none;margin-top:10px">
        ${s.thumb ? `<img src="${esc(s.thumb)}" alt="">` : '<span class="noimg"></span>'}
        <div><b style="display:block">${esc(s.name || 'Shot')}</b><small class="muted">${esc(fmtDate(s.created))} · style ${esc(f1(cur.style))} vs ${esc(splitName(s.closest).title)}</small>
        ${d != null ? `<small class="${d >= 0 ? 'pos' : 'neg'} num" style="display:block">${d >= 0 ? '▲' : '▼'} ${Math.abs(d).toFixed(1)} vs previous</small>` : ''}</div>
        <span class="sc num" title="Mechanics score">${f1(cur.mech)}</span></a>
      <div class="row-between" style="margin-top:10px"><span class="muted" style="font-size:12px">${list.length} session${list.length > 1 ? 's' : ''} saved on this device</span><a class="btn sm" href="#/history">History</a></div>`;
  }).catch(() => {});
}
function pickFile(file, source) {
  if (!file) return;
  if (file.type && !/^video\//.test(file.type) && !/\.(mp4|mov|m4v|webm|mkv|avi|3gp)$/i.test(file.name || '')) {
    return showError({ icon: 'warn', title: "That doesn't look like a video", msg: `“${file.name}” is a ${file.type || 'unknown'} file.`, fixes: ['Pick a video from your camera roll (MP4, MOV or WebM).'], actions: [{ label: 'Choose another', primary: true, go: '#/' }] });
  }
  if (state.url) URL.revokeObjectURL(state.url);
  state.file = file; state.url = URL.createObjectURL(file); state.source = source; state.vmeta = null; state.fpsEst = null;
  go('#/clip');
}
$('#file-upload').addEventListener('change', e => { pickFile(e.target.files[0], 'upload'); e.target.value = ''; });
$('#file-capture').addEventListener('change', e => { pickFile(e.target.files[0], 'record'); e.target.value = ''; });
$('#btn-record').addEventListener('click', () => {
  const canRec = navigator.mediaDevices && navigator.mediaDevices.getUserMedia && window.MediaRecorder;
  if (COARSE || !canRec) $('#file-capture').click(); else go('#/record');
});
if (COARSE) $('#record-sub').textContent = 'Opens your camera · 2–6 seconds';
$('#btn-sample').addEventListener('click', runSample);

// ------------------------------------------------------------------ recorder (desktop / in-page)
let rec = null;
async function enterRecord() {
  const v = $('#rec-video'), go_ = $('#rec-go'), stop = $('#rec-stop'), flip = $('#rec-flip');
  rec = { facing: 'environment', stream: null, mr: null, chunks: [], timer: null, cd: null };
  const start = async () => {
    stopStream();
    try {
      rec.stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: { facingMode: rec.facing, width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 60 } } });
      v.srcObject = rec.stream; await v.play().catch(() => {});
      const s = rec.stream.getVideoTracks()[0].getSettings();
      $('#rec-note').textContent = `${s.width || '?'}×${s.height || '?'} · ${s.frameRate ? Math.round(s.frameRate) + ' fps' : ''} — 3-2-1 countdown, then records up to 6 s.`;
      if (s.width && s.height) $('.rec-stage').style.aspectRatio = s.width + '/' + s.height;
    } catch (e) {
      showError({ title: 'Camera not available', msg: e && e.name === 'NotAllowedError' ? 'Camera access was blocked.' : 'We could not open a camera on this device.',
        fixes: ['Allow camera access in your browser’s address bar, then try again.', 'Or record with your phone’s camera app and use “Upload a clip”.'],
        actions: [{ label: 'Try again', primary: true, go: '#/record' }, { label: 'Upload instead', go: '#/', click: () => $('#file-upload').click() }], code: e && (e.name + ': ' + e.message) });
    }
  };
  function stopStream() { if (rec && rec.stream) rec.stream.getTracks().forEach(t => t.stop()); }
  const record = () => {
    if (!rec.stream) return;
    go_.disabled = true; flip.disabled = true;
    let n = 3; const cEl = $('#rec-count'); cEl.textContent = n; buzz(10);
    rec.cd = setInterval(() => {
      n--; if (n > 0) { cEl.textContent = n; buzz(10); return; }
      clearInterval(rec.cd); cEl.textContent = ''; begin();
    }, 1000);
  };
  const begin = () => {
    const types = ['video/mp4;codecs=avc1', 'video/mp4', 'video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm'];
    const type = types.find(t => MediaRecorder.isTypeSupported && MediaRecorder.isTypeSupported(t)) || '';
    try { rec.mr = new MediaRecorder(rec.stream, type ? { mimeType: type, videoBitsPerSecond: 8e6 } : undefined); }
    catch (e) { rec.mr = new MediaRecorder(rec.stream); }
    rec.chunks = [];
    rec.mr.ondataavailable = e => { if (e.data && e.data.size) rec.chunks.push(e.data); };
    rec.mr.onstop = () => {
      clearInterval(rec.timer); $('#rec-badge').hidden = true; go_.classList.remove('recording');
      const mt = (rec.mr.mimeType || type || 'video/webm').split(';')[0];
      const blob = new Blob(rec.chunks, { type: mt });
      if (!blob.size) { toast('Nothing was recorded — try again'); go_.disabled = false; flip.disabled = false; return; }
      const f = new File([blob], 'recording-' + new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-') + (mt.includes('mp4') ? '.mp4' : '.webm'), { type: mt });
      stopStream(); buzz(30);
      pickFile(f, 'record');
    };
    rec.mr.start(250);
    const t0 = performance.now(); go_.classList.add('recording'); $('#rec-badge').hidden = false; stop.disabled = false;
    rec.timer = setInterval(() => {
      const el = (performance.now() - t0) / 1000;
      $('#rec-time').textContent = el.toFixed(1) + ' s'; $('#rec-prog').style.width = Math.min(100, el / 6 * 100) + '%';
      if (el >= 6) finish();
    }, 100);
  };
  const finish = () => { clearInterval(rec.timer); stop.disabled = true; if (rec.mr && rec.mr.state !== 'inactive') rec.mr.stop(); };
  go_.onclick = record; stop.onclick = finish;
  flip.onclick = () => { rec.facing = rec.facing === 'environment' ? 'user' : 'environment'; start(); };
  go_.disabled = false; flip.disabled = false; stop.disabled = true; $('#rec-badge').hidden = true; $('#rec-prog').style.width = '0'; $('#rec-count').textContent = '';
  onLeave(() => { clearInterval(rec.cd); clearInterval(rec.timer); if (rec.mr && rec.mr.state !== 'inactive') { rec.mr.onstop = null; rec.mr.stop(); } stopStream(); v.srcObject = null; });
  await start();
}

// ------------------------------------------------------------------ video helpers
function waitEv(el, ev, ms) {
  return new Promise((res, rej) => {
    const ok = () => { done(); res(true); }, bad = () => { done(); rej(new PoseError('codec', 'decode error ' + (el.error ? el.error.code : ''))); };
    const to = setTimeout(() => { done(); res(false); }, ms);
    const done = () => { clearTimeout(to); el.removeEventListener(ev, ok); el.removeEventListener('error', bad); };
    el.addEventListener(ev, ok); el.addEventListener('error', bad);
  });
}
async function loadVideo(video, url) {
  video.muted = true; video.playsInline = true; video.setAttribute('playsinline', ''); video.preload = 'auto';
  if (video.src !== url) { video.src = url; video.load(); }
  if (video.readyState < 1) { const ok = await waitEv(video, 'loadedmetadata', 12000); if (!ok) throw new PoseError('codec', 'metadata timeout'); }
  if (!Number.isFinite(video.duration) || video.duration <= 0) {           // MediaRecorder WebM: duration = Infinity until scanned
    video.currentTime = 1e7; await waitEv(video, 'durationchange', 4000); await waitEv(video, 'seeked', 2000).catch(() => {});
    video.currentTime = 0; await waitEv(video, 'seeked', 3000).catch(() => {});
  }
  if (video.readyState < 2) {
    const ok = await waitEv(video, 'loadeddata', 5000);
    if (!ok) { try { await video.play(); video.pause(); } catch (e) {} if (video.readyState < 2) await waitEv(video, 'loadeddata', 3000); }
  }
  if (!video.videoWidth || !video.videoHeight) throw new PoseError('codec', 'no video track decoded');
  return { duration: video.duration, width: video.videoWidth, height: video.videoHeight };
}
function estimateFps(video, ms = 1400) {
  return new Promise(res => {
    if (!('requestVideoFrameCallback' in HTMLVideoElement.prototype)) return res(null);
    const ts = []; const t0 = performance.now(); let h;
    const cb = (now, md) => { ts.push(md.mediaTime); if (performance.now() - t0 < ms && ts.length < 60) h = video.requestVideoFrameCallback(cb); else finish(); };
    const finish = () => {
      const d = []; for (let i = 1; i < ts.length; i++) { const x = ts[i] - ts[i - 1]; if (x > 0.002 && x < 0.2) d.push(x); }
      if (d.length < 4) return res(null);
      d.sort((a, b) => a - b); const f = 1 / d[Math.floor(d.length * 0.25)];
      const std = [24, 25, 30, 48, 50, 60, 90, 120, 240].find(s => Math.abs(s - f) / s < 0.09);
      res(std || Math.round(f));
    };
    h = video.requestVideoFrameCallback(cb);
    setTimeout(() => { if (ts.length < 60) { try { video.cancelVideoFrameCallback(h); } catch (e) {} finish(); } }, ms + 600);
  });
}

// One hidden, dedicated video for frame grabs (thumbnails, key frames). Grabs are serialized.
const grabber = { el: null, url: null, q: Promise.resolve() };
function grab(url, t, maxSide = 480) {
  const run = async () => {
    if (!grabber.el) {
      grabber.el = document.createElement('video'); grabber.el.muted = true; grabber.el.playsInline = true; grabber.el.setAttribute('playsinline', '');
      grabber.el.style.cssText = 'position:fixed;left:-10px;top:-10px;width:2px;height:2px;opacity:0;pointer-events:none';
      document.body.appendChild(grabber.el);
    }
    if (grabber.url !== url) { grabber.url = url; await loadVideo(grabber.el, url); }
    return grabFrame(grabber.el, t, maxSide);
  };
  const p = grabber.q.then(run, run); grabber.q = p.catch(() => {}); return p;
}

// ------------------------------------------------------------------ clip step
async function enterClip() {
  const v = $('#clip-video'), meta = $('#clip-meta'), btn = $('#btn-analyze');
  btn.disabled = true;
  setSeg('#opt-side', prefs.side); setSeg('#opt-model', prefs.model); modelSub();
  meta.innerHTML = '<span class="chip skel">…</span><span class="chip skel">…</span><span class="chip skel">…</span>';
  let vm;
  try { vm = state.vmeta && v.src === state.url ? state.vmeta : await loadVideo(v, state.url); }
  catch (e) { return showError(errCodec(e)); }
  state.vmeta = vm;
  $('#clip-stage').style.setProperty('--ar', vm.width + '/' + vm.height);
  const dur = vm.duration;
  if (!state.trim || state.trim[1] <= 0 || state.trim[1] > dur || state._trimFor !== state.url) { state.trim = [0, Math.min(dur, dur > MAX_WINDOW ? 8 : dur)]; state._trimFor = state.url; }
  const chips = [`<span class="chip">${fmtS(dur, 1)}</span>`, `<span class="chip">${vm.width}×${vm.height}</span>`, '<span class="chip" id="chip-fps">fps …</span>'];
  if (state.file && state.file.size) chips.push(`<span class="chip">${fmtBytes(state.file.size)}</span>`);
  chips.push(`<span class="chip">${state.source === 'record' ? 'recorded' : 'uploaded'} · stays on device</span>`);
  meta.innerHTML = chips.join('');
  if (dur > 60) toast('Long clip — drag the handles around one shot');
  setupTrim(v, dur);
  btn.disabled = false;
  // fps estimate while previewing the selection
  v.currentTime = state.trim[0];
  v.play().then(async () => {
    if (!state.fpsEst) state.fpsEst = await estimateFps(v);
    const c = $('#chip-fps'); if (c) c.textContent = state.fpsEst ? '≈ ' + state.fpsEst + ' fps' : 'fps unknown';
    if (state.fpsEst && state.fpsEst < 24) toast('Low frame rate — 30 or 60 fps gives better timing');
  }).catch(() => { const c = $('#chip-fps'); if (c) c.textContent = 'fps unknown'; });
  filmstrip(state.url, dur);
  onLeave(() => { v.pause(); });
}
function setSeg(sel, val) { $$(sel + ' button').forEach(b => { const on = b.dataset.v === val; b.classList.toggle('on', on); b.setAttribute('aria-checked', on); }); }
function modelSub() { $('#model-sub').textContent = prefs.model === 'full' ? `Accurate · ${(MODEL_SIZES.full / 1e6).toFixed(1)} MB download, ~2× slower` : `Fast · ${(MODEL_SIZES.lite / 1e6).toFixed(1)} MB download`; }
$('#opt-side').addEventListener('click', e => { const b = e.target.closest('button'); if (!b) return; prefs.side = b.dataset.v; savePrefs(); setSeg('#opt-side', prefs.side); });
$('#opt-model').addEventListener('click', e => { const b = e.target.closest('button'); if (!b) return; prefs.model = b.dataset.v; savePrefs(); setSeg('#opt-model', prefs.model); modelSub(); });

function setupTrim(v, dur) {
  const a = $('#trim-a'), b = $('#trim-b'), sel = $('#trim-sel'), K = 1000;
  const toT = x => x / K * dur, toX = t => Math.round(t / dur * K);
  a.value = toX(state.trim[0]); b.value = toX(state.trim[1]);
  const minLen = Math.min(0.5, dur);
  const paint = () => {
    const [s, e] = state.trim;
    sel.style.left = (s / dur * 100) + '%'; sel.style.width = ((e - s) / dur * 100) + '%';
    $('#trim-a-l').textContent = fmtS(s); $('#trim-b-l').textContent = fmtS(e);
    const len = e - s; $('#trim-len').textContent = len.toFixed(1) + ' s selected';
    $('#trim-len').className = 'num ' + (len > 8 ? 'warn' : 'muted');
    $('#trim-note').textContent = dur > MAX_WINDOW ? `Clip is ${dur.toFixed(0)} s — up to ${MAX_WINDOW} s around your shot is analysed.` : len > 8 ? 'Long selection: fine for several shots, slower to process.' : 'Tight trims around one shot are faster and more accurate.';
  };
  const onA = () => {
    let s = toT(+a.value), e = state.trim[1];
    if (s > e - minLen) { s = e - minLen; }
    if (e - s > MAX_WINDOW) { e = s + MAX_WINDOW; b.value = toX(e); }
    state.trim = [Math.max(0, s), e]; a.value = toX(state.trim[0]); paint(); v.pause(); v.currentTime = state.trim[0]; playBtn(false);
  };
  const onB = () => {
    let e = toT(+b.value), s = state.trim[0];
    if (e < s + minLen) { e = s + minLen; }
    if (e - s > MAX_WINDOW) { s = e - MAX_WINDOW; a.value = toX(s); }
    state.trim = [s, Math.min(dur, e)]; b.value = toX(state.trim[1]); paint(); v.pause(); v.currentTime = Math.max(0, state.trim[1] - 0.02); playBtn(false);
  };
  a.oninput = onA; b.oninput = onB;
  // the two thumbs overlap at the edges: whichever is nearer the pointer wins
  $('#trim').onpointerdown = e => {
    const r = $('#trim').getBoundingClientRect(); const x = (e.clientX - r.left) / r.width * dur;
    const da = Math.abs(x - state.trim[0]), db = Math.abs(x - state.trim[1]);
    a.style.zIndex = da < db ? 3 : 2; b.style.zIndex = da < db ? 2 : 3;
  };
  const pb = $('#trim-play');
  const playBtn = on => { pb.querySelector('span').textContent = on ? 'Pause' : 'Play selection'; pb.querySelector('svg').innerHTML = on ? '<path d="M8 5v14M16 5v14"/>' : '<path d="M8 5v14l11-7z"/>'; };
  pb.onclick = () => { if (v.paused) { if (v.currentTime < state.trim[0] || v.currentTime >= state.trim[1] - 0.05) v.currentTime = state.trim[0]; v.play().catch(() => {}); } else v.pause(); };
  v.onplay = () => playBtn(true); v.onpause = () => playBtn(false);
  v.ontimeupdate = () => { $('#clip-cur').textContent = v.currentTime.toFixed(2) + ' s'; if (!v.paused && v.currentTime >= state.trim[1]) { v.currentTime = state.trim[0]; } };
  v.loop = false; v.onended = () => { v.currentTime = state.trim[0]; v.play().catch(() => {}); };
  paint();
}
async function filmstrip(url, dur) {
  const strip = $('#trim-strip'); const n = innerWidth < 500 ? 6 : 9;
  strip.innerHTML = '<i></i>'.repeat(n);
  for (let k = 0; k < n; k++) {
    if (state.url !== url || current !== 'clip') return;
    try {
      const c = await grab(url, (k + 0.5) / n * dur, 120);
      const img = new Image(); img.alt = ''; img.src = c.toDataURL('image/jpeg', 0.6);
      const slot = strip.children[k]; if (slot) slot.replaceWith(img);
    } catch (e) { return; }
  }
}
$('#btn-analyze').addEventListener('click', () => {
  // iOS: start (and immediately pause) the processing video inside the user gesture.
  const pv = $('#proc-video'); pv.muted = true; pv.playsInline = true;
  if (pv.src !== state.url) { pv.src = state.url; pv.load(); }
  const p = pv.play(); if (p && p.then) p.then(() => pv.pause()).catch(() => {});
  runAnalysis();
});

// ------------------------------------------------------------------ processing
function stepUI(name, status, sub) {
  const li = $(`#proc-steps [data-step="${name}"]`); if (!li) return;
  if (status) { li.classList.remove('run', 'done'); if (status !== 'idle') li.classList.add(status); }
  if (sub != null) $('#st-' + name).textContent = sub;
}
function setBar(frac, label, eta) {
  const b = $('#proc-bar'); b.querySelector('i').style.width = (clamp(frac, 0, 1) * 100).toFixed(1) + '%'; b.setAttribute('aria-valuenow', Math.round(frac * 100));
  if (label != null) $('#proc-count').textContent = label;
  $('#proc-eta').textContent = eta || '';
}
function resetProcUI(title) {
  $('#h-proc').textContent = title || 'Reading your shot…';
  ['model', 'init', 'track', 'analyze'].forEach(s => stepUI(s, 'idle'));
  stepUI('model', null, `one time · ≈${((MODEL_SIZES.wasm + MODEL_SIZES[prefs.model]) / 1e6).toFixed(0)} MB, cached for offline use`);
  stepUI('init', null, 'GPU if available'); stepUI('track', null, '33 landmarks per frame'); stepUI('analyze', null, 'mechanics · timing · drills');
  setBar(0, '—', ''); $('#proc-delegate').hidden = true; $('#proc-idle').hidden = false; $('#proc-fps').textContent = '';
  const c = $('#proc-canvas'); c.getContext('2d').clearRect(0, 0, c.width, c.height);
}
function beginBusy() {
  state.busy = true; state.abort = new AbortController(); keepAwake(true);
  return state.abort.signal;
}
function endBusy() { state.busy = false; keepAwake(false); }
$('#btn-cancel').addEventListener('click', () => { if (state.abort) state.abort.abort(); });

async function runAnalysis() {
  const signal = beginBusy();
  resetProcUI();
  const vm = state.vmeta;
  $('#proc-stage').style.setProperty('--ar', vm.width + '/' + vm.height);
  go('#/processing');
  const pv = $('#proc-video');
  const [start, end] = state.trim;
  try {
    const assetsJob = getAssets();
    // 1+2: model
    stepUI('model', 'run');
    const t0 = performance.now();
    const lmk = await ensureLandmarker({
      model: prefs.model, signal,
      onProgress: p => {
        if (p.phase === 'download') { setBar(p.loaded / (p.total || 1), `Downloading AI model… ${fmtBytes(p.loaded)} / ${fmtBytes(p.total)}`, ''); stepUI('model', null, `Downloading AI model (≈${Math.round(p.total / 1e6)} MB, one time)… ${Math.round(p.loaded / (p.total || 1) * 100)}%`); }
        if (p.phase === 'init') { stepUI('model', 'done', 'downloaded · cached'); stepUI('init', 'run'); setBar(1, 'Starting the pose model…', ''); }
      },
    });
    stepUI('model', 'done', lmk.cached ? 'already loaded' : 'downloaded · cached for offline use');
    try { const ms = new Set(safeJSONArr(localStorage.getItem('jumpshot.models'))); ms.add(prefs.model); localStorage.setItem('jumpshot.models', JSON.stringify([...ms])); } catch (e) {}
    stepUI('init', 'done', lmk.delegate === 'GPU' ? 'running on GPU' : 'running on CPU');
    const dEl = $('#proc-delegate'); dEl.hidden = false; dEl.textContent = lmk.delegate; dEl.className = 'chip' + (lmk.delegate === 'GPU' ? ' pos' : '');
    if (lmk.fellBack) toast('GPU unavailable — running on CPU (slower, same result)', 3600);
    if (signal.aborted) throw new PoseError('aborted', 'Cancelled');
    // 3: track
    stepUI('track', 'run');
    await loadVideo(pv, state.url).catch(e => { throw new PoseError('codec', e.message); });
    const pc = $('#proc-canvas'), pctx = pc.getContext('2d');
    const stage = $('#proc-stage');
    const sizeCanvas = () => { const d = Math.min(2, devicePixelRatio || 1); pc.width = Math.round(stage.clientWidth * d); pc.height = Math.round(stage.clientHeight * d); };
    sizeCanvas();
    let lastDraw = 0;
    const out = await extractPose(pv, {
      start, end, landmarker: lmk.landmarker, model: prefs.model, nativeFps: state.fpsEst || 30, signal,
      onFrame: f => {
        const now = performance.now();
        const eta = f.eta != null ? (f.eta < 1.5 ? 'almost done' : '~' + Math.ceil(f.eta) + ' s left') : 'estimating…';
        setBar(f.frac, `${f.done} / ~${f.expected} frames`, eta);
        if (now - lastDraw > 40) {        // throttle preview painting
          lastDraw = now; $('#proc-idle').hidden = true;
          if (pc.width !== Math.round(stage.clientWidth * Math.min(2, devicePixelRatio || 1))) sizeCanvas();
          const r = containRect(pc.width, pc.height, f.canvas.width, f.canvas.height);
          pctx.fillStyle = '#05070a'; pctx.fillRect(0, 0, pc.width, pc.height);
          pctx.drawImage(f.canvas, r.x, r.y, r.w, r.h);
          if (f.lm) drawPose(pctx, f.lm, (x, y) => [r.x + x * r.w, r.y + y * r.h], { col: COL.accent, w: Math.max(2.5, r.h / 160), joints: true, arm: prefs.side === 'left' ? 'left' : prefs.side === 'right' ? 'right' : null });
          $('#proc-fps').textContent = `${f.t.toFixed(2)} s · ${Math.round(f.detectMs)} ms/frame`;
        }
      },
    });
    const { frames, meta, stats } = out;
    console.info('[jumpshot] pose', stats, meta);
    if (stats.delegateSwitched) toast('GPU failed mid-run — finished on CPU');
    const nPose = frames.filter(f => f.lm).length;
    stepUI('track', 'done', `${nPose} of ${frames.length} frames with a person · ${meta.fps} fps · ${stats.path === 'rvfc' ? 'frame-exact' : 'seek'} · ${((performance.now() - t0) / 1000).toFixed(1)} s`);
    state.frames = frames; state.pmeta = meta; state.trimStart = start; state.lastStats = stats; state.lastDelegate = lmk.delegate;
    // quality gates
    if (nPose < Math.max(5, frames.length * 0.15)) throw Object.assign(new Error('no person'), { code: 'no_person', nPose, n: frames.length });
    const h = bodyHeight(frames);
    if (h < 0.2 && !state._ignoreSmall) throw Object.assign(new Error('small'), { code: 'small', h });
    // 4: analyze
    stepUI('analyze', 'run');
    setBar(1, 'Scoring…', '');
    const [eng] = await Promise.all([getEngine(), assetsJob]);
    const shots = safeShots(eng, frames, meta);
    endBusy();
    state.shots = shots;
    if (shots.length === 1) return analyzeWindows([shots[0]]);
    stepUI('analyze', 'done');
    buzz(15);
    go('#/shots', true);
  } catch (e) {
    endBusy();
    try { pv.pause(); } catch (x) {}
    if (e && e.code === 'aborted') { if (current === 'processing') go('#/clip', true); return; }
    (e && ['no_person', 'small', 'codec'].includes(e.code) ? console.info : console.warn)('[jumpshot] analysis stopped:', e && (e.code || e.message));
    if (e && e.code === 'no_person') return showError(errNoPerson(e));
    if (e && e.code === 'small') return showError(errSmall(e));
    if (e && e.code === 'codec') return showError(errCodec(e));
    if (e && (e.code === 'model' || e.code === 'simd')) return showError(errModel(e));
    showError(errUnknown(e));
  }
}
function safeShots(eng, frames, meta) {
  try { const s = eng.detectShots(frames, meta); return Array.isArray(s) ? s.filter(w => w && fin(w.start) && fin(w.end)) : []; }
  catch (e) { console.warn('detectShots failed', e); return []; }
}
function bodyHeight(frames) {
  const hs = [];
  for (const f of frames) {
    if (!f.lm) continue; let lo = 1, hi = 0, n = 0;
    for (let i = 0; i < 33; i++) { const p = P(f.lm, i); if (p && p[2] > 0.4) { lo = Math.min(lo, p[1]); hi = Math.max(hi, p[1]); n++; } }
    if (n > 8) hs.push(hi - lo);
  }
  hs.sort((a, b) => a - b); return hs.length ? hs[hs.length >> 1] : 0;
}

async function analyzeWindows(windows, opts = {}) {
  const eng = await getEngine(); const assets = await getAssets();
  const frames = state.frames, meta = state.pmeta;
  const results = [];
  let lastBad = null;
  for (const w of windows) {
    let r;
    try { r = eng.analyze(frames, meta, assets, { side: prefs.side, window: w ? { start: w.start | 0, end: w.end | 0 } : null }); }
    catch (e) { console.error(e); r = { ok: false, reason: e.message }; }
    if (r && r.ok) { r._window = w ? { start: w.start | 0, end: w.end | 0, releaseGuess: w.releaseGuess } : null; results.push(r); } else lastBad = r;
  }
  if (!results.length) return showError(errAnalysis(lastBad));
  if (windows.length > 1 && results.length < windows.length) setTimeout(() => toast(`${windows.length - results.length} of ${windows.length} shots couldn't be measured and were skipped`, 4000), 600);
  const sample = state.source === 'sample';
  const best = results.length > 1 ? 0 : 0;
  const s = {
    id: sample ? 'sample' : Date.now().toString(36) + Math.random().toString(36).slice(2, 6),
    created: new Date().toISOString(),
    name: sample ? 'Sample shot' : (state.file && state.file.name ? state.file.name.replace(/\.[^.]+$/, '') : 'Shot'),
    source: state.source, model: sample ? null : prefs.model, delegate: state.lastDelegate || null, side: prefs.side,
    meta, frames, trimStart: state.trimStart || 0,
    results, primary: best,
    score: results.length > 1 ? mean(results.map(r => r.score)) : results[0].score,
    mechScore: null, styleMatch: null, scoreKind: results[best].scoreKind || null,
    closest: results[best].closest && results[best].closest.name,
    thumb: null,
  };
  { const sc = sessScores(s); s.mechScore = fin(sc.mech) ? sc.mech : null; s.styleMatch = fin(sc.style) ? sc.style : null; }
  state.session = s;
  state.sessionVideo = sample ? null : { id: s.id, url: state.url };
  // thumbnail at release with skeleton
  try { s.thumb = await keyframeImage(s, results[best], 'release', 160, 0.72); } catch (e) { console.warn('thumb', e); }
  if (!sample) { try { await DB.put(s); refreshCount(); } catch (e) { console.warn('save failed', e); toast('Could not save to history (storage full?)'); } }
  buzz([20, 40, 20]);
  go('#/results', true);
}

// ------------------------------------------------------------------ sample
async function runSample() {
  const signal = beginBusy();
  state.source = 'sample';
  resetProcUI('Loading the sample shot…');
  $('#proc-stage').style.setProperty('--ar', '3/4');
  go('#/processing');
  stepUI('model', 'done', 'not needed — sample landmarks are pre-computed');
  stepUI('init', 'done', 'skipped');
  stepUI('track', 'run', 'loading demo landmarks');
  try {
    const [eng, assets, demo] = await Promise.all([getEngine(), getAssets(), fetch(new URL('demo.json', APP_BASE)).then(r => { if (!r.ok) throw new Error('demo.json ' + r.status); return r.json(); })]);
    if (signal.aborted) throw new PoseError('aborted');
    const { frames, meta } = parseDemo(demo);
    // a short synthetic preview of the landmarks, so the step isn't a blank wait
    const pc = $('#proc-canvas'), stage = $('#proc-stage'); const d = Math.min(2, devicePixelRatio || 1);
    pc.width = Math.round(stage.clientWidth * d); pc.height = Math.round(stage.clientHeight * d); $('#proc-idle').hidden = true;
    const ctx = pc.getContext('2d'); const r = containRect(pc.width, pc.height, meta.width, meta.height);
    const steps = RM.matches ? 1 : Math.min(frames.length, 45);
    for (let k = 0; k < steps; k++) {
      if (signal.aborted) throw new PoseError('aborted');
      const i = Math.round(k / Math.max(1, steps - 1) * (frames.length - 1));
      ctx.fillStyle = '#0d1016'; ctx.fillRect(0, 0, pc.width, pc.height); floorGrid(ctx, pc.width, pc.height, r);
      if (frames[i].lm) drawPose(ctx, frames[i].lm, (x, y) => [r.x + x * r.w, r.y + y * r.h], { col: COL.accent, w: Math.max(3, r.h / 150), joints: true });
      setBar((k + 1) / steps, `${i + 1} / ${frames.length} frames`, 'synthetic preview');
      await new Promise(res => setTimeout(res, 16));
    }
    stepUI('track', 'done', `${frames.length} frames · ${meta.fps} fps · synthetic`);
    stepUI('analyze', 'run');
    state.frames = frames; state.pmeta = meta; state.trimStart = 0; state.lastDelegate = null;
    const shots = safeShots(eng, frames, meta);
    endBusy();
    state.shots = shots;
    await analyzeWindows(shots.length ? [shots[0]] : [null]);
  } catch (e) {
    endBusy();
    if (e && e.code === 'aborted') { go('#/', true); return; }
    console.error(e);
    showError({ title: 'The sample could not load', msg: 'The demo landmarks or the analysis engine failed to load.', fixes: ['Check your connection and try again (first load needs the app files).'], actions: [{ label: 'Try again', primary: true, click: runSample }, { label: 'Home', go: '#/' }], code: String(e && e.message) });
  }
}
function parseDemo(d) {
  const src = d.clip || d.user || d;
  const m = src.meta || d.meta || {};
  const meta = { fps: num(m.fps || src.fps || 30), width: num(m.width || src.width || 480), height: num(m.height || src.height || 640), duration: 0 };
  let raw = src.frames || src.landmarks || d.frames;
  const vis = src.visibility || null;
  const frames = raw.map((f, i) => {
    let lm = f && f.lm !== undefined ? f.lm : f;
    const t = f && f.t != null ? +f.t : i / meta.fps;
    if (!lm) return { t, lm: null };
    const a = new Float32Array(132);
    if (typeof lm[0] === 'object') { for (let j = 0; j < 33; j++) { const p = lm[j] || []; a[j * 4] = num(p[0]); a[j * 4 + 1] = num(p[1]); a[j * 4 + 2] = num(p[2]) || 0; a[j * 4 + 3] = p.length > 3 ? num(p[3]) : vis ? num(vis[i][j]) : 1; } }
    else { const k = Math.round(lm.length / 33); for (let j = 0; j < 33; j++) { a[j * 4] = num(lm[j * k]); a[j * 4 + 1] = num(lm[j * k + 1]); a[j * 4 + 2] = k > 2 ? num(lm[j * k + 2]) || 0 : 0; a[j * 4 + 3] = k > 3 ? num(lm[j * k + 3]) : vis ? num(vis[i][j]) : 1; } }
    for (let q = 0; q < 132; q++) if (!Number.isFinite(a[q])) a[q] = q % 4 === 3 ? 0 : NaN;
    return { t, lm: a };
  });
  meta.duration = frames.length / meta.fps;
  return { frames, meta };
}

// ------------------------------------------------------------------ shot picker
async function renderShots() {
  const box = $('#shots-body'); const shots = state.shots || [];
  const frames = state.frames;
  if (!shots.length) {
    $('#h-shots').textContent = 'No shot found';
    box.innerHTML = `<div class="card errcard" style="margin:0;max-width:none">
      <div class="eic warn"><svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M12 8v5M12 16h.01"/></svg></div>
      <h2>We tracked you, but couldn't spot a jump shot</h2>
      <p>The person was found in ${frames.filter(f => f.lm).length} of ${frames.length} frames, but no dip → lift → release pattern stood out.</p>
      <ul><li>Film from the <b>side</b> so the shooting arm's extension is visible.</li><li>Keep the <b>whole body and the ball</b> in frame through the follow-through.</li><li>Trim to start just <b>before the dip</b> and end <b>after the release</b>.</li><li>Use 30–60 fps and good light — motion blur hides the wrist.</li></ul>
      <div class="acts"><button class="btn primary" id="btn-whole" type="button">Analyze the whole clip anyway</button><a class="btn" href="#/clip">Adjust trim</a></div></div>`;
    $('#btn-whole').onclick = () => analyzeWindows([null]);
    return;
  }
  $('#h-shots').textContent = `${shots.length} shots found`;
  box.innerHTML = `<p class="lede">Tap a shot to analyse it, or analyse all of them to see how consistent your form is from shot to shot.</p>
    <div class="shots-row" id="shots-row">${shots.map((s, i) => `<button class="shot" type="button" data-i="${i}"><span class="th skel-box" style="display:block;border-radius:0"></span>
      <span class="cap"><b>Shot ${i + 1}</b><span class="num muted">${fmtS(frames[clamp(s.releaseGuess | 0, 0, frames.length - 1)].t, 1)}${fin(s.confidence) ? ' · ' + Math.round(s.confidence * 100) + '%' : ''}</span></span></button>`).join('')}</div>
    <div class="cta-bar" style="position:static"><button class="btn primary big" id="btn-all" type="button">Analyze all ${shots.length} shots · consistency</button></div>`;
  $('#btn-all').onclick = () => analyzeWindows(shots);
  $$('#shots-row .shot').forEach(b => b.onclick = () => analyzeWindows([shots[+b.dataset.i]]));
  // thumbnails at releaseGuess, skeleton burned in
  for (let i = 0; i < shots.length; i++) {
    const s = shots[i], f = clamp(s.releaseGuess | 0, 0, frames.length - 1);
    try {
      const url = state.source === 'sample' ? null : state.url;
      const src = await poseImage(url, state.trimStart + frames[f].t, frames[f].lm, state.pmeta, 320, null);
      const slot = $(`#shots-row .shot[data-i="${i}"] .th`); if (!slot) return;
      const img = new Image(); img.className = 'th'; img.alt = 'Shot ' + (i + 1) + ' at release'; img.src = src; slot.replaceWith(img);
    } catch (e) { console.warn(e); }
  }
}

// ------------------------------------------------------------------ drawing
const BONES = [[11, 12], [11, 23], [12, 24], [23, 24], [11, 13], [13, 15], [15, 17], [15, 19], [15, 21], [17, 19], [12, 14], [14, 16], [16, 18], [16, 20], [16, 22], [18, 20],
  [23, 25], [25, 27], [27, 29], [27, 31], [29, 31], [24, 26], [26, 28], [28, 30], [28, 32], [30, 32]];
const ARM = { right: new Set(['12-14', '14-16', '16-18', '16-20', '16-22', '18-20']), left: new Set(['11-13', '13-15', '15-17', '15-19', '15-21', '17-19']) };
function containRect(W, H, w, h) { const s = Math.min(W / w, H / h); const rw = w * s, rh = h * s; return { x: (W - rw) / 2, y: (H - rh) / 2, w: rw, h: rh }; }
function drawPose(ctx, lm, map, st) {
  const pts = []; const minVis = st.minVis != null ? st.minVis : 0.3;
  for (let i = 0; i < 33; i++) { const p = P(lm, i); pts.push(p && p[2] >= minVis ? map(p[0], p[1]) : null); }
  const arm = st.arm ? ARM[st.arm] : null;
  ctx.save(); ctx.lineCap = 'round'; ctx.lineJoin = 'round'; ctx.setLineDash(st.dash || []);
  if (st.halo !== false) {               // dark halo keeps the skeleton legible over bright video
    ctx.strokeStyle = 'rgba(0,0,0,.55)'; ctx.globalAlpha = (st.alpha || 1) * 0.8;
    for (const [a, b] of BONES) { if (!pts[a] || !pts[b]) continue; ctx.lineWidth = st.w + 3 + (arm && arm.has(a + '-' + b) ? 1.5 : 0); ctx.beginPath(); ctx.moveTo(pts[a][0], pts[a][1]); ctx.lineTo(pts[b][0], pts[b][1]); ctx.stroke(); }
  }
  ctx.globalAlpha = st.alpha || 1; ctx.strokeStyle = st.col;
  for (const [a, b] of BONES) {
    if (!pts[a] || !pts[b]) continue;
    ctx.lineWidth = st.w + (arm && arm.has(a + '-' + b) ? 1.5 : 0);
    ctx.beginPath(); ctx.moveTo(pts[a][0], pts[a][1]); ctx.lineTo(pts[b][0], pts[b][1]); ctx.stroke();
  }
  ctx.setLineDash([]);
  // head: circle around ears/nose
  const n = pts[0], e1 = pts[7], e2 = pts[8];
  if (n) { const ex = e1 && e2 ? (e1[0] + e2[0]) / 2 : n[0], ey = e1 && e2 ? (e1[1] + e2[1]) / 2 : n[1]; const r = Math.max(st.w * 2, Math.hypot(n[0] - ex, n[1] - ey) + st.w);
    ctx.lineWidth = st.w; ctx.setLineDash(st.dash || []); ctx.beginPath(); ctx.arc(ex, ey, r, 0, 7); ctx.stroke(); ctx.setLineDash([]); }
  if (st.joints) for (let i = 11; i < 33; i++) { const p = pts[i]; if (!p || (i > 16 && i < 23)) continue; ctx.fillStyle = st.col; ctx.beginPath(); ctx.arc(p[0], p[1], st.w * 0.95, 0, 7); ctx.fill(); ctx.fillStyle = COL.ink; ctx.beginPath(); ctx.arc(p[0], p[1], st.w * 0.4, 0, 7); ctx.fill(); }
  ctx.restore();
}
function floorGrid(ctx, W, H, r) {
  ctx.save(); ctx.strokeStyle = COL.line; ctx.lineWidth = 1; ctx.globalAlpha = 0.6; ctx.setLineDash([2, 6]);
  for (let k = 1; k < 6; k++) { const y = r.y + r.h * k / 6; ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke(); }
  ctx.restore();
}
/** Image (data URL) of frame t from the video (or a dark stage when url is null) with the skeleton burned in. */
async function poseImage(url, t, lm, meta, maxSide, ghostLm, quality = 0.82, box = null) {
  let c, ctx;
  if (url) { c = await grab(url, t, maxSide); ctx = c.getContext('2d'); ctx.fillStyle = 'rgba(5,7,10,.25)'; ctx.fillRect(0, 0, c.width, c.height); }
  else { const s = maxSide / Math.max(meta.width, meta.height); c = document.createElement('canvas'); c.width = Math.round(meta.width * s); c.height = Math.round(meta.height * s); ctx = c.getContext('2d'); ctx.fillStyle = '#0d1016'; ctx.fillRect(0, 0, c.width, c.height); floorGrid(ctx, c.width, c.height, { x: 0, y: 0, w: c.width, h: c.height }); }
  let map = (x, y) => [x * c.width, y * c.height];
  if (!url && box) { const z = Math.min(1 / box.w, 1 / box.h) * 0.92; const cx = box.x + box.w / 2, cy = box.y + box.h / 2; map = (x, y) => [c.width / 2 + (x - cx) * z * c.width, c.height / 2 + (y - cy) * z * c.height]; }
  const w = Math.max(2, c.height / 110);
  if (ghostLm) drawPose(ctx, ghostLm, map, { col: '#c9d0de', w: w * 0.8, alpha: 0.75, dash: [w * 2, w * 1.6], halo: false });
  if (lm) drawPose(ctx, lm, map, { col: COL.accent, w, joints: true });
  return c.toDataURL('image/jpeg', quality);
}

// ------------------------------------------------------------------ ghost + draw landmarks
// The engine already returns closest.ghost in the user's normalised image coords (time-warped to
// onset..ftEnd, scaled to the user's torso, anchored at their stance, turned to their facing/hand).
function ghostMapper(session, r) {
  const g = r.closest && r.closest.ghost; if (!g || !g.length) return null;
  const on = fin(r.closest.start) ? r.closest.start | 0 : (r.events && r.events.onset) | 0;
  return { on, ft: on + g.length - 1, at(i) { return g[clamp(i - on, 0, g.length - 1)]; }, map(x, y) { return [x, y]; } };
}
/** Normalised bounding box of every drawn point (user + ghost), padded, for skeleton-only views. */
function bodyBox(session, r) {
  let x0 = 1, y0 = 1, x1 = 0, y1 = 0; const gm = ghostMapper(session, r);
  const add = lm => { if (!lm) return; for (let j = 0; j < 33; j++) { const p = P(lm, j); if (p && p[2] >= 0.3) { x0 = Math.min(x0, p[0]); x1 = Math.max(x1, p[0]); y0 = Math.min(y0, p[1]); y1 = Math.max(y1, p[1]); } } };
  for (let i = 0; i < session.frames.length; i++) { add(lmAt(session, r, i)); if (gm) add(gm.at(i)); }
  if (x1 <= x0 || y1 <= y0) return { x: 0, y: 0, w: 1, h: 1 };
  const px = (x1 - x0) * 0.18 + 0.02, py = (y1 - y0) * 0.1 + 0.02;
  x0 = Math.max(0, x0 - px); x1 = Math.min(1, x1 + px); y0 = Math.max(0, y0 - py); y1 = Math.min(1, y1 + py);
  return { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
}
/** Draw-ready landmarks for frame i: the engine's gap-filled set inside the window, else the raw frame. */
function lmAt(session, r, i) { const L = r && r.landmarks; return (L && L[i]) || (session.frames[i] && session.frames[i].lm) || null; }

// ------------------------------------------------------------------ results
let scrub = null;
async function renderResults(session, idx) {
  const box = $('#results-body');
  const results = session.results || [session.result];
  idx = clamp(idx | 0, 0, results.length - 1);
  const r = results[idx];
  const eng = await getEngine(); let assets = null; try { assets = await getAssets(); } catch (e) {}
  const FEATURES = eng.FEATURES || [];
  const hasVideo = !!(state.sessionVideo && state.sessionVideo.id === session.id && state.sessionVideo.url);
  const vurl = hasVideo ? state.sessionVideo.url : null;
  const cmp = r.comparisons || [];
  const nm = splitName(r.closest && r.closest.name || (cmp[0] && cmp[0].name));
  const fbs = (r.feedback || []).slice();
  const tolOf = f => { const ft = FEATURES.find(x => x.key === f.key); return ft && ft.tolerance ? ft.tolerance : (f.unit === 'deg' ? 5 : f.unit === 's' ? 0.06 : 0.06); };
  // engine v2 already orders feedback mechanics-first (biggest point loss first); only legacy results get re-sorted
  if (!fbs.some(f => f.kind)) fbs.sort((a, b) => sevRank(b.severity) - sevRank(a.severity) || Math.abs(b.delta) / tolOf(b) - Math.abs(a.delta) / tolOf(a));
  const nonOk = fbs.filter(f => f.severity !== 'ok');
  const nOk = fbs.length - nonOk.length;
  const ms = mechScore(r), ss = styleScore(r), mk = hasMech(r) ? r.mechanics : [];
  const mc = st => mk.filter(m => m.status === st).length;
  const nMeas = mk.length - mc('unmeasured');
  const S = mechSources(assets);
  const plan = r.plan || [];
  const lostOf = key => { const f = fbs.find(x => x.kind === 'mechanics' && x.key === key); return f && fin(f.pointsLost) ? num(f.pointsLost) : NaN; };
  const q = r.quality || {}; const ql = q.level || 'ok';
  const load = metric(r, 'load'), relh = metric(r, 'relh'), fL = fbOf(r, 'load'), fH = fbOf(r, 'relh');
  const fps = session.meta.fps;
  const sample = session.source === 'sample';
  const dchip = (f, unit) => f && fin(f.delta) ? `<span class="d ${f.severity === 'ok' ? 'pos' : f.severity === 'major' ? 'neg' : 'warn'}">${fmtVal(f.delta, unit, true)}</span> vs ${esc(nm.title)} (${fmtVal(fbRef(f), unit)})` : '';

  box.innerHTML = `
    <div class="res-head">
      <div><div class="eyebrow">${sample ? 'Sample · synthetic shot' : esc(fmtDate(session.created)) + ' · ' + esc(fmtTime(session.created))}</div>
        <h1 id="h-res">${esc(session.name || 'Your shot')}${results.length > 1 ? ` <span class="muted" style="font-weight:600">· shot ${idx + 1}/${results.length}</span>` : ''}</h1></div>
      <div class="res-actions">
        <button class="btn primary" id="btn-share" type="button"><svg viewBox="0 0 24 24"><path d="M12 15V3M7 8l5-5 5 5"/><path d="M5 12v7a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2v-7"/></svg><span>Share card</span></button>
        <button class="btn" id="btn-copy" type="button"><svg viewBox="0 0 24 24"><rect x="8" y="8" width="12" height="12" rx="2"/><path d="M16 8V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h2"/></svg><span class="hide-sm">Copy summary</span></button>
        <a class="btn" href="#/" ><svg viewBox="0 0 24 24"><path d="M12 5v14M5 12h14"/></svg><span class="hide-sm">New</span></a>
      </div>
    </div>
    ${results.length > 1 ? consistencyHTML(session, idx) : ''}
    <div class="grid g4">
      <div class="card kpi"><div class="label">Mechanics score</div><b class="hero" id="kpi-score" data-v="${fin(ms) ? ms.toFixed(1) : ''}">${f1(ms)}<small>/100</small></b>
        <div class="sub">${mk.length ? (fin(ms) ? `${mc('good')} of ${nMeas} checks in range` : 'too few checks measurable') : 'not scored · older session'}</div></div>
      <div class="card kpi"><div class="label">Style match</div><b id="kpi-style">${f1(ss)}<small>/100</small></b>
        <div class="sub">vs ${esc(nm.title)}${nm.style ? ' · ' + esc(nm.style) : ''}</div></div>
      <div class="card kpi"><div class="label">Load → release</div><b>${fin(load) ? load.toFixed(2) : '—'}<span class="u"> s</span></b><div class="sub">${dchip(fL, 's')}</div></div>
      <div class="card kpi"><div class="label">Release height</div><b>${fmtVal(relh, '', true).replace(' torso', '')}<span class="u"> torso</span></b><div class="sub">${dchip(fH, 'torso')}</div></div>
    </div>
    <div class="flagrow">
      <span class="chip ${ql === 'ok' ? 'pos' : ql === 'minor' ? 'warn' : 'neg'}">${ql === 'ok' ? 'confidence ok' : ql === 'minor' ? 'minor quality issues' : 'low confidence'}</span>
      <span class="chip">hand ${esc(r.hand || '?')}</span>
      <span class="chip">${session.frames.length} f · ${fps} fps</span>
      ${session.delegate ? `<span class="chip">${esc(session.delegate)} · ${session.model === 'full' ? 'accurate' : 'fast'} model</span>` : ''}
      ${sample ? '<span class="chip accent">sample</span>' : ''}
      ${(q.flags || []).map(f => `<span class="chip warn" title="${esc(f.text || f.message || '')}">${esc(String(f.code || 'flag').replace(/_/g, ' '))}</span>`).join('')}
    </div>
    ${(q.flags || []).length ? `<ul class="note" style="padding-left:18px;margin-top:8px">${q.flags.map(f => `<li>${esc(f.text || f.message || f.code)}</li>`).join('')}</ul>` : ''}
    <p class="read-line"><b>Read:</b> ${fin(ms) ? `Mechanics <b>${ms.toFixed(1)}/100</b>: ${mc('good')} of ${nMeas} measured checks inside the published range, ${mc('close')} close, ${mc('off')} off.` : mk.length ? 'Too few mechanics checks could be measured on this clip for a mechanics score.' : 'This session was analysed before mechanics checks existed.'} Your motion is most like <b>${esc(nm.title)}</b> (style match ${f1(ss)}) — style says how alike you move, not how sound the mechanics are.</p>

    ${mechSectionHTML(r, mk, S, assets)}

    <section class="sec"><div class="sec-h"><h2>Frame by frame</h2><span class="read">${hasVideo ? 'your video · skeleton overlay · reference ghost' : sample ? 'synthetic preview · reference ghost' : 'skeleton replay (video is never stored)'}</span></div>
      <div class="card scrub" id="scrub">
        <div class="vstage" id="sc-stage">${hasVideo ? '<video id="sc-video" playsinline muted preload="auto"></video>' : ''}<canvas id="sc-canvas"></canvas>
          <span class="tag u">You</span><span class="tag r" id="sc-gtag">${esc(nm.title)} ghost</span></div>
        <div class="panel">
          <div class="tline"><span class="t" id="sc-time">0.00 s</span><span class="f" id="sc-frame">f00</span><span class="chip accent" id="sc-phase">—</span></div>
          <div class="timeline"><div class="marks" id="sc-marks"></div><input type="range" id="sc-range" min="0" max="1" value="0" step="1" aria-label="Frame"></div>
          <div class="row">
            <button class="btn" id="sc-prev" type="button" aria-label="Previous frame"><svg viewBox="0 0 24 24"><path d="m15 6-6 6 6 6"/></svg></button>
            <button class="btn" id="sc-play" type="button" style="min-width:88px"><svg viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg><span>Play</span></button>
            <button class="btn" id="sc-next" type="button" aria-label="Next frame"><svg viewBox="0 0 24 24"><path d="m9 6 6 6-6 6"/></svg></button>
            <div class="seg" role="group" aria-label="Speed"><button type="button" data-speed="0.25">¼×</button><button type="button" class="on" data-speed="0.5">½×</button><button type="button" data-speed="1">1×</button></div>
            <button class="btn on" id="sc-ghost" type="button" aria-pressed="true">Ghost</button>
          </div>
          <div class="row"><span class="lbl">Jump</span><div class="seg" role="group" aria-label="Jump to event">
            <button type="button" data-jump="onset">Onset</button><button type="button" data-jump="set">Set</button><button type="button" data-jump="release" class="on">Release</button><button type="button" data-jump="ftEnd">Follow</button></div></div>
          <div class="ro-grid" id="sc-ro"></div>
          <div class="legend"><span><i style="background:var(--accent)"></i>you</span><span><i class="dash"></i>${esc(nm.title)} (time-warped)</span><span class="hide-touch">← → step · shift ×5 · space play</span></div>
        </div>
      </div>
      <p class="read-line" id="sc-read"></p></section>

    <section class="sec"><div class="sec-h"><h2>Findings</h2><span class="read">${mk.length ? 'mechanics first, then style vs ' + esc(nm.title) : 'vs ' + esc(nm.title)}</span></div>
      <div class="grid g2">
        <div class="card radar"><h3>Style profile <span class="muted" style="font-weight:500">vs ${esc(nm.title)}</span></h3>${radarSVG(r.radar, [{ k: 'mean', col: COL.faint, dash: '3 4', w: 1.25 }, { k: 'ref', col: COL.muted, fill: 'rgba(138,147,166,.10)', w: 1.5 }, { k: 'user', col: COL.accent, fill: 'rgba(255,160,48,.16)', w: 2, dots: true }])}
          <div class="legend" style="margin-top:6px"><span><i style="background:var(--accent)"></i>you</span><span><i style="background:var(--muted)"></i>${esc(nm.title)}</span><span><i class="dash" style="border-color:var(--faint)"></i>library mean</span></div>
          <p class="note">Axes are min–max normalised over the library and you; the rim is the highest value seen, not "best".</p></div>
        <div><ul class="fb">${fbList(nonOk, nm) || '<li class="ok"><span class="lb">Everything inside range and tolerance</span><span class="chip pos">ok</span></li>'}</ul>
          ${nOk ? `<details class="more fb-more"><summary><span>${nOk} in range / within tolerance</span></summary><div class="body"><ul class="fb">${fbList(fbs.filter(f => f.severity === 'ok'), nm)}</ul></div></details>` : ''}</div>
      </div>
      <p class="read-line"><b>Read:</b> ${(() => { const mf = nonOk.filter(f => f.kind === 'mechanics'), sf = nonOk.filter(f => f.kind !== 'mechanics');
        if (!nonOk.length) return 'Every measured check is inside its range and every style metric is within tolerance of the closest reference.';
        const a = mf.length ? `Mechanics gaps, biggest point loss first: ${mf.slice(0, 2).map(f => `${esc(f.label.toLowerCase())} (${MINUS}${fmtPts(f.pointsLost)} pts)`).join(' and ')}. ` : (mk.length ? 'All measured mechanics checks are in range. ' : '');
        const b = sf.length ? `${sf.length} style difference${sf.length === 1 ? '' : 's'} vs ${esc(nm.title)} — those compare you with one library shooter, not with a published norm.` : '';
        return a + b; })()} Red = major / off, amber = minor / close.</p></section>

    <section class="sec"><div class="sec-h"><h2>Coaching plan</h2><span class="read">${(r.plan || []).length} drill${(r.plan || []).length === 1 ? '' : 's'}</span></div>
      ${plan.length ? `<ol class="plan">${plan.map(p => planItem(p, lostOf)).join('')}</ol>
      <p class="read-line"><b>Read:</b> Start with <b>${esc(plan[0].drill)}</b> — it targets ${plan[0].kind === 'mechanics' ? `the biggest mechanics point loss${fin(lostOf(plan[0].key)) ? ` (${fmtPts(lostOf(plan[0].key))} pts)` : ''}` : 'the largest gap'}. Drills are generic form-shooting work, not professional coaching.</p>`
      : '<div class="card"><p class="muted" style="margin:0">No drills needed — keep reps consistent and re-film next week.</p></div>'}</section>

    <section class="sec"><div class="sec-h"><h2>Key frames</h2><span class="read">set · release · follow-through${hasVideo ? '' : ' (skeleton)'}</span></div>
      <div class="frames" id="kf">${['set', 'release', 'ftEnd'].map(k => `<figure class="frame"><div class="ph skel-box" style="border-radius:0"></div><figcaption class="cap"><b>${{ set: 'Set point', release: 'Release', ftEnd: 'Follow-through' }[k]}</b><span class="num">${r.events && fin(r.events[k]) ? fmtS(session.frames[clamp(r.events[k], 0, session.frames.length - 1)].t) : '—'}</span></figcaption></figure>`).join('')}</div></section>

    <section class="sec"><details class="more"><summary><span>Evidence <span class="muted" style="font-weight:500">· all references · curves · quality</span></span></summary><div class="body"><div class="stack">
      <div class="card"><h3>All references <span class="muted" style="font-weight:500">· style match</span></h3>
        <p class="note" style="margin:0 0 10px">Style match = ${assets && assets.weights ? `${assets.weights.dtw} × DTW + ${assets.weights.learned} × learned` : 'DTW + learned'} similarity to each reference${cmp[0] && fin(cmp[0].dtw) ? `. Closest, ${esc(nm.title)}: DTW ${Math.round(cmp[0].dtw)}${fin(cmp[0].learned) ? ' · learned ' + Math.round(cmp[0].learned) : ''} → ${f1(cmp[0].score)}` : ''}.</p><div class="refs">${cmp.map((c, i) => `<div class="tile ${i === 0 ? 'on' : ''}"><span class="nm">${esc(c.name)}</span><span class="sc">${Math.round(c.score)}</span>
        <div class="bar"><i style="width:${clamp(c.score, 0, 100)}%;background:${i === 0 ? 'var(--text)' : 'var(--faint)'}"></i></div>
        <span class="meta">${fin(c.dtw) ? 'DTW ' + Math.round(c.dtw) : ''}${fin(c.learned) ? ' · learned ' + Math.round(c.learned) : ''}</span></div>`).join('')}</div></div>
      ${curvesHTML(r, assets, FEATURES)}
      <div class="card"><h3>Pose quality</h3><p class="note" style="margin-top:0">Visibility ${fin(q.visibility) ? Math.round(q.visibility * 100) + '%' : '—'} · ${session.frames.filter(f => f.lm).length}/${session.frames.length} frames with a person · ${fps} fps${state.lastStats && hasVideo ? ` · ${state.lastStats.path === 'rvfc' ? 'frame-exact decode' : 'seek decode'} · ${Math.round(state.lastStats.detectMs)} ms/frame` : ''}. Events: onset f${r.events.onset}, set f${r.events.set}, release f${r.events.release}, end f${r.events.ftEnd}.</p></div>
    </div></div></details></section>
    ${!sample ? `<div class="row-between" style="margin-top:18px"><a class="btn" href="#/history">All sessions</a><button class="btn" id="btn-del" type="button" style="color:var(--neg)">Delete this session</button></div>` : '<p class="note" style="margin-top:16px">Sample runs are not saved to your history.</p>'}`;

  // wire up
  $('#btn-share').onclick = () => openShare(session, r);
  $('#btn-copy').onclick = () => copySummary(session, r);
  const del = $('#btn-del'); if (del) del.onclick = async () => { if (!confirm('Delete this session from this device?')) return; await DB.del(session.id); refreshCount(); toast('Session deleted'); go('#/history'); };
  $$('#cons-tbl tbody tr').forEach(tr => tr.onclick = () => { const i = +tr.dataset.i; if (i !== idx) { renderResults(session, i); } });
  countUp($('#kpi-score'), ms);
  $$('#results-body .src-ref').forEach(b => b.onclick = () => {
    const d = $('#how-scoring'), li = $('#src-' + b.dataset.src); if (!d || !li) return;
    d.open = true; li.scrollIntoView({ block: 'center', behavior: RM.matches ? 'auto' : 'smooth' });
    li.classList.add('hl'); setTimeout(() => li.classList.remove('hl'), 1800);
  });
  if (scrub) scrub.destroy();
  scrub = new Scrubber(session, r, FEATURES, assets, vurl);
  onLeave(() => { if (scrub) { scrub.destroy(); scrub = null; } });
  keyframes(session, r, vurl);
}
function countUp(el, v) {
  if (!el || RM.matches || !fin(v)) return;
  const t0 = performance.now(), D = 700, small = el.querySelector('small').outerHTML;
  const f = now => { const k = Math.min(1, (now - t0) / D), e = 1 - Math.pow(1 - k, 3); el.innerHTML = (k < 1 ? (v * e).toFixed(1) : num(v).toFixed(1)) + small; if (k < 1) requestAnimationFrame(f); };
  requestAnimationFrame(f);
}
function fbItem(f, nm) {
  const cls = sevCls(f.severity) || 'pos';
  const sev = f.kind === 'mechanics' ? ({ ok: 'in range', minor: 'close', major: 'off' }[f.severity] || f.severity) : f.severity;
  const vals = f.kind === 'mechanics'
    ? `you <b>${esc(fmtMech(f.user, f.unit, f.key))}</b> · range ${esc(fmtRange(f.band, f.unit))}${fin(f.pointsLost) && f.pointsLost > 0 ? ` · <span class="${cls}">${MINUS}${fmtPts(f.pointsLost)} pts</span>` : ''}`
    : `you <b>${esc(fmtVal(f.user, f.unit))}</b> · ${esc(nm && nm.title ? nm.title.replace('Reference ', 'Ref ') : 'ref')} ${esc(fmtVal(fbRef(f), f.unit))} · <span class="${cls}">${esc(fmtVal(f.delta, f.unit, true))}</span>`;
  return `<li class="${esc(f.severity)}"><span class="lb">${esc(f.label)}</span><span class="chip ${cls}">${esc(sev)}</span>
    <span class="vals">${vals}</span>
    <span class="tx">${esc(fbText(f))}</span></li>`;
}
/** feedback list in engine order with a small group header whenever the kind changes (mechanics -> style) */
function fbList(items, nm) {
  let prev = null, out = '';
  const anyKind = items.some(f => f.kind);
  for (const f of items) {
    const k = f.kind === 'mechanics' ? 'mechanics' : 'style';
    if (anyKind && k !== prev) out += `<li class="fb-h">${k === 'mechanics' ? 'Mechanics · vs published ranges' : 'Style · vs ' + esc(nm && nm.title || 'closest reference')}</li>`;
    prev = k; out += fbItem(f, nm);
  }
  return out;
}
function planItem(p, lostOf) {
  const f = p.finding; const unit = p.unit;
  const dtxt = p.severity !== 'ok' && fin(p.delta) && unit ? ' ' + (p.kind === 'mechanics' ? (num(p.delta) > 0 ? '+' : MINUS) + fmtMech(Math.abs(num(p.delta)), unit, '') : fmtVal(p.delta, unit, true)) : '';
  const ftxt = (typeof f === 'string' ? f : f && f.label || '') + dtxt;
  const cls = sevCls(p.severity === 'ok' ? '' : p.severity);
  const lost = lostOf && p.kind === 'mechanics' ? lostOf(p.key) : NaN;
  return `<li><div><div class="dn">${esc(p.drill)}</div><div class="rx">${esc(p.prescription || '')}</div>
    <div class="why">${ftxt ? `<span class="chip ${cls}">${esc(ftxt)}</span>` : ''}${fin(lost) && lost > 0 ? `<span class="chip">${MINUS}${fmtPts(lost)} pts</span>` : ''}${p.kind === 'style' ? '<span class="chip">style</span>' : ''}<span>${esc(p.cue || '')}</span></div></div></li>`;
}

// ------------------------------------------------------------------ mechanics checks
function mechDomain(m) {
  const lo = num(m.band[0]), hi = num(m.band[1]); const tol = num(m.tolerance) > 0 ? num(m.tolerance) : ((hi - lo) / 2 || 1);
  let a = lo - tol * 1.25, b = hi + tol * 1.25;
  if (lo === 0 && num(m.ideal) === 0) a = 0;                       // one-sided "less is better" checks
  if (m.unit === 'deg') { a = Math.max(0, a); b = Math.min(180, b); }
  if (m.key === 'follow_through') b = Math.min(b, 1.5);            // band runs to 3 s; the interesting part is < 1.5 s
  const v = num(m.value);
  if (Number.isFinite(v)) { if (v < a) a = v - tol * 0.2; if (v > b) b = v + tol * 0.2; }
  return [a, b];
}
function mechRow(m, S) {
  const st = m.status || 'unmeasured', cls = MSTATUS[st] || 'muted';
  const sup = supRefs(m.source, S);
  const tgt = `target ${esc(fmtRange(m.band, m.unit))}${fin(m.ideal) ? ` · ideal ${esc(fmtIdeal(m.ideal, m.unit, m.key))}` : ''}`;
  if (st === 'unmeasured' || !fin(m.value)) {
    const why = m.reason ? String(m.reason).replace(/^./, c => c.toUpperCase()) : 'Not measurable from this camera angle / frames';
    return `<li class="mc unmeasured"><div class="mc-top"><span class="mc-lb">${esc(m.label)}${sup}</span><span class="mc-v muted">—</span></div>
      <div class="mc-bot"><span class="mc-why">${esc(why)}</span><span class="mc-pts num">—<span>/${fmtPts(m.maxPoints)}</span></span></div>
      <div class="mc-ax">${tgt} · not counted in the score</div></li>`;
  }
  const [a, b] = mechDomain(m), pct = x => clamp((num(x) - a) / ((b - a) || 1) * 100, 0, 100);
  const bl = pct(m.band[0]), bw = Math.max(1, pct(m.band[1]) - bl);
  const val = fmtMech(m.value, m.unit, m.key);
  return `<li class="mc ${esc(st)}"><div class="mc-top"><span class="mc-lb">${esc(m.label)}${sup}</span><span class="mc-v ${cls}">${esc(val)}</span></div>
    <div class="mc-bot"><div class="mc-bar" role="img" aria-label="${esc(m.label)}: ${esc(val)}, target ${esc(fmtRange(m.band, m.unit))}, ${MSTATUS_TXT[st]}">
        <i class="band" style="left:${bl.toFixed(1)}%;width:${bw.toFixed(1)}%"></i>${fin(m.ideal) ? `<i class="ideal" style="left:${pct(m.ideal).toFixed(1)}%"></i>` : ''}<i class="dot ${cls}" style="left:${pct(m.value).toFixed(1)}%"></i></div>
      <span class="mc-pts num"><b class="${st === 'good' ? '' : cls}">${fmtPts(m.points)}</b><span>/${fmtPts(m.maxPoints)}</span></span></div>
    <div class="mc-ax">${tgt}${gainOf(m) > 0.05 ? ` · <span class="${cls}">+${fmtPts(gainOf(m))} pts</span>` : ''}</div></li>`;
}
function mechSectionHTML(r, mk, S, assets) {
  if (!mk.length) return `<section class="sec" id="mech"><div class="sec-h"><h2>Mechanics checks</h2></div>
    <div class="card"><p class="muted" style="margin:0">This session was analysed before mechanics checks were added, so only the style match is available. Analyse the clip again to score it against the published ranges.</p></div></section>`;
  const c = st => mk.filter(m => m.status === st).length;
  const ms = mechScore(r);
  const top = topFixes(r, 1)[0];
  const nUn = c('unmeasured');
  const read = top ? `Biggest gain available: ${esc(fixPhrase(top))} (+${fmtPts(gainOf(top))} pts).`
    : fin(ms) ? 'Every measured check is inside its range — full marks on what the camera could see.' : '';
  const unTxt = nUn ? ` ${nUn} check${nUn === 1 ? '' : 's'} couldn't be measured and ${nUn === 1 ? "doesn't" : "don't"} count for or against you.` : '';
  const lowTxt = !fin(ms) ? ' Fewer than 40 of 100 points were measurable, so there is no mechanics score for this clip.' : '';
  return `<section class="sec" id="mech"><div class="sec-h"><h2>Mechanics checks</h2><span class="read">${mk.length} checks · published ranges</span></div>
    <div class="card mech-card">
      <div class="mc-sum"><span class="chip pos">${c('good')} in range</span><span class="chip warn">${c('close')} close</span><span class="chip neg">${c('off')} off</span>${nUn ? `<span class="chip">${nUn} not measured</span>` : ''}</div>
      <ul class="mcl">${mk.map(m => mechRow(m, S)).join('')}</ul>
      <div class="legend mc-legend"><span><i class="lg-band"></i>target range</span><span><i class="lg-ideal"></i>ideal</span><span><i class="lg-dot"></i>you</span><span>pts earned / weight</span></div>
    </div>
    <p class="read-line"><b>Read:</b> ${read}${unTxt}${lowTxt}</p>
    ${howScoringHTML(mk, S, assets)}</section>`;
}
function howScoringHTML(mk, S, assets) {
  const spec = (assets && assets.mechSpec && assets.mechSpec.length ? assets.mechSpec : mk);
  return `<details class="more how-scoring" id="how-scoring" style="margin-top:12px"><summary><span>How scoring works</span></summary><div class="body">
    <p class="note" style="margin-top:0">Each check has a target range from published jump-shot kinematics. Inside the range earns its full weight; outside, points fall off linearly to 0 at the range edge ± its tolerance ("close" = within half the tolerance). <b>Mechanics score</b> = points earned ÷ weight of the checks that could be measured, so a hidden joint doesn't cost you — but at least 40 of the 100 points must be measurable.</p>
    <ul class="how-l">${spec.map(m => `<li><span class="h-lb">${esc(m.label)}${supRefs(m.source, S)}</span><span class="h-w num">${fmtPts(m.points != null && m.maxPoints == null ? m.points : m.maxPoints)} pts</span>
      <span class="h-r num">${esc(fmtRange(m.band, m.unit))}${fin(m.ideal) ? ` · ideal ${esc(fmtIdeal(m.ideal, m.unit, m.key))}` : ''}${fin(m.tolerance) ? ` · tol ±${esc(fmtIdeal(m.tolerance, m.unit, ''))}` : ''}</span></li>`).join('')}</ul>
    ${S.list.length ? `<h3 style="margin:14px 0 6px">Sources</h3><ol class="src-list">${S.list.map((x, i) => `<li id="src-${i + 1}"><b>${esc(x.cite || x.id)}</b>${x.url ? ` <a href="${esc(x.url)}" target="_blank" rel="noopener">${esc(x.url.replace(/^https?:\/\//, ''))}</a>` : ''}${x.note ? `<span>${esc(x.note)}</span>` : ''}</li>`).join('')}</ol>` : ''}
    <p class="note"><b>2-D side view.</b> The published ranges come from 3-D lab measurements of skilled shooters; here they are applied to joint angles projected onto a single phone camera. Film side-on, whole body in frame — if the camera faces you, angles are foreshortened and the most affected checks are left unmeasured. Treat the ranges as coaching guides, not laws.</p>
    <p class="note"><b>Style match</b> is separate and does not feed the mechanics score: it is how closely your time-normalised motion follows the nearest reference shooter (dynamic time warping, plus a small learned encoder where available).</p>
  </div></details>`;
}
function consistencyHTML(session, sel) {
  const rs = session.results; const fr = session.frames;
  const rows = rs.map((r, i) => ({ i, mech: mechScore(r), style: styleScore(r), rel: r.events && fin(r.events.release) ? fr[clamp(r.events.release, 0, fr.length - 1)].t : NaN, elbow: metric(r, 'elbow'), relh: metric(r, 'relh'), load: metric(r, 'load'), knee: metric(r, 'knee') }));
  const cols = [['mech', 'Mech', v => v.toFixed(1)], ['style', 'Style', v => v.toFixed(1)], ['elbow', 'Elbow @ set', v => v.toFixed(0) + '°'], ['relh', 'Release ht', v => v.toFixed(2)], ['load', 'Load→rel', v => v.toFixed(2) + ' s'], ['knee', 'Knee min', v => v.toFixed(0) + '°']];
  const stat = k => { const a = rows.map(r => r[k]).filter(Number.isFinite); return { m: mean(a), s: sd(a), n: a.length }; };
  const pk = rows.some(r => Number.isFinite(r.mech)) ? 'mech' : 'style';
  const bestI = rows.reduce((b, r) => (Number.isFinite(r[pk]) && !(r[pk] <= rows[b][pk]) ? r.i : b), 0);
  const S = stat('mech'), ST = stat('style'), E = stat('elbow');
  const pm = (o, d, u = '') => (o.n ? o.m.toFixed(d) : '—') + u;
  return `<section class="sec" style="margin-top:0;margin-bottom:18px"><div class="sec-h"><h2>Consistency</h2><span class="read">${rs.length} shots · tap a row for its full breakdown</span></div>
    <div class="cons-kpis">
      <div class="card kpi"><div class="label">Mechanics</div><b>${pm(S, 1)}</b><div class="sub num">± ${S.n > 1 ? S.s.toFixed(1) : '—'} sd</div></div>
      <div class="card kpi"><div class="label">Style match</div><b>${pm(ST, 1)}</b><div class="sub num">± ${ST.n > 1 ? ST.s.toFixed(1) : '—'} sd</div></div>
      <div class="card kpi"><div class="label">Elbow @ set</div><b>${pm(E, 0, '°')}</b><div class="sub num">± ${E.n > 1 ? E.s.toFixed(1) : '—'}°</div></div>
    </div>
    <div class="tbl cons"><table id="cons-tbl"><thead><tr><th>Shot</th><th class="n">At</th>${cols.map(c => `<th class="n">${c[1]}</th>`).join('')}</tr></thead>
      <tbody>${rows.map(r => `<tr data-i="${r.i}" class="${r.i === sel ? 'sel' : ''}"><td>#${r.i + 1}${r.i === bestI ? ' <span class="chip accent" style="padding:3px 6px">best</span>' : ''}</td><td class="n num">${fmtS(r.rel, 1)}</td>${cols.map(c => `<td class="n num ${c[0] === pk && r.i === bestI ? 'best' : ''}">${fin(r[c[0]]) ? c[2](r[c[0]]) : '—'}</td>`).join('')}</tr>`).join('')}</tbody>
      <tfoot><tr><td>mean</td><td></td>${cols.map(c => { const s = stat(c[0]); return `<td class="n">${s.n ? c[2](s.m) : '—'}</td>`; }).join('')}</tr>
      <tr><td>± sd</td><td></td>${cols.map(c => { const s = stat(c[0]); return `<td class="n">${s.n > 1 ? c[2](s.s).replace('°', '°') : '—'}</td>`; }).join('')}</tr></tfoot></table></div>
    <p class="read-line"><b>Read:</b> ${(() => { const P = S.n > 1 ? S : ST; const lab = S.n > 1 ? 'mechanics score' : 'style match';
      return `Across ${rs.length} shots your ${lab} varies by <b>± ${P.n > 1 ? P.s.toFixed(1) : '—'}</b> and your elbow at the set point by <b>± ${E.n > 1 ? E.s.toFixed(1) : '—'}°</b>. ${P.n < 2 ? '' : P.s < 3 ? 'That is a repeatable stroke — now chase the gaps.' : P.s < 7 ? 'Fairly consistent; the plan below targets the selected shot.' : 'Big shot-to-shot swings: groove one motion before changing mechanics.'}`; })()}</p></section>`;
}
function refSeq(assets, name) {
  if (!assets || !name) return null;
  const lists = [assets.refs, assets.library && assets.library.references].filter(Array.isArray);
  for (const L of lists) { const e = L.find(x => x && x.name === name); if (e) return e.seq60 || e.seq || (e.features && e.features.length === 60 ? e.features : null) || null; }
  return null;
}
function curvesHTML(r, assets, FEATURES) {
  const u = r.seq60, rf = refSeq(assets, r.closest && r.closest.name);
  if (!u || !u.length || !FEATURES.length) return '';
  const path = (seq, j) => { const v = seq.map(x => num(x[j])); const f = v.filter(Number.isFinite); return { v, lo: Math.min(...f), hi: Math.max(...f) }; };
  return `<div class="card"><h3>Time-normalised curves</h3><div class="curves" style="display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:8px">${FEATURES.map((F, j) => {
    const a = path(u, j), b = rf ? path(rf, j) : null; const lo = Math.min(a.lo, b ? b.lo : a.lo), hi = Math.max(a.hi, b ? b.hi : a.hi);
    const pts = vv => vv.map((y, i) => Number.isFinite(y) ? `${(i / (vv.length - 1) * 160).toFixed(1)},${(52 - (y - lo) / ((hi - lo) || 1) * 48).toFixed(1)}` : '').filter(Boolean).join(' ');
    return `<div style="background:var(--surface-2);border:1px solid var(--line);border-radius:8px;padding:8px 10px"><div class="row-between" style="font-size:12px;color:var(--muted)"><span>${esc(F.label)}</span></div>
      <svg viewBox="0 0 160 56" preserveAspectRatio="none" style="width:100%;height:auto;display:block" aria-hidden="true">${b ? `<polyline fill="none" stroke="${COL.muted}" stroke-width="1.6" stroke-dasharray="4 3" points="${pts(b.v)}"/>` : ''}<polyline fill="none" stroke="${COL.accent}" stroke-width="1.8" points="${pts(a.v)}"/></svg></div>`; }).join('')}</div>
    <p class="note">60 samples onset → follow-through; amber you${rf ? ', dashed grey the closest reference' : ''}.</p></div>`;
}

async function keyframeImage(session, r, key, maxSide, q) {
  const ev = r.events || {}; const fr = session.frames; const i = clamp(fin(ev[key]) ? ev[key] | 0 : fr.length >> 1, 0, fr.length - 1);
  const url = state.sessionVideo && state.sessionVideo.id === session.id ? state.sessionVideo.url : null;
  const gm = ghostMapper(session, r);
  return poseImage(url, (session.trimStart || 0) + fr[i].t, lmAt(session, r, i), session.meta, maxSide, gm && maxSide > 200 ? gm.at(i) : null, q, url ? null : bodyBox(session, r));
}
async function keyframes(session, r, vurl) {
  const keys = ['set', 'release', 'ftEnd'];
  for (let k = 0; k < keys.length; k++) {
    try {
      const src = await keyframeImage(session, r, keys[k], 480, 0.82);
      const ph = $$('#kf .ph')[k]; if (!ph) return;
      ph.classList.remove('skel-box'); ph.innerHTML = `<img alt="${keys[k]} frame with skeleton" src="${src}">`;
    } catch (e) { console.warn('keyframe', e); }
  }
}

// ---- radar
function radarSVG(radar, series) {
  if (!radar || !radar.axes || !radar.axes.length) return '<p class="muted">No radar data.</p>';
  const ax = radar.axes.map(a => (typeof a === 'string' ? a : a.label || a.key || '')), n = ax.length;
  const cx = 220, cy = 152, R = 100;
  const pt = (i, v) => { const a = -Math.PI / 2 + i / n * Math.PI * 2; return [cx + Math.cos(a) * R * v, cy + Math.sin(a) * R * v]; };
  let s = `<svg class="chart" viewBox="0 0 440 300" role="img" aria-label="Radar chart of mechanics: you vs reference">`;
  for (const k of [0.25, 0.5, 0.75, 1]) s += `<polygon points="${ax.map((_, i) => pt(i, k).map(x => x.toFixed(1)).join(',')).join(' ')}" fill="none" stroke="#242a36"/>`;
  ax.forEach((_, i) => { const [x, y] = pt(i, 1); s += `<line x1="${cx}" y1="${cy}" x2="${x.toFixed(1)}" y2="${y.toFixed(1)}" stroke="#242a36"/>`; });
  for (const se of series) {
    const vals = radar[se.k]; if (!vals || !vals.length) continue;
    const pts = vals.map((v, i) => pt(i, clamp(num(v) || 0, 0, 1.08)));
    s += `<polygon points="${pts.map(p => p.map(x => x.toFixed(1)).join(',')).join(' ')}" fill="${se.fill || 'none'}" stroke="${se.col}" stroke-width="${se.w}" stroke-linejoin="round" ${se.dash ? `stroke-dasharray="${se.dash}"` : ''}/>`;
    if (se.dots) pts.forEach((p, i) => { s += `<circle cx="${p[0].toFixed(1)}" cy="${p[1].toFixed(1)}" r="3.5" fill="${se.col}"><title>${esc(ax[i])}</title></circle>`; });
  }
  ax.forEach((l, i) => { const [x, y] = pt(i, 1.2); const anc = Math.abs(x - cx) < 8 ? 'middle' : x > cx ? 'start' : 'end'; s += `<text x="${x.toFixed(1)}" y="${(y + 4).toFixed(1)}" text-anchor="${anc}">${esc(l)}</text>`; });
  return s + '</svg>';
}

// ---- scrubber over the user's real video (or a skeleton replay)
class Scrubber {
  constructor(session, r, FEATURES, assets, vurl) {
    this.s = session; this.r = r; this.F = FEATURES; this.fr = session.frames; this.N = this.fr.length; this.fps = session.meta.fps;
    this.ev = r.events || {}; this.speed = 0.5; this.ghostOn = true; this.playing = false; this.vurl = vurl;
    this.gm = ghostMapper(session, r); this.rseq = refSeq(assets, r.closest && r.closest.name);
    if (!this.gm) { $('#sc-ghost').disabled = true; $('#sc-gtag').hidden = true; this.ghostOn = false; }
    this.cv = $('#sc-canvas'); this.ctx = this.cv.getContext('2d'); this.stage = $('#sc-stage'); this.video = $('#sc-video');
    this.range = $('#sc-range'); this.range.max = this.N - 1;
    const w = r._window; this.lo = w ? clamp(w.start, 0, this.N - 1) : 0; this.hi = w ? clamp(w.end, this.lo, this.N - 1) : this.N - 1;
    if (vurl) this.stage.style.setProperty('--ar', session.meta.width + '/' + session.meta.height);
    else { const bb = bodyBox(session, r); this.box = bb; this.stage.style.setProperty('--ar', Math.max(0.6, Math.min(1.4, bb.w * session.meta.width / (bb.h * session.meta.height) * 1.25)).toFixed(3)); }
    // markers
    const pct = i => (clamp(i, 0, this.N - 1) / Math.max(1, this.N - 1) * 100).toFixed(1) + '%';
    $('#sc-marks').innerHTML = [['onset', 'onset'], ['set', 'set'], ['release', 'release'], ['ftEnd', 'end']].filter(([k]) => fin(this.ev[k])).map(([k, l]) => { const v = clamp(this.ev[k], 0, this.N - 1) / Math.max(1, this.N - 1); return `<span class="mk ${k === 'release' ? 'rel' : ''}" style="left:${(v * 100).toFixed(1)}%;transform:translateX(${v < 0.07 ? '-15%' : v > 0.93 ? '-85%' : '-50%'})">${l}</span>`; }).join('');
    this.frame = clamp(fin(this.ev.release) ? this.ev.release : 0, 0, this.N - 1);
    this.h = [];
    const on = (el, ev, fn) => { if (!el) return; el.addEventListener(ev, fn); this.h.push(() => el.removeEventListener(ev, fn)); };
    on(this.range, 'input', () => { this.pause(); this.set(+this.range.value); });
    on($('#sc-prev'), 'click', () => { this.pause(); this.set(this.frame - 1); });
    on($('#sc-next'), 'click', () => { this.pause(); this.set(this.frame + 1); });
    on($('#sc-play'), 'click', () => (this.playing ? this.pause() : this.play()));
    on($('#sc-ghost'), 'click', () => { this.ghostOn = !this.ghostOn; $('#sc-ghost').classList.toggle('on', this.ghostOn); $('#sc-ghost').setAttribute('aria-pressed', this.ghostOn); $('#sc-gtag').hidden = !this.ghostOn; this.draw(); });
    $$('#scrub [data-speed]').forEach(b => on(b, 'click', () => { this.speed = +b.dataset.speed; $$('#scrub [data-speed]').forEach(x => x.classList.toggle('on', x === b)); if (this.video && this.playing) this.video.playbackRate = this.speed; }));
    $$('#scrub [data-jump]').forEach(b => on(b, 'click', () => { this.pause(); this.set(this.ev[b.dataset.jump]); }));
    on(document, 'keydown', e => {
      if (current !== 'results') return;
      const t = document.activeElement && document.activeElement.tagName;
      if (['INPUT', 'TEXTAREA', 'SELECT'].includes(t) && document.activeElement !== this.range) return;
      if (e.key === 'ArrowRight') { this.pause(); this.set(this.frame + (e.shiftKey ? 5 : 1)); e.preventDefault(); }
      else if (e.key === 'ArrowLeft') { this.pause(); this.set(this.frame - (e.shiftKey ? 5 : 1)); e.preventDefault(); }
      else if (e.key === 'Home') { this.pause(); this.set(0); e.preventDefault(); }
      else if (e.key === 'End') { this.pause(); this.set(this.N - 1); e.preventDefault(); }
      else if (e.key === ' ' && t !== 'BUTTON' && t !== 'SUMMARY' && t !== 'A') { this.playing ? this.pause() : this.play(); e.preventDefault(); }
    });
    this.ro = new ResizeObserver(() => this.size()); this.ro.observe(this.stage);
    if (this.video) {
      on(this.video, 'seeked', () => this.draw());
      loadVideo(this.video, vurl).then(() => this.set(this.frame)).catch(() => { this.video.remove(); this.video = null; this.draw(); });
    }
    this.size(); this.set(this.frame);
    window.__jumpshot.scrub = this;
  }
  size() {
    const d = Math.min(2, devicePixelRatio || 1); const w = this.stage.clientWidth, h = this.stage.clientHeight;
    if (!w || !h) return;
    this.cv.width = Math.round(w * d); this.cv.height = Math.round(h * d); this.W = w; this.H = h; this.d = d;
    this.ctx.setTransform(d, 0, 0, d, 0, 0); this.draw();
  }
  tOf(i) { return (this.s.trimStart || 0) + this.fr[clamp(i, 0, this.N - 1)].t; }
  set(i, fromVideo) {
    if (!fin(i)) return;
    this.frame = clamp(Math.round(i), 0, this.N - 1);
    if (this.video && !fromVideo && this.video.readyState >= 1) { try { this.video.currentTime = this.tOf(this.frame) + 0.001; } catch (e) {} }
    this.draw(); this.readouts();
  }
  draw() {
    const ctx = this.ctx, W = this.W, H = this.H; if (!W) return;
    const m = this.s.meta;
    ctx.clearRect(0, 0, W, H);
    let map, r;
    if (this.video) { r = containRect(W, H, m.width, m.height); map = (x, y) => [r.x + x * r.w, r.y + y * r.h]; }
    else {      // skeleton replay: zoom to the body (and ghost) so it fills the stage
      const bb = this.box || (this.box = bodyBox(this.s, this.r));
      r = containRect(W, H, bb.w * m.width, bb.h * m.height);
      map = (x, y) => [r.x + (x - bb.x) / bb.w * r.w, r.y + (y - bb.y) / bb.h * r.h];
      ctx.fillStyle = '#0d1016'; ctx.fillRect(0, 0, W, H); floorGrid(ctx, W, H, { x: 0, y: r.y, w: W, h: r.h });
    }
    const lw = Math.max(2.2, r.h / 150);
    if (this.ghostOn && this.gm) {
      const g = this.gm.at(this.frame);
      drawPose(ctx, g, (x, y) => { const q = this.gm.map(x, y); return map(q[0], q[1]); }, { col: '#c9d0de', w: lw * 0.85, alpha: this.video ? 0.85 : 0.7, dash: [lw * 2.2, lw * 1.8], halo: !!this.video });
    }
    const ulm = lmAt(this.s, this.r, this.frame);
    if (ulm) drawPose(ctx, ulm, map, { col: COL.accent, w: lw, joints: true, arm: this.r.hand });
    else { ctx.fillStyle = COL.muted; ctx.font = '13px ui-monospace, Menlo, monospace'; ctx.textAlign = 'center'; ctx.fillText('no pose in this frame', W / 2, H / 2); }
  }
  phase(i) { const e = this.ev; if (i < e.onset) return 'stance'; if (i < e.set) return 'load'; if (i < e.release) return 'lift'; if (i === e.release) return 'release'; if (i <= e.ftEnd) return 'follow-through'; return 'recovery'; }
  readouts() {
    const i = this.frame, f = this.fr[i];
    $('#sc-time').textContent = f.t.toFixed(2) + ' s';
    $('#sc-frame').textContent = 'f' + String(i).padStart(2, '0') + ' / ' + (this.N - 1);
    $('#sc-phase').textContent = this.phase(i);
    if (+this.range.value !== i) this.range.value = i;
    $$('#scrub [data-jump]').forEach(x => x.classList.toggle('on', this.ev[x.dataset.jump] === i));
    const pf = this.r.perFrame || {}, e = this.ev;
    const span = Math.max(1, (e.ftEnd | 0) - (e.onset | 0));
    const k = clamp(Math.round((i - e.onset) / span * 59), 0, 59);
    const inShot = i >= e.onset && i <= e.ftEnd;
    let h = '';
    for (let j = 0; j < this.F.length; j++) {
      const F = this.F[j]; const arr = pf[F.key]; const u = arr ? num(arr[i]) : NaN;
      const rv = this.rseq && inShot && this.rseq[k] ? num(this.rseq[k][j]) : NaN;
      const d = u - rv; const tol = F.tolerance || (F.unit === 'deg' ? 5 : 0.06);
      const cls = !Number.isFinite(d) ? '' : Math.abs(d) <= tol ? 'pos' : Math.abs(d) >= tol * 2.5 ? 'neg' : 'warn';
      h += `<div class="ro"><div class="k" title="${esc(F.label)}">${esc(F.label)}</div><div class="v">${fmtNum(u, F.unit)}</div><div class="d">${Number.isFinite(rv) ? `ref ${fmtNum(rv, F.unit)} <span class="${cls}">${d > 0 ? '+' : ''}${fmtNum(d, F.unit)}</span>` : inShot ? '&nbsp;' : 'outside shot'}</div></div>`;
    }
    $('#sc-ro').innerHTML = h;
    const rd = $('#sc-read');
    if (rd) {
      const rel = e.release, wh = pf.wrist_height ? num(pf.wrist_height[rel]) : metric(this.r, 'relh');
      rd.innerHTML = `<b>Read:</b> At release (f${rel}, ${this.fr[clamp(rel, 0, this.N - 1)].t.toFixed(2)} s) your wrist is <b>${fmtVal(wh, 'torso')}</b> above your head${fin(fbRef(fbOf(this.r, 'relh') || {})) ? ` vs ${fmtVal(fbRef(fbOf(this.r, 'relh')), 'torso')} for ${esc(splitName(this.r.closest.name).title)}` : ''}. The ghost is the reference, time-warped to your phases, scaled to your torso and standing on your feet.`;
    }
  }
  play() {
    this.playing = true; const b = $('#sc-play'); b.classList.add('primary'); b.querySelector('span').textContent = 'Pause'; b.querySelector('svg').innerHTML = '<path d="M8 5v14M16 5v14"/>';
    if (this.frame >= this.hi) this.set(this.lo);
    if (this.video) {
      const v = this.video; v.playbackRate = this.speed;
      v.play().catch(() => this.pause());
      const tick = (now, md) => {
        if (!this.playing) return;
        const t = md ? md.mediaTime : v.currentTime;
        const i = Math.round((t - (this.s.trimStart || 0)) * this.fps);
        if (i > this.hi || v.ended) { v.currentTime = this.tOf(this.lo) + 0.001; this.set(this.lo, true); }
        else this.set(i, true);
        this._vh = 'requestVideoFrameCallback' in v ? v.requestVideoFrameCallback(tick) : requestAnimationFrame(tick);
      };
      this._vh = 'requestVideoFrameCallback' in v ? v.requestVideoFrameCallback(tick) : requestAnimationFrame(tick);
    } else {
      let last = performance.now(), acc = 0;
      const tick = now => {
        if (!this.playing) return;
        acc += (now - last) / 1000 * this.speed; last = now; let i = this.frame;
        while (acc >= 1 / this.fps) { acc -= 1 / this.fps; i = i + 1 > this.hi ? this.lo : i + 1; }
        if (i !== this.frame) this.set(i);
        this._raf = requestAnimationFrame(tick);
      };
      this._raf = requestAnimationFrame(tick);
    }
  }
  pause() {
    if (!this.playing) return;
    this.playing = false; const b = $('#sc-play'); if (b) { b.classList.remove('primary'); b.querySelector('span').textContent = 'Play'; b.querySelector('svg').innerHTML = '<path d="M8 5v14l11-7z"/>'; }
    if (this.video) { this.video.pause(); this.set(this.frame); }
    cancelAnimationFrame(this._raf);
  }
  destroy() { this.playing = false; cancelAnimationFrame(this._raf); if (this.video) this.video.pause(); this.ro.disconnect(); this.h.forEach(f => f()); }
}

// ------------------------------------------------------------------ share card + summary
function summaryText(session, r) {
  const nm = r.closest && r.closest.name || '';
  const fixes = topFixes(r, 2);
  const plan = r.plan || [];
  const lines = [`Jumpshot Analyzer — mechanics score ${f1(mechScore(r))}/100`, `Style match ${f1(styleScore(r))} · closest reference: ${nm}`];
  const load = metric(r, 'load'), relh = metric(r, 'relh'), el = metric(r, 'elbow');
  lines.push(`Load→release ${fmtVal(load, 's')} · release height ${fmtVal(relh, 'torso', true)} · elbow at set ${fmtVal(el, 'deg')}`);
  if (fixes.length) { lines.push('Top fixes:'); fixes.forEach((m, i) => { const pl = plan.find(p => p.key === m.key); lines.push(`${i + 1}. ${m.label}: ${fmtMech(m.value, m.unit, m.key)} (target ${fmtRange(m.band, m.unit)}, +${fmtPts(gainOf(m))} pts) → ${fixPhrase(m)}${pl ? ' · drill: ' + pl.drill : ''}`); }); }
  else if (hasMech(r)) lines.push('All measured mechanics checks inside their ranges.');
  if (session.results && session.results.length > 1) { const sc = session.results.map(mechScore).filter(Number.isFinite); if (sc.length > 1) lines.push(`Consistency over ${sc.length} shots: mechanics ${mean(sc).toFixed(1)} ± ${sd(sc).toFixed(1)}`); }
  lines.push(`Analysed on-device · ${fmtDate(session.created)}`);
  return lines.join('\n');
}
async function copySummary(session, r) {
  const txt = summaryText(session, r);
  try { await navigator.clipboard.writeText(txt); toast('Summary copied'); }
  catch (e) {
    const ta = document.createElement('textarea'); ta.value = txt; ta.style.cssText = 'position:fixed;opacity:0'; document.body.appendChild(ta); ta.select();
    try { document.execCommand('copy'); toast('Summary copied'); } catch (x) { toast('Copy failed'); } ta.remove();
  }
}
function wrapText(ctx, text, x, y, maxW, lh, maxLines = 3) {
  const words = String(text).split(/\s+/); let line = '', n = 0;
  for (const w of words) { const t = line ? line + ' ' + w : w; if (ctx.measureText(t).width > maxW && line) { ctx.fillText(line, x, y); y += lh; line = w; if (++n >= maxLines - 1) { /* last line */ } } else line = t; }
  if (line) { ctx.fillText(line, x, y); y += lh; } return y;
}
function rr(ctx, x, y, w, h, r) { ctx.beginPath(); ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r); ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath(); }
async function shareCard(session, r) {
  const W = 1080, H = 1350, c = document.createElement('canvas'); c.width = W; c.height = H; const ctx = c.getContext('2d');
  const SANS = '-apple-system,BlinkMacSystemFont,Inter,"Segoe UI",Roboto,Helvetica,Arial,sans-serif', MONO = 'ui-monospace,"SF Mono",Menlo,Consolas,monospace';
  ctx.fillStyle = '#0b0d12'; ctx.fillRect(0, 0, W, H);
  const gr = ctx.createRadialGradient(W * 0.85, 120, 10, W * 0.85, 120, 700); gr.addColorStop(0, 'rgba(255,160,48,.14)'); gr.addColorStop(1, 'rgba(255,160,48,0)'); ctx.fillStyle = gr; ctx.fillRect(0, 0, W, H);
  // header
  ctx.fillStyle = COL.accent; rr(ctx, 72, 72, 64, 64, 16); ctx.fill();
  ctx.fillStyle = COL.ink; ctx.font = `800 28px ${MONO}`; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText('JS', 104, 105);
  ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic'; ctx.fillStyle = COL.text; ctx.font = `700 34px ${SANS}`; ctx.fillText('Jumpshot Analyzer', 160, 100);
  ctx.fillStyle = COL.muted; ctx.font = `500 24px ${MONO}`; ctx.fillText(fmtDate(session.created).toUpperCase(), 160, 132);
  // skeleton panel (right)
  const px = 600, py = 200, pw = 408, ph = 600;
  ctx.fillStyle = '#12151c'; rr(ctx, px, py, pw, ph, 28); ctx.fill(); ctx.strokeStyle = '#242a36'; ctx.lineWidth = 2; ctx.stroke();
  const fr = session.frames, ev = r.events || {}, ri = clamp(fin(ev.release) ? ev.release : 0, 0, fr.length - 1);
  const lm = lmAt(session, r, ri);
  if (lm) {
    // fit the release pose (+ ghost) into the panel, standing on a floor line
    const gm = ghostMapper(session, r); const pts = [];
    const asp = session.meta.width / session.meta.height;
    for (let i = 0; i < 33; i++) { const p = P(lm, i); if (p && p[2] > 0.3) pts.push([p[0] * asp, p[1]]); }
    let gl = null; if (gm) { const g = gm.at(ri); gl = []; for (let i = 0; i < 33; i++) { const p = P(g, i); gl.push(p && p[2] >= 0.3 ? gm.map(p[0], p[1]) : null); } gl.forEach(p => p && pts.push([p[0] * asp, p[1]])); }
    const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]); const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
    const s = Math.min((pw - 90) / ((maxX - minX) || 1), (ph - 170) / ((maxY - minY) || 1));
    const ox = px + pw / 2 - (minX + maxX) / 2 * s, oy = py + ph - 70 - maxY * s;
    const map = (x, y) => [ox + x * asp * s, oy + y * s];
    ctx.strokeStyle = '#2e3545'; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(px + 30, oy + maxY * s + 14); ctx.lineTo(px + pw - 30, oy + maxY * s + 14); ctx.stroke();
    if (gl) { const flat = new Float32Array(132); gl.forEach((p, i) => { if (p) { flat[i * 4] = p[0]; flat[i * 4 + 1] = p[1]; flat[i * 4 + 3] = 1; } }); drawPose(ctx, flat, map, { col: '#8a93a6', w: 7, dash: [14, 12], alpha: 0.8, halo: false }); }
    drawPose(ctx, lm, map, { col: COL.accent, w: 11, joints: true, arm: r.hand, halo: false });
    ctx.fillStyle = COL.muted; ctx.font = `600 22px ${MONO}`; ctx.fillText('AT RELEASE', px + 30, py + 48);
    if (gl) { ctx.textAlign = 'right'; ctx.fillText('- - ' + splitName(r.closest.name).title.toUpperCase(), px + pw - 30, py + 48); ctx.textAlign = 'left'; }
  }
  // mechanics score (hero)
  const msc = mechScore(r), ssc = styleScore(r), mk = hasMech(r) ? r.mechanics : [];
  ctx.fillStyle = COL.muted; ctx.font = `600 26px ${MONO}`; ctx.fillText('MECHANICS SCORE', 72, 250);
  { const sc = fin(msc) ? msc.toFixed(1) : '—'; let fs = 230; ctx.font = `800 ${fs}px ${SANS}`; while (ctx.measureText(sc).width > 400 && fs > 120) { fs -= 10; ctx.font = `800 ${fs}px ${SANS}`; }
    ctx.fillStyle = COL.accent; ctx.fillText(sc, 60, 470); const sw = ctx.measureText(sc).width; ctx.fillStyle = COL.muted; ctx.font = `700 48px ${SANS}`; ctx.fillText('/100', 72 + sw, 470); }
  // per-check points strip: width = weight, fill = points earned, colour = status
  if (mk.length) {
    const sx = 72, sy = 520, sw = 500, sh = 26, tot = mk.reduce((q, m) => q + (num(m.maxPoints) || 0), 0) || 1; let x = sx;
    mk.forEach(m => { const w = (num(m.maxPoints) || 0) / tot * sw; ctx.fillStyle = '#181c25'; ctx.fillRect(x + 1, sy, Math.max(1, w - 3), sh);
      if (m.status !== 'unmeasured') { ctx.fillStyle = { good: COL.pos, close: COL.warn, off: COL.neg }[m.status] || COL.faint; ctx.fillRect(x + 1, sy, Math.max(0, (w - 3) * clamp(num(m.points) / (num(m.maxPoints) || 1), 0, 1)), sh); }
      x += w; });
    const c = st => mk.filter(m => m.status === st).length;
    ctx.fillStyle = COL.muted; ctx.font = `500 26px ${SANS}`; ctx.fillText(`${c('good')} in range · ${c('close')} close · ${c('off')} off${c('unmeasured') ? ' · ' + c('unmeasured') + ' n/m' : ''}`, 72, 588);
  }
  // style match: one small line
  const nm = splitName(r.closest && r.closest.name);
  ctx.fillStyle = COL.muted; ctx.font = `600 22px ${MONO}`; ctx.fillText('STYLE MATCH', 72, 680);
  ctx.fillStyle = COL.text; ctx.font = `700 40px ${MONO}`; ctx.fillText(f1(ssc), 72, 730); { const w0 = ctx.measureText(f1(ssc)).width;
    ctx.fillStyle = COL.muted; ctx.font = `500 28px ${SANS}`; ctx.fillText(`vs ${nm.title}`, 72 + w0 + 18, 728, 500 - w0 - 18); }
  if (nm.style) { ctx.fillStyle = COL.faint; ctx.font = `500 24px ${SANS}`; ctx.fillText(nm.style, 72, 770, 500); }
  // top fixes from the mechanics checks (largest points available first)
  const fixes = topFixes(r, 2);
  let y = 866; ctx.fillStyle = COL.muted; ctx.font = `600 26px ${MONO}`; ctx.fillText(fixes.length ? 'TOP FIXES' : 'MECHANICS CHECK', 72, y); y += 26;
  if (!fixes.length) { ctx.fillStyle = COL.text; ctx.font = `600 36px ${SANS}`; ctx.fillText(mk.length ? 'Every measured check inside its range.' : 'Re-analyse for mechanics checks.', 72, y + 44); }
  fixes.forEach(m => {
    const col = m.status === 'off' ? COL.neg : COL.warn;
    const bx = 72, bw = W - 144, bh = 150; ctx.fillStyle = '#12151c'; rr(ctx, bx, y, bw, bh, 22); ctx.fill(); ctx.strokeStyle = '#242a36'; ctx.lineWidth = 2; ctx.stroke();
    ctx.fillStyle = col; ctx.beginPath(); ctx.arc(bx + 40, y + 46, 10, 0, 7); ctx.fill();
    ctx.fillStyle = COL.text; ctx.font = `700 34px ${SANS}`; ctx.fillText(m.label, bx + 66, y + 58, bw - 300);
    ctx.textAlign = 'right'; ctx.fillStyle = col; ctx.font = `700 34px ${MONO}`; ctx.fillText(`+${fmtPts(gainOf(m))} pts`, bx + bw - 30, y + 58); ctx.textAlign = 'left';
    ctx.fillStyle = COL.muted; ctx.font = `500 26px ${MONO}`; ctx.fillText(`${fmtMech(m.value, m.unit, m.key)}  ·  target ${fmtRange(m.band, m.unit)}`, bx + 66, y + 98, bw - 100);
    const tip = fixPhrase(m); ctx.fillStyle = COL.text; ctx.font = `500 26px ${SANS}`; ctx.fillText(tip.charAt(0).toUpperCase() + tip.slice(1), bx + 66, y + 134, bw - 100);
    y += bh + 16;
  });
  // footer
  ctx.fillStyle = '#242a36'; ctx.fillRect(72, H - 120, W - 144, 2);
  ctx.fillStyle = COL.muted; ctx.font = `600 24px ${MONO}`; ctx.fillText('JUMPSHOT ANALYZER · ON-DEVICE AI', 72, H - 70);
  ctx.textAlign = 'right'; ctx.fillText(new Date(session.created).toISOString().slice(0, 10), W - 72, H - 70); ctx.textAlign = 'left';
  return new Promise(res => c.toBlob(b => res(b), 'image/png'));
}
let drawer = null;
function sheet(html) {
  if (!drawer) {
    drawer = document.createElement('div'); drawer.className = 'drawer'; drawer.setAttribute('role', 'dialog'); drawer.setAttribute('aria-modal', 'true');
    const scrim = document.createElement('div'); scrim.className = 'scrim'; scrim.onclick = closeSheet; drawer._scrim = scrim;
    document.body.append(scrim, drawer);
    document.addEventListener('keydown', e => { if (e.key === 'Escape') closeSheet(); });
  }
  drawer.innerHTML = '<div class="grab"></div>' + html; drawer.classList.add('open'); drawer._scrim.classList.add('open');
  return drawer;
}
function closeSheet() { if (drawer) { drawer.classList.remove('open'); drawer._scrim.classList.remove('open'); } }
window.__jumpshot.closeSheet = closeSheet;
async function openShare(session, r) {
  const d = sheet('<h2>Share card</h2><div class="share-prev skel-box" style="aspect-ratio:4/5"></div>');
  const blob = await shareCard(session, r);
  const url = URL.createObjectURL(blob);
  const fname = `jumpshot-${fin(mechScore(r)) ? Math.round(mechScore(r)) : 'style' + Math.round(styleScore(r) || 0)}-${session.created.slice(0, 10)}.png`;
  const file = new File([blob], fname, { type: 'image/png' });
  const canShare = !!(navigator.canShare && navigator.canShare({ files: [file] }));
  d.innerHTML = `<div class="grab"></div><h2>Share card</h2><div class="share-prev"><img id="share-img" src="${url}" alt="Share card: mechanics score ${f1(mechScore(r))}, style match ${f1(styleScore(r))}"></div>
    <div class="acts">${canShare ? '<button class="btn primary" id="sh-share" type="button">Share…</button>' : ''}
      <a class="btn ${canShare ? '' : 'primary'}" id="sh-dl" href="${url}" download="${fname}">Download PNG</a>
      <button class="btn" id="sh-copy" type="button">Copy summary</button><button class="btn ghost" id="sh-close" type="button">Close</button></div>
    <p class="note center">1080×1350 · made on this device. Only the skeleton is drawn, never your video.</p>`;
  window.__jumpshot.lastShare = { blob, size: blob.size, canShare };
  const sb = $('#sh-share'); if (sb) sb.onclick = async () => { try { await navigator.share({ files: [file], title: 'My jumpshot', text: summaryText(session, r) }); } catch (e) { if (e.name !== 'AbortError') toast('Sharing failed — use Download'); } };
  $('#sh-copy').onclick = () => copySummary(session, r);
  $('#sh-close').onclick = closeSheet;
}

// ------------------------------------------------------------------ history
async function renderHistory() {
  const box = $('#history-body');
  box.innerHTML = '<div class="skel-box" style="height:40px;width:40%;margin-bottom:12px"></div><div class="skel-box" style="height:160px;margin-bottom:12px"></div><div class="skel-box" style="height:72px;margin-bottom:8px"></div><div class="skel-box" style="height:72px"></div>';
  const list = await DB.all(); refreshCount();
  state.compare = state.compare.filter(id => list.some(s => s.id === id));
  const tools = `<div class="hist-tools"><button class="btn sm" id="h-json" type="button" ${list.length ? '' : 'disabled'}>Export JSON</button><button class="btn sm" id="h-csv" type="button" ${list.length ? '' : 'disabled'}>Export CSV</button><button class="btn sm" id="h-imp" type="button">Import JSON</button></div>`;
  if (!list.length) {
    box.innerHTML = `<div class="view-h"><div><div class="eyebrow">History</div><h1 id="h-hist" style="margin:4px 0 0">Your sessions</h1></div></div>
      <div class="card empty"><svg viewBox="0 0 24 24"><path d="M3 12a9 9 0 1 0 3-6.7L3 8"/><path d="M3 3v5h5M12 7v5l3 2"/></svg>
      <h2>No sessions yet</h2><p>Every analysis is saved here — on this device only — so you can track your score, trends and consistency over time.</p>
      <a class="btn primary" href="#/">Analyze a shot</a></div><div style="margin-top:12px">${tools}</div>`;
    wireHistoryTools(list); return;
  }
  const eng = await getEngine();
  // feature trends from the engine (style deltas); score trend computed here on the mechanics score
  let sum = null; try { sum = eng.summarizeHistory(list.map(s => { const r = primaryResult(s) || {}; return { ...r, score: fin(styleScore(r)) ? styleScore(r) : num(r.score) }; })); } catch (e) { console.warn('summarizeHistory', e); }
  const sc = list.map(sessScores);
  const scores = sc.map(x => x.mech), mf = scores.filter(Number.isFinite), styles = sc.map(x => x.style).filter(Number.isFinite);
  const first = mf[0], last = mf[mf.length - 1], best = mf.length ? Math.max(...mf) : NaN;
  const lastS = list[list.length - 1], lastSc = sc[sc.length - 1];
  const nMech = mf.length, nOld = list.length - nMech;
  let slope = null; if (nMech > 1) { const mx = (nMech - 1) / 2, my = mean(mf); let a = 0, b = 0; mf.forEach((v, i) => { a += (i - mx) * (v - my); b += (i - mx) ** 2; }); slope = b ? a / b : 0; }
  box.innerHTML = `<div class="view-h" style="justify-content:space-between;flex-wrap:wrap"><div><div class="eyebrow">History · on this device</div><h1 id="h-hist" style="margin:4px 0 0">Your sessions</h1></div>${tools}</div>
    <div class="grid g4">
      <div class="card kpi"><div class="label">Latest mechanics</div><b class="hero">${f1(lastSc.mech)}<small>/100</small></b><div class="sub">${esc(fmtDate(lastS.created))} · style ${esc(f1(lastSc.style))}</div></div>
      <div class="card kpi"><div class="label">Best mechanics</div><b>${f1(best)}</b><div class="sub">of ${nMech} scored session${nMech === 1 ? '' : 's'}</div></div>
      <div class="card kpi"><div class="label">Trend</div><b class="${nMech > 1 ? (last - first >= 0 ? 'pos' : 'neg') : ''}">${nMech > 1 ? (last - first >= 0 ? '+' : '') + (last - first).toFixed(1) : '—'}</b><div class="sub">mechanics, first → latest</div></div>
      <div class="card kpi"><div class="label">Consistency</div><b>${nMech > 1 ? '± ' + sd(mf).toFixed(1) : '—'}</b><div class="sub">mechanics sd${styles.length > 1 ? ` · style ± ${sd(styles).toFixed(1)}` : ''}</div></div>
    </div>
    <section class="sec"><div class="card"><div class="sec-h"><h2>Scores over time</h2></div>${sparkSVG(list, sc)}
      <div class="legend" style="justify-content:center;margin-top:4px"><span><i style="background:var(--accent)"></i>mechanics score</span><span><i class="dash"></i>style match</span></div>${trendsHTML(sum, slope)}</div>
      <p class="read-line"><b>Read:</b> ${nMech > 1 ? `${nMech} scored sessions: mechanics ${first.toFixed(1)} → ${last.toFixed(1)} (<span class="${last - first >= 0 ? 'pos' : 'neg'}">${last - first >= 0 ? '+' : ''}${(last - first).toFixed(1)}</span>), best ${best.toFixed(1)}.` : nMech === 1 ? 'One session with a mechanics score so far — film again after a week of drills to see your trend.' : 'No session has a mechanics score yet.'}${nOld ? ` ${nOld} older session${nOld === 1 ? ' was' : 's were'} analysed before mechanics checks existed and show${nOld === 1 ? 's' : ''} style match only (—).` : ''}</p></section>
    <section class="sec"><div class="sec-h"><h2>Sessions</h2><span class="read">tick two to compare</span></div>
      <div class="hist-list" id="hist-list">${list.slice().reverse().map(sessRow).join('')}</div>
      <div class="cmp-bar" id="cmp-bar" hidden><span id="cmp-txt" class="muted"></span><span style="display:flex;gap:8px"><button class="btn sm" id="cmp-del" type="button" style="color:var(--neg)">Delete</button><button class="btn sm primary" id="cmp-go" type="button">Compare</button></span></div></section>
    <details class="more" style="margin-top:18px"><summary><span>Data &amp; privacy</span></summary><div class="body"><p class="note" style="margin-top:0">Sessions live in this browser's IndexedDB: landmarks, results and a 160 px thumbnail with the skeleton drawn on it. No video is stored and nothing is sent anywhere. Export JSON to back up or move to another device; import merges by session id.</p>
      <button class="btn sm" id="h-clear" type="button" style="color:var(--neg)">Delete all sessions</button></div></details>`;
  wireHistoryTools(list);
  const sync = () => {
    $$('#hist-list .sess').forEach(el => el.classList.toggle('sel', state.compare.includes(el.dataset.id)));
    $$('#hist-list input[type=checkbox]').forEach(cb => { cb.checked = state.compare.includes(cb.value); });
    const n = state.compare.length; $('#cmp-bar').hidden = !n;
    $('#cmp-txt').textContent = n === 1 ? '1 selected — pick one more to compare' : n + ' selected';
    $('#cmp-go').disabled = n !== 2;
  };
  $$('#hist-list input[type=checkbox]').forEach(cb => cb.onchange = () => {
    const id = cb.value; state.compare = state.compare.filter(x => x !== id); if (cb.checked) state.compare.push(id);
    if (state.compare.length > 2) state.compare.shift(); sync();
  });
  $('#cmp-go').onclick = () => { const [a, b] = state.compare.map(id => list.find(s => s.id === id)).sort((x, y) => (x.created < y.created ? -1 : 1)); go(`#/compare/${a.id}/${b.id}`); };
  $('#cmp-del').onclick = async () => { const n = state.compare.length; if (!confirm(`Delete ${n} session${n > 1 ? 's' : ''}?`)) return; for (const id of state.compare) await DB.del(id); state.compare = []; toast('Deleted'); renderHistory(); };
  $('#h-clear').onclick = async () => { if (!confirm('Delete ALL sessions on this device? Export first if you want a backup.')) return; await DB.clear(); state.compare = []; toast('History cleared'); renderHistory(); };
  sync();
}
function primaryResult(s) { return (s.results && s.results[s.primary || 0]) || s.result; }
function sessRow(s) {
  const r = primaryResult(s) || {}; const nMaj = (r.feedback || []).filter(f => f.severity === 'major').length; const n = (s.results || []).length; const sc = sessScores(s);
  return `<div class="sess" data-id="${esc(s.id)}"><label class="ck" title="Select to compare"><input type="checkbox" value="${esc(s.id)}" aria-label="Select ${esc(s.name)} for comparison"></label>
    ${s.thumb ? `<img src="${esc(s.thumb)}" alt="">` : '<span class="noimg"></span>'}
    <div class="mid"><b>${esc(s.name || 'Shot')}</b><small>${esc(fmtDate(s.created))} · ${esc(fmtTime(s.created))}${n > 1 ? ` · ${n} shots` : ''}</small><small>style ${esc(f1(sc.style))} · ${esc(splitName(s.closest).title)}${nMaj ? ` · <span class="neg">${nMaj} major</span>` : ''}</small></div>
    <span class="sc num">${f1(sc.mech)}<small>${n > 1 ? 'mech mean' : 'mechanics'}</small></span>
    <a class="open" href="#/session/${esc(s.id)}" aria-label="Open ${esc(s.name)}"></a></div>`;
}
function sparkSVG(list, scs) {
  const W = 360, H = 150, L = 36, R = 334, T = 22, B = 122;
  scs = scs || list.map(sessScores);
  const sc = scs.map(x => x.mech), st = scs.map(x => x.style);
  const f = sc.concat(st).filter(Number.isFinite); if (!f.length) f.push(50);
  let lo = Math.floor(Math.min(...f) - 3), hi = Math.ceil(Math.max(...f) + 3); if (hi - lo < 10) { lo -= 5; hi += 5; } lo = Math.max(0, lo); hi = Math.min(100, hi);
  const x = i => (list.length === 1 ? (L + R) / 2 : L + i / (list.length - 1) * (R - L)), y = v => B - (v - lo) / ((hi - lo) || 1) * (B - T);
  const pts = sc.map((v, i) => [x(i), y(v)]).filter(p => Number.isFinite(p[1]));
  let s = `<svg class="chart spark" viewBox="0 0 ${W} ${H}" role="img" aria-label="Mechanics score and style match over sessions">`;
  [lo, hi].forEach(v => { s += `<line class="grid" x1="${L}" x2="${R}" y1="${y(v)}" y2="${y(v)}"/><text x="${L - 6}" y="${y(v) + 4}" text-anchor="end">${v}</text>`; });
  const spts = st.map((v, i) => [x(i), y(v)]).filter(p => Number.isFinite(p[1]));
  if (spts.length > 1) s += `<polyline points="${spts.map(p => p.join(',')).join(' ')}" fill="none" stroke="${COL.muted}" stroke-width="1.5" stroke-dasharray="4 4" stroke-linejoin="round"/>`;
  spts.forEach(p => { s += `<circle cx="${p[0]}" cy="${p[1]}" r="2.5" fill="${COL.muted}"/>`; });
  if (pts.length > 1) { s += `<polygon points="${pts[0][0]},${B} ${pts.map(p => p.join(',')).join(' ')} ${pts[pts.length - 1][0]},${B}" fill="rgba(255,160,48,.10)"/><polyline points="${pts.map(p => p.join(',')).join(' ')}" fill="none" stroke="${COL.accent}" stroke-width="2" stroke-linejoin="round"/>`; }
  const step = Math.ceil(list.length / 7);
  list.forEach((ss, i) => { const v = sc[i]; const last = i === list.length - 1;
    if (!Number.isFinite(v)) { if (i % step === 0 || last) s += `<text x="${x(i)}" y="142" text-anchor="middle">${esc(fmtDate(ss.created, { month: 'numeric', day: 'numeric' }))}</text>`; return; }
    s += `<a href="#/session/${esc(ss.id)}"><circle cx="${x(i)}" cy="${y(v)}" r="${last ? 5 : 3.5}" fill="${last ? COL.accent : '#181c25'}" stroke="${COL.accent}" stroke-width="2"><title>${esc(ss.name)} · ${esc(fmtDate(ss.created))} · mechanics ${v.toFixed(1)} · style ${f1(st[i])}</title></circle></a>`;
    if (list.length <= 12 || last || i % step === 0) s += `<text x="${x(i)}" y="${y(v) - 10}" text-anchor="middle" style="fill:${last ? COL.text : COL.muted};font-weight:600">${Math.round(v)}</text>`;
    if (i % step === 0 || last) s += `<text x="${x(i)}" y="142" text-anchor="middle">${esc(fmtDate(ss.created, { month: 'numeric', day: 'numeric' }))}</text>`; });
  return s + '</svg>';
}
function trendsHTML(sum, slope) {
  if (!sum) return '';
  const short = l => String(l || '').replace(/ \(.*\)$/, '');
  const arr = x => (Array.isArray(x) ? x : []);
  const ul = (xs, cls, sym) => xs.length ? `<ul>${xs.slice(0, 3).map(it => `<li><span>${esc(short(it.label || it.key))}</span><span class="num ${cls}">${sym} ${fin(it.value) ? Math.abs(num(it.value)).toFixed(1) + '×' : ''}</span></li>`).join('')}</ul>` : '<p class="note" style="margin:6px 0 0">— needs 2+ sessions</p>';
  const groups = [['Most improved', arr(sum.mostImproved), 'pos', '▲'], ['Trending worse', arr(sum.trendingWorse), 'neg', '▼'], ['Best now', arr(sum.best), 'pos', '●'], ['Worst now', arr(sum.worst), 'neg', '●']];
  if (!groups.some(g => g[1].length)) return '';
  return `<div class="trend">${groups.map(([h, xs, c, sy]) => `<div><div class="lbl">${h}</div>${ul(xs, c, sy)}</div>`).join('')}</div>
    <p class="note">Style metrics: |you − closest reference| in tolerance units (1.0× = the "ok" threshold), first session vs latest${fin(slope) ? ` · mechanics slope ${slope > 0 ? '+' : ''}${num(slope).toFixed(2)} pts/session` : ''}.</p>`;
}
function wireHistoryTools(list) {
  const j = $('#h-json'), c = $('#h-csv'), i = $('#h-imp');
  if (j) j.onclick = () => exportJSON(list);
  if (c) c.onclick = () => exportCSV(list);
  if (i) i.onclick = () => $('#file-import').click();
}
function download(name, text, type) {
  const blob = text instanceof Blob ? text : new Blob([text], { type }); const url = URL.createObjectURL(blob);
  const a = document.createElement('a'); a.href = url; a.download = name; document.body.appendChild(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(url), 4000);
}
const r5 = x => (Number.isFinite(x) ? Math.round(x * 1e5) / 1e5 : null);
function exportJSON(list) {
  const txt = JSON.stringify({ app: 'jumpshot-analyzer', version: 1, exported: new Date().toISOString(), sessions: list }, (k, v) => (ArrayBuffer.isView(v) ? Array.from(v, r5) : typeof v === 'number' && !Number.isFinite(v) ? null : v));
  download(`jumpshot-history-${new Date().toISOString().slice(0, 10)}.json`, txt, 'application/json');
  toast(`Exported ${list.length} session${list.length === 1 ? '' : 's'}`);
}
function exportCSV(list) {
  const keys = new Set(); list.forEach(s => Object.keys((primaryResult(s) || {}).summary || {}).forEach(k => keys.add(k)));
  const ks = Array.from(keys);
  const head = ['id', 'date', 'name', 'source', 'mechanics_score', 'style_match', 'score', 'shots', 'closest', 'closest_score', 'hand', 'quality', 'major', 'minor', 'model', ...ks];
  const q = v => { const s = v == null ? '' : String(v); return /[",\n]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s; };
  const rows = list.map(s => { const r = primaryResult(s) || {}; const fb = r.feedback || [];
    const sc = sessScores(s);
    return [s.id, s.created, s.name, s.source, fin(sc.mech) ? sc.mech.toFixed(2) : '', fin(sc.style) ? sc.style.toFixed(2) : '', fin(s.score) ? num(s.score).toFixed(2) : '', (s.results || []).length, s.closest, r.closest && fin(r.closest.score) ? num(r.closest.score).toFixed(2) : '', r.hand, r.quality && r.quality.level,
      fb.filter(f => f.severity === 'major').length, fb.filter(f => f.severity === 'minor').length, s.model || '', ...ks.map(k => (fin(r.summary[k]) ? num(r.summary[k]).toFixed(4) : ''))].map(q).join(','); });
  download(`jumpshot-history-${new Date().toISOString().slice(0, 10)}.csv`, [head.join(','), ...rows].join('\n'), 'text/csv');
  toast('CSV exported');
}
$('#file-import').addEventListener('change', async e => {
  const f = e.target.files[0]; e.target.value = ''; if (!f) return;
  try {
    const d = JSON.parse(await f.text()); const ss = Array.isArray(d) ? d : d.sessions;
    if (!Array.isArray(ss)) throw new Error('No "sessions" array in this file.');
    let n = 0;
    for (const s of ss) {
      if (!s || !s.id || !s.created || !(s.results || s.result)) continue;
      if (Array.isArray(s.frames)) s.frames = s.frames.map(fr => ({ t: num(fr.t), lm: fr.lm ? Float32Array.from(fr.lm, v => (v == null ? NaN : v)) : null }));
      await DB.put(s); n++;
    }
    toast(`Imported ${n} session${n === 1 ? '' : 's'}`); refreshCount();
    if (current === 'history') renderHistory(); else go('#/history');
  } catch (err) { showError({ icon: 'warn', title: 'Import failed', msg: 'That file is not a Jumpshot Analyzer export.', fixes: ['Use a file created with “Export JSON” on the History page.'], actions: [{ label: 'Back to history', primary: true, go: '#/history' }], code: String(err.message) }); }
});

// ------------------------------------------------------------------ compare
async function renderCompare(ida, idb) {
  const box = $('#compare-body');
  const [A, B] = await Promise.all([DB.get(ida), DB.get(idb)]);
  if (!A || !B) { toast('Session not found'); return go('#/history', true); }
  const ra = primaryResult(A), rb = primaryResult(B);
  const eng = await getEngine(); const FEATURES = eng.FEATURES || [];
  const kk = ['load', 'relh', 'elbow', 'knee', 'jump', 'hold', 'elbowRel', 'lean'];
  const trs = [];
  const sgn = (v, unit) => fmtVal(v, unit, true);
  const multi = (A.results || []).length > 1 || (B.results || []).length > 1 ? ' <span class="muted">(shot 1)</span>' : '';
  const ma = mechScore(ra), mb = mechScore(rb), sa = styleScore(ra), sb = styleScore(rb);
  const dRow = (va, vb) => (fin(va) && fin(vb) ? `<td class="n ${vb >= va ? 'pos' : 'neg'}">${vb - va >= 0 ? '+' : MINUS}${Math.abs(vb - va).toFixed(1)}</td>` : '<td class="n muted">—</td>');
  trs.push(`<tr><td>Mechanics score${multi}</td><td class="n"><b>${f1(ma)}</b></td><td class="n"><b>${f1(mb)}</b></td>${dRow(ma, mb)}</tr>`);
  trs.push(`<tr><td>Style match${multi}</td><td class="n">${f1(sa)}</td><td class="n">${f1(sb)}</td>${dRow(sa, sb)}</tr>`);
  if (hasMech(ra) || hasMech(rb)) {
    const keys = (hasMech(rb) ? rb.mechanics : ra.mechanics).map(m => m.key);
    const get = (r, k) => (hasMech(r) ? r.mechanics.find(m => m.key === k) : null);
    const cell = m => (m && m.status !== 'unmeasured' ? `<span class="${MSTATUS[m.status] || ''}">${fmtPts(m.points)}</span><span class="muted">/${fmtPts(m.maxPoints)}</span>` : '<span class="muted">—</span>');
    trs.push(`<tr class="grp"><td colspan="4">Mechanics checks · points</td></tr>`);
    for (const k of keys) {
      const a = get(ra, k), b = get(rb, k), lab = (b || a).label;
      const pa = a && a.status !== 'unmeasured' ? num(a.points) : NaN, pb = b && b.status !== 'unmeasured' ? num(b.points) : NaN;
      const d = fin(pa) && fin(pb) ? pb - pa : NaN;
      trs.push(`<tr><td>${esc(lab)}</td><td class="n">${cell(a)}</td><td class="n">${cell(b)}</td><td class="n ${fin(d) ? (d > 0.05 ? 'pos' : d < -0.05 ? 'neg' : 'muted') : 'muted'}">${fin(d) ? (Math.abs(d) < 0.05 ? '0' : (d > 0 ? '+' : MINUS) + fmtPts(Math.abs(d))) : '—'}</td></tr>`);
    }
    trs.push(`<tr class="grp"><td colspan="4">Style metrics</td></tr>`);
  }
  trs.push(`<tr><td>Closest ref</td><td class="n">${esc(splitName(A.closest).title.replace('Reference ', 'Ref '))}</td><td class="n">${esc(splitName(B.closest).title.replace('Reference ', 'Ref '))}</td><td></td></tr>`);
  for (const k of kk) {
    const va = metric(ra, k), vb = metric(rb, k); if (!fin(va) && !fin(vb)) continue;
    const fa = fbOf(ra, k), fbb = fbOf(rb, k);
    let cls = '';
    if (fa && fbb && fin(fa.delta) && fin(fbb.delta)) cls = Math.abs(fbb.delta) < Math.abs(fa.delta) - 1e-9 ? 'pos' : Math.abs(fbb.delta) > Math.abs(fa.delta) + 1e-9 ? 'neg' : 'muted';
    const u = MUNIT[k] === 'torso' ? '' : MUNIT[k];
    trs.push(`<tr><td>${MLABEL[k]}${MUNIT[k] === 'torso' ? ' <span class="muted">(torso)</span>' : ''}</td><td class="n">${fmtVal(va, u)}</td><td class="n">${fmtVal(vb, u)}</td><td class="n ${cls}">${sgn(vb - va, u)}</td></tr>`);
  }
  const cnt = (r, s) => (r.feedback || []).filter(f => f.severity === s).length;
  trs.push(`<tr><td>Major / minor</td><td class="n">${cnt(ra, 'major')} / ${cnt(ra, 'minor')}</td><td class="n">${cnt(rb, 'major')} / ${cnt(rb, 'minor')}</td><td></td></tr>`);
  const head = (s, l) => `<div class="card cmp-head">${s.thumb ? `<img src="${esc(s.thumb)}" alt="">` : ''}<div style="min-width:0"><span class="chip ${l === 'B' ? 'accent' : ''}">${l}</span> <b style="display:inline">${esc(s.name)}</b><small style="display:block">${esc(fmtDate(s.created))} · ${esc(fmtTime(s.created))}</small></div></div>`;
  const radar = ra.radar && rb.radar && ra.radar.axes && rb.radar.axes ? radarSVG({ axes: rb.radar.axes, a: ra.radar.user, b: rb.radar.user, ref: rb.radar.ref }, [{ k: 'ref', col: COL.faint, dash: '3 4', w: 1.2 }, { k: 'a', col: COL.muted, fill: 'rgba(138,147,166,.10)', w: 1.6 }, { k: 'b', col: COL.accent, fill: 'rgba(255,160,48,.14)', w: 2, dots: true }]) : '';
  const usesMech = fin(ma) && fin(mb); const d = usesMech ? mb - ma : sb - sa;
  box.innerHTML = `<div class="view-h"><button class="btn ghost back" data-back type="button" aria-label="Back"><svg viewBox="0 0 24 24"><path d="m15 6-6 6 6 6"/></svg></button><div><div class="eyebrow">Compare</div><h2 id="h-cmp">A → B</h2></div></div>
    <div class="cmp-grid">${head(A, 'A')}${head(B, 'B')}</div>
    <div class="grid g2" style="margin-top:12px">
      <div class="tbl cmp-tbl" style="mask-image:none;-webkit-mask-image:none"><table style="min-width:0"><thead><tr><th>Metric</th><th class="n">A</th><th class="n">B</th><th class="n">Δ B−A</th></tr></thead><tbody>${trs.join('')}</tbody></table></div>
      <div class="card radar">${radar}<div class="legend" style="margin-top:6px"><span><i style="background:var(--muted)"></i>A</span><span><i style="background:var(--accent)"></i>B</span><span><i class="dash" style="border-color:var(--faint)"></i>B's reference</span></div></div>
    </div>
    <p class="read-line"><b>Read:</b> ${fin(d) ? `${usesMech ? 'Mechanics score' : 'Style match'} ${d >= 0 ? 'up' : 'down'} <b class="${d >= 0 ? 'pos' : 'neg'}">${Math.abs(d).toFixed(1)}</b> from A to B${usesMech ? '' : ' (mechanics score not available for both sessions — older sessions show —)'}.` : 'Scores are not available for both sessions.'} Mechanics-check deltas are points earned; for style metrics, green means B is closer to its reference than A was and red means it drifted away.</p>
    <div class="row-between" style="margin-top:14px"><a class="btn" href="#/session/${esc(A.id)}">Open A</a><a class="btn primary" href="#/session/${esc(B.id)}">Open B</a></div>`;
}

// ------------------------------------------------------------------ errors
const ICON = { err: '<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M12 7v6M12 16.5h.01"/></svg>', warn: '<svg viewBox="0 0 24 24"><path d="M12 3 2 20h20L12 3z"/><path d="M12 10v4M12 17h.01"/></svg>' };
function showError(e) { state.error = e; go('#/error'); if (current === 'error') renderError(); }
function renderError() {
  const e = state.error; const box = $('#error-body');
  box.innerHTML = `<div class="eic ${e.icon === 'warn' ? 'warn' : ''}">${ICON[e.icon === 'warn' ? 'warn' : 'err']}</div><h2 id="h-err">${esc(e.title)}</h2><p>${esc(e.msg || '')}</p>
    ${e.fixes && e.fixes.length ? `<ul>${e.fixes.map(f => `<li>${f}</li>`).join('')}</ul>` : ''}
    <div class="acts">${(e.actions || [{ label: 'Start over', primary: true, go: '#/' }]).map((a, i) => `<button class="btn ${a.primary ? 'primary' : ''}" type="button" data-ea="${i}">${esc(a.label)}</button>`).join('')}</div>
    ${e.code ? `<code>${esc(e.code)}</code>` : ''}`;
  $$('[data-ea]', box).forEach(b => b.onclick = () => { const a = e.actions[+b.dataset.ea]; if (a.click) a.click(); if (a.go) go(a.go); });
}
const iosHint = /iPhone|iPad|iPod/.test(navigator.userAgent);
function errCodec(e) {
  return { title: "This video can't be played in this browser", msg: 'The file opened, but the browser could not decode its video track. This usually means HEVC (H.265) video — the iPhone default “High Efficiency” format — on a browser without HEVC support.',
    fixes: [iosHint ? 'Open this page in <b>Safari</b>, which plays iPhone video natively.' : 'Open this page in <b>Safari</b> (Mac/iPhone) or a browser with HEVC support, <b>or</b>',
      'On iPhone: <b>Settings → Camera → Formats → Most Compatible</b>, then re-record (H.264 plays everywhere).', 'Or export/share the clip as H.264 MP4 from your editor.', 'Or use <b>Record now</b> to film directly in the browser.'],
    actions: [{ label: 'Choose another clip', primary: true, go: '#/', click: () => setTimeout(() => $('#file-upload').click(), 50) }, { label: 'Record instead', click: () => $('#btn-record').click() }], code: e && e.message };
}
function errNoPerson(e) {
  return { icon: 'warn', title: 'No person detected', msg: `The pose model found a body in only ${e.nPose} of ${e.n} frames.`,
    fixes: ['Make sure your <b>whole body</b> is in the frame — feet to raised hands.', 'Film in <b>good light</b>; backlight (a window behind you) hides the body.', 'Stand <b>3–5 m</b> from the camera, side-on, with nobody else in the frame.', 'Check the trim covers the part of the clip with you in it.'],
    actions: [{ label: 'Adjust trim', primary: true, go: '#/clip' }, { label: 'Try the Accurate model', click: () => { prefs.model = 'full'; savePrefs(); go('#/clip'); } }, { label: 'New clip', go: '#/' }] };
}
function errSmall(e) {
  return { icon: 'warn', title: "You're too small in the frame", msg: `Your body fills about ${Math.round(e.h * 100)}% of the frame height — joint angles get noisy below ~25%.`,
    fixes: ['Move the camera <b>closer</b> (3–5 m) or zoom in 2× before recording.', 'Keep your whole body plus the ball at release in frame.', 'Or crop the video around yourself before uploading.'],
    actions: [{ label: 'Analyze anyway', primary: true, click: () => { state._ignoreSmall = true; go('#/clip'); setTimeout(() => $('#btn-analyze').click(), 300); } }, { label: 'New clip', go: '#/' }] };
}
function errModel(e) {
  if (e.code === 'simd') return { title: 'This browser is too old for the AI model', msg: 'The pose model needs WebAssembly SIMD (Safari 16.4+, Chrome 91+, Firefox 89+).', fixes: ['Update iOS / your browser and try again.', 'Or use “Try a sample” to see what the analysis looks like.'], actions: [{ label: 'Try a sample', primary: true, click: runSample }, { label: 'Home', go: '#/' }], code: e.message };
  if (!navigator.onLine) {
    const have = safeJSONArr(localStorage.getItem('jumpshot.models')).filter(m => m !== prefs.model);
    const name = m => (m === 'full' ? 'Accurate' : 'Fast');
    return { icon: 'warn', title: "You're offline", msg: `The ${name(prefs.model)} model hasn't been downloaded on this device yet, so it can't run without a connection.`,
      fixes: have.length ? [`The <b>${name(have[0])}</b> model is already on this device and works offline.`, 'Or reconnect once — after that, this model works offline too.'] : ['Connect once to download the model (≈20 MB); after that the analyzer works fully offline.'],
      actions: (have.length ? [{ label: `Use ${name(have[0])} model`, primary: true, click: () => { prefs.model = have[0]; savePrefs(); go('#/clip'); setTimeout(() => $('#btn-analyze').click(), 300); } }] : []).concat([{ label: 'Try the sample', primary: !have.length, click: runSample }]) };
  }
  return { title: "The AI model didn't load", msg: 'The pose model (≈20 MB, downloaded once) could not be loaded or started.',
    fixes: ['Check your connection — the first analysis needs to download the model; after that it works offline.', 'Close other heavy tabs (the model needs ~150 MB of memory) and retry.', 'If you are in a private/incognito window, storage may be restricted — try a normal window.'],
    actions: [{ label: 'Retry', primary: true, click: () => { go('#/clip'); setTimeout(() => $('#btn-analyze').click(), 300); } }, { label: 'Try a sample', click: runSample }], code: e.message + (e.cause ? ' — ' + (e.cause.message || e.cause) : '') };
}
function errAnalysis(r) {
  const flags = (r && r.quality && r.quality.flags) || [];
  return { icon: 'warn', title: "We couldn't measure this shot", msg: (r && r.reason) || 'The analysis engine could not find a complete shot motion.',
    fixes: flags.map(f => esc(f.text || f.message || f.code)).concat(['Trim to one shot: from just before the dip to after the follow-through.', 'Film side-on with the whole body and the ball in frame.']),
    actions: [{ label: 'Adjust trim', primary: true, go: state.source === 'sample' ? '#/' : '#/clip' }, { label: 'New clip', go: '#/' }] };
}
function errUnknown(e) { return { title: 'Something went wrong', msg: 'An unexpected error stopped the analysis.', fixes: ['Try again, or with a shorter trim.', 'Reload the page if it keeps happening.'], actions: [{ label: 'Back to clip', primary: true, go: state.file ? '#/clip' : '#/' }, { label: 'Home', go: '#/' }], code: e && (e.stack || e.message || String(e)) }; }

// ------------------------------------------------------------------ boot
document.documentElement.classList.toggle('touch', COARSE);
if (COARSE) { const st = document.createElement('style'); st.textContent = '.hide-touch{display:none}'; document.head.appendChild(st); }
refreshCount();
route();
// warm the engine + assets in the background (tiny; makes the first analysis snappier)
setTimeout(() => { getAssets().catch(() => {}); }, 1200);
