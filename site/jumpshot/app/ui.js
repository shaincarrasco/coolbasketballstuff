// Jumpshot — in-browser jump-shot analyzer. Video never leaves the device.
// Views: home, record, clip, processing, shots, results, history, error (hash router). Compare and details open as sheets.
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
/** Live design tokens (light/dark can change at runtime). */
function C() {
  const cs = getComputedStyle(document.documentElement); const g = (n, d) => (cs.getPropertyValue(n) || '').trim() || d;
  return { brand: g('--brand', '#e8590c'), pos: g('--pos', '#1f9d55'), warn: g('--warn', '#c77700'), neg: g('--neg', '#e5484d'), label: g('--label', '#0a0a0f'), label2: g('--label-2', '#666'), label3: g('--label-3', '#999'), sep: g('--separator', '#ccc') };
}
const STAGE_GHOST = 'rgba(235,235,245,.62)';   // media stage is always dark: dark-mode tertiary label
const STAGE_BG = '#1c1c1e';
const MAX_WINDOW = 15;
const ICONS = {
  chev: '<svg class="chev" viewBox="0 0 8 13" aria-hidden="true"><path d="M1.5 1.5 6.5 6.5 1.5 11.5" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  play: '<svg viewBox="0 0 24 24"><path d="M8 5.5v13l10.5-6.5z"/></svg>',
  pause: '<svg viewBox="0 0 24 24"><rect x="6.5" y="5.5" width="4" height="13" rx="1"/><rect x="13.5" y="5.5" width="4" height="13" rx="1"/></svg>',
  prev: '<svg viewBox="0 0 24 24"><path d="m14.5 6-6 6 6 6"/></svg>',
  next: '<svg viewBox="0 0 24 24"><path d="m9.5 6 6 6-6 6"/></svg>',
  check: '<svg viewBox="0 0 24 24"><path d="m6 12.5 4 4 8-9"/></svg>',
  history: '<svg viewBox="0 0 24 24"><path d="M3.5 12a8.5 8.5 0 1 0 2.6-6.1L3.5 8.5"/><path d="M3.5 3.5v5h5M12 7.5V12l3 2"/></svg>',
  warn: '<svg viewBox="0 0 24 24"><path d="M12 4 2.8 19.5h18.4z"/><path d="M12 10v4.5M12 17.3v.2"/></svg>',
  film: '<svg viewBox="0 0 24 24"><rect x="3.5" y="5" width="17" height="14" rx="2.5"/><path d="M3.5 9h17M3.5 15h17M8 5v4M8 15v4M16 5v4M16 15v4"/></svg>',
  person: '<svg viewBox="0 0 24 24"><circle cx="12" cy="6.5" r="2.5"/><path d="M12 9.5v6M8 21l4-5.5 4 5.5M7 12.5h10"/></svg>',
  offline: '<svg viewBox="0 0 24 24"><path d="M3 3l18 18M8.5 16.5a5 5 0 0 1 7 0M5 12.8a10 10 0 0 1 4.2-2.3M19 12.8a10 10 0 0 0-3-1.9M2 9.2a15 15 0 0 1 4.3-2.6M22 9.2A15 15 0 0 0 10.8 5.1M12 20h.01"/></svg>',
};

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
  sessionVideo: null,                              // { id, url } when the session's video is in memory
  error: null, compare: [], editing: false,
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
function toast(msg) {
  if (window.App && window.App.toast) return window.App.toast(msg);
  console.info('[jumpshot]', msg);
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

const MINUS = '−';
const fmtS = (v, d = 2) => (fin(v) ? num(v).toFixed(d) + ' s' : '—');
const fmtDate = (iso, opts) => { try { return new Date(iso).toLocaleDateString(undefined, opts || { month: 'short', day: 'numeric', year: 'numeric' }); } catch (e) { return String(iso).slice(0, 10); } };
const fmtTime = iso => { try { return new Date(iso).toLocaleTimeString(undefined, { hour: 'numeric', minute: '2-digit' }); } catch (e) { return ''; } };
const fmtWhen = iso => { const d = new Date(iso), now = new Date(); const same = d.toDateString() === now.toDateString(); const y = new Date(now - 864e5).toDateString() === d.toDateString(); return (same ? 'Today' : y ? 'Yesterday' : fmtDate(iso)) + ' at ' + fmtTime(iso); };
const fmtBytes = b => (b > 1e6 ? (b / 1e6).toFixed(1) + ' MB' : Math.round(b / 1e3) + ' KB');
function fmtVal(v, unit, signed) {
  if (!fin(v)) return '—';
  v = num(v); const sg = signed && v > 0 ? '+' : v < 0 ? MINUS : '';
  const a = signed ? Math.abs(v) : v;
  if (unit === 'deg') return sg + (signed ? a : v).toFixed(0) + '°';
  if (unit === 's') return sg + (signed ? a : v).toFixed(2) + ' s';
  if (unit === 'torso') return sg + (signed ? a : v).toFixed(2) + ' torso';
  if (unit === '%') return sg + (signed ? a : v).toFixed(0) + '%';
  return sg + (signed ? a : v).toFixed(2);
}
const fmtNum = (v, unit) => { if (!fin(v)) return '—'; v = num(v); return unit === 'deg' ? v.toFixed(0) + '°' : v.toFixed(2); };
const splitName = n => { const s = String(n || 'Reference').split(/\s+[—–-]\s+/); return { title: s[0], style: s.slice(1).join(' — ') }; };
const mean = a => a.reduce((x, y) => x + y, 0) / (a.length || 1);
const sd = a => { if (a.length < 2) return 0; const m = mean(a); return Math.sqrt(a.reduce((x, y) => x + (y - m) * (y - m), 0) / (a.length - 1)); };
const cap1 = s => String(s || '').replace(/^./, c => c.toUpperCase());
const rx = s => String(s || '').replace(/(\d)\s*x\s*(\d)/g, '$1 × $2');

// summary metric lookups (tolerant to key naming between engine versions)
const MKEYS = {
  load: ['release_time_s', 'load_to_release_s', 'load_release_s'], relh: ['release_height'], elbow: ['elbow_angle_set', 'elbow_at_set'],
  knee: ['knee_angle_min', 'knee_min'], jump: ['jump_height'], hold: ['follow_through_hold_s', 'hold_s'], elbowRel: ['elbow_angle_release'],
  fwd: ['release_forward'], lean: ['trunk_lean_release'],
};
const MUNIT = { load: 's', relh: 'torso', elbow: 'deg', knee: 'deg', jump: 'torso', hold: 's', elbowRel: 'deg', fwd: 'torso', lean: 'deg' };
const MLABEL = { load: 'Load to release', relh: 'Release height', elbow: 'Elbow at set', knee: 'Knee bend', jump: 'Jump height', hold: 'Follow-through hold', elbowRel: 'Elbow at release', fwd: 'Release forward', lean: 'Trunk lean at release' };
function metric(r, k) { const s = (r && r.summary) || {}; for (const key of MKEYS[k]) if (fin(s[key])) return num(s[key]); return NaN; }
function fbOf(r, k) { return ((r && r.feedback) || []).find(f => MKEYS[k].includes(f.key)); }
const fbRef = f => num(f.ref != null ? f.ref : f.reference);

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
const STATUS_CLS = { good: 'pos', close: 'warn', off: 'neg', unmeasured: 'mute' };
const STATUS_TXT = { good: 'In range', close: 'Close', off: 'Off', unmeasured: 'Not measured' };
const gainOf = m => (m && m.status !== 'unmeasured' && fin(m.points) && fin(m.maxPoints) ? Math.max(0, num(m.maxPoints) - num(m.points)) : 0);
const fmtPts = v => (fin(v) ? (Math.abs(num(v) - Math.round(num(v))) < 0.05 ? String(Math.round(num(v))) : num(v).toFixed(1)) : '—');
function topFixes(r, n = 2) { return (hasMech(r) ? r.mechanics : []).filter(m => (m.status === 'off' || m.status === 'close') && gainOf(m) > 0).sort((a, b) => gainOf(b) - gainOf(a)).slice(0, n); }
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
  if (unit === 's') { if (lo >= 0) return `${tn(lo)}–${tn(hi)} s`; const f = x => (x < 0 ? MINUS : '') + tn(Math.abs(x)); return `${f(lo)} to ${f(hi)} s`; }
  if (unit === 'torso') return `${tn(lo)}–${tn(hi)} torso`;
  return `${lo}–${hi}`;
}

// ---- plain-language copy for each mechanics check
const SHORT = { set_elbow: 'Elbow at set', dip_knee: 'Knee dip', release_elbow: 'Elbow at release', release_shoulder: 'Arm angle at release', release_timing: 'Release timing', follow_through: 'Follow-through', balance_lean: 'Trunk lean', balance_drift: 'Hip drift' };
const shortLabel = m => SHORT[m.key] || m.label || m.key;
const WHAT = {
  set_elbow: 'The angle of your shooting elbow at the set point, when the ball is loaded just before you push up.',
  dip_knee: 'How far your knees bend at the bottom of the dip. Your legs start the shot.',
  release_elbow: 'How straight your shooting arm is when the ball leaves your hand.',
  release_shoulder: 'How high your upper arm is raised at release, measured from your body.',
  release_timing: 'When the ball leaves your hand compared with the top of your jump. Negative means before the top.',
  follow_through: 'How long you hold your shooting arm up after the ball leaves your hand.',
  balance_lean: 'How much your upper body tilts between the set point and release.',
  balance_drift: 'How far your hips travel forward or back while you are in the air.',
};
const isHigh = m => fin(m.value) && m.band && num(m.value) > num(m.band[1]);
/** [headline instruction, why it matters, verdict clause] for a check outside its range. */
function fixCopy(m) {
  const up = isHigh(m);
  const T = {
    set_elbow: up ? ['Fold your elbow more at the set', 'Your elbow is more open than the target when the ball is loaded, so you lose some push.', 'your elbow is too open at the set']
      : ['Open your elbow at the set', 'Your elbow is folded tighter than the target at the set point.', 'your elbow is too tight at the set'],
    dip_knee: up ? ['Sink deeper in your dip', 'Your knees barely bend before you lift, so your legs add little power.', 'your dip is too shallow']
      : ['Ease off your dip', 'You sink deeper than you need to, which slows the shot down.', 'your dip is too deep'],
    release_elbow: up ? ['Soften your elbow at release', 'Your elbow locks out hard as the ball leaves.', 'your arm locks out at release']
      : ['Extend fully at release', "Your arm isn't fully straight when the ball leaves your hand.", "you don't extend fully at release"],
    release_shoulder: up ? ['Keep the ball in front at release', 'The ball goes up from behind your head instead of in front of it.', 'your release is behind your head']
      : ['Finish higher at release', 'Your arm finishes low, which flattens the arc.', 'your release point is low'],
    release_timing: up ? ['Release at the top of your jump', 'The ball leaves your hand on the way down, after the top of your jump.', 'you release late']
      : ['Release at the top of your jump', 'The ball leaves your hand on the way up, before you reach the top of your jump.', 'you release before the top of your jump'],
    follow_through: up ? ['Relax your follow-through sooner', 'You hold the follow-through much longer than needed.', 'you hold the follow-through too long']
      : ['Hold your follow-through', 'Your shooting arm drops quickly after release.', 'your follow-through is short'],
    balance_lean: ['Keep your upper body steady', 'Your upper body tilts between the set point and release.', 'your upper body tilts'],
    balance_drift: ['Jump straight up', 'Your hips travel forward or back while you are in the air.', 'you drift in the air'],
  }[m.key];
  return T || [`Bring ${String(shortLabel(m)).toLowerCase()} into range`, m.text || '', `${String(shortLabel(m)).toLowerCase()} is out of range`];
}
function verdict(r) {
  const ms = mechScore(r);
  if (!hasMech(r)) return 'Analysed before mechanics checks existed';
  if (!fin(ms)) return 'Not enough of your body was visible to score this shot';
  const base = ms >= 85 ? 'Excellent mechanics' : ms >= 70 ? 'Solid base' : ms >= 55 ? 'Good foundation' : 'Plenty of room to grow';
  const top = topFixes(r, 1)[0];
  return top ? `${base}, but ${fixCopy(top)[2]}` : `${base}. Every measured check is in range`;
}
function mechSources(assets) {
  const list = (assets && assets.sources && assets.sources.length ? assets.sources : (assets && assets.library && assets.library.mechanics && assets.library.mechanics.sources)) || [];
  return { list, n: Object.fromEntries(list.map((s, i) => [s.id, i + 1])) };
}

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

// ------------------------------------------------------------------ sheets (foundation .sheet + .scrim)
let sheetEl = null, scrimEl = null, sheetFocus = null;
function sheet(title, html) {
  if (!sheetEl) {
    scrimEl = document.createElement('div'); scrimEl.className = 'scrim'; scrimEl.onclick = closeSheet;
    sheetEl = document.createElement('div'); sheetEl.className = 'sheet'; sheetEl.setAttribute('role', 'dialog'); sheetEl.setAttribute('aria-modal', 'true');
    document.body.append(scrimEl, sheetEl);
    document.addEventListener('keydown', e => { if (e.key === 'Escape' && sheetEl.classList.contains('open')) closeSheet(); });
    // drag down on the grabber / header to dismiss (mobile)
    let y0 = null, dy = 0;
    sheetEl.addEventListener('pointerdown', e => { if (!e.target.closest('.grabber,.sheet-hd') || e.target.closest('button') || innerWidth >= 900) return; y0 = e.clientY; dy = 0; sheetEl.style.transition = 'none'; sheetEl.setPointerCapture(e.pointerId); });
    sheetEl.addEventListener('pointermove', e => { if (y0 == null) return; dy = Math.max(0, e.clientY - y0); sheetEl.style.transform = `translateY(${dy}px)`; });
    const end = () => { if (y0 == null) return; y0 = null; sheetEl.style.transition = ''; sheetEl.style.transform = ''; if (dy > 90) closeSheet(); };
    sheetEl.addEventListener('pointerup', end); sheetEl.addEventListener('pointercancel', end);
  }
  sheetFocus = document.activeElement;
  sheetEl.setAttribute('aria-label', title);
  sheetEl.innerHTML = `<div class="grabber"></div><div class="sheet-hd"><h3>${esc(title)}</h3><button class="btn plain" type="button" data-close>Done</button></div><div class="sheet-bd">${html}</div>`;
  sheetEl.querySelector('[data-close]').onclick = closeSheet;
  sheetEl.scrollTop = 0;
  void sheetEl.offsetHeight;
  sheetEl.classList.add('open'); scrimEl.classList.add('open');
  setTimeout(() => { const b = sheetEl.querySelector('[data-close]'); if (b) b.focus({ preventScroll: true }); }, 60);
  return sheetEl;
}
function closeSheet() {
  if (!sheetEl || !sheetEl.classList.contains('open')) return;
  sheetEl.classList.remove('open'); scrimEl.classList.remove('open');
  if (sheetFocus && sheetFocus.focus) try { sheetFocus.focus({ preventScroll: true }); } catch (e) {}
}
window.__jumpshot.closeSheet = closeSheet;

// ------------------------------------------------------------------ router
const VIEWS = ['home', 'record', 'clip', 'processing', 'shots', 'results', 'history', 'error'];
const TITLES = { home: 'Jumpshot', history: 'Jumpshot', record: 'Record', clip: 'Trim', processing: 'Analyzing', shots: 'Pick a shot', results: 'Your shot', error: 'Jumpshot' };
let current = null, leaveHooks = [];
function go(h, replace) { if (replace) history.replaceState(null, '', h); else if (location.hash !== h) { location.hash = h; return; } route(); }
function onLeave(fn) { leaveHooks.push(fn); }
function chrome(view) {
  const tabbed = view === 'home' || view === 'history';
  $('#js-head').hidden = !tabbed;
  $$('.js-seg [data-nav]').forEach(b => { const on = b.dataset.nav === view; b.classList.toggle('on', on); b.setAttribute('aria-selected', on); });
  const t = $('.ap-top .ap-title'); if (t) t.textContent = TITLES[view] || 'Jumpshot';
  const sh = $('#tb-share'); if (sh) sh.hidden = view !== 'results';
  const mo = $('#tb-more'); if (mo) mo.hidden = view !== 'history';
  document.body.classList.toggle('has-pin', view === 'clip' || view === 'shots' || (view === 'history' && state.editing));
}
async function route() {
  const h = location.hash.replace(/^#\/?/, '');
  const [name, a, b] = h.split('/');
  let view = name || 'home';
  if (view === 'session') view = 'results';
  if (view === 'compare') view = 'history';
  if (!VIEWS.includes(view)) view = 'home';
  // guards
  if (view === 'clip' && !state.file) return go('#/', true);
  if (view === 'processing' && !state.busy) return go(state.session ? '#/results' : '#/', true);
  if (view === 'shots' && !state.shots) return go('#/', true);
  if (view === 'results' && name === 'results' && !state.session) return go('#/history', true);
  if (view === 'error' && !state.error) return go('#/', true);
  if (current === 'processing' && view !== 'processing' && state.busy) { state.abort && state.abort.abort(); toast('Analysis cancelled'); }
  leaveHooks.splice(0).forEach(f => { try { f(); } catch (e) {} });
  closeSheet();
  if (view !== 'history') state.editing = false;
  current = view;
  for (const v of VIEWS) { const el = $(`[data-view="${v}"]`); if (el) el.hidden = v !== view; }
  chrome(view);
  window.scrollTo(0, 0);
  try {
    if (view === 'home') enterHome();
    else if (view === 'record') enterRecord();
    else if (view === 'clip') enterClip();
    else if (view === 'shots') renderShots();
    else if (view === 'results') {
      if (name === 'session') { const s = await DB.get(a); if (!s) { toast('Session not found'); return go('#/history', true); } state.session = s; }
      await renderResults(state.session, b != null ? +b : state.session.primary || 0);
    }
    else if (view === 'history') { await renderHistory(); if (name === 'compare' && a && b) openCompare(a, b); }
    else if (view === 'error') renderError();
  } catch (e) { console.error(e); showError(errUnknown(e)); }
}
window.addEventListener('hashchange', route);
document.addEventListener('click', e => { const b = e.target.closest('[data-back]'); if (b) { e.preventDefault(); history.length > 1 ? history.back() : go('#/'); } });
$$('.js-seg [data-nav]').forEach(b => b.addEventListener('click', () => go(b.dataset.nav === 'history' ? '#/history' : '#/')));

// ------------------------------------------------------------------ home
function enterHome() {
  DB.all().then(list => {
    const box = $('#home-last'); const s = list[list.length - 1];
    if (!s) { box.innerHTML = ''; return; }
    const cur = sessScores(s);
    box.innerHTML = `<section class="section"><div class="section-h"><h2>Last session</h2><a class="more" href="#/history">See all</a></div>
      <a class="card tappable last-card" href="#/session/${esc(s.id)}">
        ${s.thumb ? `<img src="${esc(s.thumb)}" alt="">` : '<span class="noimg"></span>'}
        <div class="grow"><div class="t">${esc(s.name || 'Shot')}</div><div class="s">${esc(fmtWhen(s.created))}</div></div>
        <div class="sc"><b>${f1(cur.mech)}</b><span>Mechanics</span></div>${ICONS.chev}</a></section>`;
  }).catch(() => {});
}
function pickFile(file, source) {
  if (!file) return;
  if (file.type && !/^video\//.test(file.type) && !/\.(mp4|mov|m4v|webm|mkv|avi|3gp)$/i.test(file.name || '')) {
    return showError({ icon: 'film', title: "That isn't a video", msg: `“${file.name}” can't be analysed. Pick a video from your camera roll, such as an MP4 or MOV file.`, actions: [{ label: 'Choose another clip', primary: true, go: '#/', click: () => setTimeout(() => $('#file-upload').click(), 50) }, { label: 'Try a sample', click: runSample }] });
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
$('#btn-sample').addEventListener('click', runSample);
$('#btn-about').addEventListener('click', () => openAbout());

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
      if (s.width && s.height) $('.rec-stage').style.setProperty('--ar', s.width + '/' + s.height);
    } catch (e) {
      showError({ icon: 'film', title: 'Camera not available', msg: e && e.name === 'NotAllowedError' ? 'Camera access was blocked. Allow it in your browser, or record with your camera app and upload the clip.' : "This device doesn't have a camera the browser can use. Record with your camera app and upload the clip instead.",
        actions: [{ label: 'Upload a clip', primary: true, go: '#/', click: () => setTimeout(() => $('#file-upload').click(), 50) }, { label: 'Try again', go: '#/record' }], code: e && (e.name + ': ' + e.message) });
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
      if (!blob.size) { toast('Nothing was recorded. Try again.'); go_.disabled = false; flip.disabled = false; return; }
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
function clipMeta(vm) {
  const parts = [fmtS(vm.duration, 1), `${vm.width}×${vm.height}`];
  if (state.fpsEst) parts.push(state.fpsEst + ' fps');
  return parts.join(', ');
}
async function enterClip() {
  const v = $('#clip-video'), meta = $('#clip-meta'), btn = $('#btn-analyze');
  btn.disabled = true;
  setSeg('#opt-side', prefs.side); setSeg('#opt-model', prefs.model); modelSub();
  meta.innerHTML = '&nbsp;';
  let vm;
  try { vm = state.vmeta && v.src === state.url ? state.vmeta : await loadVideo(v, state.url); }
  catch (e) { return showError(errCodec(e)); }
  state.vmeta = vm;
  $('#clip-stage').style.setProperty('--ar', vm.width + '/' + vm.height);
  const dur = vm.duration;
  if (!state.trim || state.trim[1] <= 0 || state.trim[1] > dur || state._trimFor !== state.url) { state.trim = [0, Math.min(dur, dur > MAX_WINDOW ? 8 : dur)]; state._trimFor = state.url; }
  meta.textContent = clipMeta(vm);
  if (dur > 60) toast('Long clip. Drag the handles around one shot.');
  setupTrim(v, dur);
  btn.disabled = false;
  // fps estimate while previewing the selection
  v.currentTime = state.trim[0];
  v.play().then(async () => {
    if (!state.fpsEst) state.fpsEst = await estimateFps(v);
    if (current === 'clip') meta.textContent = clipMeta(vm);
    if (state.fpsEst && state.fpsEst < 24) toast('Low frame rate. 30 or 60 fps gives better timing.');
  }).catch(() => {});
  filmstrip(state.url, dur);
  onLeave(() => { v.pause(); });
}
function setSeg(sel, val) { $$(sel + ' button').forEach(b => { const on = b.dataset.v === val; b.classList.toggle('on', on); b.setAttribute('aria-checked', on); }); }
function modelSub() {
  $('#model-sub').textContent = prefs.model === 'full'
    ? `High accuracy is about twice as slow and downloads ${(MODEL_SIZES.full / 1e6).toFixed(0)} MB once.`
    : `Fast works for most clips. It downloads ${(MODEL_SIZES.lite / 1e6).toFixed(0)} MB once, then runs offline.`;
}
$('#opt-side').addEventListener('click', e => { const b = e.target.closest('button'); if (!b) return; prefs.side = b.dataset.v; savePrefs(); setSeg('#opt-side', prefs.side); });
$('#opt-model').addEventListener('click', e => { const b = e.target.closest('button'); if (!b) return; prefs.model = b.dataset.v; savePrefs(); setSeg('#opt-model', prefs.model); modelSub(); });

function setupTrim(v, dur) {
  const a = $('#trim-a'), b = $('#trim-b'), sel = $('#trim-sel'), K = 1000;
  const toT = x => x / K * dur, toX = t => Math.round(t / dur * K);
  a.value = toX(state.trim[0]); b.value = toX(state.trim[1]);
  const minLen = Math.min(0.5, dur);
  const paint = () => {
    const [s, e] = state.trim; const W = $('#trim').clientWidth || 1, hw = 14 / W * 100;
    // the frame's 14 px side bars sit on top of the strip; map the selection so the bars hug the chosen range
    const l = s / dur * (100 - 2 * hw), r = e / dur * (100 - 2 * hw) + 2 * hw;
    sel.style.left = l + '%'; sel.style.width = (r - l) + '%';
    $('#trim-shade-l').style.width = l + '%'; $('#trim-shade-r').style.width = (100 - r) + '%';
    $('#trim-range').textContent = `${s.toFixed(2)} – ${e.toFixed(2)} s`;
    const len = e - s; $('#trim-len').textContent = len.toFixed(1) + ' s selected';
    $('#trim-note').textContent = dur > MAX_WINDOW ? `Up to ${MAX_WINDOW} seconds around your shot are analysed.` : len > 8 ? 'Long selections work for several shots but take longer.' : 'A tight trim around one shot is faster and more accurate.';
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
  const playBtn = on => { pb.innerHTML = on ? ICONS.pause : ICONS.play; pb.setAttribute('aria-label', on ? 'Pause' : 'Play selection'); };
  pb.onclick = () => { if (v.paused) { if (v.currentTime < state.trim[0] || v.currentTime >= state.trim[1] - 0.05) v.currentTime = state.trim[0]; v.play().catch(() => {}); } else v.pause(); };
  v.onplay = () => playBtn(true); v.onpause = () => playBtn(false);
  v.ontimeupdate = () => { $('#clip-cur').textContent = v.currentTime.toFixed(2) + ' s'; if (!v.paused && v.currentTime >= state.trim[1]) { v.currentTime = state.trim[0]; } };
  v.loop = false; v.onended = () => { v.currentTime = state.trim[0]; v.play().catch(() => {}); };
  const ro = new ResizeObserver(paint); ro.observe($('#trim')); onLeave(() => ro.disconnect());
  paint();
}
async function filmstrip(url, dur) {
  const strip = $('#trim-strip'); const n = innerWidth < 500 ? 7 : 10;
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
function setProc(frac, status, sub) {
  const b = $('#proc-bar'); if (frac != null) { b.querySelector('i').style.width = (clamp(frac, 0, 1) * 100).toFixed(1) + '%'; b.setAttribute('aria-valuenow', Math.round(frac * 100)); }
  if (status != null) $('#proc-status').textContent = status;
  if (sub != null) $('#proc-eta').innerHTML = sub ? esc(sub) : '&nbsp;';
}
function resetProcUI(status) {
  setProc(0, status || 'Getting ready…', '');
  $('#proc-idle').hidden = false;
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
    const t0 = performance.now();
    const lmk = await ensureLandmarker({
      model: prefs.model, signal,
      onProgress: p => {
        if (p.phase === 'download') setProc(p.loaded / (p.total || 1), `Downloading the pose model… ${Math.round(p.loaded / (p.total || 1) * 100)}%`, `One-time download, ${fmtBytes(p.total)}`);
        if (p.phase === 'init') setProc(0, 'Starting up…', '');
      },
    });
    try { const ms = new Set(safeJSONArr(localStorage.getItem('jumpshot.models'))); ms.add(prefs.model); localStorage.setItem('jumpshot.models', JSON.stringify([...ms])); } catch (e) {}
    if (lmk.fellBack) toast('Running on the CPU. Slower, same result.');
    if (signal.aborted) throw new PoseError('aborted', 'Cancelled');
    setProc(0, 'Finding your body…', '');
    await loadVideo(pv, state.url).catch(e => { throw new PoseError('codec', e.message); });
    const pc = $('#proc-canvas'), pctx = pc.getContext('2d');
    const stage = $('#proc-stage');
    const sizeCanvas = () => { const d = Math.min(2, devicePixelRatio || 1); pc.width = Math.round(stage.clientWidth * d); pc.height = Math.round(stage.clientHeight * d); };
    sizeCanvas();
    let lastDraw = 0; const brand = C().brand;
    const out = await extractPose(pv, {
      start, end, landmarker: lmk.landmarker, model: prefs.model, nativeFps: state.fpsEst || 30, signal,
      onFrame: f => {
        const now = performance.now();
        const eta = f.eta != null ? (f.eta < 1.5 ? 'Almost done' : `About ${Math.ceil(f.eta)} s left`) : '';
        setProc(f.frac, `Finding your body… ${f.done} of ${f.expected} frames`, eta);
        if (now - lastDraw > 40) {        // throttle preview painting
          lastDraw = now; $('#proc-idle').hidden = true;
          if (pc.width !== Math.round(stage.clientWidth * Math.min(2, devicePixelRatio || 1))) sizeCanvas();
          const r = containRect(pc.width, pc.height, f.canvas.width, f.canvas.height);
          pctx.fillStyle = '#000'; pctx.fillRect(0, 0, pc.width, pc.height);
          pctx.drawImage(f.canvas, r.x, r.y, r.w, r.h);
          if (f.lm) drawPose(pctx, f.lm, (x, y) => [r.x + x * r.w, r.y + y * r.h], { col: brand, w: Math.max(2.5, r.h / 160), joints: true, arm: prefs.side === 'left' ? 'left' : prefs.side === 'right' ? 'right' : null });
        }
      },
    });
    const { frames, meta, stats } = out;
    console.info('[jumpshot] pose', stats, meta, ((performance.now() - t0) / 1000).toFixed(1) + ' s');
    if (stats.delegateSwitched) toast('Finished on the CPU');
    const nPose = frames.filter(f => f.lm).length;
    state.frames = frames; state.pmeta = meta; state.trimStart = start; state.lastStats = stats; state.lastDelegate = lmk.delegate;
    // quality gates
    if (nPose < Math.max(5, frames.length * 0.15)) throw Object.assign(new Error('no person'), { code: 'no_person', nPose, n: frames.length });
    const h = bodyHeight(frames);
    if (h < 0.2 && !state._ignoreSmall) throw Object.assign(new Error('small'), { code: 'small', h });
    setProc(1, 'Scoring your shot…', '');
    const [eng] = await Promise.all([getEngine(), assetsJob]);
    const shots = safeShots(eng, frames, meta);
    endBusy();
    state.shots = shots;
    if (shots.length === 1) return analyzeWindows([shots[0]]);
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

async function analyzeWindows(windows) {
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
  if (windows.length > 1 && results.length < windows.length) setTimeout(() => toast(`${windows.length - results.length} of ${windows.length} shots couldn't be measured`), 600);
  const sample = state.source === 'sample';
  const best = 0;
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
  if (!sample) { try { await DB.put(s); } catch (e) { console.warn('save failed', e); toast("Couldn't save to History. Storage may be full."); } }
  buzz([20, 40, 20]);
  go('#/results', true);
}

// ------------------------------------------------------------------ sample
async function runSample() {
  const signal = beginBusy();
  state.source = 'sample';
  resetProcUI('Loading the sample…');
  $('#proc-stage').style.setProperty('--ar', '3/4');
  go('#/processing');
  try {
    const [eng, assets, demo] = await Promise.all([getEngine(), getAssets(), fetch(new URL('demo.json', APP_BASE)).then(r => { if (!r.ok) throw new Error('demo.json ' + r.status); return r.json(); })]);
    if (signal.aborted) throw new PoseError('aborted');
    const { frames, meta } = parseDemo(demo);
    // a short preview of the pre-computed landmarks, so the step isn't a blank wait
    const pc = $('#proc-canvas'), stage = $('#proc-stage'); const d = Math.min(2, devicePixelRatio || 1);
    pc.width = Math.round(stage.clientWidth * d); pc.height = Math.round(stage.clientHeight * d); $('#proc-idle').hidden = true;
    const ctx = pc.getContext('2d'); const r = containRect(pc.width, pc.height, meta.width, meta.height);
    const steps = RM.matches ? 1 : Math.min(frames.length, 45); const brand = C().brand;
    for (let k = 0; k < steps; k++) {
      if (signal.aborted) throw new PoseError('aborted');
      const i = Math.round(k / Math.max(1, steps - 1) * (frames.length - 1));
      ctx.fillStyle = STAGE_BG; ctx.fillRect(0, 0, pc.width, pc.height);
      if (frames[i].lm) drawPose(ctx, frames[i].lm, (x, y) => [r.x + x * r.w, r.y + y * r.h], { col: brand, w: Math.max(3, r.h / 150), joints: true });
      setProc((k + 1) / steps, `Finding your body… ${i + 1} of ${frames.length} frames`, 'Sample shot');
      await new Promise(res => setTimeout(res, 16));
    }
    setProc(1, 'Scoring your shot…', 'Sample shot');
    state.frames = frames; state.pmeta = meta; state.trimStart = 0; state.lastDelegate = null;
    const shots = safeShots(eng, frames, meta);
    endBusy();
    state.shots = shots;
    await analyzeWindows(shots.length ? [shots[0]] : [null]);
  } catch (e) {
    endBusy();
    if (e && e.code === 'aborted') { go('#/', true); return; }
    console.error(e);
    showError({ icon: 'offline', title: "The sample didn't load", msg: 'Check your connection and try again. The first visit needs to download the app files.', actions: [{ label: 'Try again', primary: true, click: runSample }, { label: 'Back to Jumpshot', go: '#/' }], code: String(e && e.message) });
  }
}
function parseDemo(d) {
  const src = d.clip || d.user || d;
  const m = src.meta || d.meta || {};
  const meta = { fps: num(m.fps || src.fps || 30), width: num(m.width || src.width || 480), height: num(m.height || src.height || 640), duration: 0 };
  const raw = src.frames || src.landmarks || d.frames;
  const vis = src.visibility || null;
  const frames = raw.map((f, i) => {
    const lm = f && f.lm !== undefined ? f.lm : f;
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
    document.body.classList.remove('has-pin');
    box.innerHTML = `<div class="empty err"><div class="ic">${ICONS.person}</div><h3>No jump shot found</h3>
      <p>We found you in ${frames.filter(f => f.lm).length} of ${frames.length} frames, but couldn't spot a dip, lift and release.</p>
      <div class="acts"><button class="btn primary" id="btn-whole" type="button">Analyze the whole clip</button><a class="btn plain" href="#/clip">Adjust trim</a></div></div>`;
    $('#btn-whole').onclick = () => analyzeWindows([null]);
    return;
  }
  box.innerHTML = `<h1 class="large-title">${shots.length} shots found</h1><p class="subtitle">Pick one, or analyze them all to see how consistent you are.</p>
    <div class="shots-grid" id="shots-row">${shots.map((s, i) => `<button class="card tappable shot" type="button" data-i="${i}"><span class="th skel" style="border-radius:0"></span>
      <span class="cap"><b>Shot ${i + 1}</b><span class="num">${fmtS(frames[clamp(s.releaseGuess | 0, 0, frames.length - 1)].t, 1)}</span></span></button>`).join('')}</div>
    <div class="pin"><button class="btn primary block" id="btn-all" type="button">Analyze all ${shots.length} shots</button></div>`;
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
    ctx.strokeStyle = 'rgba(0,0,0,.5)'; ctx.globalAlpha = (st.alpha || 1) * 0.8;
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
  if (st.joints) for (let i = 11; i < 33; i++) { const p = pts[i]; if (!p || (i > 16 && i < 23)) continue; ctx.fillStyle = st.col; ctx.beginPath(); ctx.arc(p[0], p[1], st.w * 0.95, 0, 7); ctx.fill(); ctx.fillStyle = '#000'; ctx.beginPath(); ctx.arc(p[0], p[1], st.w * 0.4, 0, 7); ctx.fill(); }
  ctx.restore();
}
/** Image (data URL) of frame t from the video (or a dark stage when url is null) with the skeleton burned in. */
async function poseImage(url, t, lm, meta, maxSide, ghostLm, quality = 0.82, box = null) {
  let c, ctx;
  if (url) { c = await grab(url, t, maxSide); ctx = c.getContext('2d'); ctx.fillStyle = 'rgba(0,0,0,.18)'; ctx.fillRect(0, 0, c.width, c.height); }
  else { const s = maxSide / Math.max(meta.width, meta.height); c = document.createElement('canvas'); c.width = Math.round(meta.width * s); c.height = Math.round(meta.height * s); ctx = c.getContext('2d'); ctx.fillStyle = STAGE_BG; ctx.fillRect(0, 0, c.width, c.height); }
  let map = (x, y) => [x * c.width, y * c.height];
  if (!url && box) { const z = Math.min(1 / box.w, 1 / box.h) * 0.92; const cx = box.x + box.w / 2, cy = box.y + box.h / 2; map = (x, y) => [c.width / 2 + (x - cx) * z * c.width, c.height / 2 + (y - cy) * z * c.height]; }
  const w = Math.max(2, c.height / 110);
  if (ghostLm) drawPose(ctx, ghostLm, map, { col: STAGE_GHOST, w: w * 0.8, dash: [w * 2, w * 1.6], halo: false });
  if (lm) drawPose(ctx, lm, map, { col: C().brand, w, joints: true });
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
/** Domain of a check's range bar: the target band plus a margin, widened to include the user's value. */
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
function rangeBar(m, lg) {
  if (!m.band || !fin(m.value)) return `<div class="rbar${lg ? ' lg' : ''}"></div>`;
  const [a, b] = mechDomain(m), pct = x => clamp((num(x) - a) / ((b - a) || 1) * 100, 0, 100);
  const bl = pct(m.band[0]), bw = Math.max(1, pct(m.band[1]) - bl);
  const cls = STATUS_CLS[m.status] || 'mute';
  return `<div class="rbar${lg ? ' lg' : ''}" role="img" aria-label="${esc(fmtMech(m.value, m.unit, m.key))}, target ${esc(fmtRange(m.band, m.unit))}"><i class="band" style="left:${bl.toFixed(1)}%;width:${bw.toFixed(1)}%"></i><i class="mk bg-${cls}" style="left:${pct(m.value).toFixed(1)}%"></i></div>`;
}
const playLabel = (on) => (on ? ICONS.pause : ICONS.play);

async function renderResults(session, idx) {
  const box = $('#results-body');
  const results = sessResults(session);
  idx = clamp(idx | 0, 0, results.length - 1);
  const r = results[idx];
  const eng = await getEngine(); let assets = null; try { assets = await getAssets(); } catch (e) {}
  const FEATURES = eng.FEATURES || [];
  const hasVideo = !!(state.sessionVideo && state.sessionVideo.id === session.id && state.sessionVideo.url);
  const vurl = hasVideo ? state.sessionVideo.url : null;
  const nm = splitName(r.closest && r.closest.name || ((r.comparisons || [])[0] || {}).name);
  const ms = mechScore(r), ss = styleScore(r), mk = hasMech(r) ? r.mechanics : [];
  const nMeas = mk.filter(m => m.status !== 'unmeasured').length, nGood = mk.filter(m => m.status === 'good').length;
  const plan = r.plan || [];
  const q = r.quality || {}; const ql = q.level || 'ok';
  const sample = session.source === 'sample';
  const top = topFixes(r, 1)[0];
  const tot = mk.reduce((s, m) => s + (num(m.maxPoints) || 0), 0) || 1;

  // ---- 1. hero
  const hero = `<div class="hero">
      <div class="hero-score"><b id="kpi-score" class="num">${f1(ms)}</b>${fin(ms) ? '<small>/100</small>' : ''}</div>
      <p class="hero-cap">Mechanics score</p>
      <h1 class="hero-verdict" id="h-res">${esc(verdict(r))}</h1>
      <p class="hero-meta">${sample ? 'Sample shot' : esc(session.name || 'Your shot')}, ${esc(fmtWhen(session.created))}</p>
      ${mk.length ? `<div class="pts-strip" aria-hidden="true">${mk.map(m => `<i style="flex:${(num(m.maxPoints) || 1) / tot * 100}"><b class="bg-${STATUS_CLS[m.status] || 'mute'}" style="width:${m.status === 'unmeasured' ? 0 : clamp(num(m.points) / (num(m.maxPoints) || 1) * 100, 0, 100).toFixed(0)}%"></b></i>`).join('')}</div>
      <p class="hero-cap">${nGood} of ${nMeas} checks in range${mk.length > nMeas ? `, ${mk.length - nMeas} not measured` : ''}</p>` : ''}
      ${ql !== 'ok' || (q.flags || []).length ? `<p class="notice ${ql === 'major' ? 'neg' : 'warn'}">${ICONS.warn}<span>${esc(((q.flags || [])[0] && ((q.flags[0].text || q.flags[0].message))) || (ql === 'major' ? 'Low tracking confidence. Treat these numbers as rough.' : 'Some frames were hard to track.'))}</span></p>` : ''}
    </div>`;

  // multi-shot consistency
  let multi = '';
  if (results.length > 1) {
    const sc = results.map(mechScore), f = sc.filter(Number.isFinite);
    multi = `<section class="section multi"><div class="section-h"><h2>Your shots</h2></div>
      <div class="list">${results.map((x, i) => { const rel = x.events && fin(x.events.release) ? session.frames[clamp(x.events.release, 0, session.frames.length - 1)].t : NaN;
        return `<button class="row${i === idx ? ' selected' : ''}" type="button" data-shot="${i}" aria-pressed="${i === idx}"><div class="grow"><div class="title">Shot ${i + 1}</div><div class="sub">Released at ${fmtS(rel, 1)}</div></div><span class="value">${f1(sc[i])}</span>${ICONS.chev}</button>`; }).join('')}</div>
      ${f.length > 1 ? `<p class="section-foot">Your score varies by ±${sd(f).toFixed(1)} points across ${f.length} shots. ${sd(f) < 3 ? 'That is a repeatable stroke.' : sd(f) < 7 ? 'Fairly consistent.' : 'Groove one motion before changing mechanics.'}</p>` : ''}</section>`;
  }

  // ---- 2. fix this first
  let fix = '';
  if (top) {
    const [head, why] = fixCopy(top); const p = plan.find(x => x.key === top.key);
    fix = `<section class="section"><div class="section-h"><h2>Fix this first</h2></div>
      <div class="card fix" id="fix-card">
        <h3 class="fix-h">${esc(head)}</h3><p class="fix-what">${esc(why)}</p>
        <div class="fix-nums"><div><div class="v">${esc(fmtMech(top.value, top.unit, top.key))}</div><div class="c">You</div></div>
          <div><div class="v">${esc(fmtRange(top.band, top.unit))}</div><div class="c">Target</div></div>
          <div><div class="v">+${fmtPts(gainOf(top))}</div><div class="c">Points to gain</div></div></div>
        <div class="fix-bar">${rangeBar(top, true)}</div>
        ${p ? `<div class="fix-drill"><div class="k">Drill</div><div class="n">${esc(p.drill)}</div><div class="rx">${esc(rx(p.prescription))}</div><p class="cue">${esc(p.cue || '')}</p></div>` : ''}
      </div></section>`;
  } else if (mk.length && fin(ms)) {
    fix = `<section class="section"><div class="card fix good"><h3 class="fix-h">Nothing to fix</h3><p class="fix-what">Every check the camera could see is in range. Keep your reps consistent and film again next week.</p></div></section>`;
  }

  // ---- 3. video + scrubber
  const ev = r.events || {}; const N = session.frames.length;
  const video = `<section class="section"><div class="section-h"><h2>Your shot</h2></div>
      <div class="stage bleed scrub-stage" id="sc-stage">${hasVideo ? '<video id="sc-video" playsinline muted preload="auto"></video>' : ''}<canvas id="sc-canvas"></canvas>
        <span class="stage-tag" id="sc-gtag">Dashed: ${esc(nm.title)}</span></div>
      <div class="scrub-tl"><div class="ticks" id="sc-marks"></div><input type="range" id="sc-range" min="0" max="${Math.max(1, N - 1)}" value="0" step="1" aria-label="Frame"></div>
      <div class="scrub-cap"><span><b id="sc-phase">Release</b> <span id="sc-time" class="num">0.00 s</span></span><button class="btn plain" id="sc-ghost" type="button" aria-pressed="true">Hide reference</button></div>
      <div class="scrub-ctl">
        <button class="icon-btn" id="sc-prev" type="button" aria-label="Previous frame">${ICONS.prev}</button>
        <button class="icon-btn play" id="sc-play" type="button" aria-label="Play">${ICONS.play}</button>
        <button class="icon-btn" id="sc-next" type="button" aria-label="Next frame">${ICONS.next}</button>
        <div class="seg" role="group" aria-label="Playback speed"><button type="button" data-speed="0.25">¼×</button><button type="button" class="on" data-speed="0.5">½×</button><button type="button" data-speed="1">1×</button></div>
      </div>
      <div class="keys" id="kf">${[['set', 'Set point'], ['release', 'Release'], ['ftEnd', 'Follow-through']].map(([k, l]) => `<button class="key" type="button" data-jump="${k}"${fin(ev[k]) ? '' : ' disabled'}><span class="ph skel"></span><span>${l}</span></button>`).join('')}</div>
      <div class="list" style="margin-top:16px"><button class="row" id="sc-ro-tg" type="button" aria-expanded="false"><div class="grow"><div class="title">Angles at this frame</div></div>${ICONS.chev}</button><div id="sc-ro" hidden></div></div>
    </section>`;

  // ---- 4. mechanics
  let mech;
  if (!mk.length) {
    mech = `<section class="section"><div class="section-h"><h2>Your mechanics</h2></div><div class="list"><div class="row"><div class="grow"><div class="title wrap">Not scored</div><div class="sub" style="white-space:normal">This session was analysed before mechanics checks existed. Analyse the clip again to score it.</div></div></div></div></section>`;
  } else {
    mech = `<section class="section"><div class="section-h"><h2>Your mechanics</h2><button class="more" id="btn-how" type="button">About the score</button></div>
      <div class="list" id="mech-list">${mk.map((m, i) => {
        const un = m.status === 'unmeasured' || !fin(m.value);
        if (un) return `<button class="row mrow off-m" type="button" data-mk="${i}"><div class="grow"><div class="title">${esc(shortLabel(m))}</div><div class="sub">${esc(cap1(m.reason || 'Not visible from this camera angle'))}</div></div><div class="right"><div class="v">—</div><div class="p">Not counted</div></div>${ICONS.chev}</button>`;
        return `<button class="row mrow" type="button" data-mk="${i}"><div class="grow"><div class="title">${esc(shortLabel(m))}</div>${rangeBar(m)}</div><div class="right"><div class="v">${esc(fmtMech(m.value, m.unit, m.key))}</div><div class="p">${fmtPts(m.points)} of ${fmtPts(m.maxPoints)}</div></div>${ICONS.chev}</button>`;
      }).join('')}</div>
      ${!fin(ms) ? '<p class="section-foot">Fewer than 40 of 100 points could be measured, so this clip has no mechanics score.</p>' : ''}</section>`;
  }

  // ---- 5. practice plan
  const SHOW = 4;
  const planHTML = plan.length ? `<section class="section"><div class="section-h"><h2>Practice plan</h2>${plan.length > SHOW ? `<button class="more" id="plan-all" type="button">Show all ${plan.length}</button>` : ''}</div>
      <div class="list" id="plan-list">${plan.map((p, i) => `<button class="row drill" type="button" data-d="${i}" aria-expanded="false"${i >= SHOW ? ' hidden' : ''}><span class="num-ic">${i + 1}</span><div class="grow"><div class="title">${esc(p.drill)}</div><div class="sub">${esc(rx(p.prescription || ''))}</div></div>${ICONS.chev}</button><div class="drill-cue" data-cue="${i}"${i >= SHOW ? ' hidden' : ''}><div><p>${esc(p.cue || '')}</p></div></div>`).join('')}</div>
      <p class="section-foot">Ordered by what costs you the most points. General form-shooting work, not professional coaching.</p></section>` : '';

  // ---- 6. style + details + actions
  const more = `<section class="section"><div class="section-h"><h2>More</h2></div><div class="list">
      <button class="row" id="row-style" type="button"><div class="grow"><div class="title">Closest style</div><div class="sub">${esc(nm.title)}${nm.style ? `, ${esc(nm.style)}` : ''}</div></div><span class="value">${f1(ss)} match</span>${ICONS.chev}</button>
      <button class="row" id="row-details" type="button"><div class="grow"><div class="title">Details</div><div class="sub">Tracking quality and motion curves</div></div>${ICONS.chev}</button>
    </div></section>
    <section class="section">
      <a class="btn block" href="#/">Analyze another shot</a>
      ${sample ? '<p class="section-foot center">Sample results are not saved to History.</p>' : '<button class="btn plain block danger" id="btn-del" type="button" style="margin-top:8px">Delete this session</button>'}
    </section>`;

  box.innerHTML = hero + multi + fix + video + mech + planHTML + more;

  // ---- wire up
  const sh = $('#tb-share'); if (sh) sh.onclick = () => openShare(session, r);
  $('#row-style').onclick = () => openStyle(r, assets);
  $('#row-details').onclick = () => openDetails(session, r, assets, FEATURES, hasVideo);
  const how = $('#btn-how'); if (how) how.onclick = () => openAbout(assets, mk);
  const del = $('#btn-del'); if (del) del.onclick = async () => { if (!confirm('Delete this session from this device?')) return; await DB.del(session.id); toast('Session deleted'); go('#/history'); };
  $$('#results-body [data-shot]').forEach(b => b.onclick = () => { const i = +b.dataset.shot; if (i !== idx) renderResults(session, i).then(() => window.scrollTo(0, 0)); });
  $$('#mech-list [data-mk]').forEach(b => b.onclick = () => openCheck(mk[+b.dataset.mk], r, assets));
  $$('#plan-list .drill').forEach(b => b.onclick = () => {
    const open = b.getAttribute('aria-expanded') !== 'true'; b.setAttribute('aria-expanded', open);
    const c = $(`#plan-list [data-cue="${b.dataset.d}"]`); if (c) c.classList.toggle('open', open);
  });
  const pa = $('#plan-all'); if (pa) pa.onclick = () => { $$('#plan-list [hidden]').forEach(x => { x.hidden = false; }); pa.remove(); };
  const tg = $('#sc-ro-tg'); tg.onclick = () => { const open = tg.getAttribute('aria-expanded') !== 'true'; tg.setAttribute('aria-expanded', open); $('#sc-ro').hidden = !open; tg.querySelector('.chev').style.transform = open ? 'rotate(90deg)' : ''; if (scrub) scrub.readouts(); };
  countUp($('#kpi-score'), ms);
  if (scrub) scrub.destroy();
  scrub = new Scrubber(session, r, FEATURES, assets, vurl);
  onLeave(() => { if (scrub) { scrub.destroy(); scrub = null; } });
  keyframes(session, r);
}
function countUp(el, v) {
  if (!el || RM.matches || !fin(v)) return;
  const t0 = performance.now(), D = 650;
  const f = now => { const k = Math.min(1, (now - t0) / D), e = 1 - Math.pow(1 - k, 3); el.textContent = (k < 1 ? (v * e).toFixed(1) : num(v).toFixed(1)); if (k < 1) requestAnimationFrame(f); };
  requestAnimationFrame(f);
}

// ---- sheets: one check, about the score, style, details
function openCheck(m, r, assets) {
  const S = mechSources(assets); const p = (r.plan || []).find(x => x.key === m.key);
  const un = m.status === 'unmeasured' || !fin(m.value);
  const cls = STATUS_CLS[m.status] || 'mute';
  let body = `<div class="sheet-big"><b>${esc(fmtMech(m.value, m.unit, m.key))}</b><span class="badge ${cls === 'mute' ? '' : cls}">${STATUS_TXT[m.status] || ''}</span></div>`;
  if (!un) {
    const [a, b] = mechDomain(m), pct = x => clamp((num(x) - a) / ((b - a) || 1) * 100, 0, 100);
    body += rangeBar(m, true) + `<div class="rbar-ax"><span style="left:${pct(m.band[0]).toFixed(1)}%">${esc(fmtIdeal(m.band[0], m.unit, m.key))}</span><span style="left:${pct(m.band[1]).toFixed(1)}%">${esc(fmtIdeal(m.band[1], m.unit, m.key))}</span></div>`;
  }
  body += `<div class="sec"><div class="list">
      <div class="row"><div class="grow"><div class="title">Target</div></div><span class="value">${esc(fmtRange(m.band, m.unit))}</span></div>
      ${fin(m.ideal) ? `<div class="row"><div class="grow"><div class="title">Ideal</div></div><span class="value">${esc(fmtIdeal(m.ideal, m.unit, m.key))}</span></div>` : ''}
      <div class="row"><div class="grow"><div class="title">Points</div></div><span class="value">${un ? 'Not counted' : `${fmtPts(m.points)} of ${fmtPts(m.maxPoints)}`}</span></div>
    </div></div>
    <div class="sec"><h4>What this measures</h4><p class="body">${esc(WHAT[m.key] || m.label)}</p>
      ${un ? `<p class="body">${esc(cap1(m.reason || 'This check could not be measured from this camera angle.'))} It doesn't count for or against you.</p>` : m.status === 'good' ? '<p class="body">You are inside the target range. Keep doing what you are doing.</p>' : `<p class="body">${esc(fixCopy(m)[1])}</p>`}</div>`;
  if (p) body += `<div class="sec"><h4>Drill</h4><div class="list"><div class="row"><div class="grow"><div class="title wrap">${esc(p.drill)}</div><div class="sub">${esc(rx(p.prescription))}</div></div></div></div><p class="body" style="margin-top:10px">${esc(p.cue || '')}</p></div>`;
  const src = (m.source || []).map(id => S.list[S.n[id] - 1]).filter(Boolean);
  if (src.length) body += `<div class="sec"><h4>Source</h4><ol class="src-list">${src.map(x => `<li>${esc(x.cite || x.id)}${x.url ? ` <a href="${esc(x.url)}" target="_blank" rel="noopener">Link</a>` : ''}</li>`).join('')}</ol></div>`;
  sheet(shortLabel(m), body);
}
function openAbout(assets, mk) {
  const go_ = a => {
    const S = mechSources(a); const spec = (a && a.mechSpec && a.mechSpec.length ? a.mechSpec : mk || []);
    sheet('How the analysis works', `
      <p class="body">A pose model runs in this browser and finds your joints in every frame. Your video is never uploaded.</p>
      <p class="body">Your <b>mechanics score</b> checks eight parts of your shot against ranges from published jump-shot research. A check inside its range earns full points; outside it, points drop off. Checks the camera can't see don't count for or against you.</p>
      ${spec.length ? `<div class="sec"><h4>The checks</h4><div class="list">${spec.map(m => `<div class="row"><div class="grow"><div class="title">${esc(shortLabel(m))}</div><div class="sub">Target ${esc(fmtRange(m.band, m.unit))}</div></div><span class="value">${fmtPts(m.maxPoints != null ? m.maxPoints : m.points)} pts</span></div>`).join('')}</div></div>` : ''}
      <div class="sec"><h4>Closest style</h4><p class="body">Separately, your motion is compared with a small library of reference shooters to find the one you move most like. It says how alike you move, not how good your mechanics are, and doesn't affect the score.</p></div>
      <div class="sec"><h4>Limits</h4><p class="body">The ranges come from 3-D lab studies; here they're applied to a single 2-D phone video, so treat them as coaching guides. Film side-on with your whole body in frame. Ball flight and makes aren't measured. The references are synthetic stand-ins, and the drills are general form-shooting work.</p></div>
      ${S.list.length ? `<div class="sec"><h4>Sources</h4><ol class="src-list">${S.list.map(x => `<li>${esc(x.cite || x.id)}${x.url ? ` <a href="${esc(x.url)}" target="_blank" rel="noopener">Link</a>` : ''}</li>`).join('')}</ol></div>` : ''}`);
  };
  if (assets) return go_(assets);
  getAssets().then(go_).catch(() => go_(null));
}
function openStyle(r, assets) {
  const nm = splitName(r.closest && r.closest.name); const cmp = r.comparisons || [];
  sheet('Closest style', `
    <div class="sheet-big"><b>${f1(styleScore(r))}</b><span class="muted">match with ${esc(nm.title)}</span></div>
    <p class="body">You move most like <b>${esc(nm.title)}</b>${nm.style ? ` (${esc(nm.style)})` : ''}. This compares the shape and timing of your motion with each reference. It doesn't affect your mechanics score.</p>
    ${r.radar && r.radar.axes ? `<div class="sec"><h4>Profile</h4><div class="radar-wrap">${radarSVG(r.radar, [{ k: 'ref', col: 'var(--label-3)', w: 1.5, dash: '4 4' }, { k: 'user', col: 'var(--brand)', fill: 'var(--brand-tint)', w: 2, dots: true }])}</div>
      <div class="legend"><span><i style="background:var(--brand)"></i>You</span><span><i class="dash"></i>${esc(nm.title)}</span></div></div>` : ''}
    ${cmp.length ? `<div class="sec"><h4>All references</h4><div class="list">${cmp.map(c => { const n = splitName(c.name); return `<div class="row"><div class="grow"><div class="title">${esc(n.title)}</div>${n.style ? `<div class="sub">${esc(cap1(n.style))}</div>` : ''}</div><span class="value">${f1(c.score)}</span></div>`; }).join('')}</div></div>` : ''}`);
}
function openDetails(session, r, assets, FEATURES, hasVideo) {
  const q = r.quality || {}; const fr = session.frames; const ev = r.events || {};
  const tAt = i => (fin(i) ? fmtS(fr[clamp(i, 0, fr.length - 1)].t) : '—');
  const ql = { ok: 'Good', minor: 'Some issues', major: 'Low' }[q.level || 'ok'];
  const rows = [
    ['Tracking quality', ql], ['Body visible', fin(q.visibility) ? Math.round(q.visibility * 100) + '%' : '—'],
    ['Frames with you in them', `${fr.filter(f => f.lm).length} of ${fr.length}`], ['Frame rate', session.meta.fps + ' fps'],
    ['Shooting hand', cap1(r.hand || '—')],
  ];
  if (session.model) rows.push(['Pose model', (session.model === 'full' ? 'High accuracy' : 'Fast') + (session.delegate ? ` on ${session.delegate}` : '')]);
  const keyRows = [['Start of the shot', ev.onset], ['Set point', ev.set], ['Release', ev.release], ['End of follow-through', ev.ftEnd]];
  sheet('Details', `
    <div class="list">${rows.map(([k, v]) => `<div class="row"><div class="grow"><div class="title">${esc(k)}</div></div><span class="value">${esc(v)}</span></div>`).join('')}</div>
    ${(q.flags || []).length ? `<div class="sec"><h4>Notes</h4>${q.flags.map(f => `<p class="body">${esc(f.text || f.message || f.code)}</p>`).join('')}</div>` : ''}
    <div class="sec"><h4>Key moments</h4><div class="list">${keyRows.map(([k, i]) => `<div class="row"><div class="grow"><div class="title">${k}</div></div><span class="value">${tAt(i)}</span></div>`).join('')}</div></div>
    ${curvesHTML(r, assets, FEATURES)}
    <p class="section-foot">${hasVideo ? 'Your video stays in memory only while this page is open.' : 'Videos are never stored, so saved sessions replay as a skeleton.'}</p>`);
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
  return `<div class="sec"><h4>Motion curves</h4><div class="curves">${FEATURES.slice(0, 8).map((F, j) => {
    const a = path(u, j), b = rf ? path(rf, j) : null; const lo = Math.min(a.lo, b ? b.lo : a.lo), hi = Math.max(a.hi, b ? b.hi : a.hi);
    const pts = vv => vv.map((y, i) => Number.isFinite(y) ? `${(i / (vv.length - 1) * 160).toFixed(1)},${(50 - (y - lo) / ((hi - lo) || 1) * 44).toFixed(1)}` : '').filter(Boolean).join(' ');
    return `<div class="cv"><span>${esc(F.label)}</span><svg viewBox="0 0 160 54" preserveAspectRatio="none" aria-hidden="true">${b ? `<polyline fill="none" stroke="var(--label-3)" stroke-width="1.5" stroke-dasharray="4 3" points="${pts(b.v)}"/>` : ''}<polyline fill="none" stroke="var(--brand)" stroke-width="2" stroke-linejoin="round" points="${pts(a.v)}"/></svg></div>`; }).join('')}</div>
    <div class="legend"><span><i style="background:var(--brand)"></i>You, start to finish</span>${rf ? `<span><i class="dash"></i>${esc(splitName(r.closest.name).title)}</span>` : ''}</div></div>`;
}

async function keyframeImage(session, r, key, maxSide, q) {
  const ev = r.events || {}; const fr = session.frames; const i = clamp(fin(ev[key]) ? ev[key] | 0 : fr.length >> 1, 0, fr.length - 1);
  const url = state.sessionVideo && state.sessionVideo.id === session.id ? state.sessionVideo.url : null;
  const gm = ghostMapper(session, r);
  return poseImage(url, (session.trimStart || 0) + fr[i].t, lmAt(session, r, i), session.meta, maxSide, gm && maxSide > 200 ? gm.at(i) : null, q, url ? null : bodyBox(session, r));
}
async function keyframes(session, r) {
  const keys = ['set', 'release', 'ftEnd'];
  for (let k = 0; k < keys.length; k++) {
    try {
      const src = await keyframeImage(session, r, keys[k], 360, 0.8);
      const ph = $$('#kf .ph')[k]; if (!ph) return;
      ph.classList.remove('skel'); ph.innerHTML = `<img alt="" src="${src}">`;
    } catch (e) { console.warn('keyframe', e); }
  }
}

// ---- radar (SVG with token colours)
function radarSVG(radar, series) {
  if (!radar || !radar.axes || !radar.axes.length) return '';
  const plain = l => String(l).replace(/\s*@\s*/g, ' at ').replace(/\bmin\b/, 'bend').replace(/\bfwd\b/, 'forward').replace(/\balign\b/, 'angle');
  const ax = radar.axes.map(a => plain(typeof a === 'string' ? a : a.label || a.key || '')), n = ax.length;
  const cx = 200, cy = 140, R = 92;
  const pt = (i, v) => { const a = -Math.PI / 2 + i / n * Math.PI * 2; return [cx + Math.cos(a) * R * v, cy + Math.sin(a) * R * v]; };
  let s = `<svg class="chart" viewBox="0 0 400 280" role="img" aria-label="Style profile compared with the reference">`;
  for (const k of [0.5, 1]) s += `<polygon points="${ax.map((_, i) => pt(i, k).map(x => x.toFixed(1)).join(',')).join(' ')}" fill="none" class="grid"/>`;
  ax.forEach((_, i) => { const [x, y] = pt(i, 1); s += `<line x1="${cx}" y1="${cy}" x2="${x.toFixed(1)}" y2="${y.toFixed(1)}" class="grid"/>`; });
  for (const se of series) {
    const vals = radar[se.k]; if (!vals || !vals.length) continue;
    const pts = vals.map((v, i) => pt(i, clamp(num(v) || 0, 0, 1.08)));
    s += `<polygon points="${pts.map(p => p.map(x => x.toFixed(1)).join(',')).join(' ')}" style="fill:${se.fill || 'none'};stroke:${se.col}" stroke-width="${se.w}" stroke-linejoin="round" ${se.dash ? `stroke-dasharray="${se.dash}"` : ''}/>`;
    if (se.dots) pts.forEach((p, i) => { s += `<circle cx="${p[0].toFixed(1)}" cy="${p[1].toFixed(1)}" r="3" style="fill:${se.col}"><title>${esc(ax[i])}</title></circle>`; });
  }
  ax.forEach((l, i) => { const [x, y] = pt(i, 1.18); const anc = Math.abs(x - cx) < 8 ? 'middle' : x > cx ? 'start' : 'end'; s += `<text x="${x.toFixed(1)}" y="${(y + 4).toFixed(1)}" text-anchor="${anc}">${esc(l)}</text>`; });
  return s + '</svg>';
}

// ---- scrubber over the user's real video (or a skeleton replay)
const PHASE = { stance: 'Stance', load: 'Dip', lift: 'Lift', release: 'Release', 'follow-through': 'Follow-through', recovery: 'Landing' };
class Scrubber {
  constructor(session, r, FEATURES, assets, vurl) {
    this.s = session; this.r = r; this.F = FEATURES; this.fr = session.frames; this.N = this.fr.length; this.fps = session.meta.fps;
    this.ev = r.events || {}; this.speed = 0.5; this.ghostOn = true; this.playing = false; this.vurl = vurl;
    this.gm = ghostMapper(session, r); this.rseq = refSeq(assets, r.closest && r.closest.name); this.brand = C().brand;
    if (!this.gm) { $('#sc-ghost').hidden = true; $('#sc-gtag').hidden = true; this.ghostOn = false; }
    this.cv = $('#sc-canvas'); this.ctx = this.cv.getContext('2d'); this.stage = $('#sc-stage'); this.video = $('#sc-video');
    this.range = $('#sc-range'); this.range.max = Math.max(1, this.N - 1);
    const w = r._window; this.lo = w ? clamp(w.start, 0, this.N - 1) : 0; this.hi = w ? clamp(w.end, this.lo, this.N - 1) : this.N - 1;
    if (vurl) this.stage.style.setProperty('--ar', session.meta.width + '/' + session.meta.height);
    else { const bb = bodyBox(session, r); this.box = bb; this.stage.style.setProperty('--ar', Math.max(0.75, Math.min(1.2, bb.w * session.meta.width / (bb.h * session.meta.height) * 1.25)).toFixed(3)); }
    $('#sc-marks').innerHTML = ['set', 'release'].filter(k => fin(this.ev[k])).map(k => `<i style="left:${(clamp(this.ev[k], 0, this.N - 1) / Math.max(1, this.N - 1) * 100).toFixed(1)}%"></i>`).join('');
    this.frame = clamp(fin(this.ev.release) ? this.ev.release : 0, 0, this.N - 1);
    this.h = [];
    const on = (el, ev, fn) => { if (!el) return; el.addEventListener(ev, fn); this.h.push(() => el.removeEventListener(ev, fn)); };
    on(this.range, 'input', () => { this.pause(); this.set(+this.range.value); });
    on($('#sc-prev'), 'click', () => { this.pause(); this.set(this.frame - 1); });
    on($('#sc-next'), 'click', () => { this.pause(); this.set(this.frame + 1); });
    on($('#sc-play'), 'click', () => (this.playing ? this.pause() : this.play()));
    on($('#sc-ghost'), 'click', () => { this.ghostOn = !this.ghostOn; const g = $('#sc-ghost'); g.textContent = this.ghostOn ? 'Hide reference' : 'Show reference'; g.setAttribute('aria-pressed', this.ghostOn); $('#sc-gtag').hidden = !this.ghostOn; this.draw(); });
    $$('#results-body [data-speed]').forEach(b => on(b, 'click', () => { this.speed = +b.dataset.speed; $$('#results-body [data-speed]').forEach(x => x.classList.toggle('on', x === b)); if (this.video && this.playing) this.video.playbackRate = this.speed; }));
    $$('#kf [data-jump]').forEach(b => on(b, 'click', () => { this.pause(); this.set(this.ev[b.dataset.jump]); }));
    on(document, 'keydown', e => {
      if (current !== 'results' || (sheetEl && sheetEl.classList.contains('open'))) return;
      const t = document.activeElement && document.activeElement.tagName;
      if (['INPUT', 'TEXTAREA', 'SELECT'].includes(t) && document.activeElement !== this.range) return;
      if (e.key === 'ArrowRight') { this.pause(); this.set(this.frame + (e.shiftKey ? 5 : 1)); e.preventDefault(); }
      else if (e.key === 'ArrowLeft') { this.pause(); this.set(this.frame - (e.shiftKey ? 5 : 1)); e.preventDefault(); }
      else if (e.key === ' ' && t !== 'BUTTON' && t !== 'A') { this.playing ? this.pause() : this.play(); e.preventDefault(); }
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
      ctx.fillStyle = STAGE_BG; ctx.fillRect(0, 0, W, H);
    }
    const lw = Math.max(2.2, r.h / 150);
    if (this.ghostOn && this.gm) {
      const g = this.gm.at(this.frame);
      drawPose(ctx, g, (x, y) => { const q = this.gm.map(x, y); return map(q[0], q[1]); }, { col: STAGE_GHOST, w: lw * 0.85, dash: [lw * 2.2, lw * 1.8], halo: !!this.video });
    }
    const ulm = lmAt(this.s, this.r, this.frame);
    if (ulm) drawPose(ctx, ulm, map, { col: this.brand, w: lw, joints: true, arm: this.r.hand });
    else { ctx.fillStyle = 'rgba(235,235,245,.6)'; ctx.font = '500 13px -apple-system, system-ui, sans-serif'; ctx.textAlign = 'center'; ctx.fillText('No body found in this frame', W / 2, H / 2); }
  }
  phase(i) { const e = this.ev; if (i < e.onset) return 'stance'; if (i < e.set) return 'load'; if (i < e.release) return 'lift'; if (i === e.release) return 'release'; if (i <= e.ftEnd) return 'follow-through'; return 'recovery'; }
  readouts() {
    const i = this.frame, f = this.fr[i];
    $('#sc-time').textContent = f.t.toFixed(2) + ' s';
    $('#sc-phase').textContent = PHASE[this.phase(i)];
    if (+this.range.value !== i) this.range.value = i;
    $$('#kf [data-jump]').forEach(x => x.classList.toggle('on', this.ev[x.dataset.jump] === i));
    const ro = $('#sc-ro'); if (!ro || ro.hidden) return;
    const pf = this.r.perFrame || {}, e = this.ev;
    const span = Math.max(1, (e.ftEnd | 0) - (e.onset | 0));
    const k = clamp(Math.round((i - e.onset) / span * 59), 0, 59);
    const inShot = i >= e.onset && i <= e.ftEnd;
    let h = '';
    for (let j = 0; j < this.F.length; j++) {
      const F = this.F[j]; const arr = pf[F.key]; const u = arr ? num(arr[i]) : NaN;
      if (!Number.isFinite(u)) continue;
      const rv = this.rseq && inShot && this.rseq[k] ? num(this.rseq[k][j]) : NaN;
      h += `<div class="row"><div class="grow"><div class="title">${esc(F.label)}</div>${Number.isFinite(rv) ? `<div class="sub">Reference ${fmtNum(rv, F.unit)}</div>` : ''}</div><span class="value">${fmtNum(u, F.unit)}${F.unit === 'torso' ? ' torso' : ''}</span></div>`;
    }
    ro.innerHTML = h || '<div class="row"><div class="grow"><div class="sub">No measurements in this frame</div></div></div>';
  }
  play() {
    this.playing = true; const b = $('#sc-play'); b.innerHTML = ICONS.pause; b.setAttribute('aria-label', 'Pause');
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
    this.playing = false; const b = $('#sc-play'); if (b) { b.innerHTML = ICONS.play; b.setAttribute('aria-label', 'Play'); }
    if (this.video) { this.video.pause(); this.set(this.frame); }
    cancelAnimationFrame(this._raf);
  }
  destroy() { this.playing = false; cancelAnimationFrame(this._raf); if (this.video) this.video.pause(); this.ro.disconnect(); this.h.forEach(f => f()); }
}

// ------------------------------------------------------------------ share card + summary
function summaryText(session, r) {
  const nm = splitName(r.closest && r.closest.name);
  const fixes = topFixes(r, 2); const plan = r.plan || [];
  const lines = [`Jumpshot: mechanics score ${f1(mechScore(r))}/100`, verdict(r) + '.'];
  if (fixes.length) {
    lines.push('', 'What to fix:');
    fixes.forEach((m, i) => { const pl = plan.find(p => p.key === m.key); lines.push(`${i + 1}. ${fixCopy(m)[0]}. ${shortLabel(m)} ${fmtMech(m.value, m.unit, m.key)}, target ${fmtRange(m.band, m.unit)} (+${fmtPts(gainOf(m))} pts).${pl ? ' Drill: ' + pl.drill + ', ' + rx(pl.prescription) + '.' : ''}`); });
  }
  lines.push('', `Closest style: ${nm.title}${nm.style ? ' (' + nm.style + ')' : ''}, ${f1(styleScore(r))} match.`);
  if (sessResults(session).length > 1) { const sc = sessResults(session).map(mechScore).filter(Number.isFinite); if (sc.length > 1) lines.push(`Consistency over ${sc.length} shots: ${mean(sc).toFixed(1)} ± ${sd(sc).toFixed(1)}.`); }
  lines.push(`Analysed on-device, ${fmtDate(session.created)}.`);
  return lines.join('\n');
}
async function copySummary(session, r) {
  const txt = summaryText(session, r);
  try { await navigator.clipboard.writeText(txt); toast('Summary copied'); }
  catch (e) {
    const ta = document.createElement('textarea'); ta.value = txt; ta.style.cssText = 'position:fixed;opacity:0'; document.body.appendChild(ta); ta.select();
    try { document.execCommand('copy'); toast('Summary copied'); } catch (x) { toast("Couldn't copy"); } ta.remove();
  }
}
function wrapLines(ctx, text, maxW, maxLines) {
  const words = String(text).split(/\s+/); const out = []; let line = '';
  for (const w of words) { const t = line ? line + ' ' + w : w; if (ctx.measureText(t).width > maxW && line) { out.push(line); line = w; } else line = t; }
  if (line) out.push(line);
  if (out.length > maxLines) { out.length = maxLines; out[maxLines - 1] = out[maxLines - 1].replace(/\s*\S+$/, '') + '…'; }
  return out;
}
function rr(ctx, x, y, w, h, r) { ctx.beginPath(); ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r); ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath(); }
/** 1080×1350 dark card: big score, verdict, release pose, the one fix. Fixed dark palette so it reads anywhere. */
async function shareCard(session, r) {
  const W = 1080, H = 1350, c = document.createElement('canvas'); c.width = W; c.height = H; const ctx = c.getContext('2d');
  const F = '-apple-system,BlinkMacSystemFont,"SF Pro Display",Inter,"Segoe UI",Roboto,Helvetica,Arial,sans-serif';
  const K = { bg: '#000000', panel: '#1c1c1e', fill: '#2c2c2e', label: '#ffffff', l2: 'rgba(235,235,245,.68)', l3: 'rgba(235,235,245,.38)', brand: '#ff7a1a', pos: '#30d158', warn: '#ffb340', neg: '#ff453a' };
  const M = 72;
  ctx.fillStyle = K.bg; ctx.fillRect(0, 0, W, H);
  ctx.textBaseline = 'alphabetic'; ctx.textAlign = 'left';
  // header: app mark + name + date
  ctx.fillStyle = K.brand; rr(ctx, M, M, 64, 64, 16); ctx.fill();
  ctx.strokeStyle = '#1a0a00'; ctx.lineWidth = 3.4; ctx.lineCap = 'round';
  ctx.beginPath(); ctx.arc(M + 32, M + 32, 17, 0, 7); ctx.moveTo(M + 15, M + 32); ctx.lineTo(M + 49, M + 32); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(M + 32, M + 15); ctx.quadraticCurveTo(M + 22, M + 32, M + 32, M + 49); ctx.moveTo(M + 32, M + 15); ctx.quadraticCurveTo(M + 42, M + 32, M + 32, M + 49); ctx.stroke();
  ctx.fillStyle = K.label; ctx.font = `600 38px ${F}`; ctx.fillText('Jumpshot', M + 88, M + 30);
  ctx.fillStyle = K.l2; ctx.font = `400 28px ${F}`; ctx.fillText(fmtDate(session.created, { month: 'long', day: 'numeric', year: 'numeric' }), M + 88, M + 66);

  // release pose panel (right)
  const px = 620, py = 200, pw = 388, ph = 560;
  ctx.fillStyle = K.panel; rr(ctx, px, py, pw, ph, 36); ctx.fill();
  const fr = session.frames, ev = r.events || {}, ri = clamp(fin(ev.release) ? ev.release : 0, 0, fr.length - 1);
  const lm = lmAt(session, r, ri);
  if (lm) {
    const gm = ghostMapper(session, r); const pts = [];
    const asp = session.meta.width / session.meta.height;
    for (let i = 0; i < 33; i++) { const p = P(lm, i); if (p && p[2] > 0.3) pts.push([p[0] * asp, p[1]]); }
    let gl = null; if (gm) { const g = gm.at(ri); gl = []; for (let i = 0; i < 33; i++) { const p = P(g, i); gl.push(p && p[2] >= 0.3 ? gm.map(p[0], p[1]) : null); } gl.forEach(p => p && pts.push([p[0] * asp, p[1]])); }
    const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]); const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
    const s = Math.min((pw - 100) / ((maxX - minX) || 1), (ph - 150) / ((maxY - minY) || 1));
    const ox = px + pw / 2 - (minX + maxX) / 2 * s, oy = py + ph - 80 - maxY * s;
    const map = (x, y) => [ox + x * asp * s, oy + y * s];
    if (gl) { const flat = new Float32Array(132); gl.forEach((p, i) => { if (p) { flat[i * 4] = p[0]; flat[i * 4 + 1] = p[1]; flat[i * 4 + 3] = 1; } }); drawPose(ctx, flat, map, { col: K.l3, w: 7, dash: [14, 12], halo: false }); }
    drawPose(ctx, lm, map, { col: K.brand, w: 11, joints: true, arm: r.hand, halo: false });
    ctx.fillStyle = K.l2; ctx.font = `500 26px ${F}`; ctx.textAlign = 'center'; ctx.fillText('At release', px + pw / 2, py + ph - 32); ctx.textAlign = 'left';
  }

  // score + verdict (left)
  const msc = mechScore(r), mk = hasMech(r) ? r.mechanics : [];
  ctx.fillStyle = K.l2; ctx.font = `500 32px ${F}`; ctx.fillText('Mechanics score', M, 268);
  { const sc = fin(msc) ? msc.toFixed(1) : '—'; let fs = 200; ctx.font = `700 ${fs}px ${F}`; while (ctx.measureText(sc).width > 420 && fs > 110) { fs -= 10; ctx.font = `700 ${fs}px ${F}`; }
    ctx.fillStyle = K.label; ctx.fillText(sc, M - 6, 448); const sw = ctx.measureText(sc).width; if (fin(msc)) { ctx.fillStyle = K.l2; ctx.font = `500 44px ${F}`; ctx.fillText('/100', M + sw + 4, 448); } }
  ctx.fillStyle = K.label; ctx.font = `600 42px ${F}`;
  let y = 530; wrapLines(ctx, verdict(r), 500, 3).forEach(l => { ctx.fillText(l, M, y); y += 54; });
  // checks strip
  if (mk.length) {
    const tot = mk.reduce((q, m) => q + (num(m.maxPoints) || 0), 0) || 1; let x = M; const sw = 500, sy = Math.max(y + 10, 700);
    mk.forEach(m => { const w = (num(m.maxPoints) || 0) / tot * sw; ctx.fillStyle = K.fill; rr(ctx, x, sy, Math.max(2, w - 6), 12, 6); ctx.fill();
      if (m.status !== 'unmeasured') { ctx.fillStyle = { good: K.pos, close: K.warn, off: K.neg }[m.status] || K.l3; const fw = Math.max(0, (w - 6) * clamp(num(m.points) / (num(m.maxPoints) || 1), 0, 1)); if (fw > 1) { rr(ctx, x, sy, Math.max(12, fw), 12, 6); ctx.fill(); } }
      x += w; });
    const nM = mk.filter(m => m.status !== 'unmeasured').length, nG = mk.filter(m => m.status === 'good').length;
    ctx.fillStyle = K.l2; ctx.font = `400 28px ${F}`; ctx.fillText(`${nG} of ${nM} checks in range`, M, sy + 56);
  }

  // fix this first
  const top = topFixes(r, 1)[0]; const by = 820, bh = 390;
  ctx.fillStyle = K.panel; rr(ctx, M, by, W - 2 * M, bh, 36); ctx.fill();
  const ix = M + 44;
  if (top) {
    const [head] = fixCopy(top); const pl = (r.plan || []).find(p => p.key === top.key);
    ctx.fillStyle = K.brand; ctx.font = `600 28px ${F}`; ctx.fillText('Fix this first', ix, by + 66);
    ctx.fillStyle = K.label; ctx.font = `700 50px ${F}`; ctx.fillText(head, ix, by + 132, W - 2 * M - 88);
    const cols = [[fmtMech(top.value, top.unit, top.key), 'You'], [fmtRange(top.band, top.unit), 'Target'], ['+' + fmtPts(gainOf(top)), 'Points to gain']];
    let cx = ix; cols.forEach(([v, l]) => { ctx.fillStyle = K.label; ctx.font = `600 46px ${F}`; ctx.fillText(v, cx, by + 222); const w = ctx.measureText(v).width; ctx.fillStyle = K.l2; ctx.font = `400 26px ${F}`; ctx.fillText(l, cx, by + 260); cx += Math.max(w, ctx.measureText(l).width) + 64; });
    if (pl) { ctx.fillStyle = K.fill; ctx.fillRect(ix, by + 296, W - 2 * M - 88, 2); ctx.fillStyle = K.l2; ctx.font = `400 30px ${F}`; ctx.fillText(`Drill: ${pl.drill}, ${rx(pl.prescription)}`, ix, by + 350, W - 2 * M - 88); }
  } else {
    ctx.fillStyle = K.label; ctx.font = `700 50px ${F}`; ctx.fillText(mk.length ? 'Nothing to fix' : 'Mechanics not scored', ix, by + 120);
    ctx.fillStyle = K.l2; ctx.font = `400 32px ${F}`; ctx.fillText(mk.length ? 'Every measured check is in range.' : 'Analyse the clip again for mechanics checks.', ix, by + 176, W - 2 * M - 88);
  }
  // footer
  const nm = splitName(r.closest && r.closest.name);
  ctx.fillStyle = K.l2; ctx.font = `400 28px ${F}`; ctx.fillText(`Closest style: ${nm.title}, ${f1(styleScore(r))} match`, M, H - 66, 640);
  ctx.textAlign = 'right'; ctx.fillStyle = K.l3; ctx.fillText('Analysed on device', W - M, H - 66); ctx.textAlign = 'left';
  return new Promise(res => c.toBlob(b => res(b), 'image/png'));
}
async function openShare(session, r) {
  const d = sheet('Share', '<div class="share-prev skel"></div>');
  const blob = await shareCard(session, r);
  const url = URL.createObjectURL(blob);
  const fname = `jumpshot-${fin(mechScore(r)) ? Math.round(mechScore(r)) : 'style' + Math.round(styleScore(r) || 0)}-${session.created.slice(0, 10)}.png`;
  const file = new File([blob], fname, { type: 'image/png' });
  const canShare = !!(navigator.canShare && navigator.canShare({ files: [file] }));
  d.querySelector('.sheet-bd').innerHTML = `<div class="share-prev"><img id="share-img" src="${url}" alt="Share image: mechanics score ${f1(mechScore(r))}"></div>
    <div class="acts">${canShare ? '<button class="btn primary block" id="sh-share" type="button">Share image</button>' : ''}
      <a class="btn block ${canShare ? '' : 'primary'}" id="sh-dl" href="${url}" download="${fname}">Save image</a>
      <button class="btn plain" id="sh-copy" type="button">Copy summary as text</button></div>
    <p class="section-foot center">Only the skeleton is drawn, never your video.</p>`;
  window.__jumpshot.lastShare = { blob, size: blob.size, canShare };
  const sb = $('#sh-share'); if (sb) sb.onclick = async () => { try { await navigator.share({ files: [file], title: 'My jumpshot', text: summaryText(session, r) }); } catch (e) { if (e.name !== 'AbortError') toast("Couldn't share. Use Save image."); } };
  $('#sh-copy').onclick = () => copySummary(session, r);
}

// ------------------------------------------------------------------ history
function primaryResult(s) { return (s.results && s.results[s.primary || 0]) || s.result; }
let histList = [];
async function renderHistory() {
  const box = $('#history-body');
  if (!histList.length) box.innerHTML = '<div class="skel" style="height:220px;border-radius:16px;margin-top:24px"></div><div class="skel" style="height:64px;border-radius:16px;margin-top:32px"></div>';
  const list = await DB.all(); histList = list;
  state.compare = state.compare.filter(id => list.some(s => s.id === id));
  const mo = $('#tb-more'); if (mo) mo.onclick = () => openHistoryMenu(list);
  if (!list.length) {
    state.editing = false; chrome('history');
    box.innerHTML = `<div class="empty" style="margin-top:24px">${ICONS.history}<h3>No sessions yet</h3><p>Each shot you analyse is saved here, on this device only.</p><a class="btn primary" href="#/">Analyze a shot</a></div>`;
    return;
  }
  const sc = list.map(sessScores);
  const mf = sc.map(x => x.mech).filter(Number.isFinite);
  const first = mf[0], last = mf[mf.length - 1], best = mf.length ? Math.max(...mf) : NaN;
  const d = mf.length > 1 ? last - first : NaN;
  let eng = null, sum = null; try { eng = await getEngine(); sum = eng.summarizeHistory(list.map(s => { const r = primaryResult(s) || {}; return { ...r, score: fin(styleScore(r)) ? styleScore(r) : num(r.score) }; })); } catch (e) { console.warn('summarizeHistory', e); }
  const short = l => String(l || '').replace(/ \(.*\)$/, '');
  const imp = sum && Array.isArray(sum.mostImproved) ? sum.mostImproved[0] : null, worse = sum && Array.isArray(sum.trendingWorse) ? sum.trendingWorse[0] : null;
  const rows = list.map((s, i) => ({ s, sc: sc[i], prev: i > 0 ? sc[i - 1].mech : NaN })).reverse();
  box.innerHTML = `<section class="section" style="margin-top:24px">
      <div class="card hist-top">
        <div class="head"><div class="stat"><div class="v num">${f1(last)}<small>/100</small></div><div class="c">Latest mechanics score</div></div>
          ${fin(d) ? `<div class="delta ${d >= 0 ? 'pos' : 'neg'}">${d >= 0 ? '▲' : '▼'} ${Math.abs(d).toFixed(1)}<small>since your first session</small></div>` : ''}</div>
        <div class="hist-chart" id="hist-chart">${chartSVG(list, sc)}</div>
      </div>
      <p class="section-foot">${fin(best) ? `Best ${best.toFixed(1)}. ` : ''}${list.length} session${list.length === 1 ? '' : 's'} on this device.${mf.length > 2 ? ` Consistency ±${sd(mf).toFixed(1)}.` : ''}</p>
    </section>
    <section class="section"><div class="section-h"><h2>Sessions</h2><button class="more" id="h-edit" type="button">${state.editing ? 'Done' : 'Select'}</button></div>
      <div class="list" id="hist-list">${rows.map(x => sessRow(x.s, x.sc, x.prev)).join('')}</div>
      ${state.editing ? '<p class="section-foot">Select two sessions to compare them.</p>' : ''}</section>
    ${imp || worse ? `<section class="section"><div class="section-h"><h2>Trends</h2></div><div class="list">
      ${imp ? `<div class="row"><div class="grow"><div class="title wrap">${esc(short(imp.label || imp.key))}</div><div class="sub">Most improved</div></div><span class="pos">▲</span></div>` : ''}
      ${worse ? `<div class="row"><div class="grow"><div class="title wrap">${esc(short(worse.label || worse.key))}</div><div class="sub">Slipping</div></div><span class="neg">▼</span></div>` : ''}
    </div></section>` : ''}
    <div class="pin" id="edit-bar"${state.editing ? '' : ' hidden'}><div class="edit-bar"><button class="btn plain danger" id="cmp-del" type="button">Delete</button><span id="cmp-txt"></span><button class="btn plain" id="cmp-go" type="button">Compare</button></div></div>`;
  wireChart(list, sc);
  $('#h-edit').onclick = () => { state.editing = !state.editing; if (!state.editing) state.compare = []; chrome('history'); renderHistory(); };
  $$('#hist-list .srow').forEach(el => el.onclick = () => {
    const id = el.dataset.id;
    if (!state.editing) return go('#/session/' + id);
    state.compare = state.compare.filter(x => x !== id); if (!el.classList.contains('selected')) state.compare.push(id);
    if (state.compare.length > 2) state.compare.shift(); sync();
  });
  const sync = () => {
    $$('#hist-list .srow').forEach(el => { const on = state.compare.includes(el.dataset.id); el.classList.toggle('selected', on); el.setAttribute('aria-pressed', on); });
    const n = state.compare.length; const t = $('#cmp-txt'); if (!t) return;
    t.textContent = n ? `${n} selected` : 'Select sessions';
    $('#cmp-go').disabled = n !== 2; $('#cmp-del').disabled = !n;
  };
  $('#cmp-go').onclick = () => { const [a, b] = state.compare.map(id => list.find(s => s.id === id)).sort((x, y) => (x.created < y.created ? -1 : 1)); openCompare(a.id, b.id); };
  $('#cmp-del').onclick = async () => { const n = state.compare.length; if (!confirm(`Delete ${n} session${n > 1 ? 's' : ''}?`)) return; for (const id of state.compare) await DB.del(id); state.compare = []; toast(n > 1 ? 'Sessions deleted' : 'Session deleted'); renderHistory(); };
  sync();
}
function sessRow(s, sc, prev) {
  const n = sessResults(s).length; const d = fin(sc.mech) && fin(prev) ? sc.mech - prev : NaN;
  return `<button class="row srow" type="button" data-id="${esc(s.id)}"${state.editing ? ' aria-pressed="false"' : ''}>
    ${state.editing ? `<span class="pick" aria-hidden="true">${ICONS.check}</span>` : ''}
    ${s.thumb ? `<img src="${esc(s.thumb)}" alt="">` : '<span class="noimg"></span>'}
    <div class="grow"><div class="title">${esc(s.name || 'Shot')}</div><div class="sub">${esc(fmtWhen(s.created))}${n > 1 ? `, ${n} shots` : ''}</div></div>
    <div class="right"><div class="v">${f1(sc.mech)}</div>${fin(d) && Math.abs(d) >= 0.05 ? `<div class="d ${d > 0 ? 'pos' : 'neg'}">${d > 0 ? '▲' : '▼'} ${Math.abs(d).toFixed(1)}</div>` : ''}</div>
    ${state.editing ? '' : ICONS.chev}</button>`;
}
const CH = { W: 340, H: 150, L: 8, R: 312, T: 22, B: 118 };
function chartSVG(list, scs) {
  const cw = Math.round(($('#history-body').clientWidth || 360) - 32); CH.W = clamp(cw, 280, 1000); CH.R = CH.W - 28;
  const { W, H, L, R, T, B } = CH;
  const sc = scs.map(x => x.mech); const f = sc.filter(Number.isFinite); if (!f.length) f.push(50);
  let lo = Math.floor((Math.min(...f) - 4) / 5) * 5, hi = Math.ceil((Math.max(...f) + 4) / 5) * 5; if (hi - lo < 20) { const m = (hi + lo) / 2; lo = Math.round(m - 10); hi = Math.round(m + 10); } lo = Math.max(0, lo); hi = Math.min(100, hi);
  const x = i => (list.length === 1 ? (L + R) / 2 : L + i / (list.length - 1) * (R - L)), y = v => B - (v - lo) / ((hi - lo) || 1) * (B - T);
  CH.x = x; CH.y = y;
  const pts = sc.map((v, i) => [x(i), y(v), i]).filter(p => Number.isFinite(p[1]));
  let s = `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Mechanics score over ${list.length} sessions">`;
  [lo, hi].forEach(v => { s += `<line class="grid" x1="${L}" x2="${R}" y1="${y(v).toFixed(1)}" y2="${y(v).toFixed(1)}"/><text x="${W - 2}" y="${(y(v) + 4).toFixed(1)}" text-anchor="end">${v}</text>`; });
  if (pts.length > 1) s += `<polygon class="area" points="${pts[0][0].toFixed(1)},${B} ${pts.map(p => p[0].toFixed(1) + ',' + p[1].toFixed(1)).join(' ')} ${pts[pts.length - 1][0].toFixed(1)},${B}"/><polyline class="series" points="${pts.map(p => p[0].toFixed(1) + ',' + p[1].toFixed(1)).join(' ')}"/>`;
  pts.forEach((p, k) => { const lastP = k === pts.length - 1; s += `<circle cx="${p[0].toFixed(1)}" cy="${p[1].toFixed(1)}" r="${lastP ? 4.5 : 2.5}" style="fill:${lastP ? 'var(--brand)' : 'var(--surface)'};stroke:var(--brand)" stroke-width="2"/>`; });
  const dl = (i, anchor) => `<text x="${x(i).toFixed(1)}" y="${H - 6}" text-anchor="${anchor}">${esc(fmtDate(list[i].created, { month: 'short', day: 'numeric' }))}</text>`;
  if (list.length > 1) s += dl(0, 'start') + dl(list.length - 1, 'end'); else s += dl(0, 'middle');
  s += `<g id="ch-hover" style="display:none"><line class="rule" y1="${T - 6}" y2="${B}"/><circle r="5" style="fill:var(--brand);stroke:var(--surface)" stroke-width="2"/><text class="callout" text-anchor="middle"></text></g>`;
  return s + '</svg>';
}
function wireChart(list, scs) {
  const svg = $('#hist-chart svg'); if (!svg) return;
  const g = svg.querySelector('#ch-hover'), line = g.querySelector('line'), dot = g.querySelector('circle'), txt = g.querySelector('text');
  const sc = scs.map(x => x.mech);
  const show = e => {
    const r = svg.getBoundingClientRect(); const vx = (e.clientX - r.left) / r.width * CH.W;
    let bi = -1, bd = 1e9; sc.forEach((v, i) => { if (!Number.isFinite(v)) return; const d = Math.abs(CH.x(i) - vx); if (d < bd) { bd = d; bi = i; } });
    if (bi < 0) return; const X = CH.x(bi), Y = CH.y(sc[bi]);
    g.style.display = ''; line.setAttribute('x1', X); line.setAttribute('x2', X); dot.setAttribute('cx', X); dot.setAttribute('cy', Y);
    txt.setAttribute('x', clamp(X, 40, CH.R - 20)); txt.setAttribute('y', CH.T - 8 < 10 ? 10 : CH.T - 8);
    txt.textContent = `${sc[bi].toFixed(1)}, ${fmtDate(list[bi].created, { month: 'short', day: 'numeric' })}`;
  };
  svg.addEventListener('pointermove', show); svg.addEventListener('pointerdown', show);
  svg.addEventListener('pointerleave', () => { g.style.display = 'none'; });
}
function openHistoryMenu(list) {
  const n = list.length;
  const d = sheet('History', `<div class="list menu">
      <button class="row" type="button" data-m="json"${n ? '' : ' disabled'}><div class="grow"><div class="title">Export as JSON</div><div class="sub">Back up or move to another device</div></div></button>
      <button class="row" type="button" data-m="csv"${n ? '' : ' disabled'}><div class="grow"><div class="title">Export as CSV</div><div class="sub">Open in a spreadsheet</div></div></button>
      <button class="row" type="button" data-m="imp"><div class="grow"><div class="title">Import from JSON</div></div></button>
    </div>
    ${n ? '<div class="list menu" style="margin-top:24px"><button class="row" type="button" data-m="clear"><div class="grow"><div class="title danger">Delete all sessions</div></div></button></div>' : ''}
    <p class="section-foot">Sessions are stored in this browser only: landmarks, results and a small skeleton thumbnail. No video is stored.</p>`);
  $$('[data-m]', d).forEach(b => b.onclick = async () => {
    const m = b.dataset.m; closeSheet();
    if (m === 'json') exportJSON(list);
    else if (m === 'csv') exportCSV(list);
    else if (m === 'imp') $('#file-import').click();
    else if (m === 'clear') { if (!confirm('Delete all sessions on this device? Export first if you want a backup.')) return; await DB.clear(); state.compare = []; state.editing = false; toast('History cleared'); renderHistory(); }
  });
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
      fb.filter(f => f.severity === 'major').length, fb.filter(f => f.severity === 'minor').length, s.model || '', ...ks.map(k => (fin((r.summary || {})[k]) ? num(r.summary[k]).toFixed(4) : ''))].map(q).join(','); });
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
    toast(`Imported ${n} session${n === 1 ? '' : 's'}`);
    if (current === 'history') renderHistory(); else go('#/history');
  } catch (err) { showError({ icon: 'film', title: "That file couldn't be imported", msg: 'Use a file made with Export as JSON in History.', actions: [{ label: 'Back to History', primary: true, go: '#/history' }], code: String(err.message) }); }
});

// ------------------------------------------------------------------ compare (sheet)
async function openCompare(ida, idb) {
  const [A, B] = await Promise.all([DB.get(ida), DB.get(idb)]);
  if (!A || !B) { toast('Session not found'); return; }
  const ra = primaryResult(A), rb = primaryResult(B);
  const ma = mechScore(ra), mb = mechScore(rb);
  const dv = (a, b, dp = 1) => { if (!fin(a) || !fin(b)) return '<span class="d muted">—</span>'; const d = b - a; if (Math.abs(d) < 0.05) return '<span class="d muted">0</span>'; return `<span class="d ${d > 0 ? 'pos' : 'neg'}">${d > 0 ? '+' : MINUS}${Math.abs(d).toFixed(dp)}</span>`; };
  const head = s => `<div class="cmp-head">${s.thumb ? `<img src="${esc(s.thumb)}" alt="">` : ''}<div style="min-width:0"><b>${esc(s.name || 'Shot')}</b><small>${esc(fmtDate(s.created, { month: 'short', day: 'numeric' }))}</small></div></div>`;
  const total = fin(ma) && fin(mb) ? mb - ma : NaN;
  let checks = '';
  if (hasMech(ra) || hasMech(rb)) {
    const keys = (hasMech(rb) ? rb.mechanics : ra.mechanics).map(m => m.key);
    const get = (r, k) => (hasMech(r) ? r.mechanics.find(m => m.key === k) : null);
    const pv = m => (m && m.status !== 'unmeasured' ? num(m.points) : NaN);
    checks = `<div class="sec"><h4>Points per check</h4><div class="cmp-colh"><span>Then</span><span>Now</span><span class="d">Change</span></div><div class="list">${keys.map(k => { const a = get(ra, k), b = get(rb, k), m = b || a;
      return `<div class="row cmp-row"><div class="grow"><div class="title">${esc(shortLabel(m))}</div></div><div class="vals"><span>${fmtPts(pv(a))}</span><span>${fmtPts(pv(b))}</span>${dv(pv(a), pv(b))}</div></div>`; }).join('')}</div></div>`;
  }
  const kk = ['load', 'relh', 'elbow', 'knee', 'jump', 'lean'];
  const style = kk.filter(k => fin(metric(ra, k)) || fin(metric(rb, k))).map(k => { const u = MUNIT[k]; const va = metric(ra, k), vb = metric(rb, k);
    return `<div class="row cmp-row"><div class="grow"><div class="title">${MLABEL[k]}</div></div><div class="vals"><span>${fmtVal(va, u === 'torso' ? '' : u)}</span><span>${fmtVal(vb, u === 'torso' ? '' : u)}</span><span class="d muted">${fin(va) && fin(vb) ? fmtVal(vb - va, u === 'torso' ? '' : u, true) : '—'}</span></div></div>`; }).join('');
  const radar = ra.radar && rb.radar && ra.radar.axes && rb.radar.axes ? radarSVG({ axes: rb.radar.axes, a: ra.radar.user, b: rb.radar.user }, [{ k: 'a', col: 'var(--label-3)', w: 1.5, dash: '4 4' }, { k: 'b', col: 'var(--brand)', fill: 'var(--brand-tint)', w: 2, dots: true }]) : '';
  sheet('Compare', `<div class="cmp-heads">${head(A)}${head(B)}</div>
    <div class="sheet-big"><b>${fin(total) ? (total >= 0 ? '+' : MINUS) + Math.abs(total).toFixed(1) : '—'}</b><span class="muted">${fin(total) ? `mechanics score, ${f1(ma)} to ${f1(mb)}` : 'Mechanics score not available for both'}</span></div>
    ${checks}
    ${style ? `<div class="sec"><h4>Measurements</h4><div class="cmp-colh"><span>Then</span><span>Now</span><span class="d">Change</span></div><div class="list">${style}</div></div>` : ''}
    ${radar ? `<div class="sec"><h4>Style profile</h4><div class="radar-wrap">${radar}</div><div class="legend"><span><i class="dash"></i>${esc(fmtDate(A.created, { month: 'short', day: 'numeric' }))}</span><span><i style="background:var(--brand)"></i>${esc(fmtDate(B.created, { month: 'short', day: 'numeric' }))}</span></div></div>` : ''}
    <div class="acts"><a class="btn block" href="#/session/${esc(B.id)}">Open newer session</a></div>`);
}

// ------------------------------------------------------------------ errors
function showError(e) { state.error = e; go('#/error'); if (current === 'error') renderError(); }
function renderError() {
  const e = state.error; const box = $('#error-body');
  const acts = (e.actions || [{ label: 'Back to Jumpshot', primary: true, go: '#/' }]).slice(0, 2);
  box.innerHTML = `<div class="empty err"><div class="ic">${ICONS[e.icon] || ICONS.warn}</div><h3 id="h-err">${esc(e.title)}</h3><p>${esc(e.msg || '')}</p>
    ${e.fixes && e.fixes.length ? `<ul>${e.fixes.slice(0, 3).map(f => `<li>${f}</li>`).join('')}</ul>` : ''}
    <div class="acts">${acts.map((a, i) => `<button class="btn ${i === 0 ? 'primary' : 'plain'}" type="button" data-ea="${i}">${esc(a.label)}</button>`).join('')}</div>
    ${e.code ? `<details><summary>Technical details</summary><code>${esc(e.code)}</code></details>` : ''}</div>`;
  $$('[data-ea]', box).forEach(b => b.onclick = () => { const a = acts[+b.dataset.ea]; if (a.click) a.click(); if (a.go) go(a.go); });
}
const iosHint = /iPhone|iPad|iPod/.test(navigator.userAgent);
function errCodec(e) {
  return { icon: 'film', title: "This video can't be played here", msg: 'Your browser couldn’t decode it. iPhone “High Efficiency” video is the usual cause.',
    fixes: [iosHint ? 'Open this page in Safari, which plays iPhone video.' : 'Open this page in Safari, or a browser that plays HEVC video.', 'Or set Settings, Camera, Formats to Most Compatible and film again.'],
    actions: [{ label: 'Choose another clip', primary: true, go: '#/', click: () => setTimeout(() => $('#file-upload').click(), 50) }, { label: 'Record instead', click: () => { go('#/'); setTimeout(() => $('#btn-record').click(), 50); } }], code: e && e.message };
}
function errNoPerson(e) {
  return { icon: 'person', title: "We couldn't find you in the clip", msg: `A body showed up in only ${e.nPose} of ${e.n} frames.`,
    fixes: ['Get your whole body in frame, feet to raised hands.', 'Film in good light, without a bright window behind you.', 'Make sure the trim covers the part with you in it.'],
    actions: [{ label: 'Adjust trim', primary: true, go: '#/clip' }, { label: 'Try high accuracy', click: () => { prefs.model = 'full'; savePrefs(); go('#/clip'); } }] };
}
function errSmall(e) {
  return { icon: 'person', title: "You're too small in the frame", msg: `You fill about ${Math.round(e.h * 100)}% of the frame height. Angles get unreliable below about 25%.`,
    fixes: ['Move the phone closer, 3–5 m away, or zoom in before recording.'],
    actions: [{ label: 'Analyze anyway', primary: true, click: () => { state._ignoreSmall = true; go('#/clip'); setTimeout(() => $('#btn-analyze').click(), 300); } }, { label: 'Choose another clip', go: '#/' }] };
}
function errModel(e) {
  if (e.code === 'simd') return { icon: 'warn', title: 'This browser is too old', msg: 'The pose model needs Safari 16.4, Chrome 91 or Firefox 89 or newer.', actions: [{ label: 'Try a sample', primary: true, click: runSample }, { label: 'Back to Jumpshot', go: '#/' }], code: e.message };
  if (!navigator.onLine) {
    const have = safeJSONArr(localStorage.getItem('jumpshot.models')).filter(m => m !== prefs.model);
    const name = m => (m === 'full' ? 'High accuracy' : 'Fast');
    return { icon: 'offline', title: "You're offline", msg: have.length ? `The ${name(prefs.model)} model isn't on this device yet, but ${name(have[0])} is and works offline.` : 'Connect once to download the pose model. After that, everything works offline.',
      actions: (have.length ? [{ label: `Use ${name(have[0])}`, primary: true, click: () => { prefs.model = have[0]; savePrefs(); go('#/clip'); setTimeout(() => $('#btn-analyze').click(), 300); } }] : []).concat([{ label: 'Try a sample', primary: !have.length, click: runSample }]) };
  }
  return { icon: 'offline', title: "The pose model didn't load", msg: 'Check your connection and try again. It downloads once, then works offline.',
    fixes: ['Close other heavy tabs, then retry.', 'Private windows can block storage. Try a normal window.'],
    actions: [{ label: 'Try again', primary: true, click: () => { go('#/clip'); setTimeout(() => $('#btn-analyze').click(), 300); } }, { label: 'Try a sample', click: runSample }], code: e.message + (e.cause ? ' (' + (e.cause.message || e.cause) + ')' : '') };
}
function errAnalysis(r) {
  return { icon: 'person', title: "We couldn't measure this shot", msg: 'The clip didn’t show a complete shot from dip to follow-through.',
    fixes: ((r && r.quality && r.quality.flags) || []).slice(0, 1).map(f => esc(f.text || f.message || f.code)).concat(['Trim from just before the dip to after the follow-through.', 'Film side-on with your whole body and the ball in frame.']),
    actions: [{ label: state.source === 'sample' ? 'Back to Jumpshot' : 'Adjust trim', primary: true, go: state.source === 'sample' ? '#/' : '#/clip' }, { label: 'Choose another clip', go: '#/' }], code: r && r.reason };
}
function errUnknown(e) { return { icon: 'warn', title: 'Something went wrong', msg: 'The analysis stopped unexpectedly. Try again, or with a shorter trim.', actions: [{ label: state.file ? 'Back to trim' : 'Back to Jumpshot', primary: true, go: state.file ? '#/clip' : '#/' }, { label: 'Reload', click: () => location.reload() }], code: e && (e.stack || e.message || String(e)) }; }

// ------------------------------------------------------------------ boot
document.documentElement.classList.toggle('touch', COARSE);
route();
// warm the engine + assets in the background (tiny; makes the first analysis snappier)
setTimeout(() => { getAssets().catch(() => {}); }, 1200);
