// Parity + robustness tests for engine.mjs vs the Python pipeline (01-jumpshot-analyzer).
//   node site/jumpshot/app/tests/parity.test.mjs
// No npm dependencies. Fixtures are written by `python cli.py export-web` / `demo --site`.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { analyze, detectShots, summarizeHistory, prepareAssets, loadAssets, FEATURES } from '../engine.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const APP = path.resolve(HERE, '..');
const FX = path.join(HERE, 'fixtures');
const J = p => JSON.parse(fs.readFileSync(p, 'utf8'));

let passed = 0, failed = 0;
const results = [];
function test(name, fn) {
  try { fn(); passed++; results.push(`  ok   ${name}`); }
  catch (e) { failed++; results.push(`  FAIL ${name}\n       ${e.message}`); }
}
function assert(c, msg) { if (!c) throw new Error(msg); }
const near = (a, b, tol, what) => assert(Number.isFinite(a) && Number.isFinite(b) && Math.abs(a - b) <= tol, `${what}: ${a} vs ${b} (tol ${tol})`);
const cos = (a, b) => { let s = 0, na = 0, nb = 0; for (let i = 0; i < a.length; i++) { s += a[i] * b[i]; na += a[i] ** 2; nb += b[i] ** 2; } return s / Math.sqrt(na * nb); };

const library = J(path.join(APP, 'library.json'));
const drills = J(path.join(APP, 'drills.json'));
const encoder = fs.existsSync(path.join(APP, 'encoder.json')) ? J(path.join(APP, 'encoder.json')) : null;
const assets = prepareAssets({ library, drills, encoder });
const index = J(path.join(FX, '_index.json'));
const SUMMARY_TOL = { deg: 0.5, torso: 0.01, s: 0.035 };   // s: one frame at 30 fps
const metricUnit = Object.fromEntries(library.metrics.map(m => [m.key, m.unit]));
const extraUnit = { knee_angle_set: 'deg', release_time_from_onset_s: 's', peak_wrist_velocity_torso_per_s: 'torso', shot_duration_s: 's' };

// ------------------------------------------------------------------ assets
test('FEATURES constant matches library.json features', () => {
  assert(FEATURES.length === library.features.length, 'length');
  FEATURES.forEach((f, i) => { const g = library.features[i]; for (const k of ['key', 'label', 'unit', 'tolerance', 'higherIsBetter']) assert(f[k] === g[k], `${f.key}.${k}: ${f[k]} vs ${g[k]}`); });
});
test(`learned encoder loaded (${encoder ? 'encoder.json present' : 'no encoder.json'})`, () => assert(!!assets.encoder === !!encoder, 'encoder state'));
if (encoder) {
  test('reference embeddings: cosine >= 0.999 vs PyTorch', () => {
    for (const r of assets.refs) { const c = cos(r.emb, index.refEmbeddings[r.name]); assert(c >= 0.999, `${r.name}: cos ${c}`); }
  });
}

// ------------------------------------------------------------------ per-clip parity
const parityRows = [];
const embCos = [];
const resultsBySlug = {};
const MECH_TOL = { deg: 0.5, s: 0.035, torso: 0.01 };
function parityFixture(dir, clip, tag = '') {
  const fx = J(path.join(dir, `${clip.slug}.json`));
  const e = fx.expected;
  const r = analyze(fx.frames, fx.meta, assets, { autoWindow: false });
  const T = `[${tag}${clip.slug}]`;
  if (!tag) resultsBySlug[clip.slug] = r;
  if (e.error) { test(`${T} Python rejects the clip -> JS ok:false`, () => assert(!r.ok, `JS ok with score ${r.score}; Python: ${e.error}`)); return; }
  const partial = e.quality.codes.some(c => c === 'release_not_found' || c === 'events_degenerate');
  if (partial) { test(`${T} Python flags a partial shot (${e.quality.codes}) -> JS ok:false with a reason`, () => assert(!r.ok && r.reason, `ok=${r.ok}`)); return; }
  test(`${T} analyze ok`, () => assert(r.ok, r.reason));
  if (!r.ok) return;
  test(`${T} hand / facing / mirrored`, () => {
    assert(r.hand === e.hand, `hand ${r.hand} vs ${e.hand}`); assert(r.facing === e.facing, `facing ${r.facing} vs ${e.facing}`); assert(r.mirrored === e.mirrored, 'mirrored');
  });
  test(`${T} events within ±1 frame`, () => {
    for (const k of ['onset', 'set', 'release', 'ftEnd', 'peak', 'followThrough', 'peakExtension']) near(r.events[k], e.events[k], 1, `event ${k}`);
    assert(r.events.releaseFound === e.events.releaseFound, 'releaseFound');
    assert(r.events.setMethod === e.events.setMethod, `setMethod ${r.events.setMethod} vs ${e.events.setMethod}`);
  });
  test(`${T} summary metrics within tolerance`, () => {
    for (const [k, v] of Object.entries(e.summary)) near(r.summary[k], v, SUMMARY_TOL[metricUnit[k] || extraUnit[k] || 'deg'] * (k.startsWith('peak_') ? 10 : 1), `summary ${k}`);
  });
  test(`${T} per-frame features`, () => {
    for (const f of FEATURES) {
      const a = r.perFrame[f.key], b = e.perFrame[f.key];
      assert(a.length === b.length, `${f.key} length ${a.length} vs ${b.length}`);
      const tol = f.unit === 'deg' ? 0.05 : 0.002;
      for (let i = 0; i < b.length; i++) near(a[i], b[i], tol, `${f.key}[${i}]`);
    }
  });
  test(`${T} seq60`, () => { for (let i = 0; i < 60; i++) for (let j = 0; j < 9; j++) near(r.seq60[i][j], e.seq60[i][j], FEATURES[j].unit === 'deg' ? 0.05 : 0.002, `seq60[${i}][${j}]`); });
  test(`${T} mechanics checks (value, status, points, text)`, () => {
    assert(r.mechanics.length === e.mechanics.length, 'count');
    e.mechanics.forEach((pm, i) => {
      const jm = r.mechanics[i];
      assert(jm.key === pm.key, `order ${jm.key} vs ${pm.key}`);
      assert(jm.status === pm.status, `${pm.key} status ${jm.status} vs ${pm.status} (js ${jm.value}, py ${pm.value})`);
      if (pm.value != null) near(jm.value, pm.value, MECH_TOL[jm.unit], `${pm.key} value`);
      near(jm.points, pm.points, 0.5, `${pm.key} points`);
      assert(jm.text === pm.text, `${pm.key} text:\n         js: ${jm.text}\n         py: ${pm.text}`);
    });
  });
  test(`${T} headline (mechanics) score + style match within ±1.0`, () => {
    assert(r.scoreKind === e.scoreKind, `scoreKind ${r.scoreKind} vs ${e.scoreKind}`);
    near(r.score, e.score, 1.0, 'score');
    if (e.mechanicsScore != null) near(r.mechanicsScore, e.mechanicsScore, 1.0, 'mechanicsScore');
    near(r.styleMatch, e.styleMatch, 1.0, 'styleMatch');
  });
  test(`${T} style comparisons within ±1.0`, () => {
    assert(r.closest.name === e.closest, `closest ${r.closest.name} vs ${e.closest}`);
    for (const pc of e.comparisons) {
      const jc = r.comparisons.find(c => c.name === pc.name);
      near(jc.score, pc.score, 1.0, `${pc.name} score`); near(jc.dtw, pc.dtw, 1.0, `${pc.name} dtw`);
      if (pc.learned != null) near(jc.learned, pc.learned, 1.0, `${pc.name} learned`);
    }
  });
  if (encoder && e.embedding) {
    test(`${T} embedding cosine >= 0.999`, () => { const c = cos(r.embedding, e.embedding); embCos.push(c); assert(c >= 0.999, `cos ${c}`); });
  }
  test(`${T} feedback (mechanics first) + coaching plan`, () => {
    assert(r.feedback.length === e.feedback.length, `feedback count ${r.feedback.length} vs ${e.feedback.length}`);
    e.feedback.forEach((pf, i) => {
      const jf = r.feedback[i];
      assert(jf.key === pf.key && jf.kind === pf.kind, `order ${i}: ${jf.kind}:${jf.key} vs ${pf.kind}:${pf.key}`);
      assert(jf.severity === pf.severity, `${pf.key}: ${jf.severity} vs ${pf.severity}`);
      assert(jf.text === pf.text, `${pf.key} text:\n         js: ${jf.text}\n         py: ${pf.text}`);
    });
    assert(JSON.stringify(r.plan.map(p => p.key)) === JSON.stringify(e.plan.map(p => p.key)), `plan ${r.plan.map(p => p.key)} vs ${e.plan.map(p => p.key)}`);
  });
  test(`${T} radar`, () => { for (const k of ['user', 'ref', 'mean']) e.radar[k].forEach((v, i) => near(r.radar[k][i], v, 0.01, `radar.${k}[${i}]`)); });
  test(`${T} quality`, () => {
    assert(r.quality.level === e.quality.level, `level ${r.quality.level} vs ${e.quality.level}`);
    assert(JSON.stringify(r.quality.flags.map(f => f.code)) === JSON.stringify(e.quality.codes), `flags ${r.quality.flags.map(f => f.code)} vs ${e.quality.codes}`);
  });
  test(`${T} result shape (contract)`, () => {
    const N = fx.frames.length;
    assert(r.canonical.length === N && r.canonical[r.events.release] instanceof Float32Array && r.canonical[r.events.release].length === 132, 'canonical');
    assert(r.closest.ghost.length === r.events.ftEnd - r.events.onset + 1 && r.closest.warp.length === r.closest.ghost.length, 'ghost length');
    assert(r.seq60.length === 60 && r.seq60[0] instanceof Float32Array, 'seq60');
    assert(r.radar.axes.length === 9 && typeof r.radar.axes[0] === 'string', 'radar axes');
    for (const k of ['onset', 'set', 'release', 'ftEnd']) assert(Number.isInteger(r.events[k]), `events.${k}`);
    for (const m of r.mechanics) for (const k of ['key', 'label', 'unit', 'band', 'ideal', 'status', 'points', 'maxPoints', 'source']) assert(k in m, `mechanics.${k}`);
    assert(typeof r.styleMatch === 'number' && typeof r.score === 'number', 'score / styleMatch');
  });
  const dmax = Math.max(...e.comparisons.map(pc => Math.abs(r.comparisons.find(c => c.name === pc.name).score - pc.score)));
  parityRows.push(`${(tag + clip.slug).padEnd(22)} score py ${e.score.toFixed(1).padStart(5)} js ${r.score.toFixed(1).padStart(5)} | style py ${e.styleMatch.toFixed(1).padStart(5)} js ${r.styleMatch.toFixed(1).padStart(5)} | max|Δref| ${dmax.toFixed(2)}`);
}
for (const clip of index.clips) parityFixture(FX, clip);
for (const dir of (process.env.JUMPSHOT_EXTRA_FIXTURES || '').split(path.delimiter).filter(Boolean)) {
  const ix = J(path.join(dir, '_index.json'));
  for (const clip of ix.clips) parityFixture(dir, clip, path.basename(dir) + '/');
}

// ------------------------------------------------------------------ demo.json ("Try a sample")
const demo = J(path.join(APP, 'demo.json'));
const userFx = J(path.join(FX, 'user_shot.json'));
const demoRes = analyze(demo.frames, demo.meta, assets, { side: 'auto', window: null });
test('demo.json: score equals Python user_shot score', () => { assert(demoRes.ok, demoRes.reason); near(demoRes.score, userFx.expected.score, 0.05, 'demo score'); });
test('ghost lands on the user (anchor within 0.05 of the user hip at onset)', () => {
  const g = demoRes.closest.ghost[0], u = demoRes.landmarks[demoRes.events.onset];
  near(g[4 * 24], u[4 * 24], 0.05, 'hip x'); near(g[4 * 28 + 1], u[4 * 28 + 1], 0.05, 'ankle y');
});

// ------------------------------------------------------------------ robustness
const base = userFx.frames, meta = userFx.meta, refScore = userFx.expected.score;
const clone = fr => fr && { t: fr.t, lm: Float32Array.from(fr.lm) };

test('Float32Array lm input == plain array input', () => {
  const r = analyze(base.map(clone), meta, assets); near(r.score, refScore, 0.05, 'score');
});
test('aspect ratio: same pixels, different normalisation (width x1.5) -> same result', () => {
  const W2 = meta.width * 1.5;
  const fr = base.map(f => { const c = clone(f); for (let l = 0; l < 33; l++) c.lm[4 * l] = f.lm[4 * l] * meta.width / W2; return c; });
  const r = analyze(fr, { ...meta, width: W2 }, assets); assert(r.ok, r.reason); near(r.score, refScore, 0.2, 'score');
  near(r.summary.elbow_angle_set, userFx.expected.summary.elbow_angle_set, 0.1, 'elbow angle');
});
test('mirrored clip (left-handed, facing left) -> hand left, mirrored, same score', () => {
  const fr = base.map(f => { const c = clone(f); const o = Float32Array.from(c.lm); const pairs = library.constants.mirrorPairs; const map = [...Array(33).keys()]; for (const [a, b] of pairs) { map[a] = b; map[b] = a; } for (let l = 0; l < 33; l++) { const d = map[l]; c.lm[4 * d] = 1 - o[4 * l]; c.lm[4 * d + 1] = o[4 * l + 1]; c.lm[4 * d + 2] = o[4 * l + 2]; c.lm[4 * d + 3] = o[4 * l + 3]; } return c; });
  const r = analyze(fr, meta, assets); assert(r.ok, r.reason);
  assert(r.hand === 'left' && r.mirrored && r.facing === 'left', `hand ${r.hand} facing ${r.facing}`);
  near(r.score, refScore, 0.5, 'score');
});
test('dropped frames (null entries) -> ok, score within ±2', () => {
  const fr = base.map((f, i) => (i % 7 === 3 ? null : clone(f)));
  const r = analyze(fr, meta, assets); assert(r.ok, r.reason); near(r.score, refScore, 2, 'score');
  assert(r.quality.flags.some(f => f.code === 'gaps'), 'gaps flag expected');
});
test('{t, lm:null} entries are treated as missing', () => {
  const fr = base.map((f, i) => (i % 9 === 4 ? { t: f.t, lm: null } : clone(f)));
  const r = analyze(fr, meta, assets); assert(r.ok, r.reason); near(r.score, refScore, 2, 'score');
});
test('jittered timestamps (±6 ms) -> resampled, score within ±3', () => {
  let s = 1; const rnd = () => ((s = (s * 16807) % 2147483647) / 2147483647 - 0.5);
  const fr = base.map(f => ({ t: f.t + rnd() * 0.012, lm: f.lm }));
  const r = analyze(fr, meta, assets); assert(r.ok, r.reason); near(r.score, refScore, 3, 'score');
  assert(r.timing === 'resampled', `timing ${r.timing}`);
});
test('skipped frames (every 3rd frame absent from the list, t kept) -> slotted grid, score within ±2', () => {
  const fr = base.filter((_, i) => i % 3 !== 1);
  const r = analyze(fr, { ...meta, fps: 30 }, assets); assert(r.ok, r.reason); near(r.score, refScore, 2, 'score');
  assert(r.timing === 'slotted', `timing ${r.timing}`);
  assert(r.perFrame.elbow_angle.length === fr.length, 'perFrame aligned to input');
  near(fr[r.events.release].t, base[userFx.expected.events.release].t, 0.07, 'release time');
});
test('60 fps upsampled clip -> events at the same time, score within ±3', () => {
  const fr = [];
  for (let i = 0; i < base.length - 1; i++) { fr.push(base[i]); const a = base[i].lm, b = base[i + 1].lm; fr.push({ t: (base[i].t + base[i + 1].t) / 2, lm: a.map((v, k) => (v + b[k]) / 2) }); }
  fr.push(base[base.length - 1]);
  const r = analyze(fr, { ...meta, fps: 60 }, assets); assert(r.ok, r.reason); near(r.score, refScore, 3, 'score');
  near(fr[r.events.release].t, base[userFx.expected.events.release].t, 0.05, 'release time');
});
test('frame rate changes mid-clip (30 fps then 20 fps) -> resampled, score within ±4', () => {
  const at = t => { const p = t * 30, i = Math.min(base.length - 2, Math.floor(p)), u = p - i; return base[i].lm.map((v, k) => v + (base[i + 1].lm[k] - v) * u); };
  const out = [...base.slice(0, 30).map(f => ({ t: f.t, lm: f.lm }))];
  for (let t = 1.0; t <= (base.length - 1) / 30 + 1e-9; t += 1 / 20) out.push({ t, lm: at(t) });
  const r = analyze(out, meta, assets); assert(r.ok, r.reason); near(r.score, refScore, 4, 'score');
  assert(r.timing === 'resampled', `timing ${r.timing}`);
});
for (const fps of [15, 24, 25, 60]) {
  test(`clip captured at ${fps} fps -> score within ±3`, () => {
    const at = t => { const p = t * 30, i = Math.min(base.length - 2, Math.floor(p)), u = p - i; return base[i].lm.map((v, k) => v + (base[i + 1].lm[k] - v) * u); };
    const out = []; for (let k = 0; k / fps <= (base.length - 1) / 30 + 1e-9; k++) out.push({ t: k / fps, lm: at(k / fps) });
    const r = analyze(out, { ...meta, fps }, assets); assert(r.ok, r.reason); near(r.score, refScore, 3, 'score');
  });
}
test('low-visibility joints (< 0.5) are treated as missing', () => {
  const fr = base.map((f, i) => { const c = clone(f); if (i >= 30 && i < 34) c.lm[4 * 16 + 3] = 0.3; return c; });
  const r = analyze(fr, meta, assets); assert(r.ok, r.reason); near(r.score, refScore, 2, 'score');
});
test('partial clip: stops before the release -> ok:false with a helpful reason', () => {
  const rel = userFx.expected.events.release;
  const r = analyze(base.slice(0, rel - 6), meta, assets);
  assert(!r.ok && /release|dip|shot/i.test(r.reason), `ok=${r.ok} reason=${r.reason}`);
});
test('partial clip: starts after the dip -> ok:false', () => {
  const r = analyze(base.slice(userFx.expected.events.set + 2), meta, assets);
  assert(!r.ok && r.reason, `ok=${r.ok} score=${r.score}`);
});
test('shooting arm out of frame -> ok:false naming the joint', () => {
  const fr = base.map(f => { const c = clone(f); for (const l of [14, 16, 18, 20, 22]) c.lm[4 * l + 3] = 0.1; return c; });
  const r = analyze(fr, meta, assets, { side: 'right' }); assert(!r.ok && /elbow|wrist|hand/.test(r.reason), r.reason);
});
test('bad input never throws', () => {
  for (const [fr, m] of [[[], meta], [null, meta], [[null, null], meta], [base.map(() => null), meta], [[{ t: 0, lm: [1, 2] }], meta], [base, null], ['x', {}]]) {
    const r = analyze(fr, m, assets); assert(r && r.ok === false && typeof r.reason === 'string', JSON.stringify(r).slice(0, 80));
  }
  assert(analyze(base, meta, null).ok === false, 'no assets');
  assert(Array.isArray(detectShots(null, meta)) && detectShots([], meta).length === 0, 'detectShots bad input');
});
test('analyze with explicit window == analyze on the sliced clip (indices into input frames)', () => {
  const pad = Array.from({ length: 20 }, (_, i) => ({ t: i / 30, lm: base[0].lm }));
  const fr = [...pad, ...base.map(f => ({ t: f.t + 20 / 30, lm: f.lm }))];
  const r = analyze(fr, meta, assets, { window: { start: 20, end: fr.length - 1 } });
  assert(r.ok, r.reason); near(r.score, refScore, 0.05, 'score');
  assert(r.events.release === userFx.expected.events.release + 20, `release ${r.events.release}`);
  assert(Number.isNaN(r.perFrame.elbow_angle[5]) && r.canonical[5] === null, 'outside window -> NaN/null');
});

// ------------------------------------------------------------------ v2: events + mechanics robustness
const userExp = userFx.expected;
test('v2: score is the mechanics score; styleMatch is the best reference similarity', () => {
  assert(demoRes.scoreKind === 'mechanics' && demoRes.score === demoRes.mechanicsScore, `${demoRes.scoreKind} ${demoRes.score} ${demoRes.mechanicsScore}`);
  assert(demoRes.styleMatch === demoRes.comparisons[0].score, 'styleMatch');
  const tot = demoRes.mechanics.filter(m => m.status !== 'unmeasured').reduce((q, m) => q + m.maxPoints, 0);
  near(demoRes.score, 100 * demoRes.mechanics.reduce((q, m) => q + m.points, 0) / tot, 0.06, 'score = sum(points)/measured');
  assert(demoRes.feedback[0].kind === 'mechanics' && demoRes.plan[0].kind === 'mechanics', 'mechanics first');
});
test('v2: MediaPipe warm-up garbage in the first frames is masked (same events, score ±1)', () => {
  const fr = base.map((f, i) => { const c = clone(f); if (i < 5) for (const l of [25, 26, 27, 28]) { c.lm[4 * l + 1] = f.lm[4 * 24 + 1] + 0.02 * (l - 24); } return c; });
  const r = analyze(fr, meta, assets); assert(r.ok, r.reason);
  for (const k of ['set', 'release', 'ftEnd']) near(r.events[k], userExp.events[k], 1, k);
  near(r.score, userExp.score, 1, 'score');
});
test('v2: scene cut before the shot -> baseline and set search stay after the cut', () => {
  const other = base.slice(0, 20).map(f => { const c = clone(f); for (let l = 0; l < 33; l++) { c.lm[4 * l] = 0.2 + (f.lm[4 * l] - 0.45) * 0.6; c.lm[4 * l + 1] = 0.3 + (f.lm[4 * l + 1] - 0.5) * 0.6; } return c; });
  const fr = [...other, ...base.map(clone)].map((f, i) => ({ t: i / 30, lm: f.lm }));
  const r = analyze(fr, meta, assets, { autoWindow: false }); assert(r.ok, r.reason);
  assert(r.events.onset >= 20, `onset ${r.events.onset} before the cut`);
  for (const k of ['set', 'release']) near(r.events[k] - 20, userExp.events[k], 1, k);
  near(r.score, userExp.score, 2, 'score');
});
test('v2: shooting arm hidden at the set point -> set_elbow unmeasured, points re-normalised (not 0)', () => {
  const sp = userExp.events.set;
  const fr = base.map((f, i) => { const c = clone(f); if (Math.abs(i - sp) <= 3) for (const l of [14, 16]) c.lm[4 * l + 3] = 0.1; return c; });
  const r = analyze(fr, meta, assets); assert(r.ok, r.reason);
  const m = r.mechanics.find(x => x.key === 'set_elbow');
  assert(m.status === 'unmeasured' && m.value === null && /hidden/.test(m.reason), JSON.stringify(m));
  const tot = r.mechanics.filter(x => x.status !== 'unmeasured').reduce((q, x) => q + x.maxPoints, 0);
  assert(tot === 80, `measured max points ${tot}`);
  near(r.score, 100 * r.mechanics.reduce((q, x) => q + x.points, 0) / 80, 0.06, 'renormalised');
});
test('v2: camera off the side -> off_side_view flag; frontal -> arm angles unmeasured', () => {
  const widen = k => base.map(f => { const c = clone(f); const tor = Math.abs(f.lm[4 * 12 + 1] - f.lm[4 * 24 + 1]) * meta.height / meta.width; c.lm[4 * 11] = f.lm[4 * 12] - k * tor; return c; });
  const r1 = analyze(widen(0.6), meta, assets); assert(r1.ok, r1.reason);
  assert(r1.quality.flags.some(f => f.code === 'off_side_view'), 'flag');
  assert(r1.mechanics.find(x => x.key === 'set_elbow').status !== 'unmeasured', 'still measured at 0.6');
  const r2 = analyze(widen(0.9), meta, assets); assert(r2.ok, r2.reason);
  for (const k of ['set_elbow', 'release_elbow', 'release_shoulder', 'balance_lean']) assert(r2.mechanics.find(x => x.key === k).status === 'unmeasured', k);
});
test('v2: library.json carries mechanics checks + sources; drills cover every check', () => {
  assert(library.mechanics.checks.length === 8 && library.mechanics.sources.length >= 4, 'library');
  assert(library.mechanics.checks.reduce((q, c) => q + c.points, 0) === 100, 'weights sum to 100');
  for (const c of library.mechanics.checks) { assert(drills.drills[c.key] && drills.drills[c.key].high && drills.drills[c.key].low, `drill ${c.key}`); for (const sid of c.source) assert(library.mechanics.sources.some(x => x.id === sid), `source ${sid}`); }
});

// ------------------------------------------------------------------ detectShots
// stance padding cycles through a few real frames (a single repeated noisy frame would be an
// unrealistically static baseline for the onset detector)
const stance = (n, t0, lmOrFrames) => Array.from({ length: n }, (_, i) => ({ t: t0 + i / 30, lm: Array.isArray(lmOrFrames) && lmOrFrames[0] && lmOrFrames[0].lm ? lmOrFrames[i % lmOrFrames.length].lm : lmOrFrames }));
const shift = (fr, dt) => fr.map(f => ({ t: f.t + dt, lm: f.lm }));
test('detectShots: single demo clip -> 1 window containing the release', () => {
  const s = detectShots(base, meta);
  assert(s.length === 1, `got ${s.length}`);
  const rel = userFx.expected.events.release;
  assert(s[0].start <= userFx.expected.events.onset + 3 && s[0].end >= userFx.expected.events.ftEnd - 3, `window ${s[0].start}-${s[0].end}`);
  near(s[0].releaseGuess, rel, 4, 'releaseGuess'); assert(s[0].confidence > 0.5, `confidence ${s[0].confidence}`);
});
test('detectShots: long clip with two shots -> 2 sorted windows; analyze(window) scores each', () => {
  const A = base, d1 = 3, d2 = d1 + 2 + 2.5;
  const head = A.slice(0, 4), tail = A.slice(-4);
  const fr = [...stance(90, 0, head), ...shift(A, d1), ...stance(75, d1 + 2, tail), ...stance(15, 0, head), ...shift(A, d2), ...stance(45, d2 + 2, tail)];
  fr.forEach((f, i) => { f.t = i / 30; });
  const s = detectShots(fr, { ...meta, duration: fr.length / 30 });
  assert(s.length === 2 && s[0].start < s[1].start && s[0].end < s[1].start, JSON.stringify(s));
  for (const w of s) { const r = analyze(fr, meta, assets, { window: w }); assert(r.ok, r.reason); near(r.score, refScore, 3, `score in window ${w.start}-${w.end}`); }
  const auto = analyze(fr, meta, assets, {});            // > 3.5 s, no window -> auto window
  assert(auto.ok && auto.window.source === 'auto', `auto window: ${auto.window && auto.window.source}`); near(auto.score, refScore, 3, 'auto score');
});
test('detectShots: windows never span a scene cut', () => {
  const other = base.map(f => { const c = clone(f); for (let l = 0; l < 33; l++) { c.lm[4 * l] = 0.25 + (f.lm[4 * l] - 0.45) * 0.7; c.lm[4 * l + 1] = 0.35 + (f.lm[4 * l + 1] - 0.5) * 0.7; } return c; });
  const fr = [...other, ...base.map(clone)].map((f, i) => ({ t: i / 30, lm: f.lm }));
  const sh = detectShots(fr, meta);
  assert(sh.length === 2 && sh[0].end < 60 && sh[1].start >= 60, JSON.stringify(sh.map(x => [x.start, x.end])));
});
test('detectShots: standing still -> no shots', () => {
  assert(detectShots(stance(120, 0, base[0].lm), meta).length === 0, 'expected none');
});

// ------------------------------------------------------------------ history
test('summarizeHistory over the demo sessions', () => {
  const seq = ['session_01', 'session_02', 'session_03', 'session_04', 'user_shot'].map(k => resultsBySlug[k]).filter(Boolean);
  const h = summarizeHistory(seq);
  assert(h.scores.length === seq.length && h.trend.direction === 'up', JSON.stringify(h.trend));
  assert(Array.isArray(h.mostImproved) && Array.isArray(h.trendingWorse), 'lists');
  const h2 = summarizeHistory([{ score: 50, deltas: { knee_angle_min: 12 } }, { score: 60, deltas: { knee_angle_min: 3 } }]);
  assert(h2.mostImproved[0].key === 'knee_angle_min' && h2.trend.change === 10, JSON.stringify(h2));
});

// ------------------------------------------------------------------ loadAssets (file: URL)
const asyncTests = [];
asyncTests.push(loadAssets(new URL('../', import.meta.url).href).then(a => {
  test('loadAssets(file: URL) loads library/drills/encoder', () => assert(a.refs.length === library.references.length && !!a.drills && a.learned === !!encoder, 'assets'));
}).catch(e => test('loadAssets(file: URL)', () => { throw e; })));

await Promise.all(asyncTests);
console.log(results.join('\n'));
console.log('\nParity (Python vs JS, same landmarks):\n  ' + parityRows.join('\n  '));
if (embCos.length) console.log(`  embedding cosine JS vs PyTorch: min ${Math.min(...embCos).toFixed(6)} over ${embCos.length} clips (+ ${assets.refs.length} references)`);
console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);
