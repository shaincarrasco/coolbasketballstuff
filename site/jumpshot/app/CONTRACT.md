# Jumpshot web app — engine ⇄ UI contract

Everything runs client-side. Video never leaves the device.

```
site/jumpshot/
  index.html            UI app (owner: UI agent)
  app/
    ui.js, ui.css       UI app code (owner: UI agent)
    pose.mjs            MediaPipe wrapper: video -> frames (owner: UI agent)
    engine.mjs          pure analysis engine, no DOM (owner: engine agent)
    library.json        reference library exported from Python (owner: engine agent)
    demo.json           demo user clip landmarks, for "Try a sample" (owner: engine agent)
    drills.json         finding -> drill table (owner: engine agent)
    encoder.json        optional: learned encoder weights (owner: engine agent)
    tests/              node parity tests vs Python (owner: engine agent)
  sample/index.html     the static Python-generated sample report (owner: engine agent via report.py --site)
  vendor/               MediaPipe tasks-vision 1.2.2 bundle + wasm + pose_landmarker_{lite,full}.task (commander; don't edit)
```

## Frame format (pose.mjs -> engine)
```js
// one entry per processed frame, in time order
{ t: Number,              // seconds from clip start
  lm: Float32Array(33*4)  // MediaPipe normalized image landmarks: x, y (0..1, y down), z, visibility
  // or null when no person was detected in that frame
}
// plus clip meta: { fps: Number, width: Number, height: Number, duration: Number }
```

## Engine API (engine.mjs, ES module, no DOM, works in browser and Node ≥18)
```js
export async function loadAssets(baseUrl)             // fetches library.json, drills.json, encoder.json(optional) -> assets
export function detectShots(frames, meta)             // -> [{start, end, releaseGuess, confidence}] frame-index windows; [] if none
export function analyze(frames, meta, assets, opts)   // opts: {side:'auto'|'left'|'right', window:{start,end}|null}
  // -> Result (below). Never throws on bad input: returns {ok:false, reason, quality}
export function summarizeHistory(results)             // -> {scores:[...], trend, mostImproved, trendingWorse}
export const FEATURES                                  // ordered [{key,label,unit,tolerance,higherIsBetter|null}]
```

### Result
```js
{
  ok: true,
  quality: { level:'ok'|'minor'|'major', flags:[{code,text}], visibility:0..1, frames:Number },
  hand: 'right'|'left', facing:'left'|'right', mirrored:Boolean,
  events: { onset, set, release, ftEnd },                 // frame indices into the input frames
  perFrame: { [featureKey]: Float32Array },               // aligned to input frames (NaN where missing)
  summary: { [metricKey]: Number },                       // elbow_at_set, knee_min, release_height, load_to_release_s, hold_s, ...
  seq60: Array<Float32Array>,                             // 60 x F time-normalized onset->ftEnd
  canonical: Array<Float32Array|null>,                    // per-frame landmarks after mirror normalization (for drawing)
  comparisons: [{ name, score, dtw, learned|null }],      // sorted desc
  closest: { name, score, ghost: Array<Float32Array>, warp: Int32Array },  // ghost = ref landmarks time-warped to user frames onset..ftEnd
  feedback: [{ key, label, user, ref, delta, unit, severity:'ok'|'minor'|'major', text }],
  plan: [{ finding, drill, prescription, cue, severity }],
  radar: { axes:[...], user:[...], ref:[...], mean:[...] },   // 0..1
  score: Number                                            // 0..100 headline score = mechanicsScore (falls back to styleMatch)

  // ---- engine v2 additions (additive; everything above is unchanged in shape)
  mechanics: [{ key, label, value|null, unit, band:[lo,hi], ideal, tolerance,
                status:'good'|'close'|'off'|'unmeasured', points, maxPoints,
                source:[sourceId], text, reason?, frame }],       // 8 checks, see library.json mechanics.sources
  mechanicsScore: Number|null,      // 0..100: sum(points) / sum(maxPoints of measured checks); null if < 40 pts measurable
  styleMatch: Number,               // 0..100 old headline: 0.6*DTW + 0.4*learned similarity to the closest reference
  scoreKind: 'mechanics'|'style',   // what `score` is
  // feedback: mechanics findings first (kind:'mechanics', biggest point loss first, ref = nearest range edge,
  //           pointsLost, band), then style-match deltas vs the closest reference (kind:'style')
  // plan:     mechanics drills first, then style drills; each drill listed once; item.kind
  // events:   + peakExtension, setMethod ('elbow_min'|'knee_min'|'legacy')
  // quality.flags may include {code:'off_side_view'} (camera not side-on; angles are 2-D projections)
}
```

Mechanics check keys: set_elbow (20 pts), dip_knee (15), release_elbow (20), release_shoulder (15),
release_timing (15, s relative to the jump peak, negative = before), follow_through (10),
balance_lean (2), balance_drift (3). Ranges, weights and citations live in `library.json → mechanics`
(`checks`, `sources`); `drills.json` has a drill for every check key (`high` = above range, `low` = below).

Scores must match the Python pipeline on the same landmarks within ±1.0 point (parity tests).
