# AI Jumpshot Analyzer

Computer-vision tool that analyzes a basketball jumpshot video, extracts shot
mechanics from body pose, and compares them against a library of reference
shooters — producing a 0–100 mechanics score (eight checks against published
jump-shot kinematics), a style match against the closest reference,
measured per-feature coaching feedback, a drill plan, a side-by-side skeleton
video, and an interactive, mobile-first HTML report with a frame scrubber,
mechanics radar, coaching plan, session history and batch comparison — also
emitted as a page of the shared static site (`--site`).

**Tech stack:** Python 3.13 · OpenCV · MediaPipe (33-landmark pose) · PyTorch · NumPy/SciPy

> **Honesty note.** Everything under `demo/` is *procedurally generated*. No real
> game footage, no real player data and no real player names are used anywhere.
> The reference shooters are named generically ("Reference A — high release", …).
> See [What is synthetic](#what-is-synthetic-and-what-is-not).

---

## Features

* **Pose → mechanics.** 33-landmark pose, 9 body-size-invariant per-frame features, automatic
  shot events (onset, set point, release, follow-through), 60-sample time normalisation.
* **Scoring.** DTW on z-scored sequences + contrastively trained 1-D conv embedding; ranked
  against every reference with per-feature DTW cost attribution.
* **Interactive frame scrubber.** The report embeds every frame's landmarks and features;
  a canvas draws the user skeleton with the **time-warped reference as a ghost** overlay, scaled
  to the same torso length on a shared floor line. Slider, ← / → (shift = ×5), space to play,
  ¼× / ½× / 1× speed, jump-to-event buttons, timecode and live readouts of all 9 features with
  deltas; timeline markers for onset / set point / release / follow-through end.
* **Mechanics radar.** Inline SVG of the 9 features: user vs closest reference vs library mean.
* **Coaching plan.** Every major/minor finding maps to a concrete, generic drill with a
  prescription (sets × reps / hold time) and cue (`drills.py`).
* **Progress tracking.** Each `analyze`/`batch` run appends to `output/history.jsonl`; the report's
  *Sessions* panel shows a score sparkline, most-improved / trending-worse features and the last
  12 sessions. The demo seeds 4 synthetic earlier sessions (labelled `demo`).
* **Batch mode.** `cli.py batch <folder>` scores every clip → `output/batch.csv` + a comparison
  table in the report.
* **Shooting-hand auto-detect + mirror normalisation.** Features use the detected side and facing
  sign; for overlays a left-handed shooter is mirrored into the canonical right-handed view and
  the reference is flipped to face the same way.
* **Quality / confidence flags.** Low body or arm landmark visibility, too few frames, gap-filled
  frames, release not found, degenerate events, clip cut during follow-through — shown as pills
  in the header and a dedicated panel; `confidence` ∈ {ok, minor, major} is stored in results.
* **Exports.** `results.json`, `results.csv` (summary metrics with deltas/severity),
  `features_per_frame.csv`, `batch.csv`, `history.jsonl`, `side_by_side.mp4`.
* **`analyze --webcam`** prints how to record a good clip (no camera is opened).

## Pipeline

```
video ──► pose.py ──► features.py ──► model.py ──► compare.py ──► drills.py ──► overlay.py + report.py
          33 lm/frame   9 features/frame   1-D conv embed   score + feedback   drill plan   MP4 + HTML report
                        + shot events      + DTW baseline    + radar profile                (scrubber, radar, sessions,
                        + quality flags                                                      batch table, CSV/JSONL)
                        + 60-frame resample
```

| module | what it does |
|---|---|
| `pose.py` | Extracts MediaPipe-layout landmarks `(T, 33, 3)` + visibility. Backends: `mediapipe.solutions.pose` if present, else MediaPipe **tasks** `PoseLandmarker` (needs a `pose_landmarker_*.task` file), else — for synthetic clips only — a `<clip>.landmarks.npz` sidecar. Low-visibility landmarks become NaN; `fill_missing` linearly interpolates gaps. |
| `features.py` | Per-frame mechanics features (see below), shot-event detection (onset, set point, peak wrist velocity, release, follow-through, follow-through end), Savitzky-Golay smoothing, torso-length normalisation, and resampling of the onset→follow-through window to **60 frames**. |
| `model.py` | `ShotEncoder`: small 1-D conv net → 64-d unit embedding, trained with a supervised-contrastive (NT-Xent) loss on landmark-level augmentations (time-warp, speed change, scale/translate/roll, jitter + drift, left/right mirror) plus an auxiliary feature-profile reconstruction head. Trains in ~2 s on CPU. Also `dtw_similarity` — a plain DTW baseline on z-scored features. |
| `compare.py` | Headline = the mechanics score (`mechanics.py`); style match against every reference: `0.6 × DTW score + 0.4 × learned cosine score`, ranks references, generates per-feature feedback with measured numbers and severities, and builds the radar profile (min–max normalised over library + user). |
| `drills.py` | Table mapping each feedback key × direction (user above / below reference) to a drill: name, prescription, cue. `coaching_plan()` orders majors first. |
| `overlay.py` | Side-by-side MP4: user video with skeleton (hot-white joints / amber bones / orange shooting arm) next to the reference (video+skeleton if available, otherwise stick figure on dark background), reference time-warped to the user's shot phases, live feature readouts and deltas. H.264 via ffmpeg when available. |
| `report.py` | `output/report.html` (standalone, shared `shell.css` embedded inline) and, with `--site DIR`, `DIR/sample/index.html` + `side_by_side.mp4` linked to `../../assets/shell.css` / `shell.js` (never `DIR/index.html`, which is the web app). KPI tiles, interactive scrubber (canvas sized by ResizeObserver, inline JS + embedded JSON), radar SVG + per-feature table, coaching plan, sessions (sparkline + trends + table), film (video + key frames), evidence (reference ranking, feature curves, pose quality), batch table, "Method & limits" footer. Every section has a plain-English "Read:" line computed from the data. No CDN, no external requests. |
| `demo_data.py` | Parameterised 2-D humanoid shooter animated with OpenCV (filled limbs, head, jersey, ball, court). Generates 3 references with distinct mechanics, one flawed user clip and 4 earlier "practice session" variants, plus landmark sidecars. |
| `cli.py` | argparse CLI (`build-references`, `analyze`, `batch`, `demo`); history/CSV export helpers. |
| `mechanics.py` | The headline mechanics score: 8 checks vs published coaching ranges, sources, re-normalisation of unmeasured checks, mechanics feedback. |
| `export_web.py` | Exports `library.json`, `drills.json`, `encoder.json`, `demo.json` and parity fixtures for the in-browser engine (`site/jumpshot/app/engine.mjs`); see [Web app](#web-app). |

## Run

```bash
cd 01-jumpshot-analyzer
pip install -r requirements.txt

# everything: generate synthetic clips -> build library -> analyze -> outputs
python cli.py demo

# or step by step
python cli.py build-references demo/ref_*.mp4 --landmarks-from-synthetic
python cli.py analyze demo/user_shot.mp4 --references references/ --landmarks-from-synthetic
python cli.py analyze demo/user_shot.mp4 --label "week 3"        # label stored in history.jsonl

# every clip in a folder -> output/batch.csv (+ full report for the top clip; --no-report to skip)
python cli.py batch demo/sessions --landmarks-from-synthetic

# how to record a clip (prints instructions; does not open a camera)
python cli.py analyze --webcam

# also write the static sample report (<dir>/sample/) + the web-app data (<dir>/app/*.json)
python cli.py demo --site ../site/jumpshot
python cli.py analyze demo/user_shot.mp4 --landmarks-from-synthetic --site ../site/jumpshot

# only re-export the web-app JSON (+ parity fixtures)
python cli.py export-web --out ../site/jumpshot/app
```

`--site DIR` (on `demo`, `analyze` and `batch`) writes the same report as `DIR/sample/index.html`
with `<body data-app="jumpshot">`, *relative* links to the shared shell (`../../assets/shell.css`,
`../../assets/shell.js`), a copy of the video and an "Analyze your own shot →" link to `../index.html`,
then exports the web-app data into `DIR/app/` (see [Web app](#web-app)). It never writes
`DIR/index.html` — that page is the in-browser analyzer. The standalone `output/report.html` embeds
its own copy of `shell.css` so it still opens from `file://`.

`cli.py demo` uses the synthetic clips' exact landmark sidecars even when a MediaPipe model is
present (MediaPipe drops ~40 % of frames on the drawn cartoon); `--detect-synthetic` runs the real
detector on them instead.

The demo runs in ~5–10 s on a 2-core CPU (clips → library → 4 back-dated sessions → analysis).

### Output files (`output/`)

| file | what |
|---|---|
| `report.html` | the standalone report (shell.css embedded; open in any browser; works from `file://`) |
| `side_by_side.mp4` | user + time-warped reference skeleton video |
| `results.json` | full results: scores, per-reference feedback, events, summary, radar, quality, coaching plan |
| `results.csv` | one row per summary metric: user / reference / delta / severity |
| `features_per_frame.csv` | 9 features per frame of the user clip with event labels |
| `history.jsonl` | one line per analyze/batch run (timestamp, clip, label, score, per-feature deltas, severities, confidence) |
| `batch.csv` | one row per clip from `batch` (score, closest, major/minor counts, confidence, all deltas) |

### Design notes

Follows `../DESIGN_BRIEF.md` and the shared shell (`site/assets/shell.css`): near-black
surfaces with 1px borders, system UI type, tabular-nums monospace for figures, 375 px first
(single column), 2–4 columns from 900 px. One accent — amber `var(--accent)` — used only for
the hero score, active controls, the selected reference tile and the user skeleton; the
time-warped reference ghost is `var(--muted)` dashed. Colour otherwise means something:
deltas are `--pos` inside the per-metric "ok" tolerance, `--warn` between ok and major,
`--neg` at or beyond the major threshold (thresholds from `compare.FEEDBACK_SPEC`). Page
order: KPI tiles → scrubber → mechanics → coaching plan → Sessions / Film / Evidence / Batch
as collapsible `<details>` (Sessions and Batch open automatically on desktop) → "Method &
limits" footer. Wide tables scroll horizontally with the shell's fade hint; the scrubber
canvas resizes to its container (ResizeObserver, device-pixel-ratio aware). No grain, glow,
text-shadow or decorative chrome.

### Pose backend

MediaPipe ≥ 0.10.3x on Python 3.13 ships **only the tasks API** and no bundled
pose model; `mediapipe.solutions` is gone. To run real MediaPipe inference:

1. Download `pose_landmarker_lite.task` (or `_full` / `_heavy`) from the
   MediaPipe Pose Landmarker model card.
2. Put it in `./models/`, or pass `--pose-model path/to/pose_landmarker_lite.task`,
   or set `POSE_LANDMARKER_MODEL`.

`cli.py demo` detects whether MediaPipe can run; if it can, the synthetic clips are
run through the real detector, otherwise it falls back to the synthetic landmark
sidecars (`--landmarks-from-synthetic`) and says so on stdout and in the report
footer. `analyze` on a real clip without a model raises a clear error.

## Web app

`site/jumpshot/index.html` lets a user record or upload their own shot on a phone and get the same
analysis in the browser. Everything runs client-side; **the video never leaves the device**.

```
video (file / camera) ─► app/pose.mjs ─────────► app/engine.mjs ──────────────► app/ui.js
                         MediaPipe tasks-vision    detectShots → analyze:         results view
                         (vendor/, wasm, .task)    port of features/model/        (score, ghost, radar,
                         {t, lm[132]} per frame     compare/drills, no DOM         plan, history)
                                                       ▲
              library.json · drills.json · encoder.json · demo.json   ◄── export_web.py (this folder)
```

* **`export_web.py`** (`cli.py export-web --out DIR`, also run by `--site`) writes
  `library.json` (3 references: name, events, summary, 60×9 `seq60`, canonical landmarks over
  onset→follow-through resampled to 60 frames for the ghost, feature/metric lists with tolerances and
  wording, z-score stats, DTW band/score mapping, 0.6/0.4 weights, event-detector / smoothing /
  visibility constants, radar axes — 107 KB), `drills.json` (5 KB), `encoder.json` (ShotEncoder
  eval-mode weights as nested arrays, 6 significant figures, decoder omitted — 866 KB), `demo.json`
  (the synthetic user clip as web-app frames, 61 KB, for "Try a sample") and
  `tests/fixtures/*.json` (Python's results on 7 clips, input frames included).
* **`engine.mjs`** is a line-by-line port: visibility mask (< 0.5 = missing, same `pose.MIN_VIS`
  in Python), gap fill, scipy-exact Savitzky-Golay (`mode="interp"`), torso normalisation in pixel
  space (normalised coords × `meta.width/height`, so non-square video is handled exactly as
  `features.to_pixels`), the 9 features, hand/facing detection, events with the constants verbatim,
  quality flags, resampling, DTW (Sakoe-Chiba band `max(n,m)//4`), the conv1d/BatchNorm/GELU/pool
  encoder forward pass, feedback strings, coaching plan, radar. Phone-video robustness: frames on a
  regular 1/fps lattice with gaps are slotted (= Python's dropped frames); irregular timestamps are
  resampled onto a uniform grid at the median rate, so every time quantity uses `t`. Clips that miss
  the release or the dip return `{ok:false, reason}` ("Couldn't find a release — make sure the whole
  shot, from dip to follow-through, is in the clip."). `detectShots()` finds shot windows in long
  clips (wrist above the head with elbow extension after a low ball position, padded 0.8 s / 0.6 s);
  `analyze()` auto-picks the best window for clips > 3.5 s when none is given.
* **Scoring**: `result.score` is the mechanics score, `result.styleMatch` the similarity to the closest
  reference, `result.mechanics` the eight checks (see [How scoring works](#how-scoring-works)).
* **Parity** (`node site/jumpshot/app/tests/parity.test.mjs`, no npm deps, 134 checks; set
  `JUMPSHOT_EXTRA_FIXTURES=dir1:dir2` to add fixtures made by `export_web.export_clip_fixtures`, e.g. from a
  real clip — 12 real-derived clips (4 shots × 30/15 fps, crops, time-reversed) agree exactly locally): on the demo
  user clip, a gap-injected copy, the 4 session clips and a reference clip, JS events are identical,
  summaries match within 2e-5, every reference score / DTW / learned score is identical to Python's
  1-decimal output (|Δ| = 0.0), mechanics checks/score/feedback/plan are identical, embeddings have
  cosine 1.000000 vs PyTorch. Robustness checks:
  Float32Array input, aspect ratio, mirrored left-hander, dropped / null frames, timestamp jitter,
  skipped frames, mid-clip fps change, 15/24/25/60 fps captures (score within ±3), low visibility,
  partial clips, missing shooting arm, bad input (never throws), explicit / auto windows, multi-shot
  detection. On a real 13 s, 960×540 phone-style clip with four shots (MediaPipe lite, Python
  `pose.py`), `detectShots` found the four shots and Python/JS agreed exactly on every window.
* **Re-export** after changing the library, features or encoder:
  `python cli.py demo --site ../site/jumpshot` (or `python cli.py export-web --out ../site/jumpshot/app`),
  then run the parity test.

## Adding real reference clips

The tool does not download footage. Supply your **own, legally obtained** clips
(your own recordings, or footage you have rights to). Sources the original idea
pointed to for data/inspiration were the NBA Stats shooting pages and the public
NBA datasets on Kaggle — those give shot *charts/outcomes*, not video; the video
has to come from you.

```bash
python cli.py build-references my_clips/shooter1.mp4 my_clips/shooter2.mp4 \
    --names "Coach — catch and shoot" "Teammate — pull-up" \
    --pose-model models/pose_landmarker_full.task
python cli.py analyze my_shot.mp4 --references references/ --pose-model models/pose_landmarker_full.task
```

Tips for good results: one shooter in frame, side-ish or 3/4 view, the whole body
visible from stance to follow-through, ≥ 30 fps, one shot per clip (trim to
~2 s around the shot). Shooting hand is auto-detected (`--side right|left` to force).

## Feature definitions

All features are body-size invariant: angles in degrees, lengths in
**torso-lengths** (shoulder-midpoint → hip-midpoint distance, median over the clip;
≈ 50 cm for an adult). Computed per frame on Savitzky-Golay-smoothed pixel
coordinates of the shooting side.

| feature | definition |
|---|---|
| `elbow_angle` | shoulder–elbow–wrist angle (180° = fully extended) |
| `knee_angle` | hip–knee–ankle angle (180° = straight leg) |
| `hip_angle` | shoulder–hip–knee angle (alignment; 180° = stacked) |
| `trunk_lean` | angle of hip→shoulder vector from vertical, + = forward |
| `wrist_height` | (nose_y − wrist_y) / torso; + = wrist above head |
| `wrist_forward` | horizontal wrist offset from shoulder along facing direction / torso |
| `elbow_tuck` | horizontal elbow offset from shoulder along facing direction / torso |
| `wrist_angle` | elbow–wrist–index-finger angle; drops after release = snap |
| `hip_rise` | (hip_y_start − hip_y) / torso — jump-height proxy |

**Events (detector v2, anchored on the release).** Built for real phone footage (walk-ins,
dribbles, MediaPipe warm-up noise, edited clips) and checked frame by frame against contact sheets
of a real four-shot clip:

1. *Clean-up* — frames whose leg-bone / torso length ratio is outside 0.6–1.6× its clip median are
   treated as missing (MediaPipe locking onto the wrong limbs, typically in the first frames); a hip
   jump of > 0.8 torso in one frame is a scene cut, and the stance baseline / set search stay after it.
2. *Extension* — the peak elbow angular velocity over frames where the ball is **loaded** (shooting
   wrist above the shoulder or within 0.3 torso of it) **and the wrist is rising**, after a 0.2 s
   warm-up, and only if the arm then reaches ≥ 140° with the wrist above the head within 0.4 s. This
   excludes the dribble, the walk-in, the waist-to-set lift and the arm coming back down.
3. *Release* — first frame after that peak with the elbow ≥ 150° (or within 4° of the arm's maximum in
   the next 0.6 s). v1 used 160°; on real footage the ball visibly leaves the fingertips at ~150° (2-D),
   and the detected release is within 1 frame of the ball leaving on all four real shots. Timing metrics
   use the sub-frame threshold crossing. `release_not_found` if the arm never reaches 140°.
4. *Set point* — the last elbow-flexion minimum in the loaded zone within 0.8 s before release (flexed
   ≥ 8° within 0.3 s before it, extends ≥ 20° after it); if the shot is one continuous extension, the
   deepest knee bend in that window (`events.set_method` says which).
5. *Onset* — first departure from the stance pose (knee −4° or wrist ±0.12 torso), searched from 1 s
   before the set point. A set point inside the warm-up means the dip is not in the clip → the web
   app returns `ok:false` ("start recording about a second before you start your shot").

*Follow-through hold* = seconds after release with wrist ≥ 65 % of release height and elbow ≥ 140°.
The onset→follow-through-end window is resampled to 60 samples. On the synthetic clips v2 finds the
same set points and onsets as v1 and the release 1 frame earlier (1–3 for the session clips).

**Per-shot summary** (what the feedback compares): elbow angle & elbow tuck at set
point, deepest knee bend, hip alignment at set, trunk lean at release, release
height, release point forward offset, load-to-release time (set point → release),
follow-through hold and wrist angle, jump height, elbow extension at release.

## How scoring works

The **headline score is a mechanics score** (0–100): eight checks of measurable shot mechanics, each
against a coaching range taken from published jump-shot kinematics or, where marked, a coaching
convention. Inside the range = full points; outside, points fall off linearly to 0 at range ±
tolerance. A check that cannot be measured reliably on the clip (joint hidden around that frame,
camera too frontal, no jump, clip cut) is **unmeasured** and its points are removed from the total
(re-normalised); it is never silently scored 0. If fewer than 40 points are measurable, the score falls
back to the style match.

| check | measured as | range | tol | pts | source |
|---|---|---|---|---|---|
| Elbow at set point | elbow angle at the set point | 70–100° | 20° | 20 | [1] (72–88°) |
| Knee dip | deepest knee angle (interior) within 0.8 s before release | 100–130° | 20° | 15 | [1] (102–115°), [2] |
| Elbow at release | elbow angle at release | 125–180° | 20° | 20 | [3] (126.7 ± 9.4°), [1] (164°) |
| Shoulder at release | angle between trunk axis and upper arm | 120–150° | 20° | 15 | [3] (129.8 ± 9.2°) |
| Release timing | release − peak of hip height (s, − = before) | −0.12…+0.03 s | 0.12 s | 15 | [1], [4] |
| Follow-through | hold with wrist high and elbow ≥ 140° | ≥ 0.3 s | 0.2 s | 10 | coaching convention |
| Balance: lean | trunk-lean change set → release | 0–10° | 10° | 2 | coaching convention, [4] |
| Balance: drift | horizontal hip travel set → release + 0.4 s | 0–0.4 torso | 0.4 | 3 | coaching convention, [4] |

1. *Kinematic analysis of jump shot with backward take-off* (CROSBI 683397, 2019) — model jump shot:
   elbow 72–88° at the set position, knee flexion 102–115°, elbow ~164° at release, release at or just
   before the peak of the jump.
2. *Frontiers in Sports and Active Living* (2026), doi:10.3389/fspor.2026.1732293 — proficient
   3-point shooters show greater hip, knee and ankle flexion.
3. Taborda et al. — elbow 126.7 ± 9.4° and shoulder flexion 129.8 ± 9.2° at release.
4. França et al. (2021), *Acta of Bioengineering and Biomechanics* 24(1) — release at/just before the
   jump peak; jump close to vertical.

These are conservative **coaching ranges, not laws**: the studies measured small groups of skilled
shooters in 3-D; here every angle is a 2-D projection from one phone camera. A camera that is not
side-on (shoulder width > 45 % of torso length) raises the `off_side_view` quality flag; above 75 % the
arm and trunk angles are reported as unmeasured. The release-elbow range is open to 180° so a fully
extended finish is never penalised. Feedback and the drill plan lead with the mechanics findings
(biggest point loss first); style-match deltas follow.

**Style match** (the v1 headline, still reported): DTW on z-scored 60-sample sequences (library
mean/std with physical floors) mapped by `100·exp(−(d/2.2)^1.5)`, blended 0.6/0.4 with a sharpened
cosine of the learned embeddings. It measures resemblance to the closest of the three *synthetic*
reference shooters — useful for the ghost overlay, not a quality grade: real shooters land at ≈ 40–50.

**Real clip, v1 → v2** (13 s, 960×540 phone-style clip of one shooter taking four shots, MediaPipe lite;
used only as a local test fixture, not committed):

| shot | v1 events (onset/set/release/end) | v1 score (similarity) | v2 events | v2 mechanics | style match |
|---|---|---|---|---|---|
| 1 | failed: release not found | — | 2 / 14 / 29 / 42 | 95.1 (set elbow + 2 balance checks unmeasured: noisy first frames) | 38.7 |
| 2 | 82 / 91 (walk-in) / 134 / 160 | 44.3 | 105 / 116 / 134 / 160 | 89.3 (set elbow 61°) | 47.1 |
| 3 | 190 / 203 (walk-in) / 244 / 273 | 40.7 | 214 / 230 / 243 / 273 | 99.5 | 47.3 |
| 4 | 293 / 336 / 344 / 353 | 36.5 | 300 / 328 / 343 / 353 | 92.2 (set elbow 63°) | 43.7 |

The shooter's mechanics are inside most ranges, so the scores are high. That is the honest result of
these bands, not tuning; the main deductions are a pocket elbow tighter than 70° and the trunk lean
change in the dip.

## What is synthetic and what is not

* **Synthetic:** all four demo clips (procedural stick-figure animation), their
  landmark sidecars (exact synthetic joint positions + ~1–2 px Gaussian jitter),
  and therefore every number in the demo report.
* **Real:** the pipeline code. `pose.py` implements the MediaPipe tasks/solutions
  paths; features, events, DTW, encoder training, overlay and report all operate
  on the standard 33-landmark format and work unchanged on real detections.
* **MediaPipe:** `models/pose_landmarker_lite.task` runs here. On the drawn demo figure it misses
  ~40 % of frames, so `cli.py demo` uses the exact sidecars (`--detect-synthetic` to use MediaPipe).
  On a real basketball clip (used only as a local test fixture, not committed) the detector, the
  Python pipeline and the web engine all run end to end and agree exactly.

## Limitations

* The coaching plan maps deltas to *generic* form-shooting drills; it is a demonstration of the
  mapping, not professional coaching or medical advice.
* The sessions panel in the demo is seeded with synthetic earlier sessions (labelled `demo`);
  `cli.py demo` resets `history.jsonl` so the demo is reproducible. Real runs append.
* Landmarks with visibility < 0.5 are treated as missing everywhere (`pose.MIN_VIS`); `mediapipe` visibility scores are used for the quality flags; the synthetic sidecars carry
  hand-set visibilities (far-side joints 0.4–0.7), so the demo shows 100 % body visibility.
* 2-D, single-camera: true elbow *flare* (lateral) is unobservable from a side
  view; `elbow_tuck` is the sagittal proxy. A front/3-4 view changes feature values,
  so compare clips shot from similar angles.
* With three library shots the contrastive encoder behaves like a 3-way classifier
  (embeddings cluster tightly; learned scores are near 100 or near 20). It becomes
  meaningful with a larger library; the DTW term keeps scores graded meanwhile.
* Time normalisation removes absolute tempo from the sequence comparison; tempo is
  compared explicitly through `release_time_s` / hold duration in the feedback.
* Event detection was developed on one real clip (four shots, one shooter, side-on); other camera
  angles, shooters and shot types (step-backs, fades, set shots without a jump) are untested. The
  release is kinematic (elbow ≥ 150°), not ball tracking, and lands within ~1 frame of the ball leaving.
  At 15 fps the set-point elbow is sampled coarsely (real-shot scores move by up to ~10 points vs 30 fps).
* The mechanics ranges come from small 3-D studies of skilled shooters and are applied to 2-D
  projections; the style match compares against synthetic references (real shooters ≈ 40–50).
* Release detection assumes one shot per clip (per `detectShots` window in the web app) and a visible shooting arm. Heavy
  occlusion (defender, ball covering the wrist) will degrade events.
* Lengths are reported in torso units; cm conversions in messages assume a 50 cm torso.
* No ball tracking — "release" is a kinematic definition, not the ball leaving the hand.

## Resume bullets

* Built an end-to-end computer-vision jumpshot analyzer (Python, OpenCV, MediaPipe,
  PyTorch): 33-landmark pose extraction → body-size-invariant mechanics features →
  automatic shot-event detection (set point, release, follow-through) → similarity
  scoring and measured coaching feedback, with a skeleton side-by-side video
  and a self-contained HTML report.
* Designed a hybrid similarity metric combining dynamic time warping on z-scored
  feature sequences with a contrastively trained 1-D convolutional embedding
  (NT-Xent on time-warp / jitter / mirror augmentations, CPU training in ~2 s).
* Engineered a parameterised procedural motion generator to produce labelled
  synthetic shooters with known mechanics, enabling offline validation of event
  detection (release frame within 1–2 frames of ground truth) without licensed footage.
* Diagnosed and fixed a 60× CPU slowdown from BLAS thread oversubscription in small
  PyTorch models on 2-core machines.
* Shipped an interactive, dependency-free HTML "film room": canvas frame scrubber with a
  time-warped reference ghost, radar chart, session history and batch comparison, plus a
  findings-to-drills coaching plan and clip-quality confidence flags.
