# AI Jumpshot Analyzer

Computer-vision tool that analyzes a basketball jumpshot video, extracts shot
mechanics from body pose, and compares them against a library of reference
shooters — producing a 0–100 form-similarity score, the closest reference,
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
| `compare.py` | Scores the user shot against every reference: `0.6 × DTW score + 0.4 × learned cosine score`, ranks references, generates per-feature feedback with measured numbers and severities, and builds the radar profile (min–max normalised over library + user). |
| `drills.py` | Table mapping each feedback key × direction (user above / below reference) to a drill: name, prescription, cue. `coaching_plan()` orders majors first. |
| `overlay.py` | Side-by-side MP4: user video with skeleton (hot-white joints / amber bones / orange shooting arm) next to the reference (video+skeleton if available, otherwise stick figure on dark background), reference time-warped to the user's shot phases, live feature readouts and deltas. H.264 via ffmpeg when available. |
| `report.py` | `output/report.html` (standalone, shared `shell.css` embedded inline) and, with `--site DIR`, `DIR/index.html` + `side_by_side.mp4` linked to `../assets/shell.css` / `shell.js`. KPI tiles, interactive scrubber (canvas sized by ResizeObserver, inline JS + embedded JSON), radar SVG + per-feature table, coaching plan, sessions (sparkline + trends + table), film (video + key frames), evidence (reference ranking, feature curves, pose quality), batch table, "Method & limits" footer. Every section has a plain-English "Read:" line computed from the data. No CDN, no external requests. |
| `demo_data.py` | Parameterised 2-D humanoid shooter animated with OpenCV (filled limbs, head, jersey, ball, court). Generates 3 references with distinct mechanics, one flawed user clip and 4 earlier "practice session" variants, plus landmark sidecars. |
| `cli.py` | argparse CLI (`build-references`, `analyze`, `batch`, `demo`); history/CSV export helpers. |

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

# also write the static-site page: <dir>/index.html + side_by_side.mp4, linking ../assets/shell.{css,js}
python cli.py demo --site ../site/jumpshot
python cli.py analyze demo/user_shot.mp4 --landmarks-from-synthetic --site ../site/jumpshot
```

`--site DIR` (on `demo`, `analyze` and `batch`) writes the same report as `DIR/index.html` with
`<body data-app="jumpshot">` and *relative* links to the shared shell (`../assets/shell.css`,
`../assets/shell.js` — the shell draws the top bar / app switcher), plus a copy of the video. The
standalone `output/report.html` embeds its own copy of `shell.css` so it still opens from `file://`.

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

**Events.** The shot is anchored in the order *dip → extension → wrist top*:
the *wrist top* is the first frame within 5 % of the wrist's highest point; the
*set point* is the deepest knee bend before it; the *extension peak* is the peak
elbow angular velocity between set point and wrist top (peak wrist or elbow
velocity over the whole clip also fires on the lift from waist to set point,
especially for an under-loaded shooter). *Release* = first frame after the
extension peak where the elbow reaches ≥ 160°, or within 4° of the arm's own
maximum in the following 0.6 s when it never gets there; if the arm never
reaches 150° the `release_not_found` quality flag is raised. *Onset* = first
departure from the stance pose (knee −4° or wrist ±0.12 torso). On the eight
synthetic clips the detected release is within 1–3 frames of ground truth.
*Follow-through hold* = seconds after release with wrist ≥ 65 % of release height and
elbow ≥ 140°. The onset→follow-through-end window is resampled to 60 samples.

**Per-shot summary** (what the feedback compares): elbow angle & elbow tuck at set
point, deepest knee bend, hip alignment at set, trunk lean at release, release
height, release point forward offset, load-to-release time (set point → release),
follow-through hold and wrist angle, jump height, elbow extension at release.

**Score.** DTW on z-scored sequences (library mean/std with physical floors so a
near-constant feature cannot dominate) is mapped by `100·exp(−(d/2.2)^1.5)`;
calibrated on the synthetic library so augmented copies of a shot score ≈ 85–95
and clearly different mechanics ≈ 40–55. The learned score is a sharpened cosine
of embeddings. Final = 0.6·DTW + 0.4·learned; both are shown separately.

## What is synthetic and what is not

* **Synthetic:** all four demo clips (procedural stick-figure animation), their
  landmark sidecars (exact synthetic joint positions + ~1–2 px Gaussian jitter),
  and therefore every number in the demo report.
* **Real:** the pipeline code. `pose.py` implements the MediaPipe tasks/solutions
  paths; features, events, DTW, encoder training, overlay and report all operate
  on the standard 33-landmark format and work unchanged on real detections.
* **Not verified here:** MediaPipe's detection quality on the drawn figure. No
  pose model file was available offline in the build environment, so the demo
  used the sidecar path. If you drop a `.task` model in `models/`, `cli.py demo`
  will run the real detector on the synthetic clips automatically.

## Limitations

* The coaching plan maps deltas to *generic* form-shooting drills; it is a demonstration of the
  mapping, not professional coaching or medical advice.
* The sessions panel in the demo is seeded with synthetic earlier sessions (labelled `demo`);
  `cli.py demo` resets `history.jsonl` so the demo is reproducible. Real runs append.
* `mediapipe` visibility scores are used for the quality flags; the synthetic sidecars carry
  hand-set visibilities (far-side joints 0.4–0.7), so the demo shows 100 % body visibility.
* 2-D, single-camera: true elbow *flare* (lateral) is unobservable from a side
  view; `elbow_tuck` is the sagittal proxy. A front/3-4 view changes feature values,
  so compare clips shot from similar angles.
* With three library shots the contrastive encoder behaves like a 3-way classifier
  (embeddings cluster tightly; learned scores are near 100 or near 20). It becomes
  meaningful with a larger library; the DTW term keeps scores graded meanwhile.
* Time normalisation removes absolute tempo from the sequence comparison; tempo is
  compared explicitly through `release_time_s` / hold duration in the feedback.
* Release detection assumes one shot per clip and a visible shooting arm. Heavy
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
