# Project context: ASL Fingerspelling Recognizer

Read this before touching the code. It's the map of what exists, what's
missing, and what NOT to rebuild.

## What this is

Real-time recognition of **static ASL fingerspelling letters** (24 of 26 —
J and Z need motion, excluded on purpose, see below) from a webcam. Not a
general sign-language translator. Not a CNN-on-images project.

Stack: **MediaPipe Tasks (HandLandmarker)** finds the hand and returns 21
joint coordinates → a small **scikit-learn** classifier (RandomForest or
MLP) maps those 63 numbers to a letter → **OpenCV** does all camera I/O and
the on-screen overlay. Full rationale is in [README.md](README.md) — it's
unusually thorough, read it before asking "why not a CNN."

## Current state (as of last check)

- Code: **complete**, all files implemented, nothing is a stub.
- `data/`, `models/`, `outputs/` dirs: **do not exist yet** — no training
  data has been collected, no model has been trained.
- `models/hand_landmarker.task`: **not downloaded**. Run `setup_model.py` first.
- Python deps: installed and import cleanly (`mediapipe`, `cv2`, `sklearn`,
  `pandas`, `numpy`, `joblib`, `matplotlib`). Verified working with
  `mediapipe.tasks.python.vision` (the Tasks API, not the legacy
  `mp.solutions` — that API is gone from current mediapipe, don't write
  code against it).

## Why landmarks, not pixels

63 floats (21 points × x,y,z) beat a CNN on raw hand photos: ~150 frames
needed per letter instead of thousands, trains on CPU in seconds, and
lighting/background are irrelevant since the classifier never sees pixels.
This is the whole architectural bet of the project — don't propose "just
train a CNN on images" as an improvement, it's a strictly worse approach
here.

## Pipeline (in run order)

```
python setup_model.py      # downloads hand_landmarker.task (~7MB) - DO THIS FIRST
python collect_data.py     # webcam, press a-y to record each letter, ~90 min
python train_model.py      # trains RF + MLP, prints naive vs honest accuracy
python inference.py        # sanity check: one letter at a time
python spell_mode.py       # the actual demo: hold letters, spells words
```

Tests (no camera needed):
```
python test_landmarks.py       # feature-invariance math
python test_spell_tracker.py   # hold-to-commit state machine
```

## File map

| File | Role |
|---|---|
| `config.py` | every tunable constant — letters, paths, thresholds, colors |
| `landmarks.py` | **only file that imports mediapipe**. Extracts + normalizes 21 landmarks → 63 floats |
| `setup_model.py` | downloads the `.task` model file (Tasks API doesn't bundle weights) |
| `collect_data.py` | webcam → `data/landmarks.csv`, tags each recording session with a burst/group id |
| `dataset_from_images.py` | optional: converts a folder-per-class image dataset (e.g. Kaggle ASL Alphabet) into the same CSV format |
| `train_model.py` | trains RandomForest + MLP, reports **naive split** (leaky, optimistic) vs **group split** (honest, holds out whole bursts) |
| `inference.py` | live single-letter webcam test |
| `spell_tracker.py` | pure-logic hold-to-commit FSM (majority vote + timed hold + lock). No I/O — unit-testable without a camera |
| `spell_mode.py` | the full demo: HUD, hold-progress ring, alphabet rail, output strip |
| `ui.py` | all OpenCV drawing helpers (scrims, glow skeleton, rings, pills). Zero ML in this file |
| `test_landmarks.py`, `test_spell_tracker.py` | fast, no-camera unit tests |

## The one bug every tutorial has (and this project fixes)

Frames within one recording burst are near-duplicates. Randomly splitting
train/test scatters duplicates across both sides → 99% accuracy that means
nothing. `train_model.py` tags every burst with a group id at collection
time and does `leave_one_burst_out()` — held-out test bursts, one per
class, never seen in train. **Always report the group-split number, not
the naive one.**

## Dataset decision (already made, don't relitigate without reason)

**Record your own via `collect_data.py`.** ~90 minutes, 24 letters × 150
frames, 2-3 bursts per letter, move the hand while recording (tilt/rotate/
drift — static frames = model memorizes one pose). This beats public data
because the model transfers to *your* webcam, and 63-float landmark models
trained on strangers' hands generalize worse than expected.

If recording isn't an option, `dataset_from_images.py` converts a
Kaggle-style folder-per-class image set into the CSV format — see the
"dataset question" section in README.md for named datasets, tradeoffs, and
expected detection-rate loss (crops are often too tight for MediaPipe's
palm detector; Sign Language MNIST at 28×28 is unusable for this
architecture, don't recommend it).

## Things not to "fix"

- J and Z missing from `LETTERS` in `config.py` — intentional, they need
  motion, a single-frame classifier cannot represent them.
- `mp.solutions.hands` anywhere — don't add it, it's the deprecated API and
  isn't in current mediapipe releases.
- Default `HOLD_SECONDS = 1.5` — deliberate tradeoff, tunable live with
  `[`/`]` keys in `spell_mode.py`, documented in README.md.
