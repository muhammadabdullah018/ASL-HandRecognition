# ASL Fingerspelling Recognition

Real-time recognition of static ASL fingerspelling letters from a webcam — hold a letter, it locks in, words spell themselves out on screen.

**<img width="800" height="470" alt="bandicam2026-07-1701-22-58-759-ezgif com-video-to-gif-converter" src="https://github.com/user-attachments/assets/5682f48e-1069-4325-9cc3-05f77c43be31" />**

---

## How it works

A webcam frame goes through **MediaPipe Tasks (HandLandmarker)**, which returns 21 hand-joint coordinates — not pixels, just geometry. Those 63 numbers (21 points × x, y, z) get normalized (translation/scale/mirror invariant) and fed into a small **scikit-learn** classifier that predicts one of 24 letters. **OpenCV** handles the camera and renders the whole UI; a hold-to-commit state machine locks in a letter after it's held steady for ~1.5 seconds, so words get spelled out one letter at a time.

## Tech stack

- **MediaPipe Tasks** (`HandLandmarker`) — hand detection + 21-point landmark extraction
- **scikit-learn** — RandomForest / MLP classifier on the 63-float landmark vector
- **OpenCV** — camera I/O, frame processing
- **Pillow** — real typography (Inter / JetBrains Mono) rendered onto the OpenCV overlay
- **pandas / NumPy** — dataset handling and feature math
- **matplotlib** — evaluation charts

## Setup

```bash
git clone https://github.com/muhammadabdullah018/ASL-HandRecognition.git
cd ASL-HandRecognition
pip install -r requirements.txt
python setup_model.py      # downloads hand_landmarker.task (~7MB)
```

Then, in order:

```bash
python collect_data.py     # record your own hand: press a-y, hold each sign
python train_model.py      # trains the classifier, prints accuracy, saves charts to outputs/
python inference.py        # sanity check: one letter at a time
python spell_mode.py       # the full demo: hold letters, spell words
```

Tests (no camera needed):

```bash
python test_landmarks.py
python test_spell_tracker.py
```

## Folder structure

```
ASL-HandRecognition/
├── config.py                # every tunable constant — letters, paths, thresholds, colors
├── fonts.py                  # loads Inter / JetBrains Mono for the UI
├── landmarks.py               # the only file that imports MediaPipe — extraction + normalization
├── setup_model.py             # downloads the MediaPipe .task model file
├── collect_data.py            # webcam -> data/landmarks.csv, tagged with burst/group ids
├── dataset_from_images.py      # optional: folder-per-class image dataset -> same CSV format
├── train_model.py              # trains + evaluates (naive vs honest split) + generates charts
├── inference.py                 # live single-letter webcam test
├── spell_tracker.py             # hold-to-commit + gesture state machines (pure logic, no I/O)
├── spell_mode.py                 # the full demo — HUD, gestures, hold-progress ring
├── ui.py                          # all drawing primitives (OpenCV shapes + Pillow text)
├── test_landmarks.py               # unit tests for the feature math (no camera)
├── test_spell_tracker.py            # unit tests for the state machines (no camera)
├── requirements.txt
├── assets/fonts/                     # bundled Inter + JetBrains Mono (SIL OFL license)
├── outputs/                           # confusion matrix + accuracy charts (generated)
├── data/                               # gitignored — created by collect_data.py
└── models/                             # gitignored — created by setup_model.py / train_model.py
```

---

## Why landmarks instead of a CNN on images

The obvious approach is a CNN trained on photos of hands. This project doesn't do that, on purpose.

| | CNN on raw images | MediaPipe landmarks → small classifier |
|---|---|---|
| Data needed | thousands of images per letter | a few hundred frames per letter |
| Training | GPU, tens of minutes+ | CPU, seconds |
| Lighting / background | learns your room, breaks elsewhere | irrelevant — it never sees pixels |
| Input size | 200×200×3 = 120,000 numbers | 21 points × 3 = **63 numbers** |

MediaPipe already solved the hard vision problem — finding a hand and locating its joints. Putting a 63-input classifier on top of that is the correct architecture, not a shortcut.

## Scope: 24 letters, not 26

**J and Z are excluded.** Both are *motion* signs — J draws a hook, Z draws a zigzag. A single-frame classifier has no time axis, so it structurally cannot represent them. This is a stated scope decision, not a bug: adding them would mean an LSTM/GRU over landmark sequences, which is a different project.

## Results

Trained on **14,400 samples** — 600 per letter (all 24), across 4 separate recording bursts each, with the hand tilted/rotated/moved between bursts so the model doesn't memorize one exact pose.

```
naive split (random shuffle, leaky)    98.6%
group split (whole bursts held out)    89.6%   <- the real number
```

That ~9-point gap **is the single most interesting result in this project.** Frames recorded inside one burst are near-duplicates of each other — a random train/test split scatters those duplicates across both sides, so the model gets tested on frames it's effectively already seen. `train_model.py` fixes this with a custom `leave_one_burst_out()` split: one whole burst held out per letter, guaranteed no overlap. Report the honest number.

**19 of 24 letters score ≥ 90% accuracy** on the honest split; 12 are at a perfect 100% (Y, L, W, V, B, T, E, R, Q, P, O, F). The weakest letters are the classic near-identical-fist confusions: **G** (mistaken for P), **S** (for M), **K** (for U), **I** (for A), **U** (for R) — see `outputs/confusion_matrix.png` and `outputs/per_letter_accuracy.png` for the full breakdown.

## Tuning

Everything lives in `config.py`.

| Symptom | Fix |
|---|---|
| letters fire before you're ready | `HOLD_SECONDS` ↑, or press `]` live in `spell_mode.py` |
| feels sluggish | `HOLD_SECONDS` ↓, or press `[` live |
| flickers between two letters | `SMOOTHING_WINDOW` ↑ to 7-9 |
| shows garbage when your hand is moving | `CONF_THRESHOLD` ↑ to 0.90 |
| one letter is consistently wrong | record more bursts of just that letter, retrain |

## Gestures

- **Hold a letter** ~1.5s → it locks in (majority-vote smoothing + timed hold + lock, so it won't spam-repeat)
- **Make a fist with both hands, press them together** → inserts a space
- **Raise both hands near the top of frame, hold ~1s** → a celebratory reveal of whatever you've spelled so far, with confetti — doesn't clear your text or interrupt recognition

## What this is not

This recognizes **static fingerspelled letters** — it does not "understand sign language." Real ASL is a full language with its own grammar, facial/non-manual markers, and two-handed signs; fingerspelling is a small corner of it, mostly used for proper nouns. Someone who actually signs will notice the difference — being precise about scope is worth more than overclaiming.
