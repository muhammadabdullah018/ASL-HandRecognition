"""
OPTIONAL. Convert a folder-per-class image dataset (e.g. Kaggle ASL Alphabet)
into landmarks.csv, so you can bootstrap without recording everything yourself.

    python dataset_from_images.py /path/to/asl_alphabet_train --per-class 300

Expects:
    root/A/*.jpg   root/B/*.jpg   ...

READ THIS BEFORE YOU BOTTLE OUT OF RECORDING YOUR OWN DATA
----------------------------------------------------------
Public ASL image sets are a worse starting point than they look:

  - most are tightly cropped to the hand, and MediaPipe's palm detector wants
    some context around it. Expect to silently lose 20-50% of images to
    "no hand found". That's normal, not a bug in this script.
  - they're other people's hands, other cameras, other lighting. A landmark
    model trained on them transfers to YOUR webcam worse than 150 frames of
    your own hand does.
  - Sign Language MNIST is 28x28 grayscale. MediaPipe cannot use it at all.
    Don't bother.

Use this to pre-train and then top up with your own bursts, or use it if you
truly can't record. Otherwise collect_data.py wins for this project.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

import config
from landmarks import aspect_correct, normalize_landmarks

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_static_detector():
    opts = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(config.LANDMARKER_TASK)),
        running_mode=vision.RunningMode.IMAGE,   # stills, not video
        num_hands=1,
        min_hand_detection_confidence=0.3,       # loosened: crops are hard
    )
    return vision.HandLandmarker.create_from_options(opts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--per-class", type=int, default=300)
    ap.add_argument("--pad", type=int, default=40,
                    help="border px added around the image; helps palm detection on tight crops")
    ap.add_argument("--out", type=Path, default=config.DATA_DIR / "landmarks_public.csv")
    a = ap.parse_args()

    if not config.LANDMARKER_TASK.exists():
        sys.exit("run setup_model.py first")
    if not a.root.is_dir():
        sys.exit(f"not a directory: {a.root}")

    det = build_static_detector()
    rows, stats = [], {}

    for letter in config.LETTERS:
        d = a.root / letter
        if not d.is_dir():
            d = a.root / letter.lower()
        if not d.is_dir():
            print(f"  {letter}: no folder, skipped")
            continue

        files = sorted(p for p in d.iterdir() if p.suffix.lower() in EXTS)[:a.per_class]
        hits = 0
        for i, fp in enumerate(files):
            img = cv2.imread(str(fp))
            if img is None:
                continue
            img = cv2.copyMakeBorder(img, a.pad, a.pad, a.pad, a.pad,
                                     cv2.BORDER_REPLICATE)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = det.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            if not res.hand_landmarks:
                continue
            handed = res.handedness[0][0].category_name
            h, w = rgb.shape[:2]
            feats = normalize_landmarks(aspect_correct(res.hand_landmarks[0], w, h), handed)
            # group by chunk so train/test splits stay meaningful
            rows.append([letter, f"{letter}_pub{i // 50}"] + [f"{x:.6f}" for x in feats])
            hits += 1
        rate = hits / max(len(files), 1)
        stats[letter] = (hits, len(files), rate)
        print(f"  {letter}: {hits}/{len(files)} detected ({rate*100:.0f}%)")

    if not rows:
        sys.exit("nothing detected - wrong folder layout, or images too tightly cropped")

    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "group"] + [f"{ax}{i}" for i in range(21) for ax in ("x", "y", "z")])
        w.writerows(rows)

    overall = np.mean([s[2] for s in stats.values()])
    print(f"\nwrote {len(rows)} rows -> {a.out}")
    print(f"mean detection rate {overall*100:.0f}%  <- worth mentioning in the post")
    print(f"to use it: cp {a.out} {config.DATASET_CSV}   (or concat with your own)")


if __name__ == "__main__":
    main()
