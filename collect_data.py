"""
Step 2. Record training samples from your webcam.

Usage:
    python collect_data.py



WHILE RECORDING, MOVE. Tilt the hand, rotate the wrist, drift nearer and
further, shift left and right in frame. 150 identical frames teach the model
nothing - it just memorises one pose and falls apart the moment your hand
sits half an inch differently. Variation is the whole point.

RECORD EACH LETTER IN AT LEAST 2 SEPARATE BURSTS. Every burst gets a group id.
train_model.py splits on that group so no near-duplicate frame lands in both
train and test. One burst per letter = you cannot measure yourself honestly.

Keys:  a-y = record that letter   |   TAB = show tally   |   q / ESC = quit
"""
from __future__ import annotations

import csv
import time
from collections import Counter

import cv2
import numpy as np

import config
import fonts as F
import ui
from landmarks import HandTracker, FEATURE_DIM


def load_tally() -> Counter:
    tally = Counter()
    if config.DATASET_CSV.exists():
        with open(config.DATASET_CSV, newline="") as f:
            r = csv.reader(f)
            next(r, None)
            for row in r:
                if row:
                    tally[row[0]] += 1
    return tally


def ensure_header():
    if not config.DATASET_CSV.exists():
        with open(config.DATASET_CSV, "w", newline="") as f:
            cols = ["label", "group"] + [f"{a}{i}" for i in range(21) for a in ("x", "y", "z")]
            csv.writer(f).writerow(cols)


def tally_strip(layer, x, y, tally, spacing=28):
    """Checklist row: each letter in its done/not-done color, monospaced
    so the row stays aligned as letters complete."""
    font = F.mono(15, "medium")
    for L in config.LETTERS:
        done = tally[L] >= config.SAMPLES_PER_LETTER
        layer.text(L, (x, y), font, config.COL_FG if done else config.COL_FAINT, anchor="lm")
        x += spacing


def main():
    ensure_header()
    tally = load_tally()
    tracker = HandTracker(num_hands=1)

    cap = cv2.VideoCapture(config.CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_H)
    if not cap.isOpened():
        raise RuntimeError(f"camera {config.CAM_INDEX} would not open")

    state = "idle"          # idle | countdown | recording
    target = None
    group_id = ""
    t_state = 0.0
    buffer: list[np.ndarray] = []

    print("ready. press a letter key (a-y) to record. BACKSPACE or ESC to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)                       # selfie view
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ts = int(time.perf_counter() * 1000)

        out = tracker.process(rgb, ts)
        if out is not None:
            px, feats, handed = out
            ui.skeleton(frame, px)
        else:
            feats = None

        now = time.perf_counter()
        layer = ui.TextLayer()

        # ---------------- state machine
        if state == "countdown":
            left = config.COUNTDOWN_SEC - (now - t_state)
            if left <= 0:
                state, buffer = "recording", []
                group_id = f"{target}_{int(time.time())}"   # burst id -> leak-free splits
            else:
                layer.text(str(int(left) + 1), (config.FRAME_W // 2, 400),
                           F.inter(140, "bold"), config.COL_ACCENT, anchor="mm")

        elif state == "recording":
            if feats is not None:
                buffer.append(feats)
            pct = len(buffer) / config.SAMPLES_PER_LETTER
            ui.bar(frame, 40, config.FRAME_H - 60, config.FRAME_W - 80, 14, pct, config.COL_GOOD)
            layer.text(f"REC {target}   {len(buffer)}/{config.SAMPLES_PER_LETTER}   keep moving",
                       (40, config.FRAME_H - 74), F.inter(15, "medium"), config.COL_GOOD, anchor="lm")

            if len(buffer) >= config.SAMPLES_PER_LETTER:
                with open(config.DATASET_CSV, "a", newline="") as f:
                    w = csv.writer(f)
                    for v in buffer:
                        w.writerow([target, group_id] + [f"{x:.6f}" for x in v])
                tally[target] += len(buffer)
                print(f"saved {len(buffer)} for {target}  (total {tally[target]})")
                state, target, buffer = "idle", None, []

        # ---------------- hud
        ui.rounded_panel(frame, 0, 0, config.FRAME_W, 84, radius=0, shadow=False)
        done = sum(1 for L in config.LETTERS if tally[L] >= config.SAMPLES_PER_LETTER)
        layer.text(f"{done}/24 letters", (24, 30), F.inter(16, "semibold"), config.COL_FG, anchor="lm")
        layer.text(f"{sum(tally.values())} samples", (150, 31), F.mono(13, "regular"),
                   config.COL_DIM, anchor="lm")
        tally_strip(layer, 24, 60, tally)

        if feats is None:
            ui.pill(frame, layer, config.FRAME_W - 150, 20, "NO HAND", config.COL_BAD)

        frame = layer.render(frame)
        cv2.imshow("collect  |  press a-y to record  |  backspace/esc to quit", frame)

        # ---------------- keys
        # NOTE: quit is backspace/ESC, not 'q' - Q is a real letter this
        # project trains (see config.LETTERS), and it's picked with the same
        # a-y keys as every other letter below. 'q' used to quit the whole
        # app instead of ever starting a Q recording.
        k = cv2.waitKey(1) & 0xFF
        if k in (8, 127, 27):
            break
        if k == 9:
            print({L: tally[L] for L in config.LETTERS})
        if state == "idle" and (97 <= k <= 122 or 65 <= k <= 90):
            ch = chr(k).upper()
            if ch in config.LETTERS:
                target, state, t_state = ch, "countdown", now
                print(f"get ready: {ch}")
            elif ch in ("J", "Z"):
                print("J and Z need motion - not supported by a single-frame model")



    cap.release()
    cv2.destroyAllWindows()
    tracker.close()
    print("\nfinal:", {L: tally[L] for L in config.LETTERS})
    missing = [L for L in config.LETTERS if tally[L] < config.SAMPLES_PER_LETTER]
    print("still short:", missing if missing else "nothing, go train")


if __name__ == "__main__":
    main()
