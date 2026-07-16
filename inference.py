"""
Step 4a. Live single-letter recognition. The sanity check before the demo.

Usage:
    python inference.py

Shows the predicted letter + confidence + top-3. If a letter is consistently
wrong here, go record more bursts of it and retrain. q to quit.
"""
from __future__ import annotations

import time
from collections import deque

import cv2
import joblib
import numpy as np

import config
import fonts as F
import ui
from landmarks import HandTracker


def main():
    if not config.CLASSIFIER_PKL.exists():
        raise SystemExit("no model - run train_model.py first")
    bundle = joblib.load(config.CLASSIFIER_PKL)
    model, labels = bundle["model"], bundle["labels"]
    print(f"{bundle['name']}  {bundle['accuracy']*100:.1f}%  [{bundle.get('eval','?')}]")

    tracker = HandTracker(num_hands=1)
    cap = cv2.VideoCapture(config.CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_H)
    if not cap.isOpened():
        raise SystemExit(f"camera {config.CAM_INDEX} would not open")

    fps_hist = deque(maxlen=30)
    prev = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        out = tracker.process(rgb, int(time.perf_counter() * 1000))

        layer = ui.TextLayer()
        ui.rounded_panel(frame, 20, 20, 330, 190, radius=20)

        if out is None:
            layer.text("No hand detected", (44, 74), F.inter(15, "medium"),
                       config.COL_BAD, anchor="lm")
        else:
            px, feats, handed = out
            proba = model.predict_proba(feats.reshape(1, -1))[0]
            top = np.argsort(proba)[::-1][:3]
            best, conf = labels[top[0]], float(proba[top[0]])
            col = config.COL_GOOD if conf >= config.CONF_THRESHOLD else config.COL_DIM
            ui.skeleton(frame, px, color=col if conf >= config.CONF_THRESHOLD else None)

            layer.text(best if conf >= config.CONF_THRESHOLD else "–", (44, 132),
                       F.inter(64, "semibold"), col, anchor="ls")
            layer.text(f"{conf*100:.0f}%  ·  {handed} hand", (150, 88),
                       F.inter(13, "medium"), config.COL_DIM, anchor="lm")
            ui.bar(frame, 150, 98, 170, 6, conf, col)
            for i, k in enumerate(top):
                layer.text(f"{labels[k]}", (150, 132 + i * 22), F.mono(13, "medium"),
                           config.COL_DIM, anchor="lm")
                layer.text(f"{proba[k]*100:4.1f}%", (320, 132 + i * 22), F.mono(12, "regular"),
                           config.COL_FAINT, anchor="rm")

        now = time.perf_counter()
        fps_hist.append(1.0 / max(now - prev, 1e-6)); prev = now
        layer.text(f"{np.mean(fps_hist):.0f} fps", (config.FRAME_W - 110, 40),
                   F.mono(13, "medium"), config.COL_DIM, anchor="lm")

        frame = layer.render(frame)
        cv2.imshow("asl - letter test  |  backspace/esc to quit", frame)
        if cv2.waitKey(1) & 0xFF in (8, 127, 27):
            break

    cap.release(); cv2.destroyAllWindows(); tracker.close()


if __name__ == "__main__":
    main()
