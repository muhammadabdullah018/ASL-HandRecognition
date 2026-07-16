"""
Step 4b. SPELL MODE. This is the one you screen record.

    python spell_mode.py

Hold a letter -> a ring fills around it -> it locks into the word strip at the
bottom. Break the pose, hold the next one. Make a fist with both hands and
press them together to add a space instead of reaching for the keyboard.
Spell HELLO and the screen types it.

WHO DOES WHAT
-------------
MediaPipe : finds up to 2 hands, returns 21 joint coordinates each. That's
            its whole job.
OpenCV    : opens the camera, reads/flips frames, draws every shape in this
            overlay, shows the window, reads keys.
PIL       : renders every piece of text (Inter / JetBrains Mono) - OpenCV's
            built-in font can't do that on its own. See ui.py's docstring.
They are not alternatives. Each does the one job the others can't.

GESTURES
    Both hands in a fist, pressed together, held briefly
        -> adds a space. SPACE key still works too - the gesture is an
        alternative, not a replacement. Requiring a fist (not just
        proximity) makes it a deliberate pose you won't drift into by
        resting your hands near each other between letters.
    Both hands raised to the top of frame, held a beat longer
        -> big "ta-da" reveal of whatever you've spelled so far, with
        confetti. Purely a celebration overlay - it doesn't clear the text
        or pause recognition, everything keeps running underneath it.
    Both use the same hold-then-lock shape as letters (GestureTracker in
    spell_tracker.py): won't refire while you hold the pose, has to break
    and reform to fire again.

KEYS
    [ / ]          hold time -0.1 / +0.1 s     <- tune it live, on camera
    SPACE          insert a space (same effect as the two-hands gesture)
    D              delete last letter
    C              clear the output
    R              clean mode (hides the HUD for recording)
    BACKSPACE/ESC  quit

    Quit moved off Q on purpose: Q is a real letter this model recognizes,
    and collect_data.py uses the same a-y keys to pick which letter to
    record - 'q' used to quit the whole app instead of ever letting you
    record Q. Backspace was free (this file's own delete-letter action
    moved to D to make room).
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
from landmarks import HandTracker, hands_raised, is_fist
from spell_tracker import GestureTracker, SpellTracker


def main():
    if not config.CLASSIFIER_PKL.exists():
        raise SystemExit("no model - run train_model.py first")
    bundle = joblib.load(config.CLASSIFIER_PKL)
    model, labels = bundle["model"], bundle["labels"]
    print(f"{bundle['name']}  {bundle['accuracy']*100:.1f}%  [{bundle.get('eval','?')}]")
    print(f"hold {config.HOLD_SECONDS}s to lock a letter.  [ and ] to adjust.  "
          f"hands together = space.  hands up = celebrate.  backspace/esc to quit.")

    tracker = HandTracker(num_hands=2)
    speller = SpellTracker(config.HOLD_SECONDS, config.SMOOTHING_WINDOW,
                           config.CONF_THRESHOLD)
    gesture = GestureTracker(config.SPACE_HOLD_SECONDS)
    celebrate_gesture = GestureTracker(config.CELEBRATE_HOLD_SECONDS)

    cap = cv2.VideoCapture(config.CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_H)
    if not cap.isOpened():
        raise SystemExit(f"camera {config.CAM_INDEX} would not open")

    W, H = config.FRAME_W, config.FRAME_H
    fps_hist = deque(maxlen=30)
    prev = time.perf_counter()
    clean = False
    flash_until = 0.0
    last_locked = ""
    celebrate_until = 0.0
    confetti: list[tuple[int, int, int, tuple[int, int, int]]] = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(cv2.flip(frame, 1), (W, H))
        ui.vignette(frame)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        now = time.perf_counter()

        hands = tracker.process_all(rgb, int(now * 1000))

        letter, conf, proba = None, 0.0, None
        together, raised = False, False
        if len(hands) >= 2:
            (pxA, _, _, corrA), (pxB, _, _, corrB) = hands[:2]
            together = is_fist(corrA) and is_fist(corrB)
            raised = (not together) and hands_raised(pxA, pxB, H, config.HANDS_UP_FRAC)
        elif len(hands) == 1:
            _, feats, _, _ = hands[0]
            proba = model.predict_proba(feats.reshape(1, -1))[0]
            k = int(np.argmax(proba))
            letter, conf = labels[k], float(proba[k])

        before = len(speller.text)
        g_progress, g_fired = gesture.update(together, now)
        if g_fired:
            speller.space()
        progress = speller.update(letter, conf, now)
        if len(speller.text) > before:
            flash_until = now + 0.35
            last_locked = speller.text[-1]

        c_progress, c_fired = celebrate_gesture.update(raised, now)
        if c_fired:
            celebrate_until = now + config.CELEBRATE_SHOW_SECONDS
            rng = np.random.default_rng()
            palette = [config.COL_GOOD, config.COL_ACCENT, (170, 210, 255)]
            confetti = [
                (W // 2 + int(rng.uniform(-330, 330)), H // 2 + int(rng.uniform(-150, 150)),
                 int(rng.uniform(3, 7)), palette[int(rng.integers(0, len(palette)))])
                for _ in range(28)
            ]
        celebrating = now < celebrate_until

        flashing = now < flash_until
        layer = ui.TextLayer()

        # ---------------------------------------------------------- skeleton
        if len(hands) >= 2:
            sk_col = config.COL_GOOD if (flashing or celebrating) else (
                config.COL_ACCENT if (together or raised) else config.COL_BONE)
            for px, *_ in hands[:2]:
                ui.skeleton(frame, px, color=sk_col)
        elif len(hands) == 1:
            if flashing:
                bone = config.COL_GOOD
            elif progress > 0:
                bone = config.COL_ACCENT
            else:
                bone = config.COL_BONE
            ui.skeleton(frame, hands[0][0], color=bone)

        # ---------------------------------------------------------- left card
        ui.rounded_panel(frame, 24, 24, 300, 372, radius=22)
        cx, cy = 174, 150

        active_progress = g_progress if together else (c_progress if raised else progress)
        ui.ring(frame, (cx, cy), 74, active_progress, thickness=6)
        if flashing:
            ui.ring(frame, (cx, cy), 74, 1.0, thickness=6)

        if together:
            col = config.COL_GOOD if flashing else config.COL_ACCENT
            ui.bar(frame, cx - 34, cy - 11, 68, 22, 1.0, col)
            layer.text("SPACE", (cx, cy + 40), F.inter(13, "semibold"), col, anchor="mm")
        elif raised:
            col = config.COL_ACCENT
            tri = np.array([[cx, cy - 18], [cx - 16, cy + 10], [cx + 16, cy + 10]], np.int32)
            cv2.fillPoly(frame, [tri], col, cv2.LINE_AA)
            layer.text("CELEBRATE", (cx, cy + 40), F.inter(12, "semibold"), col, anchor="mm")
        elif letter is not None and conf >= config.CONF_THRESHOLD:
            col = config.COL_GOOD if flashing else config.COL_FG
            layer.text(letter, (cx, cy + 4), F.inter(80, "semibold"), col, anchor="mm")
        else:
            layer.text("–", (cx, cy + 4), F.inter(80, "semibold"), config.COL_FAINT, anchor="mm")

        if flashing:
            label = "SPACE ADDED" if last_locked == " " else f"LOCKED {last_locked}"
            layer.text(label, (cx, cy + 100), F.inter(13, "semibold"),
                       config.COL_GOOD, anchor="mm")
        elif active_progress > 0:
            hold_total = (config.SPACE_HOLD_SECONDS if together else
                          config.CELEBRATE_HOLD_SECONDS if raised else config.HOLD_SECONDS)
            remain = hold_total * (1 - active_progress)
            layer.text(f"{remain:.1f}s", (cx, cy + 100), F.mono(13, "medium"),
                       config.COL_ACCENT, anchor="mm")

        if proba is not None:
            ui.caption(layer, "top 3", (54, 270))
            ui.topk_bars(frame, layer, 54, 288, labels, proba, k=3, w=150)

        # ---------------------------------------------------------- status
        if len(hands) == 0:
            ui.pill(frame, layer, 24, 410, "NO HAND", config.COL_BAD)
        elif flashing:
            label = "SPACE ADDED" if last_locked == " " else f"LOCKED  {last_locked}"
            ui.pill(frame, layer, 24, 410, label, config.COL_GOOD)
        elif together:
            ui.pill(frame, layer, 24, 410, "SPACE", config.COL_ACCENT)
        elif raised:
            ui.pill(frame, layer, 24, 410, "CELEBRATE", config.COL_ACCENT)
        elif len(hands) == 1 and conf < config.CONF_THRESHOLD:
            ui.pill(frame, layer, 24, 410, "UNSURE", config.COL_DIM)
        elif progress > 0:
            ui.pill(frame, layer, 24, 410, f"HOLDING  {letter}", config.COL_ACCENT)
        elif len(hands) == 1:
            ui.pill(frame, layer, 24, 410, f"READY  {letter}", config.COL_DIM)
        else:
            ui.pill(frame, layer, 24, 410, "TWO HANDS", config.COL_DIM)

        # ---------------------------------------------------------- rail
        if not clean:
            ui.alphabet_rail(frame, layer, letter if (conf >= config.CONF_THRESHOLD and not together) else None,
                             W // 2, H - 178)

        # ---------------------------------------------------------- output
        strip_h = 122
        sy = H - strip_h - 20
        ui.rounded_panel(frame, 24, sy, W - 48, strip_h, radius=22)
        ui.caption(layer, "output", (48, sy + 20))

        caret = "_" if int(now * 2) % 2 == 0 else " "
        body = config.COL_GOOD if flashing else config.COL_FG
        ui.fit_text(layer, speller.text + caret, (48, sy + 92), W - 200, body,
                    sizes=(40, 34, 28, 24, 20), anchor="lm")
        layer.text(f"{len(speller.text)}", (W - 72, sy + 92), F.mono(15, "regular"),
                   config.COL_FAINT, anchor="rm")

        # ---------------------------------------------------------- hud
        if not clean:
            ui.rounded_panel(frame, W - 264, 24, 240, 96, radius=18, shadow=False)
            layer.text(f"{np.mean(fps_hist) if fps_hist else 0:.0f} fps",
                       (W - 244, 50), F.mono(13, "medium"), config.COL_DIM, anchor="lm")
            layer.text(f"hold {config.HOLD_SECONDS:.1f}s", (W - 244, 76),
                       F.mono(13, "regular"), config.COL_FAINT, anchor="lm")
            layer.text(f"{bundle['accuracy']*100:.0f}% acc  ·  {len(labels)} letters",
                       (W - 244, 100), F.mono(12, "regular"), config.COL_FAINT, anchor="lm")

        # ---------------------------------------------------------- celebration
        if celebrating:
            remain_frac = (celebrate_until - now) / config.CELEBRATE_SHOW_SECONDS
            ui.celebration(frame, layer, speller.text, W, H, confetti, remain_frac)

        now2 = time.perf_counter()
        fps_hist.append(1.0 / max(now2 - prev, 1e-6))
        prev = now2

        frame = layer.render(frame)
        cv2.imshow("ASL fingerspelling", frame)

        # ---------------------------------------------------------- keys
        k = cv2.waitKey(1) & 0xFF
        if k in (8, 127, 27):
            break
        elif k == 32:
            speller.space()
        elif k == ord("d"):
            speller.backspace()
        elif k == ord("c"):
            speller.clear()
        elif k == ord("r"):
            clean = not clean
        elif k == ord("]"):
            config.HOLD_SECONDS = min(5.0, config.HOLD_SECONDS + 0.1)
            speller.hold_seconds = config.HOLD_SECONDS
        elif k == ord("["):
            config.HOLD_SECONDS = max(0.3, config.HOLD_SECONDS - 0.1)
            speller.hold_seconds = config.HOLD_SECONDS

    cap.release()
    cv2.destroyAllWindows()
    tracker.close()
    print(f"\nspelled: {speller.text!r}")


if __name__ == "__main__":
    main()
