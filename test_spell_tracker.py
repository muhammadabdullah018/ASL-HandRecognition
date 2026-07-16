"""
python test_spell_tracker.py
Fake clock, no webcam. Runs in ~10ms.
"""
from spell_tracker import GestureTracker, SpellTracker

HOLD, WIN, CONF = 0.9, 5, 0.80
GHOLD = 0.6


def feed(t, letter, seconds, conf=0.95, t0=0.0, step=1 / 30):
    """Simulate holding `letter` for `seconds` at 30fps."""
    n = int(seconds / step)
    for i in range(n):
        t.update(letter, conf, t0 + i * step)
    return t0 + n * step


def test_commits_after_hold():
    t = SpellTracker(HOLD, WIN, CONF)
    feed(t, "A", 1.2)
    assert t.text == "A", t.text


def test_no_commit_if_too_short():
    t = SpellTracker(HOLD, WIN, CONF)
    feed(t, "A", 0.5)
    assert t.text == "", t.text


def test_no_refire_while_holding():
    t = SpellTracker(HOLD, WIN, CONF)
    feed(t, "A", 5.0)
    assert t.text == "A", f"spammed: {t.text}"


def test_low_confidence_blocked():
    t = SpellTracker(HOLD, WIN, CONF)
    feed(t, "A", 3.0, conf=0.40)
    assert t.text == "", t.text


def test_double_letter_needs_a_break():
    t = SpellTracker(HOLD, WIN, CONF)
    now = feed(t, "L", 1.2)
    now = feed(t, None, 0.5, t0=now)      # break the pose
    feed(t, "L", 1.2, t0=now)
    assert t.text == "LL", t.text


def test_spells_hello():
    t = SpellTracker(HOLD, WIN, CONF)
    now = 0.0
    for ch in "HELLO":
        now = feed(t, ch, 1.2, t0=now)
        now = feed(t, None, 0.4, t0=now)  # drop hand between letters
    assert t.text == "HELLO", t.text


def test_switching_letters_resets_progress():
    t = SpellTracker(HOLD, WIN, CONF)
    now = feed(t, "A", 0.6)
    feed(t, "B", 0.6, t0=now)
    assert t.text == "", t.text          # neither held long enough


def test_editing():
    t = SpellTracker(HOLD, WIN, CONF)
    feed(t, "A", 1.2)
    t.space(); t.backspace(); t.backspace()
    assert t.text == "", t.text


# ------------------------------------------------------------ GestureTracker
def feed_gesture(g, active, seconds, t0=0.0, step=1 / 30):
    n = int(seconds / step)
    fired_at = None
    for i in range(n):
        t = t0 + i * step
        _, fired = g.update(active, t)
        if fired and fired_at is None:
            fired_at = t
    return t0 + n * step, fired_at


def test_gesture_fires_after_hold():
    g = GestureTracker(GHOLD)
    _, fired_at = feed_gesture(g, True, 1.0)
    assert fired_at is not None, "never fired"


def test_gesture_no_fire_if_too_short():
    g = GestureTracker(GHOLD)
    _, fired_at = feed_gesture(g, True, 0.3)
    assert fired_at is None, "fired too early"


def test_gesture_fires_only_once_while_held():
    g = GestureTracker(GHOLD)
    fire_count = 0
    now = 0.0
    for i in range(int(2.0 / (1 / 30))):
        now = i / 30
        _, fired = g.update(True, now)
        fire_count += fired
    assert fire_count == 1, f"fired {fire_count} times while held"


def test_gesture_refires_after_break_and_reform():
    g = GestureTracker(GHOLD)
    now, first = feed_gesture(g, True, 1.0)
    assert first is not None
    now, _ = feed_gesture(g, False, 0.3, t0=now)   # hands separate
    _, second = feed_gesture(g, True, 1.0, t0=now)
    assert second is not None, "did not refire after breaking the gesture"


def test_gesture_progress_resets_on_break():
    g = GestureTracker(GHOLD)
    now, _ = feed_gesture(g, True, 0.3)   # partway through the hold
    progress, fired = g.update(False, now)
    assert progress == 0.0 and not fired


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} passed")
