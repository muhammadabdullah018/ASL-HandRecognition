"""
Hold-to-confirm logic. No OpenCV, no MediaPipe, no clock of its own -
time is injected. That's the whole reason it's a separate file: it means
you can unit test it in milliseconds instead of waving at a webcam.

Behaviour:
  - majority-vote the last N frames -> kills single-frame jitter
  - a letter must survive HOLD_SECONDS continuously before it commits
  - once committed it LOCKS: holding the same pose won't spam it
  - to type a double letter (HELLO) you break the pose and re-form it
"""
from __future__ import annotations

from collections import Counter, deque


class SpellTracker:
    def __init__(self, hold_seconds: float, window: int, conf_threshold: float):
        self.hold_seconds = hold_seconds
        self.conf_threshold = conf_threshold
        self.recent: deque[str | None] = deque(maxlen=window)
        self.text = ""
        self.candidate: str | None = None
        self.t_start: float | None = None
        self.locked = False          # already fired for this hold

    # ------------------------------------------------------------ core
    def update(self, letter: str | None, confidence: float, now: float) -> float:
        """
        Returns hold progress in [0,1] for the UI ring.
        Commits into self.text as a side effect.
        """
        if letter is None or confidence < self.conf_threshold:
            self.recent.append(None)
        else:
            self.recent.append(letter)

        smoothed = self._vote()

        if smoothed != self.candidate:
            # pose changed -> new hold, and the lock is released
            self.candidate = smoothed
            self.t_start = now if smoothed is not None else None
            self.locked = False
            return 0.0

        if smoothed is None or self.locked or self.t_start is None:
            return 0.0

        progress = (now - self.t_start) / self.hold_seconds
        if progress >= 1.0:
            self.text += smoothed
            self.locked = True
            return 1.0
        return progress

    def _vote(self) -> str | None:
        """Majority over the window. None wins ties and empty windows."""
        if not self.recent:
            return None
        counts = Counter(x for x in self.recent if x is not None)
        if not counts:
            return None
        letter, n = counts.most_common(1)[0]
        # needs a real majority of the window, not just the top of a mess
        return letter if n > len(self.recent) / 2 else None

    # ------------------------------------------------------------ editing
    def space(self):
        self.text += " "
        self._break()

    def backspace(self):
        self.text = self.text[:-1]
        self._break()

    def clear(self):
        self.text = ""
        self._break()

    def _break(self):
        self.recent.clear()
        self.candidate = None
        self.t_start = None
        self.locked = False


class GestureTracker:
    """
    Hold-to-confirm for a boolean gesture (e.g. two hands pressed together
    -> space), same shape as SpellTracker's letter hold but simpler: no
    smoothing window, no candidate letter, just active/not-active.

    Fires exactly once per hold (locked, like SpellTracker), and won't fire
    again until the gesture breaks (hands separate) and reforms - so holding
    your hands together doesn't spam spaces.
    """

    def __init__(self, hold_seconds: float):
        self.hold_seconds = hold_seconds
        self.t_start: float | None = None
        self.locked = False

    def update(self, active: bool, now: float) -> tuple[float, bool]:
        """Returns (hold progress in [0,1], just_fired this frame)."""
        if not active:
            self.t_start = None
            self.locked = False
            return 0.0, False

        if self.t_start is None:
            self.t_start = now

        if self.locked:
            return 1.0, False

        progress = (now - self.t_start) / self.hold_seconds
        if progress >= 1.0:
            self.locked = True
            return 1.0, True
        return progress, False
