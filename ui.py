"""
Drawing primitives. This file is why the demo looks like a product instead
of a homework screenshot. Nothing here is ML.

DESIGN SYSTEM
-------------
Dark, Apple-system palette (config.py) + Inter/JetBrains Mono (fonts.py)
instead of OpenCV's built-in Hershey fonts. Everything is built from one
repeated shape - the rounded, softly-shadowed glass panel - the same way
real interfaces reuse one card component everywhere instead of one-off
shapes per screen.

OpenCV can't draw anti-aliased custom-font text, so text goes through PIL.
Converting frame<->PIL is not free, so text calls don't draw immediately -
they queue on a TextLayer and get stamped in ONE conversion per frame via
.render(). Shapes (panels, rings, bars, skeleton) stay pure OpenCV and draw
straight onto the frame array as before, cheaply, at 30fps+.

Ordering rule: draw every shape for the frame first (as before), queue all
text alongside it, then call layer.render(frame) once, right before
cv2.imshow. Nothing in this UI draws a shape on top of another panel's
text, so deferring all text to the end is safe.
"""
from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw

import config
import fonts as F


# ============================================================== text layer
class TextLayer:
    """Queues text draws; stamps them all in one PIL round-trip via render()."""

    def __init__(self):
        self._ops = []

    def text(self, s, xy, font, color, anchor="la", tracking=0):
        self._ops.append((str(s), (int(xy[0]), int(xy[1])), font, color, anchor, tracking))

    def render(self, frame_bgr: np.ndarray) -> np.ndarray:
        if not self._ops:
            return frame_bgr
        img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        for s, xy, font, color, anchor, tracking in self._ops:
            rgb = (color[2], color[1], color[0])
            if tracking <= 0:
                draw.text(xy, s, font=font, fill=rgb, anchor=anchor)
            else:
                x, y = xy
                va = anchor[1] if len(anchor) > 1 else "a"
                for ch in s:
                    draw.text((x, y), ch, font=font, fill=rgb, anchor="l" + va)
                    x += draw.textlength(ch, font=font) + tracking
        self._ops.clear()
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def text_width(s: str, font) -> int:
    return int(font.getlength(s))


def line_height(font) -> int:
    a, d = font.getmetrics()
    return a + d


def caption(layer: TextLayer, s, xy, color=None, anchor="la"):
    """Small tracked-caps section label, e.g. OUTPUT / TOP 3. Vercel/Apple habit:
    quiet gray, wide letter-spacing, tells you what a panel is without shouting."""
    layer.text(s.upper(), xy, F.inter(11, "semibold"), color or config.COL_FAINT,
               anchor=anchor, tracking=2)


def fit_text(layer: TextLayer, s, xy, max_w, color, sizes=(40, 34, 28, 24, 20),
             weight="semibold", mono=True, anchor="lm"):
    """Pick the largest size that fits max_w. For the growing output word."""
    font_fn = F.mono if mono else F.inter
    for size in sizes:
        f = font_fn(size, weight)
        if text_width(s, f) <= max_w or size == sizes[-1]:
            layer.text(s, xy, f, color, anchor=anchor)
            return size
    return sizes[-1]


# ============================================================== rounded shapes
@lru_cache(maxsize=64)
def _rounded_mask(w, h, radius):
    w, h = max(w, 1), max(h, 1)
    radius = max(0, min(radius, w // 2, h // 2))
    mask = np.zeros((h, w), np.uint8)
    cv2.rectangle(mask, (radius, 0), (w - radius, h), 255, -1)
    cv2.rectangle(mask, (0, radius), (w, h - radius), 255, -1)
    for cx, cy in ((radius, radius), (w - radius, radius),
                   (radius, h - radius), (w - radius, h - radius)):
        cv2.circle(mask, (cx, cy), radius, 255, -1)
    return mask


def _paint(roi, mask, color_or_array):
    """Copy `color_or_array` into `roi` wherever `mask` is nonzero.

    This looks equivalent to `roi[mask > 0] = src[mask > 0]`, but that numpy
    boolean-fancy-index form measured ~15ms for a single ~300x372 panel -
    almost the entire per-frame render budget on its own, because fancy
    indexing on a 3-channel array does an unvectorized element-by-element
    gather instead of a blit. cv2.copyTo does the identical masked copy in
    C++ in ~0.05ms. This one swap is most of the FPS difference."""
    src = color_or_array if isinstance(color_or_array, np.ndarray) else \
        np.full_like(roi, color_or_array)
    cv2.copyTo(src, mask, roi)


def rounded_panel(frame, x, y, w, h, radius=18, fill=None, alpha=0.80,
                   border=True, shadow=False, border_color=None):
    """Rounded translucent glass card with a soft drop shadow and hairline
    border - the one component every panel in this UI is built from."""
    H, W = frame.shape[:2]

    if shadow:
        pad = 10
        sx, sy = max(x - pad, 0), max(y - pad + 6, 0)
        ex, ey = min(x + w + pad, W), min(y + h + pad + 6, H)
        if ex > sx and ey > sy:
            roi = frame[sy:ey, sx:ex]
            glow = np.zeros_like(roi)
            gx0, gy0 = x - sx, y - sy
            m = _rounded_mask(w, h, radius)
            gy1, gx1 = min(gy0 + h, roi.shape[0]), min(gx0 + w, roi.shape[1])
            if gy1 > max(gy0, 0) and gx1 > max(gx0, 0):
                yy0, xx0 = max(gy0, 0), max(gx0, 0)
                sub_m = m[yy0 - gy0:gy1 - gy0, xx0 - gx0:gx1 - gx0]
                _paint(glow[yy0:gy1, xx0:gx1], sub_m, (255, 255, 255))
                glow = cv2.GaussianBlur(glow, (0, 0), 10)
                cv2.subtract(roi, (glow * 0.45).astype(np.uint8), dst=roi)

    x2, y2 = min(x + w, W), min(y + h, H)
    xa, ya = max(x, 0), max(y, 0)
    if x2 <= xa or y2 <= ya:
        return
    roi = frame[ya:y2, xa:x2]
    mask = _rounded_mask(x2 - xa, y2 - ya, radius)
    plate = np.full_like(roi, fill or config.COL_PANEL)
    blended = cv2.addWeighted(roi, 1 - alpha, plate, alpha, 0)
    _paint(roi, mask, blended)

    if border:
        eroded = cv2.erode(mask, np.ones((2, 2), np.uint8))
        edge = cv2.subtract(mask, eroded)
        _paint(roi, edge, border_color or config.COL_BORDER)


def panel(frame, x, y, w, h, alpha=0.72, radius=18):
    """Back-compat alias for the old flat scrim call sites."""
    rounded_panel(frame, x, y, w, h, radius=radius, alpha=alpha, shadow=False)


scrim = panel  # legacy name


def bar(frame, x, y, w, h, frac, color, track=None):
    """Rounded-cap progress bar."""
    H, W = frame.shape[:2]
    x2, y2 = min(x + w, W), min(y + h, H)
    if x2 <= x or y2 <= y:
        return
    roi = frame[y:y2, x:x2]
    r = (y2 - y) // 2
    mask = _rounded_mask(x2 - x, y2 - y, r)
    _paint(roi, mask, track or config.COL_BORDER)
    if frac > 0:
        fw = min(max(int((x2 - x) * min(frac, 1.0)), y2 - y), x2 - x)
        fmask = _rounded_mask(fw, y2 - y, r)
        _paint(roi[:, :fw], fmask, color)


def ring(frame, center, radius, progress, thickness=6):
    """Circular hold-progress arc, 12 o'clock clockwise, rounded caps."""
    cv2.circle(frame, center, radius, config.COL_BORDER, thickness, cv2.LINE_AA)
    if progress <= 0:
        return
    col = config.COL_GOOD if progress >= 1.0 else config.COL_ACCENT
    end_deg = int(360 * min(progress, 1.0))
    cv2.ellipse(frame, center, (radius, radius), -90, 0, end_deg, col, thickness, cv2.LINE_AA)
    cap_r = max(thickness // 2, 2)
    cv2.circle(frame, (center[0], center[1] - radius), cap_r, col, -1, cv2.LINE_AA)
    theta = np.deg2rad(end_deg - 90)
    ex = int(center[0] + radius * np.cos(theta))
    ey = int(center[1] + radius * np.sin(theta))
    cv2.circle(frame, (ex, ey), cap_r, col, -1, cv2.LINE_AA)


def pill(frame, layer: TextLayer, x, y, label, color, h=30):
    """Tinted status capsule: colored dot + colored label on a low-alpha
    tint of the same color. Same language as Vercel deployment-status pills."""
    font = F.inter(13, "medium")
    w = text_width(label, font) + h + 14
    H, W = frame.shape[:2]
    x2, y2 = min(x + w, W), min(y + h, H)
    if x2 > x and y2 > y:
        roi = frame[y:y2, x:x2]
        mask = _rounded_mask(x2 - x, y2 - y, (y2 - y) // 2)
        tint = np.full_like(roi, color)
        blended = cv2.addWeighted(roi, 0.80, tint, 0.20, 0)
        _paint(roi, mask, blended)
        eroded = cv2.erode(mask, np.ones((2, 2), np.uint8))
        edge = cv2.subtract(mask, eroded)
        _paint(roi, edge, color)
    cv2.circle(frame, (x + h // 2, y + h // 2), 4, color, -1, cv2.LINE_AA)
    layer.text(label, (x + h // 2 + 12, y + h // 2 + 1), font, color, anchor="lm")
    return w


# ============================================================== hand skeleton
def skeleton(img, px, color=None, glow=True):
    """
    Hand skeleton with a restrained additive glow - soft white idle state,
    closer to visionOS hand-tracking than a neon CV demo.

    NOTE: the glow is composited from a separate layer on purpose. The
    obvious version - GaussianBlur(img, dst=img) - blurs the ENTIRE frame
    including your face. Looks like a broken webcam driver on video.
    """
    col = color or config.COL_BONE
    if glow:
        layer = np.zeros_like(img)
        for a, b in config.HAND_EDGES:
            cv2.line(layer, tuple(px[a]), tuple(px[b]), col, 5, cv2.LINE_AA)
        layer = cv2.GaussianBlur(layer, (0, 0), 6)
        cv2.add(img, (layer * 0.55).astype(np.uint8), dst=img)
    for a, b in config.HAND_EDGES:
        cv2.line(img, tuple(px[a]), tuple(px[b]), col, 2, cv2.LINE_AA)
    for i, (x, y) in enumerate(px):
        r = 5 if i in (4, 8, 12, 16, 20) else 3
        cv2.circle(img, (x, y), r, config.COL_JOINT, -1, cv2.LINE_AA)
        cv2.circle(img, (x, y), r, config.COL_BG, 1, cv2.LINE_AA)


# ============================================================== composite widgets
def alphabet_rail(frame, layer: TextLayer, active, cx, y, spacing=42):
    """
    The 24 supported letters in a row, active one lit in a soft accent
    capsule. Doubles as documentation: it shows at a glance that J and Z
    aren't there, so nobody in the comments has to ask.
    """
    letters = config.LETTERS
    total = len(letters) * spacing
    x = cx - total // 2
    active_font = F.inter(16, "semibold")
    idle_font = F.inter(13, "regular")
    H, W = frame.shape[:2]
    for L in letters:
        on = (L == active)
        cxp = x + spacing // 2
        if on:
            r = 17
            x1, y1 = max(cxp - r, 0), max(y - r, 0)
            x2, y2 = min(cxp + r, W), min(y + r, H)
            if x2 > x1 and y2 > y1:
                roi = frame[y1:y2, x1:x2]
                mask = _rounded_mask(x2 - x1, y2 - y1, r)
                tint = np.full_like(roi, config.COL_ACCENT)
                blended = cv2.addWeighted(roi, 0.72, tint, 0.28, 0)
                _paint(roi, mask, blended)
            layer.text(L, (cxp, y + 1), active_font, config.COL_FG, anchor="mm")
        else:
            layer.text(L, (cxp, y), idle_font, config.COL_FAINT, anchor="mm")
        x += spacing


def topk_bars(frame, layer: TextLayer, x, y, labels, probs, k=3, w=150):
    """Top-k probabilities. Shows the model's doubt instead of hiding it."""
    order = np.argsort(probs)[::-1][:k]
    label_font = F.mono(13, "medium")
    pct_font = F.mono(12, "regular")
    for i, idx in enumerate(order):
        yy = y + i * 24
        col = config.COL_FG if i == 0 else config.COL_FAINT
        layer.text(labels[idx], (x, yy + 7), label_font, col, anchor="lm")
        bar(frame, x + 24, yy + 1, w, 6, float(probs[idx]),
            config.COL_ACCENT if i == 0 else config.COL_BORDER)
        layer.text(f"{probs[idx]*100:3.0f}%", (x + 34 + w, yy + 7), pct_font,
                   config.COL_FAINT, anchor="lm")


def celebration(frame, layer: TextLayer, text, W, H, confetti=(), remain_frac=1.0):
    """
    Big centered reveal for the both-hands-up gesture: what you spelled,
    shown large, with a confetti scatter and a countdown bar for when it'll
    auto-dismiss. A deliberate "ta-da" moment - separate from the quiet
    per-letter lock flash, on purpose: this one is for showing the room.
    """
    w, h = min(int(W * 0.72), 860), 260
    x, y = (W - w) // 2, (H - h) // 2

    for cx, cy, r, col in confetti:
        cv2.circle(frame, (cx, cy), r, col, -1, cv2.LINE_AA)

    rounded_panel(frame, x, y, w, h, radius=28, alpha=0.93,
                  border_color=config.COL_GOOD)
    caption(layer, "spelled", (x + 32, y + 30), config.COL_GOOD)
    fit_text(layer, text or "—", (x + w // 2, y + h // 2 + 14), w - 100,
             config.COL_FG, sizes=(64, 54, 46, 38, 30), weight="bold",
             mono=False, anchor="mm")
    bar(frame, x + 28, y + h - 26, w - 56, 6, remain_frac, config.COL_GOOD)


# ============================================================== frame treatment
_vignette_cache: dict[tuple[int, int], np.ndarray] = {}


def vignette(frame, strength=0.32):
    pass
