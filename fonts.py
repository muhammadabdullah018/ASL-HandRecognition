"""
Typography. cv2.putText only has the built-in Hershey stroke fonts - they're
why every OpenCV demo looks like a lab prototype instead of a product. This
loads real variable fonts (Inter for UI text, JetBrains Mono for numeric
readouts) once and caches instances per (size, weight).

Both are open-source (SIL OFL, see assets/fonts/OFL-*.txt) and both ship as
single variable-font files, so "weight" is an axis we set at load time
rather than a separate file per weight.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from PIL import ImageFont

FONT_DIR = Path(__file__).parent / "assets" / "fonts"
INTER = FONT_DIR / "Inter-Variable.ttf"
MONO = FONT_DIR / "JetBrainsMono-Variable.ttf"

WEIGHTS = {"regular": 400, "medium": 500, "semibold": 600, "bold": 700}


def _check():
    for f in (INTER, MONO):
        if not f.exists():
            raise FileNotFoundError(
                f"Missing font: {f}\n"
                "assets/fonts/ should ship with the repo. If it's missing, "
                "re-download Inter and JetBrains Mono (variable, SIL OFL) "
                "from https://github.com/google/fonts/tree/main/ofl."
            )


@lru_cache(maxsize=64)
def inter(size: int, weight: str = "regular") -> ImageFont.FreeTypeFont:
    _check()
    f = ImageFont.truetype(str(INTER), size)
    opsz = max(14, min(32, size))  # Inter's optical-size axis: bigger text, wider cut
    f.set_variation_by_axes([opsz, WEIGHTS[weight]])
    return f


@lru_cache(maxsize=64)
def mono(size: int, weight: str = "regular") -> ImageFont.FreeTypeFont:
    _check()
    f = ImageFont.truetype(str(MONO), size)
    f.set_variation_by_axes([WEIGHTS[weight]])
    return f
