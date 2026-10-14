"""Pixel sizes for the two stages.

Stage 1 is the AI master: the source fitted into a square box, never enlarged more than 4x, because past 4x
SeedVR2 starts inventing material. Stage 2 is a plain resize to a pixel budget, aspect kept.
"""

import math

BOX = 8192
MAX_FACTOR = 4.0
DIVISIBLE_BY = 8
# 4 x 3 m at 300 DPI is 47244 x 35433 px, the pixel budget we aim for whatever the aspect
POSTER_MEGAPIXELS = 47244 * 35433 / 1e6


def stage1_size(src_w: int, src_h: int, box: int = BOX, max_factor: float = MAX_FACTOR) -> tuple[int, int, bool]:
    """Size of the stage 1 master, mirroring ImageResizeKJv2 in 'resize' mode. Returns (w, h, capped)."""
    fit = min(box / src_w, box / src_h)
    ratio = min(fit, max_factor)
    w, h = round(src_w * ratio), round(src_h * ratio)
    return w - w % DIVISIBLE_BY, h - h % DIVISIBLE_BY, ratio < fit


def poster_size(w: int, h: int, megapixels: float = POSTER_MEGAPIXELS, long_side: int | None = None) -> tuple[int, int]:
    """Final size for a w x h master, from a pixel budget or a long side, aspect kept."""
    scale = long_side / max(w, h) if long_side else math.sqrt(megapixels * 1e6 / (w * h))
    return round(w * scale), round(h * scale)


def physical_cm(px: int, dpi: float) -> float:
    return px / dpi * 2.54
