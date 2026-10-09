"""Unit tests of the ink-projection row bands.

The bands are pure image analysis (numpy/scipy/PIL), so these tests need neither
kraken nor the segmentation model: a synthetic page with drawn ink rows is
enough to check where the bands start and end.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from app.htr.infrastructure.kraken.rows import otsu_threshold, row_bands

PAPER = 235
INK = 25


def _page(width=400, height=600, bars=(), desk=False):
    """A light page with dark text rows; optionally inset on a dark desk."""
    canvas = np.full((height, width), INK if desk else PAPER, dtype=np.uint8)
    left, right, top, bottom = 0, width, 0, height
    if desk:
        left, right, top, bottom = 60, width - 60, 60, height - 60
        canvas[top:bottom, left:right] = PAPER
    for bar_top, bar_bottom in bars:
        canvas[bar_top:bar_bottom, left + 20 : right - 20] = INK
    return Image.fromarray(canvas, mode="L")


def test_otsu_splits_paper_from_ink():
    gray = np.array([PAPER] * 100 + [INK] * 100, dtype=np.uint8)
    assert INK < otsu_threshold(gray) < PAPER


def test_every_drawn_row_becomes_a_band():
    bars = [(100, 140), (250, 290), (400, 440)]
    bands = row_bands(_page(bars=bars))

    assert len(bands) == 3
    for (top, bottom), (bar_top, bar_bottom) in zip(bands, bars):
        assert abs(top - bar_top) <= 3
        assert abs(bottom - bar_bottom) <= 3


def test_the_desk_around_the_page_is_not_a_row():
    bars = [(200, 240), (400, 440), (600, 640)]
    bands = row_bands(_page(height=800, bars=bars, desk=True))

    assert len(bands) == 3
    # inside the page inset, never the dark frame at the top/bottom of the photo
    assert all(60 <= top and bottom <= 740 for top, bottom in bands)


def test_a_page_without_ink_has_no_rows():
    assert row_bands(_page()) == []


def test_bands_come_back_in_the_image_coordinates():
    """The mask is built downscaled; the bands must not be."""
    bands = row_bands(_page(height=900, bars=[(700, 740)]), max_side=200)

    assert len(bands) == 1
    assert abs(bands[0][0] - 700) <= 6
    assert abs(bands[0][1] - 740) <= 6


def test_bands_are_ordered_top_to_bottom():
    bands = row_bands(_page(bars=[(400, 440), (100, 140), (250, 290)]))

    tops = [top for top, _ in bands]
    assert tops == sorted(tops)
