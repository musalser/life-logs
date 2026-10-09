"""Physical text rows found in the ink of the page.

A baseline segmenter answers "where is a baseline", not "what is a row": on a
logbook page whose words and numbers stand far apart it returns several
baselines per physical row, and recognition would see each of them as a line.
The horizontal profile of the page ink answers the other question — where the
rows are — because the pieces of one row share a band of dark pixels.

The bands are deliberately used to *add* merges to the geometric one (see
``recognizer.merge_collinear_lines``): a band that is too wide can glue two
tightly spaced rows, so the geometric decision is never undone, and a page
without a usable profile falls back to it entirely.

Measured on the corpus (phone photos of a diary and of a logbook): the bands
leave the 28–32 rows of the diary pages alone and turn the logbook page that the
geometric merge kept at 8 lines into its 7 physical rows.
"""
from __future__ import annotations

import logging

import numpy as np
from PIL import Image
from scipy import ndimage

logger = logging.getLogger(__name__)

#: the mask is built on a downscaled copy: full-resolution morphology on a
#: 3472×4640 phone photo costs seconds and finds the same bands
MASK_MAX_SIDE = 1500
#: how much darker than the paper a pixel must be to count as ink. The printed
#: grid of a squared notebook is lighter than this, so it does not become a row.
INK_MARGIN = 25
#: width of the moving average over the profile, as a fraction of the image
#: height: it closes the gaps between words without merging two rows
SMOOTHING_FRACTION = 1 / 150
#: a row band starts where the smoothed profile passes this fraction of its
#: maximum. Tightly spaced diary rows need the higher end (their rows nearly
#: touch), sparse logbook rows are far below it either way.
DEFAULT_THRESHOLD_RATIO = 0.4
#: bands shorter than this (in downscaled pixels) are paper texture, not text
MIN_BAND_PIXELS = 3


def row_bands(
    image: Image.Image,
    *,
    threshold_ratio: float = DEFAULT_THRESHOLD_RATIO,
    max_side: int = MASK_MAX_SIDE,
) -> list[tuple[int, int]]:
    """``(top, bottom)`` y ranges of the page's text rows, top to bottom.

    Empty when there is no usable ink: a blank page, or paper that cannot be
    told from the background. Coordinates are in ``image`` pixels, never in the
    downscaled mask's.
    """
    gray = _downscaled_gray(image, max_side)
    if gray.size == 0:
        return []
    threshold = otsu_threshold(gray)
    paper = _paper_mask(gray, threshold)
    if not paper.any():
        return []
    ink = (gray < threshold - INK_MARGIN) & paper
    profile = _moving_average(
        ink.sum(axis=1).astype(np.float32),
        max(3, round(gray.shape[0] * SMOOTHING_FRACTION)),
    )
    peak = float(profile.max()) if profile.size else 0.0
    if peak <= 0:
        return []
    scale = image.height / gray.shape[0]
    return [
        (round(top * scale), round(bottom * scale))
        for top, bottom in _bands_above(profile, threshold_ratio * peak)
    ]


def otsu_threshold(gray: np.ndarray) -> float:
    """Otsu's threshold of an 8-bit grayscale array (histogram only).

    The split with the largest between-class variance is turned into the
    midpoint of the two class means. Returning the level itself would be
    degenerate on a high-contrast page: every level between the ink and the paper
    maximises the variance, and the first one (the ink value) would be picked,
    leaving nothing below ``level - INK_MARGIN``.
    """
    histogram = np.bincount(gray.reshape(-1), minlength=256).astype(np.float64)
    total = histogram.sum()
    if total <= 0:
        return 0.0
    probability = histogram / total
    omega = np.cumsum(probability)
    levels = np.arange(256, dtype=np.float64)
    mu = np.cumsum(probability * levels)
    denominator = omega * (1.0 - omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        between = np.where(
            denominator > 0, (mu[-1] * omega - mu) ** 2 / denominator, 0.0
        )
    index = int(np.argmax(between))
    if omega[index] <= 0 or omega[index] >= 1:
        return float(index)
    background_mean = mu[index] / omega[index]
    foreground_mean = (mu[-1] - mu[index]) / (1.0 - omega[index])
    return float((background_mean + foreground_mean) / 2)


def _downscaled_gray(image: Image.Image, max_side: int) -> np.ndarray:
    gray = image.convert("L")
    longest = max(gray.size)
    if longest > max_side > 0:
        scale = max_side / longest
        size = (max(1, round(gray.width * scale)), max(1, round(gray.height * scale)))
        gray = gray.resize(size, Image.BILINEAR)
    return np.asarray(gray, dtype=np.uint8)


def _paper_mask(gray: np.ndarray, threshold: float) -> np.ndarray:
    """The largest bright region: the page itself, without the desk around it."""
    mask = gray > threshold
    mask = ndimage.binary_fill_holes(mask)
    mask = ndimage.binary_opening(mask, iterations=2)
    labeled, count = ndimage.label(mask)
    if count > 1:
        sizes = ndimage.sum(mask, labeled, index=np.arange(1, count + 1))
        mask = labeled == (1 + int(np.argmax(sizes)))
    return mask


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size < window:
        return values
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(values, kernel, mode="same").astype(np.float32)


def _bands_above(profile: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(profile):
        if value > threshold and start is None:
            start = index
        elif value <= threshold and start is not None:
            if index - start >= MIN_BAND_PIXELS:
                bands.append((start, index))
            start = None
    if start is not None and len(profile) - start >= MIN_BAND_PIXELS:
        bands.append((start, len(profile)))
    return bands
