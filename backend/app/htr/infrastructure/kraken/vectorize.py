"""bLLA post-processing inside kraken: faster, and with the lines glued first.

kraken's neural segmenter turns its heatmap into lines in
``kraken.lib.vgsl.spred.vec_lines``: baselines are vectorized from the heatmap,
then every baseline gets its outline (``calculate_polygonal_environment``), then
everything is scaled back to the page. This module replaces that one function
(:func:`install`), for two reasons.

**Speed, with identical output.** On a 30-line page the network forward pass
takes well under a second, while the post-processing took ~30 s on one core:

* ``skimage.filters.sato`` (≈9–17 s): five Hessian scales computed one after
  another, the smallest one with a Gaussian truncated at 100 σ. The scales are
  now computed in parallel threads (scipy's filters release the GIL) and the
  σ = 1 kernel is cut at 8 σ like the others — its tail is below float32
  resolution, measured max |difference| = 0 on two pages;
* ``boundary_tracing`` (≈5 s): a Moore-neighbourhood walk that builds numpy
  arrays at every pixel step. The same walk on Python ints traces the same
  pixels in the same order (verified on every blob of a page) in 0.2 s;
* the outlines (≈13 s): independent per line, now computed in worker processes
  (:mod:`.parallel`).

**The split lines are glued before kraken draws the outlines.** bLLA breaks one
physical line into several baselines at wide gaps between words. The old repair
ran after segmentation and gave the glued line the convex hull of its pieces —
a crude outline (16 points against ~80) that on a curved line swallows the
ascenders of its neighbours. Here the pieces are grouped with the same rules
(:func:`.recognizer.group_collinear_lines`, the ink rows of :mod:`.rows`), their
baselines are joined, and kraken's own polygonizer draws the outline of the
whole line, with every other line as an obstacle. The hull remains only as the
fallback when the polygonizer refuses a joined baseline.

The options of one segmentation call travel in a context variable
(:func:`segmentation_options`), because kraken calls ``vec_lines`` itself, deep
inside ``SegmentationTaskModel.predict``. Without options the function does
exactly what kraken's does, only faster.
"""
from __future__ import annotations

import contextvars
import logging
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from itertools import combinations_with_replacement
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np

from . import parallel

logger = logging.getLogger(__name__)


@dataclass
class SegmentationOptions:
    #: glue the pieces of one physical line before the outlines are drawn
    merge_lines: bool = True
    #: also union the pieces that share one ink row (see ``rows.row_bands``)
    rows: bool = True
    rows_threshold: float = 0.4
    #: the page image itself: the ink rows are measured on it, in page
    #: coordinates, exactly as the page-level repair measured them
    image: Any = None
    #: filled in by :func:`vec_lines`: how many pieces were glued, and whether
    #: the merge ran here at all (the caller then skips its own fallback)
    report: dict = field(default_factory=dict)


_OPTIONS: contextvars.ContextVar[SegmentationOptions | None] = contextvars.ContextVar(
    "htr_segmentation_options", default=None
)


@contextmanager
def segmentation_options(options: SegmentationOptions):
    token = _OPTIONS.set(options)
    try:
        yield options
    finally:
        _OPTIONS.reset(token)


# ---------------------------------------------------------------------------
# Faster drop-ins for two kraken helpers
# ---------------------------------------------------------------------------

_SATO_POOL: ThreadPoolExecutor | None = None
_SATO_POOL_LOCK = threading.Lock()


def _sato_pool() -> ThreadPoolExecutor:
    global _SATO_POOL
    with _SATO_POOL_LOCK:
        if _SATO_POOL is None:
            _SATO_POOL = ThreadPoolExecutor(max_workers=5, thread_name_prefix="htr-sato")
        return _SATO_POOL


def _hessian(image: np.ndarray, sigma: float, mode: str, cval: float) -> list[np.ndarray]:
    """``skimage.feature.hessian_matrix(..., use_gaussian_derivatives=True)``.

    Same two successive first-order Gaussian derivatives as skimage 0.25, except
    for the truncation of σ ≤ 1, which skimage widens to 100 σ against aliasing
    of *float64* input: a float32 heatmap cannot represent that tail.
    """
    import scipy.ndimage as ndi

    scaled = sigma / math.sqrt(2)
    common = dict(sigma=(scaled, scaled), mode=mode, cval=cval, truncate=8)
    orders = ([1, 0], [0, 1])
    gradients = [ndi.gaussian_filter(image, order=order, **common) for order in orders]
    return [
        ndi.gaussian_filter(gradients[first], order=orders[second], **common)
        for first, second in combinations_with_replacement(range(2), 2)
    ]


def fast_sato(image, sigmas=range(1, 10, 2), black_ridges=True, mode="reflect", cval=0):
    """``skimage.filters.sato`` for a 2-D float32 image, scales in parallel."""
    from skimage.feature import hessian_matrix_eigvals
    from skimage.filters import sato

    values = np.asarray(image)
    if values.ndim != 2 or values.dtype != np.float32:
        return sato(image, sigmas=sigmas, black_ridges=black_ridges, mode=mode, cval=cval)
    if not black_ridges:
        values = -values

    def tubeness(sigma: float) -> np.ndarray:
        eigvals = hessian_matrix_eigvals(_hessian(values, sigma, mode, cval))[:-1]
        return sigma**2 * np.prod(np.maximum(eigvals, 0), 0) ** (1 / len(eigvals))

    filtered = np.zeros_like(values)
    for scale in _sato_pool().map(tubeness, list(sigmas)):
        filtered = np.maximum(filtered, scale)
    return filtered


class _FastFilters:
    """``skimage.filters`` as kraken's segmentation module sees it."""

    def __init__(self, module):
        self._module = module

    def __getattr__(self, name):
        return getattr(self._module, name)

    sato = staticmethod(fast_sato)


_MOORE = ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))


def boundary_tracing(region) -> np.ndarray:
    """``kraken.lib.segmentation.boundary_tracing`` on plain integers.

    Same start pixel, same backtrack rule, same neighbourhood order as kraken's
    version (itself copied from deepwings), so the traced boundary is the same
    array — kraken allocates numpy arrays at every pixel step, this walks a flat
    byte mask with integer offsets.
    """
    coords = region.coords
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    height = int(maxs[0] - mins[0] + 3)
    width = int(maxs[1] - mins[1] + 3)
    ys = coords[:, 0] - mins[0] + 1
    xs = coords[:, 1] - mins[1] + 1
    mask = np.zeros(height * width, dtype=np.uint8)
    mask[ys * width + xs] = 1
    filled = mask.tobytes()
    offsets = [dy * width + dx for dy, dx in _MOORE]
    direction_of = {offset: index for index, offset in enumerate(offsets)}

    index = 0
    while True:  # the starting point must not be isolated
        y, x = int(ys[index]), int(xs[index])
        window = mask.reshape(height, width)[y - 1:y + 2, x - 1:x + 2]
        if int(window.sum()) > 1:
            break
        index += 1
    start = y * width + x
    if not filled[start + width] and not filled[start + width - 1]:
        backtrack_start = start + width
    else:
        backtrack_start = start - 1

    current, backtrack = start, backtrack_start
    trail = []
    while True:
        first = direction_of[backtrack - current]
        found = 0
        for step in range(8):
            if filled[current + offsets[(first + step) % 8]]:
                found = step
                break
        trail.append(current)
        backtrack = current + offsets[(first + found - 1) % 8]
        current = current + offsets[(first + found) % 8]
        if current == start and backtrack == backtrack_start:
            break
    flat = np.asarray(trail, dtype=np.int64)
    return np.stack([flat // width, flat % width], axis=1) + [mins[0] - 1, mins[1] - 1]


# ---------------------------------------------------------------------------
# vec_lines
# ---------------------------------------------------------------------------


def vec_lines(
    heatmap,
    cls_map: dict,
    scale,
    text_direction: str = "horizontal-lr",
    regions: list | None = None,
    scal_im: np.ndarray | None = None,
    suppl_obj: list | None = None,
    topline: bool | None = False,
    raise_on_error: bool = False,
    **kwargs,
) -> list[dict[str, Any]]:
    """Drop-in for ``kraken.lib.vgsl.spred.vec_lines`` (kraken 7.1).

    Like kraken's version, ``suppl_obj`` is accepted and ignored: kraken builds
    the obstacles of every line from the other baselines and the regions.
    """
    import shapely.geometry as geom
    from kraken.lib.segmentation import (
        is_in_region,
        scale_polygonal_lines,
        vectorize_lines,
    )
    from scipy.ndimage import gaussian_filter
    from skimage.filters import sobel

    regions = list(regions or [])
    start_sep = cls_map["aux"]["_start_separator"]
    end_sep = cls_map["aux"]["_end_separator"]
    baselines: list[tuple[str, list]] = []
    for bl_type, index in cls_map["baselines"].items():
        baselines.extend(
            (bl_type, line)
            for line in vectorize_lines(
                heatmap[(start_sep, end_sep, index), :, :], text_direction=text_direction[:-3]
            )
        )

    im_feats = gaussian_filter(sobel(scal_im), 0.5)
    region_polygons = [geom.Polygon(region) for region in regions]

    def obstacles(baseline, others: Sequence[Sequence]) -> list:
        line = geom.LineString(baseline)
        found = [list(other) for other in others]
        for region_index, polygon in enumerate(region_polygons):
            if is_in_region(line, polygon):
                found.append(regions[region_index])
        return found

    targets = [
        (line, obstacles(line, [other for _, other in baselines[:i] + baselines[i + 1:]]))
        for i, (_, line) in enumerate(baselines)
    ]
    polygons = parallel.polygonize(im_feats, targets, topline, raise_on_error)
    lines = [
        (bl_type, line, polygon)
        for (bl_type, line), polygon in zip(baselines, polygons)
        if polygon is not None
    ]

    options = _OPTIONS.get()
    if options is not None and options.merge_lines and len(lines) > 1:
        lines = _merge_pieces(
            lines, scal_im, im_feats, options, obstacles, topline, raise_on_error, scale,
            regions=regions, text_direction=text_direction,
        )
        options.report["merged_in_kraken"] = True

    scaled = scale_polygonal_lines([line[1:] for line in lines], scale)
    return [
        {"tags": {"type": [{"type": bl_type}]}, "baseline": baseline, "boundary": boundary}
        for (bl_type, _, _), (baseline, boundary) in zip(lines, scaled)
    ]


def _merge_pieces(
    lines, scal_im, im_feats, options, obstacles, topline, raise_on_error, scale=None,
    regions=None, text_direction: str = "horizontal-lr",
):
    """Glue the pieces of one physical line and give each glued line its outline.

    The *decision* is taken in page coordinates — the pieces scaled exactly as
    kraken scales its output, the ink rows measured on the page image — so it is
    the decision the page-level repair made, point for point. The thresholds are
    relative to the line pitch, but the direction of a baseline end comes from
    its last short segment, and the network's rounding flipped a borderline case
    (a word written under the end of a line, page 178). Joining and drawing the
    outline then happen in the network's coordinates, where kraken works.
    """
    from .recognizer import _union_band_groups, group_collinear_lines, merged_boundary

    pieces = [
        SimpleNamespace(baseline=[tuple(point) for point in line], boundary=[tuple(p) for p in polygon])
        for _, line, polygon in lines
    ]
    decided = _page_pieces(lines, scale) if scale is not None else pieces
    # The pairwise test is not symmetric for pieces that overlap horizontally
    # (which end is compared with which start depends on the argument order),
    # so the union-find result depends on the order of the pieces. The
    # page-level repair saw them in kraken's reading order; so does this one.
    order = _kraken_reading_order(decided, regions, scale, text_direction)
    ordered = [decided[index] for index in order]
    groups = group_collinear_lines(ordered)
    bands = _ink_rows(scal_im, options, scale) if options.rows else None
    if bands:
        groups = _union_band_groups(ordered, groups, bands)
    groups = [[order[index] for index in group] for group in groups]
    if all(len(group) == 1 for group in groups):
        return lines

    joined: list[tuple[list[int], list]] = []
    for group in sorted(groups, key=min):
        if len(group) == 1:
            joined.append((group, list(lines[group[0]][1])))
        else:
            joined.append((group, joined_baseline([pieces[index] for index in group])))

    # outlines only for the glued lines; every other final line is an obstacle
    to_draw = [position for position, (group, _) in enumerate(joined) if len(group) > 1]
    targets = [
        (
            joined[position][1],
            obstacles(
                joined[position][1],
                [baseline for other, (_, baseline) in enumerate(joined) if other != position],
            ),
        )
        for position in to_draw
    ]
    drawn = dict(zip(to_draw, parallel.polygonize(im_feats, targets, topline, raise_on_error)))

    merged = []
    glued = 0
    for position, (group, baseline) in enumerate(joined):
        if len(group) == 1:
            merged.append(lines[group[0]])
            continue
        glued += len(group)
        group_pieces = [pieces[index] for index in group]
        polygon = drawn.get(position)
        if polygon is None:
            logger.info(
                "HTR segmentation: kraken refused the outline of a glued line, "
                "using the hull of its %s pieces", len(group),
            )
            polygon = merged_boundary(group_pieces, [tuple(point) for point in baseline])
        logger.info(
            "HTR segmentation: glued %s piece(s) into one line (%s..%s)",
            len(group), baseline[0][0], baseline[-1][0],
        )
        merged.append((lines[group[0]][0], baseline, np.asarray(polygon)))
    options.report["glued_pieces"] = glued
    return merged


def _kraken_reading_order(pieces, regions, scale, text_direction) -> list[int]:
    """kraken's own reading order of the pieces (identity when unavailable)."""
    try:
        from kraken.lib.segmentation import polygonal_reading_order

        page_regions = [
            SimpleNamespace(boundary=(np.asarray(region) * (scale if scale is not None else 1))
                            .astype("int").tolist())
            for region in (regions or [])
        ]
        order = list(polygonal_reading_order(
            lines=pieces, regions=page_regions, text_direction=text_direction[-2:]
        ))
        if sorted(order) == list(range(len(pieces))):
            return order
    except Exception as exc:  # pragma: no cover - ordering is a refinement only
        logger.debug("HTR segmentation: no kraken reading order for the merge (%s)", exc)
    return list(range(len(pieces)))


def _page_pieces(lines, scale) -> list[SimpleNamespace]:
    """The pieces as kraken hands them to the page (``scale_polygonal_lines``)."""
    from kraken.lib.segmentation import scale_polygonal_lines

    scaled = scale_polygonal_lines([(line, polygon) for _, line, polygon in lines], scale)
    return [
        SimpleNamespace(baseline=[tuple(p) for p in baseline], boundary=[tuple(p) for p in boundary])
        for baseline, boundary in scaled
    ]


def joined_baseline(pieces: Sequence[Any]) -> list[list[int]]:
    """One baseline through the pieces of a line, left to right.

    A piece standing *inside* the span of a longer one (a tiny "+" between two
    words) is left out: its baseline would zigzag through the joined one. The
    result runs strictly left to right, because kraken's polygonizer and line
    extractor walk the baseline as a polyline.
    """
    from .recognizer import baseline_ends

    ordered = sorted(pieces, key=lambda piece: baseline_ends(piece)[0])
    spans = [baseline_ends(piece) for piece in ordered]
    kept = []
    for index, span in enumerate(spans):
        covered = any(
            other != index
            and other_span[0] <= span[0]
            and other_span[2] >= span[2]
            and (other_span[2] - other_span[0]) > (span[2] - span[0])
            for other, other_span in enumerate(spans)
        )
        if not covered:
            kept.append(ordered[index])
    points: list[list[int]] = []
    for piece in kept or ordered:
        line = [(int(round(x)), int(round(y))) for x, y in piece.baseline]
        if line and line[0][0] > line[-1][0]:
            line.reverse()
        for x, y in line:
            if not points or x > points[-1][0]:
                points.append([x, y])
    return points


def _ink_rows(scal_im, options, scale=None) -> list[tuple[int, int]] | None:
    """The page's ink rows, in the coordinates the grouping decides in.

    With the page image (the recognizer passes it) the rows are measured on it,
    in page coordinates — what the page-level repair did. Without it they come
    from the network's copy of the page, scaled up.
    """
    from PIL import Image

    from .rows import row_bands

    try:
        if options.image is not None:
            return row_bands(options.image, threshold_ratio=options.rows_threshold) or None
        if scal_im is None:
            return None
        bands = row_bands(Image.fromarray(np.asarray(scal_im, dtype=np.uint8)),
                          threshold_ratio=options.rows_threshold)
        y_scale = float(np.atleast_1d(scale)[-1]) if scale is not None else 1.0
        return [(top * y_scale, bottom * y_scale) for top, bottom in bands] or None
    except Exception as exc:  # pragma: no cover - the profile must never fail a page
        logger.warning("HTR segmentation: ink rows unavailable (%s: %s)", type(exc).__name__, exc)
        return None


# ---------------------------------------------------------------------------

_INSTALLED = False
_INSTALL_LOCK = threading.Lock()


def install() -> bool:
    """Put the faster pieces into kraken (idempotent). False when impossible."""
    global _INSTALLED
    with _INSTALL_LOCK:
        if _INSTALLED:
            return True
        try:
            import kraken.lib.segmentation as segmentation
            import kraken.lib.vgsl.spred as spred
        except Exception as exc:  # pragma: no cover - kraken missing or reorganized
            logger.warning("HTR: cannot hook kraken's line vectorization (%s)", exc)
            return False
        if not hasattr(spred, "vec_lines") or not hasattr(segmentation, "boundary_tracing"):
            logger.warning(
                "HTR: this kraken has no vec_lines/boundary_tracing to replace; "
                "keeping its own (slower) post-processing"
            )
            return False
        segmentation.filters = _FastFilters(segmentation.filters)
        segmentation.boundary_tracing = boundary_tracing
        spred.vec_lines = vec_lines
        _INSTALLED = True
        logger.info("HTR: kraken line vectorization replaced (parallel, glued lines first)")
        return True
