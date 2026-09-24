"""Line extraction for training crops, using kraken's own inference code.

Training on the line's bounding box leaves the pretrained model with the wrong
picture of a line: a box around a curved line drags in slivers of its
neighbours, and kraken then records the samples as ``bbox`` while the model was
trained and is used on *baseline* strips (the warning
"Neural network has been trained on baselines image information but training set
is bbox").

Here the stored line outline is turned back into a baseline + bounding polygon
and fed to :func:`kraken.lib.segmentation.extract_polygons` — the very function
recognition uses to cut lines out of a page. The crop is therefore dewarped
exactly like the input the model sees at inference time.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ...domain.entities import LineGeometry
from ...domain.errors import CorruptImageError
from ..storage import PilLineCropper, open_oriented_image

logger = logging.getLogger(__name__)

#: one baseline point per this many pixels of line length
BIN_SIZE = 16
MAX_BASELINE_POINTS = 64


def baseline_from_boundary(
    polygon: list[tuple[int, int]] | None,
) -> list[tuple[float, float]] | None:
    """Centre line of a line's outline, left to right.

    The outline of a text line is a thin ribbon around it: points on one side
    have a negative offset from the line's axis, points on the other a positive
    one. Sorting each side along the axis and averaging the two sides therefore
    gives the baseline itself.

    Splitting the outline into "the two halves of the loop" instead is fragile:
    a rectangle (what a merged line looks like) has *two* corners at each end,
    and binning vertices by position jumps between the edges when they are
    sampled unevenly.
    """
    if not polygon or len(polygon) < 4:
        return None
    points = np.asarray(polygon, dtype=float)
    centre = points.mean(axis=0)
    centred = points - centre
    try:
        _, singular, axes = np.linalg.svd(centred, full_matrices=False)
    except np.linalg.LinAlgError:  # pragma: no cover - defensive
        return None
    if singular[0] <= 0:
        return None
    direction = axes[0]
    if direction[0] < 0:  # always left to right
        direction = -direction
    normal = np.array([-direction[1], direction[0]])

    along = centred @ direction
    across = centred @ normal
    span = float(along.max() - along.min())
    if span < BIN_SIZE:
        return None

    upper = points[across <= 0]
    lower = points[across > 0]
    if len(upper) < 2 or len(lower) < 2:
        # everything on one side: not a ribbon, nothing to average
        return None

    count = int(min(MAX_BASELINE_POINTS, max(2, round(span / BIN_SIZE))))
    edge_up = _resample_across_axis(upper, direction, count)
    edge_down = _resample_across_axis(lower, direction, count)
    if edge_up is None or edge_down is None:
        return None
    midline = (edge_up + edge_down) / 2
    return [(float(x), float(y)) for x, y in midline]


def _resample_across_axis(
    points: np.ndarray, direction: np.ndarray, count: int
) -> np.ndarray | None:
    """Resample one edge at ``count`` even positions along the line's axis."""
    along = points @ direction
    order = np.argsort(along)
    along = along[order]
    ordered = points[order]
    if along[-1] - along[0] <= 0:
        return None
    grid = np.linspace(along[0], along[-1], count)
    return np.column_stack(
        [np.interp(grid, along, ordered[:, 0]), np.interp(grid, along, ordered[:, 1])]
    )


class KrakenLineCropper:
    """Cut training crops the way kraken cuts lines at recognition time."""

    def __init__(self, fallback: PilLineCropper | None = None):
        self.fallback = fallback or PilLineCropper()
        self._warned = False

    # ------------------------------------------------------------------

    def crop_line(
        self, page_image_path: str, geometry: LineGeometry, output_path: str
    ) -> str:
        polygon = geometry.polygon
        if not polygon or len(polygon) < 4:
            return self.fallback.crop_line(page_image_path, geometry, output_path)
        try:
            return self._crop_polygon(page_image_path, polygon, output_path)
        except CorruptImageError:
            raise
        except Exception as exc:
            # a bad outline must never fail a training run: the box is a safe,
            # if cruder, crop
            if not self._warned:
                logger.warning(
                    "HTR training: falling back to bbox crops (%s: %s)",
                    type(exc).__name__, exc,
                )
                self._warned = True
            return self.fallback.crop_line(page_image_path, geometry, output_path)

    # ------------------------------------------------------------------

    def _crop_polygon(self, page_image_path: str, polygon, output_path: str) -> str:
        from kraken.containers import BaselineLine, Segmentation
        from kraken.lib.segmentation import extract_polygons

        baseline = baseline_from_boundary(polygon)
        if baseline is None:
            raise ValueError("the line outline has no usable centre line")

        with open_oriented_image(page_image_path) as image:
            width, height = image.size
            boundary = _clamp(polygon, width, height)
            line = BaselineLine(
                id="line",
                baseline=[(int(round(x)), int(round(y))) for x, y in _clamp(baseline, width, height)],
                boundary=boundary,
            )
            segmentation = Segmentation(
                type="baselines",
                imagename=str(page_image_path),
                text_direction="horizontal-lr",
                script_detection=False,
                lines=[line],
            )
            try:
                crop, _ = next(extract_polygons(image, segmentation))
            except StopIteration as exc:  # pragma: no cover - defensive
                raise ValueError("kraken extracted no line image") from exc

            target = Path(output_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            crop.save(target)
        return output_path


def _clamp(points, width: int, height: int) -> list[tuple[int, int]]:
    """Keep every point inside the image: kraken rejects out-of-bounds shapes."""
    return [
        (
            max(0, min(int(round(float(x))), width - 1)),
            max(0, min(int(round(float(y))), height - 1)),
        )
        for x, y in points
    ]
