"""Unit tests of baseline-aligned training crops.

The crops must look like what the recognizer feeds the model — a dewarped strip
along the line's baseline — instead of a box around the line. These tests pin
down the geometry derivation and the fallback behaviour.
"""
from __future__ import annotations

import json

import pytest

from app.htr.domain.entities import BoundingBox, LineGeometry
from app.htr.infrastructure.kraken.lines import KrakenLineCropper, baseline_from_boundary
from app.htr.infrastructure.storage import PilLineCropper

pytest.importorskip("PIL")

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"


# ---------------------------------------------------------------------------
# centre line of an outline
# ---------------------------------------------------------------------------


def ribbon(top, bottom):
    """A closed outline: along one edge, back along the other."""
    return list(top) + list(reversed(bottom)) + [top[0]]


def test_baseline_is_the_middle_of_a_straight_ribbon():
    outline = ribbon([(0, 0), (100, 0)], [(0, 20), (100, 20)])
    baseline = baseline_from_boundary(outline)
    assert baseline is not None
    ys = [y for _, y in baseline]
    xs = [x for x, _ in baseline]
    assert min(xs) == pytest.approx(0, abs=1)
    assert max(xs) == pytest.approx(100, abs=1)
    assert min(ys) == pytest.approx(10, abs=1)  # halfway between the edges
    assert max(ys) == pytest.approx(10, abs=1)


def test_baseline_follows_a_curved_ribbon():
    top = [(x, (x - 50) ** 2 / 100) for x in range(0, 101, 5)]
    bottom = [(x, y + 20) for x, y in top]
    baseline = baseline_from_boundary(ribbon(top, bottom))
    assert baseline is not None
    ys = [y for _, y in baseline]
    # the middle sags with the line instead of staying on the chord
    assert max(ys) - min(ys) > 5


def test_baseline_ignores_uneven_edge_sampling():
    """Regression: binning vertices used to jump between the two edges."""
    top = [(x, 0) for x in range(0, 101, 25)]          # few points on top
    bottom = [(x, 20) for x in range(0, 101, 2)]       # many on the bottom
    baseline = baseline_from_boundary(ribbon(top, bottom))
    assert baseline is not None
    assert all(y == pytest.approx(10, abs=2) for _, y in baseline)


def test_baseline_runs_left_to_right_even_if_the_outline_is_reversed():
    forward = ribbon([(0, 0), (100, 0)], [(0, 20), (100, 20)])
    backward = list(reversed(forward))
    for outline in (forward, backward):
        baseline = baseline_from_boundary(outline)
        assert baseline is not None
        assert baseline[0][0] < baseline[-1][0]


def test_baseline_rejects_shapes_it_cannot_interpret():
    assert baseline_from_boundary(None) is None
    assert baseline_from_boundary([(0, 0), (1, 1), (2, 2)]) is None
    assert baseline_from_boundary([(0, 0), (1, 0), (2, 0), (3, 0)]) is None  # no span across


# ---------------------------------------------------------------------------
# the cropper
# ---------------------------------------------------------------------------


def _page(tmp_path, width=200, height=120):
    from PIL import Image, ImageDraw

    image = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(image)
    draw.line((10, 60, 190, 60), fill=0, width=3)
    path = tmp_path / "page.png"
    image.save(path)
    return path


def test_cropper_uses_the_outline_rather_than_the_box(tmp_path):
    page = _page(tmp_path)
    # a box that covers everything, an outline that covers only the strip
    geometry = LineGeometry(
        bbox=BoundingBox(0, 0, 200, 120),
        polygon=ribbon([(10, 50), (190, 50)], [(10, 70), (190, 70)]),
    )
    output = tmp_path / "line.png"

    KrakenLineCropper().crop_line(str(page), geometry, str(output))

    from PIL import Image

    with Image.open(output) as crop:
        # the strip is the line's own height, not the 120 px of the box
        assert crop.height < 60
        assert crop.width > crop.height


def test_cropper_falls_back_to_the_box_without_an_outline(tmp_path):
    page = _page(tmp_path)
    geometry = LineGeometry(bbox=BoundingBox(0, 0, 50, 40), polygon=None)
    output = tmp_path / "line.png"

    KrakenLineCropper().crop_line(str(page), geometry, str(output))

    from PIL import Image

    with Image.open(output) as crop:
        assert crop.size == (50, 40)


def test_cropper_falls_back_when_the_outline_is_unusable(tmp_path):
    """A bad outline must never fail a training run."""
    page = _page(tmp_path)
    geometry = LineGeometry(
        bbox=BoundingBox(0, 0, 40, 30), polygon=[(0, 0), (0, 0), (0, 0), (0, 0)]
    )
    output = tmp_path / "line.png"

    KrakenLineCropper().crop_line(str(page), geometry, str(output))

    from PIL import Image

    with Image.open(output) as crop:
        assert crop.size == (40, 30)  # the box crop


def test_cropper_keeps_out_of_bounds_outlines_inside_the_image(tmp_path):
    """kraken rejects shapes extending beyond the image; they are clamped."""
    page = _page(tmp_path)
    geometry = LineGeometry(
        bbox=BoundingBox(-20, -20, 300, 200),
        polygon=ribbon([(-10, 50), (210, 50)], [(-10, 70), (210, 70)]),
    )
    output = tmp_path / "line.png"

    KrakenLineCropper().crop_line(str(page), geometry, str(output))

    from PIL import Image

    with Image.open(output) as crop:
        assert crop.width > 0 and crop.height > 0
