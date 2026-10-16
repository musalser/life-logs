"""The faster pieces of kraken's line vectorization, and the in-kraken merge."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from app.htr.infrastructure.kraken import parallel, vectorize

pytest.importorskip("kraken")


def _blobs():
    from scipy import ndimage
    from skimage.measure import label, regionprops

    rng = np.random.default_rng(0)
    mask = ndimage.binary_opening(
        ndimage.gaussian_filter(rng.random((300, 400)), 5) > 0.5, iterations=2
    )
    return [region for region in regionprops(label(mask)) if region.area > 30]


def test_boundary_tracing_matches_kraken_pixel_for_pixel():
    from kraken.lib.segmentation import boundary_tracing as kraken_tracing

    regions = _blobs()
    assert regions
    for region in regions:
        assert np.array_equal(vectorize.boundary_tracing(region), kraken_tracing(region))


def test_fast_sato_matches_skimage_on_a_float32_heatmap():
    from skimage.filters import sato

    rng = np.random.default_rng(1)
    heatmap = rng.random((120, 160)).astype(np.float32)

    ours = vectorize.fast_sato(heatmap, black_ridges=False, mode="constant")
    reference = sato(heatmap, black_ridges=False, mode="constant")

    assert ours.dtype == reference.dtype
    assert np.allclose(ours, reference, atol=1e-6)


def test_fast_sato_hands_other_inputs_to_skimage():
    from skimage.filters import sato

    image = np.random.default_rng(2).random((40, 50))  # float64
    assert np.array_equal(vectorize.fast_sato(image), sato(image))


def _piece(x0, x1, y, height=20):
    return SimpleNamespace(
        baseline=[(x0, y), (x1, y)],
        boundary=[(x0, y - height), (x1, y - height), (x1, y + 4), (x0, y + 4)],
    )


def test_joined_baseline_runs_left_to_right_and_skips_a_covered_mark():
    word, plus, rest = _piece(10, 60, 100), _piece(200, 210, 98), _piece(80, 400, 102)

    baseline = vectorize.joined_baseline([rest, plus, word])

    xs = [x for x, _ in baseline]
    assert xs == sorted(xs) and len(set(xs)) == len(xs)
    assert baseline[0] == [10, 100] and baseline[-1] == [400, 102]
    assert [200, 98] not in baseline  # the "+" inside "rest" is not a vertex


def _lines(*pieces):
    return [
        ("default", [list(p) for p in piece.baseline], np.asarray(piece.boundary))
        for piece in pieces
    ]


def _no_obstacles(baseline, others):
    return []


def test_split_pieces_are_glued_and_kraken_draws_the_outline(monkeypatch):
    drawn = []

    def fake_polygonize(im_feats, targets, topline=False, raise_on_error=False):
        drawn.extend(targets)
        return [[[0, 0], [1, 0], [1, 1]] for _ in targets]

    monkeypatch.setattr(parallel, "polygonize", fake_polygonize)
    options = vectorize.SegmentationOptions(rows=False)
    lines = _lines(
        _piece(10, 60, 100), _piece(80, 400, 101),       # one line, split in two
        _piece(10, 400, 160), _piece(10, 400, 220), _piece(10, 400, 280),
    )

    merged = vectorize._merge_pieces(lines, None, None, options, _no_obstacles, False, False)

    assert len(merged) == 4
    # exactly one outline was drawn — for the glued line, through both pieces
    assert len(drawn) == 1 and drawn[0][0][0] == [10, 100] and drawn[0][0][-1] == [400, 101]
    assert merged[0][2].tolist() == [[0, 0], [1, 0], [1, 1]]
    assert options.report["glued_pieces"] == 2


def test_a_refused_outline_falls_back_to_the_hull(monkeypatch):
    monkeypatch.setattr(parallel, "polygonize", lambda im, targets, *a, **k: [None for _ in targets])
    options = vectorize.SegmentationOptions(rows=False)
    lines = _lines(
        _piece(10, 60, 100), _piece(80, 400, 101),
        _piece(10, 400, 160), _piece(10, 400, 220), _piece(10, 400, 280),
    )

    merged = vectorize._merge_pieces(lines, None, None, options, _no_obstacles, False, False)

    hull = merged[0][2]
    assert hull[:, 0].min() == 10 and hull[:, 0].max() == 400


def test_nothing_to_glue_keeps_the_lines_untouched(monkeypatch):
    monkeypatch.setattr(parallel, "polygonize", lambda *a, **k: pytest.fail("no outline needed"))
    lines = _lines(_piece(10, 400, 100), _piece(10, 400, 160), _piece(10, 400, 220), _piece(10, 400, 280))

    merged = vectorize._merge_pieces(
        lines, None, None, vectorize.SegmentationOptions(rows=False), _no_obstacles, False, False
    )

    assert merged is lines


def test_install_replaces_kraken_vec_lines():
    import kraken.lib.segmentation as segmentation
    import kraken.lib.vgsl.spred as spred

    assert vectorize.install() is True
    assert spred.vec_lines is vectorize.vec_lines
    assert segmentation.boundary_tracing is vectorize.boundary_tracing
    assert segmentation.filters.sato is vectorize.fast_sato


def _rect_piece(x0, x1, y, height):
    top, bottom = y - int(height * 0.7), y + int(height * 0.3)
    return ("default", [[x0, y], [x1, y]], np.asarray([[x0, top], [x1, top], [x1, bottom], [x0, bottom]]))


def test_a_word_written_under_the_end_of_a_line_stays_its_own_line(monkeypatch):
    """Page 178: «(от другой» ends the first line, «матери» is written under it.

    The pairwise test is not symmetric for pieces that overlap horizontally, so
    the result depends on the order of the pieces; kraken's raw order glued
    «матери» into the line, the reading order (what the page-level repair saw)
    does not. Coordinates are the page's own.
    """
    monkeypatch.setattr(
        parallel, "polygonize", lambda im, targets, *a, **k: [[[0, 0], [1, 0], [1, 1]] for _ in targets]
    )
    lines = [
        _rect_piece(2161, 2277, 99, 80),   # (от другой
        _rect_piece(2092, 2264, 159, 88),  # матери
        _rect_piece(38, 2106, 123, 136),   # the first line
        _rect_piece(5, 2248, 243, 146),
        _rect_piece(17, 2161, 373, 138),
        _rect_piece(20, 2200, 500, 140),
        _rect_piece(20, 2200, 625, 140),
    ]
    options = vectorize.SegmentationOptions(rows=False)

    merged = vectorize._merge_pieces(
        lines, None, None, options, _no_obstacles, False, False, scale=(1.0, 1.0)
    )

    assert len(merged) == 6
    assert options.report["glued_pieces"] == 2
    spans = sorted((line[1][0][0], line[1][-1][0]) for line in merged)
    assert (2092, 2264) in spans          # «матери» on its own
    assert (38, 2277) in spans            # the first line ends with «(от другой»
