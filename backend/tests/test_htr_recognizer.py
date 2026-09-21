"""Unit tests of the Kraken recognizer adapter.

The adapter is the only place that talks to kraken; everything testable without
the optional backend (word reconstruction, geometry, device mapping, error
handling) is covered here.
"""
from __future__ import annotations

import sys

import pytest

from app.htr.domain.entities import BoundingBox
from app.htr.domain.errors import RecognitionError
from app.htr.infrastructure.kraken.recognizer import (
    KrakenRecognizer,
    lightning_device,
    line_bbox,
    polygon_bbox,
    word_spans,
    words_from_record,
)

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402


class FakeRecord:
    """Stand-in for kraken.containers.BBoxOCRRecord (one 4-point cut per char)."""

    def __init__(self, text, boxes, confidences=None, bbox=(0, 0, 200, 20)):
        self.prediction = text
        self.cuts = boxes
        self.confidences = confidences or [0.9] * len(boxes)
        self.bbox = bbox

    def __getitem__(self, key):
        if not isinstance(key, slice):
            raise TypeError("only slicing is used by the adapter")
        indices = range(*key.indices(len(self.prediction)))
        chars = [self.prediction[i] for i in indices]
        boxes = [self.cuts[i] for i in indices]
        xs = [point[0] for box in boxes for point in box]
        ys = [point[1] for box in boxes for point in box]
        cut = ((min(xs), min(ys)), (max(xs), min(ys)), (max(xs), max(ys)), (min(xs), max(ys)))
        confidence = sum(self.confidences[i] for i in indices) / len(indices)
        return "".join(chars), cut, confidence


def char_cuts(text, char_width=10, height=20):
    """One square cut per character, laid out left to right."""
    return [
        (
            (i * char_width, 0),
            ((i + 1) * char_width, 0),
            ((i + 1) * char_width, height),
            (i * char_width, height),
        )
        for i in range(len(text))
    ]


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------


def test_word_spans_splits_on_whitespace():
    assert word_spans("Уж очень дед") == [(0, 2), (3, 8), (9, 12)]
    assert word_spans("  leading and trailing  ") == [(2, 9), (10, 13), (14, 22)]
    assert word_spans("") == []
    assert word_spans("   ") == []


def test_polygon_bbox_from_polygon_and_from_baseline_cut():
    assert polygon_bbox(((5, 7), (15, 7), (15, 9), (5, 9))) == BoundingBox(5, 7, 15, 9)
    # baseline-style cuts are (x_start, x_end) pairs; vertical extent is unknown
    assert polygon_bbox((3, 9)) == BoundingBox(3, 0, 9, 0)
    assert polygon_bbox(None) is None
    assert polygon_bbox(()) is None


def test_words_from_record_rebuilds_words_with_geometry():
    text = "Уж очень"
    record = FakeRecord(
        text,
        char_cuts(text),
        confidences=[0.9] * len(text),
    )
    words = words_from_record(record, "3")

    assert [w.text for w in words] == ["Уж", "очень"]
    assert [w.id for w in words] == ["3:0", "3:1"]
    assert words[0].bbox == BoundingBox(0, 0, 20, 20)
    assert words[1].bbox == BoundingBox(30, 0, 80, 20)
    assert words[0].confidence == pytest.approx(0.9)


def test_words_from_record_averages_character_confidence():
    text = "да"
    record = FakeRecord(text, char_cuts(text), confidences=[0.8, 0.6])
    (word,) = words_from_record(record, "0")
    assert word.confidence == pytest.approx(0.7)


def test_words_from_record_skips_unaligned_and_empty_text():
    # multi-codepoint labels: fewer cuts than code points
    assert words_from_record(FakeRecord("ab", char_cuts("a")), "0") == []
    assert words_from_record(FakeRecord("   ", []), "0") == []


def test_line_bbox_prefers_record_bbox_and_falls_back_to_cuts():
    record = FakeRecord("ab", char_cuts("ab"), bbox=(1, 2, 3, 4))
    assert line_bbox(record) == BoundingBox(1, 2, 3, 4)

    baseline_like = FakeRecord("ab", char_cuts("ab"))
    baseline_like.bbox = None
    assert line_bbox(baseline_like) == BoundingBox(0, 0, 20, 20)

    nothing = FakeRecord("", [], bbox=None)
    assert line_bbox(nothing) is None


@pytest.mark.parametrize(
    ("device", "expected"),
    [
        ("cpu", ("cpu", "auto")),
        ("auto", ("auto", "auto")),
        ("cuda:0", ("gpu", [0])),
        ("cuda:1", ("gpu", [1])),
        ("cuda", ("gpu", "auto")),
    ],
)
def test_lightning_device_mapping(device, expected):
    assert lightning_device(device) == expected


# ---------------------------------------------------------------------------
# error handling (no kraken backend involved)
# ---------------------------------------------------------------------------


def test_recognize_without_model_file_reports_clearly(tmp_path):
    image = tmp_path / "page.png"
    Image.new("L", (10, 10), 255).save(image)

    with pytest.raises(RecognitionError) as excinfo:
        KrakenRecognizer().recognize(str(image), str(tmp_path / "missing.safetensors"))

    assert "does not exist" in str(excinfo.value)


def test_recognize_reports_missing_backend(monkeypatch, tmp_path):
    image = tmp_path / "page.png"
    Image.new("L", (10, 10), 255).save(image)
    model = tmp_path / "model.safetensors"
    model.write_bytes(b"not-a-model")

    # simulate a deployment without the optional kraken backend
    monkeypatch.setitem(sys.modules, "kraken.models", None)

    with pytest.raises(RecognitionError) as excinfo:
        KrakenRecognizer().recognize(str(image), str(model))

    assert "install the optional HTR backend" in str(excinfo.value)


def test_recognize_reports_unreadable_image(tmp_path):
    image = tmp_path / "page.png"
    image.write_bytes(b"definitely not an image")
    model = tmp_path / "model.safetensors"
    model.write_bytes(b"not-a-model")

    with pytest.raises(RecognitionError):
        KrakenRecognizer().recognize(str(image), str(model))


def test_to_result_rejects_records_without_geometry():
    with pytest.raises(RecognitionError):
        KrakenRecognizer._to_result((10, 10), [FakeRecord("", [], bbox=None)])


def test_to_result_uses_oriented_page_size_and_line_order():
    records = [
        FakeRecord("раз", char_cuts("раз"), bbox=None),
        FakeRecord("два", char_cuts("два"), bbox=None),
    ]
    result = KrakenRecognizer._to_result((3472, 4640), records)

    assert (result.page_width, result.page_height) == (3472, 4640)
    assert [line.text for line in result.lines] == ["раз", "два"]
    # no record bbox -> geometry falls back to the character cuts
    assert [line.bbox.x2 for line in result.lines] == [30, 30]


def test_to_result_normalizes_unicode_predictions():
    text = "мои\u0306"  # 'мой' with a combining breve, as kraken decodes it
    record = FakeRecord(text, char_cuts(text), bbox=None)

    result = KrakenRecognizer._to_result((10, 10), [record])

    assert result.lines[0].text == "мой"
    assert [w.text for w in result.lines[0].words] == ["мой"]


def test_module_import_does_not_require_kraken():
    """The adapter must stay importable when the optional backend is absent."""
    assert "app.htr.infrastructure.kraken.recognizer" in sys.modules
