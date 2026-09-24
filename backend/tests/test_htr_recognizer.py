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
    available_lightning_device,
    baseline_ends,
    group_collinear_lines,
    is_continuation,
    lightning_device,
    line_bbox,
    line_polygon,
    merge_collinear_lines,
    polygon_bbox,
    polygon_points,
    word_spans,
    words_from_record,
)

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402


class FakeRecord:
    """Stand-in for kraken.containers.BBoxOCRRecord (one 4-point cut per char)."""

    def __init__(self, text, boxes, confidences=None, bbox=(0, 0, 200, 20),
                 boundary=None, baseline=None):
        self.prediction = text
        self.cuts = boxes
        self.confidences = confidences or [0.9] * len(boxes)
        self.bbox = bbox
        self.boundary = boundary
        self.baseline = baseline

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


def test_polygon_points_normalizes_and_rejects_baseline_cuts():
    assert polygon_points([[1, 2], [3, 4], [5, 6]]) == [(1, 2), (3, 4), (5, 6)]
    # baseline-style (x_start, x_end) cuts carry no vertical extent
    assert polygon_points((3, 9)) is None
    assert polygon_points([[1, 2], [3, 4]]) is None
    assert polygon_points(None) is None


def test_line_polygon_prefers_boundary_over_baseline():
    boundary = [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]
    baseline = [[0, 5], [5, 6], [10, 5]]
    record = FakeRecord("ab", char_cuts("ab"), boundary=boundary, baseline=baseline)
    assert line_polygon(record) == [(0, 0), (10, 0), (10, 10), (0, 10), (0, 0)]

    record.boundary = None
    assert line_polygon(record) == [(0, 5), (5, 6), (10, 5)]

    record.baseline = [[0, 5], [10, 5]]  # a two-point baseline is not a shape
    assert line_polygon(record) is None

    record.baseline = None
    assert line_polygon(record) is None


# ---------------------------------------------------------------------------
# gluing a detached first word back onto its line
# ---------------------------------------------------------------------------


class FakeBaselineLine:
    """Geometry-only stand-in for kraken's BaselineLine."""

    def __init__(self, line_id, baseline, height=40, boundary=None):
        self.id = line_id
        self.baseline = baseline
        if boundary is None:
            xs = [point[0] for point in baseline]
            ys = [point[1] for point in baseline]
            boundary = [
                (min(xs), min(ys) - height // 2),
                (max(xs), min(ys) - height // 2),
                (max(xs), max(ys) + height // 2),
                (min(xs), max(ys) + height // 2),
                (min(xs), min(ys) - height // 2),
            ]
        self.boundary = boundary


def test_baseline_ends_normalizes_direction():
    assert baseline_ends(FakeBaselineLine("a", [(100, 50), (10, 52)])) == (10.0, 52.0, 100.0, 50.0)
    assert baseline_ends(FakeBaselineLine("b", [(1, 1)])) is None


def test_detached_first_word_is_glued_back():
    # "сейчас" split off to the left of the line that continues at x=120
    word = FakeBaselineLine("w", [(10, 100), (90, 101)])
    rest = FakeBaselineLine("r", [(120, 102), (900, 108)])
    assert is_continuation(word.baseline, rest.baseline, scale=40)

    merged = merge_collinear_lines([word, rest], scale=40)

    assert len(merged) == 1
    assert merged[0].baseline[0] == (10, 100)
    assert merged[0].baseline[-1] == (900, 108)
    assert len(merged[0].boundary) >= 4


def test_normal_consecutive_lines_are_not_glued():
    first = FakeBaselineLine("1", [(10, 100), (900, 104)])
    second = FakeBaselineLine("2", [(10, 140), (900, 144)])

    assert not is_continuation(first.baseline, second.baseline, scale=40)
    assert group_collinear_lines([first, second]) == [[0], [1]]
    assert len(merge_collinear_lines([first, second])) == 2


def test_slanted_consecutive_lines_are_not_glued():
    # strongly slanted page: raw y at the junction is close, but the offset is
    # perpendicular to the baseline, i.e. one line spacing
    first = FakeBaselineLine("1", [(100, 300), (500, 200), (900, 100)])
    second = FakeBaselineLine("2", [(100, 420), (500, 320), (900, 220)])

    assert not is_continuation(first.baseline, second.baseline, scale=100)
    assert group_collinear_lines([first, second]) == [[0], [1]]


def test_distant_columns_are_not_glued():
    left = FakeBaselineLine("l", [(10, 100), (200, 100)])
    right = FakeBaselineLine("r", [(900, 100), (1100, 100)])
    assert not is_continuation(left.baseline, right.baseline, scale=40)
    assert len(merge_collinear_lines([left, right])) == 2


def test_merge_keeps_reading_order_and_untouched_lines():
    word = FakeBaselineLine("w", [(10, 100), (90, 100)])
    rest = FakeBaselineLine("r", [(120, 101), (900, 104)])
    following = FakeBaselineLine("f", [(10, 140), (900, 144)])

    merged = merge_collinear_lines([word, rest, following], scale=40)

    assert len(merged) == 2
    assert merged[0].baseline[0][0] == 10 and merged[0].baseline[-1][0] == 900
    assert merged[1].id == "f"


def test_words_from_record_keeps_character_cut_polygon():
    text = "Уж"
    record = FakeRecord(text, char_cuts(text), boundary=[[0, 0], [20, 0], [20, 20], [0, 20]])
    (word,) = words_from_record(record, "0")
    assert word.polygon is not None
    assert word.bbox == BoundingBox(0, 0, 20, 20)


def test_to_result_carries_line_polygon():
    text = "да"
    boundary = [[0, 0], [30, 0], [30, 20], [0, 20], [0, 0]]
    record = FakeRecord(text, char_cuts(text), bbox=(0, 0, 30, 20), boundary=boundary)

    result = KrakenRecognizer._to_result((100, 50), [record])

    assert result.lines[0].polygon == [(0, 0), (30, 0), (30, 20), (0, 20), (0, 0)]
    assert result.lines[0].words[0].polygon is not None


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


class FakeBaselineRecord:
    """Models kraken's BaselineOCRRecord: cut *positions*, not boxes.

    kraken aligns most code points at a single position on the line
    (``_cuts[i] = [d, d]``), so slicing a one-code-point span yields a
    zero-width section with the full line height.
    """

    def __init__(self, text, offsets, baseline, boundary, bl_length):
        self.prediction = text
        self._cuts = [[offset, offset] for offset in offsets]
        self._bl_length = bl_length
        self.baseline = baseline
        self.boundary = boundary
        # only the length is used by the adapter's alignment guard
        self.cuts = [(0, 0)] * len(offsets)
        self.confidences = [0.9] * len(offsets)

    def _section(self, low, high):
        from kraken.lib.segmentation import compute_polygon_section

        return list(compute_polygon_section(self.baseline, self.boundary, low, high))

    def __getitem__(self, key):
        if isinstance(key, slice):
            indices = list(range(*key.indices(len(self.prediction))))
            offsets = [value for i in indices for value in self._cuts[i]]
            low = min(offsets if min(offsets) else [0.0])
            high = max(offsets)
            if low == high:  # the degenerate case kraken produces
                high = low
            confidence = sum(self.confidences[i] for i in indices) / len(indices)
            return (
                "".join(self.prediction[i] for i in indices),
                self._section(low, high),
                confidence,
            )
        return (
            self.prediction[key],
            self._section(self._cuts[key][0], self._cuts[key][1]),
            self.confidences[key],
        )


def _baseline_record(text, offsets, length=200.0):
    baseline = [(0.0, 0.0), (length, 0.0)]
    boundary = [
        (0.0, -15.0), (length, -15.0), (length, 15.0), (0.0, 15.0), (0.0, -15.0),
    ]
    return FakeBaselineRecord(text, offsets, baseline, boundary, bl_length=length)


def test_single_letter_words_get_a_real_box():
    """Regression: 'я', 'о', 'у' used to be drawn as zero-width slivers."""
    record = _baseline_record("я о", offsets=[10, 40, 60])

    words = words_from_record(record, "0")

    assert [w.text for w in words] == ["я", "о"]
    # the slice kraken offers for one code point is degenerate ...
    assert record[0:1][1][0][0] == record[0:1][1][1][0]
    # ... while the word now spans to the next code point's position
    assert words[0].bbox.x2 - words[0].bbox.x1 == pytest.approx(30, abs=2)
    assert words[0].bbox.x1 == pytest.approx(10, abs=2)
    # the last word runs to the end of the line
    assert words[1].bbox.x2 == pytest.approx(200, abs=2)


def test_every_word_of_a_line_covers_its_characters():
    text = "уж очень дед"          # 12 code points, one position each
    offsets = [index * 30 for index in range(len(text))]
    record = _baseline_record(text, offsets, length=400.0)

    words = words_from_record(record, "1")

    assert [w.text for w in words] == ["уж", "очень", "дед"]
    widths = [w.bbox.x2 - w.bbox.x1 for w in words]
    assert all(width > 0 for width in widths)
    # 0->60, 90->240 and 270->the end of the line
    assert widths[0] == pytest.approx(60, abs=2)
    assert widths[1] == pytest.approx(150, abs=2)
    assert widths[2] == pytest.approx(130, abs=2)


def test_last_word_does_not_taper_into_a_triangle():
    """The line outline ends in a point; the word box must not follow it."""
    text = "уж дед"
    offsets = [index * 30 for index in range(len(text))]
    baseline = [(0.0, 0.0), (200.0, 0.0)]
    # a ribbon that is full height and converges to a point only in its last
    # 20 px, like the outlines the segmenter returns
    boundary = [
        (0.0, -15.0), (180.0, -15.0), (200.0, 0.0),
        (180.0, 15.0), (0.0, 15.0), (0.0, -15.0),
    ]
    record = FakeBaselineRecord(text, offsets, baseline, boundary, bl_length=200.0)

    # what a section up to the very end looks like: a wedge
    tapered = record._section(float(offsets[3]), 200.0)
    assert tapered[2] == tapered[3]

    last = words_from_record(record, "0")[-1]

    assert len({(round(p[0]), round(p[1])) for p in last.polygon}) == 4
    assert last.polygon[2][0] == last.polygon[3][0]      # a vertical right edge
    assert abs(last.polygon[2][1] - last.polygon[3][1]) > 2


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


def test_available_device_falls_back_to_cpu_without_cuda(monkeypatch):
    from app.htr.infrastructure.kraken import recognizer as module

    monkeypatch.setattr(module, "gpu_available", lambda: False)
    # a configured GPU must not abort recognition on a CUDA-less process
    assert available_lightning_device("cuda:0") == ("cpu", "auto")
    assert available_lightning_device("gpu") == ("cpu", "auto")
    assert available_lightning_device("cpu") == ("cpu", "auto")


def test_available_device_keeps_gpu_when_present(monkeypatch):
    from app.htr.infrastructure.kraken import recognizer as module

    monkeypatch.setattr(module, "gpu_available", lambda: True)
    assert available_lightning_device("cuda:0") == ("gpu", [0])
    assert available_lightning_device("auto") == ("auto", "auto")


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


# ---------------------------------------------------------------------------
# decoder selection: beam only when the language model can be trusted
# ---------------------------------------------------------------------------


class FakeLanguageModel:
    """Minimal stand-in for CharNGram (only what the guard reads)."""

    def __init__(self, running_text_chars, has_space=True):
        self.meta = {"running_text_chars": running_text_chars}
        self._has_space = has_space

    def char_id(self, char):
        return 0 if (char == " " and not self._has_space) else 1


class FakeCodec:
    """Only the label map ``PrefixBeamSearch`` reads."""

    l2c_single = {0: "", 1: "а", 2: " "}


class FakeCodecModel:
    codec = FakeCodec()


def test_beam_decoder_requires_enough_running_text(monkeypatch):
    import importlib

    module = importlib.import_module("app.htr.infrastructure.kraken.recognizer")
    recognizer = KrakenRecognizer(
        decoder="beam", lm_path="lm.npz", min_lm_text_chars=1000
    )

    monkeypatch.setattr(module, "_loaded_language_model", lambda path: FakeLanguageModel(999))
    assert recognizer._build_decoder(FakeCodecModel(), None) is None

    monkeypatch.setattr(module, "_loaded_language_model", lambda path: FakeLanguageModel(1000))
    assert recognizer._build_decoder(FakeCodecModel(), None) is not None

    # a word-form-only model knows letters but not spacing: never beam
    monkeypatch.setattr(
        module,
        "_loaded_language_model",
        lambda path: FakeLanguageModel(5_000_000, has_space=False),
    )
    assert recognizer._build_decoder(FakeCodecModel(), None) is None


class FakeChecker:
    def __init__(self, known):
        self.known = set(known)

    @property
    def is_available(self):
        return True

    def is_known(self, word):
        return word in self.known


def test_the_author_checker_is_used_for_the_word_bonus(monkeypatch):
    """The application layer's vocabulary wins over the dictionary file."""
    import importlib

    module = importlib.import_module("app.htr.infrastructure.kraken.recognizer")
    monkeypatch.setattr(
        module, "_loaded_language_model", lambda path: FakeLanguageModel(5000)
    )
    recognizer = KrakenRecognizer(
        decoder="beam", lm_path="lm.npz", min_lm_text_chars=1000
    )
    checker = FakeChecker({"корова"})

    decoder = recognizer._build_decoder(FakeCodecModel(), None, checker)

    # the codec emits decomposed text; the dictionary holds composed words
    assert decoder.known_word("корова") is True
    assert decoder.known_word("коров") is False


def test_greedy_decoder_never_builds_a_beam_search():
    recognizer = KrakenRecognizer(decoder="greedy", lm_path="lm.npz")
    assert recognizer._build_decoder(FakeCodecModel(), None) is None


def test_author_language_model_wins_over_the_general_one(tmp_path):
    general = tmp_path / "ru_char_lm.npz"
    general.write_bytes(b"")
    recognizer = KrakenRecognizer(decoder="beam", lm_path=str(general))

    # no per-author artifact yet -> the general model
    assert recognizer._lm_for(7) == str(general)

    author = tmp_path / "author_7_char_lm.npz"
    author.write_bytes(b"")
    assert recognizer._lm_for(7) == str(author)
    # another author still gets the general model
    assert recognizer._lm_for(8) == str(general)
