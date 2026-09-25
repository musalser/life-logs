"""Kraken implementation of HTRRecognizer.

All Kraken-specific code is confined to this module; nothing outside the
infrastructure layer may import kraken. Kraken is imported lazily inside the
methods so that the application keeps working (with a clear error) when the
optional HTR backend is not installed.

Pipeline for one page::

    PIL image (EXIF orientation normalized)
      -> line segmentation (neural bLLA, or the classical projection profile)
      -> CTC recognition (kraken.models.load_models + kraken.tasks)

Two deliberate choices:

* Recognition models are loaded from the ``safetensors`` format (e.g. the
  multilingual PP-OCRv6 models, which cover Russian/Cyrillic handwriting).
* Segmentation defaults to the bundled neural bLLA model
  (:meth:`kraken.tasks.SegmentationTaskModel.load_model`). It returns *baseline
  polygons* that follow curved lines instead of axis-aligned boxes, which is
  what handwritten photos of notebooks need; the classical
  :mod:`kraken.pageseg` projection-profile segmenter remains available through
  ``segmentation_engine='classical'`` (and as an automatic fallback). The bLLA
  weights ship as ``kraken/blla.mlmodel`` in the CoreML container format, so
  loading them needs ``coremltools`` — available on Linux/WSL, not on Windows.

Verified against kraken 7.1.1.
"""
from __future__ import annotations

import logging
import math
import threading
import unicodedata
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable, Sequence

from ...domain.interfaces import LexiconChecker
from ...domain.entities import (
    BoundingBox,
    RecognitionResult,
    RecognizedLine,
    RecognizedWord,
)
from ...domain.entities import WordAlternative
from ...domain.errors import CorruptImageError, RecognitionError
from ...domain.text import align_word_alternatives, map_spans_to_reference
from ...domain.interfaces import HTRRecognizer
from ..storage import open_oriented_image

logger = logging.getLogger(__name__)

# Prepared models hold torch weights and a Lightning Fabric; rebuilding them per
# request would dominate the request time, so they are cached per model file and
# inference parameters. Kraken inference is serialized because a prepared model
# keeps per-call state (scaling factors, precision module).
_MODEL_CACHE: "OrderedDict[tuple, Any]" = OrderedDict()
_MODEL_CACHE_LOCK = threading.Lock()
_RECOGNITION_LOCK = threading.Lock()
_MAX_CACHED_MODELS = 4

# Character language models are tens of megabytes; the last few paths are kept.
_LM_CACHE: "OrderedDict[tuple, Any]" = OrderedDict()
_MAX_CACHED_LMS = 2
#: the word-level models are ~1 GB each; one at a time is plenty
_MAX_CACHED_WORD_MODELS = 1
_WORD_MODEL_CACHE: "OrderedDict[str, Any]" = OrderedDict()


def _loaded_word_model(path: str | None):
    """Load (and cache) the word-level KenLM model; None when unusable.

    ``kenlm`` is an optional dependency: the pipeline works without it, only the
    second pass is skipped.
    """
    if not path:
        return None
    cached = _WORD_MODEL_CACHE.get(path)
    if cached is not None:
        return cached
    try:
        import kenlm
    except ImportError:
        logger.warning(
            "HTR decoding: kenlm is not installed, skipping the second pass (%s)", path
        )
        return None
    if not Path(path).is_file():
        logger.warning("HTR decoding: no word-level model at %s, skipping it", path)
        return None
    try:
        model = kenlm.Model(path)
    except Exception as exc:
        logger.warning("HTR decoding: cannot load %s (%s), skipping it", path, exc)
        return None
    logger.info("HTR decoding: word-level model %s (order %s) loaded", path, model.order)
    _WORD_MODEL_CACHE[path] = model
    while len(_WORD_MODEL_CACHE) > _MAX_CACHED_WORD_MODELS:
        _WORD_MODEL_CACHE.popitem(last=False)
    return model


def _loaded_language_model(path: str | None):
    """Load (and cache) a character language model, or None when unavailable."""
    if not path:
        return None
    target = Path(path)
    try:
        stat = target.stat()
    except OSError:
        logger.warning("HTR decoding: no language model at %s, using greedy decoding", path)
        return None
    key = (str(target.resolve()), stat.st_mtime_ns, stat.st_size)
    with _MODEL_CACHE_LOCK:
        cached = _LM_CACHE.get(key)
        if cached is not None:
            _LM_CACHE.move_to_end(key)
            return cached
    try:
        from ..lm.char_ngram import CharNGram

        model = CharNGram.load(target)
    except Exception as exc:
        logger.warning("HTR decoding: cannot load %s (%s), using greedy decoding", path, exc)
        return None
    with _MODEL_CACHE_LOCK:
        _LM_CACHE[key] = model
        while len(_LM_CACHE) > _MAX_CACHED_LMS:
            _LM_CACHE.popitem(last=False)
    logger.info("HTR decoding: language model %s (order %s) loaded", path, model.order)
    return model

# The bundled bLLA segmentation model is a singleton: loading it is expensive
# and it is stateless, unlike the recognition nets.
_SEGMENTATION_MODEL: Any = None
_SEGMENTATION_MODEL_LOCK = threading.Lock()


def _segmentation_model():
    global _SEGMENTATION_MODEL
    with _SEGMENTATION_MODEL_LOCK:
        if _SEGMENTATION_MODEL is None:
            from kraken.tasks import SegmentationTaskModel

            logger.info("Loading the default bLLA line segmentation model")
            _SEGMENTATION_MODEL = SegmentationTaskModel.load_model()
        return _SEGMENTATION_MODEL


# ---------------------------------------------------------------------------
# Pure helpers (no kraken import, unit-testable)
# ---------------------------------------------------------------------------


def normalize_text(text: str) -> str:
    """NFC-normalize a prediction.

    Kraken decodes exactly what the model's codec contains, which for Cyrillic
    means decomposed sequences (``и`` + U+0306 instead of ``й``). Storing those
    unchanged would make corrections and CER/WER comparisons count combining
    marks as separate characters.
    """
    return unicodedata.normalize("NFC", text)


def word_spans(text: str) -> list[tuple[int, int]]:
    """Character spans of the whitespace separated tokens of ``text``."""
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for index, char in enumerate(text):
        if char.isspace():
            if start is not None:
                spans.append((start, index))
                start = None
        elif start is None:
            start = index
    if start is not None:
        spans.append((start, len(text)))
    return spans


def polygon_bbox(polygon: Sequence) -> BoundingBox | None:
    """Axis-aligned bounding box of a 4-point polygon or (x_start, x_end) cut."""
    if polygon is None:
        return None
    points = list(polygon)
    if not points:
        return None
    if isinstance(points[0], (int, float)):
        # baseline-style cut: the vertical extent comes from the line box
        if len(points) < 2:
            return None
        return BoundingBox(int(points[0]), 0, int(points[1]), 0)
    try:
        xs = [int(round(point[0])) for point in points]
        ys = [int(round(point[1])) for point in points]
    except (TypeError, IndexError):
        return None
    return BoundingBox(min(xs), min(ys), max(xs), max(ys))


def polygon_points(polygon: Sequence) -> list[tuple[int, int]] | None:
    """Normalize a kraken polygon into a list of integer points.

    Returns None for baseline-style ``(x_start, x_end)`` cuts, which carry no
    vertical extent and therefore cannot be rendered as a shape.
    """
    if polygon is None:
        return None
    points = list(polygon)
    if len(points) < 3 or isinstance(points[0], (int, float)):
        return None
    try:
        return [(int(round(point[0])), int(round(point[1]))) for point in points]
    except (TypeError, IndexError):
        return None


def line_polygon(record: Any) -> list[tuple[int, int]] | None:
    """Bounding polygon of the line a record belongs to (baseline models)."""
    boundary = getattr(record, "boundary", None)
    points = polygon_points(boundary) if boundary else None
    if points:
        return points
    baseline = getattr(record, "baseline", None)
    return polygon_points(baseline) if baseline else None


# ---------------------------------------------------------------------------
# Gluing split-off line pieces back together before recognition
# ---------------------------------------------------------------------------

def baseline_ends(line: Any) -> tuple[float, float, float, float] | None:
    """(x_start, y_start, x_end, y_end) of a line's baseline, left to right."""
    baseline = getattr(line, "baseline", None)
    if not baseline or len(baseline) < 2:
        return None
    start, end = baseline[0], baseline[-1]
    x0, y0 = float(start[0]), float(start[1])
    x1, y1 = float(end[0]), float(end[1])
    if x0 > x1:  # right-to-left text: normalize so callers can assume lr
        x0, y0, x1, y1 = x1, y1, x0, y0
    return x0, y0, x1, y1


def line_height(line: Any) -> float | None:
    boundary = getattr(line, "boundary", None)
    if boundary and len(boundary) >= 3:
        ys = [float(point[1]) for point in boundary]
        height = max(ys) - min(ys)
        if height > 0:
            return height
    return None


def unit_direction(points: Sequence, at_end: bool) -> tuple[float, float] | None:
    """Unit vector of the baseline near its start or end (local slope)."""
    if not points or len(points) < 2:
        return None
    a, b = (points[-2], points[-1]) if at_end else (points[0], points[1])
    dx, dy = float(b[0]) - float(a[0]), float(b[1]) - float(a[1])
    norm = math.hypot(dx, dy)
    if norm == 0:
        return None
    return dx / norm, dy / norm


def estimate_pitch(baselines: list[list]) -> float | None:
    """Rough distance between two consecutive text lines.

    Boundary heights are a bad scale here: the outline of a wavy line can be
    taller than the line pitch. The span of all baseline midpoints divided by
    the number of gaps is stable and needs no image dimensions.
    """
    mids = sorted(
        (float(points[0][1]) + float(points[-1][1])) / 2
        for points in baselines
        if len(points) >= 2
    )
    if len(mids) < 4:
        return None
    span = mids[-1] - mids[0]
    if span <= 0:
        return None
    return span / (len(mids) - 1)


def is_continuation(
    first: Sequence | None,
    second: Sequence | None,
    scale: float,
    gap_factor: float = 1.5,
    perp_factor: float = 0.5,
    overlap_factor: float = 0.5,
    angle_factor: float = 0.5,
) -> bool:
    """True when two baselines are pieces of the same physical line.

    The segmenter occasionally detaches the first word of a line into its own
    record. The connecting vector between the two pieces then runs *along* the
    baseline (small perpendicular component). A genuinely following line is
    offset *across* the baseline by about one line pitch, so its perpendicular
    component is large — even on a strongly slanted page, where comparing raw
    ``y`` values alone would wrongly glue lines together.

    ``scale`` is the line pitch (see :func:`estimate_pitch`), not the glyph
    height.
    """
    first_points = list(first or [])
    second_points = list(second or [])
    if len(first_points) < 2 or len(second_points) < 2:
        return False
    if first_points[-1][0] > second_points[0][0]:
        first_points, second_points = second_points, first_points

    direction = unit_direction(first_points, at_end=True)
    if direction is None:
        return False

    ax, ay = float(first_points[-1][0]), float(first_points[-1][1])
    bx, by = float(second_points[0][0]), float(second_points[0][1])
    vx, vy = bx - ax, by - ay
    along = vx * direction[0] + vy * direction[1]
    perpendicular = abs(vx * direction[1] - vy * direction[0])

    if perpendicular > perp_factor * scale:
        return False
    if along > gap_factor * scale or along < -overlap_factor * scale:
        return False

    other = unit_direction(second_points, at_end=False)
    if other is not None and abs(direction[0] * other[1] - direction[1] * other[0]) > angle_factor:
        return False
    return True


def group_collinear_lines(lines: list[Any], scale: float | None = None) -> list[list[int]]:
    """Group indices of segments that form one line (union-find over geometry).

    ``scale`` is the line pitch; it is estimated from the baselines when not
    given (tests pass it explicitly).
    """
    baselines = [list(getattr(line, "baseline", None) or []) for line in lines]
    if not scale:
        scale = estimate_pitch(baselines)
    if not scale:
        return [[index] for index in range(len(lines))]

    parent = list(range(len(lines)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            if is_continuation(baselines[i], baselines[j], scale):
                root_i, root_j = find(i), find(j)
                if root_i != root_j:
                    parent[max(root_i, root_j)] = min(root_i, root_j)

    groups: dict[int, list[int]] = {}
    for index in range(len(lines)):
        groups.setdefault(find(index), []).append(index)
    return list(groups.values())


def merge_collinear_lines(lines: list[Any], scale: float | None = None) -> list[Any]:
    """Glue split-off pieces back together, keeping the reading order.

    This runs *before* recognition so the model sees the whole line; merging
    afterwards would only hide the split. A merged record keeps the position of
    its earliest piece.
    """
    groups = group_collinear_lines(lines, scale)
    if all(len(group) == 1 for group in groups):
        return list(lines)

    from copy import copy

    merged: list[Any] = []
    for group in groups:
        if len(group) == 1:
            merged.append(lines[group[0]])
            continue
        pieces = sorted(
            (lines[index] for index in group),
            key=lambda line: baseline_ends(line)[0],
        )
        baseline = [
            (int(round(point[0])), int(round(point[1])))
            for piece in pieces
            for point in (piece.baseline or [])
        ]
        boundary = merged_boundary(pieces, baseline)
        logger.info(
            "HTR segmentation: merged %s segment(s) into one line (%s..%s)",
            len(pieces), baseline[0][0] if baseline else "?", baseline[-1][0] if baseline else "?",
        )
        merged_line = copy(pieces[0])
        merged_line.baseline = baseline
        merged_line.boundary = boundary
        merged.append(merged_line)
    return merged


def merged_boundary(
    pieces: list[Any], baseline: list[tuple[int, int]]
) -> list[tuple[int, int]] | None:
    """Convex hull of the pieces' outlines, with a quad as a safety net."""
    points = [
        (float(point[0]), float(point[1]))
        for piece in pieces
        for point in (getattr(piece, "boundary", None) or [])
    ]
    if points:
        from shapely.geometry import MultiPoint

        hull = MultiPoint(points).convex_hull
        if hull.geom_type == "Polygon":
            return [(int(round(x)), int(round(y))) for x, y in hull.exterior.coords]
    heights = sorted(h for h in (line_height(piece) for piece in pieces) if h)
    if not heights or not baseline:
        return getattr(pieces[0], "boundary", None)
    half = heights[len(heights) // 2] / 2
    (x0, y0), (x1, y1) = baseline[0], baseline[-1]
    return [
        (int(round(x0)), int(round(y0 - half))),
        (int(round(x1)), int(round(y1 - half))),
        (int(round(x1)), int(round(y1 + half))),
        (int(round(x0)), int(round(y0 + half))),
        (int(round(x0)), int(round(y0 - half))),
    ]


def words_from_record(
    record: Any,
    line_id: str,
    text: str | None = None,
    alternatives: dict[int, list[WordAlternative]] | None = None,
) -> list[RecognizedWord]:
    """Words of a kraken OCR record, derived from its per-character geometry.

    Kraken yields one text plus per-character cuts per line; the word layer used
    by the editor is rebuilt here by slicing that text on whitespace.

    The cut of a single code point is a *position* on the line (the CTC
    alignment puts most characters at ``[d, d]``), not a box. Slicing the record
    over a one-character span therefore yields a zero-width section with the
    full line height — which is what used to draw single-letter words (``я``,
    ``о``, ``у``) as thin vertical slivers. A word is instead cut from the start
    of its first code point to the start of the code point *after* it (or to the
    end of the line for the last word), which is the word's real extent.
    """
    predicted = record.prediction or ""
    # a beam-search text may differ from what kraken decoded; the per-character
    # geometry still describes the greedy path, so it is only reused when the
    # two tokenizations have the same shape (see below)
    text = predicted if text is None else text
    if not text.strip():
        return []
    if len(getattr(record, "cuts", ()) or ()) != len(predicted):
        # multi-codepoint labels: per-character geometry is not aligned
        logger.warning(
            "Skipping word geometry for line %s: %s cuts for %s code points",
            line_id, len(getattr(record, "cuts", ()) or ()), len(predicted),
        )
        return []

    spans = word_spans(text)
    # The decoder's text and kraken's own greedy text can disagree — the beam
    # may merge ("где -то" -> "где-то"), split or replace characters. The cuts
    # describe the *greedy* text, so the display words are mapped onto it by a
    # character alignment instead of dropping the whole line's geometry (which
    # left lines with a correct transcription and no word boxes at all).
    ranges = map_spans_to_reference(text, predicted, spans)
    if text != predicted and len(ranges) != len(spans):  # pragma: no cover - defensive
        logger.warning("Line %s: cannot align the decoded text, keeping greedy", line_id)
        return []

    words: list[RecognizedWord] = []
    for order, (start, end) in enumerate(spans):
        span = ranges[order] if order < len(ranges) else None
        if span is None:
            logger.debug(
                "Line %s: word %s has no counterpart in the greedy text", line_id, order
            )
            continue
        geometry = word_cut(record, span[0], span[1])
        if geometry is None:
            continue
        points, confidence = geometry
        bbox = polygon_bbox(points)
        if bbox is None:
            continue
        words.append(
            RecognizedWord(
                id=f"{line_id}:{order}",
                bbox=bbox,
                text=normalize_text(text[start:end]),
                confidence=confidence,
                polygon=polygon_points(points),
                alternatives=(alternatives or {}).get(order, []),
            )
        )
    if not words:
        # used to be silent: a line could end up with a correct transcription
        # and no word boxes at all, which reads as an interface bug
        logger.warning(
            "Line %s: no word geometry could be built (%s spans, %s cuts)",
            line_id, len(spans), len(getattr(record, "cuts", ()) or ()),
        )
    return words


def word_cut(
    record: Any, start: int, end: int
) -> tuple[list, float | None] | None:
    """Polygon and mean confidence of the code points ``[start, end)``.

    Prefers the exact section between the span's outer cut positions; falls back
    to kraken's own slicing (which degenerates for single code points) when the
    record does not expose them.
    """
    cut = _section_between(record, start, end)
    if cut is None:
        try:
            _, fallback_cut, confidence = record[start:end]
        except Exception:  # pragma: no cover - defensive
            return None
        if not polygon_points(fallback_cut):
            return None
        return list(fallback_cut), float(confidence)
    confidence = _mean_confidence(record, start, end)
    return cut, confidence


#: the outline of a line converges to a point at its very end, so a section
#: ending exactly there is a triangle; the end is pulled back in these steps
#: until the right edge is a real one again
TAIL_STEP = 2.0
TAIL_LIMIT = 40.0


def _section_between(record: Any, start: int, end: int) -> list | None:
    """Line polygon section between the outer cuts of a span."""
    cuts = getattr(record, "_cuts", None)
    baseline = getattr(record, "baseline", None)
    boundary = getattr(record, "boundary", None)
    if not cuts or not baseline or not boundary or len(cuts) != len(record.prediction or ""):
        return None
    try:
        start_offset = float(cuts[start][0])
        if end < len(cuts):
            end_offset = float(cuts[end][0])
            if end_offset <= start_offset:
                return None
            return _line_section(baseline, boundary, start_offset, end_offset)

        # last word of the line: run to the end of the baseline
        length = float(getattr(record, "_bl_length", 0.0) or 0.0)
        if length <= start_offset:
            return None
        best = _line_section(baseline, boundary, start_offset, length)
        if best is None or _has_real_right_edge(best):
            return best
        back = TAIL_STEP
        while back <= TAIL_LIMIT:
            candidate = _line_section(baseline, boundary, start_offset, length - back)
            if candidate is not None and _has_real_right_edge(candidate):
                return candidate
            back += TAIL_STEP
        return best
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Falling back to slice geometry: %s", exc)
        return None


def _line_section(baseline: Sequence, boundary: Sequence, start: float, end: float) -> list | None:
    from kraken.lib.segmentation import compute_polygon_section

    return list(compute_polygon_section(baseline, boundary, start, end))


def _has_real_right_edge(points: Sequence) -> bool:
    """False when the section tapers to a point at the end of the line."""
    if len(points) != 4:
        return True
    right = math.hypot(
        float(points[3][0]) - float(points[2][0]), float(points[3][1]) - float(points[2][1])
    )
    left = math.hypot(
        float(points[1][0]) - float(points[0][0]), float(points[1][1]) - float(points[0][1])
    )
    return right >= 0.5 * left


def _mean_confidence(record: Any, start: int, end: int) -> float | None:
    values: list[float] = []
    for index in range(start, end):
        try:
            values.append(float(record[index][2]))
        except Exception:  # pragma: no cover - defensive
            return None
    return sum(values) / len(values) if values else None


def line_bbox(record: Any) -> BoundingBox | None:
    """Bounding box of the line the record belongs to."""
    bbox = getattr(record, "bbox", None)
    if bbox:
        return BoundingBox(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
    # baseline-segmented records have no bbox: fall back to the cut extent
    cuts = list(getattr(record, "cuts", []) or [])
    if not cuts:
        return None
    boxes = [polygon_bbox(cut) for cut in cuts]
    boxes = [box for box in boxes if box is not None]
    if not boxes:
        return None
    return BoundingBox(
        min(box.x1 for box in boxes),
        min(box.y1 for box in boxes),
        max(box.x2 for box in boxes),
        max(box.y2 for box in boxes),
    )


def lightning_device(device: str) -> tuple[str, Any]:
    """Map a kraken device string onto Lightning's (accelerator, devices)."""
    value = (device or "auto").strip()
    if value == "auto":
        return "auto", "auto"
    if value in ("cpu", "mps"):
        return value, "auto"
    if ":" in value:
        kind, _, index = value.partition(":")
        accelerator = "gpu" if kind == "cuda" else kind
        return accelerator, [int(index)]
    if value in ("cuda", "gpu"):
        return "gpu", "auto"
    return value, "auto"


def gpu_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover - torch missing/broken
        return False


def available_lightning_device(device: str) -> tuple[str, Any]:
    """Like :func:`lightning_device`, but never asks for a GPU that is absent.

    A configured ``cuda:0`` on a machine (or a process) without a usable CUDA
    backend used to abort recognition with *"No supported gpu backend found!"*.
    Falling back to the CPU keeps the feature working; the caller logs it.
    """
    accelerator, devices = lightning_device(device)
    if accelerator in ("gpu", "cuda") and not gpu_available():
        logger.warning(
            "HTR: device %r has no usable CUDA backend in this process; using CPU instead",
            device,
        )
        return "cpu", "auto"
    return accelerator, devices


# ---------------------------------------------------------------------------
# Recognizer
# ---------------------------------------------------------------------------


class KrakenRecognizer(HTRRecognizer):
    def __init__(
        self,
        device: str = "cpu",
        batch_size: int = 8,
        padding: int = 16,
        text_direction: str = "horizontal-lr",
        num_line_workers: int = 0,
        maxcolseps: int = 2,
        no_hlines: bool = True,
        temperature: float = 1.0,
        segmentation_engine: str = "neural",
        merge_lines: bool = True,
        return_logits: bool = False,
        decoder: str = "greedy",
        lm_path: str | None = None,
        beam_config: Any | None = None,
        lexicon_path: str | None = None,
        min_lm_text_chars: int = 0,
        word_lm_path: str | None = None,
        rescore_config: Any | None = None,
        rescore_n: int = 10,
        word_alternatives: int = 5,
    ):
        self.device = device
        self.batch_size = batch_size
        self.padding = padding
        self.text_direction = text_direction
        self.num_line_workers = num_line_workers
        self.maxcolseps = maxcolseps
        self.no_hlines = no_hlines
        self.temperature = temperature
        # 'neural' = bundled bLLA model (baseline polygons that follow curved
        # lines), 'classical' = projection-profile segmenter (straight boxes)
        self.segmentation_engine = segmentation_engine
        # glue a detached first word back onto its line before recognition
        self.merge_lines = merge_lines
        # 'greedy' = kraken's own decoder; 'beam' = prefix beam search with a
        # character language model and an optional lexicon bonus
        self.decoder = (decoder or "greedy").strip().lower()
        self.lm_path = lm_path
        self.beam_config = beam_config
        self.lexicon_path = lexicon_path
        self.min_lm_text_chars = max(0, int(min_lm_text_chars))
        # second pass: a word-level KenLM re-ranks the beam's N best hypotheses.
        # None (or a missing kenlm) simply means single-pass decoding.
        self.word_lm_path = word_lm_path
        self.rescore_config = rescore_config
        self.rescore_n = max(1, int(rescore_n))
        #: how many alternative readings per word to keep (0 = do not keep any).
        #: They come from the same N-best list the second pass uses, so keeping
        #: them costs nothing beyond the search that already runs.
        self.word_alternatives = max(0, int(word_alternatives))
        # the beam decoder needs the probability matrix *before* kraken's own
        # greedy decoder, so logits are requested whenever it is active
        self.return_logits = return_logits or self.decoder == "beam"

    # ------------------------------------------------------------------

    def recognize(
        self,
        image_path: str,
        model_path: str,
        author_id: int | None = None,
        word_checker: LexiconChecker | None = None,
    ) -> RecognitionResult:
        model_file = Path(model_path)
        if not model_file.is_file():
            raise RecognitionError(
                f"Recognition model {model_path} does not exist; fetch it with "
                "`python scripts/download_htr_model.py` or point "
                "HTR_DEFAULT_MODEL_PATH at an existing model"
            )

        try:
            image = open_oriented_image(image_path)
        except CorruptImageError as exc:
            raise RecognitionError(str(exc)) from exc

        try:
            with _RECOGNITION_LOCK:
                model, config = self._prepared_model(model_file)
                decoder = self._build_decoder(model, author_id, word_checker)
                rescorer = (
                    self._build_rescorer(word_checker)
                    if decoder is not None
                    else None
                )
                segmentation = self._merge_split_lines(self._segment(image))
                if not segmentation.lines:
                    raise RecognitionError(
                        "No text lines were found on the page; check the scan "
                        "orientation and quality"
                    )
                records = list(model.predict(image, segmentation))
            result = self._to_result(
                image.size,
                records,
                decoder,
                rescorer,
                self.rescore_n,
                self.word_alternatives,
            )
        except RecognitionError:
            raise
        except Exception as exc:
            raise RecognitionError(f"Kraken recognition failed: {exc}") from exc
        finally:
            image.close()

        logger.info(
            "HTR recognition finished: image=%s model=%s lines=%s",
            image_path, model_path, len(result.lines),
        )
        return result

    # ------------------------------------------------------------------

    def _build_decoder(
        self,
        model: Any,
        author_id: int | None,
        word_checker: LexiconChecker | None = None,
    ):
        """Beam-search decoder with the language model of this author, or None.

        ``word_checker`` is the vocabulary of this author as the application
        layer sees it (general dictionary *plus* their own confirmed words and
        knowledge terms); the word bonus uses it to prefer words the user has
        already used. Without one the general dictionary alone is used, and
        without that the beam simply runs with no lexicon bonus at all.
        """
        if self.decoder != "beam":
            logger.info("HTR decoding: greedy CTC (htr_decoder=%s)", self.decoder)
            return None
        lm = _loaded_language_model(self._lm_for(author_id))
        if lm is None:
            return None
        min_chars = int(self.min_lm_text_chars)
        running_chars = int(lm.meta.get("running_text_chars", 0) or 0)
        if lm.char_id(" ") == 0 or running_chars < min_chars:
            # A model built from bare word forms knows letters but not what
            # follows a space: measured, it turned WER 0.30 into 0.95, and with
            # a few hundred characters of real text it is still a coin flip.
            # Below the threshold greedy decoding is the safer choice.
            logger.warning(
                "HTR decoding: language model %s has too little running text "
                "(%s characters, need %s), keeping greedy decoding",
                self.lm_path, running_chars, min_chars,
            )
            return None
        from .beam import BeamSearchConfig, PrefixBeamSearch

        config = self.beam_config or BeamSearchConfig()

        logger.info(
            "HTR decoding: prefix beam search over the CTC matrix "
            "(lm=%s, width=%s, top_k=%s, alpha=%s, beta=%s, word_bonus=%s, "
            "author_vocabulary=%s)",
            self._lm_for(author_id), config.beam_width, config.top_k,
            config.alpha, config.beta, config.word_bonus,
            word_checker is not None,
        )

        return PrefixBeamSearch(
            codec=model.codec, lm=lm, config=config,
            known_word=self._known_word_fn(word_checker),
        )

    def _known_word_fn(self, word_checker: LexiconChecker | None):
        """Vocabulary callable shared by the beam bonus and the rescoring guard."""
        checker = word_checker or self._file_lexicon_checker()
        if checker is None:
            return None

        def known(word: str) -> bool:
            # the codec emits decomposed text, the dictionary holds composed
            # words, so normalization happens exactly here
            return checker.is_known(unicodedata.normalize("NFC", word))

        return known

    def _build_rescorer(self, word_checker: LexiconChecker | None):
        """Second-pass re-ranker over the beam's N best, or None."""
        if not self.word_lm_path:
            return None
        model = _loaded_word_model(self.word_lm_path)
        if model is None:
            return None
        from ..lm.word_rescorer import KenLMSentenceScorer, RescoreConfig, WordRescorer

        config = self.rescore_config or RescoreConfig()
        logger.info(
            "HTR decoding: second pass with the word-level model %s "
            "(weight=%s, n=%s, guard=%s)",
            self.word_lm_path, config.weight, self.rescore_n, config.guard,
        )
        return WordRescorer(
            KenLMSentenceScorer(model), config, known_word=self._known_word_fn(word_checker)
        )

    def _file_lexicon_checker(self):
        """Fallback vocabulary: the dictionary file alone, no author words."""
        if not self.lexicon_path:
            return None
        try:
            from ..lexicon import LayeredLexiconChecker, load_file_lexicon

            lexicon = load_file_lexicon(self.lexicon_path)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("HTR decoding: lexicon %s unusable: %s", self.lexicon_path, exc)
            return None
        if not lexicon.is_available:
            return None
        return LayeredLexiconChecker(base=lexicon)

    def _lm_for(self, author_id: int | None) -> str | None:
        """Author-specific language model when one was built, else the general one."""
        if author_id is not None and self.lm_path:
            candidate = Path(self.lm_path).with_name(f"author_{author_id}_char_lm.npz")
            if candidate.is_file():
                return str(candidate)
        return self.lm_path

    @staticmethod
    def _logits_matrix(logits: Any, index: int = 0):
        """The CTC matrix of one record, whatever shape kraken handed over.

        With the neural (baseline) segmenter ``record.logits`` is the CTC tensor
        (classes × time) and everything works. The classical (bbox) segmenter
        does **not** attach a matrix at all: measured on ``BBoxOCRRecord``, its
        ``logits`` is a list of per-character tuples ``(str, int, int, float)``,
        which is why that path has neither beam decoding nor per-word
        alternatives and cleanly falls back to greedy. Taking a tensor for
        granted there used to surface as a confusing "matrix must be 2-D"
        warning; now such a payload is reported as "no CTC matrix".
        """
        if logits is None:
            return None
        import numpy as np

        value = logits
        if isinstance(value, (list, tuple)):
            if index >= len(value):
                return None
            value = value[index]
            # (logits, lengths)-style pair from the classical path
            if isinstance(value, (list, tuple)):
                value = value[0] if value else None
                if value is None:
                    return None
        if hasattr(value, "detach"):
            value = value.detach().cpu()
        try:
            matrix = np.asarray(value, dtype=np.float32)
        except (TypeError, ValueError):
            # the classical path hands over per-character tuples of strings and
            # numbers, which cannot be a probability matrix at all
            return None
        if matrix.ndim != 2 or min(matrix.shape) < 2:
            return None
        return matrix

    @staticmethod
    def _word_alternatives(
        text: str, candidates: list[Any], limit: int
    ) -> dict[int, list[WordAlternative]]:
        """Other readings per word index, from the line's N-best list."""
        if limit <= 0 or len(candidates) < 2 or not text.strip():
            return {}
        chosen_words = [text[start:end] for start, end in word_spans(text)]
        hypotheses = [
            ([piece for piece in candidate.text.split() if piece], candidate.score)
            for candidate in candidates
        ]
        raw = align_word_alternatives(chosen_words, hypotheses, limit)
        return {
            index: [WordAlternative(text=word, score=score) for word, score in items]
            for index, items in raw.items()
        }

    @classmethod
    def _decoded_text(
        cls,
        record: Any,
        decoder,
        greedy: str,
        line_id: str,
        rescorer: Any | None = None,
        rescore_n: int = 10,
    ) -> str:
        """Just the text of a line (kept for callers that need nothing else)."""
        return cls._decoded_line(record, decoder, greedy, line_id, rescorer, rescore_n)[0]

    @staticmethod
    def _decoded_line(
        record: Any,
        decoder,
        greedy: str,
        line_id: str,
        rescorer: Any | None = None,
        rescore_n: int = 10,
        alternatives_n: int = 0,
        record_index: int = 0,
    ) -> tuple[str, list[Any]]:
        """Beam-search text of a line, falling back to kraken's own decoding.

        With a rescorer the beam returns its N best hypotheses and a word-level
        model re-ranks them; the guard inside the rescorer decides whether it is
        allowed to change the winner at all.
        """
        logits = getattr(record, "logits", None)
        matrix = KrakenRecognizer._logits_matrix(logits, record_index)
        if matrix is None:
            logger.warning(
                "Line %s: no CTC matrix on the record, keeping the greedy text", line_id
            )
            return greedy, []
        try:
            # the N-best list is needed by the second pass and by the per-word
            # alternatives, so it is requested once for both
            n = max(rescore_n if rescorer is not None else 0, alternatives_n)
            if n <= 1:
                return normalize_text(decoder.decode(matrix) or greedy), []
            candidates = decoder.decode_nbest(matrix, n=n)
            if not candidates:
                return normalize_text(decoder.decode(matrix) or greedy), []
            if rescorer is None:
                return normalize_text(candidates[0].text or greedy), candidates
            outcome = rescorer.choose(candidates)
            if outcome.changed:
                logger.info("Line %s: %s", line_id, outcome)
            return normalize_text(outcome.text or greedy), candidates
        except Exception as exc:  # pragma: no cover - the decoder must not break recognition
            logger.warning("Line %s: beam search failed (%s), keeping greedy", line_id, exc)
            return greedy, []

    def _segment(self, image):
        if self.segmentation_engine == "neural":
            try:
                return self._neural_segment(image)
            except Exception as exc:
                logger.warning(
                    "Neural line segmentation unavailable (%s); falling back to the "
                    "classical projection-profile segmenter", exc,
                )
        return self._classical_segment(image)

    def _merge_split_lines(self, segmentation):
        """Re-join segments that the segmenter split off the same line."""
        if not self.merge_lines:
            return segmentation
        lines = list(getattr(segmentation, "lines", None) or [])
        if len(lines) < 2 or not all(getattr(line, "baseline", None) for line in lines):
            return segmentation
        merged = merge_collinear_lines(lines)
        if len(merged) == len(lines):
            return segmentation
        from dataclasses import replace

        return replace(segmentation, lines=merged)

    def _neural_segment(self, image):
        """Bundled bLLA model: polygons and baselines that follow curved lines."""
        from kraken.configs import SegmentationInferenceConfig
        from kraken.tasks import SegmentationTaskModel

        model = _segmentation_model()
        accelerator, devices = available_lightning_device(self.device)
        config = SegmentationInferenceConfig(
            accelerator=accelerator,
            device=devices,
            text_direction=self.text_direction,
        )
        return model.predict(image, config)

    def _classical_segment(self, image):
        from kraken.binarization import nlbin
        from kraken.pageseg import segment

        binary = nlbin(image)
        return segment(
            binary,
            text_direction=self.text_direction,
            maxcolseps=self.maxcolseps,
            no_hlines=self.no_hlines,
        )

    def _prepared_model(self, model_file: Path):
        stat = model_file.stat()
        key = (
            str(model_file.resolve()),
            stat.st_mtime_ns,
            stat.st_size,
            self.device,
            self.batch_size,
            self.padding,
            self.num_line_workers,
            self.temperature,
            self.return_logits,
        )
        with _MODEL_CACHE_LOCK:
            cached = _MODEL_CACHE.get(key)
            if cached is not None:
                _MODEL_CACHE.move_to_end(key)
                return cached
            prepared = self._load_model(model_file)
            _MODEL_CACHE[key] = prepared
            while len(_MODEL_CACHE) > _MAX_CACHED_MODELS:
                _MODEL_CACHE.popitem(last=False)
            return prepared

    def _load_model(self, model_file: Path):
        try:
            from kraken.configs import RecognitionInferenceConfig
            from kraken.tasks import RecognitionTaskModel
        except ImportError as exc:
            raise RecognitionError(
                "kraken is not installed; install the optional HTR backend "
                "(see app/htr/README.md) to enable recognition"
            ) from exc

        models = self._deserialize_models(model_file)
        if not models:
            raise RecognitionError(
                f"{model_file} contains no recognition model (was it trained "
                "for segmentation?)"
            )
        task = RecognitionTaskModel(models)
        accelerator, devices = available_lightning_device(self.device)
        config = RecognitionInferenceConfig(
            accelerator=accelerator,
            device=devices,
            batch_size=self.batch_size,
            padding=self.padding,
            num_line_workers=self.num_line_workers,
            temperature=self.temperature,
            return_logits=self.return_logits,
        )
        net = task.net
        net.prepare_for_inference(config)
        return net, config

    @staticmethod
    def _deserialize_models(model_file: Path) -> list:
        """Load models from a kraken model file.

        ``kraken.models.load_models`` walks all registered format loaders and
        only tolerates ``ValueError``, so on this platform the CoreML loader
        aborts the walk with ``ModuleNotFoundError: coremltools`` (no Windows
        wheels exist). safetensors files are therefore handed to their loader
        directly; other formats keep the generic entry point.
        """
        from kraken.models import load_models, load_safetensors

        if model_file.suffix.lower() == ".safetensors":
            return load_safetensors(str(model_file), tasks=["recognition"])
        try:
            return load_models(str(model_file), tasks=["recognition"])
        except ImportError as exc:  # pragma: no cover - platform dependent
            raise RecognitionError(
                f"Cannot read {model_file.name}: it requires the CoreML loader "
                f"(coremltools), which is not installable on this platform ({exc}). "
                "Use a kraken safetensors model."
            ) from exc

    # ------------------------------------------------------------------

    @classmethod
    def _to_result(
        cls,
        page_size: tuple[int, int],
        records: Iterable[Any],
        decoder: "PrefixBeamSearch | None" = None,
        rescorer: Any | None = None,
        rescore_n: int = 10,
        alternatives_n: int = 0,
    ) -> RecognitionResult:
        lines: list[RecognizedLine] = []
        for order, record in enumerate(records):
            line_id = str(order)
            bbox = line_bbox(record)
            if bbox is None:
                logger.warning("Skipping line %s: no bounding box in record", line_id)
                continue
            greedy = normalize_text(record.prediction or "")
            text = greedy
            candidates: list[Any] = []
            if decoder is not None:
                text, candidates = cls._decoded_line(
                    record, decoder, greedy, line_id, rescorer, rescore_n,
                    alternatives_n, order,
                )
            alternatives = cls._word_alternatives(text, candidates, alternatives_n)
            lines.append(
                RecognizedLine(
                    id=line_id,
                    bbox=bbox,
                    text=text,
                    words=words_from_record(
                        record, line_id, text=text, alternatives=alternatives
                    ),
                    polygon=line_polygon(record),
                )
            )
        if not lines:
            raise RecognitionError(
                "Recognition produced no lines with usable geometry"
            )
        return RecognitionResult(
            page_width=int(page_size[0]),
            page_height=int(page_size[1]),
            lines=lines,
        )
