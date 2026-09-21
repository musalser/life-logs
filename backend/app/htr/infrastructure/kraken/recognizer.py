"""Kraken implementation of HTRRecognizer.

All Kraken-specific code is confined to this module; nothing outside the
infrastructure layer may import kraken. Kraken is imported lazily inside the
methods so that the application keeps working (with a clear error) when the
optional HTR backend is not installed.

Pipeline for one page::

    PIL image (EXIF orientation normalized)
      -> binarization (kraken.binarization.nlbin)
      -> line segmentation (kraken.pageseg.segment)
      -> CTC recognition (kraken.models.load_models + kraken.tasks)

Two deliberate choices, both driven by the Windows deployment:

* Recognition models are loaded from the ``safetensors`` format (e.g. the
  multilingual PP-OCRv6 models, which cover Russian/Cyrillic handwriting).
* Segmentation uses the classical projection-profile segmenter
  (:mod:`kraken.pageseg`) instead of the neural bLLA segmenter, whose weights
  ship as ``kraken/blla.mlmodel`` in the CoreML format. Loading CoreML requires
  ``coremltools``, which publishes no Windows wheels, while ``pageseg`` only
  needs numpy/scipy. The same applies to the Kraken ``.mlmodel`` recognition
  format.

Verified against kraken 7.1.1.
"""
from __future__ import annotations

import logging
import threading
import unicodedata
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable, Sequence

from ...domain.entities import (
    BoundingBox,
    RecognitionResult,
    RecognizedLine,
    RecognizedWord,
)
from ...domain.errors import CorruptImageError, RecognitionError
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


def words_from_record(record: Any, line_id: str) -> list[RecognizedWord]:
    """Words of a kraken OCR record, derived from its per-character geometry.

    Kraken yields one text plus per-character cuts/confidences per line; the word
    layer used by the editor and by training is rebuilt here by slicing the
    record on whitespace boundaries.
    """
    text = record.prediction or ""
    if not text.strip():
        return []
    cuts = list(getattr(record, "cuts", []) or [])
    if len(cuts) != len(text):
        # multi-codepoint labels: per-character geometry is not aligned
        logger.warning(
            "Skipping word geometry for line %s: %s cuts for %s code points",
            line_id, len(cuts), len(text),
        )
        return []
    words: list[RecognizedWord] = []
    for order, (start, end) in enumerate(word_spans(text)):
        try:
            token, cut, confidence = record[start:end]
        except Exception:  # pragma: no cover - defensive
            continue
        bbox = polygon_bbox(cut)
        if bbox is None:
            continue
        words.append(
            RecognizedWord(
                id=f"{line_id}:{order}",
                bbox=bbox,
                text=normalize_text(token),
                confidence=float(confidence),
            )
        )
    return words


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
    ):
        self.device = device
        self.batch_size = batch_size
        self.padding = padding
        self.text_direction = text_direction
        self.num_line_workers = num_line_workers
        self.maxcolseps = maxcolseps
        self.no_hlines = no_hlines
        self.temperature = temperature

    # ------------------------------------------------------------------

    def recognize(self, image_path: str, model_path: str) -> RecognitionResult:
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
                segmentation = self._segment(image)
                if not segmentation.lines:
                    raise RecognitionError(
                        "No text lines were found on the page; check the scan "
                        "orientation and quality"
                    )
                records = list(model.predict(image, segmentation))
            result = self._to_result(image.size, records)
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

    def _segment(self, image):
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
        accelerator, devices = lightning_device(self.device)
        config = RecognitionInferenceConfig(
            accelerator=accelerator,
            device=devices,
            batch_size=self.batch_size,
            padding=self.padding,
            num_line_workers=self.num_line_workers,
            temperature=self.temperature,
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

    @staticmethod
    def _to_result(page_size: tuple[int, int], records: Iterable[Any]) -> RecognitionResult:
        lines: list[RecognizedLine] = []
        for order, record in enumerate(records):
            line_id = str(order)
            bbox = line_bbox(record)
            if bbox is None:
                logger.warning("Skipping line %s: no bounding box in record", line_id)
                continue
            lines.append(
                RecognizedLine(
                    id=line_id,
                    bbox=bbox,
                    text=normalize_text(record.prediction or ""),
                    words=words_from_record(record, line_id),
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
