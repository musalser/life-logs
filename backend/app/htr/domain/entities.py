"""Framework-independent domain model of the HTR module.

No ML-framework (Kraken, TrOCR, ...) types may appear here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class PageStatus(str, Enum):
    UPLOADED = "UPLOADED"
    RECOGNIZED = "RECOGNIZED"
    EDITING = "EDITING"
    CONFIRMED = "CONFIRMED"


class ModelVersionStatus(str, Enum):
    TRAINING = "TRAINING"
    READY = "READY"
    ACTIVE = "ACTIVE"
    FAILED = "FAILED"


class TrainingRunStatus(str, Enum):
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class TrainingOutcome(str, Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    # another training run for the same author is already in progress
    BUSY = "BUSY"
    # training finished but the result is not better than the base model, so it
    # was not activated (previous active model stays)
    NO_IMPROVEMENT = "NO_IMPROVEMENT"


class ConfidenceLevel(str, Enum):
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass(frozen=True)
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int


# ---------------------------------------------------------------------------
# Recognition output (produced by a future HTRRecognizer implementation)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LineGeometry:
    """Where a line sits on its page image.

    ``bbox`` is the cheap envelope; ``polygon`` is the (possibly curved)
    outline produced by the segmenter. Training crops follow the outline when
    there is one, because that is what recognition does at inference time.
    """

    bbox: BoundingBox
    polygon: list[tuple[int, int]] | None = None


@dataclass
class RecognizedWord:
    id: str
    bbox: BoundingBox
    text: str
    confidence: float
    # polygon following the (possibly curved) baseline; bbox stays as envelope
    polygon: list[tuple[int, int]] | None = None


@dataclass
class RecognizedLine:
    id: str
    bbox: BoundingBox
    words: list[RecognizedWord]
    text: str
    polygon: list[tuple[int, int]] | None = None


@dataclass
class RecognitionResult:
    page_width: int
    page_height: int
    lines: list[RecognizedLine]


# ---------------------------------------------------------------------------
# Read model of stored pages (prediction vs. ground truth are kept apart)
# ---------------------------------------------------------------------------

@dataclass
class WordView:
    id: int
    order: int
    bbox: BoundingBox
    predicted_text: str | None
    confidence: float | None
    corrected_text: str | None = None
    polygon: list[tuple[int, int]] | None = None
    # vocabulary check (see app.htr.application.lexicon): True when the word is
    # in the dictionary, False when the user should look at it, None when the
    # check could not run (no dictionary installed) or there is nothing to check
    in_lexicon: bool | None = None

    @property
    def effective_text(self) -> str | None:
        return self.corrected_text if self.corrected_text is not None else self.predicted_text


@dataclass(frozen=True)
class SuggestionChange:
    """One word the model proposes to replace, with its dictionary verdict.

    ``in_lexicon`` is False when the new word is in no dictionary and in no
    author vocabulary — exactly the kind of change that must not be applied in
    bulk, because that is where a language model invents things.

    ``start``/``end`` are the character offsets of ``before`` in the line the
    proposal refers to (``start == end`` for an insertion), so a single change
    can be applied on its own — for the user who likes some of the proposed
    words but not all of them.
    """

    before: str
    after: str
    in_lexicon: bool | None = None
    start: int = 0
    end: int = 0


@dataclass
class LineView:
    id: int
    order: int
    bbox: BoundingBox
    predicted_text: str | None
    corrected_text: str | None = None
    # who wrote corrected_text: only 'user' now (accepting a proposal is a
    # human decision); kept for historical rows
    corrected_by: str | None = None
    words: list[WordView] = field(default_factory=list)
    # True when word bboxes no longer match the line transcription tokenization
    words_stale: bool = False
    polygon: list[tuple[int, int]] | None = None
    # model proposal, kept apart from corrected_text so the raw recognition is
    # never overwritten by machine output
    suggested_text: str | None = None
    suggested_by: str | None = None
    suggestion_changes: list[SuggestionChange] = field(default_factory=list)
    # words of effective_text that are missing from the dictionary
    oov_count: int = 0
    # the same misses as text, in reading order (badge and list must agree)
    oov_words: list[str] = field(default_factory=list)

    @property
    def geometry(self) -> LineGeometry:
        return LineGeometry(bbox=self.bbox, polygon=self.polygon)

    @property
    def effective_text(self) -> str | None:
        """Ground truth for training: user correction wins, never auto-fixed."""
        return self.corrected_text if self.corrected_text is not None else self.predicted_text

    @property
    def has_valid_transcription(self) -> bool:
        """Empty text is valid only when the user explicitly confirmed it."""
        if self.corrected_text is not None:
            return True
        return bool(self.predicted_text and self.predicted_text.strip())

    @property
    def has_suggestion(self) -> bool:
        """True when the model proposes something different from what is stored."""
        if not self.suggested_text:
            return False
        return self.suggested_text.strip() != (self.effective_text or "").strip()

    @property
    def suggestion_verified(self) -> bool:
        """Safe to accept in bulk: every change is confirmed by the dictionary.

        A change nobody could verify (no dictionary installed, ``in_lexicon``
        None) blocks the bulk accept, and so does a proposal that only touches
        punctuation — there is nothing to check there either, so the user reads
        it themselves.
        """
        if not self.has_suggestion or not self.suggestion_changes:
            return False
        return all(change.in_lexicon is True for change in self.suggestion_changes)


@dataclass
class PageView:
    id: int
    user_id: int
    author_id: int
    file_path: str
    status: PageStatus
    width: int | None
    height: int | None
    created_at: datetime | None
    confirmed_at: datetime | None = None
    recognition_model_version_id: int | None = None
    prediction_cer: float | None = None
    prediction_wer: float | None = None
    lines: list[LineView] = field(default_factory=list)
    # vocabulary check over the whole page (filled in on read, not stored)
    oov_count: int = 0
    lexicon_available: bool = False


@dataclass
class PageSummary:
    """Lightweight page row for lists (no line/word payload)."""

    id: int
    author_id: int
    status: PageStatus
    file_path: str
    created_at: datetime | None
    confirmed_at: datetime | None
    line_count: int
    prediction_cer: float | None = None
    prediction_wer: float | None = None
    # words missing from the dictionary; None when the check is unavailable
    oov_count: int | None = None
    lexicon_available: bool = False


# ---------------------------------------------------------------------------
# Models & training
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelRef:
    """Reference to a model artifact usable as base for training/recognition."""
    id: str
    path: str


@dataclass
class ModelVersionInfo:
    id: int
    author_id: int
    version: int
    base_model_id: str
    file_path: str
    status: ModelVersionStatus
    dataset_hash: str | None = None
    metrics: dict[str, Any] | None = None
    training_config: dict[str, Any] | None = None
    created_at: datetime | None = None


@dataclass(frozen=True)
class TrainingSample:
    page_id: int
    line_id: int
    image_path: str
    transcription: str
    # outline the crop was cut along; part of the dataset identity
    geometry: LineGeometry | None = None


@dataclass
class TrainingDataset:
    author_id: int
    samples: list[TrainingSample]
    dataset_hash: str


@dataclass
class TrainingConfig:
    """Framework-independent fine-tuning parameters.

    ``backend_options`` carries knobs only a concrete HTR backend understands
    (kraken: resize policy, line height, max line width, ...). Keeping them in
    an opaque mapping stops backend-specific types from leaking into the
    application layer: another backend reads its own keys and ignores the rest.
    """

    device: str = "cuda:0"
    epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 1e-4
    validation_split: float = 0.1
    confidence_warning_threshold: float = 0.90
    confidence_critical_threshold: float = 0.70
    random_seed: int = 42
    min_epochs: int = 0
    backend_options: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "validation_split": self.validation_split,
            "confidence_warning_threshold": self.confidence_warning_threshold,
            "confidence_critical_threshold": self.confidence_critical_threshold,
            "random_seed": self.random_seed,
            "min_epochs": self.min_epochs,
            "backend_options": dict(self.backend_options),
        }


@dataclass
class TrainingRunResult:
    """Unified result returned by a concrete HTRTrainer backend."""
    model_path: str
    training_metrics: dict[str, Any] = field(default_factory=dict)
    validation_metrics: dict[str, Any] | None = None
    # metrics of the untouched base model on the *same* validation split; the
    # application layer refuses to activate a model that does not beat it
    baseline_metrics: dict[str, Any] | None = None
    holdout_used: bool = False
    note: str | None = None


@dataclass
class TrainingResult:
    outcome: TrainingOutcome
    author_id: int
    message: str | None = None
    model_version: ModelVersionInfo | None = None
    metrics: dict[str, Any] | None = None
    dataset_hash: str | None = None
    training_run_id: int | None = None
    # how much confirmed training material exists / is required; lets the UI
    # explain why a confirmation did not trigger a new model version yet
    lines_collected: int | None = None
    lines_required: int | None = None
    words_collected: int | None = None
    words_required: int | None = None


# ---------------------------------------------------------------------------
# LLM-assisted correction of the raw prediction
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConfusionPair:
    """A character confusion learned from the author's own corrections.

    ``recognized`` is what the HTR model produced, ``correct`` what the user
    wrote instead (e.g. ``н`` -> ``п``).
    """

    recognized: str
    correct: str
    count: int


@dataclass
class CorrectionContext:
    """Author-specific material handed to the text corrector.

    Nothing here is model-specific: it is the author's own vocabulary, the
    confusions the recognizer makes on their hand, real correction examples and
    the domain vocabulary extracted from the knowledge base.
    """

    lexicon: list[str] = field(default_factory=list)
    vocabulary: list[str] = field(default_factory=list)
    confusions: list[ConfusionPair] = field(default_factory=list)
    examples: list[tuple[str, str]] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not (self.lexicon or self.vocabulary or self.confusions or self.examples)
