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

@dataclass
class RecognizedWord:
    id: str
    bbox: BoundingBox
    text: str
    confidence: float


@dataclass
class RecognizedLine:
    id: str
    bbox: BoundingBox
    words: list[RecognizedWord]
    text: str


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

    @property
    def effective_text(self) -> str | None:
        return self.corrected_text if self.corrected_text is not None else self.predicted_text


@dataclass
class LineView:
    id: int
    order: int
    bbox: BoundingBox
    predicted_text: str | None
    corrected_text: str | None = None
    words: list[WordView] = field(default_factory=list)
    # True when word bboxes no longer match the line transcription tokenization
    words_stale: bool = False

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
