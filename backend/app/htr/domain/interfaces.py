"""Abstractions the application layer depends on.

Concrete implementations live in app.htr.infrastructure; dependencies point
inward (API -> Application -> Domain <- Infrastructure).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Protocol

from .entities import (
    BoundingBox,
    ModelRef,
    ModelVersionInfo,
    PageView,
    RecognitionResult,
    TrainingConfig,
    TrainingDataset,
    TrainingRunResult,
)


class HTRTrainer(ABC):
    """Fine-tunes a recognition model. Backend-specific code stays behind this."""

    @abstractmethod
    def train(
        self,
        base_model: ModelRef,
        dataset: TrainingDataset,
        output_model_path: str,
        config: TrainingConfig,
    ) -> TrainingRunResult:
        ...


class HTRRecognizer(ABC):
    """Transcribes a page image with a given model.

    Implemented in app.htr.infrastructure.kraken.recognizer; the application
    layer only sees this interface (image in, lines/words out).
    """

    @abstractmethod
    def recognize(self, image_path: str, model_path: str) -> RecognitionResult:
        """Raises RecognitionError when the backend or the model is unusable."""
        ...


class LineCropper(Protocol):
    def crop_line(self, page_image_path: str, bbox: BoundingBox, output_path: str) -> str:
        """Write the line crop to output_path and return it. Raises CorruptImageError."""
        ...


class PageRepository(Protocol):
    def create_page(
        self, user_id: int, author_id: int, file_path: str, width: int, height: int
    ) -> PageView: ...

    def get_page(self, page_id: int) -> PageView | None: ...

    def get_confirmed_pages(self, author_id: int) -> list[PageView]: ...

    def save_recognition(
        self, page_id: int, result: RecognitionResult, model_version_id: int | None
    ) -> PageView: ...

    def apply_word_update(
        self,
        page_id: int,
        line_id: int,
        word_id: int,
        corrected_text: str,
        line_corrected_text: str,
        words_stale: bool,
    ) -> PageView: ...

    def apply_line_update(self, page_id: int, line_id: int, corrected_text: str) -> PageView: ...

    def confirm_page(
        self,
        page_id: int,
        confirmed_at: datetime,
        prediction_cer: float | None,
        prediction_wer: float | None,
    ) -> PageView: ...


class ModelRepository(Protocol):
    def get_default_model(self) -> ModelRef: ...

    def get_active_model(self, author_id: int) -> ModelVersionInfo | None: ...

    def get_active_model_ref(self, author_id: int) -> ModelRef: ...

    def create_version(
        self,
        author_id: int,
        base_model_id: str,
        training_config: dict[str, Any],
        dataset_hash: str,
        environment: dict[str, Any] | None = None,
    ) -> ModelVersionInfo: ...

    def mark_ready(self, model_version_id: int, model_path: str, metrics: dict[str, Any]) -> None: ...

    def mark_failed(self, model_version_id: int, error: str) -> None: ...

    def fail_stale_training(self, author_id: int, reason: str) -> int:
        """Mark versions stuck in TRAINING as FAILED; returns how many."""
        ...

    def get_version(self, model_version_id: int) -> ModelVersionInfo | None: ...

    def activate_model(self, author_id: int, model_version_id: int) -> None: ...

    def clear_active_model(self, author_id: int) -> None:
        """Demote the active version so the default model is used again."""
        ...

    def list_versions(self, author_id: int) -> list[ModelVersionInfo]: ...


class TrainingRunRepository(Protocol):
    def start_run(self, author_id: int, dataset_hash: str) -> int: ...

    def finish_run(
        self,
        run_id: int,
        status: str,
        metrics: dict[str, Any] | None = None,
        error: str | None = None,
        model_version_id: int | None = None,
    ) -> None: ...


class DatasetBuilder(Protocol):
    def build_for_author(self, author_id: int) -> TrainingDataset: ...
