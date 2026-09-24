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
    CorrectionContext,
    LineGeometry,
    ModelRef,
    ModelVersionInfo,
    PageSummary,
    PageView,
    RecognitionResult,
    TrainingConfig,
    TrainingDataset,
    TrainingRunResult,
)

class TextCorrector(Protocol):
    """Proposes a corrected version of one predicted line using an LLM.

    The proposal is never written into the transcription: it is stored as a
    suggestion and only the user moves it into ``corrected_text``.
    Implementations must be best-effort: transport or model failures return
    ``None`` (the raw prediction is kept), they never raise.
    """

    def correct_line(
        self,
        text: str,
        context_lines: list[str],
        context: CorrectionContext,
    ) -> str | None: ...


class VocabularyProvider(Protocol):
    """Proper nouns / domain terms of a user (from the knowledge base)."""

    def vocabulary(self, user_id: int) -> list[str]: ...


class WordList(Protocol):
    """Read-only set of known dictionary forms (a Bloom filter, a file, ...)."""

    @property
    def is_available(self) -> bool:
        """False when the dictionary artifact is missing or unusable."""
        ...

    def __contains__(self, word: str) -> bool: ...


class LexiconChecker(Protocol):
    """Decides whether a recognized word exists in the known vocabulary."""

    @property
    def is_available(self) -> bool: ...

    def is_known(self, word: str) -> bool: ...


class LexiconProvider(Protocol):
    """Vocabulary access for the pages of one author.

    ``checker`` combines the general dictionary with the words this author
    really writes; ``page_transcriptions`` gives the effective line texts per
    page so lists can show an OOV total without loading every line.
    """

    def checker(self, author_id: int) -> LexiconChecker: ...

    def page_transcriptions(self, author_id: int) -> dict[int, list[str]]: ...


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
    def recognize(
        self,
        image_path: str,
        model_path: str,
        author_id: int | None = None,
        word_checker: LexiconChecker | None = None,
    ) -> RecognitionResult:
        """Raises RecognitionError when the backend or the model is unusable.

        ``author_id`` lets the backend pick author-specific decoding material
        (a language model built from that author's confirmed pages).
        ``word_checker`` is the vocabulary the decoder should prefer; the
        application layer passes the author's own checker so decoding can favour
        words the user has already confirmed.
        """
        ...


class LineCropper(Protocol):
    def crop_line(
        self, page_image_path: str, geometry: LineGeometry, output_path: str
    ) -> str:
        """Write the line crop to output_path and return it. Raises CorruptImageError."""
        ...


class PageRepository(Protocol):
    def create_page(
        self, user_id: int, author_id: int, file_path: str, width: int, height: int
    ) -> PageView: ...

    def get_page(self, page_id: int) -> PageView | None: ...

    def delete_page(self, page_id: int) -> None:
        """Remove the page and its lines/words (cascade)."""
        ...

    def list_page_summaries(self, author_id: int) -> list[PageSummary]: ...

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

    def save_line_suggestion(
        self,
        page_id: int,
        line_id: int,
        suggested_text: str,
        suggested_by: str,
    ) -> PageView:
        """Store a model proposal; ``corrected_text`` is left untouched."""
        ...

    def accept_line_suggestions(self, page_id: int, line_ids: list[int]) -> PageView:
        """Move the proposals of these lines into the user's transcription."""
        ...

    def apply_line_change(self, page_id: int, line_id: int, corrected_text: str) -> PageView:
        """Write a partially accepted proposal: text yes, proposal stays.

        The user took some of the proposed words but not all of them, so the
        rest of the proposal must remain visible for review.
        """
        ...

    def clear_line_suggestions(self, page_id: int, line_ids: list[int]) -> PageView:
        """Drop proposals (dismissed by the user)."""
        ...

    def record_suggestion_event(
        self,
        *,
        page_id: int,
        line_id: int,
        user_id: int,
        action: str,
        suggested_text: str | None = None,
        original_text: str | None = None,
        change_before: str | None = None,
        change_after: str | None = None,
        in_lexicon: bool | None = None,
        suggested_by: str | None = None,
    ) -> None:
        """Append one accepted/dismissed proposal to the feedback log."""
        ...

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
