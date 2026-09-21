"""Page lifecycle: upload -> recognition -> editing -> confirmation.

Prediction and ground truth are strictly separated: predicted_text is never
modified, corrections live in corrected_text and are never auto-fixed.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Protocol

from ..domain.entities import PageStatus, PageView, RecognitionResult
from ..domain.errors import (
    InvalidTranscriptionError,
    NotFoundError,
    PageStateError,
    RecognitionError,
)
from ..domain.interfaces import HTRRecognizer, ModelRepository, PageRepository
from .metrics import MetricsEvaluator

logger = logging.getLogger(__name__)


class PageImageStore(Protocol):
    def save_page_image(self, user_id: int, author_id: int, filename: str, content: bytes) -> str: ...

    def probe_image(self, content: bytes) -> tuple[int, int]:
        """Return (width, height); raises CorruptImageError."""
        ...


class HandwritingPageService:
    def __init__(
        self,
        page_repository: PageRepository,
        model_repository: ModelRepository,
        image_store: PageImageStore,
        metrics_evaluator: MetricsEvaluator | None = None,
        recognizer: HTRRecognizer | None = None,
    ):
        self.page_repository = page_repository
        self.model_repository = model_repository
        self.image_store = image_store
        self.metrics_evaluator = metrics_evaluator or MetricsEvaluator()
        self.recognizer = recognizer

    # ------------------------------------------------------------------

    def upload_page(
        self, user_id: int, author_id: int, filename: str, content: bytes
    ) -> PageView:
        width, height = self.image_store.probe_image(content)
        file_path = self.image_store.save_page_image(user_id, author_id, filename, content)
        page = self.page_repository.create_page(
            user_id=user_id,
            author_id=author_id,
            file_path=file_path,
            width=width,
            height=height,
        )
        logger.info(
            "HTR page uploaded: user_id=%s author_id=%s page_id=%s",
            user_id, author_id, page.id,
        )
        return page

    def recognize_page(self, page_id: int) -> PageView:
        """Run the configured recognizer over the stored page image.

        The active model of the author is used when one exists, otherwise the
        configured default recognition model. The result is stored as the
        prediction; ground truth stays untouched.
        """
        page = self._get_page(page_id)
        if page.status not in (PageStatus.UPLOADED, PageStatus.RECOGNIZED):
            raise PageStateError(
                f"Page {page_id} is {page.status}; recognition is only possible "
                "before the page is edited or confirmed"
            )
        if self.recognizer is None:
            raise RecognitionError("No HTR recognizer is configured")
        active = self.model_repository.get_active_model(page.author_id)
        model_path = (
            active.file_path if active is not None
            else self.model_repository.get_default_model().path
        )
        if not model_path:
            raise RecognitionError("No recognition model is configured")
        result = self.recognizer.recognize(page.file_path, model_path)
        logger.info(
            "HTR page recognized: page_id=%s model=%s lines=%s",
            page_id, model_path, len(result.lines),
        )
        return self.apply_recognition(page_id, result)

    def apply_recognition(self, page_id: int, result: RecognitionResult) -> PageView:
        """Store a recognition result (from a recognizer or external import)."""
        page = self._get_page(page_id)
        if page.status not in (PageStatus.UPLOADED, PageStatus.RECOGNIZED):
            raise PageStateError(
                f"Recognition result cannot be applied to page in status {page.status}"
            )
        active = self.model_repository.get_active_model(page.author_id)
        model_version_id = active.id if active else None
        page = self.page_repository.save_recognition(page_id, result, model_version_id)
        logger.info(
            "HTR recognition stored: page_id=%s lines=%s model_version_id=%s",
            page_id, len(result.lines), model_version_id,
        )
        return page

    # ------------------------------------------------------------------

    def update_word(
        self, page_id: int, line_id: int, word_id: int, corrected_text: str
    ) -> PageView:
        page = self._get_editable_page(page_id)
        line = self._find_line(page, line_id)
        word = next((w for w in line.words if w.id == word_id), None)
        if word is None:
            raise NotFoundError(f"Word {word_id} not found in line {line_id}")

        word.corrected_text = corrected_text
        # line transcription is canonical: rebuild it from effective word texts
        parts = [w.effective_text for w in sorted(line.words, key=lambda w: w.order)]
        line_text = " ".join(p for p in parts if p)
        # split/merge (spaces or deletion) breaks word-bbox alignment
        words_stale = line.words_stale or (" " in corrected_text.strip()) or corrected_text == ""
        return self.page_repository.apply_word_update(
            page_id, line_id, word_id, corrected_text, line_text, words_stale
        )

    def update_line(self, page_id: int, line_id: int, corrected_text: str) -> PageView:
        page = self._get_editable_page(page_id)
        self._find_line(page, line_id)
        return self.page_repository.apply_line_update(page_id, line_id, corrected_text)

    # ------------------------------------------------------------------

    def confirm_page(self, page_id: int) -> PageView:
        page = self._get_editable_page(page_id)
        for line in page.lines:
            if not line.has_valid_transcription:
                raise InvalidTranscriptionError(
                    f"Line {line.id} has an empty transcription that was not explicitly "
                    "confirmed by the user"
                )
        # Prediction quality on this page vs. the user's ground truth (section 19).
        pairs = [(line.effective_text or "", line.predicted_text or "") for line in page.lines]
        metrics = self.metrics_evaluator.evaluate_pairs(pairs) if pairs else None
        page = self.page_repository.confirm_page(
            page_id,
            confirmed_at=datetime.now(timezone.utc),
            prediction_cer=metrics["cer"] if metrics else None,
            prediction_wer=metrics["wer"] if metrics else None,
        )
        logger.info(
            "HTR page confirmed: page_id=%s author_id=%s prediction_cer=%s prediction_wer=%s",
            page_id, page.author_id, page.prediction_cer, page.prediction_wer,
        )
        return page

    # ------------------------------------------------------------------

    def _get_page(self, page_id: int) -> PageView:
        page = self.page_repository.get_page(page_id)
        if page is None:
            raise NotFoundError(f"Page {page_id} not found")
        return page

    def _get_editable_page(self, page_id: int) -> PageView:
        page = self._get_page(page_id)
        if page.status not in (PageStatus.RECOGNIZED, PageStatus.EDITING):
            raise PageStateError(
                f"Page {page_id} is {page.status}; expected RECOGNIZED or EDITING"
            )
        return page

    @staticmethod
    def _find_line(page: PageView, line_id: int):
        line = next((l for l in page.lines if l.id == line_id), None)
        if line is None:
            raise NotFoundError(f"Line {line_id} not found on page {page.id}")
        return line
