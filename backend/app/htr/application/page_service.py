"""Page lifecycle: upload -> recognition -> suggestions -> editing -> confirm.

Three kinds of text are kept strictly apart:

* ``predicted_text`` — what kraken produced; never modified by anything;
* ``suggested_text`` — what the language model proposes for that prediction,
  stored next to the line and highlighting *which words* it would change;
* ``corrected_text`` — the user's transcription, the only ground truth, written
  by human edits or by accepting a proposal.

So the language model never touches the training corpus on its own: it can only
make a suggestion, and the user decides (per line, or in bulk for the lines
whose every change the dictionary confirms).
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Protocol

from ..domain.entities import PageStatus, PageSummary, PageView, RecognitionResult
from ..domain.errors import (
    InvalidTranscriptionError,
    NotFoundError,
    PageStateError,
    RecognitionError,
)
from ..domain.interfaces import (
    HTRRecognizer,
    ModelRepository,
    PageRepository,
    TextCorrector,
)
from .correction_context import CorrectionContextBuilder
from .lexicon import LexiconAnnotator
from .metrics import MetricsEvaluator
from .suggestions import apply_change, build_changes

logger = logging.getLogger(__name__)

# A model that reports "no answer" twice in a row is treated as unavailable:
# otherwise a hanging Ollama would cost one timeout per line of the page.
MAX_CONSECUTIVE_CORRECTION_FAILURES = 2

#: column limits of the page name/source path (see app.models.HTRPage)
FILE_NAME_MAX_LENGTH = 1024
SOURCE_PATH_MAX_LENGTH = 2048


def split_page_path(raw: str | None) -> tuple[str, str | None]:
    """Split an upload name into (basename, full path).

    Browsers send only the basename for a plain file input, but a folder upload
    (``webkitRelativePath``) or an API client can send a path. The basename is
    what the sidebar shows; the full path is kept only when it really contains a
    directory, because that is what tells two equal file names apart.
    """
    if not raw:
        return "page", None
    normalized = raw.strip().replace("\\", "/")
    name = normalized.rsplit("/", 1)[-1].strip()
    if not name:
        return "page", None
    return name[:FILE_NAME_MAX_LENGTH], (normalized[:SOURCE_PATH_MAX_LENGTH] if "/" in normalized else None)


class PageImageStore(Protocol):
    def save_page_image(self, user_id: int, author_id: int, filename: str, content: bytes) -> str: ...

    def probe_image(self, content: bytes) -> tuple[int, int]:
        """Return (width, height); raises CorruptImageError."""
        ...

    def delete_page_assets(self, page_id: int, file_path: str) -> None:
        """Best-effort removal of the page image and its line crops."""
        ...


class HandwritingPageService:
    def __init__(
        self,
        page_repository: PageRepository,
        model_repository: ModelRepository,
        image_store: PageImageStore,
        metrics_evaluator: MetricsEvaluator | None = None,
        recognizer: HTRRecognizer | None = None,
        corrector: TextCorrector | None = None,
        correction_context_builder: CorrectionContextBuilder | None = None,
        lexicon_annotator: LexiconAnnotator | None = None,
        # manual by default: the correction runs on POST /pages/{id}/correct
        auto_correct: bool = False,
        correction_context_lines: int = 2,
    ):
        self.page_repository = page_repository
        self.model_repository = model_repository
        self.image_store = image_store
        self.metrics_evaluator = metrics_evaluator or MetricsEvaluator()
        self.recognizer = recognizer
        self.corrector = corrector
        self.correction_context_builder = correction_context_builder
        # always present: besides the OOV marks it computes the word-level diff
        # of every model proposal, which needs no dictionary at all
        self.lexicon_annotator = lexicon_annotator or LexiconAnnotator(None)
        self.auto_correct = auto_correct
        self.correction_context_lines = max(0, correction_context_lines)

    # ------------------------------------------------------------------

    @property
    def lexicon_available(self) -> bool:
        return bool(self.lexicon_annotator and self.lexicon_annotator.is_available)

    def get_page(self, page_id: int) -> PageView:
        """One page with its vocabulary annotation (the read path of the UI)."""
        return self._annotate(self._get_page(page_id))

    def list_page_summaries(self, author_id: int) -> list[PageSummary]:
        """Sidebar rows of one author, with the OOV total of each page."""
        summaries = self.page_repository.list_page_summaries(author_id)
        if self.lexicon_annotator is None:
            return summaries
        return self.lexicon_annotator.annotate_summaries(summaries)

    def reorder_pages(self, author_id: int, page_ids: list[int]) -> list[PageSummary]:
        """Apply the sidebar order the user dragged the pages into."""
        summaries = self.page_repository.reorder_pages(author_id, page_ids)
        if self.lexicon_annotator is None:
            return summaries
        return self.lexicon_annotator.annotate_summaries(summaries)

    def rename_page(self, page_id: int, file_name: str) -> PageView:
        """Rename the displayed file name of one page."""
        self.page_repository.rename_page(page_id, file_name)
        return self.get_page(page_id)

    def _annotate(self, page: PageView) -> PageView:
        if self.lexicon_annotator is None:
            return page
        return self.lexicon_annotator.annotate(page)

    # ------------------------------------------------------------------

    def upload_page(
        self,
        user_id: int,
        author_id: int,
        filename: str,
        content: bytes,
        source_path: str | None = None,
    ) -> PageView:
        width, height = self.image_store.probe_image(content)
        file_path = self.image_store.save_page_image(user_id, author_id, filename, content)
        file_name, original_path = split_page_path(source_path or filename)
        page = self.page_repository.create_page(
            user_id=user_id,
            author_id=author_id,
            file_path=file_path,
            width=width,
            height=height,
            file_name=file_name,
            source_path=original_path,
        )
        logger.info(
            "HTR page uploaded: user_id=%s author_id=%s page_id=%s file_name=%s",
            user_id, author_id, page.id, file_name,
        )
        return self._annotate(page)

    def delete_page(self, page_id: int) -> None:
        """Remove a page from the corpus (and its image/crops from disk).

        Deleting a CONFIRMED page removes it from every future training dataset
        (each fine-tune rebuilds the corpus from the confirmed pages that exist
        at that moment). Already trained model versions keep the dataset hash
        they were built from, so the change stays auditable.
        """
        page = self._get_page(page_id)
        self.page_repository.delete_page(page_id)
        self.image_store.delete_page_assets(page_id, page.file_path)
        logger.info(
            "HTR page deleted: user_id=%s author_id=%s page_id=%s status=%s",
            page.user_id, page.author_id, page_id, page.status.value,
        )

    def recognize_page(self, page_id: int, force: bool = False) -> PageView:
        """Run the configured recognizer over the stored page image.

        The active model of the author is used when one exists, otherwise the
        configured default recognition model. The result is stored as the
        prediction; ground truth stays untouched.

        ``force`` re-segments a page whose structure is already fixed: an
        EDITING page (corrections are dropped) or a CONFIRMED one (its
        confirmation and transcript are dropped, and the page goes back to
        RECOGNIZED). This is the escape hatch for pages recognized with an older
        segmenter, so callers must confirm it with the user first.
        """
        page = self._get_page(page_id)
        allowed = (PageStatus.UPLOADED, PageStatus.RECOGNIZED)
        forceable = (PageStatus.EDITING, PageStatus.CONFIRMED)
        if page.status not in allowed and not (force and page.status in forceable):
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
        # the beam decoder scores word frequency and known words; give it the
        # author's own vocabulary (dictionary + confirmed words + knowledge
        # terms) instead of the general dictionary alone
        checker = (
            self.lexicon_annotator.checker_for(page.author_id)
            if self.lexicon_annotator
            else None
        )
        result = self.recognizer.recognize(
            page.file_path,
            model_path,
            author_id=page.author_id,
            word_checker=checker,
        )
        logger.info(
            "HTR page recognized: page_id=%s model=%s lines=%s force=%s",
            page_id, model_path, len(result.lines), force,
        )
        page = self.apply_recognition(page_id, result, force=force)
        if self.auto_correct and self.corrector is not None:
            # an unreachable LLM must not silently turn the automatic step into
            # a per-line timeout inside recognition
            problem = self.corrector_status()
            if problem is not None:
                logger.warning(
                    "HTR page %s: skipping the automatic correction: %s",
                    page_id, problem,
                )
            else:
                page = self.suggest_corrections(page_id)
        return self._annotate(page)

    # ------------------------------------------------------------------

    def corrector_status(self) -> str | None:
        """Why the LLM cannot be used right now, or None when it can.

        A cheap preflight, meant to be called *before* a page-long correction
        run: the difference for the user is an immediate explanation instead of
        a couple of timeouts and an empty page.
        """
        if self.corrector is None:
            return (
                "правки языковой моделью выключены "
                "(htr_correction_enabled=false)"
            )
        check = getattr(self.corrector, "is_available", None)
        if check is None:
            return None
        host = getattr(self.corrector, "host", None)
        try:
            reachable = bool(check())
        except Exception as exc:  # pragma: no cover - defensive
            return f"Ollama недоступна ({type(exc).__name__}: {exc})"
        if reachable:
            return None
        where = f" по адресу {host}" if host else ""
        return (
            f"Ollama недоступна{where} — запустите её и повторите; "
            "распознанный текст не изменялся"
        )

    def suggest_corrections(self, page_id: int) -> PageView:
        """Ask the LLM for corrections and store them as *proposals*.

        The transcription is not touched: each proposal goes to
        ``suggested_text`` with the list of word changes the dictionary could or
        could not confirm. The page status stays as it was, because nothing was
        edited yet.

        Lines the user has already written themselves are left alone, as are
        lines without a transcription. Best-effort by design: any failure
        (Ollama down, unusable answer) simply leaves that line without a
        proposal.
        """
        page = self._get_page(page_id)
        if page.status == PageStatus.CONFIRMED:
            # the transcription of a confirmed page is ground truth for training
            raise PageStateError(
                f"Page {page_id} is CONFIRMED; its transcription is ground truth "
                "and is not re-corrected"
            )
        if self.corrector is None or self.correction_context_builder is None:
            return self._annotate(page)
        lines = list(page.lines)
        if not lines:
            return self._annotate(page)
        try:
            context = self.correction_context_builder.build(
                page.author_id, exclude_page_id=page.id
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("HTR correction: cannot build context for page %s: %s", page_id, exc)
            return self._annotate(page)

        checker = self.lexicon_annotator.checker_for(page.author_id) if self.lexicon_annotator else None
        model_name = self.corrector_model_name()

        suggested = 0
        failures = 0
        for index, line in enumerate(lines):
            if line.corrected_text is not None:
                # the user has already written this line themselves; a proposal
                # about the discarded raw reading would only be noise
                continue
            raw = (line.predicted_text or "").strip()
            if not raw:
                continue
            neighbors = self._neighbor_texts(lines, index)
            text = self.corrector.correct_line(raw, neighbors, context)
            if text is None:
                # a hanging/unreachable LLM must not cost one timeout per line
                failures += 1
                if failures >= MAX_CONSECUTIVE_CORRECTION_FAILURES:
                    logger.warning(
                        "HTR correction aborted for page %s: %s failures in a row "
                        "(LLM unavailable?); no proposals stored",
                        page_id, failures,
                    )
                    break
                continue
            failures = 0
            if text.strip() == raw:
                continue
            page = self.page_repository.save_line_suggestion(
                page.id, line.id, text, model_name
            )
            suggested += 1

        logger.info(
            "HTR suggestions stored: page_id=%s lines=%s suggested=%s",
            page_id, len(lines), suggested,
        )
        return self._annotate(page)

    def accept_suggestion(self, page_id: int, line_id: int) -> PageView:
        """Move one proposal into the user's transcription."""
        page = self._annotate(self._get_editable_page(page_id))
        line = self._find_line(page, line_id)
        if not line.has_suggestion:
            raise NotFoundError(f"Line {line_id} has no model suggestion to accept")
        original = line.effective_text or ""
        page = self.page_repository.accept_line_suggestions(page_id, [line_id])
        self._log_suggestion(
            page, line, "accepted_line", original_text=original,
        )
        logger.info(
            "HTR suggestion accepted: page_id=%s line_id=%s words=%s",
            page_id, line_id, len(line.suggestion_changes),
        )
        return self._annotate(page)

    def accept_suggestion_change(self, page_id: int, line_id: int, index: int) -> PageView:
        """Accept exactly one proposed word, leaving the rest of the proposal.

        For the common case "the model is right about these words but wrong
        about that one": the user clicks the change they want instead of
        taking or rejecting the whole line.
        """
        page = self._annotate(self._get_editable_page(page_id))
        line = self._find_line(page, line_id)
        if not line.has_suggestion:
            raise NotFoundError(f"Line {line_id} has no model suggestion to accept")
        text = apply_change(line.effective_text or "", line.suggestion_changes, index)
        if text is None:
            raise NotFoundError(
                f"Line {line_id} has no suggestion change #{index} any more"
            )
        change = line.suggestion_changes[index]
        page = self.page_repository.apply_line_change(page_id, line_id, text)
        self._log_suggestion(
            page,
            line,
            "accepted_change",
            original_text=line.effective_text or "",
            change=change,
        )
        logger.info(
            "HTR suggestion change accepted: page_id=%s line_id=%s index=%s",
            page_id, line_id, index,
        )
        return self._annotate(page)

    def accept_verified_suggestions(self, page_id: int) -> tuple[PageView, int]:
        """Accept every proposal whose every change the dictionary confirms.

        Lines with an unverified change (a word nobody could check, a deletion)
        are left for the user to read — that is the whole point of verifying.
        """
        # annotated on purpose: suggestion_verified comes from the read-time
        # diff, which is not stored in the database
        page = self._annotate(self._get_editable_page(page_id))
        verified = [line.id for line in page.lines if line.suggestion_verified]
        if not verified:
            return self._annotate(page), 0
        originals = {line.id: line.effective_text or "" for line in page.lines if line.id in set(verified)}
        proposals = {line.id: line for line in page.lines if line.id in set(verified)}
        page = self.page_repository.accept_line_suggestions(page_id, verified)
        for line_id in verified:
            self._log_suggestion(
                page,
                proposals[line_id],
                "accepted_bulk",
                original_text=originals[line_id],
            )
        logger.info(
            "HTR suggestions accepted in bulk: page_id=%s lines=%s",
            page_id, len(verified),
        )
        return self._annotate(page), len(verified)

    def dismiss_suggestion(self, page_id: int, line_id: int) -> PageView:
        """Drop a proposal the user does not want to see."""
        page = self._annotate(self._get_editable_page(page_id))
        line = self._find_line(page, line_id)
        original = line.effective_text or ""
        self.page_repository.clear_line_suggestions(page_id, [line_id])
        self._log_suggestion(page, line, "dismissed", original_text=original)
        page = self._get_page(page_id)
        return self._annotate(page)

    def _log_suggestion(
        self,
        page,
        line,
        action: str,
        *,
        original_text: str | None = None,
        change=None,
    ) -> None:
        """Record what the user did with a proposal (best effort).

        The log is what makes the correction loop measurable: acceptance rate
        overall, per dictionary verdict, and per model.
        """
        try:
            self.page_repository.record_suggestion_event(
                page_id=page.id,
                line_id=line.id,
                user_id=page.user_id,
                action=action,
                suggested_text=line.suggested_text,
                original_text=original_text,
                change_before=getattr(change, "before", None),
                change_after=getattr(change, "after", None),
                in_lexicon=getattr(change, "in_lexicon", None),
                suggested_by=line.suggested_by,
            )
        except Exception as exc:  # pragma: no cover - the log must never break a request
            logger.warning("HTR suggestion log failed: %s", exc)

    def corrector_model_name(self) -> str:
        """Name of the model behind a proposal, for the UI/audit trail."""
        return getattr(self.corrector, "model", None) or "llm"

    def _neighbor_texts(self, lines: list, index: int) -> list[str]:
        """Surrounding lines as context for the line being corrected."""
        if not self.correction_context_lines:
            return []
        start = max(0, index - self.correction_context_lines)
        end = min(len(lines), index + self.correction_context_lines + 1)
        neighbors: list[str] = []
        for position in range(start, end):
            if position == index:
                continue
            text = (lines[position].effective_text or "").strip()
            if text:
                neighbors.append(text)
        return neighbors

    def apply_recognition(
        self, page_id: int, result: RecognitionResult, force: bool = False
    ) -> PageView:
        """Store a recognition result (from a recognizer or external import)."""
        page = self._get_page(page_id)
        if page.status not in (PageStatus.UPLOADED, PageStatus.RECOGNIZED) and not (
            force and page.status in (PageStatus.EDITING, PageStatus.CONFIRMED)
        ):
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
        return self._annotate(page)

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
        return self._annotate(
            self.page_repository.apply_word_update(
                page_id, line_id, word_id, corrected_text, line_text, words_stale
            )
        )

    def update_line(self, page_id: int, line_id: int, corrected_text: str) -> PageView:
        page = self._get_editable_page(page_id)
        self._find_line(page, line_id)
        return self._annotate(
            self.page_repository.apply_line_update(page_id, line_id, corrected_text)
        )

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
            "HTR page confirmed: user_id=%s author_id=%s page_id=%s "
            "prediction_cer=%s prediction_wer=%s",
            page.user_id, page.author_id, page_id,
            page.prediction_cer, page.prediction_wer,
        )
        return self._annotate(page)

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
