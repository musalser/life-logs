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
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

from ..domain.entities import PageStatus, PageSummary, PageView, RecognitionResult
from ..domain.errors import (
    DuplicatePageError,
    InvalidTranscriptionError,
    NotFoundError,
    PageStateError,
    RecognitionError,
)
from ..domain.text import alignment_preserved, iter_words, normalize_word
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
from .training_service import author_training_in_progress

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConfirmationReport:
    """The confirmed page plus the vocabulary its confirmation taught."""

    page: PageView
    #: words that entered the author's vocabulary with this page
    added_author_words: tuple[str, ...] = ()
    #: subset the general dictionary does not know: they stop being OOV marks
    learned_words: tuple[str, ...] = ()


@dataclass(frozen=True)
class ConfirmationPreview:
    """What confirming a page *would* teach the author's vocabulary.

    Computed without writing anything, so the user can be shown the words a
    confirmation is about to make known — and fix the ones that should not be.
    """

    #: words that would enter the author's vocabulary with this page
    added_author_words: tuple[str, ...] = ()
    #: subset the general dictionary does not know: they would stop being OOV
    learned_words: tuple[str, ...] = ()

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

    # ------------------------------------------------------------------
    # The author's own vocabulary (the dictionary panel)

    def author_dictionary(self, author_id: int) -> dict:
        """Words taken from the author's confirmed pages (+ knowledge terms)."""
        payload = self.lexicon_annotator.dictionary(author_id)
        return payload or {"available": False, "words": [], "ignored": []}

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
        file_name, original_path = split_page_path(source_path or filename)
        # A page is identified by its name *and* where it came from: the same name
        # from another folder is another page (that is what `source_path` is kept
        # for), while the same name from the same source is one page uploaded
        # twice — a duplicate in the corpus, not a new entry. The check runs
        # before the image is written, so a refused upload leaves no orphan file.
        duplicate = self.page_repository.find_page_by_name(author_id, file_name, original_path)
        if duplicate is not None:
            raise DuplicatePageError(
                f"Page '{file_name}' from this source is already uploaded (page {duplicate.id})"
            )
        file_path = self.image_store.save_page_image(user_id, author_id, filename, content)
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

    def reopen_page(self, page_id: int) -> PageView:
        """Return a confirmed page to editing — the reverse of ``confirm_page``.

        The page stops being ground truth: the next training run leaves it out
        and its words stop counting as the author's vocabulary (the corpus cache
        is invalidated), while every correction the user made is kept and
        editing works again. A model already trained on the page is *not*
        changed: untraining is impossible, only the next run is affected.
        """
        page = self._get_page(page_id)
        if page.status != PageStatus.CONFIRMED:
            raise PageStateError(
                f"Page {page_id} is {page.status}; only a CONFIRMED page can be "
                "returned to editing"
            )
        if author_training_in_progress(page.author_id):
            raise PageStateError(
                "A training run for this author is in progress; wait for it to "
                "finish before returning the page to editing"
            )
        page = self.page_repository.reopen_page(page_id)
        logger.info(
            "HTR page reopened for editing: user_id=%s author_id=%s page_id=%s",
            page.user_id, page.author_id, page_id,
        )
        return self._annotate(page)

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
        # replacing one word with another keeps box i on token i; only a split
        # («не знаю» typed into one box) or a deletion invalidates the geometry
        words_stale = not alignment_preserved(len(line.words), line_text)
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

    def delete_line(self, page_id: int, line_id: int) -> PageView:
        """Drop a line the segmenter invented (or that is not text at all)."""
        page = self._get_editable_page(page_id)
        self._find_line(page, line_id)
        logger.info(
            "HTR line deleted: user_id=%s author_id=%s page_id=%s line_id=%s",
            page.user_id, page.author_id, page_id, line_id,
        )
        return self._annotate(self.page_repository.delete_line(page_id, line_id))

    # ------------------------------------------------------------------

    def confirm_page(self, page_id: int) -> PageView:
        """Confirm the page (kept for callers that only need the page itself)."""
        return self.confirm_page_report(page_id).page

    def preview_confirmation(self, page_id: int) -> ConfirmationPreview:
        """What confirming the page would add to the author's dictionary.

        The read-only twin of :meth:`confirm_page_report`: the same computation
        on the same vocabulary, but nothing is written. The UI opens it when the
        user asks to confirm a page, so a word that is about to be taught to the
        dictionary by mistake can still be fixed.
        """
        page = self._get_confirmable_page(page_id)
        vocabulary_before = self.lexicon_annotator.author_words(page.author_id)
        vocabulary_after = self.lexicon_annotator.author_words_after_confirming(page)
        added, learned = self._dictionary_delta(page, vocabulary_before, vocabulary_after)
        return ConfirmationPreview(added_author_words=added, learned_words=learned)

    def confirm_page_report(self, page_id: int) -> ConfirmationReport:
        """Confirm the page and report what its words added to the dictionary.

        A confirmation is what puts a page's words into the author's vocabulary,
        so it is the moment to show them: the user can spot a word the page
        taught by mistake while it is still fresh, rather than meeting it later
        as an unexpected "known" word.
        """
        page = self._get_confirmable_page(page_id)
        # Prediction quality on this page vs. the user's ground truth (section 19).
        pairs = [(line.effective_text or "", line.predicted_text or "") for line in page.lines]
        metrics = self.metrics_evaluator.evaluate_pairs(pairs) if pairs else None
        vocabulary_before = self.lexicon_annotator.author_words(page.author_id)
        page = self.page_repository.confirm_page(
            page_id,
            confirmed_at=datetime.now(timezone.utc),
            prediction_cer=metrics["cer"] if metrics else None,
            prediction_wer=metrics["wer"] if metrics else None,
        )
        vocabulary_after = self.lexicon_annotator.author_words(page.author_id)
        added, learned = self._dictionary_delta(page, vocabulary_before, vocabulary_after)
        if learned:
            logger.info(
                "HTR author vocabulary grew: page_id=%s added=%s learned=%s",
                page_id, len(added), list(learned[:20]),
            )
        logger.info(
            "HTR page confirmed: user_id=%s author_id=%s page_id=%s "
            "prediction_cer=%s prediction_wer=%s",
            page.user_id, page.author_id, page_id,
            page.prediction_cer, page.prediction_wer,
        )
        return ConfirmationReport(
            page=self._annotate(page),
            added_author_words=added,
            learned_words=learned,
        )

    def _get_confirmable_page(self, page_id: int) -> PageView:
        """A page in an editable state whose every line has a transcription.

        Shared by the confirmation and its preview, so the preview cannot offer
        words the confirmation would then refuse to take.
        """
        page = self._get_editable_page(page_id)
        for line in page.lines:
            if not line.has_valid_transcription:
                raise InvalidTranscriptionError(
                    f"Line {line.id} has an empty transcription that was not explicitly "
                    "confirmed by the user"
                )
        return page

    def _dictionary_delta(
        self,
        page: PageView,
        before: frozenset[str] | None,
        after: frozenset[str] | None,
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """The words a page adds to the author dictionary and what they change.

        ``before`` / ``after`` are the author vocabularies around the write; the
        preview projects ``after`` instead of confirming. Returns the words the
        page contributes and the subset the general dictionary does not know —
        the ones that stop being flagged as unknown.
        """
        if before is None or after is None:
            return (), ()
        # only the words actually written on the page: the vocabulary also holds
        # lookup variants ("еврея" brings "евреё" along), and showing those would
        # confuse the very report meant to be checkable by eye
        page_words = {
            normalize_word(token)
            for line in page.lines
            for token in iter_words(line.effective_text or "")
            if any(character.isalpha() for character in token)
        }
        new_words = sorted(word for word in after - before if word in page_words)
        # built on the *after* vocabulary: those words are the author's own now,
        # so only `is_author_only` can tell them from the general dictionary's
        checker = self.lexicon_annotator.checker_with_words(after)
        added = tuple(new_words)
        learned = tuple(
            word for word in new_words if checker is not None and checker.is_author_only(word)
        )
        return added, learned

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
