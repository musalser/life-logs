"""Vocabulary annotation of a recognized page.

The recognizer confidently produces words that do not exist (proper nouns,
dialect, or plain misreadings), and the confidence score does not catch those:
it is high precisely when the model is sure of the wrong thing. A dictionary
check catches them, which is why every word carries ``in_lexicon`` and every
line/page an ``oov_count`` on read.

The check is deliberately read-only and best-effort:

* it never modifies the prediction or the user's correction;
* when no dictionary is installed the annotation stays empty
  (``in_lexicon = None``, ``lexicon_available = False``) so the UI can say so
  instead of marking the whole page;
* the author's own confirmed transcriptions and knowledge-base terms count as
  known, so their names and dialect words are not flagged.
"""
from __future__ import annotations

import logging

from ..domain.entities import LineView, PageSummary, PageView
from ..domain.interfaces import LexiconChecker, LexiconProvider
from ..domain.text import iter_words
from .suggestions import build_changes

logger = logging.getLogger(__name__)


class LexiconAnnotator:
    def __init__(self, provider: LexiconProvider | None = None):
        self.provider = provider

    @property
    def is_available(self) -> bool:
        return self.provider is not None

    def checker_for(self, author_id: int) -> LexiconChecker | None:
        """The dictionary check of one author, or None when there is none.

        Used outside the page annotation as well: model proposals are verified
        against the same vocabulary before the user is offered to accept them.
        """
        if self.provider is None:
            return None
        try:
            return self.provider.checker(author_id)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("HTR lexicon: no checker for author %s: %s", author_id, exc)
            return None

    # ------------------------------------------------------------------

    def annotate(self, page: PageView) -> PageView:
        """Fill in_lexicon / oov_count on a page read model (in place)."""
        available = False
        checker: LexiconChecker | None = None
        if self.provider is not None:
            try:
                checker = self.provider.checker(page.author_id)
                available = checker.is_available
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("HTR lexicon: check unavailable for page %s: %s", page.id, exc)
                checker, available = None, False

        page.lexicon_available = available
        total = 0
        for line in page.lines:
            total += self._annotate_line(line, checker, available)
        page.oov_count = total
        return page

    def annotate_summaries(self, summaries: list[PageSummary]) -> list[PageSummary]:
        """Add per-page OOV totals to sidebar rows (no line payload needed)."""
        if self.provider is None or not summaries:
            return summaries
        by_author: dict[int, tuple[LexiconChecker, dict[int, list[str]]]] = {}
        for summary in summaries:
            entry = by_author.get(summary.author_id)
            if entry is None:
                try:
                    checker = self.provider.checker(summary.author_id)
                    available = checker.is_available
                    texts = self.provider.page_transcriptions(summary.author_id) if available else {}
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning("HTR lexicon: summaries unavailable: %s", exc)
                    checker, available, texts = None, False, {}
                entry = (checker, texts) if available else (None, {})
                by_author[summary.author_id] = entry
            checker, texts = entry
            summary.lexicon_available = checker is not None
            if checker is None:
                summary.oov_count = None
                continue
            summary.oov_count = self._count_oov(texts.get(summary.id, []), checker)
        return summaries

    # ------------------------------------------------------------------

    def _annotate_line(
        self, line: LineView, checker: LexiconChecker | None, available: bool
    ) -> int:
        words = sorted(line.words, key=lambda word: word.order)
        # the model proposal is reviewed as a list of word changes, each with
        # its dictionary verdict; without a dictionary nothing can be verified
        # (in_lexicon stays None, which blocks the bulk accept)
        current = (line.effective_text or "").strip()
        proposal = (line.suggested_text or "").strip()
        line.suggestion_changes = (
            build_changes(current, proposal, checker if available else None)
            if proposal and proposal != current
            else []
        )
        if not available or checker is None:
            line.oov_count = 0
            line.oov_words = []
            for word in words:
                word.in_lexicon = None
            return 0

        tokens = iter_words(line.effective_text)
        # the word boxes are aligned with the tokens of the transcription
        # unless the user rewrote the line (words_stale) or changed the count
        aligned = not line.words_stale and len(tokens) == len(words)
        for index, word in enumerate(words):
            if aligned:
                word.in_lexicon = checker.is_known(tokens[index])
            else:
                own = iter_words(word.effective_text)
                word.in_lexicon = (
                    all(checker.is_known(token) for token in own) if own else None
                )

        # the badge and the list follow the transcription the user sees, even
        # when the (stale) word boxes no longer match it
        line.oov_words = [token for token in tokens if not checker.is_known(token)]
        line.oov_count = len(line.oov_words)
        return line.oov_count

    @staticmethod
    def _count_oov(texts: list[str], checker: LexiconChecker) -> int:
        return sum(
            1
            for text in texts
            for token in iter_words(text)
            if not checker.is_known(token)
        )
