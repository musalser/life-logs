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
  known, so their names and dialect words are not flagged. A word that only the
  author's pages vouch for is marked ``author_only`` instead of being silently
  accepted: it may be a name — or a typo that was confirmed once. Such words are
  not part of ``oov_count``, and the dictionary panel can remove them.
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

    def author_words(self, author_id: int) -> frozenset[str] | None:
        """The author's own vocabulary, or None when there is no dictionary.

        Used to tell the user what confirming a page just taught the dictionary.
        """
        provider = self.provider
        getter = getattr(provider, "author_words", None) if provider else None
        if getter is None:
            return None
        try:
            return frozenset(getter(author_id))
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("HTR lexicon: no vocabulary for author %s: %s", author_id, exc)
            return None

    def author_words_after_confirming(self, page: PageView) -> frozenset[str] | None:
        """The vocabulary the author would have if ``page`` were confirmed.

        The read-only twin of :meth:`author_words`: the confirmation preview
        asks what a page is about to teach, so nothing is written.
        """
        provider = self.provider
        getter = getattr(provider, "author_words_with_page", None) if provider else None
        if getter is None:
            return None
        try:
            # reading order matters: the line-break rules join the tail of one
            # line to the head of the next
            texts = [
                line.effective_text or ""
                for line in sorted(page.lines, key=lambda item: item.order)
            ]
            return frozenset(getter(page.author_id, page.id, texts))
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(
                "HTR lexicon: no projected vocabulary for page %s: %s", page.id, exc
            )
            return None

    def checker_with_words(self, author_words) -> LexiconChecker | None:
        """A check whose author layer is given explicitly (confirmation preview).

        ``None`` when the provider cannot build one; callers then report nothing
        rather than guessing which words the general dictionary knows.
        """
        provider = self.provider
        getter = getattr(provider, "checker_with_words", None) if provider else None
        if getter is None:
            return None
        try:
            return getter(author_words)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("HTR lexicon: no preview check: %s", exc)
            return None

    # -- the author dictionary panel ------------------------------------

    def dictionary(self, author_id: int) -> dict | None:
        """The author's own vocabulary + the words they removed.

        None when the provider cannot list words (no dictionary configured or a
        provider without the panel methods).
        """
        provider = self.provider
        if provider is None or not hasattr(provider, "author_terms"):
            return None
        return {
            "available": self.is_available,
            "words": provider.author_terms(author_id),
        }


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
            line.author_only_count = 0
            line.author_only_words = []
            for word in words:
                word.in_lexicon = None
                word.author_only = False
            return 0

        tokens = iter_words(line.effective_text)
        # the word boxes are aligned with the tokens of the transcription
        # unless the user rewrote the line (words_stale) or changed the count
        aligned = not line.words_stale and len(tokens) == len(words)
        for index, word in enumerate(words):
            if aligned:
                token = tokens[index]
                word.in_lexicon = checker.is_known(token)
                word.author_only = checker.is_author_only(token)
            else:
                own = iter_words(word.effective_text)
                word.in_lexicon = (
                    all(checker.is_known(token) for token in own) if own else None
                )
                # a box is weak when it is known at all, but only the author's
                # own pages vouch for it
                word.author_only = (
                    bool(own)
                    and all(checker.is_known(token) for token in own)
                    and any(checker.is_author_only(token) for token in own)
                )

        # the badge and the list follow the transcription the user sees, even
        # when the (stale) word boxes no longer match it
        line.oov_words = [token for token in tokens if not checker.is_known(token)]
        line.oov_count = len(line.oov_words)
        line.author_only_words = [
            token for token in tokens if checker.is_author_only(token)
        ]
        line.author_only_count = len(line.author_only_words)
        return line.oov_count

    @staticmethod
    def _count_oov(texts: list[str], checker: LexiconChecker) -> int:
        return sum(
            1
            for text in texts
            for token in iter_words(text)
            if not checker.is_known(token)
        )
