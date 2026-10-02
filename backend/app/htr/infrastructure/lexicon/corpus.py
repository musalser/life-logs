"""Per-author text corpus used by the vocabulary check.

Two things come from the same snapshot and are therefore cached together:

* **known words** — the vocabulary of the author's *confirmed* pages (their own
  ground truth) plus the proper nouns of the knowledge base. Without it every
  toponym and dialect word of the author would be flagged as unknown;
* **page transcriptions** — the effective line texts per page, so the page list
  can show an OOV total without loading every line of every page.

The two vocabularies are kept apart because they mean different things: the
knowledge base and the general dictionary are *strong* evidence, while the
author's own confirmed pages are *weak* — they also contain the typos the user
confirmed by accident. :class:`~app.htr.infrastructure.lexicon.checker.LayeredLexiconChecker`
turns that into separate verdicts, and :class:`HTRLexiconIgnore` rows let the
user subtract a word from their own vocabulary.

The snapshot is invalidated by :class:`~app.htr.infrastructure.page_repository.SqlAlchemyPageRepository`
on every write, so a confirmation or an edit is visible on the next read.
"""
from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from ....models import HTRAuthor, HTRLexiconIgnore, HTRLine, HTRPage

logger = logging.getLogger(__name__)
from ..knowledge_vocabulary import SqlAlchemyKnowledgeVocabulary
from .words import words_of
from ...domain.text import (
    WORD_RE,
    iter_words,
    line_break_analysis,
    normalize_word,
    word_variants,
)

CONFIRMED = "CONFIRMED"

#: the dictionary panel is read whole into the browser; this keeps the payload
#: bounded once an author has a very large confirmed corpus
MAX_AUTHOR_TERMS = 20000

#: how many places per word the panel can jump to, and for how many words they
#: are collected at all — the panel lists at most ``MAX_AUTHOR_TERMS``
MAX_OCCURRENCES_PER_WORD = 8
MAX_OCCURRENCE_WORDS = 30000


@dataclass(frozen=True)
class WordOccurrence:
    """Where a word of the author's vocabulary actually stands."""

    page_id: int
    line_id: int
    line_order: int
    #: the spelling as it appears in the line (the panel lists normalized forms)
    surface: str


@dataclass(frozen=True)
class AuthorCorpusSnapshot:
    #: everything the checker accepts (union, kept for existing callers)
    known_words: frozenset[str] = frozenset()
    #: forms known only from the author's confirmed pages (weak verdict)
    author_words: frozenset[str] = frozenset()
    #: kept for callers of older snapshots; the knowledge base is *not* a
    #: dictionary source any more (entity extraction produces word pieces)
    page_texts: dict[int, list[str]] = field(default_factory=dict)
    #: ``(normalized surface, occurrences)`` of the confirmed pages, by count
    author_terms: tuple[tuple[str, int], ...] = ()
    #: normalized knowledge-base terms
    #: words the user removed (normalized surface)
    ignored: tuple[str, ...] = ()
    #: words the writer split over a line break (see ``line_break_analysis``)
    line_breaks: tuple = ()
    #: normalized word -> places it occurs, for the dictionary panel. Only words
    #: the general dictionary does not know are collected: they are the ones the
    #: panel lists, and collecting every "и" would cost more than it is worth.
    occurrences: dict[str, tuple[WordOccurrence, ...]] = field(default_factory=dict)


class AuthorCorpusCache:
    """Process-wide cache of author snapshots.

    The key carries whether the general dictionary was available, because that
    decides how words broken at a line break are treated: a snapshot built
    without it must never be served to a caller that asked for the check.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: dict[tuple[int, bool], AuthorCorpusSnapshot] = {}

    def get(self, author_id: int, with_dictionary: bool = False):
        with self._lock:
            return self._data.get((author_id, with_dictionary))

    def put(
        self, author_id: int, snapshot: AuthorCorpusSnapshot, with_dictionary: bool = False
    ) -> None:
        with self._lock:
            self._data[(author_id, with_dictionary)] = snapshot

    def invalidate(self, author_id: int) -> None:
        with self._lock:
            for key in [k for k in self._data if k[0] == author_id]:
                del self._data[key]

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


# Shared by the factory (reader) and the page repository (invalidator).
author_corpus_cache = AuthorCorpusCache()


class SqlAlchemyAuthorCorpus:
    def __init__(
        self,
        db: Session,
        cache: AuthorCorpusCache | None = None,
        vocabulary_provider: SqlAlchemyKnowledgeVocabulary | None = None,
        base_is_known: Callable[[str], bool] | None = None,
    ):
        self.db = db
        self.cache = cache if cache is not None else author_corpus_cache
        self.vocabulary_provider = vocabulary_provider or SqlAlchemyKnowledgeVocabulary(db)
        #: the general dictionary, used to tell a broken word from a real one:
        #: "лась" is not a word, "был" is. Without it (a caller that only needs
        #: the raw corpus) line-break fragments are left as they are.
        self.base_is_known = base_is_known

    # ------------------------------------------------------------------

    def snapshot(self, author_id: int) -> AuthorCorpusSnapshot:
        with_dictionary = self.base_is_known is not None
        cached = self.cache.get(author_id, with_dictionary)
        if cached is not None:
            return cached
        snapshot = self._load(author_id)
        self.cache.put(author_id, snapshot, with_dictionary)
        return snapshot

    def invalidate(self, author_id: int) -> None:
        self.cache.invalidate(author_id)

    # ------------------------------------------------------------------

    def _load(self, author_id: int) -> AuthorCorpusSnapshot:
        rows = (
            self.db.query(
                HTRPage.id,
                HTRPage.status,
                HTRPage.user_id,
                HTRLine.corrected_text,
                HTRLine.predicted_text,
                HTRLine.id,
                HTRLine.order_index,
            )
            .outerjoin(HTRLine, HTRLine.page_id == HTRPage.id)
            .filter(HTRPage.author_id == author_id)
            .order_by(HTRPage.id, HTRLine.order_index)
            .all()
        )

        page_texts: dict[int, list[str]] = {}
        confirmed_pages: dict[int, list[str]] = {}
        occurrences: dict[str, list[WordOccurrence]] = {}

        for page_id, status, _user_id, corrected, predicted, line_id, line_order in rows:
            texts = page_texts.setdefault(page_id, [])
            text = (corrected if corrected is not None else predicted) or ""
            text = text.strip()
            if not text:
                continue
            texts.append(text)
            if status != CONFIRMED:
                continue
            confirmed_pages.setdefault(page_id, []).append(text)
            for token in iter_words(text):
                if not any(character.isalpha() for character in token):
                    continue
                normalized = normalize_word(token)
                # remember where it stands, but only for the words the panel
                # shows (the general dictionary does not know them): a place
                # for every "и" would be collected for nothing
                if not normalized or line_id is None:
                    continue
                if len(occurrences) >= MAX_OCCURRENCE_WORDS:
                    continue
                if self.base_is_known is not None and self.base_is_known(normalized):
                    continue
                places = occurrences.setdefault(normalized, [])
                if len(places) < MAX_OCCURRENCES_PER_WORD:
                    places.append(
                        WordOccurrence(
                            page_id=page_id,
                            line_id=line_id,
                            line_order=line_order or 0,
                            surface=token,
                        )
                    )

        author_known, author_counts, line_breaks = self._vocabulary(author_id, confirmed_pages)
        author_terms = tuple(
            sorted(author_counts.items(), key=lambda item: (-item[1], item[0]))[
                :MAX_AUTHOR_TERMS
            ]
        )
        return AuthorCorpusSnapshot(
            known_words=author_known,
            author_words=author_known,
            page_texts=page_texts,
            author_terms=author_terms,
            line_breaks=line_breaks,
            occurrences={
                word: tuple(places) for word, places in occurrences.items()
            },
        )

    def _vocabulary(
        self, author_id: int, pages: dict[int, list[str]]
    ) -> tuple[frozenset[str], dict[str, int], tuple]:
        """Vocabulary and word counts of the given confirmed page texts.

        Split out of :meth:`_load` because the confirmation preview needs the
        vocabulary the author *would* have if one more page were confirmed, and
        it must apply exactly the same line-break rules as a stored page —
        otherwise the preview would list the halves of a word the writer broke
        at the right edge, which confirmation never adds.
        """
        author_known: set[str] = set()
        author_counts: dict[str, int] = {}
        #: words that occur *inside* a line: the only precise evidence available
        #: for "this half is a word of its own" (the dictionary answers "yes" to
        #: about one string in a hundred, and it also contains morphemes like
        #: "ным"/"ровать", so it cannot be asked that question)
        interior_words: set[str] = set()

        for texts in pages.values():
            for text in texts:
                author_known.update(words_of(text))
                tokens = iter_words(text)
                for token in tokens:
                    if not any(character.isalpha() for character in token):
                        continue
                    normalized = normalize_word(token)
                    if normalized:
                        author_counts[normalized] = author_counts.get(normalized, 0) + 1
                interior_words.update(normalize_word(token) for token in tokens[1:-1])

        # A word the writer broke at the end of a line arrives as two halves
        # ("слуша-" + "лась"), and both look like words the author confirmed —
        # they are not, and they must not teach the OOV highlighting that the
        # fragment is a word. The joined form is what deserves to be known.
        line_breaks: tuple = ()
        if self.base_is_known is not None and pages:
            analysis = line_break_analysis(
                list(pages.values()),
                self.base_is_known,
                is_word=interior_words.__contains__,
            )
            for fragment in analysis.fragments:
                author_known -= words_of(fragment)
                author_counts.pop(fragment, None)
            for word in analysis.words:
                author_known.update(words_of(word))
                author_counts[word] = author_counts.get(word, 0) + 1
            line_breaks = analysis.joins
            if analysis.fragments:
                logger.info(
                    "HTR author corpus: %s line-break fragment(s) and %s joined "
                    "word(s) for author_id=%s",
                    len(analysis.fragments), len(analysis.words), author_id,
                )
        return frozenset(author_known), author_counts, line_breaks

    def author_words_with_page(
        self, author_id: int, page_id: int, page_texts: Sequence[str]
    ) -> frozenset[str]:
        """The vocabulary as if ``page_texts`` were already confirmed.

        Backs the confirmation preview: nothing is written, but the candidate
        page goes through the same rules as the confirmed ones, so the words the
        preview offers to review are exactly the ones the write would add.
        """
        pages = self.confirmed_texts(author_id)
        pages[page_id] = [text.strip() for text in page_texts if (text or "").strip()]
        known, _counts, _breaks = self._vocabulary(author_id, pages)
        return known

    # ------------------------------------------------------------------

    def known_words(self, author_id: int) -> frozenset[str]:
        return self.snapshot(author_id).known_words

    def author_words(self, author_id: int) -> frozenset[str]:
        """Forms known only from the author's confirmed pages (weak verdict)."""
        return self.snapshot(author_id).author_words

    def page_texts(self, author_id: int) -> dict[int, list[str]]:
        return self.snapshot(author_id).page_texts

    def terms(self, author_id: int) -> tuple[tuple[str, int], ...]:
        """``(word, occurrences)`` of the confirmed pages, most frequent first."""
        return self.snapshot(author_id).author_terms

    # ------------------------------------------------------------------

    def confirmed_texts(self, author_id: int) -> dict[int, list[str]]:
        """Effective line texts of the confirmed pages only (for calibration)."""
        rows = (
            self.db.query(HTRPage.id, HTRLine.corrected_text, HTRLine.predicted_text)
            .join(HTRLine, HTRLine.page_id == HTRPage.id)
            .filter(HTRPage.author_id == author_id, HTRPage.status == CONFIRMED)
            .order_by(HTRPage.id, HTRLine.order_index)
            .all()
        )
        texts: dict[int, list[str]] = {}
        for page_id, corrected, predicted in rows:
            text = ((corrected if corrected is not None else predicted) or "").strip()
            if text:
                texts.setdefault(page_id, []).append(text)
        return texts
