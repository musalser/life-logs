"""Per-author text corpus used by the vocabulary check.

Two things come from the same snapshot and are therefore cached together:

* **known words** — the vocabulary of the author's *confirmed* pages (their own
  ground truth) plus the proper nouns of the knowledge base. Without it every
  toponym and dialect word of the author would be flagged as unknown;
* **page transcriptions** — the effective line texts per page, so the page list
  can show an OOV total without loading every line of every page.

The snapshot is invalidated by :class:`~app.htr.infrastructure.page_repository.SqlAlchemyPageRepository`
on every write, so a confirmation or an edit is visible on the next read.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from ....models import HTRAuthor, HTRLine, HTRPage
from ..knowledge_vocabulary import SqlAlchemyKnowledgeVocabulary
from .words import words_of

CONFIRMED = "CONFIRMED"


@dataclass(frozen=True)
class AuthorCorpusSnapshot:
    known_words: frozenset[str] = frozenset()
    page_texts: dict[int, list[str]] = field(default_factory=dict)


class AuthorCorpusCache:
    """Process-wide cache of author snapshots (one entry per author)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: dict[int, AuthorCorpusSnapshot] = {}

    def get(self, author_id: int) -> AuthorCorpusSnapshot | None:
        with self._lock:
            return self._data.get(author_id)

    def put(self, author_id: int, snapshot: AuthorCorpusSnapshot) -> None:
        with self._lock:
            self._data[author_id] = snapshot

    def invalidate(self, author_id: int) -> None:
        with self._lock:
            self._data.pop(author_id, None)

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
    ):
        self.db = db
        self.cache = cache if cache is not None else author_corpus_cache
        self.vocabulary_provider = vocabulary_provider or SqlAlchemyKnowledgeVocabulary(db)

    # ------------------------------------------------------------------

    def snapshot(self, author_id: int) -> AuthorCorpusSnapshot:
        cached = self.cache.get(author_id)
        if cached is not None:
            return cached
        snapshot = self._load(author_id)
        self.cache.put(author_id, snapshot)
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
            )
            .outerjoin(HTRLine, HTRLine.page_id == HTRPage.id)
            .filter(HTRPage.author_id == author_id)
            .order_by(HTRPage.id, HTRLine.order_index)
            .all()
        )

        known: set[str] = set()
        page_texts: dict[int, list[str]] = {}
        user_ids: set[int] = set()

        owner = (
            self.db.query(HTRAuthor.user_id).filter(HTRAuthor.id == author_id).first()
        )
        if owner is not None:
            user_ids.add(owner[0])

        for page_id, status, user_id, corrected, predicted in rows:
            user_ids.add(user_id)
            texts = page_texts.setdefault(page_id, [])
            text = (corrected if corrected is not None else predicted) or ""
            text = text.strip()
            if not text:
                continue
            texts.append(text)
            if status == CONFIRMED:
                known.update(words_of(text))

        for user_id in sorted(user_ids):
            for term in self.vocabulary_provider.vocabulary(user_id):
                known.update(words_of(term))

        return AuthorCorpusSnapshot(known_words=frozenset(known), page_texts=page_texts)

    # ------------------------------------------------------------------

    def known_words(self, author_id: int) -> frozenset[str]:
        return self.snapshot(author_id).known_words

    def page_texts(self, author_id: int) -> dict[int, list[str]]:
        return self.snapshot(author_id).page_texts

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
