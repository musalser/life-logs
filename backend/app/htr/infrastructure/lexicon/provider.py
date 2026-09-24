"""Composition of the vocabulary check for one author."""
from __future__ import annotations

from ...domain.interfaces import WordList
from .checker import LayeredLexiconChecker
from .corpus import SqlAlchemyAuthorCorpus


class SqlAlchemyLexiconProvider:
    """Implements the domain ``LexiconProvider`` on top of the DB + dictionary."""

    def __init__(self, corpus: SqlAlchemyAuthorCorpus, base: WordList | None = None):
        self.corpus = corpus
        self.base = base

    def checker(self, author_id: int) -> LayeredLexiconChecker:
        return LayeredLexiconChecker(base=self.base, extra=self.corpus.known_words(author_id))

    def page_transcriptions(self, author_id: int) -> dict[int, list[str]]:
        return self.corpus.page_texts(author_id)
