"""Composition of the vocabulary check for one author."""
from __future__ import annotations

from typing import Iterable, Sequence

from ...domain.interfaces import WordList
from .checker import LayeredLexiconChecker
from .corpus import SqlAlchemyAuthorCorpus


class SqlAlchemyLexiconProvider:
    """Implements the domain ``LexiconProvider`` on top of the DB + dictionary."""

    def __init__(self, corpus: SqlAlchemyAuthorCorpus, base: WordList | None = None):
        self.corpus = corpus
        self.base = base

    def checker(self, author_id: int) -> LayeredLexiconChecker:
        """The dictionary check: the Russian word list plus the author's own
        confirmed words. Nothing else — terms of the knowledge base are produced
        by entity extraction and are not vocabulary (see the README)."""
        snapshot = self.corpus.snapshot(author_id)
        return LayeredLexiconChecker(
            base=self.base,
            author=snapshot.author_words,
        )

    def page_transcriptions(self, author_id: int) -> dict[int, list[str]]:
        return self.corpus.page_texts(author_id)

    # -- the dictionary panel (view / edit the author's own vocabulary) -----

    def author_terms(self, author_id: int) -> list[dict]:
        snapshot = self.corpus.snapshot(author_id)
        checker = LayeredLexiconChecker(base=self.base, author=snapshot.author_words)
        # Only the words the general dictionary does *not* have are interesting
        # here: removing one of those is what changes the highlighting. An
        # ordinary word ("дом") stays known through the base dictionary anyway.
        terms = [
            {
                "word": word,
                "count": count,
                "source": "author",
                # where the word actually stands, so the panel can jump there
                "occurrences": [
                    {
                        "page_id": place.page_id,
                        "line_id": place.line_id,
                        "line_order": place.line_order,
                        "surface": place.surface,
                    }
                    for place in snapshot.occurrences.get(word, ())
                ],
            }
            for word, count in snapshot.author_terms
            if checker.is_author_only(word)
        ]
        return terms

    def author_words(self, author_id: int) -> frozenset[str]:
        """The author's own vocabulary (words their confirmed pages taught)."""
        return self.corpus.author_words(author_id)

    # -- the confirmation preview (what a write *would* teach) --------------

    def author_words_with_page(
        self, author_id: int, page_id: int, page_texts: Sequence[str]
    ) -> frozenset[str]:
        """The vocabulary the author would have if ``page_texts`` were confirmed."""
        return self.corpus.author_words_with_page(author_id, page_id, page_texts)

    def checker_with_words(self, author_words: Iterable[str]) -> LayeredLexiconChecker:
        """A dictionary check with an explicit author layer.

        The confirmation preview passes the vocabulary the author *would* have:
        building the check from the stored one would call the very words under
        review unknown instead of the author's own.
        """
        return LayeredLexiconChecker(base=self.base, author=author_words)

    def invalidate(self, author_id: int) -> None:
        self.corpus.invalidate(author_id)
