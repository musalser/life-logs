"""Layered vocabulary check: general dictionary + the author's own words."""
from __future__ import annotations

from typing import Iterable

from ...domain.interfaces import WordList
from ...domain.text import word_variants

#: verdict of :meth:`LayeredLexiconChecker.verdict`
KNOWN = "known"      # the general dictionary (or the knowledge base) knows it
AUTHOR = "author"    # only the author's own confirmed pages know it
UNKNOWN = "unknown"  # nobody knows it


class LayeredLexiconChecker:
    """Answers "is this recognized word known?" for one author.

    ``base`` is the general dictionary (a :class:`FileLexicon`), ``extra`` the
    strong additions (proper nouns of the knowledge base) and ``author`` the
    weak ones (words the author confirmed on their own pages).

    The three verdicts matter to the UI: a word the general dictionary does not
    have, but the author's own pages do, is *probably* one of their names — or a
    typo they confirmed once. Such a word is not flagged as an unknown word, but
    it is marked weakly so a self-confirmed mistake can still be caught.

    Unknown words in the base table are reported as *known* here: a number, an
    abbreviation or a mixed token is not something the user can fix by editing
    a word box.
    """

    def __init__(
        self,
        base: WordList | None = None,
        extra: Iterable[str] = (),
        *,
        author: Iterable[str] = (),
        available: bool | None = None,
    ):
        self._base = base
        self._extra = frozenset(extra)
        self._author = frozenset(author)
        if available is None:
            available = base is not None and bool(getattr(base, "is_available", True))
        self._available = bool(available)

    @property
    def is_available(self) -> bool:
        return self._available

    def verdict(self, word: str) -> str:
        """Where the word is known: ``known``, ``author`` or ``unknown``."""
        if not word:
            return KNOWN
        # numbers, dates and abbreviations are not dictionary words
        if not any(character.isalpha() for character in word):
            return KNOWN
        if any(character.isdigit() for character in word):
            return KNOWN
        # word lists start at two letters, while "и", "в", "с", "к", "у", "а",
        # "я" are perfectly good one-letter words; flagging them would only add
        # noise (a lone letter is never something the user edits)
        if len(word) == 1:
            return KNOWN
        if "-" in word:
            parts = [part for part in word.split("-") if part]
            if parts:
                verdicts = [self.verdict(part) for part in parts]
                if all(verdict != UNKNOWN for verdict in verdicts):
                    # a compound is only "author known" when a part is
                    return AUTHOR if AUTHOR in verdicts else KNOWN
        forms = word_variants(word)
        if self._base is not None and any(form in self._base for form in forms):
            return KNOWN
        if self._extra and any(form in self._extra for form in forms):
            return KNOWN
        if self._author and any(form in self._author for form in forms):
            return AUTHOR
        return UNKNOWN

    def is_known(self, word: str) -> bool:
        return self.verdict(word) != UNKNOWN

    def is_strongly_known(self, word: str) -> bool:
        """Known without counting the author's own confirmed pages."""
        return self.verdict(word) == KNOWN

    def is_author_only(self, word: str) -> bool:
        """Known only because the author confirmed it on one of their pages."""
        return self.verdict(word) == AUTHOR
