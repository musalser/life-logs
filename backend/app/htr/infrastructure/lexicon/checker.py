"""Layered vocabulary check: general dictionary + the author's own words."""
from __future__ import annotations

from typing import Iterable

from ...domain.interfaces import WordList
from ...domain.text import word_variants


class LayeredLexiconChecker:
    """Answers "is this recognized word known?" for one author.

    ``base`` is the general dictionary (a :class:`FileLexicon`), ``extra`` the
    normalized forms taken from the author's confirmed pages and knowledge base.
    Unknown words in the base table are reported as *known* here: a number, an
    abbreviation or a mixed token is not something the user can fix by editing
    a word box.
    """

    def __init__(
        self,
        base: WordList | None = None,
        extra: Iterable[str] = (),
        *,
        available: bool | None = None,
    ):
        self._base = base
        self._extra = frozenset(extra)
        if available is None:
            available = base is not None and bool(getattr(base, "is_available", True))
        self._available = bool(available)

    @property
    def is_available(self) -> bool:
        return self._available

    def is_known(self, word: str) -> bool:
        if not word:
            return True
        # numbers, dates and abbreviations are not dictionary words
        if not any(character.isalpha() for character in word):
            return True
        if any(character.isdigit() for character in word):
            return True
        # word lists start at two letters, while "и", "в", "с", "к", "у", "а",
        # "я" are perfectly good one-letter words; flagging them would only add
        # noise (a lone letter is never something the user edits)
        if len(word) == 1:
            return True
        if "-" in word:
            parts = [part for part in word.split("-") if part]
            if parts and all(self.is_known(part) for part in parts):
                return True
        forms = word_variants(word)
        if self._extra and any(form in self._extra for form in forms):
            return True
        if self._base is None:
            return False
        return any(form in self._base for form in forms)
