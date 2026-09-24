"""Turning transcriptions into dictionary forms."""
from __future__ import annotations

from ...domain.text import iter_words, word_variants


def words_of(text: str | None) -> set[str]:
    """All dictionary forms a text contributes (used to extend the lexicon)."""
    forms: set[str] = set()
    for word in iter_words(text):
        if any(character.isalpha() for character in word):
            forms.update(word_variants(word))
    return forms
