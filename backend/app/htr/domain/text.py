"""Plain-text helpers shared by vocabulary checking and dictionary building.

Pure Python on purpose: the application layer (page annotation), the
infrastructure layer (dictionary provisioning) and the CLI scripts all need the
same notion of "a word", so it lives in the domain instead of being duplicated.

The manuscripts of this project are Russian and often pre-1918, while every
available word list is modern. ``normalize_word`` and :func:`word_variants`
therefore map old spellings onto their modern counterparts *as extra lookup
candidates* — a word is accepted when any candidate is in the dictionary, so an
aggressive rule can only ever turn a red flag into a green one, never the other
way round.
"""
from __future__ import annotations

import re
from functools import lru_cache

# Words as the recognizer and the user produce them: letters/digits plus the
# hyphen ("кто-то") and the apostrophe some hands use.
WORD_RE = re.compile(r"[\w'’\-]+", re.UNICODE)

# Pre-1918 letters mapped to their modern equivalents (lowercase only:
# normalize_word folds the case first).
_ARCHAIC_LETTERS = str.maketrans({"ѣ": "е", "і": "и", "ѳ": "ф", "ѵ": "и"})

# Old endings that survive into modern spelling. Applied only to build extra
# lookup candidates, never to rewrite what the recognizer produced.
_ENDING_RULES: tuple[tuple[str, str], ...] = (
    ("аго", "ого"),  # новаго -> нового
    ("яго", "его"),  # синяго -> синего
    ("ыя", "ые"),    # новыя -> новые
    ("ія", "ие"),    # синія -> синие (after the і -> и mapping)
    ("ия", "ие"),
    ("ея", "её"),    # ея -> её
)

_VARIANTS_CACHE_SIZE = 200_000


def iter_word_spans(text: str | None) -> list[tuple[int, int, str]]:
    """Tokenize a transcription into ``(start, end, word)`` character spans.

    The offsets let a single proposed change be applied to the text without
    rebuilding it from tokens — punctuation and the author's spacing survive.
    """
    if not text:
        return []
    return [(match.start(), match.end(), match.group()) for match in WORD_RE.finditer(text)]


def iter_words(text: str | None) -> list[str]:
    """Tokenize a transcription into words (punctuation is dropped)."""
    return [word for _, _, word in iter_word_spans(text)]


def normalize_word(word: str) -> str:
    """Canonical dictionary form of one word: casefold, archaic letters, no ъ.

    Both the dictionary and the lookup go through this function, so a
    dictionary built from a modern word list matches old spellings such as
    «домъ» or «всѣ».
    """
    text = word.casefold().translate(_ARCHAIC_LETTERS)
    if text.endswith("ъ"):
        text = text[:-1]
    return text


@lru_cache(maxsize=_VARIANTS_CACHE_SIZE)
def word_variants(word: str) -> frozenset[str]:
    """Every dictionary form a word may legitimately match.

    Contains the plain folded spelling, the archaic-letter mapping, the form
    without a word-final hard sign and the modern endings of the old ones. A
    word counts as known when *any* variant is in the dictionary.
    """
    forms = {word.casefold()}
    forms.add(normalize_word(word))
    for form in tuple(forms):
        if form.endswith("ъ"):
            forms.add(form[:-1])
    for form in tuple(forms):
        for old, new in _ENDING_RULES:
            if form.endswith(old):
                forms.add(form[: -len(old)] + new)
    return frozenset(forms)
