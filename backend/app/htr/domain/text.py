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

from difflib import SequenceMatcher

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

def align_word_alternatives(
    chosen_words: list[str],
    hypotheses: list[tuple[list[str], float]],
    limit: int = 5,
) -> dict[int, list[tuple[str, float]]]:
    """Other readings of each chosen word, taken from the line-level N-best.

    ``hypotheses`` is ``[(words, score), ...]`` ordered best-first, the first
    entry being the chosen reading itself. Words are matched with
    :class:`difflib.SequenceMatcher`, so hypotheses with a different word count
    (a merged ``на зывалось`` vs ``называлось``, a split, an extra token) still
    contribute: in a block whose lengths differ, the unpaired words of the
    hypothesis are offered as alternatives for the first word of the block.

    Returns ``{word_index: [(text, score), ...]}`` with alternatives sorted
    best-first, deduplicated by text and without the chosen reading.
    """
    if not chosen_words or limit <= 0:
        return {}
    alternatives: dict[int, list[tuple[str, float]]] = {}
    for words, score in hypotheses[1:]:
        if not words:
            continue
        matcher = SequenceMatcher(None, chosen_words, words, autojunk=False)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            pairs = min(i2 - i1, j2 - j1)
            for offset in range(pairs):
                index = i1 + offset
                candidate = words[j1 + offset]
                if candidate and candidate != chosen_words[index]:
                    alternatives.setdefault(index, []).append((candidate, score))
            # lengths differ inside the block: a split of a chosen word (the
            # hypothesis carries more words than the chosen reading) or a merge
            # (fewer). The unpaired words become alternatives for the nearest
            # chosen word — for a split that is the word being split, even when
            # the block sits between two chosen words.
            if j2 - j1 > pairs:
                target: int | None
                if i2 > i1:
                    target = i1
                elif i1 < len(chosen_words):
                    target = i1
                elif chosen_words:
                    target = len(chosen_words) - 1
                else:
                    target = None
                if target is not None:
                    for extra in words[j1 + pairs : j2]:
                        if extra and extra != chosen_words[target]:
                            alternatives.setdefault(target, []).append((extra, score))

    result: dict[int, list[tuple[str, float]]] = {}
    for index, items in alternatives.items():
        seen: set[str] = set()
        best: list[tuple[str, float]] = []
        for text, score in sorted(items, key=lambda item: item[1], reverse=True):
            # equality, not containment: "на" is a legitimate alternative for
            # the split "называлось" even though it is a substring of it
            if text in seen or text == chosen_words[index]:
                continue
            seen.add(text)
            best.append((text, score))
            if len(best) >= limit:
                break
        if best:
            result[index] = best
    return result

def map_spans_to_reference(
    display: str,
    reference: str,
    spans: list[tuple[int, int]],
) -> list[tuple[int, int] | None]:
    """Character ranges of ``reference`` behind each span of ``display``.

    Used to keep word geometry when the beam's text and kraken's own greedy text
    differ. The two are aligned at the *character* level, which makes the
    mapping work for every divergence at once: a merge (``где -то`` →
    ``где-то``), a split (``называлось`` → ``на зывалось``), a letter change or
    an inserted/removed token. A span with no counterpart in the reference (the
    beam inserted characters greedy never had) maps to ``None``; the caller then
    loses the geometry of that single word instead of the whole line's.
    """
    if not spans:
        return []
    if display == reference:
        return [(start, end) for start, end in spans]

    blocks = SequenceMatcher(None, display, reference, autojunk=False).get_opcodes()
    mapped: list[tuple[int, int] | None] = []
    for start, end in spans:
        low: int | None = None
        high: int | None = None
        for tag, i1, i2, j1, j2 in blocks:
            if tag == "insert":
                # display characters with no counterpart in the reference
                continue
            if tag == "delete":
                # reference characters with no counterpart in the display: they
                # belong to whichever word sits at that position
                if start <= i1 <= end:
                    low = j1 if low is None else min(low, j1)
                    high = j2 if high is None else max(high, j2)
                continue
            lo, hi = max(i1, start), min(i2, end)
            if lo >= hi:
                continue
            if tag == "equal":
                # one-to-one: the offset inside the block is the same on both sides
                piece = (j1 + (lo - i1), j1 + (hi - i1))
            else:  # replace: map the overlap proportionally
                width = i2 - i1
                ref_width = j2 - j1
                piece = (
                    j1 + round((lo - i1) * ref_width / width),
                    j1 + round((hi - i1) * ref_width / width),
                )
            if piece[1] <= piece[0]:
                continue
            low = piece[0] if low is None else min(low, piece[0])
            high = piece[1] if high is None else max(high, piece[1])
        mapped.append((low, high) if low is not None and high is not None and high > low else None)
    return mapped
