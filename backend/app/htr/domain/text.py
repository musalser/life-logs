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

from dataclasses import dataclass, field
from difflib import SequenceMatcher

import re
import unicodedata
from collections.abc import Callable, Iterable, Sequence
from functools import lru_cache

# Unicode combining marks (category Mn): the codec emits a *decomposed* "й" as
# "и" + U+0306, and a stray stress mark sometimes lands mid-word. They belong to
# the word around them, so the tokenizer keeps them and ``normalize_word`` either
# composes them (NFC) or drops what cannot be composed.
_COMBINING_MARKS = "\u0300-\u036f\u0483-\u0489\u2de0-\u2dff\ua674-\ua67f\ufe20-\ufe2f"

# Words as the recognizer and the user produce them: letters/digits plus the
# hyphen ("кто-то"), the apostrophe some hands use and the combining marks above.
# Without the marks "строгий" (written as "и" + breve) would tokenise as
# "строгии" -- a word the dictionary has never seen.
WORD_RE = re.compile(rf"[\w'’\-{_COMBINING_MARKS}]+", re.UNICODE)

#: A combining mark that survived NFC: nothing to compose it with, so it is an
#: artifact of recognition (a stray stress mark) rather than part of the word.
_LOOSE_MARK_RE = re.compile(rf"[{_COMBINING_MARKS}]")

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


def alignment_preserved(word_count: int, text: str | None) -> bool:
    """True when the stored word boxes still map 1:1 onto the text's words.

    Word boxes are built from the *whitespace* tokens of the recognised line, so
    the same tokenisation is compared here. A rewrite that keeps the number of
    words — fixing a typo, adding a comma («еврея» → «еврея,»), or accepting a
    word alternative — leaves box *i* pointing at word *i*, so the geometry
    stays usable and the line must not be flagged as stale. A split or a merge
    («не знаю» → «незнаю», a deleted word) shifts every following box, which is
    exactly what ``words_stale`` warns about.
    """
    if word_count <= 0:
        return True
    return len((text or "").split()) == word_count


def normalize_word(word: str) -> str:
    """Canonical dictionary form of one word: NFC, casefold, archaic letters, no ъ.

    Both the dictionary and the lookup go through this function, so a
    dictionary built from a modern word list matches old spellings such as
    «домъ» or «всѣ».

    Unicode normalization comes first because the codec emits decomposed text:
    «строгий» arrives as "строги" + "и" + U+0306, and only NFC turns that back
    into the composed «строгий» the word list holds. A mark that still has
    nothing to compose with after NFC is a stray stress/artifact mark and is
    dropped instead of being glued to the lookup key.
    """
    text = unicodedata.normalize("NFC", word).casefold().translate(_ARCHAIC_LETTERS)
    text = _LOOSE_MARK_RE.sub("", text)
    if text.endswith("ъ"):
        text = text[:-1]
    return text


@lru_cache(maxsize=_VARIANTS_CACHE_SIZE)
def word_variants(word: str) -> frozenset[str]:
    """Every dictionary form a word may legitimately match.

    Contains the plain folded spelling, the archaic-letter mapping, the form
    without a word-final hard sign and the modern endings of the old ones. A
    word counts as known when *any* variant is in the dictionary.

    The plain spelling is normalized too, so a decomposed «строгий» carries the
    composed «строгий» the word list actually holds.
    """
    forms = {unicodedata.normalize("NFC", word).casefold()}
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


# ---------------------------------------------------------------------------
# words the writer broke at the end of a line
# ---------------------------------------------------------------------------

#: One-character confusions the recognizer actually makes. They are used only to
#: *recognise* a line-break join when the dictionary misses it — never to rewrite
#: text. Measured on the corpus: the diarist's «эвакуи|ровать» came back as
#: «звакуи|ровать», so the exact join is not in any dictionary.
_JOIN_CONFUSIONS: dict[str, str] = {
    "а": "о",
    "о": "а",
    "е": "иэ",
    "и": "ейы",
    "й": "и",
    "ы": "и",
    "з": "эс",
    "э": "з",
    "с": "з",
    "т": "г",
    "г": "тч",
    "ч": "г",
    "д": "т",
    "б": "в",
    "в": "б",
    "н": "п",
    "п": "н",
    "ш": "щ",
    "щ": "ш",
    "ь": "и",
}


def confusion_variants(
    word: str, *, positions: Iterable[int] | None = None, limit: int = 400
) -> list[str]:
    """``word`` plus one-position confusions of it (for a fuzzy dictionary check).

    ``positions`` restricts the confusions to given indices. Line-break detection
    passes the seam (where a broken word continues) and the first character (the
    model reads «эвакуи» as «звакуи»): anywhere else a "confusion" is a random
    string that a Bloom filter answers "yes" to about once in 2500 lookups, and
    every such hit would invent a word the writer never broke.
    """
    variants = [word]
    indices = range(len(word)) if positions is None else positions
    for index in indices:
        if not 0 <= index < len(word):
            continue
        char = word[index]
        for replacement in _JOIN_CONFUSIONS.get(char, ""):
            if len(variants) >= limit:
                return variants
            variants.append(word[:index] + replacement + word[index + 1 :])
    return variants


@dataclass(frozen=True)
class LineBreakJoin:
    """A word the writer split over a line break."""

    #: the half that is not a word of its own
    fragment: str
    #: first word of the following line
    head: str
    #: the joined form worth adding to the vocabulary
    word: str
    hyphenated: bool
    #: the exact join is not a dictionary word; a one-letter confusion is
    fuzzy: bool = False


@dataclass(frozen=True)
class LineBreakAnalysis:
    joins: tuple[LineBreakJoin, ...] = ()
    #: forms that exist only because of a break and must not become vocabulary
    fragments: frozenset[str] = field(default_factory=frozenset)

    @property
    def words(self) -> frozenset[str]:
        """Joined forms that are certain enough to add as the author's words."""
        return frozenset(join.word for join in self.joins if not join.fuzzy)


def line_break_analysis(
    pages: Sequence[Sequence[str]],
    is_known: Callable[[str], bool],
    *,
    is_word: Callable[[str], bool] | None = None,
    min_fragment_length: int = 2,
) -> LineBreakAnalysis:
    """Find words split across line breaks, decided by the general dictionary.

    ``pages`` is a sequence of pages, each a sequence of line texts in reading
    order; a page never joins to the next one. The last word of a line and the
    first word of the following line are a split when they make up a dictionary
    word: with the hyphen removed (``слуша-`` + ``лась``) or through a single
    confusion (``звакуи`` + ``ровать`` → эвакуировать).

    Only the halves that are **not words on their own** are reported as
    fragments — a line may legitimately end with «и» or «был», and such a word
    stays in the vocabulary. A fragment is never rewritten in the text: this only
    decides what the dictionary and the OOV highlighting consider a word. For a
    fuzzy join nothing is added either, because the joined form carries the
    recognition error that hid it from the dictionary in the first place.
    """
    is_word = is_word or is_known
    joins: list[LineBreakJoin] = []
    fragments: set[str] = set()

    for lines in pages:
        tokens_per_line: list[tuple[list[str], list[str]]] = []
        for text in lines:
            raw = [token for token in iter_words(text) if any(c.isalpha() for c in token)]
            normalized = [normalize_word(token) for token in raw]
            tokens_per_line.append((raw, normalized))

        for index in range(len(tokens_per_line) - 1):
            raw_tail, tail_words = tokens_per_line[index]
            _raw_head, head_words = tokens_per_line[index + 1]
            if not tail_words or not head_words:
                continue
            tail = tail_words[-1]
            head = head_words[0]
            if len(tail) < min_fragment_length or len(head) < min_fragment_length:
                continue
            hyphenated = raw_tail[-1].rstrip().endswith(("-", "‐", "–"))

            stripped = tail.rstrip("-‐–")
            candidate = stripped + head
            joined: str | None = None
            fuzzy = False
            if hyphenated:
                # the hyphen is the writer saying "this word continues", so an
                # exact dictionary word is enough. Nothing fuzzy here: a Bloom
                # filter answers "yes" to roughly one string in a hundred, which
                # would invent joins such as "ездили-стелили" + "лежневку"
                if is_known(candidate):
                    joined = candidate
                elif is_known(tail + head):
                    # a real hyphenated compound broken after its hyphen
                    joined = tail + head
            elif not is_word(tail) and not is_word(head):
                # no hyphen: the pair has to look like one broken word, so
                # neither half may be a word of its own — "ямы" + "во" and
                # "она" + "не" are two words, not "ямыво"/"онане", whatever the
                # dictionary says about their concatenation. What they make must
                # be a dictionary word, allowing the one-letter slip (of the
                # recognizer, not of the dictionary) that hid it.
                if is_known(candidate):
                    joined = candidate
                else:
                    seam = len(stripped)
                    variants = confusion_variants(candidate, positions=(0, seam - 1, seam))
                    if any(is_known(variant) for variant in variants):
                        joined = candidate
                        fuzzy = True
            if joined is None:
                continue

            split = LineBreakJoin(
                fragment=tail, head=head, word=joined, hyphenated=hyphenated, fuzzy=fuzzy
            )
            joins.append(split)
            for half, is_head in ((tail, False), (head, True)):
                if len(half) < min_fragment_length or is_word(half):
                    continue
                # the head of the next line is a whole word more often than not
                # ("пастбища" + "стадо" is two words, not one): only call it a
                # fragment when the dictionary does not know it either
                if is_head and is_known(half):
                    continue
                fragments.add(half)

    return LineBreakAnalysis(joins=tuple(joins), fragments=frozenset(fragments))
