"""Word-level diff between the raw recognition and the model's proposal.

The proposal is stored as a whole line, but a human reviews *changes*: each one
can be applied on its own, because a user who likes «очен → очень» may well not
want the «Судиславль» the model silently dropped.

Every change carries a dictionary verdict:

* ``in_lexicon = True`` — the new word exists in the general dictionary or in
  the author's own vocabulary (their confirmed pages, knowledge base terms);
* ``False`` — nobody has ever seen that word; this is where a language model
  invents things, so such a change is never applied in bulk;
* ``None`` — no dictionary is installed (or the change removes a word, which no
  dictionary can vouch for).
"""
from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher

from ..domain.entities import SuggestionChange
from ..domain.interfaces import LexiconChecker
from ..domain.text import iter_word_spans, iter_words

#: a word needs no space after these (opening bracket, opening quote)
_OPENERS = "([{\"'«„"
#: and none before these (closing bracket, quote, punctuation)
_ATTACHED = ")]}\"'»“.,;:!?…"


@dataclass(frozen=True)
class _WordChange:
    """A change plus which tokens of the *current* line it replaces."""

    before: str
    after: str
    first_token: int
    last_token: int  # exclusive


def _same_letters(left: list[str], right: list[str]) -> bool:
    """True when the two token runs differ only in where the spaces fell."""
    return "".join(left).casefold() == "".join(right).casefold()


def _word_ops(before: str, after: str) -> list[_WordChange]:
    left = iter_word_spans(before)
    left_words = [word for _, _, word in left]
    right_words = [word for _, _, word in iter_word_spans(after)]
    if left_words == right_words:
        return []

    ops: list[_WordChange] = []
    matcher = SequenceMatcher(a=left_words, b=right_words, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        # a block with a different token count on both sides is usually a word
        # the segmenter glued or split: "на зывалось" -> "называлось". When the
        # letters really are the same, that is one change the user can accept
        # with a single click — and one the dictionary can verify.
        if (i2 - i1) != (j2 - j1) and _same_letters(left_words[i1:i2], right_words[j1:j2]):
            ops.append(
                _WordChange(
                    " ".join(left_words[i1:i2]), " ".join(right_words[j1:j2]), i1, i2
                )
            )
            continue
        # otherwise pair the block positionally; what is left over on either
        # side is a real deletion/insertion and must be reported, otherwise a
        # proposal could silently drop a word and still count as "verified"
        paired = min(i2 - i1, j2 - j1)
        for offset in range(paired):
            ops.append(
                _WordChange(left_words[i1 + offset], right_words[j1 + offset], i1 + offset, i1 + offset + 1)
            )
        for index in range(i1 + paired, i2):
            ops.append(_WordChange(left_words[index], "", index, index + 1))
        insert_at = i1 + paired
        for word in right_words[j1 + paired : j2]:
            # an insertion sits between two tokens: its span is empty
            ops.append(_WordChange("", word, insert_at, insert_at))
    return ops


def word_changes(before: str, after: str) -> list[tuple[str, str]]:
    """Word replacements turning ``before`` into ``after`` (in reading order).

    Pure punctuation edits produce no pairs: there is nothing to verify, and
    the caller still sees that the two strings differ.
    """
    return [(op.before, op.after) for op in _word_ops(before, after)]


def build_changes(
    before: str,
    after: str,
    checker: LexiconChecker | None = None,
) -> list[SuggestionChange]:
    """Changes with character offsets and the dictionary verdict.

    Inserting or replacing a word is checkable (the new word either exists or
    not); *removing* a word is not, so a deletion stays unverified and the user
    decides on it.
    """
    available = bool(checker is not None and checker.is_available)
    spans = iter_word_spans(before)
    changes: list[SuggestionChange] = []
    for op in _word_ops(before, after):
        verdict: bool | None = None
        if available and op.after and checker is not None:
            # a split produces several words ("наеврея" -> "на еврея"): every
            # one of them has to be known. The checker itself knows that
            # numbers, dates and one-letter words are not dictionary entries.
            words = iter_words(op.after) or [op.after]
            verdict = all(checker.is_known(word) for word in words)
        if op.first_token < len(spans):
            start = spans[op.first_token][0]
        else:
            start = len(before)
        if 0 < op.last_token <= len(spans):
            end = spans[op.last_token - 1][1]
        else:
            end = start
        if op.first_token == op.last_token and op.after and start < len(before):
            # pure insertion before an existing word: keep the word itself
            end = start
        changes.append(
            SuggestionChange(
                before=op.before, after=op.after, in_lexicon=verdict, start=start, end=end
            )
        )
    return changes


def apply_change(text: str, changes: list[SuggestionChange], index: int) -> str | None:
    """Text with a single change applied, keeping everything else as it was.

    ``None`` when there is no such change — the caller turns that into a 404,
    because the client referred to a diff that no longer exists.
    """
    if index < 0 or index >= len(changes):
        return None
    change = changes[index]
    start = max(0, min(change.start, len(text)))
    end = max(start, min(change.end, len(text)))
    if not change.after:
        return _remove(text, start, end)
    return _replace(text, start, end, change.after)


def _join(left: str, right: str) -> str:
    """Glue two fragments back together with the smallest sane whitespace."""
    if not left or not right:
        return left + right
    if left[-1].isspace() or right[0].isspace():
        return left + right
    if left[-1] in _OPENERS or right[0] in _ATTACHED:
        return left + right
    return f"{left} {right}"


def _remove(text: str, start: int, end: int) -> str:
    return _join(text[:start].rstrip(), text[end:].lstrip())


def _replace(text: str, start: int, end: int, word: str) -> str:
    # the replacement is a word: it separates from its neighbours like one
    left = _join(text[:start].rstrip(), word)
    return _join(left, text[end:].lstrip())
