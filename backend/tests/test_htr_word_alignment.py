"""The rule that decides whether a page's word layout is still valid.

Word boxes come from the whitespace tokens of the recognised line, so a rewrite
that keeps the number of words leaves box *i* on word *i* and must not raise the
«разметка устарела» flag. Only a split, a merge or a deletion shifts the boxes.
"""
from __future__ import annotations

from app.htr.domain.text import alignment_preserved


def test_same_word_count_keeps_the_alignment():
    assert alignment_preserved(3, "Уж очень дед") is True
    # a spelling fix, a capital and a comma are all same-length rewrites
    assert alignment_preserved(3, "Уж очень дед,") is True
    assert alignment_preserved(2, "На еврея") is True
    assert alignment_preserved(5, "Уж очень дед был похож") is True


def test_split_merge_and_deletion_break_the_alignment():
    assert alignment_preserved(2, "незнаю") is False        # merged
    assert alignment_preserved(1, "не знаю") is False       # split
    assert alignment_preserved(3, "Уж очень") is False      # a word was dropped
    assert alignment_preserved(2, "") is False              # the line was cleared


def test_a_line_without_word_boxes_is_never_stale():
    # nothing to align: the flag would only produce a meaningless badge
    assert alignment_preserved(0, "любой текст") is True
    assert alignment_preserved(0, "") is True
