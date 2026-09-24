"""Unit tests of the model-proposal diff and its dictionary verdicts.

The proposal itself is a whole line; what a human reviews is the *change*.
Getting that list right decides whether the bulk accept is safe, so these
tests pin down replacements, insertions, deletions and the meaning of
``in_lexicon``.
"""
from __future__ import annotations

import pytest

from app.htr.application.suggestions import apply_change, build_changes, word_changes
from app.htr.domain.entities import BoundingBox, LineView, SuggestionChange
from app.htr.infrastructure.lexicon import LayeredLexiconChecker


def checker_with(*words: str) -> LayeredLexiconChecker:
    return LayeredLexiconChecker(base=frozenset(words))


def make_line(predicted: str, suggested: str | None, changes=None) -> LineView:
    return LineView(
        id=1,
        order=0,
        bbox=BoundingBox(0, 0, 10, 10),
        predicted_text=predicted,
        suggested_text=suggested,
        suggestion_changes=list(changes or []),
    )


# ---------------------------------------------------------------------------
# diff
# ---------------------------------------------------------------------------


def test_word_changes_reports_replacements_in_reading_order():
    assert word_changes("Уж очен дед был похож", "Уж очень дед был похож") == [("очен", "очень")]
    assert word_changes("и звали с нили искупаеся", "и звали с нили искупаться") == [
        ("искупаеся", "искупаться")
    ]


def test_word_changes_reports_insertions_and_deletions():
    assert word_changes("на еврея", "на еврея, Захарьевка") == [("", "Захарьевка")]
    assert word_changes("на еврея Захарьевка", "на еврея") == [("Захарьевка", "")]


def test_word_changes_ignores_punctuation_only_edits():
    # nothing to verify in a comma, but the caller still sees the texts differ
    assert word_changes("на еврея", "на еврея,") == []
    assert word_changes("одно и то же", "одно и то же") == []


def test_word_changes_pairs_a_rewritten_block_positionally():
    changes = word_changes("дажее с тумбы", "даже с тумбы высоко")
    assert ("дажее", "даже") in changes


def test_word_changes_reports_what_a_rewritten_block_drops():
    """Regression: a shorter replacement must not hide deleted words.

    On a real page the model rewrote
    "пазиі. организоцни и. Судиславль" into "партийной организации." — the
    dropped words have to show up, otherwise the proposal counts as fully
    verified and the bulk accept would silently delete them.
    """
    changes = word_changes(
        "был секретарем пазиі. организоцни и. Судиславль",
        "был секретарем партийной организации.",
    )
    assert changes == [
        ("пазиі", "партийной"),
        ("организоцни", "организации"),
        ("и", ""),
        ("Судиславль", ""),
    ]


# ---------------------------------------------------------------------------
# dictionary verdicts
# ---------------------------------------------------------------------------


def test_build_changes_marks_a_replacement_found_in_the_dictionary():
    changes = build_changes("Уж очен дед", "Уж очень дед", checker_with("уж", "очень", "дед"))
    assert [(c.before, c.after, c.in_lexicon) for c in changes] == [("очен", "очень", True)]
    # the offsets point at the word inside the line, so one click can splice it
    assert "Уж очен дед"[changes[0].start : changes[0].end] == "очен"


def test_build_changes_marks_an_invented_word_as_unknown():
    changes = build_changes("на еврея", "на еврея, Захарьевка", checker_with("на", "еврея"))
    assert [(c.before, c.after, c.in_lexicon) for c in changes] == [("", "Захарьевка", False)]
    assert changes[0].start == changes[0].end == len("на еврея")


def test_build_changes_leaves_everything_unverified_without_a_dictionary():
    changes = build_changes("Уж очен дед", "Уж очень дед", None)
    assert changes[0].in_lexicon is None


def test_build_changes_treats_a_deletion_as_unverifiable():
    changes = build_changes("на еврея Захарьевка", "на еврея", checker_with("на", "еврея"))
    assert [(c.before, c.after, c.in_lexicon) for c in changes] == [("Захарьевка", "", None)]


def test_build_changes_accepts_numbers_without_a_dictionary_hit():
    changes = build_changes("с 1д46 годе", "с 1946 годе", checker_with())
    assert changes[0].in_lexicon is True  # digits are not dictionary words


# ---------------------------------------------------------------------------
# LineView verdict used by the UI and by the bulk accept
# ---------------------------------------------------------------------------


def test_line_without_a_proposal_is_not_a_suggestion():
    assert make_line("дед", None).has_suggestion is False
    assert make_line("дед", "дед").has_suggestion is False


def test_line_suggestion_verified_only_when_every_change_is_confirmed():
    verified = make_line("дед", "дед был", [SuggestionChange("", "был", True)])
    assert verified.has_suggestion is True
    assert verified.suggestion_verified is True

    unknown = make_line("дед", "дед был", [SuggestionChange("", "был", False)])
    assert unknown.suggestion_verified is False

    unchecked = make_line("дед", "дед был", [SuggestionChange("", "был", None)])
    assert unchecked.suggestion_verified is False

    mixed = make_line(
        "дед",
        "дед был",
        [SuggestionChange("", "был", True), SuggestionChange("", "Захарьевка", False)],
    )
    assert mixed.suggestion_verified is False


def test_punctuation_only_proposal_is_not_bulk_accepted():
    # nothing to check against, so it is reviewed line by line
    line = make_line("на еврея", "на еврея,", [])
    assert line.has_suggestion is True
    assert line.suggestion_verified is False


def test_proposal_that_drops_a_word_is_not_bulk_accepted():
    line = make_line(
        "секретарем пазиі и. Судиславль",
        "секретарем партийной",
        [
            SuggestionChange("пазиі", "партийной", True),
            SuggestionChange("и", "", None),
            SuggestionChange("Судиславль", "", None),
        ],
    )
    assert line.suggestion_verified is False


def test_effective_text_still_prefers_the_user_over_the_model():
    line = make_line("уж очен", "уж очень")
    line.corrected_text = "уж очень дед"
    assert line.effective_text == "уж очень дед"
    # the raw prediction stays available for metrics and retraining
    assert line.predicted_text == "уж очен"


# ---------------------------------------------------------------------------
# applying one change on its own
# ---------------------------------------------------------------------------


def apply_one(before: str, after: str, index: int) -> str:
    return apply_change(before, build_changes(before, after), index)


def test_apply_change_replaces_a_single_word():
    assert apply_one("уж очен дед был", "уж очень дед был", 0) == "уж очень дед был"
    assert apply_one("Уж очен дед", "Уж очень дед", 0) == "Уж очень дед"


def test_apply_change_keeps_the_punctuation_of_the_original():
    # the model moved the full stop; taking one word must not move it
    assert apply_one("был секретарем пазиі. дальше", "был секретарем партийной дальше", 0) == (
        "был секретарем партийной. дальше"
    )


def test_apply_change_inserts_a_word_with_a_separator():
    assert apply_one("на еврея", "на еврея Захарьевка", 0) == "на еврея Захарьевка"


def test_apply_change_removes_a_word_without_leaving_double_spaces():
    assert apply_one("на еврея Захарьевка", "на еврея", 0) == "на еврея"
    assert apply_one("организцни и. Судиславль", "организцни. Судиславль", 0) == (
        "организцни. Судиславль"
    )


def test_apply_change_handles_a_bracket():
    assert apply_one("он (сторугі", "он (стоял", 0) == "он (стоял"


def test_apply_change_reports_an_index_that_is_gone():
    changes = build_changes("уж очен дед", "уж очень дед")
    assert apply_change("уж очен дед", changes, 1) is None
    assert apply_change("уж очен дед", changes, -1) is None


def test_apply_change_leaves_the_rest_of_the_proposal_reachable():
    """Taking one word of a two-word proposal keeps the other one offered."""
    before, after = "был секретарем пазиі. дальше", "был секретарем партийной дальше"
    once = apply_one(before, after, 0)
    assert once == "был секретарем партийной. дальше"
    remaining = build_changes(once, after)
    assert [(c.before, c.after) for c in remaining] == [(".", "")] or remaining == []


def test_merge_of_two_words_is_one_change():
    """The segmenter splitting a word is the most common HTR error."""
    changes = build_changes("на зывалось «Аленушкино»", "называлось «Аленушкино»")
    assert [(c.before, c.after) for c in changes] == [("на зывалось", "называлось")]
    assert changes[0].in_lexicon is None  # no checker passed
    assert apply_change("на зывалось «Аленушкино»", changes, 0) == "называлось «Аленушкино»"


def test_merge_is_verified_when_the_merged_word_is_known():
    changes = build_changes(
        "на зывалось «Аленушкино»", "называлось «Аленушкино»", checker_with("называлось")
    )
    assert changes[0].in_lexicon is True


def test_split_of_a_word_is_one_change():
    changes = build_changes("наеврея Захарьевка", "на еврея Захарьевка", checker_with("на", "еврея"))
    assert [(c.before, c.after) for c in changes] == [("наеврея", "на еврея")]
    assert changes[0].in_lexicon is True


def test_a_block_that_drops_a_word_is_not_a_merge():
    """Letters differ, so it is not a glued word: the lost word must show up."""
    changes = build_changes("пазиі. организцни и. Судиславль", "партийной организации.")
    assert [(c.before, c.after) for c in changes] == [
        ("пазиі", "партийной"),
        ("организцни", "организации"),
        ("и", ""),
        ("Судиславль", ""),
    ]
