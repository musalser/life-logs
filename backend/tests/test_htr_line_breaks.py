"""Words the diarist broke over a line break.

The author's vocabulary used to collect both halves as if they were words
(«слуша-» + «лась», «звакуи» + «ровать»), which taught the OOV highlighting that
a fragment is vocabulary. These tests pin the rules that decide otherwise, and
the two signals they use: the general dictionary confirms the *join*, while the
corpus itself (a word seen inside a line) confirms that a half is a word in its
own right.
"""
from __future__ import annotations

from app.htr.domain.text import confusion_variants, line_break_analysis


def analyse(pages, known, words=None):
    known_set = set(known)
    word_set = set(words) if words is not None else known_set
    return line_break_analysis(pages, known_set.__contains__, is_word=word_set.__contains__)


def test_hyphen_split_is_joined_and_both_halves_drop_out():
    analysis = analyse([["она часто слуша-", "лась музыки"]], known={"слушалась"})

    assert [join.word for join in analysis.joins] == ["слушалась"]
    assert analysis.fragments == {"слуша-", "лась"}
    assert analysis.words == {"слушалась"}


def test_split_without_a_hyphen_is_found_through_the_dictionary():
    analysis = analyse(
        [["дом ремонтирова", "лись всем селом"]], known={"ремонтировались"}
    )

    assert analysis.fragments == {"ремонтирова", "лись"}
    assert analysis.words == {"ремонтировались"}


def test_a_recognition_slip_at_the_seam_is_still_a_split():
    """Measured on page 26: «зажитог» + «ным» is зажиточным, not two words."""
    analysis = analyse([["а дед был зажитог", "ным тогда"]], known={"зажиточным"})

    assert [join.word for join in analysis.joins] == ["зажитогным"]
    assert analysis.joins[0].fuzzy is True
    assert analysis.fragments == {"зажитог", "ным"}
    # the joined form carries the recognition error, so it is not vocabulary
    assert analysis.words == frozenset()


def test_a_recognition_slip_at_the_start_of_the_word_is_found_too():
    """«эвакуи|ровать» was read as «звакуи|ровать»."""
    analysis = analyse([["их всех звакуи", "ровать решили"]], known={"эвакуировать"})

    assert analysis.fragments == {"звакуи", "ровать"}
    assert analysis.words == frozenset()


def test_two_words_across_a_break_are_left_alone():
    analysis = analyse([["скот гоняли на пастбища", "стадо вернулось"]], known={"стадо"})

    assert analysis.joins == ()
    assert analysis.fragments == frozenset()


def test_word_seen_inside_a_line_keeps_its_place_in_the_vocabulary():
    """«ямы» + «во» is not «ямыво», however the dictionary answers about it."""
    analysis = analyse(
        [["мы копали ямы", "во дворе"]],
        known={"ямыво"},
        words={"ямы", "во"},
    )

    assert analysis.joins == ()
    assert analysis.fragments == frozenset()


def test_the_head_of_the_next_line_is_kept_when_it_is_a_word():
    """A hyphen may also break a compound: «домо-» + «строительном»."""
    analysis = analyse(
        [["работал в домо-", "строительном техникуме"]],
        known={"домостроительном", "строительном"},
    )

    assert analysis.words == {"домостроительном"}
    assert analysis.fragments == {"домо-"}, "строительном — настоящее слово"


def test_pages_never_join_to_each_other():
    analysis = analyse([["конец строки"], ["начало страницы"]], known={"строкиначало"})

    assert analysis.joins == ()


def test_short_and_punctuation_only_tokens_are_ignored():
    analysis = analyse([["он сказал и", "о поехал"]], known={"ио"})

    assert analysis.joins == ()


def test_confusions_can_be_restricted_to_the_seam():
    word = "пастбищастадо"

    everywhere = confusion_variants(word)
    seam_only = confusion_variants(word, positions=(0, len("пастбища") - 1, len("пастбища")))

    assert len(seam_only) < len(everywhere)
    assert "настбищастадо" in everywhere
    assert set(seam_only) <= set(everywhere)
