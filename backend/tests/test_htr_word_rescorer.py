"""Unit tests of the second-pass (word-level) re-ranking and its guard."""
from __future__ import annotations

import pytest

from app.htr.infrastructure.kraken.beam import BeamCandidate
from app.htr.infrastructure.lm.word_rescorer import (
    RescoreConfig,
    WordRescorer,
)


class FakeScorer:
    """A word-level model as a table of sentence scores."""

    def __init__(self, scores: dict[str, float]):
        self.scores = scores
        self.calls: list[str] = []

    def score(self, sentence: str) -> float:
        self.calls.append(sentence)
        return self.scores.get(sentence, -100.0)


def candidate(text: str, score: float = -5.0, acoustic: float | None = None):
    return BeamCandidate(
        text=text,
        score=score,
        acoustic=acoustic if acoustic is not None else score,
        lm=0.0,
        words=0.0,
    )


def test_rescoring_picks_the_higher_word_score_within_the_weight():
    scorer = FakeScorer({"корова мычала": -3.0, "карова мычала": -9.0})
    rescorer = WordRescorer(scorer, RescoreConfig(weight=1.0, normalize_by_words=False))

    outcome = rescorer.choose([candidate("карова мычала", -4.0), candidate("корова мычала", -5.0)])

    assert outcome.text == "корова мычала"
    assert outcome.changed is True


def test_small_weight_leaves_the_beam_winner_alone():
    scorer = FakeScorer({"корова мычала": 0.0, "карова мычала": -100.0})
    rescorer = WordRescorer(scorer, RescoreConfig(weight=0.001, normalize_by_words=False))

    outcome = rescorer.choose([candidate("карова мычала", -4.0), candidate("корова мычала", -5.0)])

    assert outcome.text == "карова мычала"
    assert outcome.changed is False


def test_guard_keeps_a_confident_fully_known_top1():
    """A rare toponym must not be traded for a frequent word by a general LM."""
    scorer = FakeScorer({"в Судиславль": 0.0, "в судиславле": -100.0})
    rescorer = WordRescorer(
        scorer,
        RescoreConfig(weight=10.0, normalize_by_words=False),
        known_word=lambda word: word in {"в", "Судиславль", "судиславле"},
    )

    outcome = rescorer.choose(
        [candidate("в Судиславль", -1.0, acoustic=-1.0), candidate("в судиславле", -1.2, acoustic=-1.2)]
    )

    assert outcome.changed is False
    assert "fully known" in outcome.reason
    assert scorer.calls == [], "модель даже не должна запрашиваться"


def test_guard_allows_rescoring_when_the_top1_has_an_unknown_word():
    scorer = FakeScorer({"корова мычала": 0.0, "карова мычала": -50.0})
    rescorer = WordRescorer(
        scorer,
        RescoreConfig(weight=100.0, normalize_by_words=False),
        known_word=lambda word: word == "корова",
    )

    outcome = rescorer.choose(
        [candidate("карова мычала", -1.0, acoustic=-1.0), candidate("корова мычала", -1.5, acoustic=-1.5)]
    )

    assert outcome.changed is True
    assert "out of dictionary" in outcome.reason


def test_guard_allows_rescoring_when_the_top1_is_uncertain():
    scorer = FakeScorer({"корова мычала": 0.0, "карова мычала": -50.0})
    rescorer = WordRescorer(
        scorer,
        RescoreConfig(weight=100.0, normalize_by_words=False, min_mean_acoustic=-0.2),
        known_word=lambda word: True,
    )

    outcome = rescorer.choose(
        [candidate("карова мычала", -1.0, acoustic=-5.0), candidate("корова мычала", -1.1, acoustic=-5.1)]
    )

    assert outcome.changed is True
    assert "low confidence" in outcome.reason


def test_sentence_score_is_normalized_by_word_count():
    scorer = FakeScorer({"одно": -10.0, "одно два три четыре": -10.0})
    rescorer = WordRescorer(scorer, RescoreConfig(weight=1.0, normalize_by_words=True))

    assert rescorer.score_text("одно") == pytest.approx(-10.0)
    assert rescorer.score_text("одно два три четыре") == pytest.approx(-2.5)


def test_candidates_are_normalized_to_nfc_before_scoring():
    """The codec emits и + U+0306; the word model was trained on й."""
    scorer = FakeScorer({"мой дом": -1.0})
    rescorer = WordRescorer(scorer, RescoreConfig(weight=1.0))

    assert rescorer.score_text("мои\u0306 дом") == pytest.approx(-0.5)
    assert scorer.calls == ["мой дом"]


def test_single_candidate_is_returned_untouched():
    scorer = FakeScorer({})
    rescorer = WordRescorer(scorer, RescoreConfig())

    outcome = rescorer.choose([candidate("дом")])

    assert outcome.changed is False
    assert scorer.calls == []


def test_empty_candidate_list_is_an_error():
    rescorer = WordRescorer(FakeScorer({}), RescoreConfig())

    with pytest.raises(ValueError):
        rescorer.choose([])
