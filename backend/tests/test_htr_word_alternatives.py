"""Unit tests of the per-word alternative readings from the N-best list."""
from __future__ import annotations

import pytest

from app.htr.domain.entities import WordAlternative
from app.htr.domain.text import align_word_alternatives
from app.htr.infrastructure.kraken.beam import BeamCandidate
from app.htr.infrastructure.kraken.recognizer import KrakenRecognizer


# ---------------------------------------------------------------------------
# alignment of the chosen words against the other hypotheses
# ---------------------------------------------------------------------------


def test_replacement_lands_on_the_right_word():
    chosen = ["дед", "был", "похож"]
    alternatives = align_word_alternatives(
        chosen,
        [(chosen, 0.0), (["дедъ", "был", "похож"], -0.5), (["дед", "быль", "похож"], -0.7)],
        limit=5,
    )

    assert alternatives == {0: [("дедъ", -0.5)], 1: [("быль", -0.7)]}


def test_merged_pair_is_offered_as_one_alternative():
    """'на зывалось' -> 'называлось': exactly the case a character LM cannot see."""
    alternatives = align_word_alternatives(
        ["на", "зывалось", "дом"],
        [(["на", "зывалось", "дом"], 0.0), (["называлось", "дом"], -1.0)],
        limit=5,
    )

    assert alternatives[0][0][0] == "называлось"


def test_split_word_is_offered_too():
    alternatives = align_word_alternatives(
        ["называлось", "дом"],
        [(["называлось", "дом"], 0.0), (["на", "зывалось", "дом"], -1.0)],
        limit=5,
    )

    assert any("на" == text for text, _score in alternatives.get(0, []))


def test_alternatives_are_deduplicated_and_limited():
    chosen = ["дом"]
    hypotheses = [(chosen, 0.0)]
    for index in range(6):
        hypotheses.append(([f"дом{index}"], -float(index)))

    alternatives = align_word_alternatives(chosen, hypotheses, limit=3)

    assert len(alternatives[0]) == 3
    assert [text for text, _score in alternatives[0]] == ["дом0", "дом1", "дом2"]


def test_chosen_reading_is_never_listed_as_an_alternative():
    chosen = ["дом"]
    alternatives = align_word_alternatives(
        chosen, [(chosen, 0.0), (["дом"], -1.0)], limit=5
    )

    assert alternatives == {}


def test_hypotheses_are_sorted_by_score_not_by_input_order():
    chosen = ["дом"]
    alternatives = align_word_alternatives(
        chosen,
        [(chosen, 0.0), (["домъ"], -5.0), (["домб"], -1.0)],
        limit=5,
    )

    assert [text for text, _score in alternatives[0]] == ["домб", "домъ"]


def test_limit_zero_disables_the_feature():
    assert align_word_alternatives(["дом"], [(["дом"], 0.0), (["домъ"], -1.0)], limit=0) == {}


# ---------------------------------------------------------------------------
# the recognizer turns candidates into per-word alternatives
# ---------------------------------------------------------------------------


def candidate(text: str, score: float) -> BeamCandidate:
    return BeamCandidate(text=text, score=score, acoustic=score, lm=0.0, words=0.0)


def test_recognizer_collects_alternatives_by_word_index():
    candidates = [
        candidate("дед был похож на еврея", 0.0),
        candidate("дедъ был похож на еврея", -0.4),
        candidate("дед был похож на еврея.", -0.9),
    ]

    alternatives = KrakenRecognizer._word_alternatives(
        candidates[0].text, candidates, limit=5
    )

    # index 0: another spelling of the first word; index 4: the last word with
    # a trailing full stop — both are readings the beam actually considered
    assert list(alternatives) == [0, 4]
    assert [item.text for item in alternatives[0]] == ["дедъ"]
    assert [item.text for item in alternatives[4]] == ["еврея."]
    assert all(isinstance(item, WordAlternative) for item in alternatives[0])


def test_recognizer_needs_at_least_two_hypotheses():
    only_one = [candidate("дом", 0.0)]

    assert KrakenRecognizer._word_alternatives("дом", only_one, limit=5) == {}
    assert KrakenRecognizer._word_alternatives("", only_one * 2, limit=5) == {}


def test_recognizer_respects_the_limit():
    candidates = [candidate("дом", 0.0)] + [
        candidate(f"дом{index}", -float(index)) for index in range(1, 5)
    ]

    alternatives = KrakenRecognizer._word_alternatives("дом", candidates, limit=2)

    assert len(alternatives[0]) == 2


# ---------------------------------------------------------------------------
# persistence helpers
# ---------------------------------------------------------------------------


def test_alternatives_survive_a_dump_load_round_trip():
    from app.htr.infrastructure.page_repository import (
        _dump_alternatives,
        _load_alternatives,
    )

    items = [WordAlternative(text="домъ", score=-1.23456), WordAlternative(text="домб", score=-2.0)]

    restored = _load_alternatives(_dump_alternatives(items))

    assert [item.text for item in restored] == ["домъ", "домб"]
    assert restored[0].score == pytest.approx(-1.2346, abs=1e-4)


def test_empty_alternatives_are_stored_as_null():
    from app.htr.infrastructure.page_repository import _dump_alternatives

    assert _dump_alternatives([]) is None
    assert _dump_alternatives(None) is None


def test_broken_json_in_the_column_is_ignored():
    from app.htr.infrastructure.page_repository import _load_alternatives

    assert _load_alternatives("not json") == []
    assert _load_alternatives('[{"no_text": 1}]') == []
    assert _load_alternatives(None) == []


# ---------------------------------------------------------------------------
# keeping word geometry when the beam and greedy tokenizations differ
# ---------------------------------------------------------------------------


def test_word_geometry_survives_a_merge():
    """'где -то' (greedy) vs 'где-то' (beam) used to drop the whole line's boxes."""
    from app.htr.domain.text import map_spans_to_reference
    from app.htr.infrastructure.kraken.recognizer import word_spans

    display = "где-то погиб. Мамин"
    reference = "где -то погиб. Мамин"

    ranges = map_spans_to_reference(display, reference, word_spans(display))

    assert ranges[0] == (0, 7), "слитое слово должно взять оба куска greedy"
    assert reference[ranges[0][0] : ranges[0][1]] == "где -то"
    assert ranges[1] == (8, 14)
    assert ranges[2] == (15, 20)


def test_word_geometry_survives_a_split():
    from app.htr.domain.text import map_spans_to_reference
    from app.htr.infrastructure.kraken.recognizer import word_spans

    display = "называлось дом"
    reference = "на зывалось дом"

    ranges = map_spans_to_reference(display, reference, word_spans(display))

    assert reference[ranges[0][0] : ranges[0][1]] == "на зывалось"
    assert ranges[1] == (12, 15)


def test_word_geometry_survives_a_letter_change():
    from app.htr.domain.text import map_spans_to_reference
    from app.htr.infrastructure.kraken.recognizer import word_spans

    ranges = map_spans_to_reference("дед был", "дедъ был", word_spans("дед был"))

    assert [range_ for range_ in ranges if range_] == [(0, 3), (5, 8)]


def test_word_inserted_by_the_beam_has_no_geometry_but_the_rest_do():
    from app.htr.domain.text import map_spans_to_reference
    from app.htr.infrastructure.kraken.recognizer import word_spans

    display = "и я тут"
    reference = "и тут"

    ranges = map_spans_to_reference(display, reference, word_spans(display))

    assert ranges[0] is not None
    assert ranges[1] is None, "у вставленного слова нет геометрии — и только у него"
    assert ranges[2] is not None


def test_identical_texts_keep_their_own_spans():
    from app.htr.domain.text import map_spans_to_reference
    from app.htr.infrastructure.kraken.recognizer import word_spans

    text = "дед был похож"

    assert map_spans_to_reference(text, text, word_spans(text)) == word_spans(text)
