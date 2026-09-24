"""Unit tests of the character language model and the CTC prefix beam search.

The decoder is the piece that decides what text lands in the corpus, so the
tests pin down the two things that can go silently wrong: the matrix is
normalized over the *class* axis (a first version normalized over time and
produced gibberish), and the language model can actually overrule the acoustic
model on a genuinely confusable frame.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.htr.infrastructure.kraken.beam import (
    BeamSearchConfig,
    PrefixBeamSearch,
    greedy_text,
)
from app.htr.infrastructure.lm.char_ngram import CharNGram


# ---------------------------------------------------------------------------
# character language model
# ---------------------------------------------------------------------------


def test_lm_scores_a_known_continuation_above_an_unknown_one():
    lm = CharNGram.build(["корова мычала. " * 50], order=4, min_count=1)
    context = [lm.char_id(char) for char in "коров"]
    assert lm.logprob(context, lm.char_id("а")) > lm.logprob(context, lm.char_id("ы"))
    # a character the model has never seen falls back to the floor
    assert lm.logprob(context, lm.char_id("ж")) == pytest.approx(lm.floor_logprob)


def test_lm_ranks_a_plausible_word_above_a_scrambled_one():
    lm = CharNGram.build(["корова мычала во дворе. " * 50], order=5, min_count=1)
    assert lm.text_logprob("корова мычала") > lm.text_logprob("карова мячяла")


def test_lm_survives_a_save_and_load_round_trip(tmp_path):
    lm = CharNGram.build(["наша улица вела к озеру. " * 40], order=5, min_count=1)
    path = lm.save(tmp_path / "lm.npz")

    restored = CharNGram.load(path)

    assert restored.order == lm.order
    assert restored.chars == lm.chars
    assert restored.text_logprob("наша улица") == pytest.approx(lm.text_logprob("наша улица"))


def test_lm_records_what_it_was_built_from(tmp_path):
    """The recognizer needs to tell running text from bare word forms."""
    lm = CharNGram.build(
        ["корова мычала во дворе. " * 10, "к\nо\nр\nо\nв\nа"],
        order=5,
        min_count=1,
        meta={"running_text_chars": 240, "word_form_chars": 11},
    )
    assert lm.meta["running_text_chars"] == 240

    restored = CharNGram.load(lm.save(tmp_path / "meta.npz"))
    assert restored.meta == lm.meta


def test_lm_without_metadata_reports_no_running_text():
    """Artifacts built before the counters existed must not enable beam search."""
    lm = CharNGram.build(["корова мычала. " * 50], order=4, min_count=1)
    assert lm.meta.get("running_text_chars", 0) == 0


def test_lm_backs_off_to_a_shorter_context():
    """An unseen 4-gram must fall back, not return the floor."""
    lm = CharNGram.build(["корова мычала. " * 50], order=6, min_count=1)
    unseen = [lm.char_id(char) for char in "ыжю"]
    score = lm.logprob(unseen, lm.char_id("а"))
    assert score > lm.floor_logprob


def test_lm_storage_is_compact():
    """A dict of tuple keys would be gigabytes; the packed form must not be."""
    text = "корова мычала во дворе. " * 500
    lm = CharNGram.build([text], order=6, min_count=1)
    assert lm.size_bytes < 20_000_000


# ---------------------------------------------------------------------------
# beam search
# ---------------------------------------------------------------------------


def _codec(charset: str):
    from kraken.lib.codec import PytorchCodec

    return PytorchCodec(list(charset))


def _blank_matrix(codec, text: str, frames_per_char: int = 6) -> np.ndarray:
    """A sharp matrix that decodes greedily to ``text``."""
    labels = [codec.c2l[char][0] for char in text]
    width = len(labels) * frames_per_char
    matrix = np.zeros((len(codec.l2c_single) + 1, width), dtype=np.float32)
    matrix[0] = 0.9
    for index, label in enumerate(labels):
        for frame in range(index * frames_per_char, (index + 1) * frames_per_char - 1):
            matrix[:, frame] = 0.001
            matrix[label, frame] = 0.9
    matrix /= matrix.sum(axis=0, keepdims=True)
    return matrix


def test_probabilities_are_normalized_over_classes():
    codec = _codec("абв ")
    search = PrefixBeamSearch(codec=codec)
    matrix = np.full((len(codec.l2c_single) + 1, 7), 3.0, dtype=np.float32)

    probs = search.probabilities(matrix)

    assert probs.shape[0] == len(codec.l2c_single) + 1
    assert np.allclose(probs.sum(axis=0), 1.0)


def test_beam_without_a_language_model_agrees_with_greedy():
    codec = _codec("абв ")
    matrix = _blank_matrix(codec, "аб в")
    search = PrefixBeamSearch(codec=codec, config=BeamSearchConfig(alpha=0.0, beta=0.0))

    assert search.decode(matrix) == greedy_text(matrix, codec) == "аб в"


def _confusable_matrix(
    codec, correct: str, wrong: str, position: int = 1, frames_per_char: int = 8
) -> np.ndarray:
    """A matrix whose ``position``-th character is a near tie the argmax loses.

    Every frame of that character offers the wrong character at 0.45, the right
    one at 0.42 and a blank at 0.13 — exactly what a CTC model emits for a
    letter it cannot separate, and something greedy decoding cannot recover
    from because it never looks at the language.
    """
    matrix = _blank_matrix(codec, correct, frames_per_char=frames_per_char)
    wrong_label = codec.c2l[wrong][0]
    right_label = codec.c2l[correct[position]][0]
    start = position * frames_per_char
    for frame in range(start, start + frames_per_char):
        matrix[:, frame] = 0.0
        matrix[0, frame] = 0.13
        matrix[wrong_label, frame] = 0.45
        matrix[right_label, frame] = 0.42
    matrix /= matrix.sum(axis=0, keepdims=True)
    return matrix


def test_language_model_overrules_the_acoustic_model():
    """The classic case: 'о' and 'а' are acoustically close, the LM is not."""
    codec = _codec("акорв ")
    lm = CharNGram.build(["корова " * 200], order=5, min_count=1)
    matrix = _confusable_matrix(codec, "корова", wrong="а")

    assert greedy_text(matrix, codec).startswith("ка")  # the acoustic model errs
    search = PrefixBeamSearch(
        codec=codec, lm=lm, config=BeamSearchConfig(alpha=0.8, beta=0.0, word_bonus=0.0)
    )
    assert search.decode(matrix).startswith("ко")       # the language model fixes it


def test_lexicon_bonus_prefers_a_known_word():
    codec = _codec("акорв ")
    matrix = _confusable_matrix(codec, "корова", wrong="а")

    search = PrefixBeamSearch(
        codec=codec,
        lm=None,
        config=BeamSearchConfig(alpha=0.0, beta=0.0, word_bonus=8.0),
        known_word=lambda word: word.strip() == "корова",
    )

    assert search.decode(matrix).startswith("ко")


def test_beam_keeps_the_decomposed_text_the_codec_emits():
    """й arrives as и + U+0306; the decoder must not compose it away."""
    codec = _codec("и\u0306й ")
    matrix = _blank_matrix(codec, "и\u0306")
    search = PrefixBeamSearch(codec=codec, config=BeamSearchConfig(alpha=0.0, beta=0.0))

    assert search.decode(matrix) == "и\u0306"
