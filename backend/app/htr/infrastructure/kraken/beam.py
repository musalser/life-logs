"""CTC prefix beam search for the kraken recognizer.

kraken 7 ships a single decoder (`kraken.lib.ctc_decoder.greedy_decoder`), which
takes the best path through the matrix independently of any notion of language.
This module adds the standard prefix beam search (Graves et al., Maas et al.)
on top of the *same* matrix, with two soft priors:

* a character n-gram language model (:mod:`app.htr.infrastructure.lm`), which
  fixes letter-level confusions and does not care whether the resulting word is
  in any dictionary — that is what makes toponyms and dialect spellings survive;
* a lexicon bonus for a completed word, which is deliberately *a bonus, not a
  constraint*: an out-of-vocabulary word loses a little score instead of being
  forbidden.

Scoring is ``log P_ctc + alpha * log P_lm + beta * words + bonus * known words``
— the familiar pyctcdecode/KenLM formulation. Everything is log10.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

logger = logging.getLogger(__name__)

BLANK = 0


@dataclass
class BeamSearchConfig:
    #: prefixes kept per frame
    beam_width: int = 60
    #: characters considered per frame (per beam)
    top_k: int = 12
    #: language model weight (per character)
    alpha: float = 0.6
    #: word insertion bonus
    beta: float = 1.2
    #: extra bonus for a word the lexicon knows
    word_bonus: float = 0.8
    #: drop beams this far (in log10) behind the best one
    prune_width: float = 9.0

    def describe(self) -> dict:
        return {
            "beam_width": self.beam_width,
            "top_k": self.top_k,
            "alpha": self.alpha,
            "beta": self.beta,
            "word_bonus": self.word_bonus,
            "prune_width": self.prune_width,
        }


@dataclass
class _Beam:
    prefix: tuple[int, ...]
    blank: float = -math.inf
    non_blank: float = -math.inf
    lm: float = 0.0
    words: float = 0.0
    #: label ids of the word being built; only joined when it is completed
    word: tuple[int, ...] = ()

    @property
    def prob(self) -> float:
        return _log_add(self.blank, self.non_blank)

    @property
    def score(self) -> float:
        return self.prob + self.lm + self.words


def _log_add(a: float, b: float) -> float:
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    if a < b:
        a, b = b, a
    return a + math.log10(1.0 + 10.0 ** (b - a))


class PrefixBeamSearch:
    """Decodes one CTC matrix (classes × time) into a string."""

    def __init__(
        self,
        codec,
        lm=None,
        config: BeamSearchConfig | None = None,
        known_word: Callable[[str], bool] | None = None,
        space: str = " ",
    ):
        self.codec = codec
        self.lm = lm
        self.config = config or BeamSearchConfig()
        self.known_word = known_word
        self.space = space
        #: label id -> grapheme / language model id, computed once
        self._graphemes = self._build_graphemes()
        self._lm_ids = np.array(
            [self.lm.char_id(grapheme) if lm is not None else 0 for grapheme in self._graphemes],
            dtype=np.int64,
        )
        self._space_label = next(
            (index for index, grapheme in enumerate(self._graphemes) if grapheme == space), None
        )
        self._class_count = len(self._graphemes)
        #: (context ids, label) -> log10 P; the same context repeats across beams
        self._lm_cache: dict[tuple[tuple[int, ...], int], float] = {}
        self._word_cache: dict[tuple[int, ...], float] = {}

    # ------------------------------------------------------------------

    def _build_graphemes(self) -> list[str]:
        """Label id -> grapheme, index 0 being the CTC blank."""
        size = max(self.codec.l2c_single) + 1 if self.codec.l2c_single else 1
        graphemes = [""] * max(size, 1)
        for label, grapheme in self.codec.l2c_single.items():
            graphemes[int(label)] = grapheme
        return graphemes

    def text(self, prefix: Sequence[int]) -> str:
        return "".join(self._graphemes[label] for label in prefix)

    # ------------------------------------------------------------------

    def decode(self, matrix: np.ndarray, temperature: float = 1.0) -> str:
        """Decode one line. ``matrix`` is kraken's record matrix: (classes, time)."""
        probs = self.probabilities(matrix, temperature)
        beams = {(): _Beam(prefix=(), blank=0.0)}
        for frame in range(probs.shape[1]):
            column = probs[:, frame]
            width = min(self.config.top_k, column.size)
            candidates = np.argpartition(column, -width)[-width:]
            candidates = candidates[np.argsort(-column[candidates])]
            log_probs = np.log10(np.maximum(column[candidates], 1e-12))
            beams = self._advance(beams, candidates, log_probs)
        if not beams:
            return ""
        # the last word of the line has no trailing space, so its reward is
        # applied here, once, for every surviving beam
        best = max(
            beams.values(),
            key=lambda beam: beam.score + self._word_reward(beam.word),
        )
        return self.text(best.prefix)

    def probabilities(self, matrix: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """Normalize kraken's matrix into (classes, time) probabilities.

        The matrix attached to a kraken record is already softmaxed (its columns
        sum to one), so nothing is exponentiated here; a temperature is applied
        as ``p ** (1 / T)`` for callers that want a flatter or sharper
        distribution.
        """
        values = np.asarray(matrix, dtype=np.float32)
        if values.ndim == 3:
            values = values[0]
        if values.ndim != 2:
            raise ValueError(f"CTC matrix must be 2-D, got {values.shape}")
        # orient by the number of classes rather than by shape: a long line has
        # more frames than classes
        if values.shape[0] != self._class_count and values.shape[1] == self._class_count:
            values = values.T
        if values.shape[0] != self._class_count:
            raise ValueError(
                f"CTC matrix has {values.shape[0]} classes, codec expects {self._class_count}"
            )
        if temperature != 1.0:
            values = np.power(np.maximum(values, 1e-30), 1.0 / max(temperature, 1e-6))
        # normalise per frame, i.e. across the class axis
        totals = values.sum(axis=0, keepdims=True)
        return values / np.maximum(totals, 1e-30)

    # ------------------------------------------------------------------

    def _advance(self, beams, candidates: np.ndarray, log_probs: np.ndarray):
        updated: dict[tuple[int, ...], _Beam] = {}
        best_score = -math.inf
        for beam in beams.values():
            score = beam.score
            if score > best_score:
                best_score = score
        threshold = best_score - self.config.prune_width

        labels = candidates.tolist()
        for beam in beams.values():
            prefix = beam.prefix
            blank_prob = beam.blank
            prob = _log_add(blank_prob, beam.non_blank)
            same = updated.get(prefix)
            for label, log_prob in zip(labels, log_probs.tolist()):
                if label == BLANK:
                    if same is None:
                        same = _Beam(prefix=prefix, lm=beam.lm, words=beam.words, word=beam.word)
                        updated[prefix] = same
                    same.blank = _log_add(same.blank, prob + log_prob)
                    continue
                repeated = prefix and prefix[-1] == label
                if repeated:
                    if same is None:
                        same = _Beam(prefix=prefix, lm=beam.lm, words=beam.words, word=beam.word)
                        updated[prefix] = same
                    same.non_blank = _log_add(same.non_blank, beam.non_blank + log_prob)
                    base = blank_prob
                else:
                    base = prob
                if base == -math.inf:
                    continue
                extension = prefix + (label,)
                target = updated.get(extension)
                if target is None:
                    lm_score, words, word = self._extend(beam, label)
                    target = _Beam(prefix=extension, lm=lm_score, words=words, word=word)
                    updated[extension] = target
                target.non_blank = _log_add(target.non_blank, base + log_prob)

        ranked = sorted(updated.values(), key=lambda item: item.score, reverse=True)
        kept = [beam for beam in ranked if beam.score >= threshold][: self.config.beam_width]
        if not kept:
            kept = ranked[: self.config.beam_width]
        return {beam.prefix: beam for beam in kept}

    def _extend(self, beam: _Beam, label: int) -> tuple[float, float, tuple[int, ...]]:
        """Language-model and lexicon scores for appending ``label``."""
        lm_score = beam.lm
        words = beam.words
        word = beam.word
        if self.lm is not None:
            context = beam.prefix[-(self.lm.order - 1) :] if self.lm.order > 1 else ()
            context_ids = tuple(int(self._lm_ids[item]) for item in context)
            lm_id = int(self._lm_ids[label])
            key = (context_ids, lm_id)
            cached = self._lm_cache.get(key)
            if cached is None:
                cached = self.lm.logprob(context_ids, lm_id)
                if len(self._lm_cache) > 2_000_000:  # pragma: no cover - safety valve
                    self._lm_cache.clear()
                self._lm_cache[key] = cached
            lm_score += self.config.alpha * cached

        if label == self._space_label:
            if word:
                words += self._word_reward(word)
            word = ()
        else:
            word = word + (label,)
        return lm_score, words, word

    def _word_reward(self, word_labels: tuple[int, ...]) -> float:
        word = "".join(self._graphemes[label] for label in word_labels).strip()
        if not word:
            return 0.0
        cached = self._word_cache.get(word_labels)
        if cached is not None:
            return cached
        reward = self.config.beta
        if self.known_word is not None and self.known_word(word):
            reward += self.config.word_bonus
        if len(self._word_cache) > 500_000:  # pragma: no cover - safety valve
            self._word_cache.clear()
        self._word_cache[word_labels] = reward
        return reward

    # ------------------------------------------------------------------

def greedy_text(matrix: np.ndarray, codec, temperature: float = 1.0, blank: int = BLANK) -> str:
    """Greedy (best path) decoding of the same matrix, for comparison."""
    values = np.asarray(matrix, dtype=np.float32)
    if values.ndim == 3:
        values = values[0]
    if temperature != 1.0:
        values = values / max(temperature, 1e-6)
    labels = values.argmax(axis=0)
    graphemes = [""] * (max(codec.l2c_single) + 1) if codec.l2c_single else []
    for label, grapheme in codec.l2c_single.items():
        graphemes[int(label)] = grapheme
    out: list[str] = []
    previous = blank
    for label in labels.tolist():
        if label != previous and label != blank:
            out.append(graphemes[label] if label < len(graphemes) else "")
        previous = label
    return "".join(out)
