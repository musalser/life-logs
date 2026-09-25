"""Two-pass decoding: re-rank the beam's N-best list with a word-level LM.

Stage 1 (:mod:`app.htr.infrastructure.kraken.beam`) scores *characters* with a
cheap character n-gram, which is what makes a 1.5 M-lookup-per-page search
affordable. Stage 2 takes the N best hypotheses of that search and re-scores
them with a word-level model (KenLM trained on a large Russian corpus and
interpolated with the author's own text), where word order and word choice
actually matter.

The re-ranking weight has to stay small and the winner may only change when the
first pass is genuinely unsure. Both rules come from measurement, not taste:

* an over-weighted character LM was already measured to make WER *worse* than no
  LM at all (weight 0.45 gave 28.7 % against 25.5 % for plain greedy);
* a word-level LM is trained on general text, so it pulls towards frequent
  words. Applied without a guard it will happily trade a rare surname or
  toponym for a common word — the same failure the LLM step shows occasionally,
  but without the OOV colouring that makes it visible.

So the default guard is: **change the winner only if the first-pass top-1
contains a word the dictionary does not know, or its mean acoustic confidence is
below the threshold.** If the top-1 is fully known and confident, it is kept.

Scoring (log10, like everywhere else in this pipeline)::

    final = beam_score + weight * word_score
"""

from __future__ import annotations

import logging
import unicodedata
from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

logger = logging.getLogger(__name__)


class KenLMSentenceScorer:
    """Adapter over ``kenlm.Model`` (the import stays lazy and optional)."""

    def __init__(self, model):
        self.model = model

    def score(self, sentence: str) -> float:
        return float(self.model.score(sentence, bos=True, eos=True))


class SentenceScorer(Protocol):
    """A word-level model that scores a whole line (KenLM, in practice)."""

    def score(self, sentence: str) -> float:
        """log10 P(sentence); higher (closer to zero) is better."""
        ...


@dataclass
class RescoreConfig:
    #: how much the word-level model may move the ranking (`w` in the formula)
    weight: float = 0.5
    #: divide the sentence score by its word count. KenLM returns a *total*
    #: log-probability, which grows with length; without this a shorter
    #: candidate wins by being shorter rather than by being better.
    normalize_by_words: bool = True
    #: only re-rank when the first pass is unsure (see the module docstring)
    guard: bool = True
    #: mean acoustic log10 per character below which the line counts as unsure
    min_mean_acoustic: float = -0.30

    def describe(self) -> dict:
        return {
            "weight": self.weight,
            "normalize_by_words": self.normalize_by_words,
            "guard": self.guard,
            "min_mean_acoustic": self.min_mean_acoustic,
        }


@dataclass
class RescoreOutcome:
    """What the second pass did, and why — worth logging for every line."""

    text: str
    changed: bool
    reason: str
    first_pass_text: str
    first_pass_score: float
    rescored_score: float | None = None

    def __str__(self) -> str:  # pragma: no cover - logging helper
        if not self.changed:
            return f"kept first pass ({self.reason})"
        return (
            f"rescored ({self.reason}): {self.first_pass_text!r} -> {self.text!r} "
            f"({self.first_pass_score:.3f} -> {self.rescored_score:.3f})"
        )


class WordRescorer:
    """Re-ranks the N best beam hypotheses with a word-level language model."""

    def __init__(
        self,
        scorer: SentenceScorer,
        config: RescoreConfig | None = None,
        known_word: Callable[[str], bool] | None = None,
    ):
        self.scorer = scorer
        self.config = config or RescoreConfig()
        self.known_word = known_word

    # ------------------------------------------------------------------

    def score_text(self, text: str) -> float:
        """Word-level score of one hypothesis, normalized for length."""
        if not text.strip():
            return 0.0
        # the codec emits decomposed Cyrillic; the model was trained on
        # composed text, so this is the same й/ё trap as everywhere else
        value = float(self.scorer.score(unicodedata.normalize("NFC", text)))
        if self.config.normalize_by_words:
            from ...domain.text import iter_words

            value /= max(1, len(iter_words(text)))
        return value

    def needs_help(self, candidate) -> tuple[bool, str]:
        """Is the first pass unsure enough to allow a re-ranking?"""
        if not self.config.guard:
            return True, "guard disabled"
        if self.config.min_mean_acoustic is not None:
            if candidate.mean_acoustic < self.config.min_mean_acoustic:
                return True, f"low confidence ({candidate.mean_acoustic:.3f})"
        if self.known_word is not None:
            from ...domain.text import iter_words

            unknown = [
                word
                for word in iter_words(unicodedata.normalize("NFC", candidate.text))
                if not self.known_word(word)
            ]
            if unknown:
                return True, f"out of dictionary: {unknown[0]!r}"
        return False, "top-1 is confident and fully known"

    def choose(self, candidates: Sequence) -> RescoreOutcome:
        """Pick the winner among ``candidates`` (best-first from the beam)."""
        if not candidates:
            raise ValueError("no candidates to rescore")
        first = candidates[0]
        if len(candidates) == 1:
            return RescoreOutcome(
                text=first.text, changed=False, reason="only one hypothesis",
                first_pass_text=first.text, first_pass_score=first.score,
            )

        allowed, why = self.needs_help(first)
        if not allowed:
            return RescoreOutcome(
                text=first.text, changed=False, reason=why,
                first_pass_text=first.text, first_pass_score=first.score,
            )

        scored = []
        for candidate in candidates:
            final = candidate.score + self.config.weight * self.score_text(candidate.text)
            scored.append((final, candidate))
        scored.sort(key=lambda item: item[0], reverse=True)
        best_final, best = scored[0]

        changed = best.text != first.text
        return RescoreOutcome(
            text=best.text,
            changed=changed,
            reason=why if changed else f"{why}, but the ranking did not change",
            first_pass_text=first.text,
            first_pass_score=first.score,
            rescored_score=best_final,
        )
