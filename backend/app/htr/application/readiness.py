"""Decides whether a confirmed-page corpus is large enough for fine-tuning.

Training is started explicitly by the user, but a single page (a few dozen
lines) is rarely enough to adapt a model without overfitting it. The dataset
therefore has to reach a configurable size before a training run does any work;
until then the request succeeds and the training result reports how much
material is still missing (``INSUFFICIENT_DATA``).

The threshold is expressed in lines by default. An optional word threshold can
be enabled as well (``min_words > 0``); both conditions must then hold.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..domain.entities import TrainingDataset


@dataclass(frozen=True)
class TrainingReadiness:
    ready: bool
    lines: int
    min_lines: int
    words: int
    min_words: int
    reason: str | None = None


class TrainingReadinessPolicy:
    def __init__(self, min_lines: int = 50, min_words: int = 0):
        if min_lines < 1 and min_words < 1:
            raise ValueError("at least one training threshold must be positive")
        self.min_lines = min_lines
        self.min_words = min_words

    def evaluate(self, dataset: TrainingDataset) -> TrainingReadiness:
        lines = len(dataset.samples)
        words = sum(len(sample.transcription.split()) for sample in dataset.samples)

        missing: list[str] = []
        if lines < self.min_lines:
            missing.append(f"{lines}/{self.min_lines} line(s)")
        if self.min_words > 0 and words < self.min_words:
            missing.append(f"{words}/{self.min_words} word(s)")

        if missing:
            return TrainingReadiness(
                ready=False,
                lines=lines,
                min_lines=self.min_lines,
                words=words,
                min_words=self.min_words,
                reason=(
                    "Not enough confirmed training data yet: "
                    + ", ".join(missing)
                    + "; confirm more pages, then start training again"
                ),
            )
        return TrainingReadiness(
            ready=True,
            lines=lines,
            min_lines=self.min_lines,
            words=words,
            min_words=self.min_words,
        )
