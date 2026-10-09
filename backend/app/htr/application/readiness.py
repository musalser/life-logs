"""Decides whether a confirmed-page corpus is large enough for fine-tuning.

Training is started explicitly by the user, and confirming a page is already
their decision that the material is worth learning from — so the size threshold
is **disabled by default** (``min_lines = 0``) and any non-empty corpus is
trained on. A positive ``min_lines`` / ``min_words`` restores the old gate for
deployments that want it; a single page (a few dozen lines) can overfit a model,
but that is the user's call, not a hidden refusal.

The one thing that is always required is *some* material: a run over an empty
corpus cannot produce a model, so it stays ``INSUFFICIENT_DATA`` with a reason.
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
    def __init__(self, min_lines: int = 0, min_words: int = 0):
        self.min_lines = max(0, min_lines)
        self.min_words = max(0, min_words)

    def evaluate(self, dataset: TrainingDataset) -> TrainingReadiness:
        lines = len(dataset.samples)
        words = sum(len(sample.transcription.split()) for sample in dataset.samples)

        missing: list[str] = []
        if lines < self.min_lines:
            missing.append(f"{lines}/{self.min_lines} line(s)")
        if self.min_words > 0 and words < self.min_words:
            missing.append(f"{words}/{self.min_words} word(s)")
        if not missing and lines == 0:
            # not a threshold but a floor: there is nothing to train on at all
            missing.append("no confirmed lines at all")

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
