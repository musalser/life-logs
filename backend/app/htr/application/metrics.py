"""CER/WER computed via Levenshtein edit distance.

CER = char-level edit distance / reference length,
WER = word-level edit distance / reference word count,
aggregated over pairs as total distance / total reference length.
"""
from __future__ import annotations

from typing import Any, Sequence


def levenshtein(a: Sequence, b: Sequence) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _rate(distance: int, ref_len: int, hyp_len: int) -> float:
    if ref_len == 0:
        return 0.0 if hyp_len == 0 else 1.0
    return distance / ref_len


def cer(reference: str, hypothesis: str) -> float:
    return _rate(levenshtein(reference, hypothesis), len(reference), len(hypothesis))


def wer(reference: str, hypothesis: str) -> float:
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    return _rate(levenshtein(ref_words, hyp_words), len(ref_words), len(hyp_words))


class MetricsEvaluator:
    """Aggregates CER/WER over (reference, hypothesis) pairs."""

    def evaluate_pairs(self, pairs: list[tuple[str, str]]) -> dict[str, Any]:
        char_dist = char_ref = char_hyp = 0
        word_dist = word_ref = word_hyp = 0
        for reference, hypothesis in pairs:
            char_dist += levenshtein(reference, hypothesis)
            char_ref += len(reference)
            char_hyp += len(hypothesis)
            ref_words = reference.split()
            hyp_words = hypothesis.split()
            word_dist += levenshtein(ref_words, hyp_words)
            word_ref += len(ref_words)
            word_hyp += len(hyp_words)
        return {
            "cer": _rate(char_dist, char_ref, char_hyp),
            "wer": _rate(word_dist, word_ref, word_hyp),
            "lines": len(pairs),
        }
