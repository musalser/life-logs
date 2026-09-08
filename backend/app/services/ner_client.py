from __future__ import annotations

import re
from typing import List, Dict

from transformers import pipeline as hf_pipeline

from ..config import settings


class NERService:
    def __init__(self):
        self.pipeline = hf_pipeline(
            "ner",
            model=settings.ner_model,
            aggregation_strategy="simple",
            device=-1,  # CPU; set to 0 to use GPU
        )
        print(f"✓ NERService модель загружена: {settings.ner_model}")

    def extract_entities(self, text: str, min_score: float = 0.75) -> List[Dict]:
        """
        Returns a list of dicts:
          entity_type  – NEREL label (PERSON, ORG, LOC, DATE, …)
          mention_text – surface form found in text
          score        – confidence [0, 1]
          start        – char offset in text
          end          – char offset in text
          sentence     – surrounding sentence for context
        """
        results = self.pipeline(text)
        entities = []
        for r in results:
            if r["score"] < min_score:
                continue
            mention = r["word"].strip()
            if not mention:
                continue
            entities.append(
                {
                    "entity_type": r["entity_group"],
                    "mention_text": mention,
                    "score": float(r["score"]),
                    "start": r["start"],
                    "end": r["end"],
                    "sentence": _extract_sentence(text, r["start"], r["end"]),
                }
            )
        return entities


# ── helpers ──────────────────────────────────────────────────────────────────

_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+")


def _extract_sentence(text: str, start: int, end: int) -> str:
    """Return the sentence that contains the span [start, end)."""
    sentences = _SENT_SPLIT.split(text)
    pos = 0
    for sent in sentences:
        sent_end = pos + len(sent)
        if pos <= start < sent_end:
            return sent.strip()
        pos = sent_end + 1  # +1 for the consumed space
    return text[max(0, start - 60): end + 60].strip()
