"""Builds the author-specific material the LLM text corrector needs.

Everything here comes from data the project already stores:

* **lexicon** — words the author really writes, taken from the confirmed pages
  (the user's own transcription, never the raw prediction);
* **confusions** — character substitutions between the raw prediction and the
  user's correction, i.e. the mistakes this recognizer makes on this hand;
* **examples** — real ``(prediction -> correction)`` line pairs used as
  few-shot examples;
* **vocabulary** — proper nouns and terms from the knowledge base (entities,
  events, goals, habits), which a generic language model cannot guess.

Nothing model-specific leaks into the domain: the result is a plain
:class:`~app.htr.domain.entities.CorrectionContext`.
"""
from __future__ import annotations

import logging
import re
from collections import Counter

from ..domain.entities import ConfusionPair, CorrectionContext
from ..domain.interfaces import PageRepository, VocabularyProvider

logger = logging.getLogger(__name__)

WORD_RE = re.compile(r"[\w'’\-]+", re.UNICODE)


class CorrectionContextBuilder:
    def __init__(
        self,
        page_repository: PageRepository,
        vocabulary_provider: VocabularyProvider | None = None,
        *,
        max_lexicon: int = 200,
        max_vocabulary: int = 200,
        max_confusions: int = 25,
        max_examples: int = 5,
        min_word_length: int = 3,
        min_confusion_count: int = 2,
    ):
        self.page_repository = page_repository
        self.vocabulary_provider = vocabulary_provider
        self.max_lexicon = max_lexicon
        self.max_vocabulary = max_vocabulary
        self.max_confusions = max_confusions
        self.max_examples = max_examples
        self.min_word_length = min_word_length
        self.min_confusion_count = min_confusion_count

    # ------------------------------------------------------------------

    def build(self, author_id: int, exclude_page_id: int | None = None) -> CorrectionContext:
        pages = [
            page
            for page in self.page_repository.get_confirmed_pages(author_id)
            if page.id != exclude_page_id
        ]

        lexicon: Counter[str] = Counter()
        confusions: Counter[tuple[str, str]] = Counter()
        examples: list[tuple[str, str]] = []
        user_ids: set[int] = set()

        for page in pages:
            user_ids.add(page.user_id)
            for line in page.lines:
                predicted = (line.predicted_text or "").strip()
                corrected = line.corrected_text
                truth = (corrected if corrected is not None else predicted).strip()
                if not truth:
                    continue
                for word in WORD_RE.findall(truth):
                    if len(word) >= self.min_word_length:
                        lexicon[word] += 1
                if corrected is not None and predicted and predicted != corrected:
                    for recognized, correct in self._substitutions(predicted, corrected):
                        confusions[(recognized, correct)] += 1
                    examples.append((predicted, corrected))

        context = CorrectionContext(
            lexicon=[word for word, _ in lexicon.most_common(self.max_lexicon)],
            vocabulary=self._vocabulary(user_ids),
            confusions=[
                ConfusionPair(recognized=recognized, correct=correct, count=count)
                for (recognized, correct), count in confusions.most_common(self.max_confusions)
                if count >= self.min_confusion_count
            ],
            examples=self._pick_examples(examples),
        )
        logger.info(
            "HTR correction context for author_id=%s: %s lexicon words, %s confusions, "
            "%s examples, %s vocabulary terms (from %s confirmed pages)",
            author_id, len(context.lexicon), len(context.confusions),
            len(context.examples), len(context.vocabulary), len(pages),
        )
        return context

    # ------------------------------------------------------------------

    @staticmethod
    def _substitutions(predicted: str, corrected: str) -> list[tuple[str, str]]:
        from .metrics import align_substitutions

        return [
            (recognized, correct)
            for recognized, correct in align_substitutions(predicted, corrected)
            if recognized.isalnum() and correct.isalnum()
        ]

    def _vocabulary(self, user_ids: set[int]) -> list[str]:
        if self.vocabulary_provider is None or not user_ids:
            return []
        terms: list[str] = []
        seen: set[str] = set()
        for user_id in sorted(user_ids):
            for term in self.vocabulary_provider.vocabulary(user_id):
                cleaned = (term or "").strip()
                if len(cleaned) < 2 or cleaned.casefold() in seen:
                    continue
                seen.add(cleaned.casefold())
                terms.append(cleaned)
        return terms[: self.max_vocabulary]

    def _pick_examples(self, examples: list[tuple[str, str]]) -> list[tuple[str, str]]:
        """Short, diverse corrections teach the most per prompt token."""
        unique: dict[str, tuple[str, str]] = {}
        for predicted, corrected in examples:
            unique.setdefault(corrected, (predicted, corrected))
        ordered = sorted(
            unique.values(),
            key=lambda pair: (abs(len(pair[0]) - len(pair[1])), len(pair[1])),
        )
        return ordered[: self.max_examples]
