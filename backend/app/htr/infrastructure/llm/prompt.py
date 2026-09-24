"""Prompt assembly and answer sanitising for LLM text correction.

Pure functions: no network, no LLM client, so they are unit-testable on their
own. The correction context (lexicon, confusions, examples, vocabulary) is
rendered as extra sections of the strict correction prompt.
"""
from __future__ import annotations

from ...domain.entities import CorrectionContext

BASE_RULES = """Ты — профессиональный эксперт по текстологии и архивному делу. Твоя задача — исправить механические ошибки OCR/HTR распознавания в приведенном тексте, используя контекст.
СТРОГИЕ ПРАВИЛА:
1. Исправляй ТОЛЬКО явные опечатки, склейки слов и ошибки распознавания букв (например, "к0рова" -> "корова", "нрншел" -> "пришел").
2. Ни в коем случае НЕ изменяй стиль автора, орфографию, пунктуацию и структуру текста.
3. НЕ пытайся "улучшить" или сделать текст более красивым/современным.
4. Если слово полностью неразборчиво или превратилось в хаотичный набор символов (например, "щфвл9"), и контекст не дает 100% уверенности — оставь его как есть, рядом добавь [неразборчиво]. Не галлюцинируй и не выдумывай факты.
5. Выведи ТОЛЬКО исправленный текст этой одной строки, без твоих комментариев, пояснений и вводных фраз."""

# Answers that are clearly longer/shorter than the input mean the model started
# explaining itself or dropped the line: keep the raw prediction instead.
MIN_LENGTH_RATIO = 0.34
MAX_LENGTH_RATIO = 3.0

_LEADING_LABELS = (
    "исправленный текст:",
    "исправленный вариант:",
    "стало:",
    "итог:",
    "ответ:",
    "corrected text:",
)


def build_correction_prompt(
    text: str,
    context_lines: list[str],
    context: CorrectionContext,
) -> str:
    """Render the correction prompt for a single line."""
    parts: list[str] = [BASE_RULES, ""]

    if context.lexicon or context.vocabulary:
        parts.append("СЛОВАРИ (если слово в тексте похоже на слово из словаря — используй написание из словаря):")
        if context.lexicon:
            parts.append("частые слова этого автора: " + ", ".join(context.lexicon))
        if context.vocabulary:
            parts.append(
                "имена, топонимы и термины из базы знаний: " + ", ".join(context.vocabulary)
            )
        parts.append("")

    if context.confusions:
        parts.append(
            "ЧАСТЫЕ ОШИБКИ РАСПОЗНАВАНИЯ НА ЭТОМ ПОЧЕРКЕ (распознано → должно быть):"
        )
        parts.append(
            ", ".join(
                f"{pair.recognized}→{pair.correct} ({pair.count})" for pair in context.confusions
            )
        )
        parts.append("")

    if context.examples:
        parts.append("ПРИМЕРЫ ИСПРАВЛЕНИЙ ЭТОГО ЖЕ АВТОРА (стиль и степень правки):")
        for predicted, corrected in context.examples:
            parts.append(f"  было: {predicted}")
            parts.append(f"  стало: {corrected}")
        parts.append("")

    if context_lines:
        parts.append("СОСЕДНИЕ СТРОКИ (только для понимания смысла, исправлять их НЕ нужно):")
        for line in context_lines:
            parts.append(f"  {line}")
        parts.append("")

    parts.append("ИСПРАВЬ ТОЛЬКО ЭТУ СТРОКУ И ВЫВЕДИ ТОЛЬКО ЕЁ:")
    parts.append(text)
    return "\n".join(parts)


def sanitize_correction(raw: str | None, original: str) -> str | None:
    """Clean a model answer, or return None when it cannot be trusted.

    ``None`` means "keep the raw prediction"; the caller decides what to do.
    An answer is rejected when it is empty, spans several lines (we ask for
    exactly one), or differs so much in length that the model clearly started
    explaining itself or dropped the line.
    """
    if not raw:
        return None
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) != 1:
        return None

    candidate = lines[0].lstrip("-•*").strip()
    lowered = candidate.casefold()
    for label in _LEADING_LABELS:
        if lowered.startswith(label):
            candidate = candidate[len(label):].strip()
            break

    if len(candidate) >= 2 and _is_quoted(candidate):
        candidate = candidate[1:-1].strip()

    if not candidate:
        return None
    original_length = max(len(original.strip()), 1)
    ratio = len(candidate) / original_length
    if ratio < MIN_LENGTH_RATIO or ratio > MAX_LENGTH_RATIO:
        return None
    return candidate


_QUOTE_PAIRS = {'"': '"', "'": "'", "«": "»", "“": "”"}


def _is_quoted(text: str) -> bool:
    return _QUOTE_PAIRS.get(text[0]) == text[-1]
