"""Unit tests of the LLM correction layer.

No Ollama is involved: the corrector is exercised with a fake client, and the
context/prompt/sanitising logic is pure Python.
"""
from __future__ import annotations

import pytest

from app.htr.application.correction_context import CorrectionContextBuilder
from app.htr.application.metrics import align_substitutions
from app.htr.domain.entities import ConfusionPair, CorrectionContext
from app.htr.infrastructure.llm.ollama_corrector import OllamaLineCorrector
from app.htr.infrastructure.llm.prompt import build_correction_prompt, sanitize_correction
from tests.htr_fakes import FakePageRepository, make_line, make_page


# ---------------------------------------------------------------------------
# alignment / confusion profile
# ---------------------------------------------------------------------------


def test_align_substitutions_reports_letter_confusions():
    assert align_substitutions("Шналозавод", "Шпалозавод") == [("н", "п")]
    assert align_substitutions("пришел", "пришёл") == [("е", "ё")]
    assert align_substitutions("", "текст") == []
    assert align_substitutions("текст", "") == []


def test_align_substitutions_ignores_insertions_and_deletions():
    assert align_substitutions("доом", "дом") == []
    assert align_substitutions("дом", "доом") == []


# ---------------------------------------------------------------------------
# context builder (Phase 0)
# ---------------------------------------------------------------------------


def test_context_builder_collects_lexicon_confusions_and_examples():
    pages = [
        make_page(
            1,
            lines=[
                make_line(10, predicted="Шналозавод стоит", corrected="Шпалозавод стоит"),
                make_line(11, predicted="Шналозавод виден", corrected="Шпалозавод виден"),
                make_line(12, predicted="он пришел", corrected=None),
            ],
        )
    ]

    context = CorrectionContextBuilder(FakePageRepository(pages)).build(1)

    assert "Шпалозавод" in context.lexicon
    assert "пришел" in context.lexicon
    assert ConfusionPair(recognized="н", correct="п", count=2) in context.confusions
    assert ("Шналозавод стоит", "Шпалозавод стоит") in context.examples


def test_context_builder_takes_ground_truth_from_corrections():
    """The lexicon must never be built from the raw HTR prediction."""
    page = make_page(1, lines=[make_line(10, predicted="нпалозавод", corrected="шпалозавод")])
    context = CorrectionContextBuilder(FakePageRepository([page])).build(1)

    assert "шпалозавод" in context.lexicon
    assert "нпалозавод" not in context.lexicon


def test_context_builder_excludes_the_page_being_corrected():
    pages = [
        make_page(1, lines=[make_line(10, predicted="а", corrected="б")]),
        make_page(2, lines=[make_line(20, predicted="в", corrected="г")]),
    ]

    context = CorrectionContextBuilder(FakePageRepository(pages)).build(1, exclude_page_id=1)

    assert context.examples == [("в", "г")]


def test_context_builder_uses_the_knowledge_vocabulary():
    class FakeVocabulary:
        def vocabulary(self, user_id: int) -> list[str]:
            assert user_id == 1
            return ["Судиславль", "Шпалозавод", "Судиславль"]

    builder = CorrectionContextBuilder(FakePageRepository([make_page(1)]), FakeVocabulary())
    context = builder.build(1)

    assert context.vocabulary == ["Судиславль", "Шпалозавод"]


def test_context_builder_marks_pages_without_corrections():
    page = make_page(1, lines=[make_line(10, predicted="чисто", corrected=None)])
    context = CorrectionContextBuilder(FakePageRepository([page])).build(1)

    assert context.confusions == []
    assert context.examples == []
    assert "чисто" in context.lexicon


# ---------------------------------------------------------------------------
# prompt
# ---------------------------------------------------------------------------


def test_prompt_contains_rules_context_and_target():
    context = CorrectionContext(
        lexicon=["Шпалозавод"],
        vocabulary=["Судиславль"],
        confusions=[ConfusionPair(recognized="н", correct="п", count=3)],
        examples=[("Шналозавод", "Шпалозавод")],
    )

    prompt = build_correction_prompt("Шналозавод виден", ["предыдущая строка"], context)

    assert "СТРОГИЕ ПРАВИЛА" in prompt
    assert "Шпалозавод" in prompt and "Судиславль" in prompt
    assert "н→п (3)" in prompt
    assert "предыдущая строка" in prompt
    assert prompt.rstrip().endswith("Шналозавод виден")


def test_prompt_without_context_is_still_valid():
    prompt = build_correction_prompt("текст", [], CorrectionContext())

    assert prompt.rstrip().endswith("текст")
    assert "СЛОВАРИ" not in prompt


# ---------------------------------------------------------------------------
# answer sanitising
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("домик", "домик"),
        ("  дом  ", "дом"),
        ("```\nдом\n```", "дом"),
        ('"дом"', "дом"),
        ("«дом»", "дом"),
        ("Исправленный текст: дом", "дом"),
        ("стало: дом", "дом"),
        ("- дом", "дом"),
        ("x" * 4, "x" * 4),
    ],
)
def test_sanitize_accepts_repairable_answers(raw, expected):
    assert sanitize_correction(raw, "дом") == expected


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "   ",
        None,
        "совершенно другой текст неизвестно откуда",
        "дом\nкомментарий модели",
    ],
)
def test_sanitize_rejects_unusable_answers(raw):
    assert sanitize_correction(raw, "дом") is None


# ---------------------------------------------------------------------------
# Ollama corrector (fake client, no network)
# ---------------------------------------------------------------------------


class FakeOllamaClient:
    def __init__(self, answer: str | None = None, error: Exception | None = None):
        self.answer = answer
        self.error = error
        self.calls: list[dict] = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return {"response": self.answer}


def make_corrector(client) -> OllamaLineCorrector:
    corrector = OllamaLineCorrector(model="gpt-oss:20b", host="http://ollama.invalid")
    corrector._client = client  # test seam: no real client is created
    return corrector


def test_corrector_returns_sanitized_answer():
    client = FakeOllamaClient(answer="Шпалозавод виден")
    corrector = make_corrector(client)

    assert corrector.correct_line("Шналозавод виден", [], CorrectionContext()) == "Шпалозавод виден"
    assert client.calls[0]["model"] == "gpt-oss:20b"
    assert client.calls[0]["options"]["temperature"] == 0


def test_corrector_keeps_raw_prediction_when_ollama_fails():
    corrector = make_corrector(FakeOllamaClient(error=RuntimeError("connection refused")))

    assert corrector.correct_line("текст", [], CorrectionContext()) is None


def test_corrector_rejects_verbose_answer():
    corrector = make_corrector(
        FakeOllamaClient(answer="Конечно! Вот исправленный текст, который я подготовил для вас")
    )

    assert corrector.correct_line("дом", [], CorrectionContext()) is None


def test_corrector_skips_blank_lines_without_calling_the_model():
    client = FakeOllamaClient(answer="что-то")
    corrector = make_corrector(client)

    assert corrector.correct_line("   ", [], CorrectionContext()) is None
    assert client.calls == []


# ---------------------------------------------------------------------------
# connecting must fail fast: a stopped Ollama drops the connection silently
# ---------------------------------------------------------------------------


def test_connect_timeout_is_separate_from_generation_timeout():
    """One shared budget made a dead Ollama cost 90 s per line."""
    corrector = OllamaLineCorrector(
        model="gemma3:12b", host="http://ollama.invalid", timeout=90.0, connect_timeout=5.0
    )

    # ollama.Client keeps the httpx client (and therefore the Timeout) inside
    timeout = corrector.client._client.timeout

    assert timeout.connect == 5.0
    assert timeout.read == 90.0
    assert timeout.pool == 5.0


class FakeListClient(FakeOllamaClient):
    def __init__(self, error: Exception | None = None):
        super().__init__()
        self.list_error = error
        self.list_calls = 0

    def list(self):
        self.list_calls += 1
        if self.list_error is not None:
            raise self.list_error
        return {"models": []}


def test_availability_probe_does_not_generate():
    client = FakeListClient()
    corrector = make_corrector(client)

    assert corrector.is_available() is True
    assert client.list_calls == 1
    assert client.calls == []  # no generation was attempted


def test_availability_probe_reports_an_unreachable_host():
    corrector = make_corrector(FakeListClient(error=ConnectionError("timed out")))

    assert corrector.is_available() is False


def test_default_connect_timeout_is_short():
    """Guards the 5 s default: 90 s of waiting for a dead host is the bug."""
    corrector = OllamaLineCorrector(model="gemma3:12b", host="http://ollama.invalid")

    assert corrector.connect_timeout <= 10.0
    assert corrector.timeout == 90.0
