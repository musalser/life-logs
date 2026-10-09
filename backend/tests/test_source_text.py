"""Тесты сборки текста источника: карта строк, офсеты, чанки RAG.

Это фундамент «ссылок на реальный текст»: если офсет перестанет указывать на
строку рукописи, факт будет подсвечиваться не там, где он написан.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from app.services.source_text import (
    assemble_lines,
    chunk_text,
    line_for_offset,
    line_span,
    line_texts,
    manuscript_title,
    neighbor_context,
    parse_line_map,
    text_hash,
)

BACKEND_DIR = Path(__file__).resolve().parents[1]


def test_assemble_lines_counts_the_separator_in_offsets():
    text, line_map = assemble_lines([(703, "первая"), (704, "вторая")])

    assert text == "первая\nвторая"
    assert [entry["start"] for entry in parse_line_map(line_map)] == [0, 7]
    assert [entry["end"] for entry in parse_line_map(line_map)] == [6, 13]
    assert line_texts(text, line_map) == [(703, "первая"), (704, "вторая")]


def test_assemble_lines_keeps_empty_lines_in_place():
    text, line_map = assemble_lines([(1, "верх"), (2, ""), (3, "низ")])

    assert text == "верх\n\nниз"
    assert line_texts(text, line_map) == [(1, "верх"), (2, ""), (3, "низ")]


def test_assemble_lines_handles_missing_and_empty_input():
    assert assemble_lines([]) == ("", "[]")
    text, line_map = assemble_lines([(5, None)])
    assert text == ""
    assert line_texts(text, line_map) == [(5, "")]


def test_line_for_offset_inside_a_line():
    text, line_map = assemble_lines([(10, "abcdef"), (11, "ghij")])

    assert line_for_offset(line_map, 0) == 10
    assert line_for_offset(line_map, 5) == 10
    assert line_for_offset(line_map, 7) == 11
    assert line_for_offset(line_map, 10) == 11


def test_line_for_offset_lands_on_a_separator():
    _, line_map = assemble_lines([(10, "abc"), (11, "def")])

    # офсет 3 — это "\n": относим его к строке слева, там начиналось предложение
    assert line_for_offset(line_map, 3) == 10


def test_line_for_offset_tolerates_broken_input():
    assert line_for_offset(None, 3) is None
    assert line_for_offset("не json", 3) is None
    assert line_for_offset('[{"line_id": 1, "start": "нет", "end": 2}]', 1) is None
    assert line_for_offset('[{"line_id": 1, "start": 0, "end": 3}]', None) is None


def test_line_span_and_round_trip():
    text, line_map = assemble_lines([(1, "раз"), (2, "два-три")])

    assert line_span(line_map, 2) == (4, 11)
    assert text[slice(*line_span(line_map, 2))] == "два-три"
    assert line_span(line_map, 99) is None


def test_text_hash_changes_with_the_text():
    assert text_hash("abc") == text_hash("abc")
    assert text_hash("abc") != text_hash("abc ")
    assert len(text_hash("")) == 64


def test_chunk_text_keeps_a_short_source_whole():
    body = "короткая запись"
    assert chunk_text(body, 1200) == [(0, len(body))]
    assert chunk_text("   ", 1200) == []


def test_chunk_text_splits_on_whitespace_with_overlap():
    body = " ".join(f"слово{index}" for index in range(60))

    spans = chunk_text(body, max_chars=100, overlap_chars=20)

    assert len(spans) > 1
    for start, end in spans:
        assert end > start
        assert not body[start:end].startswith(" ")
        assert not body[start:end].endswith(" ")
    # перекрытие: следующий кусок начинается раньше конца предыдущего
    assert spans[1][0] < spans[0][1]
    # текст покрыт целиком, ничего не потеряно
    assert spans[0][0] == 0
    assert spans[-1][1] == len(body)


def test_manuscript_title_falls_back_to_the_page_number():
    assert manuscript_title("scan_01.jpg", 12) == "scan_01.jpg"
    assert manuscript_title("  ", 12) == "Стр. #12"
    assert manuscript_title(None, 12) == "Стр. #12"


def test_neighbor_context_takes_tail_and_head():
    context = neighbor_context(
        before=[(1, "конец прошлой страницы")],
        after=[(2, "начало следующей")],
        limit=10,
    )

    assert context.startswith("…")
    assert "й страницы" in context  # ровно хвост предыдущей страницы
    assert "начало сле" in context  # голова следующей
    assert neighbor_context(before=None, after=None, limit=10) == ""
    assert neighbor_context(before=[(1, "x")], after=None, limit=0) == ""


def test_migration_and_runtime_assemble_identically():
    """Миграция несёт замороженную копию сборки — она обязана совпадать.

    Если разойдутся, у страниц, перенесённых миграцией, офсеты фактов укажут на
    другое место, чем у страниц, добавленных после неё.
    """
    candidates = sorted((BACKEND_DIR / "revisions" / "versions").glob("*knowledge_sources_perspective*.py"))
    if not candidates:
        pytest.skip("миграция knowledge_sources не найдена")

    spec = importlib.util.spec_from_file_location("knowledge_migration", candidates[0])
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    rows = [(703, "первая строка"), (704, ""), (705, None), (706, "четвёртая")]
    assert module._assemble_lines(rows) == assemble_lines(rows)
