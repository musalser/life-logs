"""Сборка текста источника: страница рукописи -> текст + карта строк.

Единственная реализация того, что миграция ``70d54e2ac8e3`` делает собственной
замороженной копией: конвейер извлечения (этап 3) и backfill обязаны собирать
текст одинаково, иначе офсеты фактов перестанут указывать на строки рукописи.

Карта строк переводит глобальный офсет в тексте источника обратно в строку
страницы, а строка — в координаты на изображении: это и есть «ссылка на
реальный текст» из требований.
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Iterable, Sequence

logger = logging.getLogger(__name__)

#: how the lines of a manuscript page are glued into one text; the separator
#: counts towards the offsets, so it must be one character long
LINE_SEPARATOR = "\n"


def text_hash(text: str) -> str:
    """sha256 of the source text: a change means the knowledge is stale."""
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def assemble_lines(rows: Iterable[tuple[int, str | None]]) -> tuple[str, str]:
    """Ordered (line_id, text) pairs -> (page text, line map JSON).

    ``rows`` must already be in reading order (the repository orders by
    ``order_index``). An empty line keeps its place: dropping it would shift
    every offset after it.
    """
    parts: list[str] = []
    line_map: list[dict] = []
    offset = 0
    for index, (line_id, raw) in enumerate(rows):
        value = raw or ""
        if index:
            parts.append(LINE_SEPARATOR)
            offset += len(LINE_SEPARATOR)
        start = offset
        parts.append(value)
        offset += len(value)
        line_map.append({"line_id": line_id, "start": start, "end": offset})
    return "".join(parts), json.dumps(line_map, ensure_ascii=False)


def parse_line_map(line_map: str | None) -> list[dict]:
    """Reads a stored line map; a broken one is reported, never guessed."""
    if not line_map:
        return []
    try:
        entries = json.loads(line_map)
    except (json.JSONDecodeError, TypeError):
        logger.warning("Unreadable line map: %r", str(line_map)[:200])
        return []
    return entries if isinstance(entries, list) else []


def line_for_offset(line_map: str | None, offset: int | None) -> int | None:
    """The line id that contains a character offset of the source text.

    The map's ``end`` is exclusive. An offset that lands on the separator
    between two lines belongs to the line on its left: that is where the
    sentence the fact was read from started.
    """
    if offset is None:
        return None
    previous: int | None = None
    for entry in parse_line_map(line_map):
        start, end = entry.get("start"), entry.get("end")
        if not isinstance(start, int) or not isinstance(end, int):
            continue
        if start <= offset < end:
            return entry.get("line_id")
        if start <= offset:
            previous = entry.get("line_id")
    return previous


def line_span(line_map: str | None, line_id: int) -> tuple[int, int] | None:
    """The offset range of one line in the source text."""
    for entry in parse_line_map(line_map):
        if entry.get("line_id") == line_id:
            start, end = entry.get("start"), entry.get("end")
            if isinstance(start, int) and isinstance(end, int):
                return start, end
    return None


def line_texts(text: str, line_map: str | None) -> list[tuple[int, str]]:
    """Splits an assembled text back into (line_id, text) pairs."""
    result: list[tuple[int, str]] = []
    for entry in parse_line_map(line_map):
        start, end = entry.get("start"), entry.get("end")
        if isinstance(start, int) and isinstance(end, int):
            result.append((entry.get("line_id"), text[start:end]))
    return result


def chunk_text(text: str, max_chars: int, overlap_chars: int = 0) -> list[tuple[int, int]]:
    """Splits a source text into RAG passages as (start, end) offsets.

    Splitting is done on whitespace near ``max_chars`` so a chunk does not start
    mid-word, and consecutive chunks overlap by ``overlap_chars`` characters
    (a fact can sit exactly on a boundary). Short texts stay one chunk.
    """
    body = text or ""
    if not body.strip():
        return []
    if max_chars <= 0 or len(body) <= max_chars:
        return [(0, len(body))]

    spans: list[tuple[int, int]] = []
    start = 0
    while start < len(body):
        end = min(len(body), start + max_chars)
        if end < len(body):
            cut = body.rfind(" ", start, end)
            if cut > start:
                end = cut
        spans.append((start, end))
        if end >= len(body):
            break
        start = max(end - overlap_chars, start + 1) if overlap_chars > 0 else end
    return spans


def manuscript_title(file_name: str | None, page_id: int) -> str:
    """How a manuscript page is named in the UI when the file name is unknown."""
    return (file_name or "").strip() or f"Стр. #{page_id}"


def neighbor_context(
    before: Sequence[tuple[int, str | None]] | None,
    after: Sequence[tuple[int, str | None]] | None,
    limit: int,
) -> str:
    """Tail of the previous page and head of the next one, for extraction.

    A sentence can start at the bottom of one page and end at the top of the
    next; the LLM gets that context to read it correctly, while the extracted
    knowledge is still attributed to the current page only.
    """
    if limit <= 0:
        return ""
    blocks: list[str] = []
    if before:
        tail = LINE_SEPARATOR.join((value or "") for _, value in before)
        if tail.strip():
            blocks.append("…" + tail[-limit:])
    if after:
        head = LINE_SEPARATOR.join((value or "") for _, value in after)
        if head.strip():
            blocks.append(head[:limit] + "…")
    return "\n".join(blocks)
