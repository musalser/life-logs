"""Точка расширения конвейера извлечения знаний.

Этап 3 (сервис источников) владеет жизненным циклом: собрать текст, занять
источник, очистить знание прошлого прогона, записать результат в
``extraction_runs`` и обновить версию базы знаний. Сам конвейер — вызовы LLM,
дедупликация, запись знаний — реализуется на этапе 4 вот здесь, в
``run_extraction``.

Пока конвейер не реализован, функция поднимает :class:`ExtractionNotImplemented`;
сервис трактует это как «запускать нечего»: источник возвращается в ``pending``,
а запись о прогоне не остаётся. Так подтверждение страницы не выглядит падением
и не помечает источник ``failed`` из-за незавершённой разработки.

Требования к реализации этапа 4:

* работать в транзакции вызывающего: ``reset_source_knowledge`` уже вызван,
  а сервис закоммитит результат (или откатит всё при исключении);
* вернуть ``dict`` со статистикой прогона — он попадёт в ``extraction_runs.stats``;
* писать знания только с ``source_id`` переданного источника и с ``subject_entity_id``
  его автора (см. 7.3 ТЗ), факты — с точными офсетами в ``source.text``.
"""
from __future__ import annotations

from typing import Any, Awaitable, Callable

from sqlalchemy.orm import Session

from ..models import KnowledgeSource


class ExtractionError(RuntimeError):
    """The pipeline failed in a way the caller must record on the source."""


class ExtractionNotImplemented(ExtractionError):
    """The pipeline of stage 4 is not wired in yet."""


async def run_extraction(db: Session, source: KnowledgeSource) -> dict[str, Any]:
    """Reads one source and writes the knowledge derived from it (stage 4)."""
    raise ExtractionNotImplemented(
        "конвейер извлечения ещё не реализован (этап 4 ТЗ); "
        "источник остаётся в статусе pending"
    )


#: what the service calls; tests and the Celery worker may replace it
Pipeline = Callable[[Session, KnowledgeSource], Awaitable[dict[str, Any]]]
