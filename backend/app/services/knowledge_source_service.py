"""Источники знаний: сборка текста, статусы, запуск и переизвлечение.

Разделы 6 и 7.4 ТЗ. Сервис — единственное место, которое знает, что страница
дневника и подтверждённая страница рукописи одинаково являются *источником*
текста: он собирает текст и карту строк, привязывает автора-объект, ведёт статус
обработки, занимает источник перед прогоном, очищает знание предыдущего прогона
и записывает результат в ``extraction_runs``.

Транзакции. ``refresh_*`` и ``mark_pending`` — самостоятельные операции, они
коммитят. ``reset_source_knowledge`` коммитить НЕ должен: он вызывается внутри
прогона, и удаление старого знания обязано откатиться вместе с неудачной
вставкой нового (иначе падение конвейера оставит страницу без знаний вообще).
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Sequence

from sqlalchemy import text as sql_text
from sqlalchemy.orm import Session

from ..config import settings
from ..models import (
    Chunk,
    DiaryPage,
    Entity,
    EntityMention,
    EntityRelation,
    Event,
    ExtractionRun,
    Fact,
    Goal,
    GoalProgress,
    Habit,
    HabitLog,
    HTRAuthor,
    HTRLine,
    HTRPage,
    KnowledgeSource,
    KnowledgeState,
    User,
)
from .extraction_pipeline import ExtractionNotImplemented, run_extraction
from .source_text import assemble_lines, manuscript_title, neighbor_context, text_hash

logger = logging.getLogger(__name__)

SOURCE_DIARY = "diary"
SOURCE_HTR = "htr"

STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"

RUN_RUNNING = "running"
RUN_SUCCEEDED = "succeeded"
RUN_FAILED = "failed"

HTR_CONFIRMED = "CONFIRMED"

#: manuscript authors that mean "the account owner"
SELF_NAMES = ("me", "я", "i")

#: orphaned person objects: nothing references them any more (7.4)
_ORPHAN_ENTITIES_SQL = """
DELETE FROM entities
WHERE entity_type = 'person'
  AND user_id = :user_id
  AND id NOT IN (SELECT self_entity_id FROM users WHERE self_entity_id IS NOT NULL)
  AND NOT EXISTS (SELECT 1 FROM entity_mentions m WHERE m.entity_id = entities.id)
  AND NOT EXISTS (SELECT 1 FROM facts f
                  WHERE f.subject_entity_id = entities.id OR f.object_entity_id = entities.id)
  AND NOT EXISTS (SELECT 1 FROM entity_relations r
                  WHERE r.from_entity_id = entities.id OR r.to_entity_id = entities.id)
  AND NOT EXISTS (SELECT 1 FROM goals g WHERE g.subject_entity_id = entities.id)
  AND NOT EXISTS (SELECT 1 FROM events e WHERE e.subject_entity_id = entities.id)
  AND NOT EXISTS (SELECT 1 FROM habits h WHERE h.subject_entity_id = entities.id)
"""

#: the heads an extraction may create; deleting them keeps orphans from piling up
_OWNED_CHILDREN = (EntityMention, Fact, GoalProgress, Event, HabitLog, Chunk, EntityRelation)

Pipeline = Callable[[Session, KnowledgeSource], Awaitable[dict[str, Any]]]


class SourceError(RuntimeError):
    """The source cannot be built or used as asked."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _normalize(value: str | None) -> str:
    return " ".join((value or "").split()).lower()[:255]


class KnowledgeSourceService:
    def __init__(self, db: Session, pipeline: Pipeline | None = None) -> None:
        self.db = db
        #: the pipeline of stage 4; injectable so the lifecycle can be tested
        self._pipeline = pipeline or run_extraction

    # ------------------------------------------------------------------
    # Objects: "Я" and manuscript authors
    # ------------------------------------------------------------------

    def ensure_self_entity(self, user_id: int) -> Entity:
        """The account's own person-object, created if an old row lacks it."""
        user = self.db.query(User).filter(User.id == user_id).first()
        if user is None:
            raise SourceError(f"user {user_id} does not exist")
        if user.self_entity_id:
            entity = self.db.query(Entity).filter(Entity.id == user.self_entity_id).first()
            if entity is not None:
                return entity
        for candidate in SELF_NAMES + (_normalize(user.name), _normalize(user.username)):
            normalized = _normalize(candidate)
            if not normalized:
                continue
            found = (
                self.db.query(Entity)
                .filter_by(user_id=user.id, entity_type="person", normalized_name=normalized)
                .first()
            )
            if found is not None:
                user.self_entity_id = found.id
                return found
        display_name = (user.name or user.username or "Я").strip()
        entity = Entity(
            user_id=user.id,
            entity_type="person",
            canonical_name=display_name[:255],
            normalized_name=display_name.lower()[:255],
            first_seen_at=_now(),
            last_seen_at=_now(),
        )
        self.db.add(entity)
        self.db.flush()
        user.self_entity_id = entity.id
        return entity

    def ensure_author_entity(self, author: HTRAuthor | None) -> Entity | None:
        """Binds a manuscript author to an object; "me" is the account's object."""
        if author is None:
            return None
        if author.entity_id:
            entity = self.db.query(Entity).filter(Entity.id == author.entity_id).first()
            if entity is not None:
                return entity
        normalized = _normalize(author.name)
        if normalized in SELF_NAMES:
            entity = self.ensure_self_entity(author.user_id)
            author.entity_id = entity.id
            return entity
        entity = (
            self.db.query(Entity)
            .filter_by(user_id=author.user_id, entity_type="person", normalized_name=normalized)
            .first()
        )
        if entity is None:
            entity = Entity(
                user_id=author.user_id,
                entity_type="person",
                canonical_name=(author.name or "")[:255],
                normalized_name=normalized,
                first_seen_at=_now(),
                last_seen_at=_now(),
            )
            self.db.add(entity)
            self.db.flush()
        author.entity_id = entity.id
        return entity

    # ------------------------------------------------------------------
    # Sources
    # ------------------------------------------------------------------

    def get(self, source_type: str, ref_id: int, user_id: int | None = None) -> KnowledgeSource | None:
        query = self.db.query(KnowledgeSource).filter_by(source_type=source_type, ref_id=ref_id)
        if user_id is not None:
            query = query.filter(KnowledgeSource.user_id == user_id)
        return query.first()

    def page_lines(self, page_id: int) -> list[tuple[int, str | None]]:
        """Lines of a manuscript page in reading order, corrected text first."""
        rows = (
            self.db.query(HTRLine.id, HTRLine.corrected_text, HTRLine.predicted_text)
            .filter(HTRLine.page_id == page_id)
            .order_by(HTRLine.order_index, HTRLine.id)
            .all()
        )
        return [(row[0], row[1] or row[2]) for row in rows]

    def refresh_from_diary_page(
        self, page: DiaryPage, *, commit: bool = True
    ) -> tuple[KnowledgeSource, bool, bool]:
        """Creates or refreshes the source of a diary page."""
        author = self.ensure_self_entity(page.user_id)
        source, created, changed = self._upsert(
            user_id=page.user_id,
            source_type=SOURCE_DIARY,
            ref_id=page.id,
            author_entity_id=author.id,
            title=(page.title or None),
            text=page.content or "",
            line_map=None,
        )
        if commit:
            self.db.commit()
        return source, created, changed

    def refresh_from_htr_page(
        self, page: HTRPage, *, commit: bool = True
    ) -> tuple[KnowledgeSource, bool, bool]:
        """Creates or refreshes the source of a *confirmed* manuscript page."""
        if page.status != HTR_CONFIRMED:
            raise SourceError(
                f"page {page.id} is {page.status}: a source is only built for a confirmed page"
            )
        author = self.db.query(HTRAuthor).filter(HTRAuthor.id == page.author_id).first()
        author_entity = self.ensure_author_entity(author)
        page_text, line_map = assemble_lines(self.page_lines(page.id))
        source, created, changed = self._upsert(
            user_id=page.user_id,
            source_type=SOURCE_HTR,
            ref_id=page.id,
            author_entity_id=author_entity.id if author_entity else None,
            title=manuscript_title(page.file_name, page.id),
            text=page_text,
            line_map=line_map,
        )
        if commit:
            self.db.commit()
        return source, created, changed

    def _upsert(
        self,
        *,
        user_id: int,
        source_type: str,
        ref_id: int,
        author_entity_id: int | None,
        title: str | None,
        text: str,
        line_map: str | None,
    ) -> tuple[KnowledgeSource, bool, bool]:
        source = self.get(source_type, ref_id, user_id)
        digest = text_hash(text)
        if source is None:
            source = KnowledgeSource(
                user_id=user_id,
                source_type=source_type,
                ref_id=ref_id,
                author_entity_id=author_entity_id,
                title=title,
                text=text,
                line_map=line_map,
                text_hash=digest,
                extraction_status=STATUS_PENDING,
            )
            self.db.add(source)
            self.db.flush()
            return source, True, True

        changed = source.text_hash != digest
        source.title = title
        source.author_entity_id = author_entity_id or source.author_entity_id
        source.updated_at = _now()
        if changed:
            source.text = text
            source.line_map = line_map
            source.text_hash = digest
            # the knowledge of the previous text is stale: it will be replaced
            source.extraction_status = STATUS_PENDING
            source.error = None
        self.db.flush()
        return source, False, changed

    def mark_pending(self, source: KnowledgeSource, *, drop_knowledge: bool = True) -> dict[str, int]:
        """Returns a source to the queue, optionally dropping its knowledge.

        Used when a page is reopened (6.3): the text is no longer ground truth,
        so the knowledge read from it must not be shown any more. Commits.
        """
        removed = self.reset_source_knowledge(source) if drop_knowledge else {}
        source.extraction_status = STATUS_PENDING
        source.error = None
        self.db.commit()
        if removed:
            logger.info("Source %s reset: %s", source.id, removed)
        return removed

    def delete_source(self, source: KnowledgeSource) -> dict[str, int]:
        """Removes a source and everything derived from it (page deleted). Commits."""
        removed = self.reset_source_knowledge(source)
        self.db.delete(source)
        self.db.commit()
        return removed

    # ------------------------------------------------------------------
    # Re-extraction (7.4)
    # ------------------------------------------------------------------

    def reset_source_knowledge(self, source: KnowledgeSource) -> dict[str, int]:
        """Drops what the previous run of this source produced.

        Does **not** commit: the caller runs it in the same transaction as the
        new extraction, so a failing pipeline restores the old knowledge instead
        of leaving the page empty. Orphaned heads (a goal whose only progress
        came from this source, a habit nobody logs any more, a person nobody
        mentions) go with it.
        """
        removed: dict[str, int] = {}
        for model in _OWNED_CHILDREN:
            deleted = (
                self.db.query(model)
                .filter(model.source_id == source.id)
                .delete(synchronize_session="fetch")
            )
            removed[model.__tablename__] = int(deleted or 0)

        removed["goals"] = int(
            self.db.query(Goal)
            .filter(
                Goal.created_from_source_id == source.id,
                ~self.db.query(GoalProgress).filter(GoalProgress.goal_id == Goal.id).exists(),
            )
            .delete(synchronize_session="fetch")
            or 0
        )
        removed["habits"] = int(
            self.db.query(Habit)
            .filter(~self.db.query(HabitLog).filter(HabitLog.habit_id == Habit.id).exists())
            .delete(synchronize_session="fetch")
            or 0
        )
        result = self.db.execute(sql_text(_ORPHAN_ENTITIES_SQL), {"user_id": source.user_id})
        removed["entities"] = int(result.rowcount or 0)
        # the entities were deleted by raw SQL, so the session still holds them
        self.db.expire_all()
        return removed

    # ------------------------------------------------------------------
    # Neighbour context (6.2)
    # ------------------------------------------------------------------

    def _neighbor_lines(self, page: HTRPage, *, older: bool) -> list[tuple[int, str | None]]:
        query = self.db.query(HTRPage).filter(
            HTRPage.author_id == page.author_id,
            HTRPage.status == HTR_CONFIRMED,
            HTRPage.id != page.id,
        )
        if older:
            neighbor = (
                query.filter(HTRPage.order_index < page.order_index)
                .order_by(HTRPage.order_index.desc(), HTRPage.id.desc())
                .first()
            )
        else:
            neighbor = (
                query.filter(HTRPage.order_index > page.order_index)
                .order_by(HTRPage.order_index.asc(), HTRPage.id.asc())
                .first()
            )
        return self.page_lines(neighbor.id) if neighbor is not None else []

    def context_block(self, source: KnowledgeSource) -> str:
        """Tail of the previous page and head of the next one, for the prompt.

        A sentence can start on one page and end on another; the model needs the
        neighbour text to read it, but must not extract anything from it.
        """
        if source.source_type != SOURCE_HTR:
            return ""
        page = self.db.query(HTRPage).filter(HTRPage.id == source.ref_id).first()
        if page is None:
            return ""
        context = neighbor_context(
            self._neighbor_lines(page, older=True),
            self._neighbor_lines(page, older=False),
            settings.knowledge_neighbor_context_chars,
        )
        if not context:
            return ""
        return "Контекст соседних страниц (НЕ извлекай знания из него):\n" + context

    # ------------------------------------------------------------------
    # Running the pipeline
    # ------------------------------------------------------------------

    def claim(self, source_id: int) -> bool:
        """Takes the source for one run; False when a run is already in flight."""
        updated = (
            self.db.query(KnowledgeSource)
            .filter(
                KnowledgeSource.id == source_id,
                KnowledgeSource.extraction_status != STATUS_RUNNING,
            )
            .update({KnowledgeSource.extraction_status: STATUS_RUNNING}, synchronize_session=False)
        )
        self.db.commit()
        return bool(updated)

    async def run_now(self, source: KnowledgeSource) -> dict[str, Any]:
        """Runs the pipeline over one source, in this process (sync mode)."""
        source_id = source.id
        if not self.claim(source_id):
            logger.info("Source %s is already being extracted; the second run is skipped", source_id)
            return {"source_id": source_id, "status": STATUS_RUNNING, "skipped": "already running"}

        self.db.refresh(source)
        if source.author_entity_id is None:
            return self._release(
                source_id,
                None,
                STATUS_FAILED,
                error="у источника нет автора-объекта: привяжите автора рукописи к объекту",
            )

        run = ExtractionRun(source_id=source_id, status=RUN_RUNNING, model=settings.llm_model)
        self.db.add(run)
        self.db.commit()
        run_id = run.id

        try:
            # old knowledge and the new one live in one transaction: a failure
            # must restore the page, not empty it
            self.reset_source_knowledge(source)
            summary = await self._pipeline(self.db, source)
            self.db.flush()
            self._finish_success(source, run_id, summary)
            self.db.commit()
            logger.info("Source %s extracted: %s", source_id, summary)
            return {"source_id": source_id, "status": STATUS_DONE, "summary": summary}
        except ExtractionNotImplemented as exc:
            self.db.rollback()
            self._release(source_id, run_id, STATUS_PENDING, error=None, drop_run=True)
            logger.info("Source %s left pending: %s", source_id, exc)
            return {"source_id": source_id, "status": STATUS_PENDING, "skipped": str(exc)}
        except Exception as exc:  # noqa: BLE001 - любой сбой конвейера = failed
            self.db.rollback()
            self._release(source_id, run_id, STATUS_FAILED, error=str(exc))
            logger.exception("Extraction failed for source %s", source_id)
            return {"source_id": source_id, "status": STATUS_FAILED, "error": str(exc)}

    async def request_extraction(
        self, source: KnowledgeSource, *, mode: str | None = None
    ) -> dict[str, Any]:
        """Runs the pipeline now (sync) or hands it to Celery (async)."""
        chosen = (mode or settings.knowledge_processing_mode or "sync").strip().lower()
        if chosen == "async":
            # imported lazily: app.tasks imports this module through the service
            from ..tasks import extract_knowledge_task

            task = extract_knowledge_task.delay(source.id)
            logger.info("Source %s queued for extraction (task %s)", source.id, task.id)
            return {
                "source_id": source.id,
                "status": STATUS_PENDING,
                "mode": "async",
                "queued": True,
                "task_id": task.id,
            }
        return await self.run_now(source)

    def request_extraction_sync(
        self, source: KnowledgeSource, *, mode: str | None = None
    ) -> dict[str, Any]:
        """``request_extraction`` for a sync FastAPI route, Celery or a script."""
        return asyncio.run(self.request_extraction(source, mode=mode))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _finish_success(self, source: KnowledgeSource, run_id: int | None, summary: dict[str, Any]) -> None:
        source.extraction_status = STATUS_DONE
        source.last_extracted_at = _now()
        source.error = None
        run = self.db.query(ExtractionRun).filter(ExtractionRun.id == run_id).first()
        if run is not None:
            run.status = RUN_SUCCEEDED
            run.stats = json.dumps(summary, ensure_ascii=False, default=str)
            run.finished_at = _now()
        self._bump_knowledge_version(source.user_id)

    def _bump_knowledge_version(self, user_id: int) -> None:
        """Analytics caches are keyed by this version (spec 10)."""
        state = self.db.query(KnowledgeState).filter(KnowledgeState.user_id == user_id).first()
        if state is None:
            state = KnowledgeState(user_id=user_id, version=0)
            self.db.add(state)
        state.version = (state.version or 0) + 1

    def _release(
        self,
        source_id: int,
        run_id: int | None,
        status: str,
        *,
        error: str | None = None,
        drop_run: bool = False,
    ) -> dict[str, Any]:
        """Records the outcome of a run that did not produce knowledge."""
        source = self.db.query(KnowledgeSource).filter(KnowledgeSource.id == source_id).first()
        if source is not None:
            source.extraction_status = status
            source.error = (error or "")[:2000] or None
        run = (
            self.db.query(ExtractionRun).filter(ExtractionRun.id == run_id).first()
            if run_id is not None
            else None
        )
        if run is not None:
            if drop_run:
                self.db.delete(run)
            else:
                run.status = RUN_FAILED
                run.error = (error or "")[:2000] or None
                run.finished_at = _now()
        self.db.commit()
        return {
            "source_id": source_id,
            "status": status,
            **({"error": error} if error else {}),
        }


def source_status_payload(db: Session, source: KnowledgeSource) -> dict[str, Any]:
    """What the UI polls after a confirmation (8.1)."""
    last_run = (
        db.query(ExtractionRun)
        .filter(ExtractionRun.source_id == source.id)
        .order_by(ExtractionRun.id.desc())
        .first()
    )
    stats: Any = None
    if last_run is not None and last_run.stats:
        try:
            stats = json.loads(last_run.stats)
        except (json.JSONDecodeError, TypeError):
            stats = None
    return {
        "source_id": source.id,
        "source_type": source.source_type,
        "ref_id": source.ref_id,
        "status": source.extraction_status,
        "text_hash": source.text_hash,
        "last_extracted_at": source.last_extracted_at,
        "error": source.error,
        "last_run": (
            {
                "id": last_run.id,
                "status": last_run.status,
                "model": last_run.model,
                "started_at": last_run.started_at,
                "finished_at": last_run.finished_at,
                "error": last_run.error,
                "stats": stats,
            }
            if last_run is not None
            else None
        ),
    }
