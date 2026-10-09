import asyncio
import logging
from datetime import datetime, timezone
from typing import Mapping, Sequence

from .celery_app import celery
from .services.ollama_client import generate_reply

logger = logging.getLogger(__name__)


class NERExtractionError(Exception):
    """Raised when NER extraction fails before persistence."""

@celery.task(name="app.tasks.generate_reply")
def generate_reply_task(
    message: str,
    tone: str,
    history: Sequence[Mapping[str, str]] | None = None,
) -> str:
    return asyncio.run(generate_reply(message, tone, history))


@celery.task(name="app.tasks.extract_knowledge", bind=True, max_retries=1)
def extract_knowledge_task(self, source_id: int) -> dict:
    """Extracts the knowledge of one source (async mode of the pipeline).

    The worker owns its own session and its own singletons (LLM adapter,
    embedding service), and the synchronous entry point wraps the async
    pipeline — exactly like the chat task above.
    """
    from .db import SessionLocal
    from .models import KnowledgeSource
    from .services.knowledge_source_service import KnowledgeSourceService

    db = SessionLocal()
    try:
        source = (
            db.query(KnowledgeSource).filter(KnowledgeSource.id == source_id).first()
        )
        if source is None:
            logger.warning("extract_knowledge: source %s not found", source_id)
            return {"source_id": source_id, "status": "missing"}
        return asyncio.run(KnowledgeSourceService(db).run_now(source))
    finally:
        db.close()


# Process-level NER singleton – initialised once per Celery worker process.
_ner_service = None


def _get_ner():
    global _ner_service
    if _ner_service is None:
        from .services.ner_client import NERService
        _ner_service = NERService()
    return _ner_service


def extract_entities_sync(
    diary_page_id: int,
    user_id: int,
    content: str,
    *,
    task_id: str | None = None,
) -> dict[str, int]:
    from .config import settings
    from .db import SessionLocal
    from .models import Entity, EntityMention, KnowledgeSource

    logger.info(
        "Starting extract_entities task_id=%s diary_page_id=%s user_id=%s",
        task_id,
        diary_page_id,
        user_id,
    )

    try:
        ner = _get_ner()
        entities = ner.extract_entities(content, min_score=settings.ner_min_score)
    except Exception as exc:
        logger.exception(
            "NER extraction failed task_id=%s diary_page_id=%s",
            task_id,
            diary_page_id,
        )
        raise NERExtractionError("NER extraction failed") from exc

    db = SessionLocal()
    try:
        # mentions hang off a knowledge source now, not off the diary page
        source = (
            db.query(KnowledgeSource)
            .filter_by(user_id=user_id, source_type="diary", ref_id=diary_page_id)
            .first()
        )
        if source is None:
            logger.warning(
                "No knowledge source for diary page %s (task_id=%s); "
                "run `python -m app.scripts.backfill_sources`",
                diary_page_id,
                task_id,
            )
            return {"entities_extracted": 0}

        now = datetime.now(timezone.utc)
        for ent in entities:
            normalized = ent["mention_text"].lower().strip()

            entity = (
                db.query(Entity)
                .filter_by(
                    user_id=user_id,
                    entity_type=ent["entity_type"],
                    normalized_name=normalized,
                )
                .first()
            )
            if entity is None:
                entity = Entity(
                    user_id=user_id,
                    entity_type=ent["entity_type"],
                    canonical_name=ent["mention_text"],
                    normalized_name=normalized,
                    first_seen_at=now,
                    last_seen_at=now,
                )
                db.add(entity)
                db.flush()
            else:
                entity.last_seen_at = now

            mention = EntityMention(
                entity_id=entity.id,
                source_id=source.id,
                mention_text=ent["mention_text"],
                sentence_text=ent["sentence"],
                start_offset=ent["start"],
                end_offset=ent["end"],
                confidence=ent["score"],
            )
            db.add(mention)

        db.commit()
        result = {"entities_extracted": len(entities)}
        logger.info(
            "Completed extract_entities task_id=%s diary_page_id=%s extracted=%s",
            task_id,
            diary_page_id,
            result["entities_extracted"],
        )
        return result
    except Exception:
        db.rollback()
        logger.exception(
            "Database write failed in extract_entities task_id=%s diary_page_id=%s",
            task_id,
            diary_page_id,
        )
        raise
    finally:
        db.close()


@celery.task(name="app.tasks.extract_entities", bind=True, max_retries=2)
def extract_entities_task(self, diary_page_id: int, user_id: int, content: str) -> dict[str, int]:
    try:
        return extract_entities_sync(
            diary_page_id,
            user_id,
            content,
            task_id=self.request.id,
        )
    except NERExtractionError as exc:
        logger.exception(
            "extract_entities retry task_id=%s diary_page_id=%s",
            self.request.id,
            diary_page_id,
        )
        raise self.retry(exc=exc, countdown=10)
