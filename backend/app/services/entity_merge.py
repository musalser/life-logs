"""Ручное слияние объектов знаний (объединение двух персон в одну).

Автоматическая дедупликация не всегда решает: извлечение может создать объект,
который не слился по эмбеддингам, хотя для пользователя это тот же человек,
что и существующий автор. Здесь — явный механизм: все ссылки с «исходного»
объекта переносятся на «целевой», а исходный удаляется.

Переносятся: упоминания, факты (субъект и объект), рёбра графа, цели, события,
привычки, источники, привязки авторов рукописей и (через очистку) кэш
аналитики. Алиасы исходного объекта пополняют алиасы целевого.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from sqlalchemy import or_
from sqlalchemy.orm import Session

from ..models import (
    Entity,
    EntityMention,
    EntityRelation,
    Event,
    Fact,
    Goal,
    Habit,
    HTRAuthor,
    KnowledgeInsight,
    KnowledgeSource,
    KnowledgeState,
    User,
)

logger = logging.getLogger(__name__)


class MergeError(RuntimeError):
    """The merge cannot be performed as asked."""


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_aliases(raw: str | None) -> list[str]:
    if not raw:
        return []
    try:
        values = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []
    return [str(value) for value in values if isinstance(value, str) and value] if isinstance(values, list) else []


def _bump_version(db: Session, user_id: int) -> None:
    state = db.query(KnowledgeState).filter(KnowledgeState.user_id == user_id).first()
    if state is None:
        state = KnowledgeState(user_id=user_id, version=0)
        db.add(state)
    state.version = (state.version or 0) + 1


def _merge_edges(db: Session, user: User, source: Entity, target: Entity) -> None:
    """Re-points the graph edges of ``source`` at ``target``, deduping collisions.

    Reassigning can make two edges identical (``target -> X, "друг"`` from both
    the source's and the target's own text), which would violate
    ``uq_entity_relation_edge``. Every edge touching either object is therefore
    folded onto its post-merge identity first, and only the best one survives.
    """
    edges = (
        db.query(EntityRelation)
        .filter(
            EntityRelation.user_id == user.id,
            or_(
                EntityRelation.from_entity_id.in_([source.id, target.id]),
                EntityRelation.to_entity_id.in_([source.id, target.id]),
            ),
        )
        .all()
    )
    groups: dict[tuple[int, int, str], list[EntityRelation]] = {}
    for edge in edges:
        from_id = target.id if edge.from_entity_id in (source.id, target.id) else edge.from_entity_id
        to_id = target.id if edge.to_entity_id in (source.id, target.id) else edge.to_entity_id
        if from_id == to_id:
            # a self-loop has no meaning on the graph: drop it
            db.delete(edge)
            continue
        groups.setdefault((from_id, to_id, edge.relation_type), []).append(edge)

    for (from_id, to_id, relation_type), group in groups.items():
        group.sort(key=lambda e: (0 if e.confidence is not None else 1, -(e.confidence or 0.0), e.id))
        for duplicate in group[1:]:
            db.delete(duplicate)
    # Deletes must land before the surviving edge is moved onto the now-freed
    # identity: SQLAlchemy runs updates before deletes, so moving a keeper onto
    # the same (from, to, type) a duplicate still occupies would trip the unique
    # constraint mid-flush.
    db.flush()
    for (from_id, to_id, relation_type), group in groups.items():
        keep = group[0]
        keep.from_entity_id = from_id
        keep.to_entity_id = to_id
    db.flush()


def merge_entities(db: Session, user: User, source_id: int, target_id: int) -> Entity:
    """Merges ``source`` into ``target`` and returns the merged (target) object."""
    if source_id == target_id:
        raise MergeError("Нельзя объединить объект с самим собой")

    source = db.query(Entity).filter(Entity.id == source_id, Entity.user_id == user.id).first()
    target = db.query(Entity).filter(Entity.id == target_id, Entity.user_id == user.id).first()
    if source is None or target is None:
        raise MergeError("Объект не найден")
    if source.entity_type != "person" or target.entity_type != "person":
        raise MergeError("Объединять можно только персоны")
    if user.self_entity_id == source.id:
        raise MergeError("Нельзя объединить «Я» с другим объектом")

    _merge_edges(db, user, source, target)

    # simple one-to-one re-pointing; no unique constraints on these columns
    db.query(EntityMention).filter(EntityMention.entity_id == source.id).update(
        {EntityMention.entity_id: target.id}, synchronize_session=False
    )
    db.query(Fact).filter(Fact.subject_entity_id == source.id).update(
        {Fact.subject_entity_id: target.id}, synchronize_session=False
    )
    db.query(Fact).filter(Fact.object_entity_id == source.id).update(
        {Fact.object_entity_id: target.id}, synchronize_session=False
    )
    db.query(Goal).filter(Goal.subject_entity_id == source.id).update(
        {Goal.subject_entity_id: target.id}, synchronize_session=False
    )
    db.query(Event).filter(Event.subject_entity_id == source.id).update(
        {Event.subject_entity_id: target.id}, synchronize_session=False
    )
    db.query(Habit).filter(Habit.subject_entity_id == source.id).update(
        {Habit.subject_entity_id: target.id}, synchronize_session=False
    )
    db.query(KnowledgeSource).filter(KnowledgeSource.author_entity_id == source.id).update(
        {KnowledgeSource.author_entity_id: target.id}, synchronize_session=False
    )
    db.query(HTRAuthor).filter(HTRAuthor.entity_id == source.id).update(
        {HTRAuthor.entity_id: target.id}, synchronize_session=False
    )
    # якорные сущности, которые считали источник своим якорем, переезжают на цель
    db.query(Entity).filter(Entity.anchor_entity_id == source.id).update(
        {Entity.anchor_entity_id: target.id}, synchronize_session=False
    )
    # analytics cache is keyed by subject; drop the source's (target's own will
    # regenerate, and the version bump below marks it stale regardless)
    db.query(KnowledgeInsight).filter(KnowledgeInsight.subject_entity_id == source.id).delete(
        synchronize_session=False
    )

    aliases = {target.canonical_name, source.canonical_name}
    aliases.update(_parse_aliases(target.aliases_json))
    aliases.update(_parse_aliases(source.aliases_json))
    aliases.discard("")
    target.aliases_json = json.dumps(sorted(aliases), ensure_ascii=False) or None
    #: the merged vector is no longer the sum of its parts: recompute lazily
    target.embedding = None
    target.updated_at = _now()

    _bump_version(db, user.id)

    db.delete(source)
    db.commit()
    db.refresh(target)
    logger.info(
        "Merged entity %s (%s) into %s (%s) for user %s",
        source_id, source.canonical_name, target_id, target.canonical_name, user.id,
    )
    return target
