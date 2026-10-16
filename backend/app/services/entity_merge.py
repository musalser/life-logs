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
    RelationSource,
    User,
)
from .kinship import aggregate_confirmations
from .relation_store import apply_aggregate, canonical_fields, confirmation_rows

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
    """Re-points the graph edges of ``source`` at ``target``, recomputing canon.

    Переписывание концов меняет каноническую пару и токен, а они хранятся
    колонками: без пересчёта той же функцией, что и при записи, уникальность
    упадёт на первом же слиянии. Рёбра, ставшие одинаковыми по каноническому
    ключу, склеиваются вместе с подтверждениями, петли (автор → автор)
    удаляются, а концы переносятся **после** удаления дублей — SQLAlchemy
    выполняет обновления раньше удалений, и иначе flush упал бы на уникальности.
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
    planned: list[tuple[EntityRelation, int, int, dict]] = []
    for edge in edges:
        from_id = target.id if edge.from_entity_id in (source.id, target.id) else edge.from_entity_id
        to_id = target.id if edge.to_entity_id in (source.id, target.id) else edge.to_entity_id
        if from_id == to_id:
            db.delete(edge)
            continue
        fields = canonical_fields(
            kind=edge.relation_kind,
            generations=edge.generations,
            via_marriage=edge.via_marriage,
            bond=edge.bond,
            from_id=from_id,
            to_id=to_id,
            relation_type=edge.relation_type,
        )
        planned.append((edge, from_id, to_id, fields))

    groups: dict[tuple[int, int, str], list[tuple[EntityRelation, int, int, dict]]] = {}
    for item in planned:
        fields = item[3]
        key = (fields["canon_from"], fields["canon_to"], fields["canon_key"])
        groups.setdefault(key, []).append(item)

    keepers: list[tuple[EntityRelation, int, int, dict]] = []
    for group in groups.values():
        group.sort(key=lambda item: (0 if item[0].confidence is not None else 1, -(item[0].confidence or 0.0), item[0].id))
        keeper = group[0]
        keepers.append(keeper)
        for _edge, _from, _to, _fields in group[1:]:
            loser = _edge
            for row in list(loser.confirmations):
                duplicate = (
                    db.query(RelationSource)
                    .filter(
                        RelationSource.relation_id == keeper[0].id,
                        RelationSource.source_id == row.source_id,
                    )
                    .first()
                )
                if duplicate is None:
                    loser.confirmations.remove(row)
                    keeper[0].confirmations.append(row)
                else:
                    db.delete(row)
            db.delete(loser)
    db.flush()

    for edge, from_id, to_id, fields in keepers:
        edge.from_entity_id = from_id
        edge.to_entity_id = to_id
        for name, value in fields.items():
            setattr(edge, name, value)
        aggregate = aggregate_confirmations(confirmation_rows(db, edge.id))
        if aggregate is not None:
            apply_aggregate(edge, aggregate)
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
