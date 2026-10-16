"""Факты, связанные с объектом: как субъект, как дополнение или как упоминание.

Факт относится к объекту тремя способами, и каждый даёт свою причину совпадения:

* ``subject`` — объект стоит субъектом факта (``subject_entity_id``);
* ``object`` — объект стоит дополнением (``object_entity_id``);
* ``mention`` — в той же цитате источника есть упоминание объекта: офсеты
  упоминания пересекаются с офсетами факта.

Имя и алиасы дополнительно ищутся как подстрока в цитате и в ``object_text``
(``text``): модель не всегда связывает дополнение с объектом, но имя в тексте
остаётся. Короткие имена (меньше трёх символов) не ищутся, иначе совпадёт всё.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from ..models import Entity, EntityMention, Fact

MIN_NAME_LENGTH = 3

REASON_SUBJECT = "subject"
REASON_OBJECT = "object"
REASON_MENTION = "mention"
REASON_TEXT = "text"


@dataclass
class EntityFactMatch:
    fact: Fact
    reasons: list[str] = field(default_factory=list)


def _names(entity: Entity) -> list[str]:
    """Каноническое имя и алиасы, без дублей и слишком коротких."""
    raw = [entity.canonical_name]
    try:
        raw += [str(alias) for alias in json.loads(entity.aliases_json or "[]")]
    except (json.JSONDecodeError, TypeError):
        pass
    seen: set[str] = set()
    names: list[str] = []
    for value in raw:
        name = " ".join((value or "").split())
        key = name.casefold()
        if len(name) < MIN_NAME_LENGTH or key in seen:
            continue
        seen.add(key)
        names.append(name)
    return names


def _overlaps(a_start, a_end, b_start, b_end) -> bool:
    if None in (a_start, a_end, b_start, b_end):
        return False
    return a_start < b_end and b_start < a_end


def facts_for_entity(db: Session, user_id: int, entity: Entity) -> list[EntityFactMatch]:
    """Все факты пользователя, в которых участвует объект, с причинами совпадения."""
    matches: dict[int, EntityFactMatch] = {}

    def add(fact: Fact, reason: str) -> None:
        match = matches.setdefault(fact.id, EntityFactMatch(fact=fact))
        if reason not in match.reasons:
            match.reasons.append(reason)

    by_link = (
        db.query(Fact)
        .filter(Fact.user_id == user_id)
        .filter((Fact.subject_entity_id == entity.id) | (Fact.object_entity_id == entity.id))
        .all()
    )
    for fact in by_link:
        if fact.subject_entity_id == entity.id:
            add(fact, REASON_SUBJECT)
        if fact.object_entity_id == entity.id:
            add(fact, REASON_OBJECT)

    mentions = (
        db.query(EntityMention)
        .filter(EntityMention.entity_id == entity.id)
        .all()
    )
    mentioned_sources: dict[int, list[EntityMention]] = {}
    for mention in mentions:
        mentioned_sources.setdefault(mention.source_id, []).append(mention)
    if mentioned_sources:
        candidates = (
            db.query(Fact)
            .filter(Fact.user_id == user_id, Fact.source_id.in_(list(mentioned_sources)))
            .all()
        )
        for fact in candidates:
            if any(
                _overlaps(fact.start_offset, fact.end_offset, m.start_offset, m.end_offset)
                for m in mentioned_sources[fact.source_id]
            ):
                add(fact, REASON_MENTION)

    names = _names(entity)
    if names:
        for fact in db.query(Fact).filter(Fact.user_id == user_id).all():
            haystacks = [fact.source_text or "", fact.object_text or ""]
            if any(name.casefold() in text.casefold() for name in names for text in haystacks):
                add(fact, REASON_TEXT)

    return sorted(matches.values(), key=lambda match: match.fact.id, reverse=True)
