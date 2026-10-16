"""Запись рёбер: канонические поля, подтверждения и их слияние.

Одно место, которое знает, как из вида и флагов получается каноническая пара и
как ребро собирается из подтверждений. Им пользуются трое: конвейер (запись),
очистка источника (пересчёт после удаления подтверждений) и слияние сущностей
(пересчёт после переписывания концов). Если бы каноническую пару считали в
нескольких местах, ключ уникальности разошёлся бы с данными при первой правке.
"""
from __future__ import annotations

from typing import Any, Mapping

from sqlalchemy.orm import Session

from ..models import EntityRelation, RelationSource
from .kinship import canon_pair, canon_token, generations_ok, kin_of


def canonical_fields(
    *,
    kind: str | None,
    generations: int | None,
    via_marriage: bool | None,
    bond: str | None,
    from_id: int,
    to_id: int,
    relation_type: str | None = None,
) -> dict[str, Any]:
    """Канонические поля строки ребра — той же функцией, что и уровни.

    ``bond`` и брак, подразумеваемые видом, выставляет код: иначе токен и
    колонки разошлись бы.
    """
    depth = int(generations) if generations_ok(generations) else 0
    kin = kin_of(kind, depth, via_marriage=via_marriage, bond=bond)
    canon_from, canon_to = canon_pair(
        kind, from_id, to_id, depth, via_marriage=via_marriage, bond=bond
    )
    return {
        "relation_kind": (kind or "").strip().lower() or None,
        "generations": int(generations) if generations_ok(generations) else None,
        "via_marriage": kin.via_marriage if kin else bool(via_marriage),
        "bond": kin.bond if kin else bond,
        "canon_from": canon_from,
        "canon_to": canon_to,
        "canon_key": canon_token(
            kind, depth, via_marriage=via_marriage, bond=bond, relation_type=relation_type
        ),
    }


def recompute_canonical(relation: EntityRelation) -> None:
    """Пересчитывает каноническую пару и токен по текущим полям строки."""
    fields = canonical_fields(
        kind=relation.relation_kind,
        generations=relation.generations,
        via_marriage=relation.via_marriage,
        bond=relation.bond,
        from_id=relation.from_entity_id,
        to_id=relation.to_entity_id,
        relation_type=relation.relation_type,
    )
    for name, value in fields.items():
        setattr(relation, name, value)


def confirmation_payload(
    relation: EntityRelation | None, source_id: int, item: Mapping[str, Any], *, quote, start, end
) -> dict[str, Any]:
    """Одно подтверждение так, как его видит ``aggregate_confirmations``."""
    return {
        "source_id": source_id,
        "relation_kind": item.get("relation_kind"),
        "relation_type": item.get("relation_type"),
        "generations": item.get("generations"),
        "via_marriage": item.get("via_marriage"),
        "bond": item.get("bond"),
        "confidence": item.get("confidence"),
        "evidence_text": quote,
        "start_offset": start,
        "end_offset": end,
    }


def apply_aggregate(relation: EntityRelation, aggregate: Mapping[str, Any]) -> None:
    """Переносит в строку представителя и совокупную уверенность."""
    if aggregate.get("relation_type"):
        relation.relation_type = aggregate["relation_type"]
    relation.confidence = aggregate.get("confidence")
    if aggregate.get("evidence_text") is not None:
        relation.evidence_text = aggregate["evidence_text"]
        relation.start_offset = aggregate.get("start_offset")
        relation.end_offset = aggregate.get("end_offset")
    if aggregate.get("primary_source_id") is not None:
        relation.source_id = aggregate["primary_source_id"]
    recompute_canonical(relation)


def upsert_confirmation(
    db: Session, relation: EntityRelation, source_id: int, payload: Mapping[str, Any]
) -> RelationSource:
    """Подтверждение этого источника: повторный прогон обновляет строку."""
    row = (
        db.query(RelationSource)
        .filter(RelationSource.relation_id == relation.id, RelationSource.source_id == source_id)
        .first()
    )
    if row is None:
        # autoflush выключен: строка, добавленная в этом же прогоне, запросу не
        # видна, и без этой проверки flush упал бы на уникальности подтверждения
        for pending in db.new:
            if (
                isinstance(pending, RelationSource)
                and pending.relation_id == relation.id
                and pending.source_id == source_id
            ):
                row = pending
                break
    if row is None:
        row = RelationSource(relation_id=relation.id, source_id=source_id)
        db.add(row)
    row.relation_kind = (payload.get("relation_kind") or None)
    row.relation_type = payload.get("relation_type")
    row.generations = payload.get("generations")
    row.via_marriage = payload.get("via_marriage")
    row.bond = payload.get("bond")
    row.confidence = payload.get("confidence")
    row.evidence_text = payload.get("evidence_text")
    row.start_offset = payload.get("start_offset")
    row.end_offset = payload.get("end_offset")
    return row


def confirmation_rows(db: Session, relation_id: int, *, skip_source: int | None = None) -> list[dict]:
    """Подтверждения ребра в виде, пригодном для ``aggregate_confirmations``."""
    query = db.query(RelationSource).filter(RelationSource.relation_id == relation_id)
    if skip_source is not None:
        query = query.filter(RelationSource.source_id != skip_source)
    return [
        {
            "source_id": row.source_id,
            "relation_kind": row.relation_kind,
            "relation_type": row.relation_type,
            "generations": row.generations,
            "via_marriage": row.via_marriage,
            "bond": row.bond,
            "confidence": row.confidence,
            "evidence_text": row.evidence_text,
            "start_offset": row.start_offset,
            "end_offset": row.end_offset,
        }
        for row in query.all()
    ]


def merge_relation(db: Session, keeper: EntityRelation, loser: EntityRelation) -> None:
    """Переносит подтверждения проигравшего к победителю и удаляет проигравшего.

    Нужно и слиянию сущностей, и склейке рёбер, ставших одинаковыми по
    каноническому ключу: без переноса вместе с ребром пропал бы источник.
    """
    for row in list(loser.confirmations):
        duplicate = (
            db.query(RelationSource)
            .filter(
                RelationSource.relation_id == keeper.id,
                RelationSource.source_id == row.source_id,
            )
            .first()
        )
        if duplicate is None:
            # переносим через коллекции: у подтверждений стоит delete-orphan, и
            # удаление проигравшего унесло бы уже перенесённую строку
            loser.confirmations.remove(row)
            keeper.confirmations.append(row)
        else:
            db.delete(row)
    db.delete(loser)
    db.flush()
