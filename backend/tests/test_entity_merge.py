"""Тесты ручного слияния объектов: перенос ссылок, алиасы, дедуп рёбер, защита."""
from __future__ import annotations

import json

import pytest

from app.models import (
    Entity,
    EntityMention,
    EntityRelation,
    Event,
    Fact,
    Goal,
    Habit,
    HTRAuthor,
    KnowledgeSource,
    RelationSource,
    User,
)
from app.services.entity_merge import MergeError, merge_entities


def make_entity(db, user, name):
    entity = Entity(
        user_id=user.id,
        entity_type="person",
        canonical_name=name,
        normalized_name=" ".join(name.split()).lower(),
    )
    db.add(entity)
    db.commit()
    db.refresh(entity)
    return entity


def make_source(db, user, ref_id=1):
    source = KnowledgeSource(
        user_id=user.id,
        source_type="diary",
        ref_id=ref_id,
        author_entity_id=user.self_entity_id,
        title="t",
        text="text",
        text_hash="h",
        extraction_status="done",
    )
    db.add(source)
    db.commit()
    db.refresh(source)
    return source


@pytest.fixture()
def self_entity(db_session, user):
    entity = make_entity(db_session, user, "Я")
    user.self_entity_id = entity.id
    db_session.commit()
    return entity


def test_merge_moves_all_references_and_deletes_source(db_session, user, self_entity):
    source = make_entity(db_session, user, "Баба Галя (копия)")
    target = make_entity(db_session, user, "Баба Галя")
    ks = make_source(db_session, user)

    mention = EntityMention(entity_id=source.id, source_id=ks.id, mention_text="Галя")
    db_session.add(mention)
    fact = Fact(
        user_id=user.id,
        source_id=ks.id,
        subject_entity_id=source.id,
        object_entity_id=source.id,
        predicate="жила",
        object_text=None,
        source_text="жила в доме",
    )
    db_session.add(fact)
    goal = Goal(user_id=user.id, subject_entity_id=source.id, title="цель", status="active")
    db_session.add(goal)
    event = Event(user_id=user.id, source_id=ks.id, subject_entity_id=source.id, title="событие")
    db_session.add(event)
    habit = Habit(user_id=user.id, subject_entity_id=source.id, title="привычка")
    db_session.add(habit)
    author = HTRAuthor(user_id=user.id, name="Баба Галя", entity_id=source.id)
    db_session.add(author)
    author_source = KnowledgeSource(
        user_id=user.id,
        source_type="htr",
        ref_id=2,
        author_entity_id=source.id,
        title="t2",
        text="text2",
        extraction_status="done",
    )
    db_session.add(author_source)
    db_session.commit()

    merged = merge_entities(db_session, user, source.id, target.id)

    assert merged.id == target.id
    assert db_session.query(Entity).filter(Entity.id == source.id).first() is None
    assert db_session.query(EntityMention).filter(EntityMention.id == mention.id).one().entity_id == target.id
    assert db_session.query(Fact).filter(Fact.id == fact.id).one().subject_entity_id == target.id
    assert db_session.query(Fact).filter(Fact.id == fact.id).one().object_entity_id == target.id
    assert db_session.query(Goal).filter(Goal.id == goal.id).one().subject_entity_id == target.id
    assert db_session.query(Event).filter(Event.id == event.id).one().subject_entity_id == target.id
    assert db_session.query(Habit).filter(Habit.id == habit.id).one().subject_entity_id == target.id
    assert db_session.query(HTRAuthor).filter(HTRAuthor.id == author.id).one().entity_id == target.id
    assert db_session.query(KnowledgeSource).filter(KnowledgeSource.id == author_source.id).one().author_entity_id == target.id


def test_merge_collects_aliases(db_session, user, self_entity):
    source = make_entity(db_session, user, "Галя")
    source.aliases_json = json.dumps(["Галина"], ensure_ascii=False)
    target = make_entity(db_session, user, "Баба Галя")
    target.aliases_json = json.dumps(["Бабушка"], ensure_ascii=False)
    db_session.commit()

    merged = merge_entities(db_session, user, source.id, target.id)

    aliases = json.loads(merged.aliases_json)
    assert set(aliases) == {"Баба Галя", "Галя", "Галина", "Бабушка"}


def test_merge_dedupes_edges(db_session, user, self_entity):
    source = make_entity(db_session, user, "Копия")
    target = make_entity(db_session, user, "Автор")
    other = make_entity(db_session, user, "Друг")

    db_session.add(
        EntityRelation(
            user_id=user.id, from_entity_id=source.id, to_entity_id=other.id, relation_type="друг"
        )
    )
    db_session.add(
        EntityRelation(
            user_id=user.id, from_entity_id=target.id, to_entity_id=other.id, relation_type="друг"
        )
    )
    db_session.commit()

    merge_entities(db_session, user, source.id, target.id)

    edges = db_session.query(EntityRelation).filter(EntityRelation.to_entity_id == other.id).all()
    assert len(edges) == 1
    assert edges[0].from_entity_id == target.id


def test_merge_rejects_self_into_other(db_session, user, self_entity):
    target = make_entity(db_session, user, "Другой")
    with pytest.raises(MergeError):
        merge_entities(db_session, user, self_entity.id, target.id)


def test_merge_rejects_same_object(db_session, user, self_entity):
    with pytest.raises(MergeError):
        merge_entities(db_session, user, self_entity.id, self_entity.id)


def test_merge_rejects_missing(db_session, user, self_entity):
    target = make_entity(db_session, user, "Есть")
    with pytest.raises(MergeError):
        merge_entities(db_session, user, 99999, target.id)


# ---------------------------------------------------------------------------
# Канонический ключ при слиянии
# ---------------------------------------------------------------------------


def test_merge_with_common_neighbour_keeps_one_edge_and_two_confirmations(db_session, user):
    """Общий сосед после слияния — одно ребро и оба подтверждения."""
    source = make_entity(db_session, user, "Серёжа")
    target = make_entity(db_session, user, "Сергей")
    friend = make_entity(db_session, user, "Маша")
    first = make_source(db_session, user, ref_id=11)
    second = make_source(db_session, user, ref_id=12)
    edge_a = EntityRelation(
        user_id=user.id, from_entity_id=source.id, to_entity_id=friend.id,
        relation_type="подруга", relation_kind="friend", source_id=first.id, confidence=0.5,
    )
    edge_b = EntityRelation(
        user_id=user.id, from_entity_id=target.id, to_entity_id=friend.id,
        relation_type="друг", relation_kind="friend", source_id=second.id, confidence=0.9,
    )
    db_session.add_all([edge_a, edge_b])
    db_session.flush()
    db_session.add_all([
        RelationSource(relation_id=edge_a.id, source_id=first.id, relation_kind="friend",
                       relation_type="подруга", confidence=0.5),
        RelationSource(relation_id=edge_b.id, source_id=second.id, relation_kind="friend",
                       relation_type="друг", confidence=0.9),
    ])
    db_session.commit()

    merge_entities(db_session, user, source.id, target.id)

    edges = db_session.query(EntityRelation).filter(EntityRelation.to_entity_id == friend.id).all()
    assert len(edges) == 1
    assert db_session.query(RelationSource).filter(RelationSource.relation_id == edges[0].id).count() == 2


def test_merge_drops_a_self_loop(db_session, user):
    """Ребро, ставшее петлёй (автор → автор), на графе смысла не имеет."""
    source = make_entity(db_session, user, "Серёжа")
    target = make_entity(db_session, user, "Сергей")
    db_session.add(
        EntityRelation(
            user_id=user.id, from_entity_id=source.id, to_entity_id=target.id,
            relation_type="брат", relation_kind="brother",
        )
    )
    db_session.commit()

    merge_entities(db_session, user, source.id, target.id)

    assert (
        db_session.query(EntityRelation)
        .filter(EntityRelation.from_entity_id == target.id, EntityRelation.to_entity_id == target.id)
        .count()
        == 0
    )
