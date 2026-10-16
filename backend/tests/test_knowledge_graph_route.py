"""Тесты графа связей: канонические рёбра, уровни, конфликты и «вне древа».

Маршрут отдаёт весь архив сразу — корень выбирает клиент. Уровни считаются
внутри компоненты родства с постоянным сдвигом, поэтому смена корня не требует
запроса; узлы без родства получают ``level=None`` и уходят в ``unplaced``, а не
встают на ноль по умолчанию.
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.deps import get_current_user, get_db
from app.models import Entity, EntityRelation, KnowledgeState, RelationSource, User
from app.routes import knowledge as knowledge_routes


def person(db, user, name, **kwargs):
    row = Entity(
        user_id=user.id,
        entity_type=kwargs.pop("entity_type", "person"),
        canonical_name=name,
        normalized_name=name.lower(),
        **kwargs,
    )
    db.add(row)
    db.flush()
    return row


def relation(db, user, frm, to, kind, *, label=None, source=None, confidence=0.9, **kwargs):
    edge = EntityRelation(
        user_id=user.id,
        from_entity_id=frm.id,
        to_entity_id=to.id,
        relation_type=label or kind,
        relation_kind=kind,
        confidence=confidence,
        source_id=source.id if source is not None else None,
        **kwargs,
    )
    db.add(edge)
    db.flush()
    if source is not None:
        db.add(
            RelationSource(
                relation_id=edge.id,
                source_id=source.id,
                relation_kind=kind,
                relation_type=label or kind,
                confidence=confidence,
            )
        )
        db.flush()
    return edge


@pytest.fixture()
def client(db_session, user):
    db_session.add(KnowledgeState(user_id=user.id, version=7))
    db_session.commit()
    app = FastAPI()
    app.include_router(knowledge_routes.router)
    app.dependency_overrides[get_db] = lambda: db_session
    app.dependency_overrides[get_current_user] = lambda: user.username
    with TestClient(app) as test_client:
        yield test_client


def test_graph_returns_levels_and_canonical_edges(client, db_session, user):
    """Уровни считаются внутри компоненты, направление ребра — как у источника."""
    grandfather = person(db_session, user, "Дед")
    father = person(db_session, user, "Отец")
    me = person(db_session, user, "Я")
    relation(db_session, user, me, father, "father")
    relation(db_session, user, father, grandfather, "father")
    db_session.commit()

    payload = client.get("/knowledge/graph").json()

    levels = {node["id"]: node["level"] for node in payload["nodes"]}
    assert levels[me.id] == levels[father.id] + 1 == levels[grandfather.id] + 2
    edges = {(edge["from_id"], edge["to_id"]): edge for edge in payload["edges"]}
    # «Отец — это отец Я»: старший вперёд, дельта отрицательная
    assert edges[(me.id, father.id)]["delta"] == -1
    assert edges[(me.id, father.id)]["relation_kind"] == "father"
    assert payload["version"] == 7


def test_graph_reports_two_fathers_with_the_whole_parent_set(client, db_session, user):
    """Слот показывает противоречие, а список родителей остаётся полным."""
    me = person(db_session, user, "Я")
    first = person(db_session, user, "Сергей")
    mother = person(db_session, user, "Елена")
    second = person(db_session, user, "Федосив")
    relation(db_session, user, me, first, "father")
    relation(db_session, user, me, mother, "mother")
    relation(db_session, user, me, second, "father")
    db_session.commit()

    payload = client.get("/knowledge/graph").json()

    reports = [c for c in payload["conflicts"] if c["kind"] == "functional_parent"]
    assert len(reports) == 1
    assert reports[0]["parents"] == sorted([first.id, mother.id, second.id])
    assert reports[0]["duplicated_slots"] == {"father": sorted([first.id, second.id])}


def test_graph_father_and_mother_are_not_a_conflict(client, db_session, user):
    me = person(db_session, user, "Я")
    relation(db_session, user, me, person(db_session, user, "Отец"), "father")
    relation(db_session, user, me, person(db_session, user, "Мать"), "mother")
    db_session.commit()

    payload = client.get("/knowledge/graph").json()

    assert [c for c in payload["conflicts"] if c["kind"] == "functional_parent"] == []


def test_ritual_person_is_unplaced_with_a_reason_not_at_zero(client, db_session, user):
    me = person(db_session, user, "Я")
    godfather = person(db_session, user, "Крёстный")
    relation(db_session, user, me, godfather, "godfather")
    db_session.commit()

    payload = client.get("/knowledge/graph").json()

    levels = {node["id"]: node["level"] for node in payload["nodes"]}
    assert levels[godfather.id] is None
    reasons = {item["entity_id"]: item["reason"] for item in payload["unplaced"]}
    assert reasons[godfather.id] == "ritual"


def test_places_and_dates_do_not_pollute_the_circle_of_contacts(client, db_session, user):
    """«Круг общения» — про людей: место и дата в графе не появляются."""
    person(db_session, user, "Шарья", entity_type="location")
    person(db_session, user, "20 августа 1910 года", entity_type="date")
    me = person(db_session, user, "Я")
    db_session.commit()

    payload = client.get("/knowledge/graph").json()

    assert [node["id"] for node in payload["nodes"]] == [me.id]
    assert [item["entity_id"] for item in payload["unplaced"]] == [me.id]


def test_anchor_pair_is_a_merge_candidate(client, db_session, user):
    me = person(db_session, user, "Я")
    named = person(db_session, user, "Сергей")
    anchor = person(db_session, user, "Отец автора", anchor_entity_id=me.id, anchor_relation="father")
    relation(db_session, user, me, named, "father")
    relation(db_session, user, me, anchor, "father")
    db_session.commit()

    payload = client.get("/knowledge/graph").json()

    pairs = [c for c in payload["conflicts"] if c["kind"] == "anchor_named_pair"]
    assert len(pairs) == 1
    # порядок значим: сначала якорная сущность, потом именованная
    assert pairs[0]["merge_candidates"] == [anchor.id, named.id]
    assert pairs[0]["anchor"] == anchor.id and pairs[0]["named"] == named.id
    anchor_node = [node for node in payload["nodes"] if node["id"] == anchor.id][0]
    assert anchor_node["is_anchor"] is True
    assert anchor_node["anchor_relation"] == "father"


def test_etag_is_the_knowledge_version(client):
    first = client.get("/knowledge/graph")
    assert first.status_code == 200
    etag = first.headers["etag"]
    assert etag == 'W/"7"'

    second = client.get("/knowledge/graph", headers={"If-None-Match": etag})
    assert second.status_code == 304
    assert second.headers["etag"] == etag
