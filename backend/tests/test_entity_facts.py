"""Факты по объекту: субъект, дополнение, упоминание в цитате и имя в тексте.

Поиск не должен цеплять лишнее: короткие имена не ищутся подстрокой, а факт
другого пользователя не попадает в выдачу ни по одной причине.
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.deps import get_current_user, get_db
from app.models import Entity, EntityMention, Fact, KnowledgeSource, KnowledgeState, User
from app.routes import knowledge as knowledge_routes
from app.services.entity_facts import facts_for_entity


def person(db, user, name, aliases=None):
    row = Entity(
        user_id=user.id,
        entity_type="person",
        canonical_name=name,
        normalized_name=name.lower(),
        aliases_json=aliases,
    )
    db.add(row)
    db.flush()
    return row


def source(db, user, text, ref_id=1):
    row = KnowledgeSource(user_id=user.id, source_type="diary", ref_id=ref_id, text=text)
    db.add(row)
    db.flush()
    return row


def reasons_by_fact(matches):
    return {match.fact.predicate: match.reasons for match in matches}


@pytest.fixture()
def client(db_session, user):
    db_session.add(KnowledgeState(user_id=user.id, version=1))
    db_session.commit()
    app = FastAPI()
    app.include_router(knowledge_routes.router)
    app.dependency_overrides[get_db] = lambda: db_session
    app.dependency_overrides[get_current_user] = lambda: user.username
    with TestClient(app) as test_client:
        yield test_client


def test_subject_object_and_mention_reasons(db_session, user):
    grandmother = person(db_session, user, "Баба Галя")
    father = person(db_session, user, "Отец автора")
    text = "Баба Галя родилась в 1947 году. Отец автора работал плотником."
    src = source(db_session, user, text)
    birth_start = text.index("родилась")
    db_session.add_all([
        Fact(
            user_id=user.id, source_id=src.id, subject_entity_id=grandmother.id,
            predicate="родилась", object_text="в 1947 году", source_text="родилась в 1947 году",
            start_offset=birth_start, end_offset=birth_start + len("родилась в 1947 году"),
        ),
        Fact(
            user_id=user.id, source_id=src.id, subject_entity_id=father.id,
            predicate="работал", object_text="плотником", source_text="работал плотником",
            start_offset=text.index("работал"), end_offset=len(text),
        ),
    ])
    db_session.add(EntityMention(
        entity_id=grandmother.id, source_id=src.id, mention_text="Баба Галя",
        start_offset=0, end_offset=len("Баба Галя"),
    ))
    db_session.commit()

    matches = facts_for_entity(db_session, user.id, grandmother)
    reasons = reasons_by_fact(matches)

    assert reasons["родилась"] == ["subject"]
    assert "работал" not in reasons, "факт отца не должен попадать в выдачу бабушки"


def test_mention_matches_only_overlapping_quote(db_session, user):
    obj = person(db_session, user, "Виталий")
    text = "Виталий служил в Севастополе. Потом переехал в Васильков."
    src = source(db_session, user, text)
    moved = text.index("переехал")
    db_session.add_all([
        Fact(user_id=user.id, source_id=src.id, predicate="служил",
             source_text="Виталий служил в Севастополе", start_offset=0,
             end_offset=len("Виталий служил в Севастополе")),
        Fact(user_id=user.id, source_id=src.id, predicate="переехал", source_text="переехал в Васильков",
             start_offset=moved, end_offset=moved + len("переехал в Васильков")),
        EntityMention(entity_id=obj.id, source_id=src.id, mention_text="Виталий",
                      start_offset=0, end_offset=len("Виталий")),
    ])
    db_session.commit()

    reasons = reasons_by_fact(facts_for_entity(db_session, user.id, obj))
    assert reasons["служил"] == ["mention", "text"]
    assert "переехал" not in reasons, "упоминание не должно цеплять факт из другого места страницы"


def test_object_link_and_alias_text_match(db_session, user):
    mother = person(db_session, user, "Елена Юрьевна", aliases='["мама", "Лена"]')
    father = person(db_session, user, "Сергей")
    src = source(db_session, user, "Мама варила варенье.")
    db_session.add_all([
        Fact(user_id=user.id, source_id=src.id, subject_entity_id=father.id, object_entity_id=mother.id,
             predicate="женат", source_text="женат на Елене Юрьевне"),
        Fact(user_id=user.id, source_id=src.id, subject_entity_id=father.id, predicate="варил",
             source_text="Мама варила варенье", object_text="варенье"),
    ])
    db_session.commit()

    reasons = reasons_by_fact(facts_for_entity(db_session, user.id, mother))
    assert reasons["женат"] == ["object"]
    assert reasons["варил"] == ["text"], "алиас «мама» в цитате должен находить факт"


def test_short_names_are_not_searched_as_substrings(db_session, user):
    tiny = person(db_session, user, "Ян")
    src = source(db_session, user, "Пришли гости, была рыбалка на Яне.")
    db_session.add(Fact(user_id=user.id, source_id=src.id, predicate="ловил", source_text="была рыбалка"))
    db_session.commit()

    assert facts_for_entity(db_session, user.id, tiny) == []


def test_other_users_facts_never_match(db_session, user):
    other = User(username="other", password_hash="x", name="Other")
    db_session.add(other)
    db_session.commit()
    obj = person(db_session, user, "Баба Галя")
    other_obj = person(db_session, other, "Баба Галя")
    src = source(db_session, other, "Баба Галя родилась в 1947 году.")
    db_session.add(Fact(user_id=other.id, source_id=src.id, subject_entity_id=other_obj.id,
                        predicate="родилась", source_text="Баба Галя родилась в 1947 году."))
    db_session.commit()

    assert facts_for_entity(db_session, user.id, obj) == []


def test_route_returns_facts_with_reasons_and_names(client, db_session, user):
    grandmother = person(db_session, user, "Баба Галя")
    father = person(db_session, user, "Отец автора")
    src = source(db_session, user, "Баба Галя родилась в 1947 году.")
    db_session.add(Fact(user_id=user.id, source_id=src.id, subject_entity_id=grandmother.id,
                        object_entity_id=father.id, predicate="родилась", object_text="в 1947 году",
                        source_text="Баба Галя родилась в 1947 году."))
    db_session.commit()

    payload = client.get(f"/knowledge/objects/{grandmother.id}/facts").json()

    assert len(payload) == 1
    fact = payload[0]
    assert fact["predicate"] == "родилась"
    assert fact["subject_name"] == "Баба Галя"
    assert fact["object_name"] == "Отец автора"
    assert "subject" in fact["match_reasons"]


def test_route_hides_objects_of_other_users(client, db_session, user):
    other = User(username="other", password_hash="x", name="Other")
    db_session.add(other)
    db_session.commit()
    foreign = person(db_session, other, "Чужой")
    db_session.commit()

    response = client.get(f"/knowledge/objects/{foreign.id}/facts")

    assert response.status_code == 404


def test_knowledge_events_are_important_facts_shown_once(client, db_session, user):
    """Событие — факт с важностью: оно в блоке событий, а не второй раз в фактах."""
    me = person(db_session, user, "Я")
    src = source(db_session, user, "Переехали в Киев в 1960 году. Любил чай.")
    db_session.add_all([
        Fact(user_id=user.id, source_id=src.id, subject_entity_id=me.id, predicate="переехали",
             object_text="в Киев", source_text="Переехали в Киев", time_iso="1960", importance=0.8),
        Fact(user_id=user.id, source_id=src.id, subject_entity_id=me.id, predicate="любил",
             object_text="чай", source_text="Любил чай", importance=0.1),
        Fact(user_id=user.id, source_id=src.id, subject_entity_id=me.id, predicate="родился",
             source_text="Переехали", importance=None),
    ])
    db_session.commit()

    payload = client.get(f"/knowledge?subject_entity_id={me.id}").json()

    assert [event["predicate"] for event in payload["events"]] == ["переехали"]
    assert payload["events"][0]["importance"] == 0.8
    assert {fact["predicate"] for fact in payload["facts"]} == {"любил", "родился"}
    assert payload["stats"]["events"] == 1
    assert "habits" not in payload
