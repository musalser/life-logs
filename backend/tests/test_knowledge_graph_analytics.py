"""Тесты графа связей, селектора объектов и аналитики (этап 5, разделы 9 и 10).

Проверяется то, что нельзя увидеть глазами в интерфейсе: пути ограничены тремя
рёбрами и не выдумывают типы, а аналитика выбрасывает пункты без существующих
ссылок — иначе «инсайт» окажется неподтверждаемым.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from app.models import (
    Entity,
    EntityRelation,
    Event,
    Fact,
    Goal,
    Habit,
    KnowledgeInsight,
    KnowledgeSource,
    KnowledgeState,
    User,
)
from app.services.analytics_service import AnalyticsService
from app.services.knowledge_graph_service import KnowledgeGraphService


def run(coro):
    return asyncio.run(coro)


@pytest.fixture()
def user(db_session):
    row = User(username="tester", password_hash="x", name="Тестер")
    db_session.add(row)
    db_session.commit()
    return row


def person(db, user, name):
    row = Entity(
        user_id=user.id,
        entity_type="person",
        canonical_name=name,
        normalized_name=name.lower(),
    )
    db.add(row)
    db.flush()
    return row


def edge(db, user, source, target, relation_type):
    row = EntityRelation(
        user_id=user.id,
        from_entity_id=source.id,
        to_entity_id=target.id,
        relation_type=relation_type,
    )
    db.add(row)
    db.flush()
    return row


# ---------------------------------------------------------------------------
# Граф и пути
# ---------------------------------------------------------------------------


def test_direct_edges_are_not_paths(db_session, user):
    me = person(db_session, user, "Я")
    mama = person(db_session, user, "Мама")
    edge(db_session, user, me, mama, "мама")
    graph = KnowledgeGraphService(db_session, user.id)

    assert [item.relation_type for item in graph.edges(me.id)] == ["мама"]
    assert graph.paths(me.id) == {}  # прямое ребро путём не считается


def test_path_through_another_person(db_session, user):
    me = person(db_session, user, "Я")
    baba = person(db_session, user, "Баба Галя")
    baba_mama = person(db_session, user, "Прабабушка")
    edge(db_session, user, me, baba, "бабушка")
    edge(db_session, user, baba, baba_mama, "мама")
    graph = KnowledgeGraphService(db_session, user.id)

    paths = graph.paths(me.id)

    chain = paths[baba_mama.id]
    assert [step["relation_type"] for step in chain] == ["бабушка", "мама"]
    assert [step["name"] for step in chain] == ["Баба Галя", "Прабабушка"]
    # типы не пересчитываются: «прабабушка» из цепочки не выводится
    assert graph.path_label(chain) == "Прабабушка → Баба Галя"


def test_paths_stop_at_three_edges(db_session, user):
    me = person(db_session, user, "Я")
    current = me
    for index in range(5):
        nxt = person(db_session, user, f"Лицо{index}")
        edge(db_session, user, current, nxt, f"связь{index}")
        current = nxt
    graph = KnowledgeGraphService(db_session, user.id)

    paths = graph.paths(me.id)

    # пути длиной 1 не считаются (это прямые рёбра), поэтому цепочки — 2 и 3
    assert sorted(len(chain) for chain in paths.values()) == [2, 3]
    assert max(len(chain) for chain in paths.values()) == 3  # дальше трёх рёбер не идём


def test_cycles_do_not_hang_the_search(db_session, user):
    me = person(db_session, user, "Я")
    other = person(db_session, user, "Друг")
    edge(db_session, user, me, other, "друг")
    edge(db_session, user, other, me, "друг")  # цикл
    graph = KnowledgeGraphService(db_session, user.id)

    paths = graph.paths(me.id)

    assert me.id not in paths


def test_reachable_map_mixes_direct_and_paths(db_session, user):
    me = person(db_session, user, "Я")
    mama = person(db_session, user, "Мама")
    baba = person(db_session, user, "Баба")
    edge(db_session, user, me, mama, "мама")
    edge(db_session, user, mama, baba, "мама")
    graph = KnowledgeGraphService(db_session, user.id)

    reachable = graph.reachable_map(me.id)

    assert list(reachable[mama.id])[0]["relation_type"] == "мама"
    assert [step["relation_type"] for step in reachable[baba.id]] == ["мама", "мама"]


# ---------------------------------------------------------------------------
# Аналитика
# ---------------------------------------------------------------------------


class FakeAdapter:
    def __init__(self, answer):
        self.answer = answer
        self.prompts: list[str] = []

    async def generate_structured(self, prompt, schema):
        self.prompts.append(prompt)
        return json.dumps(self.answer, ensure_ascii=False)


def seed_knowledge(db, user):
    source = KnowledgeSource(user_id=user.id, source_type="diary", ref_id=1, text="Бегал по утрам.")
    db.add(source)
    db.flush()
    me = person(db, user, "Я")
    user.self_entity_id = me.id
    fact = Fact(user_id=user.id, source_id=source.id, subject_entity_id=me.id, predicate="бегает", source_text="Бегал по утрам")
    goal = Goal(user_id=user.id, subject_entity_id=me.id, title="Выучить английский")
    habit = Habit(user_id=user.id, subject_entity_id=me.id, title="бег")
    event = Event(user_id=user.id, source_id=source.id, subject_entity_id=me.id, title="Поездка на море")
    db.add_all([fact, goal, habit, event])
    db.commit()
    return me, fact, goal, habit, event


def test_analytics_keeps_only_items_with_existing_evidence(db_session, user):
    me, fact, goal, habit, event = seed_knowledge(db_session, user)
    adapter = FakeAdapter(
        {
            "insights": [
                {"text": "Бег идёт фоном", "confidence": 0.8, "evidence": [{"kind": "habit", "id": habit.id}]},
                {"text": "Выдумка без ссылок", "confidence": 0.9, "evidence": []},
                {"text": "Ссылка в пустоту", "confidence": 0.9, "evidence": [{"kind": "fact", "id": 9999}]},
            ],
            "questions": [{"text": "Что дальше?", "confidence": 0.5, "evidence": [{"kind": "goal", "id": goal.id}]}],
            "ideas": [],
            "conclusions": [{"text": "Есть событие", "confidence": 0.7, "evidence": [{"kind": "event", "id": event.id}]}],
        }
    )
    service = AnalyticsService(db_session, adapter=adapter)

    result = run(service.generate(user.id, me.id))

    payload = result["payload"]
    assert [item["text"] for item in payload["insights"]] == ["Бег идёт фоном"]
    assert payload["questions"][0]["evidence"][0]["label"].startswith("цель")
    assert len(payload["conclusions"]) == 1
    assert result["stale"] is False
    # промпт требует не выдумывать и ссылаться на записи
    assert "Не выдумывай факты" in adapter.prompts[0]
    assert "будет отброшен" in adapter.prompts[0]


def test_analytics_cache_becomes_stale_after_new_knowledge(db_session, user):
    me, *_ = seed_knowledge(db_session, user)
    adapter = FakeAdapter({"insights": [], "questions": [], "ideas": [], "conclusions": []})
    service = AnalyticsService(db_session, adapter=adapter)
    run(service.generate(user.id, me.id))
    assert service.get_cached(user.id, me.id)["stale"] is False

    # любое успешное извлечение поднимает версию знаний
    state = db_session.query(KnowledgeState).filter_by(user_id=user.id).first()
    if state is None:
        state = KnowledgeState(user_id=user.id, version=1)
        db_session.add(state)
    else:
        state.version += 1
    db_session.commit()

    cached = service.get_cached(user.id, me.id)

    assert cached["stale"] is True
    assert cached["version"] != service.state_version(user.id)


def test_analytics_without_data_returns_empty_lists(db_session, user):
    empty_user = User(username="empty", password_hash="x", name="Пусто")
    db_session.add(empty_user)
    db_session.commit()
    me = person(db_session, empty_user, "Я")
    db_session.commit()
    adapter = FakeAdapter({"insights": [{"text": "x", "confidence": 1, "evidence": []}]})

    result = run(AnalyticsService(db_session, adapter=adapter).generate(empty_user.id, me.id))

    assert result["payload"] == {"insights": [], "questions": [], "ideas": [], "conclusions": []}
    assert adapter.prompts == []  # модель не звали: анализировать нечего


def test_analytics_replaces_the_previous_cache_row(db_session, user):
    me, *_ = seed_knowledge(db_session, user)
    first = FakeAdapter({"insights": [{"text": "первый", "confidence": 1, "evidence": [{"kind": "entity", "id": me.id}]}]})
    service = AnalyticsService(db_session, adapter=first)
    run(service.generate(user.id, me.id))

    second = FakeAdapter({"insights": [{"text": "второй", "confidence": 1, "evidence": [{"kind": "entity", "id": me.id}]}]})
    result = run(AnalyticsService(db_session, adapter=second).generate(user.id, me.id))

    assert db_session.query(KnowledgeInsight).filter_by(user_id=user.id, subject_entity_id=me.id).count() == 1
    assert result["payload"]["insights"][0]["text"] == "второй"


def test_analytics_survives_a_broken_model_answer(db_session, user):
    me, *_ = seed_knowledge(db_session, user)
    service = AnalyticsService(db_session, adapter=FakeAdapter({}))
    service.adapter.generate_structured = lambda prompt, schema: _broken()

    result = run(service.generate(user.id, me.id))

    assert result["payload"] == {"insights": [], "questions": [], "ideas": [], "conclusions": []}


async def _broken():
    return "это не JSON"
