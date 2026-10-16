"""Интеграционные тесты конвейера на сквозных сценариях (раздел 13 ТЗ).

Здесь проверяется то, что видно только на стыке слоёв: сервис источников,
конвейер, граф, аналитика и API. Модель и эмбеддинги подменены, база — sqlite
из общей фикстуры, поэтому тесты быстрые и детерминированные.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from app.models import (
    FactSource,
    DiaryPage,
    Entity,
    EntityMention,
    EntityRelation,
    Fact,
    Goal,
    HTRAuthor,
    HTRLine,
    HTRPage,
    KnowledgeSource,
    KnowledgeState,
    User,
)
from app.services.analytics_service import AnalyticsService
from app.services.extraction_pipeline import ExtractionPipeline
from app.services.knowledge_graph_service import KnowledgeGraphService
from app.services.knowledge_source_service import KnowledgeSourceService, source_status_payload


def run(coro):
    return asyncio.run(coro)


class FakeAdapter:
    """Модель-заглушка: ответы по видам вызова, вызовы запоминаются."""

    def __init__(self, answers=None, fail_kinds=None):
        self.answers = answers or {}
        self.fail_kinds = fail_kinds or set()
        self.calls: list[str] = []
        self.prompts: list[str] = []

    async def generate_structured(self, prompt: str, schema: dict) -> str:
        key = "matches" if "matches" in schema.get("properties", {}) else next(iter(schema["properties"]))
        self.calls.append(key)
        self.prompts.append(prompt)
        if key in self.fail_kinds:
            raise RuntimeError(f"{key} недоступен")
        return json.dumps(self.answers.get(key, {}), ensure_ascii=False)


class FakeEmbeddings:
    """Тематический косинус: «бег» и «бегаю» — одна тема, остальное — разное."""

    THEMES = {"бег": "бег", "бегаю": "бег", "пробежка": "бег", "английский": "англ", "язык": "англ"}

    async def embed_texts(self, texts, prompt_name=None):
        vectors = []
        for text in texts:
            words = {word.strip(".,!?;:").lower() for word in str(text).split()}
            themes = {self.THEMES[word] for word in words if word in self.THEMES}
            vectors.append([1.0 if theme in themes else 0.0 for theme in ("бег", "англ")])
        return vectors


@pytest.fixture()
def user(db_session):
    row = User(username="integration", password_hash="x", name="Интеграция")
    db_session.add(row)
    db_session.commit()
    return row


@pytest.fixture()
def author(db_session, user):
    row = HTRAuthor(user_id=user.id, name="Баба Галя")
    db_session.add(row)
    db_session.commit()
    return row


def add_htr_page(db, author, order_index, lines, status="CONFIRMED"):
    page = HTRPage(
        user_id=author.user_id,
        author_id=author.id,
        file_path=f"htr_storage/pages/{order_index}.jpg",
        file_name=f"page-{order_index}.jpg",
        order_index=order_index,
        status=status,
    )
    db.add(page)
    db.flush()
    for index, text in enumerate(lines):
        db.add(
            HTRLine(
                page_id=page.id,
                order_index=index,
                x1=0,
                y1=index,
                x2=10,
                y2=index + 1,
                predicted_text=text,
            )
        )
    db.commit()
    return page


async def run_source(db, source, adapter):
    """Прогон через конвейер напрямую: без статусов и записи прогона."""
    service = KnowledgeSourceService(db)
    pipeline = ExtractionPipeline(db, source, adapter=adapter, embeddings=FakeEmbeddings(), service=service)
    summary = await pipeline.run()
    db.commit()
    return summary


def run_through_service(db, source, adapter, monkeypatch=None):
    """Полный путь как в проде: захват, очистка, прогон, статус, версия."""
    service = KnowledgeSourceService(db)
    service._pipeline = lambda s, src: _pipeline_with(adapter, s, src)
    return run(service.run_now(source))


def self_fact(predicate, quote, **extra):
    """Факт об авторе: ответ вызова фактов с пустыми необязательными полями."""
    return {
        "subject_kind": "self", "subject_name": "", "predicate": predicate,
        "object_name": "", "object_text": "", "source_text": quote,
        "time_text": "", "confidence": 0.9, **extra,
    }


def _pipeline_with(adapter, db, source):
    pipeline = ExtractionPipeline(db, source, adapter=adapter, embeddings=FakeEmbeddings(), service=KnowledgeSourceService(db))
    return pipeline.run()


def test_full_page_pipeline_writes_every_kind(db_session, user, author):
    """Сквозной прогон страницы: текст → знание → статус → карта строк."""
    page = add_htr_page(db_session, author, 0, ["Моя мама приехала в гости.", "Сегодня бегал по утрам."])
    service = KnowledgeSourceService(db_session)
    source, created, changed = service.refresh_from_htr_page(page)
    assert (created, changed) == (True, True)

    adapter = FakeAdapter(
        {
            "entities": {"entities": [{"name": "мама", "entity_type": "person", "description": "", "aliases": []}]},
            "mentions": {"mentions": [{"entity_name": "мама", "mention_text": "мама", "sentence_text": "", "confidence": 0.9}]},
            "facts": {
                "facts": [
                    {
                        "subject_kind": "self", "subject_name": "",
                        "predicate": "приехала",
                        "object_name": "мама",
                        "object_text": "",
                        "source_text": "Моя мама приехала",
                        "time_text": "",
                        # событие — это факт с важностью
                        "importance": 0.5,
                        "confidence": 0.9,
                    },
                    {
                        "subject_kind": "self", "subject_name": "",
                        "predicate": "бегал",
                        "object_name": "",
                        "object_text": "",
                        "source_text": "бегал по утрам",
                        "time_text": "",
                        # регулярность — сырьё для привычек в аналитике
                        "recurrence": "по утрам",
                        "confidence": 0.9,
                    },
                ]
            },
            "relations": {"relations": [{"person_name": "мама", "relation_kind": "mother", "source_text": "Моя мама приехала", "confidence": 0.9}]},
        }
    )
    summary = run(run_source(db_session, source, adapter))

    assert summary["mentions_kept"] == 1
    assert summary["facts_created"] == 2
    assert summary["relations_created"] == 1
    assert summary["chunks_created"] == 1
    assert adapter.calls == ["entities", "mentions", "facts", "relations"]

    # офсеты указывают в текст источника
    fact = db_session.query(Fact).filter_by(predicate="приехала").one()
    assert fact.importance == 0.5
    assert db_session.query(Fact).filter_by(predicate="бегал").one().recurrence == "по утрам"
    assert source.text[fact.start_offset : fact.end_offset] == fact.source_text
    mention = db_session.query(EntityMention).one()
    assert source.text[mention.start_offset : mention.end_offset] == mention.mention_text
    # связь идёт от автора текста
    edge = db_session.query(EntityRelation).one()
    assert edge.from_entity_id == source.author_entity_id
    # статус источника и отчёт прогона появляются, когда прогоном владеет сервис
    result = run_through_service(db_session, source, adapter)
    assert result["status"] == "done"
    db_session.refresh(source)
    assert source.extraction_status == "done"
    payload = source_status_payload(db_session, source)
    assert payload["last_run"]["status"] == "succeeded"
    assert payload["last_run"]["stats"]["facts_created"] == 2
    assert payload["last_run"]["model"]


def test_sentence_split_across_pages_is_taken_from_the_neighbour(db_session, user, author):
    """Предложение, разорванное границей страниц, читается с контекстом соседа."""
    first = add_htr_page(db_session, author, 0, ["Мы пошли в лес за грибами, а"])
    second = add_htr_page(db_session, author, 1, ["потом вернулись домой уставшие."])
    service = KnowledgeSourceService(db_session)
    source, _, _ = service.refresh_from_htr_page(second)

    adapter = FakeAdapter(
        {
            "facts": {
                "facts": [
                    {
                        "subject_kind": "self", "subject_name": "",
                        "predicate": "вернулись",
                        "object_name": "",
                        "object_text": "домой",
                        # цитата целиком со второй страницы: контекст первой нужен
                        # только чтобы понять продолжение предложения
                        "source_text": "потом вернулись домой",
                        "time_text": "",
                        "confidence": 0.8,
                    }
                ]
            }
        }
    )
    summary = run(run_source(db_session, source, adapter))

    assert summary["facts_created"] == 1
    # контекст ушёл в промпт с первой страницы и помечен запретом
    facts_prompt = [p for p in adapter.prompts if "Ты извлекаешь факты" in p][0]
    assert "Мы пошли в лес за грибами" in facts_prompt
    assert "НЕ извлекай" in facts_prompt or "НИКОГДА не извлекай" in facts_prompt
    # а знание осталось на второй странице
    fact = db_session.query(Fact).one()
    assert fact.source_id == source.id
    assert source.text[fact.start_offset : fact.end_offset] == fact.source_text


def test_reextraction_after_the_text_changes_drops_the_old_knowledge(db_session, user):
    """Правка текста: старые знания исчезают, новые появляются, дублей нет."""
    page = DiaryPage(user_id=user.id, content="Сегодня бегал по утрам.")
    db_session.add(page)
    db_session.commit()
    service = KnowledgeSourceService(db_session)
    source, _, _ = service.refresh_from_diary_page(page)

    first = FakeAdapter({"facts": {"facts": [self_fact("бегал", "бегал по утрам")]}})
    run(run_source(db_session, source, first))
    assert [fact.predicate for fact in db_session.query(Fact).all()] == ["бегал"]

    # текст изменился: на странице теперь другое знание
    page.content = "Сегодня бегал по утрам и выучил английский язык."
    db_session.commit()
    source, created, changed = service.refresh_from_diary_page(page)
    assert changed is True
    assert source.extraction_status == "pending"

    second = FakeAdapter({"facts": {"facts": [self_fact("выучил", "выучил английский язык")]}})
    result = run_through_service(db_session, source, second)
    assert result["status"] == "done"
    summary = result["summary"]
    assert summary["facts_created"] == 1
    predicates = {fact.predicate for fact in db_session.query(Fact).all()}
    assert predicates == {"выучил"}  # старого факта больше нет


def test_sync_and_async_modes_produce_the_same_result(db_session, user, monkeypatch):
    """sync и async дают одинаковый результат: разница только в том, кто работает.

    Тело Celery-задачи здесь не вызывается: воркер открывает *свою* сессию
    (`SessionLocal()`), то есть пошёл бы в боевую базу, а не в sqlite фикстуры.
    Проверяется то, что от неё зависит: постановка задачи в очередь и то, что
    тот же конвейер, выполненный «как воркер», даёт тот же результат.
    """
    pages = []
    for index in range(2):
        page = DiaryPage(user_id=user.id, content=f"Запись {index}: бегал по утрам.")
        db_session.add(page)
        db_session.commit()
        pages.append(page)

    service = KnowledgeSourceService(db_session)
    sources = [service.refresh_from_diary_page(page)[0] for page in pages]
    answers = {"facts": {"facts": [self_fact("бегал", "бегал по утрам", recurrence="по утрам")]}}

    # sync: конвейер выполняется прямо здесь
    sync_result = run_through_service(db_session, sources[0], FakeAdapter(answers))
    assert sync_result["status"] == "done"

    # async: задача ставится в очередь
    queued = {}

    class FakeTask:
        id = "task-1"

    def fake_delay(source_id):
        queued["source_id"] = source_id
        return FakeTask()

    monkeypatch.setattr("app.tasks.extract_knowledge_task.delay", fake_delay)
    result = run(service.request_extraction(sources[1], mode="async"))

    assert result["queued"] is True
    assert result["task_id"] == "task-1"
    assert queued["source_id"] == sources[1].id
    db_session.refresh(sources[1])
    assert sources[1].extraction_status == "pending"  # работа ещё не начата

    # то же, что сделает воркер: сервис, захват, конвейер
    worker_result = run_through_service(db_session, sources[1], FakeAdapter(answers))

    assert worker_result["status"] == "done"
    assert worker_result["summary"]["extracted_facts"] == sync_result["summary"]["extracted_facts"]
    db_session.refresh(sources[1])
    assert sources[1].extraction_status == "done"
    # обе записи дали одно и то же знание: один факт, подтверждённый двумя
    # источниками (по цитате на каждый), и ни одного лишнего прогона
    assert db_session.query(Fact).count() == 1
    assert db_session.query(FactSource).count() == 2


def test_perspective_reaches_the_graph_and_analytics(db_session, user, author):
    """Знание автора видно в графе и попадает в агрегат аналитики."""
    page = add_htr_page(db_session, author, 0, ["Моя мама приехала в гости."])
    service = KnowledgeSourceService(db_session)
    source, _, _ = service.refresh_from_htr_page(page)

    adapter = FakeAdapter(
        {
            "entities": {"entities": [{"name": "Елена", "entity_type": "person", "description": "", "aliases": []}]},
            "relations": {"relations": [{"person_name": "Елена", "relation_kind": "mother", "source_text": "Моя мама приехала", "confidence": 0.9}]},
            "facts": {"facts": [self_fact("встретил маму", "Моя мама приехала", importance=0.6, time_text="вчера")]},
        }
    )
    run(run_source(db_session, source, adapter))

    author_entity = db_session.query(Entity).filter(Entity.id == source.author_entity_id).one()
    graph = KnowledgeGraphService(db_session, user.id)
    edges = graph.edges(author_entity.id)
    # подпись для интерфейса — из словаря отношений, ключ — канонический
    assert [edge.relation_type for edge in edges] == ["мать"]
    assert [edge.relation_kind for edge in edges] == ["mother"]

    # агрегат аналитики строится от перспективы автора и видит его событие —
    # факт с пометкой важности
    analytics = AnalyticsService(db_session, adapter=FakeAdapter({}))
    digest, index = analytics._digest(user.id, author_entity)
    assert "встретил маму" in digest
    assert "событие, важность 0.6" in digest
    assert index["fact"]  # ссылка на факт существует, значит пункт можно сохранить


def test_analytics_cache_is_invalidated_by_a_pipeline_run(db_session, user, author):
    """Успешный прогон поднимает версию — кэш аналитики становится stale."""
    page = add_htr_page(db_session, author, 0, ["Сегодня бегал по утрам."])
    service = KnowledgeSourceService(db_session)
    source, _, _ = service.refresh_from_htr_page(page)

    before = db_session.query(KnowledgeState).filter_by(user_id=user.id).first()
    before_version = before.version if before else 0

    result = run_through_service(db_session, source, FakeAdapter({}))
    assert result["status"] == "done"

    after = db_session.query(KnowledgeState).filter_by(user_id=user.id).one()
    assert after.version == before_version + 1
    analytics = AnalyticsService(db_session, adapter=FakeAdapter({}))
    assert analytics.state_version(user.id) == after.version


def test_reopen_hides_the_knowledge_and_keeps_it_out_of_the_graph(db_session, user, author):
    """Откат подтверждения: знание страницы уходит, граф его больше не видит."""
    page = add_htr_page(db_session, author, 0, ["Моя мама приехала в гости."])
    service = KnowledgeSourceService(db_session)
    source, _, _ = service.refresh_from_htr_page(page)
    adapter = FakeAdapter(
        {
            "entities": {"entities": [{"name": "Елена", "entity_type": "person", "description": "", "aliases": []}]},
            "relations": {"relations": [{"person_name": "Елена", "relation_kind": "mother", "source_text": "Моя мама приехала", "confidence": 0.9}]},
        }
    )
    run(run_source(db_session, source, adapter))
    assert db_session.query(EntityRelation).count() == 1

    removed = service.mark_pending(source)

    assert removed["entity_relations"] == 1
    assert db_session.query(EntityRelation).count() == 0
    db_session.refresh(source)
    assert source.extraction_status == "pending"
    # объект автора остаётся: он привязан к рукописи, а не к странице
    assert db_session.query(Entity).filter(Entity.id == source.author_entity_id).count() == 1


def test_reextraction_keeps_the_author_of_the_manuscript(db_session, user, author):
    """Автор рукописи переживает переизвлечение своих же страниц.

    Найденный на живых данных баг: объект автора обычно не упоминается на его
    собственных страницах («моя мама» имеет имя, автор — нет), поэтому очистка
    осиротевших объектов удаляла его, и следующий прогон падал с «у источника
    нет автора-объекта». Объект автора — опора перспективы, его удалять нельзя.
    """
    page = add_htr_page(db_session, author, 0, ["Моя мама приехала в гости."])
    service = KnowledgeSourceService(db_session)
    source, _, _ = service.refresh_from_htr_page(page)
    author_entity_id = source.author_entity_id
    assert author_entity_id is not None

    adapter = FakeAdapter(
        {
            "entities": {"entities": [{"name": "Елена", "entity_type": "person", "description": "", "aliases": []}]},
            "relations": {"relations": [{"person_name": "Елена", "relation_kind": "mother", "source_text": "Моя мама приехала", "confidence": 0.9}]},
        }
    )
    first = run_through_service(db_session, source, adapter)
    assert first["status"] == "done"

    # второй прогон той же страницы: объект автора на месте
    db_session.refresh(source)
    assert source.author_entity_id == author_entity_id
    assert db_session.query(Entity).filter(Entity.id == author_entity_id).count() == 1

    second = run_through_service(db_session, source, adapter)
    assert second["status"] == "done", second.get("error")
    assert db_session.query(Entity).filter(Entity.id == author_entity_id).count() == 1
    # и связь по-прежнему идёт от автора
    assert db_session.query(EntityRelation).one().from_entity_id == author_entity_id
