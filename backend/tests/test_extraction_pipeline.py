"""Тесты конвейера извлечения (этап 4): вызовы A–E, дедупликация, запись.

Модель и эмбеддинги подменяются фейками: проверяется то, чем владеет конвейер —
проверка цитат и офсетов, перспектива автора, запрет слияния по датам,
идемпотентность повторного прогона и независимость вызовов.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from app.models import (
    Chunk,
    DiaryPage,
    Entity,
    EntityMention,
    EntityRelation,
    Fact,
    FactConflict,
    FactSource,
    Goal,
    GoalProgress,
    HTRAuthor,
    HTRLine,
    HTRPage,
    User,
)
from app.services.embedding_service import EmbeddingError
from app.services.extraction_pipeline import ExtractionPipeline, _quote_and_offset
from app.services.knowledge_source_service import KnowledgeSourceService
from app.services.source_text import parse_line_map


def run(coro):
    return asyncio.run(coro)


class FakeAdapter:
    """Отдаёт заранее заготовленные ответы по схеме вызова."""

    def __init__(self, answers: dict | None = None, fail_kinds: set[str] | None = None):
        self.answers = answers or {}
        self.fail_kinds = fail_kinds or set()
        self.prompts: list[str] = []
        self.calls: list[str] = []

    async def generate_structured(self, prompt: str, schema: dict) -> str:
        key = self._kind(schema)
        self.calls.append(key)
        self.prompts.append(prompt)
        if key in self.fail_kinds:
            raise RuntimeError(f"{key} недоступен")
        return json.dumps(self.answers.get(key, {}), ensure_ascii=False)

    @staticmethod
    def _kind(schema: dict) -> str:
        if "matches" in schema.get("properties", {}):
            return "matches"
        for key in ("entities", "mentions", "facts", "relations", "goals", "events", "habits"):
            if key in schema.get("properties", {}):
                return key
        return "?"


class FakeEmbeddings:
    """Косинус считает по пересечению слов: предсказуемо и без модели."""

    calls: list[list[str]] = []

    def __init__(self, unavailable: bool = False):
        self.unavailable = unavailable

    async def embed_texts(self, texts, prompt_name=None):
        if self.unavailable:
            raise EmbeddingError("модель недоступна")
        FakeEmbeddings.calls.append(list(texts))
        return [self._vector(text) for text in texts]

    def embed_texts_sync(self, texts, prompt_name=None):
        return [self._vector(text) for text in texts]

    #: «темы» набора: словоформы сведены к одной оси, стемминга нет намеренно
    THEMES = {
        "бег": "бег", "бегаю": "бег", "бег2": "бег",
        "английский": "англ", "англ": "англ",
        "мама": "мама",
        "цель": "цель",
        "сон": "сон",
    }

    @classmethod
    def _vector(cls, text: str) -> list[float]:
        words = {word.strip(".,!?;:").lower() for word in str(text).split()}
        themes = {cls.THEMES[word] for word in words if word in cls.THEMES}
        return [1.0 if theme in themes else 0.0 for theme in ("бег", "англ", "мама", "сон")]


@pytest.fixture()
def user(db_session):
    row = User(username="tester", password_hash="x", name="Тестер")
    db_session.add(row)
    db_session.commit()
    return row


@pytest.fixture()
def author(db_session, user):
    row = HTRAuthor(user_id=user.id, name="Баба Галя")
    db_session.add(row)
    db_session.commit()
    return row


def make_diary_source(db, user, content):
    page = DiaryPage(user_id=user.id, title="Запись", content=content)
    db.add(page)
    db.commit()
    service = KnowledgeSourceService(db)
    source, _, _ = service.refresh_from_diary_page(page)
    return service, source


def make_htr_source(db, author, lines):
    page = HTRPage(
        user_id=author.user_id,
        author_id=author.id,
        file_path="htr_storage/pages/p.jpg",
        file_name="p.jpg",
        order_index=0,
        status="CONFIRMED",
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
    service = KnowledgeSourceService(db)
    source, _, _ = service.refresh_from_htr_page(page)
    return service, source


def fact(quote, predicate="хочет", *, subject_kind="self", subject_name="", object_text="", **extra):
    """Ответ вызова фактов: всё, что не задано, — пустое, как вернула бы модель."""
    item = {
        "subject_kind": subject_kind,
        "subject_name": subject_name,
        "subject_relation": "",
        "subject_of": "",
        "subject_id": 0,
        "predicate": predicate,
        "predicate_kind": "other",
        "object_name": "",
        "object_text": object_text,
        "source_text": quote,
        "time_text": "",
        "time_iso": "",
        "importance": 0,
        "recurrence": "",
        "confidence": 0.9,
        "goal_title": "",
        "goal_status": "none",
    }
    item.update(extra)
    return item


def goal_fact(title, status, quote, **extra):
    """Цель — это факт с названием цели и статусом продвижения."""
    return fact(quote, object_text=title.lower(), goal_title=title, goal_status=status, **extra)


def facts(*items):
    return {"facts": {"facts": list(items)}}


async def run_pipeline(db, source, adapter, embeddings=None):
    service = KnowledgeSourceService(db)
    pipeline = ExtractionPipeline(db, source, adapter=adapter, embeddings=embeddings or FakeEmbeddings(), service=service)
    summary = await pipeline.run()
    db.commit()
    return summary


# ---------------------------------------------------------------------------
# Цитаты и офсеты
# ---------------------------------------------------------------------------


def test_quote_lookup_prefers_the_hint_and_falls_back_to_search():
    text = "мама пришла домой"
    assert _quote_and_offset(text, "пришла", 5) == ("пришла", 5, 11)
    # модель назвала неверный офсет — берём первое вхождение
    assert _quote_and_offset(text, "домой", 0) == ("домой", 12, 17)
    # пересказ вместо цитаты не принимается
    assert _quote_and_offset(text, "мама вернулась") is None
    assert _quote_and_offset(text, "") is None


def test_quote_wrapped_across_a_line_break_is_found():
    """Рукопись — стопка строк: модель читает перенос строки как пробел."""
    text = "Мы раньше жили с ними в\nодном доме, занимали маленькую половину"

    found = _quote_and_offset(text, "Мы раньше жили с ними в одном доме")

    assert found is not None
    quote, start, end = found
    # офсет указывает на реальный текст страницы, вместе с переносом строки
    assert text[start:end] == quote
    assert "\n" in quote
    # цитата с двойным пробелом тоже находится
    assert _quote_and_offset(text, "жили с ними  в одном") is not None


def test_quote_that_is_really_absent_is_still_dropped():
    text = "Мы раньше жили с ними в одном доме"
    assert _quote_and_offset(text, "Мы жили в другом городе") is None


def test_fact_with_a_paraphrased_quote_is_dropped(db_session, user):
    text = "Сегодня я купил новую машину."
    service, source = make_diary_source(db_session, user, text)
    adapter = FakeAdapter(
        {
            "facts": {
                "facts": [
                    {
                        "subject_kind": "self", "subject_name": "",
                        "predicate": "купил",
                        "object_name": "",
                        "object_text": "новую машину",
                        "source_text": "приобрёл автомобиль",  # пересказ, не цитата
                        "time_text": "сегодня",
                        "confidence": 0.9,
                    },
                    {
                        "subject_kind": "self", "subject_name": "",
                        "predicate": "купил",
                        "object_name": "",
                        "object_text": "новую машину",
                        "source_text": "купил новую машину",
                        "time_text": "сегодня",
                        "confidence": 0.9,
                    },
                ]
            }
        }
    )

    summary = run(run_pipeline(db_session, source, adapter))

    assert summary["facts_dropped"] == 1
    assert summary["facts_created"] == 1
    fact = db_session.query(Fact).one()
    assert fact.source_text == "купил новую машину"
    assert text[fact.start_offset : fact.end_offset] == "купил новую машину"
    # субъект по умолчанию — автор текста
    self_entity = service.ensure_self_entity(user.id)
    assert fact.subject_entity_id == self_entity.id


def test_mention_offsets_point_at_the_text(db_session, user):
    text = "Маша пришла, а потом ушла Маша."
    _, source = make_diary_source(db_session, user, text)
    adapter = FakeAdapter(
        {
            "entities": {"entities": [{"name": "Маша", "entity_type": "person", "description": "", "aliases": ["Мария"]}]},
            "mentions": {
                "mentions": [
                    {"entity_name": "Маша", "mention_text": "Маша", "sentence_text": "", "confidence": 0.9}
                ]
            },
        }
    )

    run(run_pipeline(db_session, source, adapter))

    mention = db_session.query(EntityMention).one()
    assert text[mention.start_offset : mention.end_offset] == "Маша"
    assert mention.start_offset == 0
    assert mention.sentence_text  # предложение подставлено из текста
    entity = db_session.query(Entity).filter_by(canonical_name="Маша").one()
    assert json.loads(entity.aliases_json) == ["Мария"]


# ---------------------------------------------------------------------------
# Перспектива автора
# ---------------------------------------------------------------------------


def test_relations_are_read_from_the_text_author(db_session, author):
    """Связь идёт от автора рукописи; человек берётся из вызова сущностей.

    Объект создаёт только вызов A: имена из вызова связей («мама») заводят
    «персону» из любого слова, а таких записей в базе уже накопилось немало.
    """
    service, source = make_htr_source(db_session, author, ["Моя мама приехала в гости."])
    adapter = FakeAdapter(
        {
            "entities": {
                "entities": [{"name": "Елена", "entity_type": "person", "description": "", "aliases": []}]
            },
            "relations": {
                "relations": [
                    {
                        "person_name": "Елена",
                        "relation_kind": "mother",
                        "source_text": "Моя мама приехала",
                        "confidence": 0.95,
                    }
                ]
            },
        }
    )

    run(run_pipeline(db_session, source, adapter))

    edge = db_session.query(EntityRelation).one()
    bad_author = service.ensure_author_entity(author)
    self_entity = service.ensure_self_entity(author.user_id)
    assert edge.from_entity_id == bad_author.id  # автор текста, не аккаунт
    assert edge.to_entity_id != self_entity.id
    # ключ из закрытого словаря, а подпись для интерфейса — русская
    assert edge.relation_kind == "mother"
    assert edge.relation_type == "мать"
    assert edge.source_id == source.id
    assert "мама приехала" in (edge.evidence_text or "")
    # prompt несёт перспективу автора
    relation_prompt = [p for p in adapter.prompts if "круг общения" in p][0]
    assert "Баба Галя" in relation_prompt
    assert "ОТ ЛИЦА АВТОРА" in relation_prompt


def test_self_relation_is_not_stored(db_session, author):
    service, source = make_htr_source(db_session, author, ["Я — Баба Галя."])
    adapter = FakeAdapter(
        {"relations": {"relations": [{"person_name": "Баба Галя", "relation_kind": "other", "relation_label": "я", "source_text": "Я — Баба Галя", "confidence": 1.0}]}}
    )

    summary = run(run_pipeline(db_session, source, adapter))

    assert db_session.query(EntityRelation).count() == 0
    assert summary.get("relations_created", 0) == 0


def test_context_is_marked_and_not_extracted(db_session, author):
    service, first = make_htr_source(db_session, author, ["конец первой страницы"])
    page = db_session.query(HTRPage).first()
    second_page = HTRPage(
        user_id=author.user_id,
        author_id=author.id,
        file_path="htr_storage/pages/p2.jpg",
        file_name="p2.jpg",
        order_index=1,
        status="CONFIRMED",
    )
    db_session.add(second_page)
    db_session.flush()
    db_session.add(HTRLine(page_id=second_page.id, order_index=0, x1=0, y1=0, x2=1, y2=1, predicted_text="начало второй"))
    db_session.commit()
    # refresh_from_htr_page возвращает (source, created, changed):
    # распаковка в две переменные передавала в конвейер bool
    second = service.refresh_from_htr_page(second_page)[0]

    adapter = FakeAdapter({})
    run(run_pipeline(db_session, second, adapter))

    prompts = adapter.prompts
    assert prompts
    assert all("НЕ извлекай" in prompt or "НИКОГДА не извлекай" in prompt for prompt in prompts)
    context_prompts = [prompt for prompt in prompts if "Контекст соседних страниц" in prompt]
    assert context_prompts
    assert "конец первой страницы" in context_prompts[0]


# ---------------------------------------------------------------------------
# Дедупликация
# ---------------------------------------------------------------------------


def test_similar_goal_is_merged_by_the_resolver(db_session, user):
    _, source = make_diary_source(db_session, user, "Хочу выучить английский язык.")
    first_adapter = FakeAdapter(facts(goal_fact("Выучить английский", "started", "выучить английский язык")))
    run(run_pipeline(db_session, source, first_adapter))
    goal = db_session.query(Goal).one()

    _, second_source = make_diary_source(db_session, user, "Продолжаю выучить английский язык.")
    second_adapter = FakeAdapter(
        {
            **facts(goal_fact("Учить английский", "progress", "выучить английский язык")),
            "matches": {"matches": [{"candidate_index": 0, "matched_id": goal.id}]},
        }
    )
    summary = run(run_pipeline(db_session, second_source, second_adapter))

    assert db_session.query(Goal).count() == 1  # слияние, а не второй заголовок
    assert summary["goals_merged"] == 1
    assert db_session.query(GoalProgress).count() == 2
    assert db_session.query(GoalProgress).filter_by(goal_id=goal.id).count() == 2


def test_dissimilar_goal_is_not_shown_to_the_model(db_session, user):
    _, source = make_diary_source(db_session, user, "Хочу выучить английский язык.")
    run(run_pipeline(db_session, source, FakeAdapter(facts(goal_fact("Выучить английский", "started", "выучить английский язык")))))

    _, second = make_diary_source(db_session, user, "Купил молоко.")
    adapter = FakeAdapter(facts(goal_fact("Купить молоко", "mentioned", "Купил молоко")))
    summary = run(run_pipeline(db_session, second, adapter))

    # кандидат не прошёл порог: резолюция даже не вызывалась
    assert "matches" not in adapter.calls
    assert summary["goals_created"] == 1
    assert db_session.query(Goal).count() == 2


def test_resolution_requires_the_model_not_the_cosine(db_session, user):
    """Высокий косинус без подтверждения модели — не слияние."""
    _, source = make_diary_source(db_session, user, "Хочу начать бег по утрам.")
    run(run_pipeline(db_session, source, FakeAdapter(facts(goal_fact("бег", "started", "начать бег по утрам")))))

    _, second = make_diary_source(db_session, user, "Бег сегодня не задался.")
    adapter = FakeAdapter(
        {
            **facts(goal_fact("бег", "progress", "Бег сегодня")),
            "matches": {"matches": [{"candidate_index": 0, "matched_id": -1}]},  # модель: это разные записи
        }
    )
    summary = run(run_pipeline(db_session, second, adapter))

    assert "matches" in adapter.calls
    assert summary["goals_created"] == 1
    assert db_session.query(Goal).count() == 2


def test_events_with_different_dates_are_never_merged(db_session, user):
    """Событие — факт: дедуп фактов не сливает разные годы одного события."""
    _, source = make_diary_source(db_session, user, "Праздновали день рождения сына в 2023 году.")
    run(run_pipeline(db_session, source, FakeAdapter(facts(
        fact("Праздновали день рождения сына", "праздновали", object_text="день рождения сына", time_text="2023", time_iso="2023", importance=0.6)
    ))))

    _, second = make_diary_source(db_session, user, "Праздновали день рождения сына в 2024 году.")
    summary = run(run_pipeline(db_session, second, FakeAdapter(facts(
        fact("Праздновали день рождения сына", "праздновали", object_text="день рождения сына", time_text="2024", time_iso="2024", importance=0.6)
    ))))

    assert summary.get("facts_merged", 0) == 0
    assert sorted(row.time_iso for row in db_session.query(Fact).all()) == ["2023", "2024"]


def test_same_event_without_a_second_date_is_one_fact(db_session, user):
    """Та же дата (или дата только у одной стороны) — подтверждение, а не дубль."""
    _, source = make_diary_source(db_session, user, "Переехали в Киев в 1960 году.")
    run(run_pipeline(db_session, source, FakeAdapter(facts(
        fact("Переехали в Киев", "переехали", object_text="в Киев", time_iso="1960", importance=0.8)
    ))))

    _, second = make_diary_source(db_session, user, "Помню, как переехали в Киев.")
    summary = run(run_pipeline(db_session, second, FakeAdapter(facts(
        fact("переехали в Киев", "переехали", object_text="в Киев", importance=0.7)
    ))))

    assert summary["facts_merged"] == 1
    moved = db_session.query(Fact).one()
    assert moved.importance == 0.8  # важность не понижается подтверждением
    assert {quote.source_id for quote in moved.quotes} == {source.id, second.id}


def test_embedding_failure_falls_back_to_the_model(db_session, user):
    _, source = make_diary_source(db_session, user, "Хочу выучить английский язык.")
    run(run_pipeline(db_session, source, FakeAdapter(facts(goal_fact("Выучить английский", "started", "выучить английский язык")))))

    _, second = make_diary_source(db_session, user, "Продолжаю учить английский язык.")
    adapter = FakeAdapter(
        {
            **facts(goal_fact("Выучить английский", "progress", "учить английский язык")),
            "matches": {"matches": [{"candidate_index": 0, "matched_id": 1}]},
        }
    )
    service = KnowledgeSourceService(db_session)
    pipeline = ExtractionPipeline(db_session, second, adapter=adapter, embeddings=FakeEmbeddings(unavailable=True), service=service)
    summary = run(pipeline.run())
    db_session.commit()

    assert summary["embedding_fallbacks"] == ["goals"]
    assert summary["goals_merged"] == 1
    assert db_session.query(Goal).count() == 1


# ---------------------------------------------------------------------------
# Независимость вызовов, идемпотентность, чанки
# ---------------------------------------------------------------------------


def test_one_failing_call_does_not_stop_the_others(db_session, user):
    _, source = make_diary_source(db_session, user, "Хочу выучить английский язык. Сегодня бегал.")
    adapter = FakeAdapter(
        facts(goal_fact("Выучить английский", "started", "выучить английский язык")),
        fail_kinds={"entities", "relations"},
    )

    summary = run(run_pipeline(db_session, source, adapter))

    assert summary["goals_created"] == 1
    assert summary["facts_created"] == 1
    assert any(error.startswith("entities") for error in summary["errors"])
    assert any(error.startswith("relations") for error in summary["errors"])
    assert db_session.query(Goal).count() == 1


def test_run_makes_four_calls_and_no_habit_or_event_calls(db_session, user):
    """Привычки — дело аналитики, события и цели — поля факта."""
    _, source = make_diary_source(db_session, user, "Бегаю по утрам.")
    adapter = FakeAdapter({})

    run(run_pipeline(db_session, source, adapter))

    assert adapter.calls == ["entities", "mentions", "facts", "relations"]


def test_every_call_starts_with_the_same_document_block(db_session, author):
    """Общий префикс с текстом страницы — то, на чём держится KV-кэш Ollama."""
    _, source = make_htr_source(db_session, author, ["Мама пекла пироги.", "Отец работал в колхозе."])
    adapter = FakeAdapter({})

    run(run_pipeline(db_session, source, adapter))

    marker = "=== ЗАДАЧА ===\n"
    prefixes = {prompt[: prompt.index(marker) + len(marker)] for prompt in adapter.prompts}
    assert len(prefixes) == 1
    (prefix,) = prefixes
    assert source.text in prefix
    assert "Баба Галя" in prefix  # автор — часть общего блока


def test_goals_are_asked_only_from_the_diary(db_session, user, author):
    _, diary = make_diary_source(db_session, user, "Хочу выучить английский язык.")
    diary_adapter = FakeAdapter({})
    run(run_pipeline(db_session, diary, diary_adapter))

    _, page = make_htr_source(db_session, author, ["Хочу выучить английский язык."])
    htr_adapter = FakeAdapter({})
    run(run_pipeline(db_session, page, htr_adapter))

    diary_facts = diary_adapter.prompts[diary_adapter.calls.index("facts")]
    htr_facts = htr_adapter.prompts[htr_adapter.calls.index("facts")]
    assert "goal_title" in diary_facts
    assert "goal_title" not in htr_facts


def test_goal_fields_from_a_manuscript_page_are_ignored(db_session, author):
    """Схема рукописи целей не просит; если модель всё же их вернула — не пишем."""
    _, page = make_htr_source(db_session, author, ["Хочу выучить английский язык."])
    adapter = FakeAdapter(facts(goal_fact("Выучить английский", "started", "выучить английский язык")))

    summary = run(run_pipeline(db_session, page, adapter))

    assert summary["facts_created"] == 1
    assert db_session.query(Goal).count() == 0


def test_importance_and_recurrence_are_stored_on_the_fact(db_session, user):
    _, source = make_diary_source(db_session, user, "Каждое утро бегаю. Вчера переехали в Казань.")
    adapter = FakeAdapter(facts(
        fact("Каждое утро бегаю", "бегает", recurrence="Каждое утро"),
        fact("переехали в Казань", "переехали", object_text="в Казань", importance=0.8),
    ))

    run(run_pipeline(db_session, source, adapter))

    rows = {row.predicate: row for row in db_session.query(Fact).all()}
    assert rows["бегает"].recurrence == "Каждое утро"
    assert rows["бегает"].importance is None  # 0 — не событие
    assert rows["переехали"].importance == 0.8
    assert rows["переехали"].recurrence is None


def test_second_run_over_the_same_source_does_not_duplicate(db_session, user):
    """Идемпотентность: сервис чистит прогон, конвейер пишет заново."""
    _, source = make_diary_source(db_session, user, "Бегаю по утрам. Хочу выучить английский язык.")
    adapter = FakeAdapter(
        {
            "entities": {"entities": [{"name": "бег", "entity_type": "object", "description": "", "aliases": []}]},
            "mentions": {"mentions": [{"entity_name": "бег", "mention_text": "Бегаю", "sentence_text": "", "confidence": 0.8}]},
            **facts(
                fact("Бегаю по утрам", "бегает", object_text="по утрам", recurrence="по утрам"),
                goal_fact("Выучить английский", "started", "выучить английский язык"),
            ),
        }
    )

    async def pipeline(db, src):
        # фейковая модель вместо настоящей: тест не должен ходить в сеть
        return await ExtractionPipeline(
            db, src, adapter=adapter, embeddings=FakeEmbeddings(), service=KnowledgeSourceService(db)
        ).run()

    service = KnowledgeSourceService(db_session, pipeline=pipeline)

    def counts():
        return {
            "facts": db_session.query(Fact).count(),
            "mentions": db_session.query(EntityMention).count(),
            "goals": db_session.query(Goal).count(),
            "progress": db_session.query(GoalProgress).count(),
            "chunks": db_session.query(Chunk).count(),
        }

    first = run(service.run_now(source))
    counts_after_first = counts()
    db_session.refresh(source)
    run(service.run_now(source))

    assert first["status"] == "done"
    assert counts_after_first["goals"] == 1
    assert counts() == counts_after_first


def test_chunks_are_written_with_embeddings(db_session, user):
    _, source = make_diary_source(db_session, user, "Первое предложение. Второе предложение.")

    summary = run(run_pipeline(db_session, source, FakeAdapter({})))

    chunks = db_session.query(Chunk).order_by(Chunk.chunk_index).all()
    assert summary["chunks_created"] == 1
    assert chunks[0].text == source.text
    assert chunks[0].start_offset == 0
    assert chunks[0].embedding is not None


def test_chunks_without_embeddings_do_not_fail_the_run(db_session, user):
    _, source = make_diary_source(db_session, user, "Текст записи.")

    summary = run(run_pipeline(db_session, source, FakeAdapter({}), embeddings=FakeEmbeddings(unavailable=True)))

    assert summary["chunks_created"] == 1
    assert summary["chunks_without_embeddings"] == 1
    assert db_session.query(Chunk).one().embedding is None


def test_empty_source_is_skipped(db_session, user):
    _, source = make_diary_source(db_session, user, "   ")

    summary = run(run_pipeline(db_session, source, FakeAdapter({})))

    assert summary["skipped"] == "empty source text"
    assert db_session.query(Chunk).count() == 0


def test_source_without_author_refuses_to_run(db_session, user):
    service, source = make_diary_source(db_session, user, "Текст")
    source.author_entity_id = None
    db_session.commit()

    pipeline = ExtractionPipeline(db_session, source, adapter=FakeAdapter({}), embeddings=FakeEmbeddings(), service=service)
    with pytest.raises(ValueError, match="автора-объекта"):
        run(pipeline.run())


def test_goal_offsets_and_sources_are_recorded(db_session, user):
    _, source = make_diary_source(db_session, user, "Решил выучить английский язык к лету.")
    adapter = FakeAdapter(facts(goal_fact("Выучить английский", "started", "выучить английский язык")))

    run(run_pipeline(db_session, source, adapter))

    progress = db_session.query(GoalProgress).one()
    assert progress.source_id == source.id
    assert progress.progress_kind == "started"
    assert source.text[source.text.find(progress.source_text) :][: len(progress.source_text)] == progress.source_text
    goal = db_session.query(Goal).one()
    assert goal.created_from_source_id == source.id
    assert goal.subject_entity_id == source.author_entity_id


def test_goal_belongs_to_the_fact_subject_not_the_author(db_session, user):
    """«Мама решила выучить английский» — цель мамы, а не автора дневника."""
    _, source = make_diary_source(db_session, user, "Мама Ольга решила выучить английский язык.")
    adapter = FakeAdapter({
        "entities": {"entities": [{"name": "Ольга", "entity_type": "person", "description": "", "aliases": []}]},
        **facts(goal_fact(
            "Выучить английский", "started", "решила выучить английский язык",
            subject_kind="named", subject_name="Ольга",
        )),
    })

    run(run_pipeline(db_session, source, adapter))

    goal = db_session.query(Goal).one()
    olga = db_session.query(Entity).filter_by(canonical_name="Ольга").one()
    assert goal.subject_entity_id == olga.id
    assert goal.subject_entity_id != source.author_entity_id


def test_duplicate_relation_in_one_run_is_stored_once(db_session, author):
    """Одно ребро, названное моделью дважды, не должно ронять прогон.

    Уникальное ограничение uq_entity_relation_edge сработало бы на flush: запрос
    к базе второго ребра не видит, потому что первое ещё не вставлено.
    """
    service, source = make_htr_source(db_session, author, ["Мой отец строил дом."])
    adapter = FakeAdapter(
        {
            "entities": {"entities": [{"name": "отец", "entity_type": "person", "description": "", "aliases": []}]},
            "relations": {
                "relations": [
                    {"person_name": "отец", "relation_kind": "father", "source_text": "Мой отец", "confidence": 0.8},
                    {"person_name": "отец", "relation_kind": "father", "source_text": "отец строил", "confidence": 0.9},
                ]
            },
        }
    )

    summary = run(run_pipeline(db_session, source, adapter))

    assert summary["relations_created"] == 1
    edges = db_session.query(EntityRelation).all()
    assert len(edges) == 1
    # побеждает более уверенное упоминание
    assert edges[0].confidence == 0.9


def test_nameless_relative_becomes_an_anchor_not_the_author(db_session, author):
    """«Мой отец» — не имя, но и не автор: это якорная сущность «Отец автора».

    Раньше конвейер подставлял автора по умолчанию, и факт об отце становился
    фактом об авторе. Теперь роль выражена схемой (subject_kind=relative,
    subject_relation=father), а в графе появляется объект с идентичностью
    (автор, father) — «Мы» и «мой отец» как имена по-прежнему отсеиваются.
    """
    service, source = make_htr_source(db_session, author, ["Мой отец строил дом, а мы ему помогали."])
    adapter = FakeAdapter(
        {
            "entities": {
                "entities": [
                    {"name": "мой отец", "entity_type": "person", "description": "", "aliases": []},
                    {"name": "мы", "entity_type": "person", "description": "", "aliases": []},
                    {"name": "Сергей", "entity_type": "person", "description": "", "aliases": []},
                ]
            },
            "facts": {
                "facts": [
                    {
                        "subject_kind": "relative",
                        "subject_name": "",
                        "subject_relation": "father",
                        "subject_of": "self",
                        "predicate": "строил",
                        "object_name": "",
                        "object_text": "дом",
                        "source_text": "Мой отец строил дом",
                        "time_text": "",
                        "confidence": 0.8,
                    }
                ]
            },
        }
    )

    summary = run(run_pipeline(db_session, source, adapter))

    names = {entity.canonical_name for entity in db_session.query(Entity).all()}
    assert "Сергей" in names
    assert "Отец автора" in names
    assert "мой отец" not in names and "мы" not in names
    assert summary["entities_skipped_nameless"] == 2
    assert summary["anchors_created"] == 1

    anchor = db_session.query(Entity).filter_by(canonical_name="Отец автора").one()
    assert anchor.anchor_entity_id == source.author_entity_id
    assert anchor.anchor_relation == "father"

    edge = db_session.query(EntityRelation).one()
    assert edge.from_entity_id == source.author_entity_id
    assert edge.to_entity_id == anchor.id
    assert edge.relation_kind == "father"
    assert edge.relation_type == "отец"

    fact = db_session.query(Fact).one()
    assert fact.subject_entity_id == anchor.id
    assert fact.subject_entity_id != source.author_entity_id
    assert fact.subject_source == "relative"


def test_fact_about_a_named_father_is_not_attributed_to_the_author(db_session, author):
    """«Отец мой Федосив Александр Михайлович родился…» — факт об отце, не об авторе.

    Живой прогон страницы 24 приписал этот факт объекту автора («Alexander»):
    модель не назвала субъекта, а конвейер по умолчанию подставлял автора
    текста. Автор здесь сын — сам себе отцом он быть не может, поэтому
    субъектом становится названный в цитате человек.
    """
    service, source = make_htr_source(
        db_session,
        author,
        [
            "Отец мой Федосив Александр Михайлович",
            "родился 20 августа 1910 года в деревне Игумновцы",
        ],
    )
    adapter = FakeAdapter(
        {
            "entities": {
                "entities": [
                    {
                        "name": "Федосив Александр Михайлович",
                        "entity_type": "person",
                        "description": "",
                        "aliases": [],
                    }
                ]
            },
            "facts": {
                "facts": [
                    {
                        "subject_kind": "relative",
                        "subject_name": "Федосив Александр Михайлович",
                        "subject_relation": "father",
                        "subject_of": "self",
                        "predicate": "родился",
                        "object_name": "",
                        "object_text": "20 августа 1910 года в деревне Игумновцы",
                        "source_text": (
                            "Отец мой Федосив Александр Михайлович\n"
                            "родился 20 августа 1910 года в деревне Игумновцы"
                        ),
                        "time_text": "20 августа 1910 года",
                        "confidence": 0.9,
                    }
                ]
            },
        }
    )

    summary = run(run_pipeline(db_session, source, adapter))

    father = (
        db_session.query(Entity)
        .filter_by(normalized_name="федосив александр михайлович")
        .one()
    )
    assert father.id != source.author_entity_id
    fact = db_session.query(Fact).one()
    assert fact.predicate == "родился"
    # субъект — отец, а не автор страницы
    assert fact.subject_entity_id == father.id
    assert fact.subject_source == "relative"
    # роль закреплена в графе: автор → отец
    edge = db_session.query(EntityRelation).one()
    assert edge.from_entity_id == source.author_entity_id
    assert edge.to_entity_id == father.id
    assert edge.relation_kind == "father"
    assert summary["facts_created"] == 1
    assert "facts_subject_invalid" not in summary


def test_second_different_birth_date_is_a_conflict_not_a_fact(db_session, author):
    """У человека одно рождение: второе, отличающееся значение — на проверку.

    Это инвариант, не зависящий от языка и от родства: «Иван родился в 1910»
    рядом с настоящей датой ловится сам, без единого слова про отца.
    """
    service, source = make_htr_source(
        db_session,
        author,
        ["Иван родился 20 августа 1910 года.", "Потом писали, что Иван родился в 1911 году."],
    )
    adapter = FakeAdapter(
        {
            "entities": {
                "entities": [{"name": "Иван", "entity_type": "person", "description": "", "aliases": []}]
            },
            "facts": {
                "facts": [
                    {
                        "subject_kind": "named",
                        "subject_name": "Иван",
                        "subject_relation": "",
                        "subject_of": "",
                        "predicate": "родился",
                        "predicate_kind": "birth",
                        "object_name": "",
                        "object_text": "",
                        "source_text": "Иван родился 20 августа 1910 года",
                        "time_text": "20 августа 1910 года",
                        "confidence": 0.9,
                    },
                    {
                        "subject_kind": "named",
                        "subject_name": "Иван",
                        "subject_relation": "",
                        "subject_of": "",
                        "predicate": "родился",
                        "predicate_kind": "birth",
                        "object_name": "",
                        "object_text": "",
                        "source_text": "писали, что Иван родился в 1911 году",
                        "time_text": "1911 год",
                        "confidence": 0.7,
                    },
                ]
            },
        }
    )

    summary = run(run_pipeline(db_session, source, adapter))

    assert summary["facts_created"] == 1
    assert summary["facts_conflicts"] == 1
    assert db_session.query(Fact).count() == 1
    conflict = db_session.query(FactConflict).one()
    assert conflict.predicate_kind == "birth"
    assert conflict.existing_value == "20 1910"
    assert conflict.incoming_value == "1911"
    assert conflict.existing_fact_id == db_session.query(Fact).one().id
    assert "1911" in (conflict.source_text or "")


def test_same_birth_date_with_less_precision_is_not_a_conflict(db_session, author):
    """«1910» и «20 августа 1910» — одно рождение: сравниваем терпимо к точности."""
    service, source = make_htr_source(
        db_session, author, ["Иван родился 20 августа 1910 года.", "Иван родился в 1910 году."]
    )
    adapter = FakeAdapter(
        {
            "entities": {
                "entities": [{"name": "Иван", "entity_type": "person", "description": "", "aliases": []}]
            },
            "facts": {
                "facts": [
                    {
                        "subject_kind": "named", "subject_name": "Иван", "subject_relation": "", "subject_of": "",
                        "predicate": "родился", "predicate_kind": "birth",
                        "object_name": "", "object_text": "",
                        "source_text": "Иван родился 20 августа 1910 года",
                        "time_text": "20 августа 1910 года", "confidence": 0.9,
                    },
                    {
                        "subject_kind": "named", "subject_name": "Иван", "subject_relation": "", "subject_of": "",
                        "predicate": "родился", "predicate_kind": "birth",
                        "object_name": "", "object_text": "",
                        "source_text": "Иван родился в 1910 году",
                        "time_text": "1910 год", "confidence": 0.8,
                    },
                ]
            },
        }
    )

    summary = run(run_pipeline(db_session, source, adapter))

    assert summary.get("facts_conflicts", 0) == 0
    assert db_session.query(FactConflict).count() == 0


def test_death_without_a_date_does_not_create_a_second_death(db_session, author):
    """Пустое значение функционального предиката дополняет известный факт.

    На живой базе у одного человека оказались две «смерти»: с датой и без —
    вторая пришла из предложения «умерла», где даты нет. Второй строки быть не
    должно: у человека одна смерть.
    """
    service, source = make_htr_source(
        db_session, author, ["Баба Галя умерла в 1955 году.", "Баба Галя умерла."]
    )
    adapter = FakeAdapter(
        {
            "entities": {
                "entities": [{"name": "Баба Галя", "entity_type": "person", "description": "", "aliases": []}]
            },
            "facts": {
                "facts": [
                    {
                        "subject_kind": "named", "subject_name": "Баба Галя", "subject_relation": "", "subject_of": "",
                        "predicate": "умерла", "predicate_kind": "death",
                        "object_name": "", "object_text": "",
                        "source_text": "Баба Галя умерла в 1955 году",
                        "time_text": "1955 год", "confidence": 0.8,
                    },
                    {
                        "subject_kind": "named", "subject_name": "Баба Галя", "subject_relation": "", "subject_of": "",
                        "predicate": "умерла", "predicate_kind": "death",
                        "object_name": "", "object_text": "",
                        "source_text": "Баба Галя умерла",
                        "time_text": "", "confidence": 0.6,
                    },
                ]
            },
        }
    )

    summary = run(run_pipeline(db_session, source, adapter))

    assert summary["facts_created"] == 1
    assert summary["facts_merged"] == 1
    assert db_session.query(FactConflict).count() == 0
    fact = db_session.query(Fact).one()
    assert fact.predicate_kind == "death"
    assert fact.time_text == "1955 год"
    assert fact.confidence == 0.8


def _birth_fact_answers(source_text, time_text, time_iso, confidence=0.8):
    return {
        "subject_kind": "named",
        "subject_name": "Иван",
        "subject_relation": "",
        "subject_of": "",
        "predicate": "родился",
        "predicate_kind": "birth",
        "object_name": "",
        "object_text": "",
        "source_text": source_text,
        "time_text": time_text,
        "time_iso": time_iso,
        "confidence": confidence,
    }


def test_same_fact_on_two_pages_keeps_both_quotes(db_session, author):
    """Один факт на двух страницах: обе цитаты сохраняются, основная не теряется.

    Раньше при дедупе цитата перезаписывалась, и прежнее подтверждение исчезало.
    """
    service, first = make_htr_source(db_session, author, ["Иван родился в 1910 году."])
    adapter = FakeAdapter(
        {
            "entities": {"entities": [{"name": "Иван", "entity_type": "person", "description": "", "aliases": []}]},
            "facts": {"facts": [_birth_fact_answers("Иван родился в 1910 году", "1910 году", "1910")]},
        }
    )
    run(run_pipeline(db_session, first, adapter))

    _, second = make_htr_source(db_session, author, ["Иван родился в 1910 году."])
    summary = run(run_pipeline(db_session, second, adapter))

    assert summary.get("facts_created", 0) == 0
    assert summary["facts_merged"] == 1
    assert db_session.query(Fact).count() == 1
    fact = db_session.query(Fact).one()
    # основная цитата остаётся от первой страницы
    assert fact.source_id == first.id
    assert fact.source_text == "Иван родился в 1910 году"
    assert {quote.source_id for quote in fact.quotes} == {first.id, second.id}
    assert db_session.query(FactSource).count() == 2


def test_iso_date_makes_two_spellings_one_fact(db_session, author):
    """«20 августа 1910» и «1910 год» — одна дата: модель привела её к 1910."""
    service, source = make_htr_source(
        db_session, author, ["Иван родился 20 августа 1910 года.", "Иван родился в 1910 году."]
    )
    adapter = FakeAdapter(
        {
            "entities": {"entities": [{"name": "Иван", "entity_type": "person", "description": "", "aliases": []}]},
            "facts": {
                "facts": [
                    _birth_fact_answers("Иван родился 20 августа 1910 года", "20 августа 1910 года", "1910-08-20", 0.9),
                    _birth_fact_answers("Иван родился в 1910 году", "1910 году", "1910", 0.8),
                ]
            },
        }
    )

    summary = run(run_pipeline(db_session, source, adapter))

    assert summary["facts_created"] == 1
    assert summary["facts_merged"] == 1
    assert db_session.query(FactConflict).count() == 0


def test_iso_dates_that_differ_are_a_conflict(db_session, author):
    """Двадцать первое августа не сходится с двадцатым, даже если текст один."""
    service, source = make_htr_source(
        db_session, author, ["Иван родился 20 августа 1910 года.", "Иван родился 21 августа 1910 года."]
    )
    adapter = FakeAdapter(
        {
            "entities": {"entities": [{"name": "Иван", "entity_type": "person", "description": "", "aliases": []}]},
            "facts": {
                "facts": [
                    _birth_fact_answers("Иван родился 20 августа 1910 года", "20 августа 1910 года", "1910-08-20", 0.9),
                    _birth_fact_answers("Иван родился 21 августа 1910 года", "21 августа 1910 года", "1910-08-21", 0.7),
                ]
            },
        }
    )

    summary = run(run_pipeline(db_session, source, adapter))

    assert summary["facts_created"] == 1
    assert summary["facts_conflicts"] == 1
    conflict = db_session.query(FactConflict).one()
    assert conflict.existing_value == "1910-08-20"
    assert conflict.incoming_value == "1910-08-21"


def test_graph_hint_resolves_the_subject_by_id(db_session, author):
    """Модель ссылается на известного человека по id: субъект берётся из графа.

    Перед извлечением ей показывают близких автора с их id, поэтому «Пётр» в
    тексте не заводится заново под другим написанием, а узнаётся.
    """
    service, source = make_htr_source(db_session, author, ["Пётр родился в 1910 году."])
    father = Entity(
        user_id=author.user_id, entity_type="person", canonical_name="Пётр", normalized_name="пётр"
    )
    db_session.add(father)
    db_session.commit()
    db_session.add(
        EntityRelation(
            user_id=author.user_id,
            from_entity_id=source.author_entity_id,
            to_entity_id=father.id,
            relation_type="отец",
            relation_kind="father",
        )
    )
    db_session.commit()

    adapter = FakeAdapter(
        {
            "facts": {
                "facts": [
                    {
                        "subject_kind": "named",
                        "subject_name": "Пётр",
                        "subject_relation": "",
                        "subject_of": "",
                        "subject_id": father.id,
                        "predicate": "родился",
                        "predicate_kind": "birth",
                        "object_name": "",
                        "object_text": "",
                        "source_text": "Пётр родился в 1910 году",
                        "time_text": "1910 году",
                        "time_iso": "1910",
                        "confidence": 0.8,
                    }
                ]
            }
        }
    )

    summary = run(run_pipeline(db_session, source, adapter))

    fact = db_session.query(Fact).one()
    assert fact.subject_entity_id == father.id
    assert fact.subject_source == "graph"
    assert summary["facts_subject_graph"] == 1
    facts_prompt = [p for p in adapter.prompts if "Ты извлекаешь факты" in p][0]
    assert "Близкие автора" in facts_prompt
    assert f"Пётр [{father.id}]" in facts_prompt


def test_relative_with_several_candidates_is_decided_by_the_resolver(db_session, author):
    """Два брата на одну роль: выбирает резолвер, а не структура.

    Якорь один на роль только для «одного отца»: братьев может быть несколько,
    и выбор между ними — вопрос смысла, а не графа.
    """
    service, source = make_htr_source(db_session, author, ["Мой брат приехал в гости."])
    brothers = []
    for name, normalized in (("Пётр", "пётр"), ("Сергей", "сергей")):
        entity = Entity(
            user_id=author.user_id, entity_type="person", canonical_name=name, normalized_name=normalized
        )
        db_session.add(entity)
        db_session.commit()
        db_session.add(
            EntityRelation(
                user_id=author.user_id,
                from_entity_id=source.author_entity_id,
                to_entity_id=entity.id,
                relation_type="брат",
                relation_kind="brother",
            )
        )
        brothers.append(entity)
    db_session.commit()

    adapter = FakeAdapter(
        {
            "facts": {
                "facts": [
                    {
                        "subject_kind": "relative",
                        "subject_name": "",
                        "subject_relation": "brother",
                        "subject_of": "self",
                        "subject_id": 0,
                        "predicate": "приехал",
                        "predicate_kind": "other",
                        "object_name": "",
                        "object_text": "в гости",
                        "source_text": "Мой брат приехал в гости",
                        "time_text": "",
                        "time_iso": "",
                        "confidence": 0.8,
                    }
                ]
            },
            "matches": {"matches": [{"candidate_index": 0, "matched_id": brothers[1].id}]},
        }
    )

    summary = run(run_pipeline(db_session, source, adapter))

    fact = db_session.query(Fact).one()
    assert fact.subject_entity_id == brothers[1].id
    assert fact.subject_source == "relative"
    assert summary["relative_resolved"] == 1


def test_common_nouns_are_not_persons(db_session, author):
    """Нарицательное слово без имени не становится объектом-человеком.

    Живой прогон страницы 33 вернул «мох», «дела» и «дом» типом person: они
    попали в список персон на вкладке «Знания» и в агрегат аналитики.
    """
    service, source = make_htr_source(db_session, author, ["Ходили за мохом в лес, дома были дела."])
    adapter = FakeAdapter(
        {
            "entities": {
                "entities": [
                    {"name": "мох", "entity_type": "person", "description": "", "aliases": []},
                    {"name": "дела", "entity_type": "person", "description": "", "aliases": []},
                    {"name": "дом", "entity_type": "person", "description": "", "aliases": []},
                    {"name": "Анна", "entity_type": "person", "description": "", "aliases": []},
                ]
            }
        }
    )

    summary = run(run_pipeline(db_session, source, adapter))

    names = {entity.canonical_name for entity in db_session.query(Entity).all()}
    assert "Анна" in names
    for word in ("мох", "дела", "дом"):
        assert word not in names
    assert summary["entities_skipped_nameless"] == 3


# ---------------------------------------------------------------------------
# Субъект факта проверяется по родству, а не только по id и имени
# ---------------------------------------------------------------------------


def _person(db, user, name, description=None):
    row = Entity(user_id=user.id, entity_type="person", canonical_name=name, normalized_name=name.lower(), description=description)
    db.add(row)
    db.flush()
    return row


def _link(db, source, frm, to, kind):
    """Ребро графа так же, как его пишет конвейер (каноническая пара)."""
    pipeline = ExtractionPipeline(db, source, adapter=FakeAdapter({}), embeddings=FakeEmbeddings(), service=KnowledgeSourceService(db))
    pipeline._ensure_relation(frm, to, kind)
    db.commit()


def _family(db, user):
    """Автор дневника, его мама Елена Юрьевна и дядя Виталий."""
    _, source = make_diary_source(db, user, "Виталий жил у дочери Лены в Киеве, она и сейчас там живет.")
    author = db.query(Entity).get(source.author_entity_id)
    mother = _person(db, user, "Елена Юрьевна", "мама автора")
    uncle = _person(db, user, "Виталий", "дядя автора")
    _link(db, source, author, mother, "mother")
    _link(db, source, author, uncle, "uncle")
    return source, author, mother, uncle


def lena_fact(**extra):
    return fact(
        "она и сейчас там живет", "живет", subject_kind="relative", subject_name="Лена",
        subject_relation="daughter", subject_of="Виталий", object_text="в Киеве", **extra,
    )


def test_subject_id_contradicting_the_kinship_is_rejected(db_session, user):
    """«Лена, дочь Виталия» с id мамы автора — это не мама автора."""
    source, author, mother, uncle = _family(db_session, user)

    summary = run(run_pipeline(db_session, source, FakeAdapter(facts(lena_fact(subject_id=mother.id)))))

    lives = db_session.query(Fact).filter_by(predicate="живет").one()
    assert lives.subject_entity_id != mother.id
    lena = db_session.query(Entity).get(lives.subject_entity_id)
    assert lena.canonical_name == "Лена"
    assert summary["facts_subject_id_rejected"] == 1
    # и родство записано от Виталия, а не от автора
    edge = db_session.query(EntityRelation).filter(EntityRelation.relation_kind == "daughter").one()
    assert {edge.from_entity_id, edge.to_entity_id} == {uncle.id, lena.id}


def test_name_merged_into_the_wrong_person_is_not_trusted(db_session, user):
    """Резолвер сущностей приписал «Лену» Елене Юрьевне — родство это опровергает."""
    source, author, mother, uncle = _family(db_session, user)
    adapter = FakeAdapter({
        "entities": {"entities": [{"name": "Лена", "entity_type": "person", "description": "дочь Виталия", "aliases": []}]},
        "matches": {"matches": [{"candidate_index": 0, "matched_id": mother.id}]},  # ошибка резолвера
        **facts(lena_fact()),
    })

    run(run_pipeline(db_session, source, adapter))

    lives = db_session.query(Fact).filter_by(predicate="живет").one()
    assert lives.subject_entity_id != mother.id
    assert db_session.query(Entity).get(lives.subject_entity_id).canonical_name == "Лена"


def test_subject_id_that_agrees_with_the_kinship_is_kept(db_session, user):
    source, author, mother, uncle = _family(db_session, user)
    adapter = FakeAdapter(facts(fact(
        "Виталий жил у дочери Лены в Киеве", "жил", subject_kind="relative", subject_name="Виталий",
        subject_relation="uncle", subject_of="self", subject_id=uncle.id,
    )))

    summary = run(run_pipeline(db_session, source, adapter))

    assert db_session.query(Fact).filter_by(predicate="жил").one().subject_entity_id == uncle.id
    assert summary["facts_subject_graph"] == 1
    assert "facts_subject_id_rejected" not in summary


def test_named_subject_id_with_another_name_falls_back_to_the_name(db_session, user):
    source, author, mother, uncle = _family(db_session, user)
    adapter = FakeAdapter(facts(fact(
        "Виталий жил у дочери Лены в Киеве", "жил", subject_kind="named", subject_name="Виталий", subject_id=mother.id,
    )))

    summary = run(run_pipeline(db_session, source, adapter))

    assert db_session.query(Fact).filter_by(predicate="жил").one().subject_entity_id == uncle.id
    assert summary["facts_subject_id_rejected"] == 1


def test_relatives_of_one_page_are_resolved_in_one_call(db_session, user):
    """Два брата на роль: все факты страницы — один вызов резолвера."""
    _, source = make_diary_source(db_session, user, "Брат Павел служил. Брат Павел воевал. Брат Виталий умер.")
    author = db_session.query(Entity).get(source.author_entity_id)
    pavel = _person(db_session, user, "Павел")
    vitaly = _person(db_session, user, "Виталий")
    _link(db_session, source, author, pavel, "brother")
    _link(db_session, source, author, vitaly, "brother")
    adapter = FakeAdapter({
        **facts(
            fact("Брат Павел служил", "служил", subject_kind="relative", subject_relation="brother", subject_of="self"),
            fact("Брат Павел воевал", "воевал", subject_kind="relative", subject_relation="brother", subject_of="self"),
            fact("Брат Виталий умер", "умер", subject_kind="relative", subject_relation="brother", subject_of="self", subject_name="Виталий"),
        ),
        "matches": {"matches": [{"candidate_index": 0, "matched_id": pavel.id}, {"candidate_index": 1, "matched_id": vitaly.id}]},
    })

    run(run_pipeline(db_session, source, adapter))

    assert adapter.calls.count("matches") == 1
    subjects = {row.predicate: row.subject_entity_id for row in db_session.query(Fact).all()}
    assert subjects == {"служил": pavel.id, "воевал": pavel.id, "умер": vitaly.id}


def test_entity_resolver_sees_descriptions_not_only_names(db_session, user):
    _, source = make_diary_source(db_session, user, "Лена приехала из Киева.")
    _person(db_session, user, "Елена Юрьевна", "мама автора")
    db_session.commit()
    adapter = FakeAdapter({
        "entities": {"entities": [{"name": "Лена", "entity_type": "person", "description": "дочь Виталия", "aliases": []}]},
    })

    run(run_pipeline(db_session, source, adapter, embeddings=FakeEmbeddings(unavailable=True)))

    resolver_prompt = adapter.prompts[adapter.calls.index("matches")]
    assert "Лена — дочь Виталия" in resolver_prompt
    assert "Елена Юрьевна — мама автора" in resolver_prompt
