"""Тесты сервиса источников знаний: сборка, статусы, запуск, переизвлечение.

Конвейер подменяется фейком, поэтому проверяется именно то, чем владеет этап 3:
текст и карта строк, привязка автора-объекта, контекст соседних страниц, защита
от двойного запуска, транзакционность переизвлечения и статусы.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from app.models import (
    DiaryPage,
    Entity,
    EntityMention,
    EntityRelation,
    Event,
    ExtractionRun,
    Fact,
    Goal,
    GoalProgress,
    Habit,
    HabitLog,
    HTRAuthor,
    HTRLine,
    HTRPage,
    KnowledgeSource,
    KnowledgeState,
    User,
)
from app.services import knowledge_source_service as module
from app.services.knowledge_source_service import (
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_PENDING,
    STATUS_RUNNING,
    KnowledgeSourceService,
    SourceError,
    source_status_payload,
)
from app.services.source_text import parse_line_map


def run(coro):
    return asyncio.run(coro)


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


def make_page(db, author, order_index=0, status="CONFIRMED", lines=(("первая строка", None), ("вторая", "вторая!"))):
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
    for index, (predicted, corrected) in enumerate(lines):
        db.add(
            HTRLine(
                page_id=page.id,
                order_index=index,
                x1=0,
                y1=index,
                x2=10,
                y2=index + 1,
                predicted_text=predicted,
                corrected_text=corrected,
            )
        )
    db.commit()
    return page


def make_diary_page(db, user, content="Сегодня бегал по утрам.", title="Утро"):
    page = DiaryPage(user_id=user.id, title=title, content=content)
    db.add(page)
    db.commit()
    return page


def fake_pipeline(created_entities):
    """Пишет ровно то, что писал бы конвейер: упоминание, факт и объект."""

    async def pipeline(db, source):
        entity = Entity(
            user_id=source.user_id,
            entity_type="person",
            canonical_name="Бег",
            normalized_name="бег",
        )
        db.add(entity)
        db.flush()
        created_entities.append(entity.id)
        db.add(
            Fact(
                user_id=source.user_id,
                source_id=source.id,
                subject_entity_id=source.author_entity_id,
                predicate="бегает",
                object_text="по утрам",
                source_text=source.text[:20],
                start_offset=0,
                end_offset=min(20, len(source.text)),
            )
        )
        return {"facts_created": 1}

    return pipeline


# ---------------------------------------------------------------------------
# Источники: дневник
# ---------------------------------------------------------------------------


def test_diary_page_becomes_a_source(db_session, user):
    service = KnowledgeSourceService(db_session)
    page = make_diary_page(db_session, user)

    source, created, changed = service.refresh_from_diary_page(page)

    assert created is True and changed is True
    assert source.source_type == "diary"
    assert source.ref_id == page.id
    assert source.text == page.content
    assert source.line_map is None
    assert source.extraction_status == STATUS_PENDING
    # автор дневника — объект самого аккаунта
    self_entity = service.ensure_self_entity(user.id)
    assert source.author_entity_id == self_entity.id
    assert user.self_entity_id == self_entity.id


def test_repeating_the_refresh_changes_nothing(db_session, user):
    service = KnowledgeSourceService(db_session)
    page = make_diary_page(db_session, user)
    service.refresh_from_diary_page(page)

    _, created, changed = service.refresh_from_diary_page(page)

    assert (created, changed) == (False, False)


def test_editing_the_page_marks_the_source_pending(db_session, user):
    service = KnowledgeSourceService(db_session)
    page = make_diary_page(db_session, user)
    source, _, _ = service.refresh_from_diary_page(page)
    source.extraction_status = STATUS_DONE
    db_session.commit()

    page.content = "Сегодня бегал по утрам и потом спал."
    db_session.commit()
    source, created, changed = service.refresh_from_diary_page(page)

    assert created is False and changed is True
    assert source.extraction_status == STATUS_PENDING
    assert source.text == page.content


# ---------------------------------------------------------------------------
# Источники: рукопись
# ---------------------------------------------------------------------------


def test_htr_page_gets_text_line_map_and_author(db_session, author):
    service = KnowledgeSourceService(db_session)
    page = make_page(db_session, author)

    source, created, _ = service.refresh_from_htr_page(page)

    assert created is True
    # правленый текст важнее предсказанного, строки склеены переводом строки
    assert source.text == "первая строка\nвторая!"
    assert source.title == "page-0.jpg"
    line_map = parse_line_map(source.line_map)
    assert [entry["start"] for entry in line_map] == [0, 14]
    assert source.text[slice(line_map[1]["start"], line_map[1]["end"])] == "вторая!"
    # автор рукописи получил собственный объект
    db_session.refresh(author)
    assert author.entity_id is not None
    assert source.author_entity_id == author.entity_id
    entity = db_session.query(Entity).filter(Entity.id == author.entity_id).one()
    assert entity.canonical_name == "Баба Галя"
    assert entity.id != service.ensure_self_entity(author.user_id).id


def test_htr_source_requires_a_confirmed_page(db_session, author):
    service = KnowledgeSourceService(db_session)
    page = make_page(db_session, author, status="EDITING")

    with pytest.raises(SourceError, match="confirmed"):
        service.refresh_from_htr_page(page)


def test_author_me_links_to_the_account_object(db_session, user):
    service = KnowledgeSourceService(db_session)
    me = HTRAuthor(user_id=user.id, name="me")
    db_session.add(me)
    db_session.commit()
    page = make_page(db_session, me)

    source, _, _ = service.refresh_from_htr_page(page)

    assert source.author_entity_id == service.ensure_self_entity(user.id).id


# ---------------------------------------------------------------------------
# Контекст соседних страниц
# ---------------------------------------------------------------------------


def test_context_block_uses_the_neighbouring_pages(db_session, author, settings_override=None):
    service = KnowledgeSourceService(db_session)
    make_page(db_session, author, order_index=0, lines=(("самый конец первой", None),))
    middle = make_page(db_session, author, order_index=1, lines=(("середина", None),))
    make_page(db_session, author, order_index=2, lines=(("начало третьей", None),))

    source, _, _ = service.refresh_from_htr_page(middle)
    block = service.context_block(source)

    assert "НЕ извлекай" in block
    assert "конец первой" in block
    assert "начало третьей" in block
    # текст самой страницы в контекст не попадает
    assert "середина" not in block


def test_context_block_is_empty_without_neighbours(db_session, author):
    service = KnowledgeSourceService(db_session)
    page = make_page(db_session, author, order_index=0)
    source, _, _ = service.refresh_from_htr_page(page)

    assert service.context_block(source) == ""


def test_context_block_is_empty_for_diary(db_session, user):
    service = KnowledgeSourceService(db_session)
    source, _, _ = service.refresh_from_diary_page(make_diary_page(db_session, user))

    assert service.context_block(source) == ""


# ---------------------------------------------------------------------------
# Запуск конвейера
# ---------------------------------------------------------------------------


def test_run_now_marks_done_and_records_the_run(db_session, user):
    service = KnowledgeSourceService(db_session, pipeline=fake_pipeline([]))
    source, _, _ = service.refresh_from_diary_page(make_diary_page(db_session, user))
    before = db_session.query(KnowledgeState).filter_by(user_id=user.id).first()

    result = run(service.run_now(source))

    assert result["status"] == STATUS_DONE
    db_session.refresh(source)
    assert source.extraction_status == STATUS_DONE
    assert source.last_extracted_at is not None
    run_row = db_session.query(ExtractionRun).filter_by(source_id=source.id).one()
    assert run_row.status == "succeeded"
    assert json.loads(run_row.stats)["facts_created"] == 1
    version = db_session.query(KnowledgeState).filter_by(user_id=user.id).one().version
    assert version == (before.version if before else 0) + 1


def test_run_now_replaces_the_previous_knowledge(db_session, user):
    service = KnowledgeSourceService(db_session, pipeline=fake_pipeline([]))
    source, _, _ = service.refresh_from_diary_page(make_diary_page(db_session, user))

    run(service.run_now(source))
    run(service.run_now(source))

    # факт один: старый удалён, новый вставлен — дублей нет
    assert db_session.query(Fact).filter_by(source_id=source.id).count() == 1
    assert db_session.query(KnowledgeSource).filter_by(id=source.id).count() == 1


def test_run_now_skips_a_source_already_running(db_session, user):
    called = []

    async def pipeline(db, source):
        called.append(source.id)
        return {}

    service = KnowledgeSourceService(db_session, pipeline=pipeline)
    source, _, _ = service.refresh_from_diary_page(make_diary_page(db_session, user))
    source.extraction_status = STATUS_RUNNING
    db_session.commit()

    result = run(service.run_now(source))

    assert result["skipped"] == "already running"
    assert called == []


def test_unimplemented_pipeline_leaves_the_source_pending(db_session, user):
    service = KnowledgeSourceService(db_session)  # конвейер этапа 4 ещё заглушка
    source, _, _ = service.refresh_from_diary_page(make_diary_page(db_session, user))

    result = run(service.run_now(source))

    assert result["status"] == STATUS_PENDING
    db_session.refresh(source)
    assert source.extraction_status == STATUS_PENDING
    assert source.error is None
    # записи о прогоне не остаётся: конвейер даже не начинался
    assert db_session.query(ExtractionRun).filter_by(source_id=source.id).count() == 0


def test_failing_pipeline_keeps_the_old_knowledge(db_session, user):
    service = KnowledgeSourceService(db_session, pipeline=fake_pipeline([]))
    source, _, _ = service.refresh_from_diary_page(make_diary_page(db_session, user))
    run(service.run_now(source))
    assert db_session.query(Fact).filter_by(source_id=source.id).count() == 1

    async def broken(db, source):
        raise RuntimeError("LLM недоступна")

    failing = KnowledgeSourceService(db_session, pipeline=broken)
    result = run(failing.run_now(source))

    assert result["status"] == STATUS_FAILED
    db_session.refresh(source)
    assert source.extraction_status == STATUS_FAILED
    assert "LLM" in (source.error or "")
    # откат вернул прежнее знание: страница не осталась пустой
    assert db_session.query(Fact).filter_by(source_id=source.id).count() == 1


def test_source_without_author_object_is_failed_clearly(db_session, user, author):
    service = KnowledgeSourceService(db_session, pipeline=fake_pipeline([]))
    page = make_page(db_session, author)
    source, _, _ = service.refresh_from_htr_page(page)
    source.author_entity_id = None
    db_session.commit()

    result = run(service.run_now(source))

    assert result["status"] == STATUS_FAILED
    db_session.refresh(source)
    assert "автора-объекта" in (source.error or "")


def test_async_mode_queues_a_celery_task(db_session, user, monkeypatch):
    service = KnowledgeSourceService(db_session, pipeline=fake_pipeline([]))
    source, _, _ = service.refresh_from_diary_page(make_diary_page(db_session, user))
    queued = []

    class FakeTask:
        id = "task-42"

    monkeypatch.setattr(
        "app.tasks.extract_knowledge_task.delay", lambda source_id: queued.append(source_id) or FakeTask()
    )

    result = run(service.request_extraction(source, mode="async"))

    assert result["queued"] is True and result["task_id"] == "task-42"
    assert queued == [source.id]
    db_session.refresh(source)
    assert source.extraction_status == STATUS_PENDING  # воркер ещё не начал


# ---------------------------------------------------------------------------
# Переизвлечение и очистка (7.4)
# ---------------------------------------------------------------------------


def test_reset_drops_knowledge_and_orphan_heads(db_session, user):
    service = KnowledgeSourceService(db_session)
    source, _, _ = service.refresh_from_diary_page(make_diary_page(db_session, user))
    self_entity = service.ensure_self_entity(user.id)
    other_source = KnowledgeSource(user_id=user.id, source_type="diary", ref_id=999, text="другое")
    db_session.add(other_source)
    db_session.flush()

    def person(name):
        row = Entity(user_id=user.id, entity_type="person", canonical_name=name, normalized_name=name.lower())
        db_session.add(row)
        db_session.flush()
        return row

    referenced = person("Маша")
    keeps_other = person("Пётр")
    unreferenced = person("Никто")
    db_session.add(Fact(user_id=user.id, source_id=source.id, predicate="видел", source_text="Маша", object_entity_id=referenced.id))
    db_session.add(Fact(user_id=user.id, source_id=other_source.id, predicate="видел", source_text="Пётр", object_entity_id=keeps_other.id))
    db_session.add(EntityMention(entity_id=referenced.id, source_id=source.id, mention_text="Маша"))
    db_session.add(EntityRelation(user_id=user.id, from_entity_id=self_entity.id, to_entity_id=referenced.id, relation_type="подруга", source_id=source.id))

    only_source = Goal(user_id=user.id, title="Бросить курить", created_from_source_id=source.id)
    kept_goal = Goal(user_id=user.id, title="Выучить испанский", created_from_source_id=source.id)
    db_session.add_all([only_source, kept_goal])
    db_session.flush()
    db_session.add(GoalProgress(goal_id=only_source.id, source_id=source.id, progress_kind="started"))
    db_session.add(GoalProgress(goal_id=kept_goal.id, source_id=other_source.id, progress_kind="progress"))
    lonely_habit = Habit(user_id=user.id, title="Забытая привычка")
    kept_habit = Habit(user_id=user.id, title="Бег")
    db_session.add_all([lonely_habit, kept_habit])
    db_session.flush()
    db_session.add(HabitLog(habit_id=lonely_habit.id, source_id=source.id))
    db_session.add(HabitLog(habit_id=kept_habit.id, source_id=other_source.id))
    db_session.commit()
    ids = {
        "only_goal": only_source.id,
        "kept_goal": kept_goal.id,
        "lonely_habit": lonely_habit.id,
        "kept_habit": kept_habit.id,
        "self": self_entity.id,
        "keeps_other": keeps_other.id,
    }

    removed = service.reset_source_knowledge(source)
    db_session.commit()

    assert removed["facts"] == 1
    assert removed["entity_mentions"] == 1
    assert removed["entity_relations"] == 1
    assert removed["goals"] == 1
    assert removed["habits"] == 1
    assert removed["entities"] == 2  # Маша и Никто; Пётр держится другим фактом
    assert db_session.query(Fact).filter_by(source_id=source.id).count() == 0
    assert db_session.query(Goal).filter_by(id=ids["only_goal"]).count() == 0
    assert db_session.query(Goal).filter_by(id=ids["kept_goal"]).count() == 1
    assert db_session.query(Habit).filter_by(id=ids["lonely_habit"]).count() == 0
    assert db_session.query(Habit).filter_by(id=ids["kept_habit"]).count() == 1
    assert db_session.query(Entity).filter_by(id=ids["self"]).count() == 1
    assert db_session.query(Entity).filter_by(id=ids["keeps_other"]).count() == 1


def test_mark_pending_hides_the_knowledge(db_session, user):
    service = KnowledgeSourceService(db_session, pipeline=fake_pipeline([]))
    source, _, _ = service.refresh_from_diary_page(make_diary_page(db_session, user))
    run(service.run_now(source))
    assert db_session.query(Fact).filter_by(source_id=source.id).count() == 1

    removed = service.mark_pending(source)

    assert removed["facts"] == 1
    db_session.refresh(source)
    assert source.extraction_status == STATUS_PENDING


def test_delete_source_removes_it_with_its_knowledge(db_session, user):
    service = KnowledgeSourceService(db_session, pipeline=fake_pipeline([]))
    source, _, _ = service.refresh_from_diary_page(make_diary_page(db_session, user))
    run(service.run_now(source))

    service.delete_source(source)

    assert db_session.query(KnowledgeSource).filter_by(id=source.id).count() == 0
    assert db_session.query(Fact).filter_by(source_id=source.id).count() == 0


def test_status_payload_reports_the_last_run(db_session, user):
    service = KnowledgeSourceService(db_session, pipeline=fake_pipeline([]))
    source, _, _ = service.refresh_from_diary_page(make_diary_page(db_session, user))
    run(service.run_now(source))

    payload = source_status_payload(db_session, source)

    assert payload["status"] == STATUS_DONE
    assert payload["ref_id"] == source.ref_id
    assert payload["source_type"] == "diary"
    assert payload["last_run"]["status"] == "succeeded"
    assert payload["last_run"]["stats"]["facts_created"] == 1
    assert payload["error"] is None
    assert module.STATUS_PENDING == "pending"
