"""Сквозная проверка резолюции сущностей на общем коде.

Юнит-тесты конвейера подменяют эмбеддинги, а здесь важно, что резолюция
сущностей действительно доходит до модели и что слияние происходит по её
решению, а не по косинусу.
"""
from __future__ import annotations

import asyncio
import json

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import Base
from app.models import DiaryPage, Entity, User
from app.services.extraction_pipeline import ExtractionPipeline
from app.services.knowledge_source_service import KnowledgeSourceService


class FakeAdapter:
    def __init__(self, answers):
        self.answers, self.calls = answers, []

    async def generate_structured(self, prompt, schema):
        key = "matches" if "matches" in schema.get("properties", {}) else next(iter(schema["properties"]))
        self.calls.append(key)
        return json.dumps(self.answers.get(key, {}), ensure_ascii=False)


class WordEmbeddings:
    """Косинус по «темам»: Иван и Ваня — одно, Маша и Пётр — разное.

    Стемминга тут нет намеренно: важно, что отбор кандидатов — это про близость
    строк, а не про знание имён, и что решение всё равно принимает модель.
    """

    THEMES = {
        "иван": "иван", "ваня": "иван", "ване": "иван", "ивану": "иван",
        "маша": "маша", "маше": "маша", "марии": "маша",
        "пётр": "пётр", "петр": "пётр", "петру": "пётр",
    }

    async def embed_texts(self, texts, prompt_name=None):
        vocabulary = ("иван", "маша", "пётр")
        vectors = []
        for text in texts:
            themes = {self.THEMES[word] for word in str(text).lower().replace(",", " ").split() if word in self.THEMES}
            vectors.append([1.0 if theme in themes else 0.0 for theme in vocabulary])
        return vectors


def make_db():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    db = sessionmaker(bind=engine)()
    user = User(username="u", password_hash="x", name="U")
    db.add(user)
    db.commit()
    return db, user


def add_page(db, user, content):
    page = DiaryPage(user_id=user.id, content=content)
    db.add(page)
    db.commit()
    service = KnowledgeSourceService(db)
    source, _, _ = service.refresh_from_diary_page(page)
    return service, source


async def run_source(db, source, service, adapter):
    pipeline = ExtractionPipeline(db, source, adapter=adapter, embeddings=WordEmbeddings(), service=service)
    summary = await pipeline.run()
    db.commit()
    return summary


def test_two_ivan_spellings_are_merged_by_the_model():
    db, user = make_db()
    service, first = add_page(db, user, "Приходил Иван.")
    asyncio.run(
        run_source(
            db,
            first,
            service,
            FakeAdapter({"entities": {"entities": [{"name": "Иван", "entity_type": "person", "description": "", "aliases": []}]}}),
        )
    )
    entity = db.query(Entity).filter_by(canonical_name="Иван").one()

    _, second = add_page(db, user, "Иван звонил вечером.")
    adapter = FakeAdapter(
        {
            "entities": {"entities": [{"name": "Иван", "entity_type": "person", "description": "", "aliases": ["Ваня"]}]},
            "matches": {"matches": [{"candidate_index": 0, "matched_id": entity.id}]},
        }
    )
    summary = asyncio.run(run_source(db, second, service, adapter))

    assert "matches" in adapter.calls  # резолюция действительно вызвана
    assert summary["entities_matched"] == 1
    assert db.query(Entity).filter(Entity.entity_type == "person").count() == 2  # Иван и «U» (self)
    merged = db.query(Entity).filter_by(canonical_name="Иван").one()
    # слияние пополнило алиасы, а не завело вторую персону
    assert json.loads(merged.aliases_json) == ["Ваня"]


def test_unrelated_names_never_reach_the_resolver():
    db, user = make_db()
    service, first = add_page(db, user, "Приходила Маша.")
    asyncio.run(
        run_source(
            db,
            first,
            service,
            FakeAdapter({"entities": {"entities": [{"name": "Маша", "entity_type": "person", "description": "", "aliases": []}]}}),
        )
    )

    _, second = add_page(db, user, "Звонил Пётр.")
    adapter = FakeAdapter({"entities": {"entities": [{"name": "Пётр", "entity_type": "person", "description": "", "aliases": []}]}})
    summary = asyncio.run(run_source(db, second, service, adapter))

    assert "matches" not in adapter.calls  # косинус отсеял, модель не тратилась
    assert summary["entities_created"] == 1
