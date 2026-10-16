"""Пробный прогон извлечения по одной странице — без записи в базу.

Чтобы сравнить модели (локальную qwen3 и облачные Claude, OpenAI, DeepSeek) на
одном и том же тексте, конвейер выполняется целиком — вызовы, резолюция,
запись, — но внутри транзакции, которая в конце откатывается. Печатается, какие
факты вернула модель, что из них записалось бы, сколько это заняло времени.

Запуск (из каталога ``backend``)::

    python -m app.scripts.try_extraction htr 177                       # модель из настроек
    python -m app.scripts.try_extraction htr 177 --provider anthropic  # claude-opus-5-5
    python -m app.scripts.try_extraction htr 177 --provider deepseek --runs 2
    python -m app.scripts.try_extraction diary 14 --provider openai --model <модель>

Ключи берутся из настроек (``backend/.env``): ``ANTHROPIC_API_KEY``,
``OPENAI_API_KEY``, ``DEEPSEEK_API_KEY``; прокси для облака — ``LLM_API_PROXY``.
Облачный прогон стоит денег: один прогон страницы — четыре-шесть вызовов.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

# запуск и как модуль, и как файл
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.adapters.cloud_llm import PROVIDERS, knowledge_adapter, knowledge_model_label  # noqa: E402
from app.config import settings  # noqa: E402
from app.db import SessionLocal  # noqa: E402
from app.models import Entity, Fact, FactSource, KnowledgeSource  # noqa: E402
from app.services.extraction_pipeline import ExtractionPipeline  # noqa: E402
from app.services.knowledge_source_service import KnowledgeSourceService  # noqa: E402

#: поля факта, которые стоит видеть при сравнении моделей
FACT_FIELDS = (
    "subject_kind", "subject_name", "subject_relation", "subject_of", "subject_id",
    "predicate", "object_text", "time_text", "importance", "recurrence", "source_text",
)


class Recorder:
    """Обёртка адаптера: запоминает ответы вызова фактов и время вызовов."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.provider = inner.provider
        self.model = inner.model
        self.calls: list[tuple[str, float]] = []
        self.facts: list[dict] = []

    async def generate_structured(self, prompt: str, schema: dict) -> str:
        kind = next(iter(schema.get("properties", {})), "?")
        started = time.perf_counter()
        raw = await self.inner.generate_structured(prompt, schema)
        self.calls.append((kind, time.perf_counter() - started))
        if kind == "facts":
            try:
                self.facts = json.loads(raw).get("facts", [])
            except (json.JSONDecodeError, AttributeError):
                self.facts = []
        return raw


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source_type", choices=("htr", "diary"))
    parser.add_argument("ref_id", type=int, help="id страницы рукописи или записи дневника")
    parser.add_argument("--provider", choices=PROVIDERS, help="по умолчанию — KNOWLEDGE_LLM_PROVIDER")
    parser.add_argument("--model", help="по умолчанию — модель провайдера")
    parser.add_argument("--runs", type=int, default=1, help="сколько раз прогнать (стабильность ответа)")
    args = parser.parse_args()

    if args.provider:
        settings.knowledge_llm_provider = args.provider
        settings.knowledge_llm_model = args.model or ""
    elif args.model:
        settings.knowledge_llm_model = args.model

    db = SessionLocal()
    try:
        source = (
            db.query(KnowledgeSource)
            .filter(KnowledgeSource.source_type == args.source_type, KnowledgeSource.ref_id == args.ref_id)
            .first()
        )
        if source is None:
            print(f"Источник {args.source_type}/{args.ref_id} не найден", file=sys.stderr)
            return 1
        print(f"Источник {source.id} ({args.source_type}/{args.ref_id}), {len(source.text or '')} символов; "
              f"модель {knowledge_model_label()}")
        for attempt in range(1, args.runs + 1):
            adapter = Recorder(knowledge_adapter())
            service = KnowledgeSourceService(db)
            # как настоящее переизвлечение: сначала знание страницы убирается
            # (в той же транзакции, откатится вместе с прогоном), иначе к новым
            # фактам примешаются старые
            service.reset_source_knowledge(source)
            pipeline = ExtractionPipeline(db, source, adapter=adapter, service=service)
            started = time.perf_counter()

            async def run_once():
                try:
                    return await pipeline.run()
                finally:
                    if hasattr(adapter.inner, "aclose"):
                        await adapter.inner.aclose()

            stats = asyncio.run(run_once())
            elapsed = time.perf_counter() - started
            db.flush()
            # кому на самом деле приписан каждый факт этой страницы (до отката)
            written = (
                db.query(Fact, Entity)
                .outerjoin(Entity, Entity.id == Fact.subject_entity_id)
                .join(FactSource, FactSource.fact_id == Fact.id)
                .filter(FactSource.source_id == source.id)
                .order_by(Fact.start_offset)
                .all()
            )
            db.rollback()  # пробный прогон: ничего не остаётся в базе

            print(f"\n=== прогон {attempt}: {elapsed:.0f} с; вызовы: "
                  + ", ".join(f"{kind} {seconds:.0f} с" for kind, seconds in adapter.calls))
            print(f"--- модель вернула фактов: {len(adapter.facts)}")
            for item in adapter.facts:
                print("  " + json.dumps({key: item.get(key) for key in FACT_FIELDS if item.get(key) not in (None, "", 0)}, ensure_ascii=False))
            print(f"--- записалось бы (субъект → факт): {len(written)}")
            for fact, subject in written:
                who = f"{subject.canonical_name} [{subject.id}]" if subject else "—"
                print(f"  {who} → {fact.predicate} {fact.object_text or ''}".rstrip())
            summary = {key: value for key, value in stats.items() if key.startswith(("facts_", "relations_", "goals_", "relative_")) or key == "errors"}
            print("--- итог записи (откатан): " + json.dumps(summary, ensure_ascii=False))
    finally:
        db.rollback()
        db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
