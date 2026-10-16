"""Пересборка графа связей и знаний пользователя с нуля.

Граф, накопленный старыми прогонами, подкрепляет собственные ошибки: близкие
автора уходят в промпт подсказкой с id, и модель приписывает новые факты тем,
кто записан неверно. Этот скрипт:

1. очищает граф пользователя (``entity_relations`` вместе с подтверждениями
   ``relation_sources``) и описания персон — они дописываются только в пустое
   поле, поэтому ошибочное описание («дочь Виталия» у мамы автора) иначе
   осталось бы навсегда;
2. переизвлекает источники по очереди тем же путём, что и кнопка «Повторить»
   (``KnowledgeSourceService.run_now``: очистка знаний страницы + конвейер):
   сначала подтверждённые страницы рукописей по порядку, потом дневник.
   Страницы рукописей, снятые с подтверждения, пропускаются — их текст не
   окончателен.

Модель — из настроек (``KNOWLEDGE_LLM_PROVIDER``) или ``--provider``.

Запуск (из каталога ``backend``)::

    python -m app.scripts.rebuild_knowledge --user musalser --dry-run   # что будет сделано
    python -m app.scripts.rebuild_knowledge --user musalser             # очистка + переизвлечение

Перед запуском снимите дамп (см. README, «Дампы и восстановление»).
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.adapters.cloud_llm import PROVIDERS, knowledge_model_label  # noqa: E402
from app.config import settings  # noqa: E402
from app.db import SessionLocal  # noqa: E402
from app.models import (  # noqa: E402
    DiaryPage,
    Entity,
    EntityRelation,
    HTRPage,
    KnowledgeSource,
    RelationSource,
    User,
)
from app.services.knowledge_source_service import KnowledgeSourceService  # noqa: E402


def ordered_sources(db, user: User) -> tuple[list[KnowledgeSource], list[KnowledgeSource]]:
    """(что извлекать, что пропустить): рукописи по порядку страниц, затем дневник."""
    htr = (
        db.query(KnowledgeSource, HTRPage)
        .join(HTRPage, HTRPage.id == KnowledgeSource.ref_id)
        .filter(KnowledgeSource.user_id == user.id, KnowledgeSource.source_type == "htr")
        .order_by(HTRPage.author_id, HTRPage.order_index, HTRPage.id)
        .all()
    )
    diary = (
        db.query(KnowledgeSource, DiaryPage)
        .join(DiaryPage, DiaryPage.id == KnowledgeSource.ref_id)
        .filter(KnowledgeSource.user_id == user.id, KnowledgeSource.source_type == "diary")
        .order_by(DiaryPage.created_at, DiaryPage.id)
        .all()
    )
    run = [source for source, page in htr if page.status == "CONFIRMED"]
    skipped = [source for source, page in htr if page.status != "CONFIRMED"]
    run += [source for source, _ in diary if (source.text or "").strip()]
    skipped += [source for source, _ in diary if not (source.text or "").strip()]
    return run, skipped


def clear_graph(db, user: User) -> dict[str, int]:
    """Граф и описания персон пользователя; коммит — здесь же."""
    relation_ids = [row.id for row in db.query(EntityRelation.id).filter(EntityRelation.user_id == user.id)]
    removed = {
        "relation_sources": db.query(RelationSource)
        .filter(RelationSource.relation_id.in_(relation_ids))
        .delete(synchronize_session=False)
        if relation_ids
        else 0,
        "relations": db.query(EntityRelation)
        .filter(EntityRelation.user_id == user.id)
        .delete(synchronize_session=False),
        "descriptions": db.query(Entity)
        .filter(Entity.user_id == user.id, Entity.entity_type == "person", Entity.description.isnot(None))
        .update({Entity.description: None}, synchronize_session=False),
    }
    db.commit()
    return removed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--user", required=True, help="username")
    parser.add_argument("--provider", choices=PROVIDERS, help="по умолчанию — KNOWLEDGE_LLM_PROVIDER")
    parser.add_argument("--model", help="по умолчанию — модель провайдера")
    parser.add_argument("--dry-run", action="store_true", help="только показать план")
    args = parser.parse_args()
    if args.provider:
        settings.knowledge_llm_provider = args.provider
        settings.knowledge_llm_model = args.model or ""
    elif args.model:
        settings.knowledge_llm_model = args.model

    db = SessionLocal()
    try:
        user = db.query(User).filter(User.username == args.user).first()
        if user is None:
            print(f"Пользователь {args.user} не найден", file=sys.stderr)
            return 1
        run, skipped = ordered_sources(db, user)
        relations = db.query(EntityRelation).filter(EntityRelation.user_id == user.id).count()
        print(f"Модель: {knowledge_model_label()}")
        print(f"Граф: {relations} рёбер будет удалено; источников к извлечению: {len(run)}, пропущено: {len(skipped)}")
        for source in skipped:
            print(f"  пропуск: {source.source_type}/{source.ref_id} «{source.title or ''}»")
        if args.dry_run:
            return 0

        print("Очистка:", clear_graph(db, user), flush=True)
        service = KnowledgeSourceService(db)
        failed = 0
        for number, source in enumerate(run, start=1):
            started = time.perf_counter()
            result = asyncio.run(service.run_now(source))
            summary = result.get("summary") or {}
            line = (
                f"[{number}/{len(run)}] {source.source_type}/{source.ref_id}: {result.get('status')} "
                f"за {time.perf_counter() - started:.0f} с; фактов {summary.get('facts_created', 0)}, "
                f"рёбер {summary.get('relations_created', 0)}"
            )
            if summary.get("errors") or result.get("error"):
                line += f"; ошибки: {summary.get('errors') or result.get('error')}"
            if result.get("status") != "done":
                failed += 1
            print(line, flush=True)
        relations = db.query(EntityRelation).filter(EntityRelation.user_id == user.id).count()
        print(f"Готово: {len(run) - failed} из {len(run)} источников, рёбер в графе: {relations}")
        return 0 if not failed else 2
    finally:
        db.close()


if __name__ == "__main__":
    sys.exit(main())
