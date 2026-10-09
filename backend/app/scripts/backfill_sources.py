"""Создание и обновление источников знаний (knowledge_sources).

Идемпотентный backfill после миграции ``70d54e2ac8e3`` и страховка на будущее:
сервис источников проверяет каждую страницу дневника и каждую подтверждённую
страницу рукописи, пересобирает текст, карту строк и хэш. Если текст изменился,
источник помечается ``pending`` — знание из него устарело и его надо извлечь
заново.

Запуск (из каталога ``backend``)::

    python -m app.scripts.backfill_sources                 # все пользователи
    python -m app.scripts.backfill_sources --user musalser # один пользователь
    python -m app.scripts.backfill_sources --dry-run       # показать, что изменится
    python -m app.scripts.backfill_sources --extract       # + запуск конвейера

Сам конвейер извлечения появляется на этапе 4 (``app.services.extraction_pipeline``),
поэтому ``--extract`` пока честно сообщает, что запускать нечего, и выходит с
кодом 2.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# запуск и как модуль (`python -m app.scripts.backfill_sources`), и как файл
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.db import SessionLocal  # noqa: E402
from app.models import DiaryPage, HTRPage, KnowledgeSource, User  # noqa: E402
from app.services.knowledge_source_service import (  # noqa: E402
    STATUS_PENDING,
    KnowledgeSourceService,
)

logger = logging.getLogger("backfill_sources")


def backfill_user(
    db, service: KnowledgeSourceService, user: User, *, dry_run: bool
) -> dict[str, int]:
    """Refreshes every source of one account; returns counters for the report."""
    stats = {"diary_created": 0, "diary_updated": 0, "htr_created": 0, "htr_updated": 0, "unchanged": 0}

    for page in db.query(DiaryPage).filter(DiaryPage.user_id == user.id).order_by(DiaryPage.id).all():
        _, created, changed = service.refresh_from_diary_page(page, commit=False)
        stats["diary_created" if created else ("diary_updated" if changed else "unchanged")] += 1

    pages = (
        db.query(HTRPage)
        .filter(HTRPage.user_id == user.id, HTRPage.status == "CONFIRMED")
        .order_by(HTRPage.id)
        .all()
    )
    for page in pages:
        _, created, changed = service.refresh_from_htr_page(page, commit=False)
        stats["htr_created" if created else ("htr_updated" if changed else "unchanged")] += 1

    if dry_run:
        db.rollback()
    else:
        db.commit()
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--user", help="только этот username (по умолчанию все)")
    parser.add_argument("--dry-run", action="store_true", help="ничего не записывать, только показать")
    parser.add_argument("--extract", action="store_true", help="запустить извлечение знаний")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.extract:
        print(
            "Конвейер извлечения ещё не реализован (этап 4 ТЗ: "
            "app/services/extraction_pipeline.run_extraction). Источники созданы и "
            "помечены pending — запустить их извлечение можно будет после этапа 4."
        )
        return 2

    db = SessionLocal()
    try:
        query = db.query(User)
        if args.user:
            query = query.filter(User.username == args.user)
        users = query.order_by(User.id).all()
        if not users:
            print("Пользователи не найдены" + (f": {args.user}" if args.user else ""))
            return 2

        service = KnowledgeSourceService(db)
        totals: dict[str, int] = {}
        for user in users:
            stats = backfill_user(db, service, user, dry_run=args.dry_run)
            print(f"{user.username}: " + ", ".join(f"{key}={value}" for key, value in stats.items()))
            for key, value in stats.items():
                totals[key] = totals.get(key, 0) + value

        pending = (
            db.query(KnowledgeSource)
            .filter(KnowledgeSource.extraction_status == STATUS_PENDING)
            .count()
        )
        print(
            f"\n{'[dry-run] ' if args.dry_run else ''}итого: "
            + ", ".join(f"{key}={value}" for key, value in sorted(totals.items()))
            + f"; источников в статусе pending: {pending}"
        )
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
