#!/usr/bin/env python
"""Регрессия распознавания Kraken на реальных страницах автора.

Зачем: тесты HTR подменяют модель и сегментацию фейками, поэтому обновление
torch/transformers/safetensors или словаря автора не ловится юнит-тестами. Этот
скрипт берёт страницы, уже распознанные конкретной версией модели, прогоняет их
той же моделью заново и сравнивает результат с сохранённым ``predicted_text``.

БД только читается: recognizer вызывается напрямую, ``recognize_page`` (которая
перезаписала бы предсказание) не используется.

Примеры:

    # проверить конкретную страницу её собственной моделью
    python scripts/verify_htr_recognition.py --page-id 24

    # проверить все подтверждённые страницы (медленно на CPU)
    python scripts/verify_htr_recognition.py --all --max-cer 0.05

Расхождение ожидаемо там, где с момента распознавания вырос словарь автора
(подтверждённые страницы, знания) или был пересобран файл лексикона: beam-декодер
намеренно учитывает текущий словарь, поэтому часть слов читается иначе — обычно
лучше. Скрипт печатает такие строки, чтобы это было видно, а не принималось на
веру.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db import SessionLocal  # noqa: E402
from app.htr import factory  # noqa: E402
from app.htr.application.metrics import levenshtein  # noqa: E402
from app.models import HTRLine, HTRPage  # noqa: E402


def _versions() -> str:
    import importlib.metadata as md

    def version(package: str) -> str:
        try:
            return md.version(package)
        except Exception:
            return "?"

    try:
        import torch

        cuda = torch.cuda.is_available()
    except Exception:
        cuda = False
    return (
        f"kraken={version('kraken')} transformers={version('transformers')} "
        f"safetensors={version('safetensors')} torch={version('torch')} cuda={cuda}"
    )


def _model_path_for(db, page: HTRPage) -> str | None:
    """The model the page was recognized with, or the default one."""
    from app.config import settings
    from app.models import HTRModelVersion

    if page.recognition_model_version_id:
        version = (
            db.query(HTRModelVersion)
            .filter(HTRModelVersion.id == page.recognition_model_version_id)
            .first()
        )
        if version is not None and Path(version.file_path).is_file():
            return version.file_path
    default = factory.build_default_model_ref()
    return default.path if Path(default.path).is_file() else settings.htr_default_model_path


def _verify(db, service, page: HTRPage, model_path: str, show_diffs: int) -> tuple[int, int, int, float, float]:
    stored = (
        db.query(HTRLine)
        .filter(HTRLine.page_id == page.id)
        .order_by(HTRLine.order_index)
        .all()
    )
    checker = (
        service.lexicon_annotator.checker_for(page.author_id)
        if service.lexicon_annotator
        else None
    )
    started = time.time()
    result = service.recognizer.recognize(
        page.file_path, model_path, author_id=page.author_id, word_checker=checker
    )
    elapsed = time.time() - started

    pairs = [
        ((line.predicted_text or ""), (new.text or ""))
        for line, new in zip(stored, result.lines)
    ]
    identical = sum(1 for old, new in pairs if old == new)
    ref_chars = sum(len(old) for old, _ in pairs) or 1
    ref_words = sum(len(old.split()) for old, _ in pairs) or 1
    cer = sum(levenshtein(old, new) for old, new in pairs) / ref_chars
    wer = sum(levenshtein(old.split(), new.split()) for old, new in pairs) / ref_words

    print(
        f"page {page.id}: {len(result.lines)} строк за {elapsed:.1f}s, "
        f"совпало {identical}/{len(pairs)}, CER={cer:.4f}, WER={wer:.4f}"
    )
    if len(stored) != len(result.lines):
        print(f"  !! число строк изменилось: было {len(stored)}, стало {len(result.lines)}")
    shown = 0
    for line, new in zip(stored, result.lines):
        old_text = (line.predicted_text or "").strip()
        new_text = (new.text or "").strip()
        if old_text != new_text and shown < show_diffs:
            shown += 1
            print(f"  строка {line.order_index}:")
            print(f"    было:  {old_text[:100]}")
            print(f"    стало: {new_text[:100]}")
    return identical, len(pairs), len(stored) - len(result.lines), cer, wer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--page-id", type=int, action="append", default=[], help="страница (можно несколько раз)")
    parser.add_argument("--all", action="store_true", help="все подтверждённые страницы с предсказанием")
    parser.add_argument("--model", help="путь к модели вместо той, которой страница распознавалась")
    parser.add_argument("--max-cer", type=float, default=0.05, help="порог провала (по умолчанию 0.05)")
    parser.add_argument("--show-diffs", type=int, default=3, help="сколько расхождений печатать на страницу")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    print(_versions())

    db = SessionLocal()
    service = factory.build_page_service(db)

    query = db.query(HTRPage).filter(HTRPage.status == "CONFIRMED")
    if args.page_id:
        query = query.filter(HTRPage.id.in_(args.page_id))
    elif not args.all:
        print("укажите --page-id или --all")
        return 2
    pages = query.order_by(HTRPage.id).all()
    if not pages:
        print("страниц не найдено")
        return 2

    worst_cer = 0.0
    totals = [0, 0]
    failed = False
    for page in pages:
        if not (db.query(HTRLine).filter(HTRLine.page_id == page.id).count()):
            print(f"page {page.id}: строк нет, пропуск")
            continue
        model_path = args.model or _model_path_for(db, page)
        if not model_path or not Path(model_path).is_file():
            print(f"page {page.id}: модель не найдена ({model_path}), пропуск")
            continue
        print(f"page {page.id}: модель {model_path}")
        identical, total, line_delta, cer, _ = _verify(db, service, page, model_path, args.show_diffs)
        totals[0] += identical
        totals[1] += total
        worst_cer = max(worst_cer, cer)
        if cer > args.max_cer or line_delta:
            failed = True

    db.close()
    if totals[1]:
        print(f"\nитого: совпало {totals[0]}/{totals[1]}, худший CER={worst_cer:.4f}, порог {args.max_cer}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
