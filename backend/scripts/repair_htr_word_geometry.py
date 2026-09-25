"""Fill in word geometry that an earlier version dropped, without touching text.

A line whose beam text and kraken text had a different number of words used to
lose **all** word boxes: the transcription was right, the page simply showed no
word markup (see the page-26 case: ``где -то`` merged into ``где-то``). The fix
aligns the two tokenizations, but the geometry of pages recognized *before* it
is missing and is not recomputed on read — the boxes come from the CTC matrix,
which is not stored.

This script repairs those rows. It re-runs recognition on the page image to get
the cuts, maps them onto the **stored** text (so nothing the user corrected is
touched) and inserts the missing ``htr_words`` rows. Only lines that currently
have no words are affected; texts, corrections, statuses and suggestions are
left alone.

Usage (from the ``backend`` directory)::

    # see what would be filled in
    .venv/bin/python scripts/repair_htr_word_geometry.py --page-id 26 --dry-run

    # do it
    .venv/bin/python scripts/repair_htr_word_geometry.py --page-id 26
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db import SessionLocal  # noqa: E402
from app.htr.infrastructure.kraken.recognizer import (  # noqa: E402
    KrakenRecognizer,
    words_from_record,
)

logger = logging.getLogger("repair_word_geometry")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page-id", type=int, required=True)
    parser.add_argument(
        "--lines",
        type=int,
        nargs="*",
        default=None,
        help="order_index строк; по умолчанию все строки без разметки",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--min-text-ratio",
        type=float,
        default=0.5,
        help="минимальное сходство текста при сопоставлении записи со строкой",
    )
    parser.add_argument(
        "--max-center-distance",
        type=float,
        default=60.0,
        help="насколько bbox записи может отличаться от bbox строки, px",
    )
    return parser.parse_args(argv)


def _beam_text(recognizer, net, record) -> str:
    """The decoder's reading of a record, used for text-based matching."""
    try:
        matrix = recognizer._logits_matrix(getattr(record, "logits", None))
        if matrix is None:
            return ""
        decoder = recognizer._build_decoder(net, None, None)
        if decoder is None:
            return ""
        return decoder.decode_nbest(matrix, n=1)[0].text
    except Exception:  # pragma: no cover - matching must never break the repair
        return ""


def center(box) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = parse_args(argv)

    from app.htr.factory import build_recognizer
    from app.htr.infrastructure.storage import open_oriented_image
    from app.models import HTRLine, HTRPage, HTRWord

    db = SessionLocal()
    try:
        page = db.get(HTRPage, args.page_id)
        if page is None:
            print(f"страница {args.page_id} не найдена", file=sys.stderr)
            return 2
        lines = (
            db.query(HTRLine)
            .filter(HTRLine.page_id == page.id)
            .order_by(HTRLine.order_index)
            .all()
        )
        targets = []
        for line in lines:
            if args.lines is not None and line.order_index not in args.lines:
                continue
            text = (line.corrected_text or line.predicted_text or "").strip()
            if not text:
                continue
            if line.words:
                logger.info("строка %s уже размечена (%s слов) — пропускаю",
                            line.order_index, len(line.words))
                continue
            targets.append(line)
        if not targets:
            print("нечего чинить: у всех строк есть разметка")
            return 0
        print(f"страница {page.id}: строк без разметки — {len(targets)} "
              f"({[line.order_index for line in targets]})")

        recognizer = build_recognizer()
        model_path = Path(recognizer._lm_for(page.author_id) or "")
        active = None
        from app.htr.factory import build_default_model_ref
        from app.models import HTRModelVersion

        if page.recognition_model_version_id:
            active = db.get(HTRModelVersion, page.recognition_model_version_id)
        model_file = Path(active.file_path) if active else Path(build_default_model_ref().path)
        net, _config = recognizer._load_model(model_file)
        logger.info("модель для достройки: %s", model_file)

        image = open_oriented_image(page.file_path)
        try:
            segmentation = recognizer._merge_split_lines(recognizer._segment(image))
            records = list(net.predict(image, segmentation))
        finally:
            image.close()
        logger.info("распознано записей: %s", len(records))

        logger.info(
            "первая строка в БД bbox=%s | первая запись bbox=%s",
            (lines[0].x1, lines[0].y1, lines[0].x2, lines[0].y2),
            getattr(records[0], "bbox", None),
        )
        recognized = 0
        used: set[int] = set()
        for line in targets:
            text = (line.corrected_text or line.predicted_text or "").strip()
            line_box = (line.x1, line.y1, line.x2, line.y2)
            best = None
            best_position = -1
            best_distance = args.max_center_distance
            for position, record in enumerate(records):
                if position in used:
                    continue
                box = getattr(record, "bbox", None)
                if box is None or len(box) != 4:
                    continue
                distance = sum(
                    (a - b) ** 2 for a, b in zip(center(box), center(line_box))
                ) ** 0.5
                if distance < best_distance:
                    best, best_distance, best_position = record, distance, position
            match = f"по геометрии ({best_distance:.0f} px)"
            if best is None:
                # the page may have been segmented differently (segments merged
                # or split), so fall back to the text: the same handwriting must
                # produce a similar transcription
                from difflib import SequenceMatcher

                best_ratio = float(args.min_text_ratio)
                for position, record in enumerate(records):
                    if position in used:
                        continue
                    candidate = _beam_text(recognizer, net, record) or getattr(
                        record, "prediction", ""
                    ) or ""
                    ratio = SequenceMatcher(None, candidate, text).ratio()
                    if ratio > best_ratio:
                        best, best_ratio, best_position = record, ratio, position
                match = f"по тексту (совпадение {best_ratio:.2f})"
                if best is None:
                    logger.warning(
                        "строка %s: ни по геометрии, ни по тексту запись не найдена",
                        line.order_index,
                    )
                    continue
            used.add(best_position)
            logger.info("строка %s -> запись %s (%s)", line.order_index, best_position, match)
            # the *stored* text is what the user sees, so geometry is mapped to it
            words = words_from_record(best, str(line.id), text=text)
            if not words:
                logger.warning("строка %s: геометрию построить не удалось", line.order_index)
                continue
            logger.info("строка %s: %s слов (%s)", line.order_index, len(words), match)
            if args.dry_run:
                for word in words[:3]:
                    logger.info("   %r bbox=%s", word.text, word.bbox)
                recognized += 1
                continue
            for order, word in enumerate(words):
                db.add(
                    HTRWord(
                        line_id=line.id,
                        order_index=order,
                        x1=word.bbox.x1,
                        y1=word.bbox.y1,
                        x2=word.bbox.x2,
                        y2=word.bbox.y2,
                        predicted_text=word.text,
                        confidence=word.confidence,
                        polygon=json.dumps([[int(x), int(y)] for x, y in (word.polygon or [])])
                        if word.polygon
                        else None,
                    )
                )
            recognized += 1
        if args.dry_run:
            print(f"dry-run: разметка построена для {recognized} строк, в БД ничего не записано")
        else:
            db.commit()
            print(f"готово: разметка добавлена для {recognized} строк")
        return 0
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
