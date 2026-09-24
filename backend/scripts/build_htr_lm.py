"""Build the character n-gram language model used by the beam-search decoder.

Three sources, all of them optional but together they cover the three failure
modes of the recognizer:

* **the author's own confirmed pages** — their vocabulary, dialect and names,
  weighted up so it dominates the general prior (``--author-weight``);
* **running Russian prose** (``--extra-text FILE``, repeatable) — letter
  sequences *across* word boundaries. Project Gutenberg carries almost no
  Russian text and the usual dumps (Tatoeba, OPUS) are throttled from this
  machine, so a large running-text corpus has to be supplied explicitly;
* **the word-form lists** already used by the dictionary (1.5 M forms plus
  877 k surnames) — exactly the within-word patterns a CTC model gets wrong,
  including the surnames and toponyms that no dialect dictionary contains.

Everything is normalized to **NFD**, because that is what the codec of the
default model emits (``й`` arrives as ``и`` + U+0306), so the language model
sees the same characters as the acoustic model.

The artifact records how much *running* text went into it
(``meta["running_text_chars"]``), because bare word forms have millions of
characters and still cannot teach what follows a space; the recognizer refuses
to run beam search below ``htr_lm_min_text_chars`` of it.

Usage (from the ``backend`` directory)::

    # per-author artifacts (author_<id>_char_lm.npz) — what production uses
    .venv/bin/python scripts/build_htr_lm.py --per-author
    # the general fallback for authors without confirmed pages
    .venv/bin/python scripts/build_htr_lm.py --no-author
    # one author, with a real corpus on top of the word forms
    .venv/bin/python scripts/build_htr_lm.py --author-id 1 --extra-text book.txt
    # leave-one-page-out (for honest measurement, not for production)
    .venv/bin/python scripts/build_htr_lm.py --exclude-page 20 --author-id 1
"""
from __future__ import annotations

import argparse
import bz2
import json
import logging
import sys
import unicodedata
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402
from app.htr.infrastructure.lm.char_ngram import CharNGram  # noqa: E402

logger = logging.getLogger("build_htr_lm")



def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(Path(settings.htr_storage_dir) / "lm" / "ru_char_lm.npz"),
    )
    parser.add_argument("--corpus-dir", default=None)
    parser.add_argument("--author-id", type=int, default=None)
    parser.add_argument("--no-author", action="store_true")
    parser.add_argument(
        "--exclude-page",
        action="append",
        type=int,
        default=[],
        help=(
            "не использовать подтверждённый текст этой страницы (для честной "
            "оценки leave-one-page-out; можно повторять)"
        ),
    )
    parser.add_argument(
        "--per-author",
        action="store_true",
        help=(
            "собрать по одной модели на автора (author_<id>_char_lm.npz) с его "
            "подтверждёнными страницами — то, что подхватывает распознаватель"
        ),
    )
    parser.add_argument("--output-dir", default=None, help="куда класть --per-author")
    parser.add_argument(
        "--author-weight",
        type=int,
        default=30,
        help="how many times the author's own text enters the counts",
    )
    parser.add_argument("--no-words", action="store_true")
    parser.add_argument(
        "--extra-text",
        action="append",
        default=[],
        help="файл с русским текстом (можно повторять): книги, корпус, дамп",
    )
    parser.add_argument("--order", type=int, default=6)
    parser.add_argument("--min-count", type=int, default=3)
    parser.add_argument("--backoff", type=float, default=0.4)
    parser.add_argument(
        "--char-budget",
        type=int,
        default=40_000_000,
        help="stop adding general text after this many characters",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# sources


def author_texts(
    author_id: int | None, exclude_pages: set[int] | None = None
) -> list[str]:
    from app.db import SessionLocal
    from app.htr.infrastructure.lexicon import SqlAlchemyAuthorCorpus
    from app.models import HTRAuthor

    exclude = exclude_pages or set()
    db = SessionLocal()
    try:
        query = db.query(HTRAuthor)
        if author_id is not None:
            query = query.filter(HTRAuthor.id == author_id)
        corpus = SqlAlchemyAuthorCorpus(db)
        lines: list[str] = []
        for author in query.all():
            for page_id, texts in corpus.confirmed_texts(author.id).items():
                if page_id in exclude:
                    continue
                lines.extend(texts)
        return lines
    finally:
        db.close()


def authors_with_text() -> list[int]:
    """Authors that have at least one confirmed page with text."""
    from app.db import SessionLocal
    from app.htr.infrastructure.lexicon import SqlAlchemyAuthorCorpus
    from app.models import HTRAuthor

    db = SessionLocal()
    try:
        corpus = SqlAlchemyAuthorCorpus(db)
        return [
            author.id
            for author in db.query(HTRAuthor).all()
            if corpus.confirmed_texts(author.id)
        ]
    finally:
        db.close()


def word_list_paths() -> list[Path]:
    """The word-form files already downloaded for the dictionary."""
    sources = Path(settings.htr_storage_dir) / "lexicon" / "downloads"
    if not sources.is_dir():
        return []
    return sorted(sources.glob("*russian*.txt"))


# ---------------------------------------------------------------------------


def normalize(text: str) -> str:
    return unicodedata.normalize("NFD", text)


def read_text(path: Path) -> str:
    """Read a text file, stripping the Project Gutenberg licence boilerplate."""
    if path.suffix == ".bz2":
        with bz2.open(path, "rt", encoding="utf-8", errors="replace") as handle:
            text = handle.read()
    elif path.suffix == ".gz":
        import gzip

        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
            text = handle.read()
    else:
        text = path.read_text(encoding="utf-8", errors="replace")
    start = text.find("*** START OF THE PROJECT GUTENBERG EBOOK")
    if start != -1:
        text = text[text.find("\n", start) + 1 :]
    end = text.find("*** END OF THE PROJECT GUTENBERG EBOOK")
    if end != -1:
        text = text[:end]
    return text


def general_texts(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    """Everything that does not depend on the author.

    Returns ``(prose, words)`` — real running text and bare word forms — so the
    metadata can tell the two apart: only running text teaches the model what
    follows a space.
    """
    parts: list[str] = []
    budget = args.char_budget
    directory = Path(args.corpus_dir or Path(settings.htr_storage_dir) / "lm" / "texts")
    extra = [Path(item).expanduser() for item in args.extra_text]
    if directory.is_dir():
        extra.extend(sorted(directory.glob("*.txt")))
    used = 0
    for path in extra:
        if not path.is_file():
            logger.warning("text: %s отсутствует", path)
            continue
        text = normalize(read_text(path))
        if not text.strip():
            continue
        parts.append(text)
        used += len(text)
        logger.info("text: %s -> %s символов", path.name, len(text))
        if used >= budget:
            logger.info("text: бюджет %s символов исчерпан", budget)
            break
    if not extra:
        logger.info("text: дополнительных текстов нет (словоформы + текст автора)")

    prose = list(parts)
    words_parts: list[str] = []
    if not args.no_words:
        for path in word_list_paths():
            words = 0
            block: list[str] = []
            for raw in path.open("rt", encoding="cp1251", errors="replace"):
                word = raw.strip()
                if not word or word.startswith("#"):
                    continue
                block.append(word)
                words += 1
            # One form per line, deliberately *without* invented spaces:
            # a word list knows letter sequences and word boundaries, and
            # nothing about what follows a space. Feeding it synthetic
            # sentences (measured) made 28 M characters of fake cross-word
            # statistics drown the 0.3 M characters of the author's real text
            # and turned a large win into a loss (WER 0.37 against greedy 0.27).
            # Spacing and punctuation must come from real text only.
            words_parts.append(normalize("\n".join(block)))
            logger.info("words: %s форм из %s", words, path.name)
    return prose, words_parts


def author_part(args: argparse.Namespace, author_id: int | None) -> list[str]:
    """The author's confirmed text, repeated ``--author-weight`` times."""
    lines = author_texts(author_id, set(args.exclude_page))
    if not lines:
        logger.warning("author text: нет подтверждённых страниц (author=%s)", author_id)
        return []
    block = "\n".join(lines)
    logger.info(
        "author text: %s строк, %s символов (author=%s, исключено страниц: %s)",
        len(lines), len(block), author_id, len(args.exclude_page),
    )
    return [block] * max(1, args.author_weight)


def collect_texts(args: argparse.Namespace) -> list[str]:
    parts = author_part(args, args.author_id) if not args.no_author else []
    prose, words = general_texts(args)
    parts.extend(prose)
    parts.extend(words)
    return parts


def build_model(
    args: argparse.Namespace,
    author_parts: list[str],
    prose: list[str],
    words: list[str],
) -> CharNGram:
    texts = list(author_parts) + list(prose) + list(words)
    author_chars = sum(len(part) for part in author_parts) // max(1, args.author_weight)
    running_chars = author_chars + sum(len(part) for part in prose)
    model = CharNGram.build(
        texts,
        order=args.order,
        min_count=args.min_count,
        backoff=args.backoff,
        meta={
            "author_weight": args.author_weight,
            "author_chars": author_chars,
            # only this teaches word boundaries; the recognizer refuses to run
            # beam search when it is too small to be trusted
            "running_text_chars": running_chars,
            "word_form_chars": sum(len(part) for part in words),
            "total_chars": sum(len(part) for part in texts),
        },
    )
    return model


def _report(model: CharNGram, path: Path, author_id: int | None) -> None:
    who = f"author={author_id}" if author_id is not None else "общая"
    print(
        f"готово ({who}): {path} ({path.stat().st_size / 1e6:.1f} MB на диске, "
        f"{model.size_bytes / 1e6:.1f} MB в памяти, алфавит {len(model.chars)}, "
        f"порядок {model.order}, текст {model.meta.get('running_text_chars', 0):,} "
        f"символов)"
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = parse_args(argv)

    prose, words = general_texts(args)
    total_general = sum(len(part) for part in prose + words)
    if not prose and not words:
        print("нет исходных текстов", file=sys.stderr)
        return 2

    if args.per_author:
        out_dir = Path(args.output_dir or Path(args.output).parent)
        if args.author_id is not None:
            author_ids = [args.author_id]
        else:
            author_ids = authors_with_text()
        if not author_ids:
            print("нет авторов с подтверждённым текстом", file=sys.stderr)
            return 2
        written = 0
        for author_id in author_ids:
            parts = author_part(args, author_id)
            if not parts:
                continue
            model = build_model(args, parts, prose, words)
            path = model.save(out_dir / f"author_{author_id}_char_lm.npz")
            _report(model, Path(path), author_id)
            written += 1
        return 0 if written else 2

    parts = author_part(args, args.author_id) if not args.no_author else []
    model = build_model(args, parts, prose, words)
    print(f"корпус: {len(prose) + len(words) + len(parts)} блоков, "
          f"{sum(len(t) for t in parts) + total_general:,} символов")
    path = model.save(args.output)
    _report(model, Path(path), args.author_id if not args.no_author else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
