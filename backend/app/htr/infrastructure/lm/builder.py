"""Assemble the character-LM corpus and build the model.

The artifact the recognizer loads (``author_<id>_char_lm.npz``) has to follow the
ground-truth corpus: a page returned to editing must disappear from it, a newly
confirmed one must appear. That refresh happens at **training time** (see
``HandwritingTrainingService``), not on every corpus change — building the model
means reading tens of millions of characters and takes tens of seconds.

The three sources are the same as in ``scripts/build_htr_lm.py``, which keeps the
extra flags needed for measurements (``--extra-text``, ``--exclude-page``,
``--per-author``):

* the author's own confirmed pages, repeated ``author_weight`` times — this is
  the only source that teaches word boundaries;
* running prose from ``htr_storage/lm/texts/*.txt`` (the Leipzig corpus), capped
  by ``char_budget``;
* the word-form lists downloaded for the dictionary (``*russian*.txt``), one form
  per line and deliberately without invented spaces.

Everything is normalized to **NFD**, because that is what the model's codec emits
(``й`` arrives as ``и`` + U+0306).
"""
from __future__ import annotations

import logging
import unicodedata
from pathlib import Path
from typing import Iterable, Sequence

from ....config import settings
from .char_ngram import CharNGram

logger = logging.getLogger(__name__)

#: how many times the author's own text enters the counts
DEFAULT_AUTHOR_WEIGHT = 30
#: stop adding general prose after this many characters
DEFAULT_CHAR_BUDGET = 40_000_000


def normalize(text: str) -> str:
    return unicodedata.normalize("NFD", text)


def word_list_paths(storage_dir: str | Path | None = None) -> list[Path]:
    """The word-form files downloaded for the general dictionary."""
    sources = Path(storage_dir or settings.htr_storage_dir) / "lexicon" / "downloads"
    if not sources.is_dir():
        return []
    return sorted(sources.glob("*russian*.txt"))


def read_text(path: Path) -> str:
    """Read a corpus file, dropping the Project Gutenberg licence boilerplate."""
    import bz2
    import gzip

    if path.suffix == ".bz2":
        with bz2.open(path, "rt", encoding="utf-8", errors="replace") as handle:
            text = handle.read()
    elif path.suffix == ".gz":
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


def prose_texts(
    directory: str | Path | None = None,
    extra_paths: Iterable[str | Path] = (),
    char_budget: int = DEFAULT_CHAR_BUDGET,
) -> list[str]:
    """Running Russian prose, one sentence per line, up to ``char_budget``."""
    files: list[Path] = [Path(item).expanduser() for item in extra_paths]
    folder = Path(directory or Path(settings.htr_storage_dir) / "lm" / "texts")
    if folder.is_dir():
        files.extend(sorted(folder.glob("*.txt")))

    parts: list[str] = []
    used = 0
    for path in files:
        if not path.is_file():
            logger.warning("HTR language model: %s is missing", path)
            continue
        text = normalize(read_text(path))
        if not text.strip():
            continue
        parts.append(text)
        used += len(text)
        logger.info("HTR language model: %s -> %s characters", path.name, len(text))
        if used >= char_budget:
            logger.info("HTR language model: character budget %s reached", char_budget)
            break
    return parts


def word_form_texts(paths: Sequence[Path]) -> list[str]:
    """The dictionary's word forms, one per line, without invented spaces."""
    parts: list[str] = []
    for path in paths:
        block: list[str] = []
        for raw in path.open("rt", encoding="cp1251", errors="replace"):
            word = raw.strip()
            if word and not word.startswith("#"):
                block.append(word)
        if block:
            parts.append(normalize("\n".join(block)))
            logger.info("HTR language model: %s word forms from %s", len(block), path.name)
    return parts


def author_lines(db, author_id: int, exclude_pages: Iterable[int] = ()) -> list[str]:
    """Effective line texts of the author's confirmed pages."""
    from ..lexicon import SqlAlchemyAuthorCorpus

    excluded = set(exclude_pages)
    lines: list[str] = []
    for page_id, texts in SqlAlchemyAuthorCorpus(db).confirmed_texts(author_id).items():
        if page_id in excluded:
            continue
        lines.extend(texts)
    return lines


def build_author_lm(
    db,
    author_id: int,
    *,
    order: int = 6,
    min_count: int = 3,
    backoff: float = 0.4,
    author_weight: int = DEFAULT_AUTHOR_WEIGHT,
    char_budget: int = DEFAULT_CHAR_BUDGET,
    storage_dir: str | Path | None = None,
) -> CharNGram:
    """Build the character n-gram model of one author from the current corpus."""
    lines = author_lines(db, author_id)
    author_parts = ["\n".join(lines)] * max(1, author_weight) if lines else []
    prose = prose_texts(
        directory=(
            Path(storage_dir or settings.htr_storage_dir) / "lm" / "texts"
        ),
        char_budget=char_budget,
    )
    words = word_form_texts(word_list_paths(storage_dir))
    texts = author_parts + prose + words
    if not texts:
        raise ValueError("no corpus sources for the language model")

    author_chars = sum(len(part) for part in author_parts) // max(1, author_weight)
    model = CharNGram.build(
        texts,
        order=order,
        min_count=min_count,
        backoff=backoff,
        meta={
            "author_id": author_id,
            "author_weight": author_weight,
            "author_chars": author_chars,
            "running_text_chars": author_chars + sum(len(part) for part in prose),
            "word_form_chars": sum(len(part) for part in words),
            "total_chars": sum(len(part) for part in texts),
        },
    )
    logger.info(
        "HTR language model built: author_id=%s lines=%s running_text=%s characters",
        author_id, len(lines), model.meta["running_text_chars"],
    )
    return model


def rebuild_author_char_lm(
    db,
    author_id: int,
    *,
    storage_dir: str | Path | None = None,
    **kwargs,
) -> Path | None:
    """Rebuild and save ``author_<id>_char_lm.npz`` where the recognizer looks.

    Returns ``None`` when the author has no confirmed text yet: there is nothing
    author-specific to save, and the recognizer falls back to the general model.
    """
    if not author_lines(db, author_id):
        logger.info(
            "HTR language model: author_id=%s has no confirmed text yet, keeping the "
            "general model", author_id,
        )
        return None
    model = build_author_lm(db, author_id, storage_dir=storage_dir, **kwargs)
    root = Path(storage_dir or settings.htr_storage_dir) / "lm"
    root.mkdir(parents=True, exist_ok=True)
    return Path(model.save(root / f"author_{author_id}_char_lm.npz"))
