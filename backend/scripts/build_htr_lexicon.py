"""Build the dictionary artifact used by the OOV (word-not-in-dictionary) check.

The backend only *reads* this artifact: it answers "is this recognized word a
known Russian word form?" so the UI can highlight the ones that are not. A
Bloom filter keeps 1.5-2.5 million word forms in ~2 MB instead of the couple of
hundred megabytes a Python set would need.

Usage (from the ``backend`` directory)::

    # default sources: the two MIT-licensed danakt/russian-words lists
    .venv/bin/python scripts/build_htr_lexicon.py

    # add the author's own confirmed words and knowledge-base terms
    .venv/bin/python scripts/build_htr_lexicon.py --include-database

    # a local word list instead of the download
    .venv/bin/python scripts/build_htr_lexicon.py --no-default-sources \
        --source /usr/share/hunspell/ru_RU.dict

Then calibrate the result on the pages the user already confirmed::

    .venv/bin/python scripts/calibrate_htr_lexicon.py --author-id 1
"""
from __future__ import annotations

import argparse
import bz2
import gzip
import hashlib
import sys
import time
import urllib.request
from pathlib import Path
from typing import Iterable, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402
from app.htr.domain.text import normalize_word  # noqa: E402
from app.htr.infrastructure.lexicon import BloomFilter  # noqa: E402

DEFAULT_SOURCES = (
    "https://raw.githubusercontent.com/danakt/russian-words/master/russian.txt",
    "https://raw.githubusercontent.com/danakt/russian-words/master/russian_surnames.txt",
)

MIN_WORD_LENGTH = 2


# ---------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------


def resolve_source(source: str, cache_dir: Path, encoding: str, refresh: bool) -> Path:
    """Return a local path for a source, downloading URLs into ``cache_dir``."""
    if not source.startswith(("http://", "https://")):
        path = Path(source).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"source not found: {path}")
        return path

    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(source.encode("utf-8")).hexdigest()[:10]
    target = cache_dir / f"{digest}-{Path(source).name}"
    if target.is_file() and not refresh:
        print(f"  cached: {target}")
        return target

    print(f"  downloading {source}")
    started = time.time()
    with urllib.request.urlopen(source, timeout=120) as response, target.open("wb") as handle:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            handle.write(chunk)
    print(f"  saved {target} ({target.stat().st_size / 1e6:.1f} MB, {time.time() - started:.1f}s)")
    return target


def detect_encoding(path: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    with path.open("rb") as handle:
        head = handle.read(1 << 16)
    try:
        head.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        return "cp1251"


def _open_text(path: Path, encoding: str):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding=encoding, errors="replace")
    if path.suffix == ".bz2":
        return bz2.open(path, "rt", encoding=encoding, errors="replace")
    return path.open("rt", encoding=encoding, errors="replace")


def iter_word_forms(path: Path, encoding: str) -> Iterator[str]:
    """Stream the dictionary forms of one file (deduplicated by the filter)."""
    with _open_text(path, encoding) as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # hunspell dictionaries carry an optional flag column
            for token in line.split("\t")[0].split("/")[0].split():
                word = normalize_word(token)
                if len(word) < MIN_WORD_LENGTH:
                    continue
                if not any(character.isalpha() for character in word):
                    continue
                if any(character.isdigit() for character in word):
                    continue
                yield word


def database_words(author_id: int | None) -> Iterable[str]:
    """The user's own confirmed transcriptions and knowledge-base terms."""
    from app.db import SessionLocal
    from app.htr.infrastructure.lexicon import SqlAlchemyAuthorCorpus
    from app.models import HTRAuthor

    db = SessionLocal()
    try:
        query = db.query(HTRAuthor)
        if author_id is not None:
            query = query.filter(HTRAuthor.id == author_id)
        authors = [author.id for author in query.all()]
        corpus = SqlAlchemyAuthorCorpus(db)
        words: set[str] = set()
        for identifier in authors:
            words.update(corpus.known_words(identifier))
        return words
    finally:
        db.close()


# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", default=[], help="URL or file (repeatable)")
    parser.add_argument("--no-default-sources", action="store_true")
    parser.add_argument("--output", default=settings.htr_lexicon_path)
    parser.add_argument("--cache-dir", default=None, help="where downloaded lists are kept")
    parser.add_argument("--encoding", default="auto", help="auto | utf-8 | cp1251 | ...")
    parser.add_argument("--false-positive-rate", type=float, default=0.01)
    parser.add_argument("--refresh", action="store_true", help="re-download cached sources")
    parser.add_argument(
        "--include-database",
        action="store_true",
        help="also add the author's confirmed words and knowledge-base terms",
    )
    parser.add_argument("--author-id", type=int, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    sources = list(args.source)
    if not args.no_default_sources:
        sources = list(DEFAULT_SOURCES) + sources
    if not sources and not args.include_database:
        print("nothing to do: no source and no --include-database", file=sys.stderr)
        return 2

    cache_dir = Path(args.cache_dir or Path(settings.htr_storage_dir) / "lexicon" / "downloads")
    print(f"lexicon sources ({len(sources)}):")
    files: list[tuple[Path, str]] = []
    for source in sources:
        path = resolve_source(source, cache_dir, args.encoding, args.refresh)
        encoding = detect_encoding(path, args.encoding)
        print(f"  {path} [{encoding}]")
        files.append((path, encoding))

    print("pass 1/2: counting word forms")
    started = time.time()
    capacity = 0
    for path, encoding in files:
        counted = sum(1 for _ in iter_word_forms(path, encoding))
        capacity += counted
        print(f"  {path.name}: {counted:,} forms")
    extra_words: list[str] = []
    if args.include_database:
        extra_words = sorted(database_words(args.author_id))
        capacity += len(extra_words)
        print(f"  database: {len(extra_words):,} forms")
    if capacity == 0:
        print("no word forms found", file=sys.stderr)
        return 1

    print(
        f"pass 2/2: filling the filter "
        f"(capacity {capacity:,}, false positive rate {args.false_positive_rate})"
    )
    bloom = BloomFilter.build(
        capacity=capacity,
        false_positive_rate=args.false_positive_rate,
        meta={
            "sources": [source for source in sources],
            "kind": "russian-word-forms",
        },
    )
    for path, encoding in files:
        for word in iter_word_forms(path, encoding):
            bloom.add(word)
    for word in extra_words:
        bloom.add(word)

    output = Path(args.output)
    bloom.save(output)
    size = output.stat().st_size
    print(
        f"done in {time.time() - started:.1f}s: {output} "
        f"({size / 1e6:.2f} MB, {bloom.hash_count} hash functions, "
        f"{len(bloom):,} insertions)"
    )
    print("now run: .venv/bin/python scripts/calibrate_htr_lexicon.py --author-id <id>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
