"""Measure how often the dictionary check would raise a false alarm.

The check marks a word the user should look at, so every word that is *missing*
from the dictionary is a potential false alarm: the user sees a violet box on a
word that was actually recognized correctly. This script measures exactly that
rate on the pages the user already confirmed (their own ground truth) — the
only unbiased material available.

Usage (from the ``backend`` directory)::

    .venv/bin/python scripts/calibrate_htr_lexicon.py --author-id 1
    .venv/bin/python scripts/calibrate_htr_lexicon.py --author-id 1 --top 60

Decision rule used by the report: an OOV rate around or below ~2-3 % of all
words is workable (the user gets a handful of violet words per page), while a
much higher rate means the dictionary does not fit this corpus — add the
author's own words (``build_htr_lexicon.py --include-database``), another word
list, or lower expectations.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402
from app.db import SessionLocal  # noqa: E402
from app.htr.domain.text import iter_words, word_variants  # noqa: E402
from app.htr.infrastructure.lexicon import (  # noqa: E402
    LayeredLexiconChecker,
    SqlAlchemyAuthorCorpus,
    load_file_lexicon,
)

TARGET_OOV_RATE = 0.03


def _word_set(lines: list[str]) -> frozenset[str]:
    """Dictionary forms a page contributes to the author's own vocabulary."""
    forms: set[str] = set()
    for line in lines:
        for token in iter_words(line):
            forms.update(word_variants(token))
    return frozenset(forms)


def percentile(values: list[int], share: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(share * (len(ordered) - 1))))
    return float(ordered[index])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-id", type=int, required=True)
    parser.add_argument("--top", type=int, default=40, help="how many OOV words to print")
    parser.add_argument("--lexicon", default=settings.htr_lexicon_path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    base = load_file_lexicon(args.lexicon)
    if not base.is_available:
        print(
            f"no dictionary at {args.lexicon} — run scripts/build_htr_lexicon.py first",
            file=sys.stderr,
        )
        return 2

    db = SessionLocal()
    try:
        corpus = SqlAlchemyAuthorCorpus(db)
        texts = corpus.confirmed_texts(args.author_id)
        # The page being measured must not contribute to the vocabulary it is
        # checked against, otherwise every word of it is trivially "known" and
        # the report is always zero. Words that also occur on another page stay
        # known — that is exactly what a newly confirmed page would see.
        author_words = corpus.known_words(args.author_id)
        page_words = {page_id: _word_set(lines) for page_id, lines in texts.items()}
    finally:
        db.close()

    everything = frozenset().union(*page_words.values()) if page_words else frozenset()
    vocabulary_only = author_words - everything

    if not texts:
        print(
            f"author {args.author_id} has no confirmed pages; confirm a page first "
            "(the calibration needs the user's own ground truth)"
        )
        return 1

    print(
        f"dictionary: {base.path} ({base.word_count:,} word forms), "
        f"author vocabulary: {len(author_words):,} forms of which "
        f"{len(vocabulary_only):,} come from the knowledge base "
        f"({len(texts)} pages, each measured without its own words)"
    )
    print(f"confirmed pages: {len(texts)}\n")
    print(f"{'page':>6}  {'words':>6}  {'oov':>5}  {'rate':>7}")
    print("-" * 30)

    unknown: Counter[str] = Counter()
    rates: list[int] = []
    total_words = 0
    total_oov = 0
    for page_id in sorted(texts):
        others = frozenset().union(
            *(words for other_id, words in page_words.items() if other_id != page_id)
        ) if len(page_words) > 1 else frozenset()
        checker = LayeredLexiconChecker(base=base, extra=others | vocabulary_only)
        words = [token for text in texts[page_id] for token in iter_words(text)]
        oov = [token for token in words if not checker.is_known(token)]
        unknown.update(oov)
        total_words += len(words)
        total_oov += len(oov)
        page_rate = (len(oov) / len(words)) if words else 0.0
        rates.append(round(page_rate * 100))
        print(f"{page_id:>6}  {len(words):>6}  {len(oov):>5}  {page_rate * 100:>6.1f}%")

    overall = (total_oov / total_words) if total_words else 0.0
    print("-" * 30)
    print(f"{'всего':>6}  {total_words:>6}  {total_oov:>5}  {overall * 100:>6.1f}%")
    print(
        f"\nper-page OOV rate: median {percentile(rates, 0.5):.0f}%, "
        f"p90 {percentile(rates, 0.9):.0f}%, max {max(rates) if rates else 0}%"
    )

    if unknown:
        print(f"\ntop {args.top} words missing from the dictionary:")
        for word, count in unknown.most_common(args.top):
            print(f"  {count:>4}  {word}")

    print()
    if overall <= TARGET_OOV_RATE:
        print(
            f"OK: {overall * 100:.1f}% of the confirmed words are unknown "
            f"(target <= {TARGET_OOV_RATE * 100:.0f}%)"
        )
        return 0
    print(
        f"TOO NOISY: {overall * 100:.1f}% of the confirmed words are unknown "
        f"(target <= {TARGET_OOV_RATE * 100:.0f}%).\n"
        "Most of them are probably names, dialect words or old spellings: rebuild the "
        "artifact with --include-database, or add a broader word list, otherwise the "
        "violet marks will mostly be false alarms."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
