"""Build the word-level KenLM model from a general corpus plus the author's text.

The character n-gram in :mod:`app.htr.infrastructure.lm.char_ngram` is what makes
the beam search affordable (≈52 000 lookups per line), but it only sees six
characters of context. This script builds a *word*-level model, where word order
and word choice are visible, and interpolates it with the author's own pages —
using KenLM's own tools rather than hand-rolled weighting:

1. the general corpus (a large Russian text dump, one sentence per line) and the
   author's confirmed pages become two plain text files;
2. ``lmplz`` estimates each of them **in intermediate format**
   (``--intermediate``), which is what ``interpolate`` consumes;
3. ``interpolate --just_tune`` tunes the two weights on the author's own text
   (a file with one sentence per line), then ``interpolate`` rebuilds the model
   with those weights;
4. ``build_binary`` packs the result into an mmap-able binary.

Everything runs on disk with a bounded amount of memory, so a several-hundred-
megabyte corpus is fine on a laptop.

The general corpus is expected as a text file or as a Leipzig corpora TSV
(``<id>\t<sentence>``); ``--general-corpus`` accepts either.

Usage (from the ``backend`` directory)::

    # the whole pipeline, Leipzig 1M sentences as the general corpus
    .venv/bin/python scripts/build_htr_word_lm.py \
        --general-corpus /tmp/leipzig/rus_news_2020_1M/rus_news_2020_1M-sentences.txt \
        --author-id 1 --kenlm-bin /tmp/kenlm-install/bin

    # see the exact commands without running them
    .venv/bin/python scripts/build_htr_word_lm.py --dry-run

    # re-interpolate only (the general intermediate is expensive to rebuild)
    .venv/bin/python scripts/build_htr_word_lm.py --reuse-intermediate
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402

logger = logging.getLogger("build_htr_word_lm")

DEFAULT_KENLM_BIN = "/tmp/kenlm-install/bin"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--general-corpus",
        default=None,
        help="большой русский корпус: обычный текст (по предложению в строке) или TSV Leipzig",
    )
    parser.add_argument(
        "--general-text",
        default=None,
        help="готовый текстовый файл корпуса (если уже извлечён из TSV)",
    )
    parser.add_argument("--author-id", type=int, default=1)
    parser.add_argument("--order", type=int, default=5, help="порядок словесной n-gram")
    parser.add_argument("--char-budget", type=int, default=200_000_000)
    parser.add_argument("--memory", default="2G", help="лимит памяти lmplz (-S)")
    parser.add_argument(
        "--kenlm-bin",
        default=DEFAULT_KENLM_BIN,
        help="каталог с lmplz, interpolate, build_binary",
    )
    parser.add_argument(
        "--work-dir",
        default=None,
        help="где держать промежуточные файлы (по умолчанию htr_storage/lm/word)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="куда положить бинарную модель (по умолчанию htr_storage/lm/word_ru.binary)",
    )
    parser.add_argument(
        "--exclude-page",
        action="append",
        type=int,
        default=[],
        help="не брать подтверждённый текст этой страницы (для честного замера)",
    )
    parser.add_argument(
        "--reuse-intermediate",
        action="store_true",
        help="не пересобирать общий intermediate (дорогой шаг)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="напечатать команды и выйти (проверка без сборки KenLM)",
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs=2,
        default=None,
        metavar=("W_GENERAL", "W_AUTHOR"),
        help=(
            "готовые веса интерполяции. Нужны, когда --just_tune не сходится: "
            "на нашем корпусе автора (235 строк) он возвращает nan"
        ),
    )
    parser.add_argument(
        "--author-arpa-only",
        action="store_true",
        help=(
            "собрать только ARPA словесной модели автора (без общего корпуса и "
            "интерполяции). Нужно для честного замера: у каждой оцениваемой "
            "страницы должна быть своя авторская модель без её текста"
        ),
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# sources


def sentences_from_file(path: Path, budget: int) -> tuple[list[str], int]:
    """Read sentences from plain text or a Leipzig ``<id>\\t<sentence>`` TSV."""
    sentences: list[str] = []
    written = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            text = line.rstrip("\n")
            if "\t" in text:
                _, _, candidate = text.partition("\t")
                # Leipzig: id<TAB>sentence; a plain text with tabs keeps them
                if candidate.strip():
                    text = candidate
            text = text.strip()
            if not text:
                continue
            sentences.append(text)
            written += len(text) + 1
            if written >= budget:
                break
    return sentences, written


def author_sentences(author_id: int, exclude_pages: set[int] | None = None) -> list[str]:
    """The author's confirmed lines, one sentence per line."""
    from app.db import SessionLocal
    from app.htr.infrastructure.lexicon import SqlAlchemyAuthorCorpus

    exclude = exclude_pages or set()
    db = SessionLocal()
    try:
        corpus = SqlAlchemyAuthorCorpus(db)
        lines: list[str] = []
        for page_id, texts in corpus.confirmed_texts(author_id).items():
            if page_id in exclude:
                continue
            lines.extend(texts)
        return lines
    finally:
        db.close()


# ---------------------------------------------------------------------------
# commands


class Pipeline:
    """Runs (or prints) the KenLM commands."""

    def __init__(self, binary_dir: Path, dry_run: bool = False):
        self.binary_dir = Path(binary_dir)
        self.dry_run = dry_run
        self.commands: list[list[str]] = []

    def tool(self, name: str) -> str:
        candidate = self.binary_dir / name
        if not self.dry_run and not candidate.is_file():
            raise SystemExit(
                f"нет {candidate}; соберите KenLM (см. app/htr/README.md) "
                "или укажите --kenlm-bin"
            )
        return str(candidate)

    def run(self, args: list[str], *, stdin_path: Path | None = None,
            stdout_path: Path | None = None, label: str = "") -> None:
        self.commands.append(args)
        pretty = " ".join(args)
        if stdin_path is not None:
            pretty += f" < {stdin_path}"
        if stdout_path is not None:
            pretty += f" > {stdout_path}"
        logger.info("%s%s", f"[{label}] " if label else "", pretty)
        if self.dry_run:
            return
        started = time.time()
        with open(stdin_path, "rb") if stdin_path else _null() as source, \
             open(stdout_path, "wb") if stdout_path else _null() as sink:
            result = subprocess.run(args, stdin=source, stdout=sink, stderr=subprocess.PIPE)
        if result.returncode != 0:
            tail = result.stderr.decode("utf-8", "replace").strip().splitlines()[-5:]
            raise SystemExit(f"команда упала ({result.returncode}): {pretty}\n" + "\n".join(tail))
        logger.info("    готово за %.0f с", time.time() - started)


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


# ---------------------------------------------------------------------------


def author_arpa_only(args: argparse.Namespace) -> int:
    """One small word-level ARPA for the author, without the general corpus."""
    output = Path(args.output or "author.arpa")
    lines = author_sentences(args.author_id, set(args.exclude_page))
    if not lines:
        print("нет подтверждённого текста автора", file=sys.stderr)
        return 2
    work_dir = Path(args.work_dir or Path(settings.htr_storage_dir) / "lm" / "word")
    work_dir.mkdir(parents=True, exist_ok=True)
    text = work_dir / f"author_{args.author_id}_{'-'.join(map(str, sorted(args.exclude_page))) or 'all'}.txt"
    text.write_text("\n".join(lines) + "\n", encoding="utf-8")
    pipeline = Pipeline(Path(args.kenlm_bin), dry_run=args.dry_run)
    output.parent.mkdir(parents=True, exist_ok=True)
    pipeline.run(
        [pipeline.tool("lmplz"), "-o", str(args.order), "-S", args.memory,
         "--discount_fallback", "--arpa", str(output)],
        stdin_path=text, label="lmplz author arpa",
    )
    logger.info(
        "авторская словесная модель: %s строк, исключено страниц %s -> %s",
        len(lines), sorted(args.exclude_page), output,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = parse_args(argv)

    if args.author_arpa_only:
        return author_arpa_only(args)

    work_dir = Path(args.work_dir or Path(settings.htr_storage_dir) / "lm" / "word")
    work_dir.mkdir(parents=True, exist_ok=True)
    output = Path(
        args.output or Path(settings.htr_storage_dir) / "lm" / "word_ru.binary"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    # --- corpora ---------------------------------------------------------
    general_txt = work_dir / "general.txt"
    if args.general_text:
        shutil.copyfile(args.general_text, general_txt)
        logger.info("общий корпус: %s (скопирован)", args.general_text)
    elif args.general_corpus:
        sentences, size = sentences_from_file(Path(args.general_corpus), args.char_budget)
        if not sentences:
            print("общий корпус пуст", file=sys.stderr)
            return 2
        general_txt.write_text("\n".join(sentences) + "\n", encoding="utf-8")
        logger.info("общий корпус: %s предложений, %s символов", f"{len(sentences):,}", f"{size:,}")
    elif not general_txt.is_file():
        print("нужен --general-corpus или уже извлечённый general.txt", file=sys.stderr)
        return 2
    else:
        logger.info("общий корпус: используем существующий %s", general_txt)

    author_lines = author_sentences(args.author_id, set(args.exclude_page))
    if len(author_lines) < 10:
        print("у автора слишком мало подтверждённого текста для словесной модели", file=sys.stderr)
        return 2
    author_txt = work_dir / "author.txt"
    author_txt.write_text("\n".join(author_lines) + "\n", encoding="utf-8")
    logger.info(
        "текст автора: %s строк, %s символов", len(author_lines),
        f"{sum(len(line) for line in author_lines):,}",
    )
    # KenLM tunes weights on a file with one sentence per line; the author's own
    # text is the only text that is in-domain for this decoder
    tuning_txt = work_dir / "tuning.txt"
    tuning_txt.write_text("\n".join(author_lines) + "\n", encoding="utf-8")

    # --- estimate --------------------------------------------------------
    pipeline = Pipeline(Path(args.kenlm_bin), dry_run=args.dry_run)
    general_base = work_dir / "general.interm"
    author_base = work_dir / "author.interm"
    order = str(args.order)

    general_ready = (general_base.with_suffix(".interm.kenlm_intermediate")).is_file()
    if args.reuse_intermediate and general_ready:
        logger.info("общий intermediate переиспользуется: %s", general_base)
    else:
        pipeline.run(
            [pipeline.tool("lmplz"), "-o", order, "-S", args.memory,
             "--intermediate", str(general_base)],
            stdin_path=general_txt, label="lmplz general",
        )
    # The author's half is tiny (thousands of tokens), and there modified
    # Kneser-Ney can estimate a *negative* discount and lmplz refuses to build:
    #   "2-gram discount out of range for adjusted count 3: -0.79 ...
    #    rerun with --discount_fallback"
    # The fallback is KenLM's own documented escape hatch for exactly this, and
    # it only affects the small half of the interpolation.
    pipeline.run(
        [pipeline.tool("lmplz"), "-o", order, "-S", args.memory, "--discount_fallback",
         "--intermediate", str(author_base)],
        stdin_path=author_txt, label="lmplz author",
    )

    # --- tune weights ----------------------------------------------------
    weights_file = work_dir / "weights.txt"
    weights: list[float] | None = list(args.weights) if args.weights else None
    if weights is not None:
        logger.info("веса заданы явно: %s", weights)
    elif args.dry_run:
        pipeline.run(
            [pipeline.tool("interpolate"), "-m", str(general_base), "-m", str(author_base),
             "-t", str(tuning_txt), "--just_tune"],
            label="interpolate --just_tune",
        )
        weights = [0.5, 0.5]
    else:
        result = subprocess.run(
            [pipeline.tool("interpolate"), "-m", str(general_base), "-m", str(author_base),
             "-t", str(tuning_txt), "--just_tune"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(result.stderr[-2000:], file=sys.stderr)
            return 1
        parsed = []
        for token in result.stdout.split():
            try:
                parsed.append(float(token))
            except ValueError:
                parsed = []
                break
        if not parsed or any(value != value for value in parsed):  # nan check
            if args.weights:
                parsed = list(args.weights)
                logger.warning(
                    "interpolate --just_tune не сошёлся (%r) — берём веса %s",
                    result.stdout.strip(), parsed,
                )
            else:
                print(
                    "interpolate --just_tune не сошёлся (nan): у автора слишком мало "
                    "текста. Задайте --weights W_GENERAL W_AUTHOR",
                    file=sys.stderr,
                )
                return 2
        weights = parsed
        weights_file.write_text(" ".join(str(weight) for weight in weights), encoding="utf-8")
        logger.info("веса интерполяции (general, author): %s", weights)

    # --- build the interpolated model ------------------------------------
    arpa = work_dir / "interpolated.arpa"
    pipeline.run(
        [pipeline.tool("interpolate"), "-m", str(general_base), "-m", str(author_base),
         "-w", *(f"{weight:.6f}" for weight in weights)],
        stdout_path=arpa, label="interpolate",
    )
    pipeline.run(
        [pipeline.tool("build_binary"), str(arpa), str(output)], label="build_binary"
    )

    meta = {
        "general_corpus": str(args.general_corpus or general_txt),
        "author_id": args.author_id,
        "excluded_pages": sorted(args.exclude_page),
        "order": args.order,
        "weights": weights,
        "author_lines": len(author_lines),
        "author_chars": sum(len(line) for line in author_lines),
        "output": str(output),
    }
    if not args.dry_run:
        output.with_suffix(".json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"готово: {output} ({output.stat().st_size / 1e6:.1f} MB), веса {weights}")
    else:
        print(json.dumps(meta, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
