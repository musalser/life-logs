"""Measure greedy vs prefix-beam decoding on the author's own confirmed pages.

Protocol (**chronological** folds, the default honest mode):

* the acoustic model is the **default** one, which has never seen these pages,
  so the numbers describe unseen material rather than a memorized page;
* with ``--chronological`` the language model of a page is rebuilt from the
  confirmed pages that come *before* it only — exactly what production has at
  the moment that page is recognized. ``--leave-one-page-out`` is the weaker
  variant that hides just the page itself. Either way the page cannot leak its
  own words into its own decoder: a leaky global model made page 20 look almost
  perfect (CER 5.6 % -> 1.3 %) and hid the fact that the LM only starts to win
  once enough real running text exists;
* below ``--min-lm-text-chars`` of real running text the decoder falls back to
  greedy, exactly as the recognizer does in production;
* every line is decoded twice from the *same* matrix (greedy and beam), so the
  only difference is the decoder;
* the reference is the user's own transcription (``corrected_text``).

Recognized matrices *and* the per-page LM folds are cached, so decoder
parameters can be re-tuned without touching the GPU or the database again::

    # honest numbers, cached: reuse matrices, rebuild folds only if missing
    .venv/bin/python scripts/measure_htr_decoders.py --chronological --reuse-cache
    # re-tune a parameter
    .venv/bin/python scripts/measure_htr_decoders.py --chronological --reuse-cache --alpha 0.4
    # measure a fine-tuned model (matrices get their own cache namespace)
    .venv/bin/python scripts/measure_htr_decoders.py --model htr_storage/models/v12.safetensors
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from app.config import settings  # noqa: E402
from app.htr.application.metrics import MetricsEvaluator  # noqa: E402
from app.htr.infrastructure.kraken.beam import (  # noqa: E402
    BeamSearchConfig,
    PrefixBeamSearch,
)
from app.htr.infrastructure.lm.char_ngram import CharNGram  # noqa: E402

logger = logging.getLogger("measure_decoders")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-id", type=int, default=1)
    parser.add_argument(
        "--model",
        default=settings.htr_default_model_path,
        help="акустическая модель (по умолчанию базовая — вневыборочно для этих страниц)",
    )
    parser.add_argument(
        "--lm", default=str(Path(settings.htr_storage_dir) / "lm" / "ru_char_lm.npz")
    )
    parser.add_argument(
        "--cache-dir", default=str(Path(settings.htr_storage_dir) / "lm" / "matrices")
    )
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument("--pages", type=int, nargs="*", default=None)
    parser.add_argument(
        "--only-corrected",
        action="store_true",
        help=(
            "считать метрики только по строкам, которые пользователь реально "
            "исправлял (corrected_text). Без этого у неподтверждённых страниц и у "
            "строк без правок эталоном становится вывод самой модели, и CER "
            "искусственно падает"
        ),
    )
    parser.add_argument(
        "--include-recognized",
        action="store_true",
        help=(
            "брать и неподтверждённые страницы (RECOGNIZED/EDITING). Эталона у них "
            "нет: за эталон берётся predicted_text, то есть вывод той модели, "
            "которая их распознала. Годится для сравнения моделей и влияния LM, "
            "но НЕ для честного CER/WER"
        ),
    )
    parser.add_argument(
        "--model-map",
        default=None,
        help=(
            "JSON {page_id: path} или {page_id: {path: ...}} — своя акустическая "
            "модель на страницу. Нужно для честного замера: страница должна "
            "распознаваться моделью, созданной ДО её подтверждения. Кэш матриц "
            "при этом получает метку модели, иначе тексты разных моделей смешаются"
        ),
    )
    parser.add_argument(
        "--leave-one-page-out",
        action="store_true",
        help=(
            "для каждой страницы собрать LM без её подтверждённого текста "
            "(страница не подсказывает декодеру собственные слова)"
        ),
    )
    parser.add_argument(
        "--chronological",
        action="store_true",
        help=(
            "LM для страницы собирается только из подтверждённых страниц с "
            "меньшим id, то есть из тех, что были подтверждены раньше. Это "
            "точная модель продакшена: к моменту распознавания страницы у "
            "автора есть только предыдущие листы. Подразумевает LOO и "
            "перекрывает его"
        ),
    )
    parser.add_argument("--author-weight", type=int, default=30)
    parser.add_argument("--order", type=int, default=6)
    parser.add_argument("--min-count", type=int, default=3)
    parser.add_argument(
        "--min-lm-text-chars",
        type=int,
        default=settings.htr_lm_min_text_chars,
        help="ниже этого объёма реального текста beam не включается (как в проде)",
    )
    parser.add_argument(
        "--fold-dir",
        default=str(Path(settings.htr_storage_dir) / "lm" / "folds"),
        help="куда кэшировать LM-фолды leave-one-page-out",
    )
    parser.add_argument(
        "--fold-tag",
        default=None,
        help=(
            "метка набора фолдов, чтобы не переиспользовать чужой кэш: "
            "имена станут <tag>_<page>_char_lm.npz (по умолчанию chrono/page). "
            "Нужна, когда меняется корпус (например, добавлен --extra-text)"
        ),
    )
    # the defaults mirror the production settings, so a bare run measures the
    # configuration that actually serves recognition
    parser.add_argument("--beam-width", type=int, default=settings.htr_beam_width)
    parser.add_argument("--top-k", type=int, default=settings.htr_beam_top_k)
    parser.add_argument("--alpha", type=float, default=settings.htr_beam_alpha, help="вес символьной LM")
    parser.add_argument("--beta", type=float, default=settings.htr_beam_beta, help="бонус за длину слова")
    parser.add_argument(
        "--word-bonus",
        type=float,
        default=settings.htr_beam_word_bonus,
        help="бонус за слово из словаря",
    )
    parser.add_argument("--no-lm", action="store_true", help="alpha = 0: только акустика")
    parser.add_argument("--no-lexicon", action="store_true")
    parser.add_argument(
        "--word-lm-general",
        default=None,
        help="словесная KenLM-модель общего корпуса (бинарник или ARPA), общая для всех фолдов",
    )
    parser.add_argument(
        "--word-lm-author-dir",
        default=None,
        help=(
            "каталог с author_<page>.arpa — словесная модель автора без текста "
            "оцениваемой страницы (честный замер rescoring)"
        ),
    )
    parser.add_argument(
        "--word-weights",
        type=float,
        nargs=2,
        default=(0.5, 0.5),
        metavar=("W_GENERAL", "W_AUTHOR"),
        help="веса интерполяции из interpolate --just_tune",
    )
    parser.add_argument(
        "--rescore-n", type=int, default=10,
        help="сколько гипотез beam отдавать на переранжирование",
    )
    parser.add_argument(
        "--rescore-weight", type=float, default=0.5,
        help="вес словесной модели в итоговом скоре (стартуйте с малого)",
    )
    parser.add_argument(
        "--no-rescore-guard", action="store_true",
        help="разрешить rescoring менять победителя всегда (для сравнения)",
    )
    parser.add_argument(
        "--min-mean-acoustic", type=float, default=-0.30,
        help="порог средней акустической уверенности символа (log10)",
    )
    parser.add_argument("--examples", type=int, default=8, help="сколько расхождений напечатать")
    parser.add_argument("--json-out", default=None)
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------


def load_pages(
    author_id: int,
    only: list[int] | None,
    statuses=("CONFIRMED",),
    only_corrected: bool = False,
):
    from app.db import SessionLocal
    from app.models import HTRLine, HTRPage

    db = SessionLocal()
    try:
        query = (
            db.query(HTRPage)
            .filter(HTRPage.author_id == author_id, HTRPage.status.in_(statuses))
            .order_by(HTRPage.id)
        )
        pages = []
        for page in query.all():
            if only and page.id not in only:
                continue
            lines = (
                db.query(HTRLine)
                .filter(HTRLine.page_id == page.id)
                .order_by(HTRLine.order_index)
                .all()
            )
            reference = {}
            for line in lines:
                if only_corrected:
                    if not line.corrected_text:
                        continue
                    reference[line.order_index] = line.corrected_text
                else:
                    reference[line.order_index] = (
                        line.corrected_text or line.predicted_text or ""
                    )
            pages.append((page.id, page.file_path, reference))
        return pages
    finally:
        db.close()


def build_known_word(author_id: int):
    from app.db import SessionLocal
    from app.htr.infrastructure.lexicon import (
        LayeredLexiconChecker,
        SqlAlchemyAuthorCorpus,
        load_file_lexicon,
    )

    db = SessionLocal()
    try:
        checker = LayeredLexiconChecker(
            base=load_file_lexicon(settings.htr_lexicon_path),
            extra=SqlAlchemyAuthorCorpus(db).known_words(author_id),
        )
    finally:
        db.close()

    def known(word: str) -> bool:
        # the decoder works on the decomposed text the codec emits
        return checker.is_known(unicodedata.normalize("NFC", word))

    return known


# ---------------------------------------------------------------------------


class InterpolatedWordScorer:
    """Word-level score of a line: KenLM general + KenLM of this author.

    KenLM's ``interpolate`` bakes the tuned weights into a single model; for the
    *measurement* we keep the two apart, because the author half must be rebuilt
    without the evaluated page (otherwise the page suggests its own words) while
    the general half is the same for every fold. Applying the same weights to
    the two scores gives the same ranking as the interpolated model.
    """

    def __init__(self, general, author, weights):
        self.general = general
        self.author = author
        self.weights = weights

    def score(self, sentence: str) -> float:
        total = 0.0
        if self.general is not None:
            total += self.weights[0] * float(self.general.score(sentence, bos=True, eos=True))
        if self.author is not None:
            total += self.weights[1] * float(self.author.score(sentence, bos=True, eos=True))
        return total


def load_word_models(args, pages: list):
    """(general model, {page_id: author model}); empty when rescoring is off."""
    if not args.word_lm_general and not args.word_lm_author_dir:
        return None, {}
    import kenlm  # optional: only the rescoring experiment needs it

    general = kenlm.Model(args.word_lm_general) if args.word_lm_general else None
    if general is not None:
        logger.info("словесная модель (общая): %s, порядок %s", args.word_lm_general, general.order)
    authors = {}
    if args.word_lm_author_dir:
        directory = Path(args.word_lm_author_dir)
        for page_id, *_rest in pages:
            path = directory / f"author_{page_id}.arpa"
            if path.is_file():
                authors[page_id] = kenlm.Model(str(path))
                logger.info("словесная модель автора для стр.%s: %s", page_id, path.name)
            else:
                logger.warning("нет авторской модели для стр.%s (%s)", page_id, path)
    return general, authors


def prepare_model(recognizer, model_path: str):
    """Load the acoustic model once with logits enabled."""
    from kraken.configs import RecognitionInferenceConfig
    from kraken.tasks import RecognitionTaskModel

    from app.htr.infrastructure.kraken.recognizer import available_lightning_device

    task = RecognitionTaskModel(recognizer._deserialize_models(Path(model_path)))
    net = task.net
    accelerator, devices = available_lightning_device(recognizer.device)
    net.prepare_for_inference(
        RecognitionInferenceConfig(
            accelerator=accelerator,
            device=devices,
            batch_size=recognizer.batch_size,
            padding=recognizer.padding,
            num_line_workers=0,
            return_logits=True,
        )
    )
    return net


def recognize_page(recognizer, net, image_path: str, cache: Path):
    from app.htr.infrastructure.storage import open_oriented_image

    started = time.time()
    image = open_oriented_image(image_path)
    try:
        segmentation = recognizer._merge_split_lines(recognizer._segment(image))
        records = list(net.predict(image, segmentation))
    finally:
        image.close()
    matrices = [
        np.asarray(record.logits.detach().cpu(), dtype=np.float16) for record in records
    ]
    payload = {
        "matrices": _object_array(matrices),
        "greedy": _object_array([record.prediction for record in records]),
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **payload)
    logger.info("распознано за %.1f с, строк %s", time.time() - started, len(records))
    return [np.asarray(item, dtype=np.float32) for item in payload["matrices"]], list(payload["greedy"])


def _lm_builder():
    """Import ``scripts/build_htr_lm.py`` by path (``scripts`` is not a package)."""
    import importlib.util

    path = Path(__file__).with_name("build_htr_lm.py")
    spec = importlib.util.spec_from_file_location("build_htr_lm", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("build_htr_lm", module)
    spec.loader.exec_module(module)
    return module


def fold_page_ids(args: argparse.Namespace, pages: list, page_id: int) -> list[int]:
    """Which confirmed pages the LM of ``page_id`` must not see.

    ``--leave-one-page-out`` hides the evaluated page itself. ``--chronological``
    hides it *and every later page*, which is what production can actually have:
    a page is recognized before its own text (and the text of later pages) has
    been confirmed.
    """
    ids = [item[0] for item in pages]
    if args.chronological:
        return [other for other in ids if other >= page_id]
    return [page_id]


def leave_one_out_lms(args: argparse.Namespace, pages: list) -> dict:
    """One LM per page, built without that page (and later ones, if asked)."""
    builder = _lm_builder()
    builder_args = builder.parse_args(
        [
            "--author-id", str(args.author_id),
            "--author-weight", str(args.author_weight),
            "--order", str(args.order),
            "--min-count", str(args.min_count),
            "--output", str(args.lm),
        ]
    )
    fold_dir = Path(args.fold_dir)
    fold_dir.mkdir(parents=True, exist_ok=True)
    # the general part (word forms, extra text) does not depend on the page
    prose, words = builder.general_texts(builder_args)
    models: dict[int, CharNGram] = {}
    prefix = args.fold_tag or ("chrono" if args.chronological else "page")
    for page_id, *_rest in pages:
        cache = fold_dir / f"{prefix}_{page_id}_char_lm.npz"
        if args.reuse_cache and cache.is_file():
            models[page_id] = CharNGram.load(str(cache))
            continue
        hidden = fold_page_ids(args, pages, page_id)
        builder_args.exclude_page = hidden
        parts = builder.author_part(builder_args, args.author_id)
        model = builder.build_model(builder_args, parts, prose, words)
        model.save(str(cache))
        logger.info(
            "фолд стр.%s: скрыто страниц %s, текст %s символов -> %s",
            page_id, len(hidden), model.meta.get("running_text_chars", 0), cache.name,
        )
        models[page_id] = model
    return models


def _object_array(items: list):
    """Object array of items with different shapes (lines have different lengths)."""
    array = np.empty(len(items), dtype=object)
    array[:] = items
    return array


def load_page_cache(cache: Path):
    with np.load(cache, allow_pickle=True) as data:
        return (
            [np.asarray(item, dtype=np.float32) for item in data["matrices"]],
            [str(item) for item in data["greedy"]],
        )


# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = parse_args(argv)

    statuses = (
        ("CONFIRMED", "RECOGNIZED", "EDITING")
        if args.include_recognized
        else ("CONFIRMED",)
    )
    pages = load_pages(args.author_id, args.pages, statuses, args.only_corrected)
    if not pages:
        print("нет подтверждённых страниц автора", file=sys.stderr)
        return 2

    folds = (
        leave_one_out_lms(args, pages)
        if (args.leave_one_page_out or args.chronological) and not args.no_lm
        else None
    )
    lm = None if args.no_lm else CharNGram.load(args.lm)
    if folds is not None:
        mode = "хронологический" if args.chronological else "leave-one-page-out"
        print(
            f"LM: {mode}, {len(folds)} фолдов, "
            f"алфавит {len(next(iter(folds.values())).chars)}"
        )
    elif lm is not None:
        print(
            f"LM: {args.lm} — порядок {lm.order}, алфавит {len(lm.chars)}, "
            f"{lm.size_bytes / 1e6:.0f} MB в памяти"
        )
    known = None if args.no_lexicon else build_known_word(args.author_id)
    config = BeamSearchConfig(
        beam_width=args.beam_width,
        top_k=args.top_k,
        alpha=0.0 if args.no_lm else args.alpha,
        beta=args.beta,
        word_bonus=args.word_bonus,
    )
    print(f"beam: {config.describe()}")

    from app.htr.factory import build_recognizer

    recognizer = build_recognizer()
    model_map: dict[int, str] = {}
    if args.model_map:
        raw = json.loads(Path(args.model_map).read_text(encoding="utf-8"))
        for key, value in raw.items():
            path = value.get("path") if isinstance(value, dict) else value
            model_map[int(key)] = str(path)
        print("модели по страницам:", {page: Path(path).parent.name for page, path in model_map.items()})

    def model_and_tag(page_id: int) -> tuple[str, str]:
        path = model_map.get(page_id, args.model)
        return path, Path(path).parent.name if page_id in model_map else "default"

    nets: dict[str, object] = {}

    def net_for(page_id: int):
        path, tag = model_and_tag(page_id)
        if tag not in nets:
            nets[tag] = prepare_model(recognizer, path)
        return nets[tag]

    first_net = net_for(pages[0][0])
    beam = PrefixBeamSearch(codec=first_net.codec, lm=lm, config=config, known_word=known)

    def search_for(page_id: int) -> PrefixBeamSearch | None:
        """The decoder this page gets in production (greedy when the LM is weak)."""
        if folds is None:
            return beam
        page_lm = folds[page_id]
        running = int(page_lm.meta.get("running_text_chars", 0) or 0)
        if page_lm.char_id(" ") == 0 or running < args.min_lm_text_chars:
            logger.info(
                "стр.%s: текста в LM %s символов (< %s) — greedy",
                page_id, running, args.min_lm_text_chars,
            )
            return None
        return PrefixBeamSearch(
            codec=net_for(page_id).codec, lm=page_lm, config=config, known_word=known
        )
    word_general, word_authors = load_word_models(args, pages)
    evaluator = MetricsEvaluator()
    cache_root = Path(args.cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    all_greedy: list[tuple[str, str]] = []
    all_beam: list[tuple[str, str]] = []
    all_rescored: list[tuple[str, str]] = []
    differences: list[tuple[int, str, str, str]] = []
    rescore_changes: list[tuple[int, str, str, str, str]] = []
    decoder_gap: list[tuple[str, str]] = []
    rescore_gap: list[tuple[str, str]] = []
    rows = []
    for page_id, image_path, reference in pages:
        page_net = net_for(page_id)
        _path, tag = model_and_tag(page_id)
        suffix = "" if tag == "default" else f"_{tag}"
        cache = cache_root / f"page_{page_id}{suffix}.npz"
        if args.reuse_cache and cache.is_file():
            matrices, greedy = load_page_cache(cache)
        else:
            matrices, greedy = recognize_page(recognizer, page_net, image_path, cache)

        page_search = search_for(page_id)
        rescorer = None
        if word_general is not None or page_id in word_authors:
            from app.htr.infrastructure.lm.word_rescorer import RescoreConfig, WordRescorer

            rescorer = WordRescorer(
                InterpolatedWordScorer(
                    word_general, word_authors.get(page_id), args.word_weights
                ),
                RescoreConfig(
                    weight=args.rescore_weight,
                    guard=not args.no_rescore_guard,
                    min_mean_acoustic=args.min_mean_acoustic,
                ),
                known_word=known,
            )
        running = (
            int(folds[page_id].meta.get("running_text_chars", 0) or 0)
            if folds is not None
            else -1
        )
        page_greedy: list[tuple[str, str]] = []
        page_beam: list[tuple[str, str]] = []
        page_rescored: list[tuple[str, str]] = []
        for index, matrix in enumerate(matrices):
            expected = (reference.get(index) or "").strip()
            if not expected:
                continue
            search = page_search if page_search is not None else None
            if search is not None:
                candidates = (
                    search.decode_nbest(matrix, n=args.rescore_n)
                    if rescorer is not None
                    else None
                )
                beam_text = candidates[0].text if candidates else search.decode(matrix)
            elif folds is not None:
                # production fallback: an unusable fold means greedy, *not* the
                # global model (that one contains this very page and would
                # report a fake improvement)
                beam_text = greedy[index]
                candidates = None
            else:
                candidates = (
                    beam.decode_nbest(matrix, n=args.rescore_n)
                    if rescorer is not None
                    else None
                )
                beam_text = candidates[0].text if candidates else beam.decode(matrix)

            rescored_text = beam_text
            if rescorer is not None and candidates:
                outcome = rescorer.choose(candidates)
                rescored_text = outcome.text
                if outcome.changed:
                    rescore_changes.append(
                        (page_id, beam_text, rescored_text, expected, outcome.reason)
                    )

            page_greedy.append((expected, greedy[index]))
            page_beam.append((expected, beam_text))
            page_rescored.append((expected, rescored_text))
            if beam_text != greedy[index]:
                differences.append((page_id, greedy[index], beam_text, expected))

        all_greedy.extend(page_greedy)
        all_beam.extend(page_beam)
        all_rescored.extend(page_rescored)
        # how much the language model *itself* moved the output, independent of
        # any reference: greedy -> beam is the character LM, beam -> rescored the
        # word LM. On strong acoustics these should shrink.
        decoder_gap.extend((g, b) for (_ref, g), (_ref2, b) in zip(page_greedy, page_beam))
        rescore_gap.extend((b, r) for (_ref, b), (_ref2, r) in zip(page_beam, page_rescored))
        greedy_metrics = evaluator.evaluate_pairs(page_greedy)
        beam_metrics = evaluator.evaluate_pairs(page_beam)
        rescored_metrics = evaluator.evaluate_pairs(page_rescored)
        rows.append(
            (page_id, len(page_greedy), greedy_metrics, beam_metrics, rescored_metrics)
        )
        extra = f" | текст {running}" if running >= 0 else ""
        rescored_note = ""
        if rescorer is not None:
            rescored_note = (
                f" | rescored CER {rescored_metrics['cer']:.4f} "
                f"WER {rescored_metrics['wer']:.4f}"
            )
        print(
            f"page {page_id:>3}: строк {len(page_greedy):>3} | "
            f"greedy CER {greedy_metrics['cer']:.4f} WER {greedy_metrics['wer']:.4f} | "
            f"beam CER {beam_metrics['cer']:.4f} WER {beam_metrics['wer']:.4f}"
            f"{rescored_note}{extra}"
        )

    total_greedy = evaluator.evaluate_pairs(all_greedy)
    total_beam = evaluator.evaluate_pairs(all_beam)
    total_rescored = (
        evaluator.evaluate_pairs(all_rescored) if rescore_changes or word_general else None
    )
    print("\nитог (все страницы, микро-среднее):")
    print(f"  greedy: CER {total_greedy['cer']:.4f}   WER {total_greedy['wer']:.4f}")
    print(f"  beam:   CER {total_beam['cer']:.4f}   WER {total_beam['wer']:.4f}")
    delta_cer = total_greedy["cer"] - total_beam["cer"]
    delta_wer = total_greedy["wer"] - total_beam["wer"]
    print(f"  выигрыш beam: CER {delta_cer:+.4f}   WER {delta_wer:+.4f}")
    if total_rescored is not None:
        print(
            f"  rescored: CER {total_rescored['cer']:.4f}   "
            f"WER {total_rescored['wer']:.4f}"
        )
        print(
            f"  выигрыш rescoring поверх beam: "
            f"CER {total_beam['cer'] - total_rescored['cer']:+.4f}   "
            f"WER {total_beam['wer'] - total_rescored['wer']:+.4f}"
        )
        print(f"  словесная модель изменила победителя в {len(rescore_changes)} строках")
    if decoder_gap:
        gap = evaluator.evaluate_pairs(decoder_gap)
        print(
            f"\nвлияние символьной LM (greedy -> beam, без эталона): "
            f"CER {gap['cer']:.4f} WER {gap['wer']:.4f} "
            f"(то есть LM изменила {gap['cer'] * 100:.1f}% символов)"
        )
    if rescore_gap:
        gap = evaluator.evaluate_pairs(rescore_gap)
        print(
            f"влияние словесной LM (beam -> rescored): "
            f"CER {gap['cer']:.4f} WER {gap['wer']:.4f}"
        )
    print(f"\nстрок изменено декодером: {len(differences)} из {len(all_greedy)}")
    for page_id, before, after, expected in differences[: args.examples]:
        print(f"  стр.{page_id}: было  {before[:70]!r}")
        print(f"           стало {after[:70]!r}")
        print(f"           эталон {expected[:70]!r}")
    for page_id, before, after, expected, reason in rescore_changes[: args.examples]:
        print(f"  [rescoring] стр.{page_id}: {before[:60]!r} -> {after[:60]!r} ({reason})")

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(
                {
                    "model": args.model,
                    "lm": None if args.no_lm else args.lm,
                    "leave_one_page_out": bool(args.leave_one_page_out),
                    "chronological": bool(args.chronological),
                    "config": config.describe(),
                    "rescore": {
                        "general_lm": args.word_lm_general,
                        "author_dir": args.word_lm_author_dir,
                        "weights": list(args.word_weights),
                        "rescore_weight": args.rescore_weight,
                        "n": args.rescore_n,
                        "guard": not args.no_rescore_guard,
                    },
                    "pages": [
                        {
                            "page": page_id,
                            "lines": lines,
                            "greedy": greedy,
                            "beam": beam,
                            "rescored": rescored,
                        }
                        for page_id, lines, greedy, beam, rescored in rows
                    ],
                    "total": {
                        "greedy": total_greedy,
                        "beam": total_beam,
                        "rescored": total_rescored,
                    },
                    "lm_effect": {
                        "char_lm_cer": evaluator.evaluate_pairs(decoder_gap)["cer"] if decoder_gap else None,
                        "char_lm_wer": evaluator.evaluate_pairs(decoder_gap)["wer"] if decoder_gap else None,
                        "word_lm_cer": evaluator.evaluate_pairs(rescore_gap)["cer"] if rescore_gap else None,
                        "word_lm_wer": evaluator.evaluate_pairs(rescore_gap)["wer"] if rescore_gap else None,
                        "lines_changed_by_decoder": len(differences),
                        "lines_changed_by_rescoring": len(rescore_changes),
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
