"""Tune the second-pass (word-level) rescoring weight on leave-one-page-out folds.

Decoding is the expensive part (~1 s per line), and the rescoring weight does not
change the beam's N-best list at all — only how those candidates are re-ranked.
So the work is split in two:

1. **one** decoding pass over the author's confirmed pages, with the same
   chronological folds and the same cached CTC matrices as
   ``measure_htr_decoders.py``, writes every line's N-best list to a cache file;
2. the grid then runs entirely offline: the candidates' word-level scores are
   computed once per candidate text, and every weight combination is pure
   arithmetic over the cache (seconds per combination instead of minutes).

The comparison is honest by construction: the character LM of a page never
contains that page (chronological fold), and the word model of a page is the
general KenLM model combined with an author model built **without** that page
(``build_htr_word_lm.py --author-arpa-only --exclude-page …``).

Usage (from the ``backend`` directory)::

    # 1) decode once and cache the N-best lists
    PYTHONPATH=/tmp/kenlm-probe2 .venv/bin/python scripts/tune_htr_rescore.py --build-cache

    # 2) grid over the rescoring weight (and the interpolation weights)
    PYTHONPATH=/tmp/kenlm-probe2 .venv/bin/python scripts/tune_htr_rescore.py \\
        --reuse-cache --rescore-weights 0.05 0.1 0.2 0.35 0.5 0.75 1.0 \\
        --interpolation-weights 1.0,0.0 0.7,0.3 0.5,0.5 --json-out htr_storage/lm/rescore_grid.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402
from app.htr.application.metrics import MetricsEvaluator  # noqa: E402
from app.htr.infrastructure.kraken.beam import BeamCandidate, PrefixBeamSearch  # noqa: E402
from app.htr.infrastructure.lm.char_ngram import CharNGram  # noqa: E402
from app.htr.infrastructure.lm.word_rescorer import (  # noqa: E402
    RescoreConfig,
    WordRescorer,
)

logger = logging.getLogger("tune_rescore")

PAGES = [3, 9, 14, 16, 17, 18, 19, 20]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-id", type=int, default=1)
    parser.add_argument("--model", default=settings.htr_default_model_path)
    parser.add_argument("--pages", type=int, nargs="*", default=PAGES)
    parser.add_argument(
        "--cache-dir",
        default=str(Path(settings.htr_storage_dir) / "lm" / "matrices"),
    )
    parser.add_argument(
        "--fold-dir",
        default=str(Path(settings.htr_storage_dir) / "lm" / "folds"),
    )
    parser.add_argument("--fold-tag", default="chrono_leipzig")
    parser.add_argument(
        "--nbest-cache",
        default=str(Path(settings.htr_storage_dir) / "lm" / "nbest_chrono.json"),
    )
    parser.add_argument("--nbest", type=int, default=10)
    parser.add_argument(
        "--only-corrected",
        action="store_true",
        help="считать только реально исправленные строки (строгий эталон)",
    )
    parser.add_argument(
        "--include-recognized",
        action="store_true",
        help="брать неподтверждённые страницы (эталон = predicted_text, см. measure_htr_decoders.py)",
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="*",
        default=None,
        help=(
            "веса символьной LM для сетки (по умолчанию один прод-вес). "
            "Каждое значение требует своего прохода декодирования, потому что "
            "alpha меняет сам список N-best"
        ),
    )
    parser.add_argument(
        "--model-map",
        default=None,
        help="JSON {page_id: path} — своя акустическая модель на страницу (честные фолды)",
    )
    parser.add_argument("--build-cache", action="store_true", help="пересобрать кэш N-best")
    parser.add_argument("--reuse-cache", action="store_true", help="использовать кэш N-best")
    parser.add_argument("--word-lm-general", default=str(
        Path(settings.htr_storage_dir) / "lm" / "word_ru.binary"
    ))
    parser.add_argument("--word-lm-author-dir", default=str(
        Path(settings.htr_storage_dir) / "lm" / "word" / "folds"
    ))
    parser.add_argument(
        "--interpolation-weights",
        nargs="*",
        default=["1.0,0.0", "0.7,0.3", "0.5,0.5"],
        help="пары W_GENERAL,W_AUTHOR для сетки",
    )
    parser.add_argument(
        "--rescore-weights",
        type=float,
        nargs="*",
        default=[0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0],
    )
    parser.add_argument(
        "--min-mean-acoustic", type=float, default=-0.30,
        help="порог уверенности для guard",
    )
    parser.add_argument("--json-out", default=None)
    return parser.parse_args(argv)


def _measure_module():
    """Import ``scripts/measure_htr_decoders.py`` by path (scripts is not a package)."""
    path = Path(__file__).with_name("measure_htr_decoders.py")
    spec = importlib.util.spec_from_file_location("measure_htr_decoders", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("measure_htr_decoders", module)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# phase 1: decode once, cache the N-best lists


def build_nbest_cache(args: argparse.Namespace, alpha: float) -> dict:
    measure = _measure_module()
    from app.htr.factory import build_recognizer, build_beam_config

    statuses = (
        ("CONFIRMED", "RECOGNIZED", "EDITING")
        if getattr(args, "include_recognized", False)
        else ("CONFIRMED",)
    )
    pages = measure.load_pages(
        args.author_id, list(args.pages), statuses,
        getattr(args, "only_corrected", False),
    )
    if not pages:
        raise SystemExit("нет подтверждённых страниц автора")
    recognizer = build_recognizer()
    model_map: dict[int, str] = {}
    if args.model_map:
        raw = json.loads(Path(args.model_map).read_text(encoding="utf-8"))
        for key, value in raw.items():
            model_map[int(key)] = str(value.get("path") if isinstance(value, dict) else value)
    tags: dict[str, object] = {}

    def net_for(page_id: int):
        path = model_map.get(page_id, args.model)
        tag = Path(path).parent.name if page_id in model_map else "default"
        if tag not in tags:
            tags[tag] = measure.prepare_model(recognizer, path)
        return tags[tag], tag

    known = measure.build_known_word(args.author_id)
    # the *production* beam parameters, with alpha from the grid
    config = build_beam_config()
    config.alpha = alpha
    logger.info("параметры beam: %s", config.describe())

    cache_root = Path(args.cache_dir)
    payload: dict = {"nbest": args.nbest, "pages": []}
    for page_id, _image_path, reference in pages:
        fold = Path(args.fold_dir) / f"{args.fold_tag}_{page_id}_char_lm.npz"
        if not fold.is_file():
            logger.warning("нет char-LM фолда для стр.%s (%s) — пропускаю", page_id, fold)
            continue
        char_lm = CharNGram.load(str(fold))
        page_net, tag = net_for(page_id)
        search = PrefixBeamSearch(
            codec=page_net.codec, lm=char_lm, config=config, known_word=known
        )
        suffix = "" if tag == "default" else f"_{tag}"
        matrices, greedy = measure.load_page_cache(cache_root / f"page_{page_id}{suffix}.npz")
        lines = []
        for index, matrix in enumerate(matrices):
            expected = (reference.get(index) or "").strip()
            if not expected:
                continue
            candidates = search.decode_nbest(matrix, n=args.nbest)
            lines.append(
                {
                    "reference": expected,
                    "greedy": greedy[index],
                    "nbest": [
                        {
                            "text": candidate.text,
                            "score": candidate.score,
                            "acoustic": candidate.acoustic,
                            "lm": candidate.lm,
                            "words": candidate.words,
                        }
                        for candidate in candidates
                    ],
                }
            )
        logger.info("стр.%s: %s строк, гипотез %s", page_id, len(lines), args.nbest)
        payload["pages"].append({"page": page_id, "lines": lines})

    path = cache_path(args, alpha)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    logger.info("кэш N-best: %s (%.1f MB)", path, path.stat().st_size / 1e6)
    return payload


def cache_path(args: argparse.Namespace, alpha: float) -> Path:
    """One N-best cache per (fold tag, alpha): alpha changes the candidates."""
    base = Path(args.nbest_cache)
    return base.with_name(f"{base.stem}_{args.fold_tag}_a{alpha:g}{base.suffix}")


# ---------------------------------------------------------------------------
# phase 2: score every candidate once with both word models


def score_candidates(payload: dict, args: argparse.Namespace) -> dict:
    import kenlm
    import unicodedata

    general = kenlm.Model(args.word_lm_general) if args.word_lm_general else None
    logger.info("общая словесная модель: %s", args.word_lm_general)
    scores: dict[str, dict[str, float]] = {}
    for page in payload["pages"]:
        page_id = page["page"]
        author_path = Path(args.word_lm_author_dir) / f"author_{page_id}.arpa"
        author = kenlm.Model(str(author_path)) if author_path.is_file() else None
        if author is None:
            logger.info("стр.%s: авторской модели нет, только общая", page_id)
        for line in page["lines"]:
            for candidate in line["nbest"]:
                key = unicodedata.normalize("NFC", candidate["text"])
                if key in scores:
                    continue
                scores[key] = {
                    "general": float(general.score(key, bos=True, eos=True)) if general else 0.0,
                    "author": float(author.score(key, bos=True, eos=True)) if author else 0.0,
                }
    logger.info("оценено уникальных гипотез: %s", len(scores))
    return scores


class PrecomputedScorer:
    """Word-level score of a hypothesis, read from the precomputed table."""

    def __init__(self, scores: dict, weights: tuple[float, float]):
        self.scores = scores
        self.weights = weights

    def score(self, sentence: str) -> float:
        entry = self.scores.get(sentence)
        if entry is None:
            return 0.0
        return self.weights[0] * entry["general"] + self.weights[1] * entry["author"]


# ---------------------------------------------------------------------------
# phase 3: the grid


def evaluate(payload: dict, scores: dict, args: argparse.Namespace,
             weights: tuple[float, float], rescore_weight: float, guard: bool,
             known) -> dict[int, list[tuple[str, str]]]:
    """Per-page (reference, chosen) pairs for one configuration."""
    per_page: dict[int, list[tuple[str, str]]] = {}
    for page in payload["pages"]:
        rescorer = WordRescorer(
            PrecomputedScorer(scores, weights),
            RescoreConfig(
                weight=rescore_weight,
                guard=guard,
                min_mean_acoustic=args.min_mean_acoustic,
            ),
            known_word=known,
        )
        pairs: list[tuple[str, str]] = []
        for line in page["lines"]:
            candidates = [
                BeamCandidate(
                    text=item["text"],
                    score=item["score"],
                    acoustic=item["acoustic"],
                    lm=item["lm"],
                    words=item["words"],
                )
                for item in line["nbest"]
            ]
            first = candidates[0].text
            # w = 0 is the baseline by definition: the word model must not move
            # anything, whatever the guard would have allowed
            chosen = first if rescore_weight == 0.0 else rescorer.choose(candidates).text
            pairs.append((line["reference"], chosen))
        per_page[page["page"]] = pairs
    return per_page


def aggregate(per_page: dict[int, list[tuple[str, str]]],
              pages: list[int] | None = None) -> dict:
    evaluator = MetricsEvaluator()
    pairs: list[tuple[str, str]] = []
    for page_id, page_pairs in per_page.items():
        if pages is not None and page_id not in pages:
            continue
        pairs.extend(page_pairs)
    return evaluator.evaluate_pairs(pairs)


def baseline_pairs(payload: dict) -> dict[int, list[tuple[str, str]]]:
    return {
        page["page"]: [(line["reference"], line["nbest"][0]["text"]) for line in page["lines"]]
        for page in payload["pages"]
    }


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = parse_args(argv)

    # one decoding pass per alpha: alpha changes the N-best list itself, so it
    # cannot be gridded from a single cache
    alphas = args.alphas or [settings.htr_beam_alpha]
    runs = []
    for alpha in alphas:
        path = cache_path(args, alpha)
        if args.build_cache or not (args.reuse_cache and path.is_file()):
            payload = build_nbest_cache(args, alpha)
        else:
            payload = json.loads(path.read_text(encoding="utf-8"))
            logger.info("кэш N-best переиспользуется: %s", path)
        runs.append((alpha, payload, score_candidates(payload, args)))

    measure = _measure_module()
    known = measure.build_known_word(args.author_id)

    print("\nбазовая линия (beam без rescoring):")
    for alpha, payload, _scores in runs:
        base = aggregate(baseline_pairs(payload))
        print(f"  alpha={alpha}: CER {base['cer']:.4f} WER {base['wer']:.4f}")

    configurations = []
    for alpha, payload, scores in runs:
        for pair in args.interpolation_weights:
            left, _, right = pair.partition(",")
            weights = (float(left), float(right or 0.0))
            for guard in (True, False):
                for rescore_weight in args.rescore_weights:
                    per_page = evaluate(
                        payload, scores, args, weights, rescore_weight, guard, known
                    )
                    metrics = aggregate(per_page)
                    configurations.append(
                        {
                            "alpha": alpha,
                            "interpolation": list(weights),
                            "rescore_weight": rescore_weight,
                            "guard": guard,
                            "metrics": metrics,
                            "per_page": per_page,
                        }
                    )

    print(f"\n{'alpha':>5} {'W_g':>4} {'W_a':>4} {'guard':>6} {'w':>5} | {'CER':>7} {'WER':>7}")
    for config in sorted(configurations, key=lambda item: (item["metrics"]["wer"], item["metrics"]["cer"]))[:15]:
        metrics = config["metrics"]
        print(
            f"{config['alpha']:>5} {config['interpolation'][0]:>4.1f} "
            f"{config['interpolation'][1]:>4.1f} "
            f"{str(config['guard']):>6} {config['rescore_weight']:>5.2f} | "
            f"{metrics['cer']:>7.4f} {metrics['wer']:>7.4f}"
        )

    best = min(configurations, key=lambda item: (item["metrics"]["wer"], item["metrics"]["cer"]))
    print(
        f"\nлучшая комбинация на всех страницах: alpha={best['alpha']}, "
        f"W_general={best['interpolation'][0]}, W_author={best['interpolation'][1]}, "
        f"guard={best['guard']}, w={best['rescore_weight']} -> "
        f"CER {best['metrics']['cer']:.4f} WER {best['metrics']['wer']:.4f}"
    )

    # --- honest selection: the weight is picked on the *other* pages --------
    pages = [page["page"] for page in runs[0][1]["pages"]]
    selected: dict[int, dict] = {}
    held_out: dict[int, list[tuple[str, str]]] = {}
    for page_id in pages:
        others = [candidate for candidate in configurations
                  if candidate["rescore_weight"] > 0.0]
        rest = [other for other in pages if other != page_id]

        def others_metric(item):
            metrics = aggregate(item["per_page"], rest)
            return (metrics["wer"], metrics["cer"])

        choice = min(others, key=others_metric)
        selected[page_id] = choice
        held_out[page_id] = choice["per_page"][page_id]
        metrics = aggregate({page_id: choice["per_page"][page_id]})
        print(
            f"  выбор для стр.{page_id:>3}: W_g={choice['interpolation'][0]} "
            f"W_a={choice['interpolation'][1]} guard={choice['guard']} "
            f"w={choice['rescore_weight']:<4} -> на ней CER {metrics['cer']:.4f} "
            f"WER {metrics['wer']:.4f}"
        )
    loo_metrics = aggregate(held_out)
    # каждая страница сравнивается с beam того alpha, который для неё выбран
    loo_beam = aggregate(
        {
            page_id: baseline_pairs(
                next(payload for alpha, payload, _s in runs if alpha == selected[page_id]["alpha"])
            )[page_id]
            for page_id in pages
        }
    )
    print(
        f"\nчестная оценка (вес выбран на остальных страницах): "
        f"CER {loo_metrics['cer']:.4f} WER {loo_metrics['wer']:.4f} "
        f"против beam CER {loo_beam['cer']:.4f} WER {loo_beam['wer']:.4f} "
        f"({(loo_beam['wer'] - loo_metrics['wer']) / loo_beam['wer'] * 100:+.1f}% WER)"
    )
    choices: dict[str, int] = {}
    for choice in selected.values():
        key = (
            f"alpha={choice['alpha']},w={choice['rescore_weight']},guard={choice['guard']},"
            f"W=({choice['interpolation'][0]},{choice['interpolation'][1]})"
        )
        choices[key] = choices.get(key, 0) + 1
    print("распределение выбранных настроек:", choices)

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(
                {
                    "baseline": {
                        str(alpha): aggregate(baseline_pairs(payload))
                        for alpha, payload, _scores in runs
                    },
                    "best_on_all_pages": {
                        key: value for key, value in best.items() if key != "per_page"
                    },
                    "leave_one_page_out": {
                        "metrics": loo_metrics,
                        "beam": loo_beam,
                        "selected": {
                            str(page_id): {
                                "alpha": choice["alpha"],
                                "interpolation": choice["interpolation"],
                                "rescore_weight": choice["rescore_weight"],
                                "guard": choice["guard"],
                            }
                            for page_id, choice in selected.items()
                        },
                    },
                    "grid": [
                        {
                            "alpha": config["alpha"],
                            "interpolation": config["interpolation"],
                            "rescore_weight": config["rescore_weight"],
                            "guard": config["guard"],
                            "cer": config["metrics"]["cer"],
                            "wer": config["metrics"]["wer"],
                        }
                        for config in configurations
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
