"""A/B the training line height: the checkpoint's own (96 px) against a forced one.

PP-OCRv6 was pretrained at 128 px, so 96 px in ``htr_training_height`` looks like
an accidental scale mismatch. It is not: an exported checkpoint declares its own
input shape and kraken overwrites ``config.height`` with it
(``kraken/train/ppocr.py``), so the setting never reaches the network. The only
way to change the height is ``height_override`` in the trainer's backend
options, and that is what this script compares — same corpus, same split, same
seed, only the input geometry differs.

Everything runs through the real trainer directly (no training service), so the
user's model list is not touched; checkpoints go to ``--output-dir``.

Usage (from the ``backend`` directory)::

    .venv/bin/python scripts/measure_htr_height.py                 # 96 vs 128
    .venv/bin/python scripts/measure_htr_height.py --heights 96 128 192
    .venv/bin/python scripts/measure_htr_height.py --json-out htr_storage/lm/height_ab.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402

logger = logging.getLogger("measure_height")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-id", type=int, default=1)
    parser.add_argument(
        "--heights",
        type=int,
        nargs="+",
        default=[0, 128],
        help=(
            "0 = leave the checkpoint's own height (the production default); "
            "any other value is forced through backend_options['height_override']"
        ),
    )
    parser.add_argument("--output-dir", default=None, help="where the checkpoints go")
    parser.add_argument("--json-out", default=None)
    return parser.parse_args(argv)


def build_dataset(author_id: int):
    """The real training corpus of one author (crops along the baselines)."""
    from app.db import SessionLocal
    from app.htr.application.dataset_builder import TrainingDatasetBuilder
    from app.htr.infrastructure.kraken.lines import KrakenLineCropper
    from app.htr.infrastructure.page_repository import SqlAlchemyPageRepository
    from app.htr.infrastructure.storage import HTRStorage, PilLineCropper

    storage = HTRStorage(settings.htr_storage_dir)
    db = SessionLocal()
    try:
        return TrainingDatasetBuilder(
            page_repository=SqlAlchemyPageRepository(db),
            line_cropper=KrakenLineCropper(fallback=PilLineCropper()),
            crop_path_provider=storage.line_crop_path,
        ).build_for_author(author_id), storage
    finally:
        db.close()


def run_one(height: int, dataset, storage, work_dir: Path):
    from app.htr.factory import build_default_model_ref, build_training_config
    from app.htr.infrastructure.kraken.trainer import KrakenTrainer

    config = build_training_config()
    config.backend_options["height_override"] = height
    checkpoint = work_dir / f"height_{height}.safetensors"
    started = time.time()
    result = KrakenTrainer(work_dir=storage.training_work_dir()).train(
        build_default_model_ref(), dataset, str(checkpoint), config
    )
    metrics = {
        "requested": height,
        "line_height": result.training_metrics.get("line_height"),
        "line_height_override_from": result.training_metrics.get("line_height_override_from"),
        "val_cer": result.validation_metrics.get("cer"),
        "val_wer": result.validation_metrics.get("wer"),
        "base_cer": result.baseline_metrics.get("cer"),
        "base_wer": result.baseline_metrics.get("wer"),
        "seconds": round(time.time() - started, 1),
        "checkpoint": str(checkpoint),
    }
    label = "checkpoint" if height == 0 else f"forced {height}"
    logger.info(
        "height %s (%s): val CER %.4f WER %.4f | base CER %.4f | line_height=%s | %.0f s",
        height, label, metrics["val_cer"] or 0.0, metrics["val_wer"] or 0.0,
        metrics["base_cer"] or 0.0, metrics["line_height"], metrics["seconds"],
    )
    return metrics


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    args = parse_args(argv)

    dataset, storage = build_dataset(args.author_id)
    if not dataset.samples:
        print("нет подтверждённых страниц автора", file=sys.stderr)
        return 2
    work_dir = Path(args.output_dir or Path(settings.htr_storage_dir) / "training" / "height_ab")
    work_dir.mkdir(parents=True, exist_ok=True)
    print(f"датасет: {len(dataset.samples)} строк (хеш {dataset.dataset_hash[:12]})")

    rows = [run_one(height, dataset, storage, work_dir) for height in args.heights]

    print("\nвысота | фактическая | val CER | val WER | base CER")
    for row in rows:
        print(
            f"{row['requested']:>6} | {str(row['line_height']):>11} | "
            f"{row['val_cer']:.4f}  | {row['val_wer']:.4f}  | {row['base_cer']:.4f}"
        )
    best = min(rows, key=lambda row: row["val_cer"] or 1.0)
    print(f"\nлучшая высота: {best['requested']} (фактическая {best['line_height']})")

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(
                {"author_id": args.author_id, "heights": rows},
                ensure_ascii=False,
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
