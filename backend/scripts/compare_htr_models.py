"""Compare HTR models on the same held-out split of an author's corpus.

Answers the question "did this fine-tune actually improve recognition?" by
running kraken's test pass for several model files over the *identical*,
deterministically chosen validation lines (the same split KrakenTrainer uses
during training, i.e. ``random.Random(htr_random_seed)``).

Usage (from the ``backend`` directory)::

    .venv/bin/python scripts/compare_htr_models.py --author-id 1 \
        --model default=htr_storage/models/default/ppocrv6_medium.safetensors \
        --model v2=htr_storage/models/author_1/v2/model.safetensors

Without ``--model`` the default model and the author's ACTIVE model are used.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings  # noqa: E402
from app.db import SessionLocal  # noqa: E402
from app.htr.application.dataset_builder import TrainingDatasetBuilder  # noqa: E402
from app.htr.domain.entities import TrainingConfig  # noqa: E402
from app.htr.infrastructure.kraken.trainer import KrakenTrainer  # noqa: E402
from app.htr.infrastructure.model_repository import SqlAlchemyModelRepository  # noqa: E402
from app.htr.infrastructure.page_repository import SqlAlchemyPageRepository  # noqa: E402
from app.htr.infrastructure.storage import HTRStorage, PilLineCropper  # noqa: E402


def build_eval_split(author_id: int, config: TrainingConfig, work_root: Path, limit: int | None):
    storage = HTRStorage(settings.htr_storage_dir)
    builder = TrainingDatasetBuilder(
        page_repository=SqlAlchemyPageRepository(SessionLocal()),
        line_cropper=PilLineCropper(),
        crop_path_provider=storage.line_crop_path,
    )
    dataset = builder.build_for_author(author_id)
    _, eval_files = KrakenTrainer._prepare_ground_truth(dataset, config, work_root)
    if not eval_files:
        raise SystemExit("no validation lines: not enough confirmed data")
    if limit:
        eval_files = eval_files[:limit]
    print(
        f"corpus: {len(dataset.samples)} lines, dataset_hash={dataset.dataset_hash[:12]}…, "
        f"validation split: {len(eval_files)} lines"
    )
    return eval_files


def evaluate(name: str, model_path: str, config: TrainingConfig, eval_files, work_root: Path):
    trainer = KrakenTrainer(work_dir=str(work_root))
    trainer._configure_runtime(config)
    convert_models, _seed, _ckpt, lightning_cls = trainer._import_kraken()
    module_cls = trainer._resolve_module_class(Path(model_path))
    arch = module_cls._arch
    model_config = trainer._build_model_config(module_cls, arch, config, work_root)
    data_module = module_cls._data_module_class(
        trainer._data_config(module_cls, config, [], [], test_files=eval_files)
    )
    accelerator, devices = trainer._device_settings(config.device)
    lightning = lightning_cls(
        accelerator=accelerator,
        devices=devices,
        precision=config.backend_options.get("precision", "32-true"),
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    model = module_cls.load_from_weights(model_path, config=model_config)
    metrics = lightning.test(model, data_module)
    result = {
        "cer": 1.0 - float(metrics.cer),
        "wer": 1.0 - float(metrics.wer),
        "char_accuracy": float(metrics.cer),
        "word_accuracy": float(metrics.wer),
    }
    print(
        f"{name:>10} ({arch}): CER={result['cer']:.4f} WER={result['wer']:.4f} "
        f"char_acc={result['char_accuracy']:.4f} word_acc={result['word_accuracy']:.4f}"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-id", type=int, required=True)
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="model to evaluate; repeatable. Defaults to default + active author model.",
    )
    parser.add_argument("--limit", type=int, default=0, help="use only the first N validation lines")
    args = parser.parse_args()

    config = TrainingConfig(
        device=settings.htr_device,
        random_seed=settings.htr_random_seed,
        validation_split=settings.htr_validation_split,
        backend_options={
            "height": settings.htr_training_height,
            "max_width": settings.htr_training_max_width,
            "precision": settings.htr_training_precision,
            "num_workers": 0,
        },
    )

    models: list[tuple[str, str]] = []
    for spec in args.model:
        name, _, path = spec.partition("=")
        models.append((name or Path(path).stem, path))
    if not models:
        db = SessionLocal()
        try:
            repo = SqlAlchemyModelRepository(db, HTRStorage(settings.htr_storage_dir),
                                             _default_ref())
            models.append(("default", repo.get_default_model().path))
            active = repo.get_active_model(args.author_id)
            if active is not None:
                models.append((f"v{active.version}", active.file_path))
        finally:
            db.close()

    work_root = Path(tempfile.mkdtemp(prefix="htr_compare_"))
    try:
        eval_files = build_eval_split(
            args.author_id, config, work_root, args.limit or None
        )
        results = {
            name: evaluate(name, path, config, eval_files, work_root)
            for name, path in models
        }
    finally:
        shutil.rmtree(work_root, ignore_errors=True)

    if len(results) > 1:
        print("\ncomparison vs. the first model:")
        baseline = next(iter(results.values()))
        for name, value in list(results.items())[1:]:
            print(
                f"  {name}: ΔCER={value['cer'] - baseline['cer']:+.4f} "
                f"ΔWER={value['wer'] - baseline['wer']:+.4f} "
                f"({'better' if value['cer'] < baseline['cer'] else 'worse'})"
            )
    return 0


def _default_ref():
    from app.htr.domain.entities import ModelRef

    return ModelRef(id=settings.htr_default_model_id, path=settings.htr_default_model_path)


if __name__ == "__main__":
    raise SystemExit(main())
