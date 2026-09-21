"""Kraken implementation of HTRTrainer.

All Kraken-specific code is confined to this module; nothing outside the
infrastructure layer may import kraken.

NOTE (spike): the exact Python training API differs between Kraken releases
(RecognitionModel/KrakenTrainer moved and the `resize` values were renamed
add/union in newer versions). This adapter targets the ketos-style API of
kraken >= 4 and reads results defensively; pin the kraken version and verify
CUDA behaviour on the target GPU before production use.
"""
from __future__ import annotations

import logging
import random
import shutil
import tempfile
from pathlib import Path

from ...domain.entities import (
    ModelRef,
    TrainingConfig,
    TrainingDataset,
    TrainingRunResult,
)
from ...domain.errors import TrainingError
from ...domain.interfaces import HTRTrainer

logger = logging.getLogger(__name__)


class KrakenTrainer(HTRTrainer):
    def __init__(self, work_dir: str | None = None):
        self.work_dir = work_dir

    def train(
        self,
        base_model: ModelRef,
        dataset: TrainingDataset,
        output_model_path: str,
        config: TrainingConfig,
    ) -> TrainingRunResult:
        try:
            from kraken.lib.train import RecognitionModel
            from kraken.lib.train import KrakenTrainer as KrakenLightningTrainer
        except ImportError as exc:
            raise TrainingError(
                "kraken is not installed; install the 'kraken' package to enable "
                "HTR fine-tuning"
            ) from exc

        if not dataset.samples:
            raise TrainingError("Training dataset is empty")

        tmp_root = Path(tempfile.mkdtemp(prefix="kraken_train_", dir=self.work_dir))
        try:
            train_files, eval_files = self._prepare_ground_truth(dataset, config, tmp_root)
            accelerator, devices = self._device_settings(config.device)

            hyper_params = {
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "lrate": config.learning_rate,
            }
            model = RecognitionModel(
                hyper_params=hyper_params,
                output=str(tmp_root / "model"),
                model=base_model.path,
                training_data=train_files,
                evaluation_data=eval_files or None,
                partition=1.0 if eval_files else max(0.0, 1.0 - config.validation_split),
                format_type="path",
                resize="union",
            )
            trainer = KrakenLightningTrainer(
                accelerator=accelerator,
                devices=devices,
                max_epochs=config.epochs,
                enable_progress_bar=False,
            )
            trainer.fit(model)

            best_path = self._resolve_best_model(model, trainer, tmp_root)
            Path(output_model_path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(best_path, output_model_path)

            val_metrics = self._extract_validation_metrics(model, trainer)
            return TrainingRunResult(
                model_path=output_model_path,
                training_metrics={
                    "epochs": config.epochs,
                    "train_samples": len(train_files),
                    "eval_samples": len(eval_files),
                },
                validation_metrics=val_metrics,
                # kraken's evaluation split comes from the training corpus,
                # it is not an independent holdout
                holdout_used=False,
            )
        except TrainingError:
            raise
        except Exception as exc:
            raise TrainingError(f"Kraken training failed: {exc}") from exc
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)

    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_ground_truth(
        dataset: TrainingDataset, config: TrainingConfig, tmp_root: Path
    ) -> tuple[list[str], list[str]]:
        """Materialize kraken 'path' format: <name>.png + <name>.gt.txt pairs."""
        gt_dir = tmp_root / "gt"
        gt_dir.mkdir(parents=True, exist_ok=True)
        files: list[str] = []
        for sample in dataset.samples:
            src = Path(sample.image_path)
            if not src.is_file():
                raise TrainingError(f"Line crop is missing: {sample.image_path}")
            name = f"p{sample.page_id}_l{sample.line_id}{src.suffix}"
            dst = gt_dir / name
            shutil.copyfile(src, dst)
            dst.with_suffix("").with_suffix(".gt.txt").write_text(
                sample.transcription, encoding="utf-8"
            )
            files.append(str(dst))

        # deterministic validation split
        rng = random.Random(config.random_seed)
        shuffled = files[:]
        rng.shuffle(shuffled)
        eval_count = int(len(shuffled) * config.validation_split)
        if eval_count == 0 or len(shuffled) - eval_count < 1:
            return shuffled, []
        return shuffled[eval_count:], shuffled[:eval_count]

    @staticmethod
    def _device_settings(device: str) -> tuple[str, int | list[int]]:
        if device.startswith("cuda"):
            _, _, index = device.partition(":")
            return "gpu", [int(index)] if index else 1
        return "cpu", 1

    @staticmethod
    def _resolve_best_model(model, trainer, tmp_root: Path) -> str:
        for candidate in (
            getattr(model, "best_model", None),
            getattr(getattr(trainer, "checkpoint_callback", None), "best_model_path", None),
        ):
            if candidate and Path(candidate).is_file():
                return str(candidate)
        artifacts = sorted(tmp_root.glob("model*.mlmodel"))
        if artifacts:
            return str(artifacts[-1])
        raise TrainingError("Kraken produced no model artifact")

    @staticmethod
    def _extract_validation_metrics(model, trainer) -> dict | None:
        accuracy = getattr(model, "best_metric", None)
        if accuracy is None:
            metrics = getattr(trainer, "callback_metrics", {}) or {}
            raw = metrics.get("val_accuracy")
            accuracy = float(raw) if raw is not None else None
        if accuracy is None:
            return None
        # kraken reports character accuracy on its evaluation split
        return {"val_accuracy": float(accuracy), "cer": 1.0 - float(accuracy)}
