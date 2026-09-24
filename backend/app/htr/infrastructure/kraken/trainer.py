"""Kraken 7 implementation of HTRTrainer.

All Kraken-specific code is confined to this module (and to
``infrastructure/kraken/recognizer.py``); nothing outside the infrastructure
layer may import kraken. Imports are lazy so the application keeps working
when the optional HTR backend is not installed.

Verified against kraken 7.1.1. Flow (mirrors ``ketos train``):

    image + .gt.txt pairs (path format)
      -> architecture auto-detected from the base weights
      -> training config object + datamodule
      -> PPOCRv6RecognitionModel.load_from_weights(default model)   # --load
      -> KrakenTrainer.fit()
      -> best checkpoint -> convert to safetensors
      -> optional test pass on the held-out validation split -> CER/WER

The base model is always the default model handed in by the application
service; this adapter never reaches for a previous custom model.
"""
from __future__ import annotations

import logging
import math
import os
import random
import shutil
import tempfile
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

from ...domain.entities import (
    ModelRef,
    TrainingConfig,
    TrainingDataset,
    TrainingRunResult,
)
from ...domain.errors import TrainingError
from ...domain.interfaces import HTRTrainer

logger = logging.getLogger(__name__)

WEIGHTS_FORMAT = "safetensors"
DEFAULT_ARCH = "vgsl"
ARCH_ENTRY_POINT = "kraken.archs.recognition"

VALIDATION_NOTE = (
    "Metrics are computed on a validation split taken from the author's own "
    "confirmed pages (no independent holdout), so they are not an objective "
    "estimate for unseen pages."
)


class KrakenTrainer(HTRTrainer):
    def __init__(self, work_dir: str | None = None, arch: str | None = None):
        # ``work_dir`` keeps temporary training files inside HTR storage.
        # ``arch`` overrides architecture detection (normally detected from
        # the base model's metadata).
        self.work_dir = work_dir
        self.arch = arch

    # ------------------------------------------------------------------

    def train(
        self,
        base_model: ModelRef,
        dataset: TrainingDataset,
        output_model_path: str,
        config: TrainingConfig,
    ) -> TrainingRunResult:
        if not dataset.samples:
            raise TrainingError("Training dataset is empty")
        base_path = Path(base_model.path)
        if not base_path.is_file():
            raise TrainingError(
                f"Base model {base_model.path} does not exist; fetch the default "
                "model with `python scripts/download_htr_model.py`"
            )

        (
            convert_models,
            seed_everything,
            checkpoint_cls,
            lightning_trainer_cls,
            freeze_callback_cls,
        ) = self._import_kraken()
        self._configure_runtime(config)
        module_cls = self._training_module_class(
            self._resolve_module_class(base_path), config
        )
        arch = getattr(module_cls, "_arch", None) or DEFAULT_ARCH
        logger.info(
            "Kraken training: arch=%s base=%s samples=%s device=%s compile=%s",
            arch, base_model.path, len(dataset.samples), config.device,
            config.backend_options.get("compile", False),
        )

        tmp_root = Path(tempfile.mkdtemp(prefix="kraken_train_", dir=self.work_dir))
        try:
            train_files, eval_files = self._prepare_ground_truth(dataset, config, tmp_root)
            seed_everything(config.random_seed, workers=True)

            model_config = self._build_model_config(module_cls, arch, config, tmp_root)
            data_module = self._build_data_module(
                module_cls, config, train_files, eval_files
            )

            checkpoint_path = tmp_root / "checkpoints"
            checkpoint_callback = checkpoint_cls(
                dirpath=str(checkpoint_path),
                save_top_k=1,
                monitor="val_metric",
                mode="max",
                auto_insert_metric_name=False,
                filename="checkpoint_{epoch:02d}-{val_metric:.4f}",
            )
            accelerator, devices = self._runtime_device(config)
            # freeze the backbone (everything but the codec projection) for the
            # first steps: the freshly resized head would otherwise drag the
            # pretrained features away (catastrophic forgetting on a tiny corpus)
            freeze_steps = self._freeze_backbone_steps(config, len(train_files))
            callbacks = [checkpoint_callback]
            if freeze_steps > 0 or config.backend_options.get("freeze_bn", True):
                callbacks.append(
                    freeze_callback_cls(
                        freeze_steps,
                        freeze_batch_norm=config.backend_options.get("freeze_bn", True),
                    )
                )
            trainer = lightning_trainer_cls(
                accelerator=accelerator,
                devices=devices,
                precision=config.backend_options.get("precision", "32-true"),
                max_epochs=config.epochs if config.epochs > 0 else -1,
                min_epochs=config.min_epochs,
                enable_progress_bar=False,
                enable_model_summary=False,
                # global seeding above covers reproducibility; deterministic
                # mode is avoided because some CUDA kernels do not support it
                deterministic=False,
                num_sanity_val_steps=0,
                callbacks=callbacks,
                val_check_interval=1.0,
            )

            with trainer.init_module(empty_init=False):
                model = module_cls.load_from_weights(str(base_path), config=model_config)

            height_metrics = self._apply_height_override(model, config)
            trainer.fit(model, data_module)

            best_checkpoint = checkpoint_callback.best_model_path
            if not best_checkpoint or not Path(best_checkpoint).is_file():
                raise TrainingError(
                    "Kraken produced no best checkpoint; the validation split is "
                    "probably empty (need at least 2 line samples)"
                )
            best_score = getattr(checkpoint_callback, "best_model_score", None)

            # Checkpoints are kraken-internal; recognition needs the weights file.
            Path(output_model_path).parent.mkdir(parents=True, exist_ok=True)
            written = convert_models(
                [best_checkpoint], output_model_path, weights_format=WEIGHTS_FORMAT
            )
            model_path = str(written)
            if not Path(model_path).is_file():
                raise TrainingError(f"Kraken did not write a model to {model_path}")

            validation_metrics = self._test_model(
                module_cls, model_config, config, train_files, eval_files,
                model_path, trainer,
            )
            # same split, untouched base model: the application layer uses this
            # to reject a fine-tune that made recognition worse
            baseline_metrics = self._test_model(
                module_cls, model_config, config, train_files, eval_files,
                str(base_path), trainer,
            )
            steps_per_epoch = max(1, math.ceil(len(train_files) / max(1, config.batch_size)))
            training_metrics: dict[str, Any] = {
                "architecture": arch,
                "epochs": config.epochs,
                "train_samples": len(train_files),
                "eval_samples": len(eval_files),
                "steps_per_epoch": steps_per_epoch,
                "freeze_backbone_steps": freeze_steps,
                **height_metrics,
                "freeze_batch_norm": config.backend_options.get("freeze_bn", True),
                "aux_nrtr": config.backend_options.get("aux_nrtr", False),
            }
            if freeze_steps >= steps_per_epoch * max(1, config.epochs):
                logger.warning(
                    "HTR training: the backbone stays frozen for the whole run "
                    "(%s steps requested, %s available) — only the output layer "
                    "will learn",
                    freeze_steps, steps_per_epoch * max(1, config.epochs),
                )
            if best_score is not None:
                score = float(best_score)
                # kraken's val_metric is character accuracy
                training_metrics["best_val_accuracy"] = score
                training_metrics["best_val_cer"] = 1.0 - score

            if validation_metrics is None and "best_val_cer" not in training_metrics:
                # an unverifiable artifact must never be activated
                raise TrainingError(
                    "Kraken produced no validation metrics, so the model cannot "
                    "be verified; refusing to publish it"
                )

            return TrainingRunResult(
                model_path=model_path,
                training_metrics=training_metrics,
                validation_metrics=validation_metrics,
                baseline_metrics=baseline_metrics,
                holdout_used=False,
                note=VALIDATION_NOTE,
            )
        except TrainingError:
            raise
        except Exception as exc:
            raise TrainingError(f"Kraken training failed: {exc}") from exc
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)

    # ------------------------------------------------------------------
    # kraken wiring
    # ------------------------------------------------------------------

    @staticmethod
    def _import_kraken():
        try:
            from lightning.pytorch import seed_everything
            from lightning.pytorch.callbacks import ModelCheckpoint
            from kraken.models.convert import convert_models
            from kraken.train import KrakenTrainer as KrakenLightningTrainer

            from .callbacks import FreezeBackboneForSteps
        except ImportError as exc:
            raise TrainingError(
                "the kraken training backend is not installed; install the "
                "optional HTR backend (see app/htr/README.md) to enable training"
            ) from exc
        return (
            convert_models,
            seed_everything,
            ModelCheckpoint,
            KrakenLightningTrainer,
            FreezeBackboneForSteps,
        )

    @staticmethod
    def _apply_height_override(model, config: TrainingConfig) -> dict[str, Any]:
        """Force a line height that differs from the checkpoint's own.

        kraken always overrides ``config.height`` with the height recorded in the
        checkpoint, so the setting alone cannot change it. The override rewrites
        the model's expected input geometry after loading, which is what the
        data module and the network read.
        """
        requested = int(config.backend_options.get("height_override", 0) or 0)
        declared = int(getattr(model, "height", 0) or 0)
        metrics: dict[str, Any] = {"line_height": declared}
        if not requested or requested == declared:
            return metrics
        net = getattr(model, "net", None)
        if net is None or not getattr(net, "input", None) or len(net.input) != 4:
            logger.warning(
                "HTR training: cannot force line height %s on %s", requested, type(net).__name__
            )
            return metrics
        batch, channels, _, width = net.input
        net.input = (batch, channels, requested, width)
        model.height = requested
        metrics["line_height"] = requested
        metrics["line_height_override_from"] = declared
        logger.info(
            "HTR training: line height forced to %s (checkpoint declares %s)",
            requested, declared,
        )
        return metrics

    def _freeze_backbone_steps(self, config: TrainingConfig, train_samples: int) -> int:
        """How many optimizer steps to keep the backbone frozen for.

        ``freeze_backbone`` is expressed in steps, not samples: kraken's help
        text says "samples" but its callback compares against
        ``trainer.global_step``, and we follow the implementation.

        * ``0`` — no freezing at all;
        * ``N > 0`` — freeze for the first N steps;
        * ``N < 0`` (default) — freeze for the first epoch, which is long enough
          for a freshly resized output layer to settle and short enough to
          fine-tune the pretrained features afterwards.
        """
        requested = int(config.backend_options.get("freeze_backbone", 0) or 0)
        if requested > 0:
            return requested
        if requested == 0:
            return 0
        steps_per_epoch = math.ceil(train_samples / max(1, config.batch_size))
        return max(1, steps_per_epoch)

    @staticmethod
    def _configure_runtime(config: TrainingConfig) -> None:
        """Apply process-wide torch settings before kraken builds the trainer.

        ``torch.compile`` (inductor) is called unconditionally by kraken's
        ``on_fit_start``. On a small fine-tuning corpus the codegen pass can
        take longer than the training itself (minutes of 100% CPU with an idle
        GPU), so it is disabled by default. ``TORCHDYNAMO_DISABLE`` is read when
        ``torch.compile`` is invoked, i.e. the value set here is honoured.
        """
        import torch

        if config.backend_options.get("compile", False):
            os.environ.pop("TORCHDYNAMO_DISABLE", None)
        else:
            os.environ["TORCHDYNAMO_DISABLE"] = "1"

        precision = config.backend_options.get("matmul_precision")
        if precision and (config.device or "").startswith(("cuda", "gpu")):
            try:
                torch.set_float32_matmul_precision(precision)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Could not set matmul precision %r: %s", precision, exc)

    @staticmethod
    def _training_module_class(module_cls, config: TrainingConfig):
        """Swap in the CTC-only module unless the auxiliary head is wanted.

        The base checkpoint of this project is an inference model without an
        NRTR head, so kraken would train one from random weights (see
        infrastructure/kraken/modules.py).
        """
        arch = getattr(module_cls, "_arch", None) or DEFAULT_ARCH
        if arch != "ppocrv6" or config.backend_options.get("aux_nrtr", False):
            return module_cls
        from .modules import CtcFineTunePPOCRv6

        return CtcFineTunePPOCRv6

    def _resolve_module_class(self, base_model_path: Path):
        """Pick the kraken LightningModule for the base weights' architecture."""
        from kraken.models.convert import find_weights_archs

        detected: set[str] | None = None
        try:
            detected = find_weights_archs(str(base_model_path), task="recognition")
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Could not detect architecture of %s: %s", base_model_path, exc)

        arch = self.arch
        if detected:
            if arch is None:
                if len(detected) > 1:
                    raise TrainingError(
                        f"{base_model_path} contains several architectures "
                        f"{sorted(detected)}; configure the one to train"
                    )
                arch = next(iter(detected))
            elif arch not in detected:
                raise TrainingError(
                    f"configured architecture {arch!r} does not match the base "
                    f"model {base_model_path} ({sorted(detected)})"
                )
        arch = arch or DEFAULT_ARCH

        for entry_point in importlib_metadata.entry_points(group=ARCH_ENTRY_POINT):
            if entry_point.name == arch:
                return entry_point.load()
        available = sorted(
            ep.name for ep in importlib_metadata.entry_points(group=ARCH_ENTRY_POINT)
        )
        raise TrainingError(
            f"kraken has no recognition architecture {arch!r}; available: {available}"
        )

    def _build_model_config(self, module_cls, arch: str, config: TrainingConfig, tmp_root: Path):
        options = config.backend_options
        accelerator, devices = self._runtime_device(config)
        kwargs: dict[str, Any] = {
            "epochs": config.epochs if config.epochs > 0 else -1,
            "batch_size": config.batch_size,
            "lrate": config.learning_rate,
            "weight_decay": options.get("weight_decay", 0.01),
            "schedule": options.get("schedule", "cosine"),
            "warmup": options.get("warmup", 0),
            "cos_min_lr": options.get("cos_min_lr", 1e-6),
            "quit": options.get("quit", "fixed" if config.epochs > 0 else "early"),
            "checkpoint_path": str(tmp_root / "checkpoints"),
            "weights_format": WEIGHTS_FORMAT,
            "resize": options.get("resize", "union"),
            "accelerator": accelerator,
            "device": devices,
            "precision": options.get("precision", "32-true"),
        }
        if arch == "ppocrv6":
            kwargs["variant"] = options.get("variant", "medium")
            kwargs["height"] = options.get("height", 96)
        elif arch == "vgsl" and options.get("spec"):
            kwargs["spec"] = options["spec"]
        return module_cls._config_class(**kwargs)

    def _build_data_module(self, module_cls, config: TrainingConfig, train_files, eval_files):
        return module_cls._data_module_class(self._data_config(module_cls, config, train_files, eval_files))

    def _data_config(self, module_cls, config: TrainingConfig, train_files, eval_files, test_files=None):
        options = config.backend_options
        kwargs: dict[str, Any] = {
            "training_data": list(train_files),
            # an explicit evaluation set disables kraken's random partition;
            # our split is already deterministic (random_seed)
            "evaluation_data": list(eval_files) or None,
            "partition": 1.0 if eval_files else 1.0 - config.validation_split,
            "format_type": "path",
            # the crops are dewarped baseline strips, not boxes: without this
            # kraken records seg_type=bbox and flips centerline normalization
            "linetype": options.get("linetype", "baselines"),
            "num_workers": options.get("num_workers", 0),
            "augment": options.get("augment", False),
            "padding": options.get("padding", 16),
            # the user's transcription is ground truth: no whitespace or
            # spelling/case/punctuation changes. NFD is canonical equivalence,
            # not a content change, and is required because the default model's
            # codec stores decomposed sequences (see htr_training_normalization).
            "normalize_whitespace": options.get("normalize_whitespace", False),
            "bidi_reordering": options.get("bidi_reordering", True),
            "normalization": options.get("normalization", "NFD"),
        }
        if test_files is not None:
            # a test datamodule must not receive training data: kraken only
            # builds ``test_set`` when training_data is empty
            kwargs["training_data"] = None
            kwargs["evaluation_data"] = None
            kwargs["test_data"] = list(test_files)
        if getattr(module_cls, "_arch", None) == "ppocrv6":
            kwargs["max_width"] = options.get("max_width", 2560)
        return module_cls._data_config_class(**kwargs)

    def _test_model(
        self,
        module_cls,
        model_config,
        config: TrainingConfig,
        train_files,
        eval_files,
        model_path: str,
        trainer,
    ) -> dict[str, Any] | None:
        """CER/WER of one model on the held-out validation lines."""
        if not eval_files:
            return None
        try:
            data_module = module_cls._data_module_class(
                self._data_config(module_cls, config, train_files, eval_files, test_files=eval_files)
            )
            scoring_model = module_cls.load_from_weights(model_path, config=model_config)
            metrics = trainer.test(scoring_model, data_module)
            # kraken names these ``cer``/``wer`` but stores accuracies
            char_accuracy = float(metrics.cer)
            word_accuracy = float(metrics.wer)
            return {
                "cer": 1.0 - char_accuracy,
                "wer": 1.0 - word_accuracy,
                "char_accuracy": char_accuracy,
                "word_accuracy": word_accuracy,
                "lines": len(eval_files),
            }
        except Exception as exc:  # metrics must not invalidate a trained model
            logger.warning("Kraken metrics for %s could not be computed: %s", model_path, exc)
            return None

    # ------------------------------------------------------------------
    # data
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
            name = f"p{sample.page_id}_l{sample.line_id}{src.suffix or '.png'}"
            dst = gt_dir / name
            shutil.copyfile(src, dst)
            dst.with_suffix("").with_suffix(".gt.txt").write_text(
                sample.transcription, encoding="utf-8"
            )
            files.append(str(dst))

        if len(files) < 2:
            raise TrainingError(
                "Kraken fine-tuning needs at least 2 line samples to form a "
                "training/validation split"
            )

        # deterministic validation split
        rng = random.Random(config.random_seed)
        shuffled = files[:]
        rng.shuffle(shuffled)
        eval_count = int(len(shuffled) * config.validation_split)
        if config.validation_split > 0:
            eval_count = max(1, eval_count)
        eval_count = min(eval_count, len(shuffled) - 1)
        return shuffled[eval_count:], shuffled[:eval_count]

    @staticmethod
    def _device_settings(device: str) -> tuple[str, Any]:
        """Map a configured device string onto Lightning's (accelerator, devices)."""
        value = (device or "auto").strip()
        if value == "auto":
            return "auto", "auto"
        if value in ("cpu", "mps"):
            return value, "auto"
        if ":" in value:
            kind, _, index = value.partition(":")
            return ("gpu" if kind == "cuda" else kind), [int(index)]
        if value in ("cuda", "gpu"):
            return "gpu", "auto"
        return value, "auto"

    @staticmethod
    def _runtime_device(config: TrainingConfig) -> tuple[str, Any]:
        """Device actually handed to Lightning (CPU if CUDA is unusable)."""
        from .recognizer import available_lightning_device

        return available_lightning_device(config.device)
