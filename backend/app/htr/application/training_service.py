"""Orchestrates fine-tuning of per-author HTR models.

Business rules enforced here:
- the base model is ALWAYS the default model, never a previous custom model;
- the dataset contains ALL confirmed pages of the author at training time;
- every successful run creates a new model version, old versions are kept;
- a new version becomes active only after a fully successful run;
- on any failure the previously active model stays active.

Contains no ML-framework code and no HTTP code, so it can later be moved
behind a background worker unchanged.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from ..domain.entities import (
    TrainingConfig,
    TrainingOutcome,
    TrainingResult,
    TrainingRunStatus,
)
from ..domain.errors import DatasetBuildError
from ..domain.interfaces import (
    DatasetBuilder,
    HTRTrainer,
    ModelRepository,
    TrainingRunRepository,
)

logger = logging.getLogger(__name__)

NO_HOLDOUT_NOTE = (
    "No separate holdout set: metrics are computed on training/validation data "
    "and must not be read as an objective estimate for unseen pages."
)


class HandwritingTrainingService:
    def __init__(
        self,
        dataset_builder: DatasetBuilder,
        model_repository: ModelRepository,
        training_run_repository: TrainingRunRepository,
        trainer: HTRTrainer,
        config: TrainingConfig,
        min_training_samples: int = 1,
        environment: dict[str, Any] | None = None,
    ):
        self.dataset_builder = dataset_builder
        self.model_repository = model_repository
        self.training_run_repository = training_run_repository
        self.trainer = trainer
        self.config = config
        self.min_training_samples = min_training_samples
        self.environment = environment or {}

    def train_author(self, author_id: int) -> TrainingResult:
        logger.info("HTR training requested for author_id=%s", author_id)
        try:
            dataset = self.dataset_builder.build_for_author(author_id)
        except DatasetBuildError as exc:
            logger.exception("HTR dataset build failed for author_id=%s", author_id)
            run_id = self.training_run_repository.start_run(author_id, "")
            self.training_run_repository.finish_run(
                run_id, TrainingRunStatus.FAILED.value, error=str(exc)
            )
            return TrainingResult(
                outcome=TrainingOutcome.FAILED,
                author_id=author_id,
                message=f"Dataset preparation failed: {exc}",
                training_run_id=run_id,
            )

        if len(dataset.samples) < self.min_training_samples:
            message = (
                f"Not enough confirmed training data for author {author_id}: "
                f"{len(dataset.samples)} sample(s), need at least {self.min_training_samples}"
            )
            logger.info("HTR training skipped: %s", message)
            run_id = self.training_run_repository.start_run(author_id, dataset.dataset_hash)
            self.training_run_repository.finish_run(
                run_id, TrainingRunStatus.INSUFFICIENT_DATA.value, error=message
            )
            return TrainingResult(
                outcome=TrainingOutcome.INSUFFICIENT_DATA,
                author_id=author_id,
                message=message,
                dataset_hash=dataset.dataset_hash,
                training_run_id=run_id,
            )

        # Mandatory rule: always fine-tune from the default model.
        base_model = self.model_repository.get_default_model()
        run_id = self.training_run_repository.start_run(author_id, dataset.dataset_hash)
        version = self.model_repository.create_version(
            author_id=author_id,
            base_model_id=base_model.id,
            training_config=self.config.to_dict(),
            dataset_hash=dataset.dataset_hash,
            environment=self.environment,
        )
        logger.info(
            "HTR training started: author_id=%s run_id=%s version=v%s base=%s "
            "dataset_hash=%s samples=%s",
            author_id, run_id, version.version, base_model.id,
            dataset.dataset_hash, len(dataset.samples),
        )
        started = time.monotonic()
        try:
            run_result = self.trainer.train(
                base_model=base_model,
                dataset=dataset,
                output_model_path=version.file_path,
                config=self.config,
            )
            metrics = self._build_metrics(run_result)
            self.model_repository.mark_ready(version.id, run_result.model_path, metrics)
            self.model_repository.activate_model(author_id, version.id)
            self.training_run_repository.finish_run(
                run_id,
                TrainingRunStatus.SUCCEEDED.value,
                metrics=metrics,
                model_version_id=version.id,
            )
            duration = time.monotonic() - started
            logger.info(
                "HTR training succeeded: author_id=%s run_id=%s version=v%s "
                "duration=%.1fs metrics=%s",
                author_id, run_id, version.version, duration, metrics,
            )
            version.metrics = metrics
            return TrainingResult(
                outcome=TrainingOutcome.SUCCESS,
                author_id=author_id,
                model_version=version,
                metrics=metrics,
                dataset_hash=dataset.dataset_hash,
                training_run_id=run_id,
            )
        except Exception as exc:
            duration = time.monotonic() - started
            logger.exception(
                "HTR training failed: author_id=%s run_id=%s version=v%s duration=%.1fs",
                author_id, run_id, version.version, duration,
            )
            # Never activate a partially trained model; previous active stays.
            self.model_repository.mark_failed(version.id, str(exc))
            self.training_run_repository.finish_run(
                run_id,
                TrainingRunStatus.FAILED.value,
                error=str(exc),
                model_version_id=version.id,
            )
            return TrainingResult(
                outcome=TrainingOutcome.FAILED,
                author_id=author_id,
                message=str(exc),
                dataset_hash=dataset.dataset_hash,
                training_run_id=run_id,
            )

    def _build_metrics(self, run_result) -> dict[str, Any]:
        holdout_note = run_result.note or (None if run_result.holdout_used else NO_HOLDOUT_NOTE)
        return {
            "training": run_result.training_metrics,
            "validation": run_result.validation_metrics,
            "holdout_used": run_result.holdout_used,
            "note": holdout_note,
        }
