"""Orchestrates fine-tuning of per-author HTR models.

Business rules enforced here:
- the base model is ALWAYS the default model, never a previous custom model;
- the dataset contains ALL confirmed pages of the author at training time;
- training only starts once the confirmed corpus reaches a configurable size
  (per-author lines/words threshold); below it the page stays confirmed and the
  result is INSUFFICIENT_DATA instead of a failure;
- every successful run creates a new model version, old versions are kept;
- a new version becomes active only after a fully successful run;
- on any failure the previously active model stays active.

Contains no ML-framework code and no HTTP code, so it can later be moved
behind a background worker unchanged.
"""
from __future__ import annotations

import logging
import threading
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
from .readiness import TrainingReadinessPolicy

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
        min_training_lines: int = 50,
        min_training_words: int = 0,
        environment: dict[str, Any] | None = None,
    ):
        self.dataset_builder = dataset_builder
        self.model_repository = model_repository
        self.training_run_repository = training_run_repository
        self.trainer = trainer
        self.config = config
        self.readiness = TrainingReadinessPolicy(
            min_lines=min_training_lines, min_words=min_training_words
        )
        self.environment = environment or {}
        # Training is synchronous, so two confirmations arriving together would
        # otherwise train the same author concurrently (and fight over the GPU).
        # The guard is process-local; a worker queue replaces it later.
        self._author_locks: dict[int, threading.Lock] = {}
        self._author_locks_guard = threading.Lock()

    def _author_lock(self, author_id: int) -> threading.Lock:
        with self._author_locks_guard:
            return self._author_locks.setdefault(author_id, threading.Lock())

    def train_author(self, author_id: int) -> TrainingResult:
        lock = self._author_lock(author_id)
        if not lock.acquire(blocking=False):
            message = f"A training run for author {author_id} is already in progress"
            logger.info("HTR training skipped: %s", message)
            return TrainingResult(
                outcome=TrainingOutcome.BUSY,
                author_id=author_id,
                message=message,
            )
        try:
            return self._train_author_locked(author_id)
        finally:
            lock.release()

    def _train_author_locked(self, author_id: int) -> TrainingResult:
        logger.info("HTR training requested for author_id=%s", author_id)
        # a version left in TRAINING by a crash/restart can never be activated
        stale = self.model_repository.fail_stale_training(
            author_id, "training was interrupted before completion"
        )
        if stale:
            logger.warning(
                "HTR: marked %s stale TRAINING model version(s) as FAILED for author_id=%s",
                stale, author_id,
            )
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

        readiness = self.readiness.evaluate(dataset)
        if not readiness.ready:
            message = f"Author {author_id}: {readiness.reason}"
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
                lines_collected=readiness.lines,
                lines_required=readiness.min_lines,
                words_collected=readiness.words,
                words_required=readiness.min_words,
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
            improved, reason = self._beats_baseline(run_result)
            if not improved:
                # Training worked, but the artifact is worse than the model it
                # would replace: publishing it would degrade recognition.
                logger.warning(
                    "HTR training rejected: author_id=%s run_id=%s version=v%s %s",
                    author_id, run_id, version.version, reason,
                )
                self.model_repository.mark_failed(version.id, reason)
                self.training_run_repository.finish_run(
                    run_id,
                    TrainingRunStatus.FAILED.value,
                    metrics=metrics,
                    error=reason,
                    model_version_id=version.id,
                )
                return TrainingResult(
                    outcome=TrainingOutcome.NO_IMPROVEMENT,
                    author_id=author_id,
                    message=reason,
                    metrics=metrics,
                    dataset_hash=dataset.dataset_hash,
                    training_run_id=run_id,
                    lines_collected=readiness.lines,
                    lines_required=readiness.min_lines,
                    words_collected=readiness.words,
                    words_required=readiness.min_words,
                )
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
            # re-read so the response carries the post-activation status
            version = self.model_repository.get_version(version.id) or version
            version.metrics = metrics
            return TrainingResult(
                outcome=TrainingOutcome.SUCCESS,
                author_id=author_id,
                model_version=version,
                metrics=metrics,
                dataset_hash=dataset.dataset_hash,
                training_run_id=run_id,
                lines_collected=readiness.lines,
                lines_required=readiness.min_lines,
                words_collected=readiness.words,
                words_required=readiness.min_words,
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
        validation = run_result.validation_metrics
        metrics: dict[str, Any] = {
            "training": run_result.training_metrics,
            "validation": validation,
            "baseline": run_result.baseline_metrics,
            "holdout_used": run_result.holdout_used,
            "note": holdout_note,
        }
        # surface the headline numbers for the model metadata consumers
        if validation:
            if "cer" in validation:
                metrics["cer"] = validation["cer"]
            if "wer" in validation:
                metrics["wer"] = validation["wer"]
        elif "best_val_cer" in run_result.training_metrics:
            metrics["cer"] = run_result.training_metrics["best_val_cer"]
        return metrics

    @staticmethod
    def _beats_baseline(run_result) -> tuple[bool, str]:
        """A fine-tune may only be activated if it beats the base model.

        Both are scored on the same held-out split (see KrakenTrainer). The
        character error rate is the primary criterion; a run whose metrics
        cannot be compared is never activated.
        """
        validation = run_result.validation_metrics
        candidate_cer = validation.get("cer") if validation else None
        if candidate_cer is None and "best_val_cer" in run_result.training_metrics:
            candidate_cer = run_result.training_metrics["best_val_cer"]

        baseline = run_result.baseline_metrics
        baseline_cer = baseline.get("cer") if baseline else None

        if candidate_cer is None:
            return False, (
                "the fine-tuned model has no validation metrics, so it cannot be "
                "compared with the base model"
            )
        if baseline_cer is None:
            return False, (
                "the base model has no validation metrics on the same split, so "
                "the fine-tune cannot be verified"
            )
        if candidate_cer < baseline_cer:
            return True, ""
        return False, (
            f"fine-tuned CER {candidate_cer:.4f} did not improve on the base "
            f"model's CER {baseline_cer:.4f} on the same validation split; "
            "the previous model stays active"
        )
