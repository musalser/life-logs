"""Composition root of the HTR module: wires settings + DB session into services."""
from __future__ import annotations

from importlib import metadata

from sqlalchemy.orm import Session

from ..config import settings
from .application.confidence import ConfidencePolicy
from .application.dataset_builder import TrainingDatasetBuilder
from .application.metrics import MetricsEvaluator
from .application.page_service import HandwritingPageService
from .application.training_service import HandwritingTrainingService
from .domain.entities import ModelRef, TrainingConfig
from .infrastructure.kraken.recognizer import KrakenRecognizer
from .infrastructure.kraken.trainer import KrakenTrainer
from .infrastructure.model_repository import (
    SqlAlchemyModelRepository,
    SqlAlchemyTrainingRunRepository,
)
from .infrastructure.page_repository import SqlAlchemyPageRepository
from .infrastructure.storage import HTRStorage, PilLineCropper


def build_training_config() -> TrainingConfig:
    return TrainingConfig(
        device=settings.htr_device,
        epochs=settings.htr_epochs,
        batch_size=settings.htr_batch_size,
        learning_rate=settings.htr_learning_rate,
        validation_split=settings.htr_validation_split,
        confidence_warning_threshold=settings.htr_confidence_warning_threshold,
        confidence_critical_threshold=settings.htr_confidence_critical_threshold,
        random_seed=settings.htr_random_seed,
    )


def build_confidence_policy() -> ConfidencePolicy:
    return ConfidencePolicy(
        warning_threshold=settings.htr_confidence_warning_threshold,
        critical_threshold=settings.htr_confidence_critical_threshold,
    )


def build_storage() -> HTRStorage:
    return HTRStorage(settings.htr_storage_dir)


def build_default_model_ref() -> ModelRef:
    return ModelRef(id=settings.htr_default_model_id, path=settings.htr_default_model_path)


def build_recognizer() -> KrakenRecognizer:
    """Recognition backend (kraken is imported lazily on first use)."""
    return KrakenRecognizer(
        device=settings.htr_recognition_device,
        batch_size=settings.htr_recognition_batch_size,
        padding=settings.htr_recognition_padding,
        text_direction=settings.htr_recognition_text_direction,
        num_line_workers=settings.htr_recognition_num_line_workers,
        maxcolseps=settings.htr_segmentation_maxcolseps,
        no_hlines=settings.htr_segmentation_no_hlines,
    )


def _environment_snapshot() -> dict:
    versions = {}
    for package in ("kraken", "torch"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return {"packages": versions}


def build_page_service(db: Session) -> HandwritingPageService:
    storage = build_storage()
    return HandwritingPageService(
        page_repository=SqlAlchemyPageRepository(db),
        model_repository=SqlAlchemyModelRepository(db, storage, build_default_model_ref()),
        image_store=storage,
        metrics_evaluator=MetricsEvaluator(),
        recognizer=build_recognizer(),
    )


def build_training_service(db: Session) -> HandwritingTrainingService:
    storage = build_storage()
    page_repository = SqlAlchemyPageRepository(db)
    dataset_builder = TrainingDatasetBuilder(
        page_repository=page_repository,
        line_cropper=PilLineCropper(),
        crop_path_provider=storage.line_crop_path,
    )
    return HandwritingTrainingService(
        dataset_builder=dataset_builder,
        model_repository=SqlAlchemyModelRepository(db, storage, build_default_model_ref()),
        training_run_repository=SqlAlchemyTrainingRunRepository(db),
        trainer=KrakenTrainer(work_dir=storage.training_work_dir()),
        config=build_training_config(),
        min_training_samples=settings.htr_min_training_samples,
        environment=_environment_snapshot(),
    )


def build_model_repository(db: Session) -> SqlAlchemyModelRepository:
    return SqlAlchemyModelRepository(db, build_storage(), build_default_model_ref())
