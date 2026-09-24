"""Composition root of the HTR module: wires settings + DB session into services."""
from __future__ import annotations

from importlib import metadata

from sqlalchemy.orm import Session

from ..config import settings
from .application.confidence import ConfidencePolicy
from .application.correction_context import CorrectionContextBuilder
from .application.dataset_builder import TrainingDatasetBuilder
from .application.lexicon import LexiconAnnotator
from .application.metrics import MetricsEvaluator
from .application.page_service import HandwritingPageService
from .application.training_service import HandwritingTrainingService
from .domain.entities import ModelRef, TrainingConfig
from .infrastructure.knowledge_vocabulary import SqlAlchemyKnowledgeVocabulary
from .infrastructure.kraken.lines import KrakenLineCropper
from .infrastructure.kraken.recognizer import KrakenRecognizer
from .infrastructure.kraken.trainer import KrakenTrainer
from .infrastructure.lexicon import (
    SqlAlchemyAuthorCorpus,
    SqlAlchemyLexiconProvider,
    load_file_lexicon,
)
from .infrastructure.llm.ollama_corrector import OllamaLineCorrector
from .infrastructure.model_repository import (
    SqlAlchemyModelRepository,
    SqlAlchemyTrainingRunRepository,
)
from .infrastructure.page_repository import SqlAlchemyPageRepository
from .infrastructure.storage import HTRStorage, PilLineCropper


def build_training_config() -> TrainingConfig:
    normalization = settings.htr_training_normalization
    if normalization and normalization.lower() in ("none", "null", ""):
        normalization = None
    return TrainingConfig(
        device=settings.htr_device,
        epochs=settings.htr_epochs,
        batch_size=settings.htr_batch_size,
        learning_rate=settings.htr_learning_rate,
        validation_split=settings.htr_validation_split,
        confidence_warning_threshold=settings.htr_confidence_warning_threshold,
        confidence_critical_threshold=settings.htr_confidence_critical_threshold,
        random_seed=settings.htr_random_seed,
        min_epochs=settings.htr_min_epochs,
        # Keys below are understood by KrakenTrainer; another backend would
        # read its own keys and ignore these.
        backend_options={
            "resize": settings.htr_training_resize,
            "normalization": normalization,
            "height": settings.htr_training_height,
            "height_override": settings.htr_training_height_override,
            "max_width": settings.htr_training_max_width,
            "variant": settings.htr_training_variant,
            "precision": settings.htr_training_precision,
            "num_workers": settings.htr_training_num_workers,
            "augment": settings.htr_training_augment,
            "schedule": settings.htr_training_schedule,
            "warmup": settings.htr_training_warmup,
            "freeze_backbone": settings.htr_training_freeze_backbone,
            "freeze_bn": settings.htr_training_freeze_bn,
            "aux_nrtr": settings.htr_training_aux_nrtr,
            "linetype": settings.htr_training_linetype,
            "weight_decay": settings.htr_training_weight_decay,
            "compile": settings.htr_training_compile,
            "matmul_precision": settings.htr_training_matmul_precision,
        },
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


def build_beam_config():
    """Beam-search parameters (imported lazily: the module needs numpy only)."""
    from .infrastructure.kraken.beam import BeamSearchConfig

    return BeamSearchConfig(
        beam_width=settings.htr_beam_width,
        top_k=settings.htr_beam_top_k,
        alpha=settings.htr_beam_alpha,
        beta=settings.htr_beam_beta,
        word_bonus=settings.htr_beam_word_bonus,
    )


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
        segmentation_engine=settings.htr_segmentation_engine,
        merge_lines=settings.htr_segmentation_merge_lines,
        decoder=settings.htr_decoder,
        lm_path=settings.htr_lm_path,
        beam_config=build_beam_config(),
        lexicon_path=settings.htr_lexicon_path,
        min_lm_text_chars=settings.htr_lm_min_text_chars,
    )


def _environment_snapshot() -> dict:
    versions = {}
    for package in ("kraken", "torch"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return {"packages": versions}


def build_corrector() -> OllamaLineCorrector | None:
    """LLM corrector (same Ollama instance as the chat), or None when disabled."""
    if not settings.htr_correction_enabled:
        return None
    return OllamaLineCorrector(
        model=settings.htr_correction_model,
        host=settings.ollama_url,
        timeout=settings.htr_correction_timeout_s,
    )


def build_correction_context_builder(db: Session) -> CorrectionContextBuilder:
    return CorrectionContextBuilder(
        page_repository=SqlAlchemyPageRepository(db),
        vocabulary_provider=SqlAlchemyKnowledgeVocabulary(db),
        max_lexicon=settings.htr_correction_max_lexicon,
        max_vocabulary=settings.htr_correction_max_vocabulary,
        max_confusions=settings.htr_correction_max_confusions,
        max_examples=settings.htr_correction_max_examples,
    )


def build_page_service(db: Session) -> HandwritingPageService:
    storage = build_storage()
    return HandwritingPageService(
        page_repository=SqlAlchemyPageRepository(db),
        model_repository=SqlAlchemyModelRepository(db, storage, build_default_model_ref()),
        image_store=storage,
        metrics_evaluator=MetricsEvaluator(),
        recognizer=build_recognizer(),
        corrector=build_corrector(),
        correction_context_builder=build_correction_context_builder(db),
        lexicon_annotator=build_lexicon_annotator(db),
        auto_correct=settings.htr_correction_auto,
        correction_context_lines=settings.htr_correction_context_lines,
    )


def build_lexicon_annotator(db: Session) -> LexiconAnnotator:
    """Dictionary check of the recognized words (optionally disabled)."""
    if not settings.htr_lexicon_enabled:
        return LexiconAnnotator(None)
    return LexiconAnnotator(
        SqlAlchemyLexiconProvider(
            corpus=SqlAlchemyAuthorCorpus(db),
            base=load_file_lexicon(settings.htr_lexicon_path),
        )
    )


def build_training_service(db: Session) -> HandwritingTrainingService:
    storage = build_storage()
    page_repository = SqlAlchemyPageRepository(db)
    dataset_builder = TrainingDatasetBuilder(
        page_repository=page_repository,
        # crops follow the line outline through kraken's own extractor, with the
        # plain box as a fallback (see infrastructure/kraken/lines.py)
        line_cropper=KrakenLineCropper(fallback=PilLineCropper()),
        crop_path_provider=storage.line_crop_path,
    )
    return HandwritingTrainingService(
        dataset_builder=dataset_builder,
        model_repository=SqlAlchemyModelRepository(db, storage, build_default_model_ref()),
        training_run_repository=SqlAlchemyTrainingRunRepository(db),
        trainer=KrakenTrainer(work_dir=storage.training_work_dir()),
        config=build_training_config(),
        min_training_lines=settings.htr_min_training_lines,
        min_training_words=settings.htr_min_training_words,
        environment=_environment_snapshot(),
    )


def build_model_repository(db: Session) -> SqlAlchemyModelRepository:
    return SqlAlchemyModelRepository(db, build_storage(), build_default_model_ref())
