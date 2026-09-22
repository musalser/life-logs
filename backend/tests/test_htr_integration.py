"""Integration test of the full workflow:

page upload -> recognition result -> word/line editing -> confirmation ->
dataset build -> training -> new model version -> activation.

Uses real SQLAlchemy repositories (sqlite), the real dataset builder with a
Pillow line cropper and a fake trainer backend.
"""
import io

import pytest

from app.htr.application.dataset_builder import TrainingDatasetBuilder
from app.htr.application.page_service import HandwritingPageService
from app.htr.application.training_service import HandwritingTrainingService
from app.htr.domain.entities import (
    BoundingBox,
    ModelRef,
    ModelVersionStatus,
    PageStatus,
    RecognitionResult,
    RecognizedLine,
    RecognizedWord,
    TrainingConfig,
    TrainingOutcome,
)
from app.htr.infrastructure.model_repository import (
    SqlAlchemyModelRepository,
    SqlAlchemyTrainingRunRepository,
)
from app.htr.infrastructure.page_repository import SqlAlchemyPageRepository
from app.htr.infrastructure.storage import HTRStorage, PilLineCropper
from tests.htr_fakes import RecordingTrainer

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

DEFAULT = ModelRef(id="default", path="/models/default.mlmodel")


def png_bytes(width=200, height=120) -> bytes:
    buf = io.BytesIO()
    Image.new("L", (width, height), color=255).save(buf, format="PNG")
    return buf.getvalue()


def recognition_result() -> RecognitionResult:
    return RecognitionResult(
        page_width=200,
        page_height=120,
        lines=[
            RecognizedLine(
                id="l1",
                bbox=BoundingBox(0, 0, 200, 40),
                text="Уж очен дед был похож",
                words=[
                    RecognizedWord("w1", BoundingBox(0, 0, 30, 40), "Уж", 0.98),
                    RecognizedWord("w2", BoundingBox(35, 0, 90, 40), "очен", 0.55),
                    RecognizedWord("w3", BoundingBox(95, 0, 130, 40), "дед", 0.95),
                    RecognizedWord("w4", BoundingBox(135, 0, 165, 40), "был", 0.92),
                    RecognizedWord("w5", BoundingBox(170, 0, 200, 40), "похож", 0.80),
                ],
            ),
            RecognizedLine(
                id="l2",
                bbox=BoundingBox(0, 50, 200, 90),
                text="на еврея",
                words=[
                    RecognizedWord("w6", BoundingBox(0, 50, 40, 90), "на", 0.97),
                    RecognizedWord("w7", BoundingBox(45, 50, 120, 90), "еврея", 0.75),
                ],
            ),
        ],
    )


@pytest.fixture()
def env(db_session, tmp_path, user, author):
    storage = HTRStorage(tmp_path / "htr")
    page_repo = SqlAlchemyPageRepository(db_session)
    model_repo = SqlAlchemyModelRepository(db_session, storage, DEFAULT)
    page_service = HandwritingPageService(
        page_repository=page_repo,
        model_repository=model_repo,
        image_store=storage,
    )
    trainer = RecordingTrainer(write_artifact=True)
    training_service = HandwritingTrainingService(
        dataset_builder=TrainingDatasetBuilder(
            page_repository=page_repo,
            line_cropper=PilLineCropper(),
            crop_path_provider=storage.line_crop_path,
        ),
        model_repository=model_repo,
        training_run_repository=SqlAlchemyTrainingRunRepository(db_session),
        trainer=trainer,
        config=TrainingConfig(device="cpu"),
        # the test pages are tiny; the production default threshold is 50 lines
        min_training_lines=1,
    )
    return page_service, training_service, model_repo, trainer, user, author


def run_page_cycle(page_service, user, author, edit=True):
    page = page_service.upload_page(user.id, author.id, "page.png", png_bytes())
    assert page.status == PageStatus.UPLOADED
    page = page_service.apply_recognition(page.id, recognition_result())
    assert page.status == PageStatus.RECOGNIZED
    if edit:
        # word-level edit: fix "очен" -> "очень"
        word = page.lines[0].words[1]
        page = page_service.update_word(page.id, page.lines[0].id, word.id, "очень")
        assert page.status == PageStatus.EDITING
        # line-level edit: replace whole second line
        page = page_service.update_line(page.id, page.lines[1].id, "на еврея,")
    return page_service.confirm_page(page.id)


def test_full_training_workflow(env):
    page_service, training_service, model_repo, trainer, user, author = env

    page = run_page_cycle(page_service, user, author)
    assert page.status == PageStatus.CONFIRMED
    assert page.confirmed_at is not None
    # prediction quality was measured against the user's ground truth
    assert page.prediction_cer is not None and page.prediction_cer > 0
    assert page.prediction_wer is not None

    # word edit rebuilt the canonical line transcription
    assert page.lines[0].corrected_text == "Уж очень дед был похож"
    assert page.lines[0].words_stale is False
    # line edit marked word alignment stale
    assert page.lines[1].corrected_text == "на еврея,"
    assert page.lines[1].words_stale is True

    result = training_service.train_author(author.id)
    assert result.outcome == TrainingOutcome.SUCCESS

    versions = model_repo.list_versions(author.id)
    assert len(versions) == 1
    v1 = versions[0]
    assert v1.status == ModelVersionStatus.ACTIVE
    assert v1.version == 1
    assert v1.base_model_id == "default"
    assert v1.dataset_hash == result.dataset_hash

    # trainer received all confirmed lines with the corrected transcriptions
    dataset = trainer.calls[0].dataset
    assert sorted(s.transcription for s in dataset.samples) == sorted(
        ["Уж очень дед был похож", "на еврея,"]
    )
    # line crops were physically produced
    import os
    assert all(os.path.isfile(s.image_path) for s in dataset.samples)


def test_second_page_retrains_from_default_with_full_corpus(env):
    page_service, training_service, model_repo, trainer, user, author = env

    run_page_cycle(page_service, user, author)
    training_service.train_author(author.id)
    run_page_cycle(page_service, user, author, edit=False)
    result = training_service.train_author(author.id)

    assert result.outcome == TrainingOutcome.SUCCESS
    versions = model_repo.list_versions(author.id)
    assert [v.version for v in versions] == [1, 2]
    assert versions[0].status == ModelVersionStatus.READY  # kept, not deleted
    assert versions[1].status == ModelVersionStatus.ACTIVE

    # second run: base is still default, dataset covers BOTH confirmed pages
    second_call = trainer.calls[1]
    assert second_call.base_model.id == "default"
    assert len({s.page_id for s in second_call.dataset.samples}) == 2
    assert len(second_call.dataset.samples) == 4

    # next recognition would use the new active model
    active_ref = model_repo.get_active_model_ref(author.id)
    assert active_ref.path == versions[1].file_path


def test_recognition_records_active_model_version(env):
    page_service, training_service, model_repo, trainer, user, author = env

    run_page_cycle(page_service, user, author)
    training_service.train_author(author.id)
    active = model_repo.get_active_model(author.id)

    page = page_service.upload_page(user.id, author.id, "p2.png", png_bytes())
    page = page_service.apply_recognition(page.id, recognition_result())
    assert page.recognition_model_version_id == active.id
