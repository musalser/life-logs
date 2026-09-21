"""Recognition flow tests: POST /pages/{id}/recognize behind the service layer.

Uses the real SQLAlchemy repositories (sqlite) and a fake recognizer, so the
tests stay independent of the optional kraken backend.
"""
from __future__ import annotations

import io

import pytest

from app.htr.application.page_service import HandwritingPageService
from app.htr.domain.entities import ModelRef, ModelVersionStatus, PageStatus
from app.htr.domain.errors import PageStateError, RecognitionError
from app.htr.infrastructure.model_repository import SqlAlchemyModelRepository
from app.htr.infrastructure.page_repository import SqlAlchemyPageRepository
from app.htr.infrastructure.storage import HTRStorage
from tests.htr_fakes import FakeRecognizer, recognition_result

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

DEFAULT = ModelRef(id="default", path="/models/default/ppocrv6_medium.safetensors")


def png_bytes(width=200, height=120) -> bytes:
    buf = io.BytesIO()
    Image.new("L", (width, height), color=255).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def env(db_session, tmp_path, user, author):
    storage = HTRStorage(tmp_path / "htr")
    page_repo = SqlAlchemyPageRepository(db_session)
    model_repo = SqlAlchemyModelRepository(db_session, storage, DEFAULT)

    def build(recognizer):
        return HandwritingPageService(
            page_repository=page_repo,
            model_repository=model_repo,
            image_store=storage,
            recognizer=recognizer,
        )

    return build, model_repo, user, author


def _ready_active_version(model_repo, author):
    version = model_repo.create_version(
        author_id=author.id,
        base_model_id=DEFAULT.id,
        training_config={"epochs": 1},
        dataset_hash="hash",
    )
    model_repo.mark_ready(version.id, f"/models/author_{author.id}/v1/model.safetensors", {})
    model_repo.activate_model(author.id, version.id)
    return model_repo.get_active_model(author.id)


def test_recognize_page_stores_prediction(env):
    build, model_repo, user, author = env
    recognizer = FakeRecognizer(result=recognition_result())
    service = build(recognizer)

    page = service.upload_page(user.id, author.id, "page.png", png_bytes())
    assert page.status == PageStatus.UPLOADED

    page = service.recognize_page(page.id)

    assert page.status == PageStatus.RECOGNIZED
    assert [line.predicted_text for line in page.lines] == ["Уж очень дед", "на еврея"]
    assert [w.predicted_text for w in page.lines[0].words] == ["Уж", "очень", "дед"]
    assert page.lines[0].words[1].confidence == pytest.approx(0.55)
    # page geometry comes from the (orientation-normalized) recognized image
    assert (page.width, page.height) == (200, 120)
    # no author model exists yet -> the configured default model is used
    assert recognizer.calls == [(page.file_path, DEFAULT.path)]
    assert page.recognition_model_version_id is None


def test_recognize_page_uses_active_author_model(env):
    build, model_repo, user, author = env
    active = _ready_active_version(model_repo, author)
    recognizer = FakeRecognizer(result=recognition_result())
    service = build(recognizer)

    page = service.upload_page(user.id, author.id, "page.png", png_bytes())
    page = service.recognize_page(page.id)

    assert recognizer.calls[0][1] == active.file_path
    assert page.recognition_model_version_id == active.id


def test_recognize_page_is_rejected_while_editing(env):
    build, model_repo, user, author = env
    service = build(FakeRecognizer(result=recognition_result()))

    page = service.upload_page(user.id, author.id, "page.png", png_bytes())
    page = service.recognize_page(page.id)
    page = service.update_line(page.id, page.lines[0].id, "Уж очень дед был")
    assert page.status == PageStatus.EDITING

    with pytest.raises(PageStateError):
        service.recognize_page(page.id)


def test_recognize_page_without_backend_is_an_error(env):
    build, model_repo, user, author = env
    service = build(None)
    page = service.upload_page(user.id, author.id, "page.png", png_bytes())

    with pytest.raises(RecognitionError):
        service.recognize_page(page.id)


def test_backend_failure_propagates(env):
    build, model_repo, user, author = env
    service = build(FakeRecognizer(error=RecognitionError("no text lines")))
    page = service.upload_page(user.id, author.id, "page.png", png_bytes())

    with pytest.raises(RecognitionError, match="no text lines"):
        service.recognize_page(page.id)


def test_recognized_page_can_be_edited_and_confirmed(env):
    build, model_repo, user, author = env
    service = build(FakeRecognizer(result=recognition_result()))

    page = service.upload_page(user.id, author.id, "page.png", png_bytes())
    page = service.recognize_page(page.id)

    word = page.lines[0].words[1]
    page = service.update_word(page.id, page.lines[0].id, word.id, "очень")
    page = service.confirm_page(page.id)

    assert page.status == PageStatus.CONFIRMED
    assert page.lines[0].corrected_text == "Уж очень дед"
    # predictions stay untouched, the measured quality drops below the demo 1.0
    assert page.prediction_cer == pytest.approx(0.0)
    assert page.prediction_wer == pytest.approx(0.0)


def test_recognition_marks_previous_prediction_replaced(env):
    build, model_repo, user, author = env
    service = build(FakeRecognizer(result=recognition_result()))

    page = service.upload_page(user.id, author.id, "page.png", png_bytes())
    page = service.recognize_page(page.id)
    first_line_ids = [line.id for line in page.lines]

    page = service.recognize_page(page.id)

    assert [line.order for line in page.lines] == [0, 1]
    assert len(page.lines) == 2
    assert all(line.id not in first_line_ids for line in page.lines)


def test_second_page_keeps_previously_ready_version(env):
    """Recognizing does not disturb the model lifecycle."""
    build, model_repo, user, author = env
    active = _ready_active_version(model_repo, author)
    service = build(FakeRecognizer(result=recognition_result()))

    page = service.upload_page(user.id, author.id, "page.png", png_bytes())
    service.recognize_page(page.id)

    versions = model_repo.list_versions(author.id)
    assert [v.status for v in versions] == [ModelVersionStatus.ACTIVE]
    assert versions[0].id == active.id
