"""HTTP-level tests of the HTR routes.

A minimal app with only the HTR router is built per test, with the database,
the authenticated user and the page service overridden. This keeps the tests
independent of Ollama, Postgres and the optional kraken backend.
"""
from __future__ import annotations

import io

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.deps import get_current_user, get_db
from app.htr.application.dataset_builder import TrainingDatasetBuilder
from app.htr.application.page_service import HandwritingPageService
from app.htr.application.training_service import HandwritingTrainingService
from app.htr.domain.entities import ModelRef, TrainingConfig
from app.htr.domain.errors import RecognitionError
from app.htr.infrastructure.model_repository import (
    SqlAlchemyModelRepository,
    SqlAlchemyTrainingRunRepository,
)
from app.htr.infrastructure.page_repository import SqlAlchemyPageRepository
from app.htr.infrastructure.storage import HTRStorage, PilLineCropper
from app.routes import htr as htr_routes
from app.routes.htr import get_page_service, get_training_service
from tests.htr_fakes import FakeRecognizer, RecordingTrainer, recognition_result

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

DEFAULT = ModelRef(id="default", path="/models/default/ppocrv6_medium.safetensors")


def png_bytes(width=200, height=120) -> bytes:
    buffer = io.BytesIO()
    Image.new("L", (width, height), color=255).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture()
def api(db_session, tmp_path, user, author):
    storage = HTRStorage(tmp_path / "htr")
    page_repo = SqlAlchemyPageRepository(db_session)
    model_repo = SqlAlchemyModelRepository(db_session, storage, DEFAULT)
    service = HandwritingPageService(
        page_repository=page_repo,
        model_repository=model_repo,
        image_store=storage,
        recognizer=FakeRecognizer(result=recognition_result()),
    )
    training_service = HandwritingTrainingService(
        dataset_builder=TrainingDatasetBuilder(
            page_repository=page_repo,
            line_cropper=PilLineCropper(),
            crop_path_provider=storage.line_crop_path,
        ),
        model_repository=model_repo,
        training_run_repository=SqlAlchemyTrainingRunRepository(db_session),
        trainer=RecordingTrainer(write_artifact=True),
        config=TrainingConfig(device="cpu"),
    )

    app = FastAPI()
    app.include_router(htr_routes.router)
    app.dependency_overrides[get_db] = lambda: db_session
    app.dependency_overrides[get_current_user] = lambda: user.username
    app.dependency_overrides[get_page_service] = lambda: service
    app.dependency_overrides[get_training_service] = lambda: training_service

    with TestClient(app) as client:
        yield client, service, author


def upload(client, author_id: int) -> int:
    response = client.post(
        f"/htr/authors/{author_id}/pages",
        files={"file": ("page.png", png_bytes(), "image/png")},
    )
    assert response.status_code == 201, response.text
    return response.json()["page_id"]


def test_recognize_endpoint_stores_lines_and_words(api):
    client, _, author = api
    page_id = upload(client, author.id)

    response = client.post(f"/htr/pages/{page_id}/recognize")

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "RECOGNIZED"
    assert body["width"] == 200 and body["height"] == 120
    assert [line["predicted_text"] for line in body["lines"]] == ["Уж очень дед", "на еврея"]
    first_word = body["lines"][0]["words"][1]
    assert first_word["predicted_text"] == "очень"
    assert first_word["confidence"] == pytest.approx(0.55)
    assert first_word["confidence_level"] in {"normal", "warning", "critical"}


def test_recognize_endpoint_reports_backend_failure(api):
    client, service, author = api
    page_id = upload(client, author.id)
    service.recognizer = FakeRecognizer(error=RecognitionError("model file is missing"))

    response = client.post(f"/htr/pages/{page_id}/recognize")

    assert response.status_code == 422
    assert "model file is missing" in response.json()["detail"]


def test_recognize_endpoint_conflicts_with_editing_page(api):
    client, service, author = api
    page_id = upload(client, author.id)
    page = service.recognize_page(page_id)
    service.update_line(page_id, page.lines[0].id, "Уж очень дед был")

    response = client.post(f"/htr/pages/{page_id}/recognize")

    assert response.status_code == 409


def test_import_endpoint_rejects_placeholder_geometry(api):
    """Regression: the Swagger example body must not be stored as a prediction."""
    client, _, author = api
    page_id = upload(client, author.id)

    response = client.post(
        f"/htr/pages/{page_id}/recognition-result",
        json={
            "page_width": 4640,
            "page_height": 3472,
            "lines": [
                {
                    "bbox": {"x1": 0, "y1": 0, "x2": 0, "y2": 0},
                    "text": "string",
                    "words": [],
                }
            ],
        },
    )

    assert response.status_code == 422
    # nothing was written
    page = client.get(f"/htr/pages/{page_id}").json()
    assert page["status"] == "UPLOADED"
    assert page["lines"] == []


def test_import_endpoint_rejects_result_without_lines(api):
    client, _, author = api
    page_id = upload(client, author.id)

    response = client.post(
        f"/htr/pages/{page_id}/recognition-result",
        json={"lines": []},
    )

    assert response.status_code == 422


def test_import_endpoint_still_accepts_real_external_results(api):
    client, _, author = api
    page_id = upload(client, author.id)

    response = client.post(
        f"/htr/pages/{page_id}/recognition-result",
        json={
            "page_width": 200,
            "page_height": 120,
            "lines": [
                {
                    "bbox": {"x1": 0, "y1": 0, "x2": 200, "y2": 40},
                    "text": "Уж очень дед",
                    "words": [
                        {
                            "bbox": {"x1": 0, "y1": 0, "x2": 30, "y2": 40},
                            "text": "Уж",
                            "confidence": 0.9,
                        }
                    ],
                }
            ],
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "RECOGNIZED"
    assert body["lines"][0]["words"][0]["predicted_text"] == "Уж"


def test_full_http_flow_recognize_edit_confirm(api):
    client, _, author = api
    page_id = upload(client, author.id)
    page = client.post(f"/htr/pages/{page_id}/recognize").json()

    line_id = page["lines"][0]["id"]
    word_id = page["lines"][0]["words"][1]["id"]
    edited = client.patch(
        f"/htr/pages/{page_id}/lines/{line_id}/words/{word_id}",
        json={"corrected_text": "очень"},
    )
    assert edited.status_code == 200

    confirmed = client.post(f"/htr/pages/{page_id}/confirm")

    assert confirmed.status_code == 200, confirmed.text
    body = confirmed.json()
    assert body["page"]["status"] == "CONFIRMED"
    assert body["page"]["lines"][0]["corrected_text"] == "Уж очень дед"
    assert body["page"]["prediction_cer"] == pytest.approx(0.0)
    # training used the confirmed page (fake trainer in the real service)
    assert body["training"]["outcome"] == "SUCCESS"
