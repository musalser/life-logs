"""HTTP-level tests of the HTR routes.

A minimal app with only the HTR router is built per test, with the database,
the authenticated user and the page service overridden. This keeps the tests
independent of Ollama, Postgres and the optional kraken backend.
"""
from __future__ import annotations

import io
import os

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.deps import get_current_user, get_db
from app.htr.application.correction_context import CorrectionContextBuilder
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
from tests.htr_fakes import (
    FakeCorrector,
    FakeLexiconProvider,
    FakeRecognizer,
    FakeWordList,
    RecordingTrainer,
    recognition_result,
)

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
        min_training_lines=1,
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


def upload_named(client, author_id: int, filename: str, source_path: str | None = None) -> int:
    data = {"source_path": source_path} if source_path is not None else {}
    response = client.post(
        f"/htr/authors/{author_id}/pages",
        files={"file": (filename, png_bytes(), "image/png")},
        data=data,
    )
    assert response.status_code == 201, response.text
    return response.json()["page_id"]


@pytest.fixture()
def lexicon_api(db_session, tmp_path, user, author):
    """Same app, but with a working dictionary check wired in."""
    from app.htr.application.lexicon import LexiconAnnotator
    from app.htr.infrastructure.lexicon import (
        AuthorCorpusCache,
        SqlAlchemyAuthorCorpus,
        SqlAlchemyLexiconProvider,
    )

    storage = HTRStorage(tmp_path / "htr")
    cache = AuthorCorpusCache()
    page_repo = SqlAlchemyPageRepository(db_session, corpus_cache=cache)
    model_repo = SqlAlchemyModelRepository(db_session, storage, DEFAULT)
    provider = SqlAlchemyLexiconProvider(
        corpus=SqlAlchemyAuthorCorpus(db_session, cache=cache),
        base=FakeWordList({"уж", "очень", "на"}),
    )
    service = HandwritingPageService(
        page_repository=page_repo,
        model_repository=model_repo,
        image_store=storage,
        recognizer=FakeRecognizer(result=recognition_result()),
        lexicon_annotator=LexiconAnnotator(provider),
    )
    app = FastAPI()
    app.include_router(htr_routes.router)
    app.dependency_overrides[get_db] = lambda: db_session
    app.dependency_overrides[get_current_user] = lambda: user.username
    app.dependency_overrides[get_page_service] = lambda: service

    with TestClient(app) as client:
        yield client, service, author


def test_page_read_reports_the_dictionary_check(lexicon_api):
    client, _, author = lexicon_api
    page_id = upload(client, author.id)
    client.post(f"/htr/pages/{page_id}/recognize")

    body = client.get(f"/htr/pages/{page_id}").json()

    assert body["lexicon_available"] is True
    assert body["oov_count"] == 2  # "дед" and "еврея" are not in the dictionary
    assert [line["oov_count"] for line in body["lines"]] == [1, 1]
    assert [line["oov_words"] for line in body["lines"]] == [["дед"], ["еврея"]]
    assert [word["in_lexicon"] for word in body["lines"][0]["words"]] == [True, True, False]
    assert [word["in_lexicon"] for word in body["lines"][1]["words"]] == [True, False]


def test_page_list_reports_oov_totals(lexicon_api):
    client, _, author = lexicon_api
    page_id = upload(client, author.id)
    client.post(f"/htr/pages/{page_id}/recognize")

    body = client.get(f"/htr/authors/{author.id}/pages").json()

    assert len(body) == 1
    assert body[0]["oov_count"] == 2
    assert body[0]["lexicon_available"] is True


def test_page_without_a_dictionary_reports_nothing(api):
    client, _, author = api
    page_id = upload(client, author.id)
    client.post(f"/htr/pages/{page_id}/recognize")

    body = client.get(f"/htr/pages/{page_id}").json()

    assert body["lexicon_available"] is False
    assert body["oov_count"] == 0
    assert all(word["in_lexicon"] is None for line in body["lines"] for word in line["words"])


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
    # curved outlines are exposed so the UI can draw polygons, not boxes
    assert body["lines"][0]["polygon"] is not None
    assert first_word["polygon"] == [[35, 0], [90, 6], [90, 40], [35, 34]]


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


def test_list_author_pages_returns_summaries(api):
    client, _, author = api
    page_id = upload(client, author.id)
    client.post(f"/htr/pages/{page_id}/recognize")

    response = client.get(f"/htr/authors/{author.id}/pages")

    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body) == 1
    assert body[0]["page_id"] == page_id
    assert body[0]["status"] == "RECOGNIZED"
    assert body[0]["line_count"] == 2  # the fake recognizer returns two lines


def test_upload_keeps_the_file_name_and_the_full_path(api):
    client, _, author = api
    page_id = upload_named(client, author.id, "scan_001.jpg", source_path="diary/scan_001.jpg")

    page = client.get(f"/htr/pages/{page_id}").json()
    assert page["file_name"] == "scan_001.jpg"
    assert page["source_path"] == "diary/scan_001.jpg"

    summary = client.get(f"/htr/authors/{author.id}/pages").json()[0]
    assert summary["file_name"] == "scan_001.jpg"
    assert summary["source_path"] == "diary/scan_001.jpg"
    assert summary["order_index"] == 0


def test_source_path_is_normalized_and_split(api):
    client, _, author = api
    page_id = upload_named(
        client, author.id, "page_07.png", source_path=r"Дневник\1975\page_07.png"
    )

    page = client.get(f"/htr/pages/{page_id}").json()
    assert page["file_name"] == "page_07.png"
    assert page["source_path"] == "Дневник/1975/page_07.png"


def test_upload_without_a_path_keeps_only_the_name(api):
    client, _, author = api
    page_id = upload_named(client, author.id, "letter.png")

    page = client.get(f"/htr/pages/{page_id}").json()
    assert page["file_name"] == "letter.png"
    assert page["source_path"] is None


def test_pages_keep_the_upload_order_until_the_user_reorders_them(api):
    client, _, author = api
    first = upload_named(client, author.id, "a.png")
    second = upload_named(client, author.id, "b.png")
    third = upload_named(client, author.id, "c.png")

    body = client.get(f"/htr/authors/{author.id}/pages").json()
    assert [row["page_id"] for row in body] == [first, second, third]
    assert [row["order_index"] for row in body] == [0, 1, 2]


def test_reorder_endpoint_stores_the_dragged_order(api):
    client, _, author = api
    first = upload_named(client, author.id, "a.png")
    second = upload_named(client, author.id, "b.png")
    third = upload_named(client, author.id, "c.png")
    # drag the last page to the very top
    response = client.put(
        f"/htr/authors/{author.id}/pages/order",
        json={"page_ids": [third, first, second]},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert [row["page_id"] for row in body] == [third, first, second]
    assert [row["order_index"] for row in body] == [0, 1, 2]

    # the stored order survives a reload
    reloaded = client.get(f"/htr/authors/{author.id}/pages").json()
    assert [row["page_id"] for row in reloaded] == [third, first, second]


def test_reorder_endpoint_keeps_pages_the_client_did_not_send(api):
    client, _, author = api
    first = upload_named(client, author.id, "a.png")
    second = upload_named(client, author.id, "b.png")
    third = upload_named(client, author.id, "c.png")

    response = client.put(
        f"/htr/authors/{author.id}/pages/order",
        json={"page_ids": [third, first]},
    )

    assert response.status_code == 200, response.text
    assert [row["page_id"] for row in response.json()] == [third, first, second]


def test_reorder_endpoint_rejects_a_foreign_page(api):
    client, _, author = api
    page_id = upload_named(client, author.id, "a.png")

    response = client.put(
        f"/htr/authors/{author.id}/pages/order",
        json={"page_ids": [page_id, 999_999]},
    )

    assert response.status_code == 404
    # nothing moved
    body = client.get(f"/htr/authors/{author.id}/pages").json()
    assert [row["page_id"] for row in body] == [page_id]


def test_rename_endpoint_changes_the_display_name(api):
    client, _, author = api
    page_id = upload_named(client, author.id, "scan_001.jpg", source_path="diary/scan_001.jpg")

    response = client.patch(f"/htr/pages/{page_id}/name", json={"file_name": "Письмо деда"})

    assert response.status_code == 200, response.text
    page = response.json()
    assert page["file_name"] == "Письмо деда"
    # a manual name is canonical: the upload path must not shadow it
    assert page["source_path"] is None
    summary = client.get(f"/htr/authors/{author.id}/pages").json()[0]
    assert summary["file_name"] == "Письмо деда"


def test_rename_endpoint_rejects_a_blank_name(api):
    client, _, author = api
    page_id = upload_named(client, author.id, "scan_001.jpg")

    response = client.patch(f"/htr/pages/{page_id}/name", json={"file_name": "   "})

    assert response.status_code == 400


def test_page_image_endpoint_serves_the_original(api):
    client, _, author = api
    page_id = upload(client, author.id)

    response = client.get(f"/htr/pages/{page_id}/image")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("image/")
    assert response.content == png_bytes()


def test_page_image_endpoint_reports_missing_file(api, db_session, tmp_path):
    client, service, author = api
    page_id = upload(client, author.id)
    # simulate a page whose file disappeared from disk (e.g. migrated from Windows)
    page = service.page_repository.get_page(page_id)
    import os

    os.remove(page.file_path)

    response = client.get(f"/htr/pages/{page_id}/image")

    assert response.status_code == 404


def test_delete_page_removes_row_and_assets(api):
    client, service, author = api
    page_id = upload(client, author.id)
    client.post(f"/htr/pages/{page_id}/recognize")
    page = service.page_repository.get_page(page_id)
    image_path = page.file_path
    crop_dir = service.image_store.root / "crops" / f"page_{page_id}"
    crop_dir.mkdir(parents=True, exist_ok=True)
    (crop_dir / "line_1.png").write_bytes(b"crop")
    assert os.path.isfile(image_path)

    response = client.delete(f"/htr/pages/{page_id}")

    assert response.status_code == 204, response.text
    assert client.get(f"/htr/pages/{page_id}").status_code == 404
    assert client.get(f"/htr/authors/{author.id}/pages").json() == []
    assert not os.path.isfile(image_path)
    assert not crop_dir.exists()


def test_delete_page_reports_missing_page(api):
    client, _, _author = api

    assert client.delete("/htr/pages/99999").status_code == 404


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
    assert body["status"] == "CONFIRMED"
    assert body["lines"][0]["corrected_text"] == "Уж очень дед"
    assert body["prediction_cer"] == pytest.approx(0.0)


def test_confirm_does_not_train_the_model(api):
    """Confirmation is ground truth only: fine-tuning is an explicit action."""
    client, _, author = api
    page_id = upload(client, author.id)
    client.post(f"/htr/pages/{page_id}/recognize")

    client.post(f"/htr/pages/{page_id}/confirm")

    # nothing was trained behind the user's back
    assert client.get(f"/htr/authors/{author.id}/models").json() == []


def test_train_endpoint_fine_tunes_the_confirmed_corpus(api):
    client, _, author = api
    page_id = upload(client, author.id)
    client.post(f"/htr/pages/{page_id}/recognize")
    client.post(f"/htr/pages/{page_id}/confirm")

    response = client.post(f"/htr/authors/{author.id}/train")

    assert response.status_code == 200, response.text
    body = response.json()
    # training used the confirmed page (fake trainer in the real service)
    assert body["outcome"] == "SUCCESS"
    # regression: the response must carry the post-activation status, not the
    # TRAINING status the version had when it was created
    assert body["model_version"]["status"] == "ACTIVE"


def test_recognition_does_not_ask_the_model_automatically(api):
    """The LLM step is manual: recognition alone must not call it at all."""
    client, service, author = api
    corrector = FakeCorrector({"Уж очень дед": "Уж очень, дед"})
    service.corrector = corrector
    service.correction_context_builder = CorrectionContextBuilder(service.page_repository)
    page_id = upload(client, author.id)

    body = client.post(f"/htr/pages/{page_id}/recognize").json()

    assert body["status"] == "RECOGNIZED"
    assert body["lines"][0]["corrected_by"] is None
    assert body["lines"][0]["suggested_text"] is None
    assert corrector.calls == []


def test_recognition_suggests_automatically_when_enabled(api):
    """Even with auto_suggest the transcription stays kraken's."""
    client, service, author = api
    service.corrector = FakeCorrector({"Уж очень дед": "Уж очень, дед"})
    service.correction_context_builder = CorrectionContextBuilder(service.page_repository)
    service.auto_correct = True
    page_id = upload(client, author.id)

    body = client.post(f"/htr/pages/{page_id}/recognize").json()

    line = body["lines"][0]
    assert body["status"] == "RECOGNIZED"          # nothing was edited
    assert line["suggested_text"] == "Уж очень, дед"
    assert line["predicted_text"] == "Уж очень дед"
    assert line["corrected_text"] is None


def test_recognition_response_carries_word_alternatives(api):
    """The editor needs them on the same page read it renders the words from."""
    from app.htr.domain.entities import WordAlternative

    client, service, author = api
    result = recognition_result()
    result.lines[0].words[0].alternatives = [
        WordAlternative(text="Ужъ", score=-1.5),
        WordAlternative(text="Уш", score=-2.5),
    ]
    service.recognizer.result = result
    page_id = upload(client, author.id)

    body = client.post(f"/htr/pages/{page_id}/recognize").json()

    word = body["lines"][0]["words"][0]
    assert [item["text"] for item in word["alternatives"]] == ["Ужъ", "Уш"]
    assert word["alternatives"][0]["score"] == -1.5
    assert body["lines"][0]["words"][1]["alternatives"] == []


def test_suggestions_endpoint_stores_proposals(api):
    client, service, author = api
    page_id = upload(client, author.id)
    client.post(f"/htr/pages/{page_id}/recognize")
    service.corrector = FakeCorrector({"на еврея": "на еврея,"})
    service.correction_context_builder = CorrectionContextBuilder(service.page_repository)

    response = client.post(f"/htr/pages/{page_id}/suggestions")

    assert response.status_code == 200, response.text
    body = response.json()
    line = body["lines"][1]
    assert line["suggested_text"] == "на еврея,"
    assert line["corrected_text"] is None
    assert line["predicted_text"] == "на еврея"
    # a punctuation-only proposal has no word change to verify
    assert line["suggestion_changes"] == []


def test_suggestion_accept_and_dismiss_endpoints(api):
    client, service, author = api
    page_id = upload(client, author.id)
    client.post(f"/htr/pages/{page_id}/recognize")
    service.corrector = FakeCorrector(
        {"Уж очень дед": "Уж очень дед был", "на еврея": "на еврея,"}
    )
    service.correction_context_builder = CorrectionContextBuilder(service.page_repository)
    body = client.post(f"/htr/pages/{page_id}/suggestions").json()
    first_id = body["lines"][0]["id"]
    second_id = body["lines"][1]["id"]

    accepted = client.put(f"/htr/pages/{page_id}/lines/{first_id}/suggestion")

    assert accepted.status_code == 200, accepted.text
    line = accepted.json()["lines"][0]
    assert line["corrected_text"] == "Уж очень дед был"
    assert line["corrected_by"] == "user"
    assert line["suggested_text"] is None
    assert accepted.json()["status"] == "EDITING"

    dismissed = client.delete(f"/htr/pages/{page_id}/lines/{second_id}/suggestion")

    assert dismissed.status_code == 200, dismissed.text
    assert dismissed.json()["lines"][1]["suggested_text"] is None
    assert dismissed.json()["lines"][1]["corrected_text"] is None


def test_suggestion_change_endpoint_applies_a_single_word(api):
    from app.htr.application.lexicon import LexiconAnnotator

    client, service, author = api
    service.lexicon_annotator = LexiconAnnotator(
        FakeLexiconProvider(known={"уж", "очень", "дед", "был", "похож", "на", "еврея"})
    )
    page_id = upload(client, author.id)
    client.post(f"/htr/pages/{page_id}/recognize")
    service.corrector = FakeCorrector({"Уж очень дед": "Уж очень дед был"})
    service.correction_context_builder = CorrectionContextBuilder(service.page_repository)
    body = client.post(f"/htr/pages/{page_id}/suggestions").json()
    line = body["lines"][0]
    assert line["suggestion_changes"][0]["after"] == "был"

    response = client.put(
        f"/htr/pages/{page_id}/lines/{line['id']}/suggestion/changes/0"
    )

    assert response.status_code == 200, response.text
    updated = response.json()["lines"][0]
    assert updated["corrected_text"] == "Уж очень дед был"
    assert updated["corrected_by"] == "user"
    # nothing left to review on this line, so the proposal disappears
    assert updated["suggested_text"] == "Уж очень дед был"


def test_suggestion_change_endpoint_reports_a_stale_index(api):
    client, service, author = api
    page_id = upload(client, author.id)
    client.post(f"/htr/pages/{page_id}/recognize")
    service.corrector = FakeCorrector({"на еврея": "на еврея,"})  # punctuation only
    service.correction_context_builder = CorrectionContextBuilder(service.page_repository)
    body = client.post(f"/htr/pages/{page_id}/suggestions").json()
    line = body["lines"][1]

    response = client.put(
        f"/htr/pages/{page_id}/lines/{line['id']}/suggestion/changes/0"
    )

    assert response.status_code == 404


def test_suggestion_accept_reports_missing_proposal(api):
    client, service, author = api
    page_id = upload(client, author.id)
    body = client.post(f"/htr/pages/{page_id}/recognize").json()

    response = client.put(f"/htr/pages/{page_id}/lines/{body['lines'][0]['id']}/suggestion")

    assert response.status_code == 404


def test_bulk_accept_endpoint_takes_verified_proposals_only(lexicon_api):
    from app.htr.application.lexicon import LexiconAnnotator

    client, service, author = lexicon_api
    service.lexicon_annotator = LexiconAnnotator(
        FakeLexiconProvider(known={"уж", "очень", "дед", "был", "на", "еврея"})
    )
    page_id = upload(client, author.id)
    client.post(f"/htr/pages/{page_id}/recognize")
    service.corrector = FakeCorrector(
        {
            "Уж очень дед": "Уж очень дед был",        # "был" is in the dictionary
            "на еврея": "на еврея, Захарьевка",        # invented place name
        }
    )
    service.correction_context_builder = CorrectionContextBuilder(service.page_repository)
    client.post(f"/htr/pages/{page_id}/suggestions")

    response = client.post(f"/htr/pages/{page_id}/suggestions/accept")

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["accepted"] == 1
    lines = payload["page"]["lines"]
    assert lines[0]["corrected_text"] == "Уж очень дед был"
    assert lines[1]["corrected_text"] is None
    assert lines[1]["suggested_text"] == "на еврея, Захарьевка"
    assert lines[1]["suggestion_verified"] is False
    assert [change["in_lexicon"] for change in lines[1]["suggestion_changes"]] == [False]


def test_automatic_correction_is_skipped_when_ollama_is_unreachable(api):
    """Recognition must not turn a dead LLM into a per-line wait."""
    client, service, author = api
    corrector = FakeCorrector({"Уж очень дед": "Уж очень, дед"}, available=False)
    service.corrector = corrector
    service.correction_context_builder = CorrectionContextBuilder(service.page_repository)
    service.auto_correct = True
    page_id = upload(client, author.id)

    body = client.post(f"/htr/pages/{page_id}/recognize").json()

    assert body["status"] == "RECOGNIZED"
    assert corrector.calls == []
    assert all(line["suggested_text"] is None for line in body["lines"])


def test_suggestions_refuse_fast_when_ollama_is_unreachable(api):
    """A stopped Ollama must not cost a timeout per line before the page returns."""
    client, service, author = api
    page_id = upload(client, author.id)
    client.post(f"/htr/pages/{page_id}/recognize")
    corrector = FakeCorrector({"на еврея": "на еврея,"}, available=False)
    service.corrector = corrector
    service.correction_context_builder = CorrectionContextBuilder(service.page_repository)

    response = client.post(f"/htr/pages/{page_id}/suggestions")

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert "Ollama недоступна" in detail
    assert "fake-ollama" in detail
    # not a single line was attempted, and nothing was proposed
    assert corrector.calls == []
    body = client.get(f"/htr/pages/{page_id}").json()
    assert all(line["suggested_text"] is None for line in body["lines"])


def test_suggestions_endpoint_refuses_a_confirmed_page(api):
    client, service, author = api
    service.corrector = FakeCorrector({"Уж очень дед": "испорчено"})
    service.correction_context_builder = CorrectionContextBuilder(service.page_repository)
    page_id = upload(client, author.id)
    client.post(f"/htr/pages/{page_id}/recognize")
    client.post(f"/htr/pages/{page_id}/confirm")

    response = client.post(f"/htr/pages/{page_id}/suggestions")

    assert response.status_code == 409
