"""Import of a PDF: one page per sheet, all of them in one folder.

The PDF path is the bulk version of the upload route, so the tests cover what
makes it different from uploading page by page: the folder named after the file,
the all-or-nothing refusal, and the ceiling that keeps a stray 2000-page scan
out of one HTTP request.
"""
from __future__ import annotations

import io

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.deps import get_current_user, get_db
from app.htr.application.page_service import HandwritingPageService
from app.htr.domain.entities import ModelRef
from app.htr.domain.errors import CorruptPdfError
from app.htr.infrastructure.model_repository import SqlAlchemyModelRepository
from app.htr.infrastructure.page_repository import SqlAlchemyPageRepository
from app.htr.infrastructure.pdf import PdfRasterizer, is_greyscale
from app.htr.infrastructure.storage import HTRStorage
from app.models import Entity, Folder, HTRAuthor, HTRPage
from app.routes import htr as htr_routes
from app.routes.htr import get_page_service, get_pdf_rasterizer
from app.services.library_service import LibraryService
from tests.htr_fakes import FakeRecognizer, recognition_result

pytest.importorskip("pypdfium2")
import pypdfium2 as pdfium  # noqa: E402

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

DEFAULT = ModelRef(id="default", path="/models/default/ppocrv6_medium.safetensors")


def pdf_bytes(pages: int) -> bytes:
    """A real multi-page PDF (blank sheets, which is enough for the pipeline)."""
    document = pdfium.PdfDocument.new()
    for _ in range(pages):
        document.new_page(200, 120)
    buffer = io.BytesIO()
    document.save(buffer)
    document.close()
    return buffer.getvalue()


def png_bytes(width=200, height=120) -> bytes:
    buffer = io.BytesIO()
    Image.new("L", (width, height), color=255).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture()
def author_entity(db_session, user, author):
    """The author of the tree: an HTR author bound to an object."""
    entity = Entity(
        user_id=user.id,
        entity_type="person",
        canonical_name="Grandfather",
        normalized_name="grandfather",
    )
    db_session.add(entity)
    db_session.commit()
    author.entity_id = entity.id
    db_session.commit()
    return entity


@pytest.fixture()
def api(db_session, tmp_path, user, author, author_entity):
    storage = HTRStorage(tmp_path / "htr")
    service = HandwritingPageService(
        page_repository=SqlAlchemyPageRepository(db_session),
        model_repository=SqlAlchemyModelRepository(db_session, storage, DEFAULT),
        image_store=storage,
        recognizer=FakeRecognizer(result=recognition_result()),
    )
    app = FastAPI()
    app.include_router(htr_routes.router)
    app.dependency_overrides[get_db] = lambda: db_session
    app.dependency_overrides[get_current_user] = lambda: user.username
    app.dependency_overrides[get_page_service] = lambda: service
    with TestClient(app) as client:
        yield client, service, author


def foreign_folder(db_session, user) -> Folder:
    """A folder that belongs to *another* author of the same account."""
    entity = Entity(
        user_id=user.id, entity_type="person", canonical_name="Баба", normalized_name="баба"
    )
    db_session.add(entity)
    db_session.commit()
    db_session.add(HTRAuthor(user_id=user.id, name="Баба", entity_id=entity.id))
    db_session.commit()
    return LibraryService(db_session).create_folder(user, "чужая", entity.id, None)


def author_node(db_session, user, entity):
    return next(node for node in LibraryService(db_session).tree(user).authors if node.entity_id == entity.id)


def post_pdf(client, author_id: int, name: str, content: bytes, **data):
    return client.post(
        f"/htr/authors/{author_id}/pdf",
        files={"file": (name, content, "application/pdf")},
        data=data,
    )


# ---------------------------------------------------------------------------
# PDF → folder → pages
# ---------------------------------------------------------------------------


def test_every_sheet_becomes_a_page_in_a_folder_named_after_the_pdf(
    api, db_session, user, author_entity
):
    client, _, author = api

    response = post_pdf(client, author.id, "Дневник 1917.pdf", pdf_bytes(3))

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["folder_name"] == "Дневник 1917"
    assert body["page_count"] == 3

    node = author_node(db_session, user, author_entity)
    assert node.pages == []
    assert len(node.folders) == 1
    folder = node.folders[0]
    assert folder.id == body["folder_id"]
    # one row per sheet, in document order, labelled by the sheet number
    assert [page.title for page in folder.pages] == ["Стр. 1.png", "Стр. 2.png", "Стр. 3.png"]
    assert [page.ref_id for page in folder.pages] == body["page_ids"]
    assert [page.order_index for page in folder.pages] == [0, 1, 2]


def test_imported_sheet_is_a_stored_png_the_viewer_can_serve(api, db_session):
    client, _, author = api

    body = post_pdf(client, author.id, "scan.pdf", pdf_bytes(1)).json()
    page_id = body["page_ids"][0]

    image = client.get(f"/htr/pages/{page_id}/image")
    assert image.status_code == 200
    assert image.content.startswith(b"\x89PNG")
    stored = db_session.query(HTRPage).filter(HTRPage.id == page_id).one()
    # the PDF itself is not kept: what is stored is the rendered sheet
    assert stored.file_path.endswith(".png")
    assert stored.source_path == "scan.pdf/Стр. 1.png"
    assert stored.folder_id == body["folder_id"]


def test_reimporting_a_pdf_is_refused_and_adds_nothing(api, db_session, user, author_entity):
    client, _, author = api
    first = post_pdf(client, author.id, "Дневник.pdf", pdf_bytes(2))
    assert first.status_code == 201
    pages_before = db_session.query(HTRPage).count()
    folders_before = db_session.query(Folder).count()

    second = post_pdf(client, author.id, "Дневник.pdf", pdf_bytes(2))

    assert second.status_code == 409
    assert "уже загружены" in second.json()["detail"]
    assert db_session.query(HTRPage).count() == pages_before
    assert db_session.query(Folder).count() == folders_before


def test_existing_folder_of_the_same_name_is_reused(api, db_session, user, author_entity):
    client, _, author = api
    existing = LibraryService(db_session).create_folder(user, "Дневник", author_entity.id, None)

    body = post_pdf(client, author.id, "Дневник.pdf", pdf_bytes(2)).json()

    assert body["folder_id"] == existing.id
    node = author_node(db_session, user, author_entity)
    assert [folder.name for folder in node.folders] == ["Дневник"]
    assert len(node.folders[0].pages) == 2


def test_pdf_can_be_imported_into_a_chosen_folder(api, db_session, user, author_entity):
    client, _, author = api
    parent = LibraryService(db_session).create_folder(user, "1975", author_entity.id, None)

    response = post_pdf(
        client, author.id, "Письма.pdf", pdf_bytes(2), parent_folder_id=str(parent.id)
    )

    assert response.status_code == 201, response.text
    node = author_node(db_session, user, author_entity)
    assert [folder.name for folder in node.folders] == ["1975"]
    letters = node.folders[0].children[0]
    assert letters.name == "Письма"
    assert len(letters.pages) == 2


def test_pdf_into_a_folder_of_another_author_is_refused(
    api, db_session, user, author_entity
):
    client, _, author = api
    foreign = foreign_folder(db_session, user)

    response = post_pdf(
        client, author.id, "Письма.pdf", pdf_bytes(1), parent_folder_id=str(foreign.id)
    )

    assert response.status_code == 400
    assert db_session.query(HTRPage).count() == 0


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_a_file_that_is_not_a_pdf_is_refused(api, db_session):
    client, _, author = api

    response = client.post(
        f"/htr/authors/{author.id}/pdf",
        files={"file": ("page.png", png_bytes(), "image/png")},
    )

    assert response.status_code == 400
    assert db_session.query(HTRPage).count() == 0


def test_a_pdf_without_pages_is_refused(api, db_session):
    client, _, author = api

    response = post_pdf(client, author.id, "пустой.pdf", b"")

    assert response.status_code == 400


def test_a_pdf_over_the_page_ceiling_is_refused(api, db_session, monkeypatch):
    client, _, author = api
    monkeypatch.setattr(htr_routes.settings, "htr_pdf_max_pages", 2)

    response = post_pdf(client, author.id, "том.pdf", pdf_bytes(3))

    assert response.status_code == 400
    assert "больше предела" in response.json()["detail"]
    assert db_session.query(HTRPage).count() == 0
    assert db_session.query(Folder).count() == 0


class HaltingRasterizer:
    """A PDF whose second sheet cannot be rendered."""

    def __init__(self, pages: int = 2) -> None:
        self.pages = pages

    def page_count(self, content: bytes) -> int:
        return self.pages

    def render_pages(self, content: bytes):
        yield png_bytes()
        raise CorruptPdfError("Не удалось отрисовать страницу 2 PDF: boom")


def test_a_pdf_that_fails_halfway_leaves_nothing_behind(api, db_session, user, author_entity):
    client, _, author = api
    client.app.dependency_overrides[get_pdf_rasterizer] = lambda: HaltingRasterizer()

    response = post_pdf(client, author.id, "Сломанный.pdf", pdf_bytes(2))

    assert response.status_code == 400
    assert db_session.query(HTRPage).count() == 0
    # the folder the failed import created is gone as well
    assert db_session.query(Folder).count() == 0
    assert author_node(db_session, user, author_entity).folders == []


# ---------------------------------------------------------------------------
# The same folder target for a single image upload
# ---------------------------------------------------------------------------


def test_an_image_can_be_uploaded_into_a_folder(api, db_session, user, author_entity):
    client, _, author = api
    folder = LibraryService(db_session).create_folder(user, "1975", author_entity.id, None)

    response = client.post(
        f"/htr/authors/{author.id}/pages",
        files={"file": ("page.png", png_bytes(), "image/png")},
        data={"folder_id": str(folder.id)},
    )

    assert response.status_code == 201, response.text
    node = author_node(db_session, user, author_entity)
    assert [page.ref_id for page in node.folders[0].pages] == [response.json()["page_id"]]
    assert node.pages == []


def test_an_image_cannot_be_uploaded_into_a_foreign_folder(api, db_session, user):
    client, _, author = api
    foreign = foreign_folder(db_session, user)

    response = client.post(
        f"/htr/authors/{author.id}/pages",
        files={"file": ("page.png", png_bytes(), "image/png")},
        data={"folder_id": str(foreign.id)},
    )

    assert response.status_code == 400
    assert db_session.query(HTRPage).count() == 0


# ---------------------------------------------------------------------------
# The rasterizer itself
# ---------------------------------------------------------------------------


def test_rasterizer_counts_and_renders_every_sheet():
    rasterizer = PdfRasterizer(dpi=72)
    content = pdf_bytes(3)

    assert rasterizer.page_count(content) == 3
    pages = list(rasterizer.render_pages(content))

    assert len(pages) == 3
    for png in pages:
        assert png.startswith(b"\x89PNG")


def test_rasterizer_refuses_garbage():
    rasterizer = PdfRasterizer(dpi=72)

    with pytest.raises(CorruptPdfError):
        rasterizer.page_count(b"this is not a pdf")
    with pytest.raises(CorruptPdfError):
        list(rasterizer.render_pages(b"this is not a pdf"))


def test_a_blank_sheet_is_stored_as_grayscale():
    """A typescript or black-and-white scan must not cost three channels."""
    rasterizer = PdfRasterizer(dpi=72)

    png = next(iter(rasterizer.render_pages(pdf_bytes(1))))

    assert Image.open(io.BytesIO(png)).mode == "L"


def test_colour_survives_the_render():
    assert is_greyscale(Image.new("RGB", (4, 4), (200, 30, 30))) is False
    assert is_greyscale(Image.new("RGB", (4, 4), (128, 128, 128))) is True
    # a scan is never perfectly neutral: a few counts of noise are still gray
    assert is_greyscale(Image.new("RGB", (4, 4), (128, 132, 129))) is True
    assert is_greyscale(Image.new("RGB", (4, 4), (128, 180, 129))) is False
