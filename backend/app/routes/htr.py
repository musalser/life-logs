import logging
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from ..config import settings
from ..deps import get_current_user, get_db, get_knowledge_source_service
from ..services.knowledge_source_service import KnowledgeSourceService, SourceError
from ..services.library_service import LibraryError, LibraryService
from ..schemas import PageKnowledgeLine, PageKnowledgeResponse
from ..services.source_text import line_for_offset
from ..htr import background, factory
from ..htr.application.confidence import ConfidencePolicy
from ..htr.application.page_service import HandwritingPageService
from ..htr.application.training_service import HandwritingTrainingService
from ..htr.domain.entities import (
    BoundingBox,
    PageStatus,
    PageSummary,
    PageView,
    RecognitionResult,
    RecognizedLine,
    RecognizedWord,
    TrainingResult,
)
from ..htr.domain.errors import (
    CorruptImageError,
    CorruptPdfError,
    DuplicatePageError,
    HTRError,
    InvalidTranscriptionError,
    NotFoundError,
    PageStateError,
    RecognitionError,
)
from ..htr.infrastructure.pdf import PdfRasterizer
from ..htr.schemas import (
    WordAlternativeSchema,
    AuthorCreateRequest,
    AuthorEntityRequest,
    AuthorLexiconResponse,
    AuthorResponse,
    BoundingBoxSchema,
    ConfidenceThresholdsResponse,
    ConfirmationPreviewResponse,
    LexiconWordResponse,
    LineResponse,
    LineUpdateRequest,
    ModelVersionResponse,
    PageNameRequest,
    PageOrderRequest,
    PageConfirmResponse,
    PageResponse,
    PageSummaryResponse,
    PageUploadResponse,
    PdfUploadResponse,
    RecognitionResultIn,
    SuggestionChangeSchema,
    SuggestionsResponse,
    TrainingResultResponse,
    WordResponse,
    WordUpdateRequest,
)
from ..models import (
    Entity,
    EntityMention,
    EntityRelation,
    RelationSource,
    Fact,
    FactSource,
    HTRAuthor,
    HTRLexiconIgnore,
    HTRLine,
    HTRModelVersion,
    HTRPage,
    KnowledgeSource,
    User,
)

router = APIRouter(prefix="/htr", tags=["Handwriting"])
logger = logging.getLogger(__name__)


def get_page_service(db: Session = Depends(get_db)) -> HandwritingPageService:
    return factory.build_page_service(db)


def get_training_service(db: Session = Depends(get_db)) -> HandwritingTrainingService:
    return factory.build_training_service(db)


def get_pdf_rasterizer() -> PdfRasterizer:
    return factory.build_pdf_rasterizer()


def _get_user(db: Session, username: str) -> User:
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user


def _get_author(db: Session, author_id: int, user: User) -> HTRAuthor:
    author = (
        db.query(HTRAuthor)
        .filter(HTRAuthor.id == author_id, HTRAuthor.user_id == user.id)
        .first()
    )
    if not author:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Author not found")
    return author


def _get_owned_page(page_service: HandwritingPageService, page_id: int, user: User) -> PageView:
    page = page_service.page_repository.get_page(page_id)
    if page is None or page.user_id != user.id:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Page not found")
    return page


def _require_author_folder(db: Session, user: User, folder_id: int, author: HTRAuthor) -> None:
    """Refuse an upload into a folder that is not this author's."""
    try:
        LibraryService(db).folder_for_upload(user, folder_id, author.entity_id)
    except LibraryError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))


def _pdf_page_names(pdf_name: str, number: int) -> tuple[str, str]:
    """Storage name and client path of page ``number`` (1-based) of ``pdf_name``.

    The path names the PDF as the page's source directory: it keeps the pages of
    two same-named PDFs apart and makes re-importing one PDF a duplicate. The
    base name of that path is what the sidebar shows.
    """
    return f"page-{number:04d}.png", f"{pdf_name}/Стр. {number}.png"


def _author_response(db: Session, author: HTRAuthor) -> AuthorResponse:
    """Автор рукописи вместе с объектом, от лица которого читаются знания."""
    entity = (
        db.query(Entity).filter(Entity.id == author.entity_id).first()
        if author.entity_id
        else None
    )
    return AuthorResponse(
        id=author.id,
        name=author.name,
        created_at=author.created_at,
        entity_id=entity.id if entity else None,
        entity_name=entity.canonical_name if entity else None,
    )


def _to_page_response(page: PageView) -> PageResponse:
    policy: ConfidencePolicy = factory.build_confidence_policy()
    return PageResponse(
        page_id=page.id,
        author_id=page.author_id,
        status=page.status.value,
        file_path=page.file_path,
        file_name=page.file_name,
        source_path=page.source_path,
        order_index=page.order_index,
        width=page.width,
        height=page.height,
        created_at=page.created_at,
        confirmed_at=page.confirmed_at,
        recognition_model_version_id=page.recognition_model_version_id,
        prediction_cer=page.prediction_cer,
        prediction_wer=page.prediction_wer,
        confidence_thresholds=ConfidenceThresholdsResponse(
            warning=policy.warning_threshold,
            critical=policy.critical_threshold,
        ),
        oov_count=page.oov_count,
        lexicon_available=page.lexicon_available,
        lines=[
            LineResponse(
                id=line.id,
                order=line.order,
                bbox=BoundingBoxSchema(**line.bbox.__dict__),
                polygon=[list(point) for point in line.polygon] if line.polygon else None,
                predicted_text=line.predicted_text,
                corrected_text=line.corrected_text,
                corrected_by=line.corrected_by,
                effective_text=line.effective_text,
                words_stale=line.words_stale,
                suggested_text=line.suggested_text,
                suggested_by=line.suggested_by,
                suggestion_changes=[
                    SuggestionChangeSchema(
                        before=change.before,
                        after=change.after,
                        in_lexicon=change.in_lexicon,
                    )
                    for change in line.suggestion_changes
                ],
                suggestion_verified=line.suggestion_verified,
                oov_count=line.oov_count,
                oov_words=line.oov_words,
                author_only_count=line.author_only_count,
                author_only_words=line.author_only_words,
                words=[
                    WordResponse(
                        id=word.id,
                        order=word.order,
                        bbox=BoundingBoxSchema(**word.bbox.__dict__),
                        polygon=[list(point) for point in word.polygon] if word.polygon else None,
                        predicted_text=word.predicted_text,
                        corrected_text=word.corrected_text,
                        effective_text=word.effective_text,
                        confidence=word.confidence,
                        confidence_level=policy.classify(word.confidence).value,
                        alternatives=[
                            WordAlternativeSchema(text=item.text, score=item.score)
                            for item in word.alternatives
                        ],
                        in_lexicon=word.in_lexicon,
                        author_only=word.author_only,
                    )
                    for word in line.words
                ],
            )
            for line in page.lines
        ],
    )


def _to_training_response(result: TrainingResult) -> TrainingResultResponse:
    version = None
    if result.model_version is not None:
        v = result.model_version
        version = ModelVersionResponse(
            id=v.id,
            author_id=v.author_id,
            version=v.version,
            base_model_id=v.base_model_id,
            file_path=v.file_path,
            status=v.status.value,
            dataset_hash=v.dataset_hash,
            metrics=v.metrics,
            training_config=v.training_config,
            created_at=v.created_at,
        )
    return TrainingResultResponse(
        outcome=result.outcome.value,
        author_id=result.author_id,
        message=result.message,
        dataset_hash=result.dataset_hash,
        training_run_id=result.training_run_id,
        metrics=result.metrics,
        model_version=version,
        lines_collected=result.lines_collected,
        lines_required=result.lines_required,
        words_collected=result.words_collected,
        words_required=result.words_required,
    )


# ---------------------------------------------------------------------------
# Authors
# ---------------------------------------------------------------------------


@router.post("/authors", response_model=AuthorResponse, status_code=status.HTTP_201_CREATED)
def create_author(
    request: AuthorCreateRequest,
    db: Session = Depends(get_db),
    username: str = Depends(get_current_user),
):
    user = _get_user(db, username)
    author = HTRAuthor(user_id=user.id, name=request.name)
    db.add(author)
    db.commit()
    db.refresh(author)
    # у автора сразу есть объект: без него конвейер не сможет отнести к кому-то
    KnowledgeSourceService(db).ensure_author_entity(author)
    db.commit()
    return _author_response(db, author)


@router.get("/authors", response_model=list[AuthorResponse])
def list_authors(
    db: Session = Depends(get_db),
    username: str = Depends(get_current_user),
):
    user = _get_user(db, username)
    authors = db.query(HTRAuthor).filter(HTRAuthor.user_id == user.id).order_by(HTRAuthor.id).all()
    return [_author_response(db, author) for author in authors]


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


@router.post(
    "/authors/{author_id}/pages",
    response_model=PageUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_page(
    author_id: int,
    file: UploadFile = File(...),
    source_path: str | None = Form(default=None),
    folder_id: int | None = Form(default=None),
    db: Session = Depends(get_db),
    username: str = Depends(get_current_user),
    page_service: HandwritingPageService = Depends(get_page_service),
):
    user = _get_user(db, username)
    author = _get_author(db, author_id, user)
    if folder_id is not None:
        _require_author_folder(db, user, folder_id, author)
    content = await file.read()
    try:
        page = page_service.upload_page(
            user.id,
            author_id,
            file.filename or "page.png",
            content,
            source_path=source_path,
            folder_id=folder_id,
        )
    except CorruptImageError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except DuplicatePageError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    background.enqueue_pages([page.id])
    return PageUploadResponse(page_id=page.id, status=page.status.value)


@router.post(
    "/authors/{author_id}/pdf",
    response_model=PdfUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_pdf(
    author_id: int,
    file: UploadFile = File(...),
    parent_folder_id: int | None = Form(default=None),
    db: Session = Depends(get_db),
    username: str = Depends(get_current_user),
    page_service: HandwritingPageService = Depends(get_page_service),
    rasterizer: PdfRasterizer = Depends(get_pdf_rasterizer),
):
    """Every page of the PDF becomes a page of the author, in one folder.

    The folder is named after the file (without the extension) and is reused
    when it already exists, so importing the same PDF again continues that
    folder instead of piling up «Дневник (2)». The import is all-or-nothing: it
    is refused while any of its pages is already in the corpus, and a PDF that
    fails halfway leaves neither pages nor an empty folder behind.
    """
    user = _get_user(db, username)
    author = _get_author(db, author_id, user)
    # a folder belongs to an author *object*; an author that predates the
    # binding gets one here, exactly as creating an author does
    KnowledgeSourceService(db).ensure_author_entity(author)
    db.commit()
    if parent_folder_id is not None:
        _require_author_folder(db, user, parent_folder_id, author)
    content = await file.read()
    if not content:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Пустой файл")
    pdf_name = Path(file.filename or "document.pdf").name
    folder_name = Path(pdf_name).stem.strip() or "PDF"
    try:
        page_count = rasterizer.page_count(content)
    except CorruptPdfError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    if page_count == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="В PDF нет страниц")
    if page_count > settings.htr_pdf_max_pages:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"В PDF {page_count} страниц — это больше предела "
                f"{settings.htr_pdf_max_pages}. Разделите файл и загрузите частями."
            ),
        )
    # the whole file is refused when one of its pages is already here: a half
    # import would look like a successful one in the tree
    duplicates = [
        number
        for number in range(1, page_count + 1)
        if page_service.find_page_by_upload_name(author_id, *_pdf_page_names(pdf_name, number))
    ]
    if duplicates:
        shown = ", ".join(str(number) for number in duplicates[:5])
        more = "" if len(duplicates) <= 5 else f" и ещё {len(duplicates) - 5}"
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Страницы этого PDF уже загружены (стр. {shown}{more}). "
                "Удалите их, если нужно загрузить файл заново."
            ),
        )
    try:
        folder, folder_created = LibraryService(db).ensure_folder(
            user, folder_name, author.entity_id, parent_folder_id
        )
    except LibraryError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))

    created: list[int] = []
    try:
        for number, image in enumerate(rasterizer.render_pages(content), start=1):
            filename, source_path = _pdf_page_names(pdf_name, number)
            page = page_service.upload_page(
                user.id,
                author_id,
                filename,
                image,
                source_path=source_path,
                folder_id=folder.id,
            )
            created.append(page.id)
    except (CorruptPdfError, CorruptImageError, DuplicatePageError) as exc:
        # all-or-nothing: a page that failed must not leave its predecessors
        # behind — nor an empty folder the import itself created
        for page_id in created:
            page_service.delete_page(page_id)
        if folder_created:
            try:
                LibraryService(db).delete_folder(user, folder.id)
            except LibraryError:  # pragma: no cover - nothing else is in it
                logger.warning("Could not remove the folder of a failed PDF import: %s", folder.id)
        code = (
            status.HTTP_409_CONFLICT
            if isinstance(exc, DuplicatePageError)
            else status.HTTP_400_BAD_REQUEST
        )
        raise HTTPException(status_code=code, detail=str(exc))
    logger.info(
        "PDF imported: user_id=%s author_id=%s folder_id=%s pages=%s file_name=%s",
        user.id, author_id, folder.id, len(created), pdf_name,
    )
    background.enqueue_pages(created)
    return PdfUploadResponse(
        folder_id=folder.id,
        folder_name=folder.name,
        page_count=len(created),
        page_ids=created,
    )


def _to_page_summary(summary: PageSummary) -> PageSummaryResponse:
    return PageSummaryResponse(
        page_id=summary.id,
        author_id=summary.author_id,
        status=summary.status.value,
        created_at=summary.created_at,
        confirmed_at=summary.confirmed_at,
        line_count=summary.line_count,
        prediction_cer=summary.prediction_cer,
        prediction_wer=summary.prediction_wer,
        oov_count=summary.oov_count,
        lexicon_available=summary.lexicon_available,
        file_name=summary.file_name,
        source_path=summary.source_path,
        file_path=summary.file_path,
        order_index=summary.order_index,
    )


@router.get("/authors/{author_id}", response_model=AuthorResponse)
def get_author(
    author_id: int,
    db: Session = Depends(get_db),
    username: str = Depends(get_current_user),
):
    """Автор рукописи: с привязкой к объекту (entity_id, entity_name)."""
    user = _get_user(db, username)
    author = _get_author(db, author_id, user)
    return _author_response(db, author)


@router.put("/authors/{author_id}/entity", response_model=AuthorResponse)
def bind_author_entity(
    author_id: int,
    request: AuthorEntityRequest,
    db: Session = Depends(get_db),
    username: str = Depends(get_current_user),
):
    """Привязывает автора рукописи к известному объекту.

    Именно от этого объекта читаются связи и факты страниц автора: «моя мама» в
    его рукописи — его мама, а не читателя. Перепривязка касается будущих
    извлечений; уже извлечённое знание остаётся у прежнего объекта до
    переизвлечения страниц.
    """
    user = _get_user(db, username)
    author = _get_author(db, author_id, user)
    entity = (
        db.query(Entity)
        .filter(Entity.id == request.entity_id, Entity.user_id == user.id)
        .first()
    )
    if entity is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Knowledge object not found",
        )
    author.entity_id = entity.id
    db.commit()
    db.refresh(author)
    return _author_response(db, author)


@router.get("/authors/{author_id}/pages", response_model=list[PageSummaryResponse])
def list_author_pages(
    author_id: int,
    db: Session = Depends(get_db),
    username: str = Depends(get_current_user),
    page_service: HandwritingPageService = Depends(get_page_service),
):
    """Pages of one author in their stored (draggable) order."""
    user = _get_user(db, username)
    _get_author(db, author_id, user)
    summaries = page_service.list_page_summaries(author_id)
    return [_to_page_summary(s) for s in summaries]


@router.put(
    "/authors/{author_id}/pages/order",
    response_model=list[PageSummaryResponse],
    summary="Store the sidebar order of the author's pages",
)
def reorder_author_pages(
    author_id: int,
    request: PageOrderRequest,
    db: Session = Depends(get_db),
    username: str = Depends(get_current_user),
    page_service: HandwritingPageService = Depends(get_page_service),
):
    """Apply drag-and-drop order.

    ``page_ids`` is the full visible order. Ids the client did not send (for
    example a page uploaded in another tab) keep their relative order at the
    end instead of being dropped from the list.
    """
    user = _get_user(db, username)
    _get_author(db, author_id, user)
    try:
        summaries = page_service.reorder_pages(author_id, request.page_ids)
    except NotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    return [_to_page_summary(s) for s in summaries]


# ---------------------------------------------------------------------------
# The author's own vocabulary (dictionary panel)
# ---------------------------------------------------------------------------


def _to_lexicon_response(payload: dict) -> AuthorLexiconResponse:
    return AuthorLexiconResponse(
        available=bool(payload.get("available")),
        words=[LexiconWordResponse(**item) for item in payload.get("words", [])],
            )


@router.get(
    "/authors/{author_id}/lexicon",
    response_model=AuthorLexiconResponse,
    summary="Words the author's confirmed pages contributed to the dictionary",
)
def get_author_lexicon(
    author_id: int,
    db: Session = Depends(get_db),
    username: str = Depends(get_current_user),
    page_service: HandwritingPageService = Depends(get_page_service),
):
    """The author's own vocabulary, most frequent first, plus the removed words.

    These are the words the check treats as known *because the author wrote
    them on a confirmed page*: names and dialect words, but also a typo that was
    confirmed once. Removing a word here makes it unknown again on every page.
    """
    user = _get_user(db, username)
    _get_author(db, author_id, user)
    return _to_lexicon_response(page_service.author_dictionary(author_id))


@router.get(
    "/pages/{page_id}/image",
    summary="Raw page image (for the annotation view)",
    response_class=FileResponse,
)
def get_page_image(
    page_id: int,
    db: Session = Depends(get_db),
    username: str = Depends(get_current_user),
    page_service: HandwritingPageService = Depends(get_page_service),
):
    user = _get_user(db, username)
    page = _get_owned_page(page_service, page_id, user)
    path = Path(page.file_path)
    if not path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Page image file is missing on the server",
        )
    return FileResponse(
        path,
        headers={"Cache-Control": "private, max-age=3600"},
    )


@router.post(
    "/pages/{page_id}/recognition-result",
    response_model=PageResponse,
    summary="Import an externally produced recognition result (not recognition)",
)
def import_recognition_result(
    page_id: int,
    request: RecognitionResultIn,
    db: Session = Depends(get_db),
    username: str = Depends(get_current_user),
    page_service: HandwritingPageService = Depends(get_page_service),
):
    """Stores a recognition result produced outside this service.

    This endpoint does *not* run recognition: it only persists line/word data
    (typically exported from another HTR tool). Geometry is validated, so an
    unfilled example payload is rejected instead of being stored as a
    prediction. To actually recognize a page use
    ``POST /htr/pages/{page_id}/recognize``.
    """
    user = _get_user(db, username)
    page = _get_owned_page(page_service, page_id, user)
    result = RecognitionResult(
        page_width=request.page_width or page.width or 0,
        page_height=request.page_height or page.height or 0,
        lines=[
            RecognizedLine(
                id=str(i),
                bbox=BoundingBox(**line.bbox.model_dump()),
                text=line.text,
                words=[
                    RecognizedWord(
                        id=f"{i}:{j}",
                        bbox=BoundingBox(**word.bbox.model_dump()),
                        text=word.text,
                        confidence=word.confidence,
                    )
                    for j, word in enumerate(line.words)
                ],
            )
            for i, line in enumerate(request.lines)
        ],
    )
    try:
        page = page_service.apply_recognition(page_id, result)
    except PageStateError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    return _to_page_response(page)


@router.post(
    "/pages/{page_id}/recognize",
    response_model=PageResponse,
    summary="Recognize the page image (segmentation + handwriting recognition)",
)
def recognize_page(
    page_id: int,
    force: bool = False,
    only_uploaded: bool = False,
    db: Session = Depends(get_db),
    username: str = Depends(get_current_user),
    page_service: HandwritingPageService = Depends(get_page_service),
):
    """Runs the HTR engine on the stored page image and stores the prediction.

    Uses the author's active model when one exists, otherwise the configured
    default recognition model. Synchronous for now: page-level recognition takes
    seconds, and the service is HTTP-agnostic so it can move to a worker later.

    ``force=true`` re-segments a page that is already in EDITING (for example
    after the segmenter changed); it replaces the prediction and drops that
    page's corrections.

    The background queue (``htr/background.py``) may be reading the same page:
    the route then waits for it and returns its result instead of a second run.
    ``only_uploaded=true`` (the client's "this page has no markup yet") also
    returns the page as it is when it was recognized in the meantime.
    """
    user = _get_user(db, username)
    _get_owned_page(page_service, page_id, user)
    try:
        with background.get_recognition_queue().page_lock(page_id) as queue_had_it:
            if queue_had_it or only_uploaded:
                # the queue committed in its own session: read the page anew
                db.expire_all()
                current = page_service.get_page(page_id)
                if current.status != PageStatus.UPLOADED and not force:
                    return _to_page_response(current)
            page = page_service.recognize_page(page_id, force=force)
    except RecognitionError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)
        )
    except PageStateError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    return _to_page_response(page)


@router.get(
    "/pages/{page_id}/recognition-status",
    summary="Is the page waiting for, or in, background recognition?",
)
def recognition_status(
    page_id: int,
    db: Session = Depends(get_db),
    username: str = Depends(get_current_user),
    page_service: HandwritingPageService = Depends(get_page_service),
):
    """``queued`` / ``running`` / ``idle`` plus the page status (cheap to poll)."""
    user = _get_user(db, username)
    page = _get_owned_page(page_service, page_id, user)
    state = (
        background.get_recognition_queue().state(page_id)
        if background.background_enabled()
        else background.IDLE
    )
    return {"page_id": page_id, "status": page.status.value, "background": state}


@router.post(
    "/pages/{page_id}/suggestions",
    response_model=PageResponse,
    summary="Ask the LLM for corrections and store them as proposals",
)
def suggest_corrections(
    page_id: int,
    db: Session = Depends(get_db),
    username: str = Depends(get_current_user),
    page_service: HandwritingPageService = Depends(get_page_service),
):
    """Runs the LLM over the raw recognition of the page.

    The transcription is **not** modified: each proposal is stored next to the
    line together with the word changes it would make, so the user reviews and
    accepts (or dismisses) it. Best-effort: when the LLM is unavailable the page
    comes back without proposals.
    """
    user = _get_user(db, username)
    _get_owned_page(page_service, page_id, user)
    # preflight: an unreachable Ollama must not cost a timeout per line before
    # the page comes back unchanged
    problem = page_service.corrector_status()
    if problem is not None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=problem
        )
    try:
        page = page_service.suggest_corrections(page_id)
    except PageStateError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    return _to_page_response(page)


@router.put(
    "/pages/{page_id}/lines/{line_id}/suggestion",
    response_model=PageResponse,
    summary="Accept the model proposal of one line",
)
def accept_suggestion(
    page_id: int,
    line_id: int,
    db: Session = Depends(get_db),
    username: str = Depends(get_current_user),
    page_service: HandwritingPageService = Depends(get_page_service),
):
    user = _get_user(db, username)
    _get_owned_page(page_service, page_id, user)
    try:
        page = page_service.accept_suggestion(page_id, line_id)
    except NotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except PageStateError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    return _to_page_response(page)


@router.put(
    "/pages/{page_id}/lines/{line_id}/suggestion/changes/{change_index}",
    response_model=PageResponse,
    summary="Accept one proposed word, leaving the rest of the proposal",
)
def accept_suggestion_change(
    page_id: int,
    line_id: int,
    change_index: int,
    db: Session = Depends(get_db),
    username: str = Depends(get_current_user),
    page_service: HandwritingPageService = Depends(get_page_service),
):
    """Applies a single word of a proposal.

    ``change_index`` is the position in ``suggestion_changes`` of that line, so
    the list the client shows is exactly the list it indexes into. The proposal
    itself is kept: after the write the remaining changes are still offered.
    """
    user = _get_user(db, username)
    _get_owned_page(page_service, page_id, user)
    try:
        page = page_service.accept_suggestion_change(page_id, line_id, change_index)
    except NotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except PageStateError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    return _to_page_response(page)


@router.post(
    "/pages/{page_id}/suggestions/accept",
    response_model=SuggestionsResponse,
    summary="Accept every proposal whose changes the dictionary confirms",
)
def accept_verified_suggestions(
    page_id: int,
    db: Session = Depends(get_db),
    username: str = Depends(get_current_user),
    page_service: HandwritingPageService = Depends(get_page_service),
):
    """Bulk accept, restricted to lines without an unverified change.

    A change is verified when the replacement word exists in the general
    dictionary or in the author's own vocabulary. Lines with anything else (an
    unknown word, a deletion, or no dictionary installed at all) are left for
    the user to read.
    """
    user = _get_user(db, username)
    _get_owned_page(page_service, page_id, user)
    try:
        page, accepted = page_service.accept_verified_suggestions(page_id)
    except PageStateError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    return SuggestionsResponse(page=_to_page_response(page), accepted=accepted)


@router.delete(
    "/pages/{page_id}/lines/{line_id}/suggestion",
    response_model=PageResponse,
    summary="Dismiss the model proposal of one line",
)
def dismiss_suggestion(
    page_id: int,
    line_id: int,
    db: Session = Depends(get_db),
    username: str = Depends(get_current_user),
    page_service: HandwritingPageService = Depends(get_page_service),
):
    user = _get_user(db, username)
    _get_owned_page(page_service, page_id, user)
    try:
        page = page_service.dismiss_suggestion(page_id, line_id)
    except NotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except PageStateError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    return _to_page_response(page)


@router.get("/pages/{page_id}", response_model=PageResponse)
def get_page(
    page_id: int,
    db: Session = Depends(get_db),
    username: str = Depends(get_current_user),
    page_service: HandwritingPageService = Depends(get_page_service),
):
    user = _get_user(db, username)
    _get_owned_page(page_service, page_id, user)
    return _to_page_response(page_service.get_page(page_id))


@router.patch(
    "/pages/{page_id}/name",
    response_model=PageResponse,
    summary="Rename the displayed file name of a page",
)
def rename_page(
    page_id: int,
    request: PageNameRequest,
    db: Session = Depends(get_db),
    username: str = Depends(get_current_user),
    page_service: HandwritingPageService = Depends(get_page_service),
):
    user = _get_user(db, username)
    _get_owned_page(page_service, page_id, user)
    file_name = request.file_name.strip()
    if not file_name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="File name must not be blank"
        )
    return _to_page_response(page_service.rename_page(page_id, file_name))


@router.delete(
    "/pages/{page_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a page (also excludes it from future training datasets)",
)
def delete_page(
    page_id: int,
    db: Session = Depends(get_db),
    username: str = Depends(get_current_user),
    page_service: HandwritingPageService = Depends(get_page_service),
    source_service: KnowledgeSourceService = Depends(get_knowledge_source_service),
):
    """Removes the page, its lines/words, its image/crops and what was read from it.

    Because every fine-tune rebuilds the corpus from the confirmed pages that
    exist at that moment, deleting a bad page keeps it out of all later
    trainings. Model versions that were trained before the deletion keep their
    dataset hash and are not touched.

    Знания уходят вместе со страницей — так же, как у дневниковой записи:
    ``KnowledgeSource.ref_id`` не имеет внешнего ключа, и без этой уборки
    упоминания, факты и рёбра оставались бы сиротами и продолжали показываться
    в «Знаниях». Страницу удаляем первой: если уборка не удастся, останется
    сирота (её можно вычистить), а не потерянные знания живой страницы.
    """
    user = _get_user(db, username)
    _get_owned_page(page_service, page_id, user)
    page_service.delete_page(page_id)
    source = source_service.get("htr", page_id, user.id)
    if source is not None:
        source_service.delete_source(source)
    return None


@router.patch("/pages/{page_id}/lines/{line_id}/words/{word_id}", response_model=PageResponse)
def update_word(
    page_id: int,
    line_id: int,
    word_id: int,
    request: WordUpdateRequest,
    db: Session = Depends(get_db),
    username: str = Depends(get_current_user),
    page_service: HandwritingPageService = Depends(get_page_service),
):
    user = _get_user(db, username)
    _get_owned_page(page_service, page_id, user)
    try:
        page = page_service.update_word(page_id, line_id, word_id, request.corrected_text)
    except NotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except PageStateError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    return _to_page_response(page)


@router.put("/pages/{page_id}/lines/{line_id}", response_model=PageResponse)
def update_line(
    page_id: int,
    line_id: int,
    request: LineUpdateRequest,
    db: Session = Depends(get_db),
    username: str = Depends(get_current_user),
    page_service: HandwritingPageService = Depends(get_page_service),
):
    user = _get_user(db, username)
    _get_owned_page(page_service, page_id, user)
    try:
        page = page_service.update_line(page_id, line_id, request.corrected_text)
    except NotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except PageStateError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    return _to_page_response(page)


@router.delete(
    "/pages/{page_id}/lines/{line_id}",
    response_model=PageResponse,
    summary="Delete one line from the page (renumbers the rest)",
)
def delete_line(
    page_id: int,
    line_id: int,
    db: Session = Depends(get_db),
    username: str = Depends(get_current_user),
    page_service: HandwritingPageService = Depends(get_page_service),
):
    """Removes a line the segmenter invented, or one that holds no text.

    The remaining lines are renumbered, because the UI numbers them by
    position. Only editable pages are accepted: a confirmed page is frozen
    ground truth and must be re-recognized to change its lines.
    """
    user = _get_user(db, username)
    _get_owned_page(page_service, page_id, user)
    try:
        page = page_service.delete_line(page_id, line_id)
    except NotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except PageStateError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    return _to_page_response(page)


@router.post(
    "/pages/{page_id}/reopen",
    response_model=PageResponse,
    summary="Return a confirmed page to editing (leaves the training corpus)",
)
def reopen_page(
    page_id: int,
    db: Session = Depends(get_db),
    username: str = Depends(get_current_user),
    page_service: HandwritingPageService = Depends(get_page_service),
    source_service: KnowledgeSourceService = Depends(get_knowledge_source_service),
):
    """Undo a confirmation.

    The page leaves the ground-truth corpus: the next training run ignores it and
    its words stop counting as the author's vocabulary. Corrections are kept, the
    transcription becomes editable again, and the prediction metrics of the
    confirmation are dropped. A model already trained on the page is untouched —
    a fine-tune cannot be undone.

    Its knowledge goes too: the text is not ground truth any more, so what was
    read from it must not be shown (spec 6.3).
    """
    user = _get_user(db, username)
    _get_owned_page(page_service, page_id, user)
    try:
        page = page_service.reopen_page(page_id)
    except PageStateError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    try:
        source = source_service.get("htr", page_id, user.id)
        if source is not None:
            source_service.mark_pending(source)
    except Exception:
        logger.exception("Could not reset the knowledge of page %s", page_id)
    return _to_page_response(page)


@router.get(
    "/pages/{page_id}/knowledge",
    response_model=PageKnowledgeResponse,
    summary="Знания, извлечённые из этой страницы, с привязкой к строкам",
)
def get_page_knowledge(
    page_id: int,
    db: Session = Depends(get_db),
    username: str = Depends(get_current_user),
):
    """Факты, упоминания и события страницы с переводом офсетов в строки.

    Офсет знания указывает в текст источника; карта строк переводит его в
    конкретную строку страницы (``line_id`` и её порядковый номер), а строка уже
    имеет координаты на изображении — это и есть «ссылка на реальный текст».
    """
    user = _get_user(db, username)
    page = (
        db.query(HTRPage)
        .filter(HTRPage.id == page_id, HTRPage.user_id == user.id)
        .first()
    )
    if page is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Page not found")

    service = KnowledgeSourceService(db)
    source = service.get("htr", page_id, user.id)
    if source is None:
        return PageKnowledgeResponse(page_id=page_id, source_id=None, status=None, items=[])

    order_by_line = {
        line.id: line.order_index
        for line in db.query(HTRLine.id, HTRLine.order_index)
        .filter(HTRLine.page_id == page_id)
        .all()
    }
    items: list[PageKnowledgeLine] = []

    def add(kind: str, record_id: int, text: str, quote: str | None, offset: int | None, confidence: float | None):
        line_id = line_for_offset(source.line_map, offset)
        items.append(
            PageKnowledgeLine(
                kind=kind,
                id=record_id,
                text=text,
                quote=quote,
                line_id=line_id,
                line_order=order_by_line.get(line_id) if line_id else None,
                confidence=confidence,
            )
        )

    # Факт может быть подтверждён на нескольких страницах: у каждой своя цитата
    # и офсеты, поэтому берём подтверждение именно этой страницы, а если его нет —
    # основную цитату факта (страница, где он был прочитан впервые).
    occurrences: dict[int, tuple[Fact, str | None, int | None, float | None]] = {}
    for fact in db.query(Fact).filter(Fact.source_id == source.id).all():
        occurrences.setdefault(fact.id, (fact, fact.source_text, fact.start_offset, fact.confidence))
    for quote_row in db.query(FactSource).filter(FactSource.source_id == source.id).all():
        if quote_row.fact is None:
            continue
        occurrences.setdefault(
            quote_row.fact_id,
            (quote_row.fact, quote_row.source_text, quote_row.start_offset, quote_row.confidence),
        )
    for fact, quote_text, offset, confidence in occurrences.values():
        subject = fact.subject_entity.canonical_name if fact.subject_entity else None
        obj = fact.object_entity.canonical_name if fact.object_entity else (fact.object_text or "")
        label = " ".join(part for part in (subject, fact.predicate, obj) if part)
        add("fact", fact.id, label.strip(), quote_text, offset, confidence)
    for mention in db.query(EntityMention).filter(EntityMention.source_id == source.id).all():
        entity = mention.entity.canonical_name if mention.entity else "?"
        add("mention", mention.id, entity, mention.mention_text, mention.start_offset, mention.confidence)
    # Ребро может быть подтверждено несколькими источниками, поэтому берём и его
    # собственный источник, и строки подтверждений: иначе страница не показала бы
    # связь, которую она подтверждает, но не «владеет» ею
    edge_occurrences: dict[int, tuple[EntityRelation, str | None, int | None, float | None]] = {}
    for edge in db.query(EntityRelation).filter(EntityRelation.source_id == source.id).all():
        edge_occurrences.setdefault(edge.id, (edge, edge.evidence_text, edge.start_offset, edge.confidence))
    for confirmation in (
        db.query(RelationSource).filter(RelationSource.source_id == source.id).all()
    ):
        if confirmation.relation is None:
            continue
        edge_occurrences.setdefault(
            confirmation.relation_id,
            (
                confirmation.relation,
                confirmation.evidence_text,
                confirmation.start_offset,
                confirmation.confidence,
            ),
        )
    for edge, quote_text, offset, confidence in edge_occurrences.values():
        person = edge.to_entity.canonical_name if edge.to_entity else "?"
        add("relation", edge.id, f"{person} — {edge.relation_type}", quote_text, offset, confidence)

    items.sort(key=lambda item: (item.line_order is None, item.line_order or 0, item.kind))
    return PageKnowledgeResponse(
        page_id=page_id,
        source_id=source.id,
        status=source.extraction_status,
        items=items,
    )


@router.get(
    "/pages/{page_id}/confirmation-preview",
    response_model=ConfirmationPreviewResponse,
    summary="What confirming the page would add to the author dictionary",
)
def preview_confirmation(
    page_id: int,
    db: Session = Depends(get_db),
    username: str = Depends(get_current_user),
    page_service: HandwritingPageService = Depends(get_page_service),
):
    """The words a confirmation is about to make known, without writing anything.

    The UI opens this before confirming: a word that is about to be taught to the
    dictionary by mistake is much easier to fix now than to meet later as an
    unexpected "known". The lists are the same ones ``POST .../confirm`` returns.
    """
    user = _get_user(db, username)
    _get_owned_page(page_service, page_id, user)
    try:
        preview = page_service.preview_confirmation(page_id)
    except InvalidTranscriptionError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except PageStateError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    return ConfirmationPreviewResponse(
        added_author_words=list(preview.added_author_words),
        author_words_learned=list(preview.learned_words),
    )


@router.post(
    "/pages/{page_id}/confirm",
    response_model=PageConfirmResponse,
    summary="Confirm the page as ground truth (does not trigger training)",
)
def confirm_page(
    page_id: int,
    db: Session = Depends(get_db),
    username: str = Depends(get_current_user),
    page_service: HandwritingPageService = Depends(get_page_service),
    source_service: KnowledgeSourceService = Depends(get_knowledge_source_service),
):
    """Marks the page as confirmed ground truth and measures its prediction error.

    Training is deliberately *not* started here: confirming several pages in a
    row must not pay for a fine-tune each time. The client calls
    ``POST /htr/authors/{author_id}/train`` when it wants a new model version.

    The response also lists the words the confirmation added to the author's own
    vocabulary, because that is the moment those words stop being unknown.

    The confirmed text becomes a knowledge source here, but it is **not**
    extracted: extraction is an LLM run of its own, and confirming a batch of
    pages must not pay for one per page. The source waits as ``pending`` until
    the user asks for it (``POST /htr/pages/{page_id}/knowledge/extract``).
    """
    user = _get_user(db, username)
    _get_owned_page(page_service, page_id, user)
    try:
        report = page_service.confirm_page_report(page_id)
    except InvalidTranscriptionError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except PageStateError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    response = PageConfirmResponse(**_to_page_response(report.page).model_dump())
    response.added_author_words = list(report.added_author_words)
    response.author_words_learned = list(report.learned_words)

    try:
        _, created, changed = source_service.refresh_from_htr_page(report.page)
        logger.info(
            "Knowledge source for page %s: created=%s changed=%s (extraction on request)",
            page_id, created, changed,
        )
    except Exception:
        logger.exception("Knowledge source could not be refreshed for page %s", page_id)
    return response


@router.post(
    "/pages/{page_id}/knowledge/extract",
    summary="Extract knowledge from a confirmed page (separate from confirming it)",
)
def extract_page_knowledge(
    page_id: int,
    db: Session = Depends(get_db),
    username: str = Depends(get_current_user),
    page_service: HandwritingPageService = Depends(get_page_service),
    source_service: KnowledgeSourceService = Depends(get_knowledge_source_service),
):
    """Runs the knowledge pipeline over the confirmed text of the page.

    Only confirmed text is ground truth, so an unconfirmed page is refused
    (``409``). The source is refreshed from the current lines first (it is
    created if the page was confirmed before sources existed), so a re-run
    always reads what the page says now. Sync mode returns when the pipeline has
    finished, async mode queues it and the UI polls
    ``GET /knowledge/sources/htr/{page_id}/status``.
    """
    user = _get_user(db, username)
    page = _get_owned_page(page_service, page_id, user)
    if page.status != PageStatus.CONFIRMED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Знания извлекаются только из подтверждённой страницы",
        )
    record = db.query(HTRPage).filter(HTRPage.id == page_id).first()
    try:
        source, _, _ = source_service.refresh_from_htr_page(record)
    except SourceError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    return source_service.request_extraction_sync(source)


# ---------------------------------------------------------------------------
# Training / models
# ---------------------------------------------------------------------------


@router.post("/authors/{author_id}/train", response_model=TrainingResultResponse)
def train_author(
    author_id: int,
    db: Session = Depends(get_db),
    username: str = Depends(get_current_user),
    training_service: HandwritingTrainingService = Depends(get_training_service),
):
    user = _get_user(db, username)
    _get_author(db, author_id, user)
    result = training_service.train_author(author_id)
    return _to_training_response(result)


@router.get("/authors/{author_id}/models", response_model=list[ModelVersionResponse])
def list_model_versions(
    author_id: int,
    db: Session = Depends(get_db),
    username: str = Depends(get_current_user),
):
    user = _get_user(db, username)
    _get_author(db, author_id, user)
    repo = factory.build_model_repository(db)
    return [
        ModelVersionResponse(
            id=v.id,
            author_id=v.author_id,
            version=v.version,
            base_model_id=v.base_model_id,
            file_path=v.file_path,
            status=v.status.value,
            dataset_hash=v.dataset_hash,
            metrics=v.metrics,
            training_config=v.training_config,
            created_at=v.created_at,
        )
        for v in repo.list_versions(author_id)
    ]
