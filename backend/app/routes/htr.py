import logging
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from ..config import settings
from ..deps import get_current_user, get_db
from ..htr import factory
from ..htr.application.confidence import ConfidencePolicy
from ..htr.application.page_service import HandwritingPageService
from ..htr.application.training_service import HandwritingTrainingService
from ..htr.domain.entities import (
    BoundingBox,
    PageSummary,
    PageView,
    RecognitionResult,
    RecognizedLine,
    RecognizedWord,
    TrainingResult,
)
from ..htr.domain.errors import (
    CorruptImageError,
    HTRError,
    InvalidTranscriptionError,
    NotFoundError,
    PageStateError,
    RecognitionError,
)
from ..htr.schemas import (
    WordAlternativeSchema,
    AuthorCreateRequest,
    AuthorResponse,
    BoundingBoxSchema,
    ConfidenceThresholdsResponse,
    LineResponse,
    LineUpdateRequest,
    ModelVersionResponse,
    PageNameRequest,
    PageOrderRequest,
    PageResponse,
    PageSummaryResponse,
    PageUploadResponse,
    RecognitionResultIn,
    SuggestionChangeSchema,
    SuggestionsResponse,
    TrainingResultResponse,
    WordResponse,
    WordUpdateRequest,
)
from ..models import HTRAuthor, User

router = APIRouter(prefix="/htr", tags=["Handwriting"])
logger = logging.getLogger(__name__)


def get_page_service(db: Session = Depends(get_db)) -> HandwritingPageService:
    return factory.build_page_service(db)


def get_training_service(db: Session = Depends(get_db)) -> HandwritingTrainingService:
    return factory.build_training_service(db)


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
    return author


@router.get("/authors", response_model=list[AuthorResponse])
def list_authors(
    db: Session = Depends(get_db),
    username: str = Depends(get_current_user),
):
    user = _get_user(db, username)
    return db.query(HTRAuthor).filter(HTRAuthor.user_id == user.id).order_by(HTRAuthor.id).all()


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
    db: Session = Depends(get_db),
    username: str = Depends(get_current_user),
    page_service: HandwritingPageService = Depends(get_page_service),
):
    user = _get_user(db, username)
    _get_author(db, author_id, user)
    content = await file.read()
    try:
        page = page_service.upload_page(
            user.id,
            author_id,
            file.filename or "page.png",
            content,
            source_path=source_path,
        )
    except CorruptImageError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    return PageUploadResponse(page_id=page.id, status=page.status.value)


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
    """
    user = _get_user(db, username)
    _get_owned_page(page_service, page_id, user)
    try:
        page = page_service.recognize_page(page_id, force=force)
    except RecognitionError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)
        )
    except PageStateError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    return _to_page_response(page)


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
):
    """Removes the page, its lines/words and its image/crops.

    Because every fine-tune rebuilds the corpus from the confirmed pages that
    exist at that moment, deleting a bad page keeps it out of all later
    trainings. Model versions that were trained before the deletion keep their
    dataset hash and are not touched.
    """
    user = _get_user(db, username)
    _get_owned_page(page_service, page_id, user)
    page_service.delete_page(page_id)
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


@router.post(
    "/pages/{page_id}/confirm",
    response_model=PageResponse,
    summary="Confirm the page as ground truth (does not trigger training)",
)
def confirm_page(
    page_id: int,
    db: Session = Depends(get_db),
    username: str = Depends(get_current_user),
    page_service: HandwritingPageService = Depends(get_page_service),
):
    """Marks the page as confirmed ground truth and measures its prediction error.

    Training is deliberately *not* started here: confirming several pages in a
    row must not pay for a fine-tune each time. The client calls
    ``POST /htr/authors/{author_id}/train`` when it wants a new model version.
    """
    user = _get_user(db, username)
    _get_owned_page(page_service, page_id, user)
    try:
        page = page_service.confirm_page(page_id)
    except InvalidTranscriptionError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except PageStateError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    return _to_page_response(page)


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
