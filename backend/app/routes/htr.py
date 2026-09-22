import logging

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from ..config import settings
from ..deps import get_current_user, get_db
from ..htr import factory
from ..htr.application.confidence import ConfidencePolicy
from ..htr.application.page_service import HandwritingPageService
from ..htr.application.training_service import HandwritingTrainingService
from ..htr.domain.entities import (
    BoundingBox,
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
    AuthorCreateRequest,
    AuthorResponse,
    BoundingBoxSchema,
    ConfidenceThresholdsResponse,
    LineResponse,
    LineUpdateRequest,
    ModelVersionResponse,
    PageConfirmResponse,
    PageResponse,
    PageUploadResponse,
    RecognitionResultIn,
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
        lines=[
            LineResponse(
                id=line.id,
                order=line.order,
                bbox=BoundingBoxSchema(**line.bbox.__dict__),
                predicted_text=line.predicted_text,
                corrected_text=line.corrected_text,
                effective_text=line.effective_text,
                words_stale=line.words_stale,
                words=[
                    WordResponse(
                        id=word.id,
                        order=word.order,
                        bbox=BoundingBoxSchema(**word.bbox.__dict__),
                        predicted_text=word.predicted_text,
                        corrected_text=word.corrected_text,
                        effective_text=word.effective_text,
                        confidence=word.confidence,
                        confidence_level=policy.classify(word.confidence).value,
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
    db: Session = Depends(get_db),
    username: str = Depends(get_current_user),
    page_service: HandwritingPageService = Depends(get_page_service),
):
    user = _get_user(db, username)
    _get_author(db, author_id, user)
    content = await file.read()
    try:
        page = page_service.upload_page(user.id, author_id, file.filename or "page.png", content)
    except CorruptImageError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    return PageUploadResponse(page_id=page.id, status=page.status.value)


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
    db: Session = Depends(get_db),
    username: str = Depends(get_current_user),
    page_service: HandwritingPageService = Depends(get_page_service),
):
    """Runs the HTR engine on the stored page image and stores the prediction.

    Uses the author's active model when one exists, otherwise the configured
    default recognition model. Synchronous for now: page-level recognition takes
    seconds, and the service is HTTP-agnostic so it can move to a worker later.
    """
    user = _get_user(db, username)
    _get_owned_page(page_service, page_id, user)
    try:
        page = page_service.recognize_page(page_id)
    except RecognitionError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)
        )
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
    page = _get_owned_page(page_service, page_id, user)
    return _to_page_response(page)


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


@router.post("/pages/{page_id}/confirm", response_model=PageConfirmResponse)
def confirm_page(
    page_id: int,
    db: Session = Depends(get_db),
    username: str = Depends(get_current_user),
    page_service: HandwritingPageService = Depends(get_page_service),
    training_service: HandwritingTrainingService = Depends(get_training_service),
):
    user = _get_user(db, username)
    _get_owned_page(page_service, page_id, user)
    try:
        page = page_service.confirm_page(page_id)
    except InvalidTranscriptionError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    except PageStateError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    # Synchronous for the first stage; the training service is HTTP-agnostic
    # and can later be dispatched to a background worker unchanged.
    training_result = training_service.train_author(page.author_id)
    return PageConfirmResponse(
        page=_to_page_response(page),
        training=_to_training_response(training_result),
    )


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
