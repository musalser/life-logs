import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session, joinedload

from ..deps import get_current_user, get_db, get_knowledge_source_service
from ..models import DiaryPage, Entity, EntityRelation, Event, Goal, Habit, HTRPage, User
from ..schemas import (
    EventResponse,
    GoalResponse,
    HabitResponse,
    KnowledgeResponse,
    RelationResponse,
)
from ..services.knowledge_source_service import (
    KnowledgeSourceService,
    SourceError,
    source_status_payload,
)


router = APIRouter(prefix="/knowledge", tags=["Knowledge"])
logger = logging.getLogger(__name__)

SOURCE_TYPES = ("diary", "htr")


def _get_user_by_username(db: Session, username: str) -> User:
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )
    return user


@router.get("", response_model=KnowledgeResponse)
async def get_knowledge(
    db: Session = Depends(get_db),
    username: str = Depends(get_current_user),
):
    logger.info("Fetching knowledge for user %s", username)
    user = _get_user_by_username(db, username)

    goals = (
        db.query(Goal)
        .options(joinedload(Goal.progress_notes))
        .filter(Goal.user_id == user.id)
        .order_by(Goal.updated_at.desc())
        .all()
    )

    # Relations are graph edges: `from` is the perspective (the author of the
    # text), `to` is the person. Migrated and diary-derived edges start at the
    # account's own object, which is what this old response shape assumes; the
    # object selector and the path display come with stage 5.
    relations = (
        db.query(EntityRelation)
        .options(
            joinedload(EntityRelation.from_entity),
            joinedload(EntityRelation.to_entity),
        )
        .filter(EntityRelation.user_id == user.id)
        .order_by(EntityRelation.updated_at.desc())
        .all()
    )

    events = (
        db.query(Event)
        .filter(Event.user_id == user.id)
        .order_by(Event.id.desc())
        .limit(200)
        .all()
    )

    habits = (
        db.query(Habit)
        .options(joinedload(Habit.logs))
        .filter(Habit.user_id == user.id)
        .order_by(Habit.id.desc())
        .all()
    )

    return KnowledgeResponse(
        goals=[GoalResponse.model_validate(goal) for goal in goals],
        relations=[
            RelationResponse(
                id=rel.id,
                entity_id=rel.to_entity_id,
                person_name=rel.to_entity.canonical_name if rel.to_entity else "?",
                relation_type=rel.relation_type,
                confidence=rel.confidence,
                evidence_text=rel.evidence_text,
                updated_at=rel.updated_at,
            )
            for rel in relations
        ],
        events=[EventResponse.model_validate(event) for event in events],
        habits=[HabitResponse.model_validate(habit) for habit in habits],
    )


def _source_type_or_404(source_type: str) -> str:
    if source_type not in SOURCE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown source type {source_type!r}; expected one of {', '.join(SOURCE_TYPES)}",
        )
    return source_type


@router.get("/sources/{source_type}/{ref_id}/status")
async def get_source_status(
    source_type: str,
    ref_id: int,
    db: Session = Depends(get_db),
    username: str = Depends(get_current_user),
):
    """Extraction status of one source: the UI polls this after a confirmation."""
    user = _get_user_by_username(db, username)
    source = KnowledgeSourceService(db).get(_source_type_or_404(source_type), ref_id, user.id)
    if source is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Knowledge source not found for this page",
        )
    return source_status_payload(db, source)


@router.post("/sources/{source_type}/{ref_id}/extract")
async def extract_source(
    source_type: str,
    ref_id: int,
    db: Session = Depends(get_db),
    username: str = Depends(get_current_user),
    source_service: KnowledgeSourceService = Depends(get_knowledge_source_service),
):
    """Re-runs the pipeline over one source, refreshing its text first.

    The text is re-read because the point of a manual re-run is usually an edit
    that the automatic path missed: a diary page may have changed while a
    manuscript page is only re-read while it is still confirmed.
    """
    user = _get_user_by_username(db, username)
    source_type = _source_type_or_404(source_type)
    source = source_service.get(source_type, ref_id, user.id)
    if source is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Knowledge source not found for this page",
        )

    try:
        if source_type == "diary":
            page = (
                db.query(DiaryPage)
                .filter(DiaryPage.id == ref_id, DiaryPage.user_id == user.id)
                .first()
            )
            if page is not None:
                source, _, _ = source_service.refresh_from_diary_page(page)
        else:
            page = (
                db.query(HTRPage)
                .filter(HTRPage.id == ref_id, HTRPage.user_id == user.id)
                .first()
            )
            if page is not None and page.status == "CONFIRMED":
                source, _, _ = source_service.refresh_from_htr_page(page)
    except SourceError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))

    return await source_service.request_extraction(source)
