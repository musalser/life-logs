import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session, joinedload

from ..deps import get_current_user, get_db
from ..models import Entity, EntityRelation, Event, Goal, Habit, User
from ..schemas import (
    EventResponse,
    GoalResponse,
    HabitResponse,
    KnowledgeResponse,
    RelationResponse,
)


router = APIRouter(prefix="/knowledge", tags=["Knowledge"])
logger = logging.getLogger(__name__)


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

    relations = (
        db.query(EntityRelation)
        .options(joinedload(EntityRelation.entity))
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
                entity_id=rel.entity_id,
                person_name=rel.entity.canonical_name if rel.entity else "?",
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
