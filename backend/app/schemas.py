from pydantic import BaseModel
from datetime import datetime


class ChatRequest(BaseModel):
    message: str
    tone: str = "coach"
    stream: bool = True


class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(BaseModel):
    username: str
    password: str
    name: str | None = None


class UserResponse(BaseModel):
    id: int
    username: str
    name: str
    tone: str
    created_at: datetime

    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class ChatResponse(BaseModel):
    reply: str
    created_at: datetime


class DiaryPageCreateRequest(BaseModel):
    content: str


class DiaryPageUpdateRequest(BaseModel):
    content: str


class DiaryPageResponse(BaseModel):
    id: int
    user_id: int
    title: str | None = None
    content: str
    created_at: datetime

    class Config:
        from_attributes = True


class Entry(BaseModel):
    id: int
    user_id: int
    text: str
    reply: str
    created_at: datetime

    class Config:
        from_attributes = True


class GoalProgressResponse(BaseModel):
    id: int
    diary_page_id: int
    progress_kind: str
    note: str | None = None
    source_text: str | None = None
    confidence: float | None = None
    created_at: datetime

    class Config:
        from_attributes = True


class GoalResponse(BaseModel):
    id: int
    title: str
    description: str | None = None
    status: str
    created_at: datetime
    updated_at: datetime | None = None
    progress_notes: list[GoalProgressResponse] = []

    class Config:
        from_attributes = True


class RelationResponse(BaseModel):
    id: int
    entity_id: int
    person_name: str
    relation_type: str
    confidence: float | None = None
    evidence_text: str | None = None
    updated_at: datetime | None = None

    class Config:
        from_attributes = True


class EventResponse(BaseModel):
    id: int
    diary_page_id: int
    title: str
    description: str | None = None
    time_text: str | None = None
    importance: float | None = None
    source_text: str | None = None
    created_at: datetime

    class Config:
        from_attributes = True


class HabitLogResponse(BaseModel):
    id: int
    diary_page_id: int
    note: str | None = None
    source_text: str | None = None
    created_at: datetime

    class Config:
        from_attributes = True


class HabitResponse(BaseModel):
    id: int
    title: str
    created_at: datetime
    logs: list[HabitLogResponse] = []

    class Config:
        from_attributes = True


class KnowledgeResponse(BaseModel):
    goals: list[GoalResponse] = []
    relations: list[RelationResponse] = []
    events: list[EventResponse] = []
    habits: list[HabitResponse] = []


class ExtractionSummaryResponse(BaseModel):
    page_id: int
    summary: dict