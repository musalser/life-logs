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