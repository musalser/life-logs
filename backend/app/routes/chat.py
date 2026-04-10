
from typing import List
from datetime import datetime

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from ..schemas import ChatRequest, ChatResponse, Entry as EntrySchema
from ..services.llama_client import LlamaService
from ..models import User, Entry
from ..config import settings
from ..deps import get_current_user, get_db, get_llama_service


HISTORY_LIMIT = 6

router = APIRouter(prefix="/chat", tags=["Chat"])



def _get_or_create_user(db: Session, tone: str) -> User:
    user = db.query(User).first()
    if not user:
        user = User(
            username=settings.auth_username,
            password_hash=settings.auth_password,
            name="User",
            tone=tone,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
    return user


def _recent_history(db: Session, user_id: int, limit: int = HISTORY_LIMIT):
    entries = (
        db.query(Entry)
        .filter(Entry.user_id == user_id)
        .order_by(Entry.id.desc())
        .limit(limit)
        .all()
    )
    entries.reverse()
    return [
        {
            "text": entry.text,
            "reply": entry.reply,
            "created_at": entry.created_at.isoformat() if entry.created_at else "",
        }
        for entry in entries
    ]


@router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    db: Session = Depends(get_db),
    llama_service: LlamaService | None = Depends(get_llama_service),
):
    user = _get_or_create_user(db, request.tone)
    history_payload = _recent_history(db, user.id)

    if llama_service is None:
        return ChatResponse(
            reply="Ошибка: LLM сервис недоступен",
            created_at=datetime.utcnow()
        )

    if request.stream:

        async def event_generator():
            reply_chunks: List[str] = []
            async for chunk in llama_service.stream_reply(request.message, request.tone, history_payload):
                reply_chunks.append(chunk)
                yield chunk

            full_reply = "".join(reply_chunks)
            entry = Entry(user_id=user.id, text=request.message, reply=full_reply)
            db.add(entry)
            db.commit()

        return StreamingResponse(event_generator(), media_type="text/plain")

    reply = await llama_service.generate_reply(request.message, request.tone, history_payload)
    entry = Entry(user_id=user.id, text=request.message, reply=reply)
    db.add(entry)
    db.commit()
    return ChatResponse(reply=reply, created_at=datetime.utcnow())



@router.get("/history", response_model=List[EntrySchema])
def get_chat_history(
    limit: int = 50,
    db: Session = Depends(get_db),
    _: str = Depends(get_current_user),
):
    user = db.query(User).first()
    if not user:
        return []

    entries = (
        db.query(Entry)
        .filter(Entry.user_id == user.id)
        .order_by(Entry.id.desc())
        .limit(limit)
        .all()
    )

    return list(reversed(entries))