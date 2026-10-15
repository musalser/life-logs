
import json
import logging
from typing import List
from datetime import datetime

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from ..schemas import ChatRequest, ChatResponse, Entry as EntrySchema
from ..adapters.ollama_adapter import OllamaAdapter
from ..models import User, Entry
from ..config import settings
from ..deps import get_current_user, get_db, get_ollama_adapter
from ..services.rag_service import RagService


HISTORY_LIMIT = 6

router = APIRouter(prefix="/chat", tags=["Chat"])
logger = logging.getLogger(__name__)



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


#: `/chat` (как зовёт фронтенд) и `/chat/` (как звали раньше) — оба пути.
#: Один лишь `/chat/` заставлял FastAPI отвечать 307 на POST /chat, а браузер
#: на таком редиректе теряет тело потокового запроса: сеть показывает
#: net::ERR_ABORTED, и чат отвечает «Дневник не смог ответить».
@router.post("", response_model=ChatResponse, include_in_schema=False)
@router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    db: Session = Depends(get_db),
    ai_adapter: OllamaAdapter = Depends(get_ollama_adapter),
):
    logger.info(
        "Processing chat request with stream=%s tone=%s rag=%s",
        request.stream, request.tone, request.rag,
    )
    user = _get_or_create_user(db, request.tone)
    history_payload = _recent_history(db, user.id)

    # RAG: вопрос ищется по архиву, найденные фрагменты уходят в промпт, а
    # список использованных сохраняется вместе с ответом, чтобы UI показал
    # источники под сообщением (ТЗ 8.4)
    chunks: List[dict] = []
    if request.rag:
        try:
            chunks = await RagService(db).search(
                user.id, request.message, subject_entity_id=request.subject_entity_id
            )
        except Exception:
            logger.exception("RAG search failed for user_id=%s; answering without it", user.id)
    context_block = RagService.context_block(chunks)
    prompt_message = f"{context_block}\n\nВопрос:\n{request.message}" if context_block else request.message
    context_json = json.dumps(chunks, ensure_ascii=False) if chunks else None

    if ai_adapter is None:
        logger.warning("Chat request failed because AI adapter is unavailable")
        return ChatResponse(
            reply="Ошибка: LLM сервис недоступен",
            created_at=datetime.utcnow()
        )

    if request.stream:

        async def event_generator():
            reply_chunks: List[str] = []
            # тон и история — часть промпта чата: без них модель видела только
            # голое сообщение, и выбор тональности ни на что не влиял
            async for chunk in ai_adapter.stream_reply(prompt_message, request.tone, history_payload):
                reply_chunks.append(chunk)
                yield chunk

            full_reply = "".join(reply_chunks)
            entry = Entry(
                user_id=user.id,
                text=request.message,
                reply=full_reply,
                context_json=context_json,
            )
            db.add(entry)
            db.commit()
            logger.info("Completed streaming chat response for user_id=%s", user.id)

        return StreamingResponse(event_generator(), media_type="text/plain")

    reply = await ai_adapter.generate_reply(prompt_message, request.tone, history_payload)
    entry = Entry(
        user_id=user.id,
        text=request.message,
        reply=reply,
        context_json=context_json,
    )
    db.add(entry)
    db.commit()
    logger.info("Completed chat response for user_id=%s", user.id)
    return ChatResponse(reply=reply, created_at=datetime.utcnow())



@router.get("/history", response_model=List[EntrySchema])
def get_chat_history(
    limit: int = 50,
    db: Session = Depends(get_db),
    username: str = Depends(get_current_user),
):
    logger.info("Fetching chat history with limit=%s", limit)
    user = db.query(User).filter(User.username == username).first()
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
