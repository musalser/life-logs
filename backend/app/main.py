
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .db import Base, engine
from .routes import auth, chat, diary
from .config import settings
from .deps import get_llama_service


# Base.metadata.create_all(bind=engine)

app = FastAPI(title="Live Journal AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router)
app.include_router(auth.router)
app.include_router(diary.router)


@app.on_event("startup")
async def warmup_llama():
    """Инициализация LlamaService при старте приложения для избежания задержек при первом запросе."""
    llama_service = get_llama_service()
    if llama_service is not None:
        print("✓ LlamaService прогрет и готов к использованию")
    else:
        print("⚠ LlamaService не инициализирован (модель может быть недоступна)")


@app.get("/")
def root():
    return {"message": "LIFE LOGS backend is running!"}
