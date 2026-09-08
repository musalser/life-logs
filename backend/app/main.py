
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.deps import get_ollama_adapter
from app.logging_utils import configure_logging, log_requests

from .config import settings
from .routes import auth, chat, diary, knowledge



# Base.metadata.create_all(bind=engine)
configure_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    ai_adapter = get_ollama_adapter()
    logger.info("Starting application warmup")
    await ai_adapter.warmup()
    logger.info("Application warmup completed")
    yield
    
app = FastAPI(title="Live Journal AI", lifespan=lifespan)
app.middleware("http")(log_requests)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router)
app.include_router(auth.router)
app.include_router(diary.router)
app.include_router(knowledge.router)

@app.get("/")
def root():
    return {"message": "LIFE LOGS backend is running!"}
