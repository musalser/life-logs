import asyncio
from typing import Mapping, Sequence
from .celery_app import celery
from .services.ollama_client import generate_reply

@celery.task(name="app.tasks.generate_reply")
def generate_reply_task(
    message: str,
    tone: str,
    history: Sequence[Mapping[str, str]] | None = None,
) -> str:
    return asyncio.run(generate_reply(message, tone, history))