from typing import AsyncGenerator, Mapping, Sequence

import ollama

from ..config import settings
from .system_prompt import build_chat_prompt

client = ollama.AsyncClient(host=settings.ollama_url)

#: one model for every LLM job (chat, titles, extraction); see config.llm_model
MODEL = settings.llm_model
KEEP_ALIVE = "30m"
GENERATION_OPTIONS = {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "num_predict": 512,
}


async def generate_reply(
    message: str,
    tone: str,
    history: Sequence[Mapping[str, str]] | None = None,
) -> str:
    prompt = build_chat_prompt(message, tone, history)
    response = await client.generate(
        model=MODEL,
        prompt=prompt,
        stream=False,
        think=settings.llm_thinking_enabled,
        options=GENERATION_OPTIONS,
        keep_alive=KEEP_ALIVE,
    )
    return response["response"]


async def stream_reply(
    message: str,
    tone: str,
    history: Sequence[Mapping[str, str]] | None = None,
) -> AsyncGenerator[str, None]:
    prompt = build_chat_prompt(message, tone, history)
    async for part in await client.generate(
        model=MODEL,
        prompt=prompt,
        stream=True,
        think=settings.llm_thinking_enabled,
        options=GENERATION_OPTIONS,
        keep_alive=KEEP_ALIVE,
    ):
        yield part["response"]
