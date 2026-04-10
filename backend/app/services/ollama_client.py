from typing import AsyncGenerator, Mapping, Sequence

import ollama

from ..config import settings
from .system_prompt import common_part, critic, supportive, philosopher


client = ollama.AsyncClient(host=settings.ollama_url)

MODEL = "gpt-oss:20b"
GENERATION_OPTIONS = {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "max_tokens": 512,
}

PROMPT_TEMPLATE = (
    common_part + "\n" +
    """{tone}\n"""
    """История последних диалогов (от старых к новым):\n{history}\n\n"""
    """Пользователь сейчас пишет:\n"{message}"\n\n"""
)


def _format_history(history: Sequence[Mapping[str, str]] | None) -> str:
    if not history:
        return "— нет сохранённой истории"

    formatted = []
    for item in history:
        text = item.get("text", "")
        reply = item.get("reply", "")
        created_at = item.get("created_at")
        timestamp = f"[{created_at}]" if created_at else ""
        formatted.append(f"{timestamp}\nПользователь: {text}\nДневник: {reply}".strip())
    return "\n\n".join(formatted)


def _build_prompt(
    message: str,
    tone: str,
    history: Sequence[Mapping[str, str]] | None,
) -> str:
    history_block = _format_history(history)
    if tone == "critic":
        tone = critic
    elif tone == "supportive":
        tone = supportive
    elif tone == "philosopher":
        tone = philosopher
    return PROMPT_TEMPLATE.format(tone=tone, history=history_block, message=message)


async def generate_reply(
    message: str,
    tone: str,
    history: Sequence[Mapping[str, str]] | None = None,
) -> str:
    prompt = _build_prompt(message, tone, history)
    response = await client.generate(
        model=MODEL,
        prompt=prompt,
        stream=False,
        options=GENERATION_OPTIONS,
    )
    return response["response"]


async def stream_reply(
    message: str,
    tone: str,
    history: Sequence[Mapping[str, str]] | None = None,
) -> AsyncGenerator[str, None]:
    prompt = _build_prompt(message, tone, history)
    async for part in await client.generate(
        model=MODEL,
        prompt=prompt,
        stream=True,
        options=GENERATION_OPTIONS,
    ):
        yield part["response"]
