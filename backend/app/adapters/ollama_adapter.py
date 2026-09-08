# Адаптер для взаимодействия с Ollama API

import logging

import ollama

from app.config import settings

MODEL = "gpt-oss:20b"
KEEP_ALIVE = "30m"
GENERATION_OPTIONS = {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "max_tokens": 512,
}

logger = logging.getLogger(__name__)

class OllamaAdapter:
    def __init__(self):
        self.client = ollama.AsyncClient(host=settings.ollama_url)

    async def warmup(self) -> None:
        """Pre-load model weights so first user request has no cold-start delay."""
        try:
            logger.info("Warming up model %s...", MODEL)
            await self.client.generate(
                model=MODEL,
                prompt="warmup",
                stream=False,
                keep_alive=KEEP_ALIVE,
                options={"temperature": 0, "num_predict": 1},
            )
        except ollama.ResponseError as e:
            logger.warning("Error warming up model %s: %s", MODEL, e)

    async def generate_reply(self, prompt: str) -> str:
        response = await self.client.generate(
            model=MODEL,
            prompt=prompt,
            stream=False,
            options=GENERATION_OPTIONS,
            keep_alive=KEEP_ALIVE,
        )
        return response["response"]

    async def stream_reply(self, prompt: str):
        async for part in await self.client.generate(
            model=MODEL,
            prompt=prompt,
            stream=True,
            options=GENERATION_OPTIONS,
            keep_alive=KEEP_ALIVE,
        ):
            yield part["response"]