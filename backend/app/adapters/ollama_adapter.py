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

    async def generate_structured(self, prompt: str, schema: dict) -> str:
        """Constrained generation: response is forced to match the JSON schema."""
        # gpt-oss: /api/generate с format возвращает пустой response (конфликт
        # грамматики с Harmony-шаблоном), поэтому используем chat API.
        response = await self.client.chat(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            format=schema,
            options={"temperature": 0.1, "top_p": 0.9, "num_predict": 2048},
            keep_alive=KEEP_ALIVE,
        )
        content = response["message"]["content"]
        if not content:
            logger.warning(
                "Empty structured response: done_reason=%s eval_count=%s",
                response.get("done_reason"),
                response.get("eval_count"),
            )
        return content

    async def stream_reply(self, prompt: str):
        async for part in await self.client.generate(
            model=MODEL,
            prompt=prompt,
            stream=True,
            options=GENERATION_OPTIONS,
            keep_alive=KEEP_ALIVE,
        ):
            yield part["response"]