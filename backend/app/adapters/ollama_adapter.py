# Адаптер для взаимодействия с Ollama API

import json
import logging
import re
from typing import Any, Mapping, Sequence

import ollama

from app.config import settings
from app.services.system_prompt import build_chat_prompt

KEEP_ALIVE = "30m"
GENERATION_OPTIONS = {
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "num_predict": 512,
}
#: extraction must copy quotes verbatim and emit a strict JSON document, so it
#: runs much colder than the chat does
STRUCTURED_OPTIONS = {"temperature": 0.1, "top_p": 0.9, "num_predict": 2048}

logger = logging.getLogger(__name__)


def _is_valid_json(raw: str | None) -> bool:
    if not raw or not raw.strip():
        return False
    try:
        json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return False
    return True


def _first_json_object(raw: str | None) -> str | None:
    """Best-effort slice of the first ``{...}`` block of a noisy answer."""
    match = re.search(r"\{[\s\S]*\}", raw or "")
    return match.group(0) if match else None


class OllamaAdapter:
    def __init__(self) -> None:
        self.client = ollama.AsyncClient(host=settings.ollama_url)
        self.model = settings.llm_model
        self.keep_alive = KEEP_ALIVE

    async def warmup(self) -> None:
        """Pre-load model weights so first user request has no cold-start delay."""
        if not settings.llm_warmup_enabled:
            return
        try:
            logger.info("Warming up model %s...", self.model)
            await self.client.generate(
                model=self.model,
                prompt="warmup",
                stream=False,
                think=False,
                keep_alive=self.keep_alive,
                options={"temperature": 0, "num_predict": 1},
            )
        except ollama.ResponseError as e:
            logger.warning("Error warming up model %s: %s", self.model, e)

    async def generate_reply(
        self,
        prompt: str,
        tone: str | None = None,
        history: Sequence[Mapping[str, str]] | None = None,
    ) -> str:
        """One-shot answer.

        With a ``tone`` the prompt is the chat prompt (message + history); used
        by the diary title generator without it. Titles and other one-shot
        calls never think — only the chat does.
        """
        if tone is not None:
            prompt = build_chat_prompt(prompt, tone, history)
        response = await self.client.generate(
            model=self.model,
            prompt=prompt,
            stream=False,
            think=settings.llm_thinking_enabled if tone is not None else False,
            options=GENERATION_OPTIONS,
            keep_alive=self.keep_alive,
        )
        return response["response"]

    async def generate_structured(self, prompt: str, schema: dict) -> str:
        """Constrained generation: the answer is forced to match the schema.

        Structured output is the contract the extraction services parse, so a
        model that ignores the grammar (or an Ollama that cannot enforce it)
        must not silently break the pipeline: the call is retried in plain JSON
        mode and finally salvaged by taking the first JSON object of the text.
        """
        content = await self._chat(prompt, format=schema, options=STRUCTURED_OPTIONS)
        if _is_valid_json(content):
            return content

        logger.warning(
            "Model %s ignored the response schema (%r); retrying in JSON mode",
            self.model,
            (content or "")[:200],
        )
        content_json = await self._chat(prompt, format="json", options=STRUCTURED_OPTIONS)
        if _is_valid_json(content_json):
            return content_json

        salvaged = _first_json_object(content_json) or _first_json_object(content)
        if salvaged is not None:
            logger.warning("Salvaged a JSON object from a non-JSON answer of %s", self.model)
            return salvaged

        logger.warning("Structured generation returned no JSON at all for %s", self.model)
        return content_json or content or ""

    async def _chat(self, prompt: str, *, format: Any, options: dict) -> str:
        response = await self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
            # thinking and grammar-constrained output do not mix: qwen3 would
            # spend the budget on reasoning tokens and the schema would be lost
            think=False,
            format=format,
            options=options,
            keep_alive=self.keep_alive,
        )
        return response["message"]["content"] or ""

    async def stream_reply(
        self,
        prompt: str,
        tone: str | None = None,
        history: Sequence[Mapping[str, str]] | None = None,
    ):
        if tone is not None:
            prompt = build_chat_prompt(prompt, tone, history)
        async for part in await self.client.generate(
            model=self.model,
            prompt=prompt,
            stream=True,
            think=settings.llm_thinking_enabled if tone is not None else False,
            options=GENERATION_OPTIONS,
            keep_alive=self.keep_alive,
        ):
            yield part["response"]
