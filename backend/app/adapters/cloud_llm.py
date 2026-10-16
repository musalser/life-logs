"""Облачные модели для знаний: Claude (Anthropic), OpenAI и DeepSeek.

Конвейер извлечения и аналитика зовут у модели одно — ``generate_structured
(prompt, schema) -> str`` с JSON по схеме, — поэтому облачный адаптер
подменяет ``OllamaAdapter`` без правок в сервисах. Какой адаптер взять, решает
``knowledge_adapter()`` по настройкам ``knowledge_llm_provider`` /
``knowledge_llm_model``; чат остаётся на Ollama.

Как каждый провайдер гарантирует JSON:

* **Claude** — ``output_config.format`` с JSON-схемой (официальный SDK
  ``anthropic``). Схема требует ``additionalProperties: false`` у каждого
  объекта — это добавляет ``strict_schema``. Для моделей, которые это
  поддерживают, включён серверный fallback на отказ (``fallbacks: "default"``):
  если классификатор безопасности отклонит запрос, его повторит другая модель.
* **OpenAI** — ``response_format`` ``json_schema`` со ``strict: true``: там все
  поля объекта обязаны быть в ``required``.
* **DeepSeek** — API совместим с OpenAI, но знает только JSON-режим
  (``json_object``) без схемы, поэтому схема дописывается в промпт.

Прокси (``llm_api_proxy``) действует только на облачные API: Ollama на хосте
ходит напрямую, и переменные окружения прокси процессу не нужны.
"""
from __future__ import annotations

import copy
import json
import logging
import os
from typing import Any, Protocol

import anthropic
import openai

from app.config import settings

logger = logging.getLogger(__name__)

PROVIDERS = ("ollama", "anthropic", "openai", "deepseek")

#: модель по умолчанию, если ``knowledge_llm_model`` пуст. У OpenAI её нет
#: намеренно: линейка меняется часто, и угаданное имя хуже явной настройки.
DEFAULT_MODELS = {
    "anthropic": "claude-opus-5-5",
    "deepseek": "deepseek-chat",
}

#: модели Claude, принимающие серверный fallback ``fallbacks: "default"``
_ANTHROPIC_FALLBACK_MODELS = frozenset(
    {"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"}
)
_ANTHROPIC_FALLBACK_BETA = "server-side-fallback-2026-07-01"

#: ответ извлечения — список фактов страницы; запас, чтобы не обрезать JSON
_MAX_OUTPUT_TOKENS = 16000
#: потолок deepseek-chat на один ответ
_DEEPSEEK_MAX_TOKENS = 8192


class StructuredLLM(Protocol):
    """Всё, что конвейер знаний и аналитика требуют от модели."""

    provider: str
    model: str

    async def generate_structured(self, prompt: str, schema: dict) -> str: ...


class LLMConfigError(RuntimeError):
    """Провайдер выбран, но не настроен (нет модели, неизвестное имя)."""


class LLMRefusal(RuntimeError):
    """Модель отказалась отвечать: вызов считается неудачным, а не пустым."""


def strict_schema(schema: dict[str, Any], *, all_required: bool = False) -> dict[str, Any]:
    """Копия схемы с ``additionalProperties: false`` у каждого объекта.

    ``all_required`` — для OpenAI strict: каждое поле объекта обязано быть
    в ``required``. Исходная схема не меняется: её же получает Ollama.
    """
    result = copy.deepcopy(schema)

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object":
                node["additionalProperties"] = False
                if all_required and isinstance(node.get("properties"), dict):
                    node["required"] = list(node["properties"])
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    visit(result)
    return result


def _http_client(module: Any) -> Any | None:
    """HTTP-клиент SDK с прокси, если он задан; иначе SDK берёт свой."""
    if not settings.llm_api_proxy:
        return None
    return module.DefaultAsyncHttpxClient(proxy=settings.llm_api_proxy)


class AnthropicAdapter:
    """Claude через официальный SDK: JSON по схеме через ``output_config.format``."""

    provider = "anthropic"

    def __init__(self, model: str, client: Any | None = None) -> None:
        self.model = model
        # api_key=None — SDK сам найдёт ANTHROPIC_API_KEY или профиль `ant auth login`
        self.client = client or anthropic.AsyncAnthropic(
            api_key=settings.anthropic_api_key or None,
            http_client=_http_client(anthropic),
            timeout=settings.llm_api_timeout_s,
        )

    async def generate_structured(self, prompt: str, schema: dict) -> str:
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": _MAX_OUTPUT_TOKENS,
            "messages": [{"role": "user", "content": prompt}],
            "output_config": {
                "effort": settings.anthropic_effort,
                "format": {"type": "json_schema", "schema": strict_schema(schema)},
            },
        }
        if self.model in _ANTHROPIC_FALLBACK_MODELS:
            response = await self.client.beta.messages.create(
                betas=[_ANTHROPIC_FALLBACK_BETA], fallbacks="default", **request
            )
        else:
            response = await self.client.messages.create(**request)

        if response.stop_reason == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None) if details else None
            raise LLMRefusal(f"{self.model} отказалась отвечать (категория: {category or '—'})")
        if response.stop_reason == "max_tokens":
            logger.warning("%s hit max_tokens: the JSON answer may be truncated", self.model)
        return "".join(block.text for block in response.content if block.type == "text")

    async def aclose(self) -> None:
        """Закрыть соединения в том же цикле событий, где они открыты."""
        await self.client.close()


class OpenAICompatibleAdapter:
    """OpenAI и DeepSeek: один SDK, разная поддержка схем."""

    def __init__(
        self,
        provider: str,
        model: str,
        *,
        api_key: str | None,
        base_url: str | None,
        client: Any | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.client = client or openai.AsyncOpenAI(
            api_key=api_key or None,
            base_url=base_url or None,
            http_client=_http_client(openai),
            timeout=settings.llm_api_timeout_s,
        )

    async def generate_structured(self, prompt: str, schema: dict) -> str:
        if self.provider == "deepseek":
            # JSON-режим без схемы: схема — часть промпта, слово «JSON» обязательно
            request: dict[str, Any] = {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"{prompt}\n\nОтветь одним JSON-объектом строго по этой JSON-схеме "
                            f"(все поля обязательны):\n{json.dumps(schema, ensure_ascii=False)}"
                        ),
                    }
                ],
                "response_format": {"type": "json_object"},
                "temperature": 0,
                "max_tokens": _DEEPSEEK_MAX_TOKENS,
            }
        else:
            request = {
                "messages": [{"role": "user", "content": prompt}],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "extraction",
                        "strict": True,
                        "schema": strict_schema(schema, all_required=True),
                    },
                },
                "max_completion_tokens": _MAX_OUTPUT_TOKENS,
            }
        response = await self.client.chat.completions.create(model=self.model, **request)
        choice = response.choices[0]
        refusal = getattr(choice.message, "refusal", None)
        if refusal:
            raise LLMRefusal(f"{self.model} отказалась отвечать: {refusal}")
        if choice.finish_reason == "length":
            logger.warning("%s hit the token limit: the JSON answer may be truncated", self.model)
        return choice.message.content or ""

    async def aclose(self) -> None:
        """Закрыть соединения в том же цикле событий, где они открыты."""
        await self.client.close()


def knowledge_model_label() -> str:
    """Что записать в журнал прогонов и кэш аналитики: провайдер и модель."""
    provider = _provider()
    model = _model(provider)
    return model if provider == "ollama" else f"{provider}:{model}"[:64]


def knowledge_adapter() -> StructuredLLM:
    """Адаптер модели знаний по настройкам; Ollama — по умолчанию."""
    provider = _provider()
    model = _model(provider)
    if provider == "ollama":
        from app.adapters.ollama_adapter import OllamaAdapter

        return OllamaAdapter(model=model)
    if provider == "anthropic":
        return AnthropicAdapter(model)
    if provider == "openai":
        key = settings.openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise LLMConfigError("Для провайдера openai задайте OPENAI_API_KEY")
        return OpenAICompatibleAdapter("openai", model, api_key=key, base_url=settings.openai_base_url)
    # ключ DeepSeek — только свой: SDK OpenAI без него взял бы OPENAI_API_KEY
    if not settings.deepseek_api_key:
        raise LLMConfigError("Для провайдера deepseek задайте DEEPSEEK_API_KEY")
    return OpenAICompatibleAdapter(
        "deepseek", model, api_key=settings.deepseek_api_key, base_url=settings.deepseek_base_url
    )


def _provider() -> str:
    provider = (settings.knowledge_llm_provider or "ollama").strip().lower()
    if provider not in PROVIDERS:
        raise LLMConfigError(
            f"Неизвестный KNOWLEDGE_LLM_PROVIDER={provider!r}; ожидается одно из: {', '.join(PROVIDERS)}"
        )
    return provider


def _model(provider: str) -> str:
    model = (settings.knowledge_llm_model or "").strip()
    if model:
        return model
    if provider == "ollama":
        return settings.llm_model
    if provider in DEFAULT_MODELS:
        return DEFAULT_MODELS[provider]
    raise LLMConfigError(f"Для провайдера {provider} задайте модель: KNOWLEDGE_LLM_MODEL")
