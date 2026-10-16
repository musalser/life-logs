"""Облачные адаптеры знаний: что уходит провайдеру и как читается ответ.

Сеть не трогаем: SDK-клиенты подменены фейками, которые запоминают запрос.
Проверяется контракт с конвейером — ``generate_structured`` возвращает строку
JSON, отказ модели — исключение (вызов падает и попадает в ``stats.errors``),
а схема для провайдера ужесточается, не меняя исходную.
"""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from app.adapters import cloud_llm
from app.adapters.cloud_llm import (
    AnthropicAdapter,
    LLMConfigError,
    LLMRefusal,
    OpenAICompatibleAdapter,
    knowledge_adapter,
    knowledge_model_label,
    strict_schema,
)
from app.adapters.ollama_adapter import OllamaAdapter
from app.config import settings
from app.services import extraction_prompts as prompts

SCHEMA = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"predicate": {"type": "string"}, "confidence": {"type": "number"}},
                "required": ["predicate"],
            },
        }
    },
    "required": ["facts"],
}


def run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Схема
# ---------------------------------------------------------------------------


def test_strict_schema_closes_every_object_and_keeps_the_original():
    strict = strict_schema(SCHEMA)

    assert strict["additionalProperties"] is False
    assert strict["properties"]["facts"]["items"]["additionalProperties"] is False
    assert strict["properties"]["facts"]["items"]["required"] == ["predicate"]
    assert "additionalProperties" not in SCHEMA  # исходную получает Ollama


def test_strict_schema_for_openai_requires_every_field():
    strict = strict_schema(SCHEMA, all_required=True)

    assert strict["properties"]["facts"]["items"]["required"] == ["predicate", "confidence"]


def test_real_extraction_schemas_are_accepted_as_strict():
    """Ни одной конструкции, которую строгий режим не принимает."""
    for schema in (prompts.ENTITIES_SCHEMA, prompts.facts_schema(True), prompts.RELATIONS_SCHEMA, prompts.RESOLUTION_SCHEMA):
        text = json.dumps(strict_schema(schema, all_required=True))
        for forbidden in ("minimum", "maximum", "minLength", "maxLength", "$ref"):
            assert forbidden not in text


# ---------------------------------------------------------------------------
# Claude
# ---------------------------------------------------------------------------


class FakeAnthropicMessages:
    def __init__(self, response):
        self.response = response
        self.requests: list[dict] = []

    async def create(self, **request):
        self.requests.append(request)
        return self.response


def anthropic_client(response):
    messages = FakeAnthropicMessages(response)
    beta_messages = FakeAnthropicMessages(response)
    client = SimpleNamespace(messages=messages, beta=SimpleNamespace(messages=beta_messages))
    return client, messages, beta_messages


def claude_response(text, stop_reason="end_turn", category=None):
    return SimpleNamespace(
        stop_reason=stop_reason,
        stop_details=SimpleNamespace(category=category) if stop_reason == "refusal" else None,
        content=[SimpleNamespace(type="fallback"), SimpleNamespace(type="text", text=text)],
    )


def test_claude_gets_the_schema_through_output_config_with_fallbacks():
    client, plain, beta = anthropic_client(claude_response('{"facts": []}'))
    adapter = AnthropicAdapter("claude-opus-5-5", client=client)

    raw = run(adapter.generate_structured("Текст страницы", SCHEMA))

    assert json.loads(raw) == {"facts": []}
    assert plain.requests == []
    (request,) = beta.requests
    assert request["model"] == "claude-opus-5-5"
    assert request["fallbacks"] == "default"
    assert request["betas"] == ["server-side-fallback-2026-07-01"]
    assert request["output_config"]["format"]["type"] == "json_schema"
    assert request["output_config"]["format"]["schema"]["additionalProperties"] is False
    assert request["output_config"]["effort"] == settings.anthropic_effort
    assert "thinking" not in request and "temperature" not in request
    assert request["messages"] == [{"role": "user", "content": "Текст страницы"}]


def test_claude_model_without_server_fallback_uses_the_plain_endpoint():
    client, plain, beta = anthropic_client(claude_response('{"facts": []}'))

    run(AnthropicAdapter("claude-haiku-5-5", client=client).generate_structured("т", SCHEMA))

    assert beta.requests == []
    assert "fallbacks" not in plain.requests[0]


def test_claude_refusal_is_an_error_not_an_empty_answer():
    client, *_ = anthropic_client(claude_response("", stop_reason="refusal", category="cyber"))

    with pytest.raises(LLMRefusal, match="cyber"):
        run(AnthropicAdapter("claude-opus-5-5", client=client).generate_structured("т", SCHEMA))


# ---------------------------------------------------------------------------
# OpenAI и DeepSeek
# ---------------------------------------------------------------------------


class FakeCompletions:
    def __init__(self, content, finish_reason="stop", refusal=None):
        self.content, self.finish_reason, self.refusal = content, finish_reason, refusal
        self.requests: list[dict] = []

    async def create(self, **request):
        self.requests.append(request)
        message = SimpleNamespace(content=self.content, refusal=self.refusal)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=self.finish_reason)])


def openai_client(completions):
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


def test_openai_gets_a_strict_json_schema():
    completions = FakeCompletions('{"facts": []}')
    adapter = OpenAICompatibleAdapter("openai", "gpt-test", api_key="k", base_url=None, client=openai_client(completions))

    raw = run(adapter.generate_structured("Текст", SCHEMA))

    assert json.loads(raw) == {"facts": []}
    request = completions.requests[0]
    assert request["model"] == "gpt-test"
    fmt = request["response_format"]
    assert fmt["type"] == "json_schema" and fmt["json_schema"]["strict"] is True
    assert fmt["json_schema"]["schema"]["properties"]["facts"]["items"]["required"] == ["predicate", "confidence"]


def test_deepseek_uses_json_mode_with_the_schema_in_the_prompt():
    completions = FakeCompletions('{"facts": []}')
    adapter = OpenAICompatibleAdapter("deepseek", "deepseek-chat", api_key="k", base_url="https://api.deepseek.com", client=openai_client(completions))

    run(adapter.generate_structured("Текст страницы", SCHEMA))

    request = completions.requests[0]
    assert request["response_format"] == {"type": "json_object"}
    assert request["temperature"] == 0
    content = request["messages"][0]["content"]
    assert content.startswith("Текст страницы")  # общий префикс промпта сохраняется
    assert "JSON" in content and '"predicate"' in content


def test_openai_refusal_raises():
    completions = FakeCompletions(None, refusal="не могу")
    adapter = OpenAICompatibleAdapter("openai", "gpt-test", api_key="k", base_url=None, client=openai_client(completions))

    with pytest.raises(LLMRefusal):
        run(adapter.generate_structured("т", SCHEMA))


# ---------------------------------------------------------------------------
# Выбор провайдера
# ---------------------------------------------------------------------------


@pytest.fixture()
def knowledge_settings(monkeypatch):
    def configure(provider, model=""):
        monkeypatch.setattr(settings, "knowledge_llm_provider", provider)
        monkeypatch.setattr(settings, "knowledge_llm_model", model)
        monkeypatch.setattr(settings, "llm_api_proxy", None)

    return configure


def test_default_is_ollama_with_the_chat_model(knowledge_settings):
    knowledge_settings("ollama")

    adapter = knowledge_adapter()

    assert isinstance(adapter, OllamaAdapter)
    assert adapter.model == settings.llm_model
    assert knowledge_model_label() == settings.llm_model


def test_ollama_knowledge_model_can_differ_from_the_chat(knowledge_settings):
    knowledge_settings("ollama", "qwen3:32b")

    assert knowledge_adapter().model == "qwen3:32b"


def test_cloud_providers_and_their_defaults(knowledge_settings, monkeypatch):
    knowledge_settings("anthropic")
    claude = knowledge_adapter()
    assert isinstance(claude, AnthropicAdapter) and claude.model == "claude-opus-5-5"
    assert knowledge_model_label() == "anthropic:claude-opus-5-5"

    monkeypatch.setattr(settings, "deepseek_api_key", "sk-deepseek")
    knowledge_settings("deepseek")
    deepseek = knowledge_adapter()
    assert deepseek.provider == "deepseek" and deepseek.model == "deepseek-chat"
    assert str(deepseek.client.base_url).startswith(settings.deepseek_base_url)


def test_openai_needs_an_explicit_model(knowledge_settings, monkeypatch):
    monkeypatch.setattr(settings, "openai_api_key", "sk-openai")
    knowledge_settings("openai")
    with pytest.raises(LLMConfigError, match="KNOWLEDGE_LLM_MODEL"):
        knowledge_adapter()

    knowledge_settings("openai", "gpt-test")
    assert knowledge_adapter().model == "gpt-test"


def test_missing_keys_are_config_errors_with_the_variable_name(knowledge_settings, monkeypatch):
    monkeypatch.setattr(settings, "deepseek_api_key", None)
    monkeypatch.setattr(settings, "openai_api_key", None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    knowledge_settings("deepseek")
    with pytest.raises(LLMConfigError, match="DEEPSEEK_API_KEY"):
        knowledge_adapter()
    # чужой ключ DeepSeek не подхватывает
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    with pytest.raises(LLMConfigError, match="DEEPSEEK_API_KEY"):
        knowledge_adapter()

    monkeypatch.delenv("OPENAI_API_KEY")
    knowledge_settings("openai", "gpt-test")
    with pytest.raises(LLMConfigError, match="OPENAI_API_KEY"):
        knowledge_adapter()


def test_unknown_provider_is_a_config_error(knowledge_settings):
    knowledge_settings("gemini")
    with pytest.raises(LLMConfigError, match="gemini"):
        knowledge_adapter()


def test_cloud_proxy_is_used_only_by_cloud_clients(knowledge_settings, monkeypatch):
    knowledge_settings("anthropic")
    monkeypatch.setattr(settings, "llm_api_proxy", "http://proxy.local:3128")
    seen = {}

    real = cloud_llm._http_client

    def spy(module):
        client = real(module)
        seen[module.__name__] = client
        return client

    monkeypatch.setattr(cloud_llm, "_http_client", spy)
    knowledge_adapter()

    assert seen["anthropic"] is not None


def test_pipeline_closes_the_adapter_it_created(monkeypatch, db_session):
    """Свой облачный клиент конвейер закрывает в том же цикле событий."""
    from app.models import DiaryPage, User
    from app.services import extraction_pipeline
    from app.services.knowledge_source_service import KnowledgeSourceService

    closed = []

    class ClosingAdapter:
        provider, model = "deepseek", "deepseek-chat"

        async def generate_structured(self, prompt, schema):
            return "{}"

        async def aclose(self):
            closed.append(True)

    class NoEmbeddings:
        async def embed_texts(self, texts, prompt_name=None):
            from app.services.embedding_service import EmbeddingError
            raise EmbeddingError("нет")

    monkeypatch.setattr(extraction_pipeline, "knowledge_adapter", lambda: ClosingAdapter())
    user = User(username="closer", password_hash="x", name="Closer")
    db_session.add(user)
    db_session.commit()
    page = DiaryPage(user_id=user.id, content="Текст.")
    db_session.add(page)
    db_session.commit()
    source, _, _ = KnowledgeSourceService(db_session).refresh_from_diary_page(page)

    pipeline = extraction_pipeline.ExtractionPipeline(
        db_session, source, embeddings=NoEmbeddings(), service=KnowledgeSourceService(db_session)
    )
    run(pipeline.run())

    assert closed == [True]
