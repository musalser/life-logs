"""Юнит-тесты адаптера Ollama: строгий JSON, запасные режимы, подсказки тона.

Клиент подменяется фейком, поэтому тесты не требуют ни Ollama, ни сети.
"""

import asyncio
import json

from app.adapters.ollama_adapter import OllamaAdapter, _first_json_object
from app.services.system_prompt import build_chat_prompt, critic

SCHEMA = {
    "type": "object",
    "properties": {"goals": {"type": "array", "items": {"type": "object"}}},
    "required": ["goals"],
}


class FakeChatClient:
    """Отдаёт заранее заготовленные ответы и запоминает режим ``format``."""

    def __init__(self, answers):
        self.answers = list(answers)
        self.formats = []
        self.think_flags = []

    async def chat(self, **kwargs):
        self.formats.append(kwargs.get("format"))
        self.think_flags.append(kwargs.get("think"))
        answer = self.answers.pop(0) if self.answers else ""
        return {"message": {"content": answer}}


def run(coro):
    return asyncio.run(coro)


def test_structured_accepts_valid_json():
    adapter = OllamaAdapter()
    adapter.client = FakeChatClient(['{"goals": [{"title": "бег"}]}'])

    raw = run(adapter.generate_structured("промпт", SCHEMA))

    assert json.loads(raw)["goals"][0]["title"] == "бег"
    assert adapter.client.formats == [SCHEMA]
    assert adapter.client.think_flags == [False]  # qwen3 не тратит бюджет на thinking


def test_structured_falls_back_to_json_mode():
    adapter = OllamaAdapter()
    adapter.client = FakeChatClient(["Конечно! Вот цели:", '{"goals": []}'])

    raw = run(adapter.generate_structured("промпт", SCHEMA))

    assert json.loads(raw) == {"goals": []}
    assert adapter.client.formats == [SCHEMA, "json"]


def test_structured_salvages_json_object():
    adapter = OllamaAdapter()
    adapter.client = FakeChatClient(['Вот результат: {"goals": [{"title": "бег"}]} — готово', "и снова проза"])

    raw = run(adapter.generate_structured("промпт", SCHEMA))

    assert json.loads(raw)["goals"][0]["title"] == "бег"


def test_structured_gives_up_without_raising():
    adapter = OllamaAdapter()
    adapter.client = FakeChatClient(["проза", "ещё проза"])

    raw = run(adapter.generate_structured("промпт", SCHEMA))

    assert raw == "ещё проза"  # сервисы сами решают, что делать с мусором


def test_first_json_object():
    assert _first_json_object('мусор {"a": 1} хвост') == '{"a": 1}'
    assert _first_json_object("нет объекта") is None
    assert _first_json_object(None) is None


def test_chat_prompt_uses_the_tone_block():
    prompt = build_chat_prompt("привет", "critic", [{"text": "вчера", "reply": "ок"}])

    assert critic.strip() in prompt
    assert "вчера" in prompt
    assert "привет" in prompt


def test_chat_prompt_keeps_an_unknown_tone_name():
    prompt = build_chat_prompt("привет", "coach", None)

    assert "coach" in prompt
