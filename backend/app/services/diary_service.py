# Сервис для функционала дневника

import json
import re
from typing import Any


class DiaryService:
    def __init__(self, ai_adapter):
        self.ai_adapter = ai_adapter

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        """Parses model output and returns normalized extraction payload."""
        if not text:
            return {"entities": [], "entity_mentions": [], "facts": []}

        candidate = text.strip()
        if candidate.startswith("```"):
            candidate = re.sub(r"^```(?:json)?", "", candidate).strip()
            candidate = re.sub(r"```$", "", candidate).strip()

        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            # Fallback: try to extract the first JSON object block from noisy output.
            match = re.search(r"\{[\s\S]*\}", candidate)
            if not match:
                return {"entities": [], "entity_mentions": [], "facts": []}
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                return {"entities": [], "entity_mentions": [], "facts": []}

        if not isinstance(parsed, dict):
            return {"entities": [], "entity_mentions": [], "facts": []}

        return {
            "entities": parsed.get("entities", []) if isinstance(parsed.get("entities", []), list) else [],
            "entity_mentions": (
                parsed.get("entity_mentions", [])
                if isinstance(parsed.get("entity_mentions", []), list)
                else []
            ),
            "facts": parsed.get("facts", []) if isinstance(parsed.get("facts", []), list) else [],
        }

    async def generate_diary_title(self, content: str) -> str:
        prompt = (
            "/no_think Create a short diary page title in Russian using 2-5 words. "
            "Return title only, without quotes, punctuation, or explanation.\n\n"
            f"Diary page:\n{content}\n\nTitle:"
        )
        return await self.ai_adapter.generate_reply(prompt)
    
    async def extract_entities(self, content: str) -> dict[str, list[dict[str, Any]]]:
        prompt = (
            "Ты модуль структурированного извлечения знаний из личного дневника.\n"
            "Выдели сущности, упоминания сущностей и факты из входного текста.\n"
            "Верни только валидный JSON без комментариев и markdown.\n\n"
            "Формат ответа:\n"
            "{\n"
            '  "entities": [\n'
            "    {\n"
            '      "entity_ref": "E1",\n'
            '      "entity_type": "person|organization|location|event|object|date|other",\n'
            '      "canonical_name": "...",\n'
            '      "normalized_name": "...",\n'
            '      "description": "..."\n'
            "    }\n"
            "  ],\n"
            '  "entity_mentions": [\n'
            "    {\n"
            '      "entity_ref": "E1",\n'
            '      "mention_text": "точная подстрока",\n'
            '      "sentence_text": "предложение",\n'
            '      "start_offset": 0,\n'
            '      "end_offset": 10,\n'
            '      "confidence": 0.0,\n'
            '      "sentiment_label": "positive|neutral|negative|null",\n'
            '      "emotion_label": "joy|sadness|anger|fear|surprise|disgust|neutral|null"\n'
            "    }\n"
            "  ],\n"
            '  "facts": [\n'
            "    {\n"
            '      "subject_entity_ref": "E1|null",\n'
            '      "predicate": "краткий_predicate_lowercase",\n'
            '      "object_entity_ref": "E2|null",\n'
            '      "object_text": "... или null",\n'
            '      "source_text": "точная цитата",\n'
            '      "time_text": "... или null",\n'
            '      "emotion_label": "joy|sadness|anger|fear|surprise|disgust|neutral|null",\n'
            '      "sentiment_score": 0.0,\n'
            '      "confidence": 0.0\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Правила:\n"
            "1) Не выдумывай факты.\n"
            "2) mention_text и source_text должны быть точными фрагментами текста.\n"
            "3) start_offset/end_offset: 0-based, end_offset эксклюзивный.\n"
            "4) Если данных нет, верни пустые массивы.\n\n"
            f"Текст дневника:\n{content}"
        )
        raw = await self.ai_adapter.generate_reply(prompt)
        return self._extract_json(raw)