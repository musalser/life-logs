# Извлечение знаний из дневника: extraction -> resolution -> aggregation

import json
import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from ..models import (
    DiaryPage,
    Entity,
    EntityRelation,
    Event,
    Goal,
    GoalProgress,
    Habit,
    HabitLog,
)

logger = logging.getLogger(__name__)

# Ollama structured-output schemas (one per extraction call).
GOALS_SCHEMA = {
    "type": "object",
    "properties": {
        "goals": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "status_update": {
                        "type": "string",
                        "enum": ["started", "progress", "completed", "abandoned", "mentioned"],
                    },
                    "note": {"type": "string"},
                    "source_text": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["title", "status_update", "note", "source_text", "confidence"],
            },
        }
    },
    "required": ["goals"],
}

RELATIONS_SCHEMA = {
    "type": "object",
    "properties": {
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "person_name": {"type": "string"},
                    "relation_type": {"type": "string"},
                    "source_text": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["person_name", "relation_type", "source_text", "confidence"],
            },
        }
    },
    "required": ["relations"],
}

EVENTS_SCHEMA = {
    "type": "object",
    "properties": {
        "events": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "time_text": {"type": "string"},
                    "importance": {"type": "number"},
                    "source_text": {"type": "string"},
                },
                "required": ["title", "description", "time_text", "importance", "source_text"],
            },
        }
    },
    "required": ["events"],
}

HABITS_SCHEMA = {
    "type": "object",
    "properties": {
        "habits": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "note": {"type": "string"},
                    "source_text": {"type": "string"},
                },
                "required": ["title", "note", "source_text"],
            },
        }
    },
    "required": ["habits"],
}

# matched_id = -1 means "no match, create new".
RESOLUTION_SCHEMA = {
    "type": "object",
    "properties": {
        "matches": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "candidate_index": {"type": "integer"},
                    "matched_id": {"type": "integer"},
                },
                "required": ["candidate_index", "matched_id"],
            },
        }
    },
    "required": ["matches"],
}


class KnowledgeService:
    def __init__(self, ai_adapter):
        self.ai_adapter = ai_adapter

    # ------------------------------------------------------------------
    # Extraction: отдельный вызов LLM на каждый тип знаний
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_json(raw: str, key: str) -> list[dict[str, Any]]:
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Failed to parse LLM JSON for key=%s: %r", key, raw[:500] if raw else raw)
            return []
        items = parsed.get(key, []) if isinstance(parsed, dict) else []
        return [item for item in items if isinstance(item, dict)]

    async def extract_goals(self, content: str) -> list[dict[str, Any]]:
        prompt = (
            "Ты извлекаешь личные цели пользователя из записи дневника.\n"
            "Цель — это осознанное намерение чего-то достичь (выучить язык, похудеть, "
            "сменить работу, накопить деньги и т.п.).\n\n"
            "Для каждой цели укажи:\n"
            "- title: краткое название цели (2-6 слов, именительный падеж)\n"
            "- status_update: started (впервые поставил цель) | progress (есть продвижение) | "
            "completed (достиг) | abandoned (отказался) | mentioned (просто упомянул)\n"
            "- note: краткое описание продвижения или контекста\n"
            "- source_text: точная цитата из текста\n"
            "- confidence: уверенность 0.0-1.0\n\n"
            "Не выдумывай цели: мимолётные желания и бытовые дела (помыть посуду) целями не считаются.\n"
            "Если целей нет, верни пустой массив goals.\n\n"
            f"Запись дневника:\n{content}"
        )
        raw = await self.ai_adapter.generate_structured(prompt, GOALS_SCHEMA)
        return self._parse_json(raw, "goals")

    async def extract_relations(self, content: str) -> list[dict[str, Any]]:
        prompt = (
            "Ты определяешь круг общения автора дневника: родственные, дружеские, "
            "рабочие и социальные связи.\n\n"
            "Для каждого упомянутого человека, чью связь с автором можно определить, укажи:\n"
            "- person_name: имя человека как в тексте (именительный падеж)\n"
            "- relation_type: тип связи одним словом в нижнем регистре "
            "(мама, папа, сестра, брат, жена, муж, сын, дочь, бабушка, дедушка, "
            "друг, подруга, коллега, начальник, сосед, знакомый, врач, тренер и т.п.)\n"
            "- source_text: точная цитата, из которой следует связь\n"
            "- confidence: уверенность 0.0-1.0\n\n"
            "Включай человека только если связь явно следует из текста.\n"
            "Если автор пишет о себе или связи не определяются, верни пустой массив relations.\n\n"
            f"Запись дневника:\n{content}"
        )
        raw = await self.ai_adapter.generate_structured(prompt, RELATIONS_SCHEMA)
        return self._parse_json(raw, "relations")

    async def extract_events(self, content: str) -> list[dict[str, Any]]:
        prompt = (
            "Ты извлекаешь важные события из записи дневника.\n"
            "Важное событие — то, что заметно влияет на жизнь автора: переезд, смена работы, "
            "экзамен, свадьба, рождение, болезнь, путешествие, крупная покупка, встреча и т.п.\n\n"
            "Для каждого события укажи:\n"
            "- title: краткое название (2-6 слов)\n"
            "- description: описание в 1-2 предложениях\n"
            "- time_text: когда произошло, как написано в тексте (или пустая строка)\n"
            "- importance: важность 0.0-1.0 (рутина < 0.3, заметное ~0.5, жизненно важное > 0.8)\n"
            "- source_text: точная цитата\n\n"
            "Рутинные действия (поел, лёг спать) событиями не считаются.\n"
            "Если событий нет, верни пустой массив events.\n\n"
            f"Запись дневника:\n{content}"
        )
        raw = await self.ai_adapter.generate_structured(prompt, EVENTS_SCHEMA)
        return self._parse_json(raw, "events")

    async def extract_habits(self, content: str) -> list[dict[str, Any]]:
        prompt = (
            "Ты извлекаешь регулярные привычки и активности из записи дневника.\n"
            "Привычка — повторяющееся действие: бег, зарядка, чтение, медитация, "
            "изучение языка, прогулки и т.п.\n\n"
            "Для каждой привычки, выполнение (или пропуск) которой упомянуто, укажи:\n"
            "- title: название привычки (1-4 слова, именительный падеж)\n"
            "- note: что именно сделано (или почему пропущено)\n"
            "- source_text: точная цитата\n\n"
            "Включай только регулярные действия, а не разовые дела.\n"
            "Если привычек нет, верни пустой массив habits.\n\n"
            f"Запись дневника:\n{content}"
        )
        raw = await self.ai_adapter.generate_structured(prompt, HABITS_SCHEMA)
        return self._parse_json(raw, "habits")

    # ------------------------------------------------------------------
    # Resolution: LLM сопоставляет извлечённое с существующими записями
    # ------------------------------------------------------------------

    async def resolve_candidates(
        self,
        kind_label: str,
        candidates: list[str],
        existing: list[tuple[int, str]],
    ) -> dict[int, int | None]:
        """Maps candidate index -> existing record id (or None to create new)."""
        if not candidates or not existing:
            return {i: None for i in range(len(candidates))}

        existing_lines = "\n".join(f"- id={rec_id}: {title}" for rec_id, title in existing)
        candidate_lines = "\n".join(f"- index={i}: {title}" for i, title in enumerate(candidates))
        prompt = (
            f"Ты сопоставляешь новые упоминания ({kind_label}) с уже существующими записями "
            "пользователя.\n"
            "Одна и та же сущность может называться по-разному "
            "(например «английский» и «выучить английский язык», «Маша» и «Мария»).\n\n"
            f"Существующие записи:\n{existing_lines}\n\n"
            f"Новые упоминания:\n{candidate_lines}\n\n"
            "Для каждого нового упоминания верни matched_id — id существующей записи, "
            "если это то же самое, иначе -1.\n"
            "Сопоставляй только при уверенном совпадении по смыслу."
        )
        raw = await self.ai_adapter.generate_structured(prompt, RESOLUTION_SCHEMA)
        matches = self._parse_json(raw, "matches")

        valid_ids = {rec_id for rec_id, _ in existing}
        result: dict[int, int | None] = {i: None for i in range(len(candidates))}
        for match in matches:
            idx = match.get("candidate_index")
            matched_id = match.get("matched_id")
            if not isinstance(idx, int) or idx not in result:
                continue
            result[idx] = matched_id if isinstance(matched_id, int) and matched_id in valid_ids else None
        return result

    # ------------------------------------------------------------------
    # Aggregation: запись в БД
    # ------------------------------------------------------------------

    @staticmethod
    def _clamp_confidence(value: Any) -> float | None:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return None

    async def _aggregate_goals(
        self, db: Session, user_id: int, page: DiaryPage, extracted: list[dict[str, Any]]
    ) -> dict[str, int]:
        stats = {"goals_created": 0, "goals_matched": 0, "progress_notes": 0}
        if not extracted:
            return stats

        existing = (
            db.query(Goal)
            .filter(Goal.user_id == user_id)
            .order_by(Goal.id.desc())
            .limit(200)
            .all()
        )
        matches = await self.resolve_candidates(
            "цели",
            [str(item.get("title", "")) for item in extracted],
            [(goal.id, goal.title) for goal in existing],
        )
        goals_by_id = {goal.id: goal for goal in existing}

        for idx, item in enumerate(extracted):
            title = str(item.get("title", "")).strip()
            if not title:
                continue
            status_update = item.get("status_update", "mentioned")

            goal = goals_by_id.get(matches.get(idx) or -1)
            if goal is None:
                goal = Goal(
                    user_id=user_id,
                    title=title[:255],
                    description=item.get("note") or None,
                    status="active",
                    created_from_page_id=page.id,
                )
                db.add(goal)
                db.flush()
                stats["goals_created"] += 1
            else:
                stats["goals_matched"] += 1

            if status_update in ("completed", "abandoned"):
                goal.status = status_update
            elif status_update in ("started", "progress"):
                goal.status = "active"

            db.add(
                GoalProgress(
                    goal_id=goal.id,
                    diary_page_id=page.id,
                    progress_kind=status_update,
                    note=item.get("note") or None,
                    source_text=item.get("source_text") or None,
                    confidence=self._clamp_confidence(item.get("confidence")),
                )
            )
            stats["progress_notes"] += 1
        return stats

    def _get_or_create_person(self, db: Session, user_id: int, name: str) -> Entity:
        normalized = name.lower().strip()
        now = datetime.now(timezone.utc)
        entity = (
            db.query(Entity)
            .filter_by(user_id=user_id, entity_type="person", normalized_name=normalized)
            .first()
        )
        if entity is None:
            entity = Entity(
                user_id=user_id,
                entity_type="person",
                canonical_name=name[:255],
                normalized_name=normalized[:255],
                first_seen_at=now,
                last_seen_at=now,
            )
            db.add(entity)
            db.flush()
        else:
            entity.last_seen_at = now
        return entity

    async def _aggregate_relations(
        self, db: Session, user_id: int, page: DiaryPage, extracted: list[dict[str, Any]]
    ) -> dict[str, int]:
        stats = {"relations_created": 0, "relations_updated": 0}
        if not extracted:
            return stats

        existing_persons = (
            db.query(Entity)
            .filter(Entity.user_id == user_id, Entity.entity_type == "person")
            .order_by(Entity.id.desc())
            .limit(200)
            .all()
        )
        matches = await self.resolve_candidates(
            "люди",
            [str(item.get("person_name", "")) for item in extracted],
            [(person.id, person.canonical_name) for person in existing_persons],
        )
        persons_by_id = {person.id: person for person in existing_persons}

        for idx, item in enumerate(extracted):
            name = str(item.get("person_name", "")).strip()
            relation_type = str(item.get("relation_type", "")).strip().lower()
            if not name or not relation_type:
                continue

            entity = persons_by_id.get(matches.get(idx) or -1)
            if entity is None:
                entity = self._get_or_create_person(db, user_id, name)
            else:
                entity.last_seen_at = datetime.now(timezone.utc)

            confidence = self._clamp_confidence(item.get("confidence"))
            relation = (
                db.query(EntityRelation)
                .filter_by(user_id=user_id, entity_id=entity.id)
                .first()
            )
            if relation is None:
                db.add(
                    EntityRelation(
                        user_id=user_id,
                        entity_id=entity.id,
                        relation_type=relation_type[:64],
                        confidence=confidence,
                        evidence_text=item.get("source_text") or None,
                        last_page_id=page.id,
                    )
                )
                stats["relations_created"] += 1
            else:
                # Обновляем тип связи только если новая уверенность не ниже прежней.
                if confidence is None or relation.confidence is None or confidence >= relation.confidence:
                    relation.relation_type = relation_type[:64]
                    relation.confidence = confidence
                    relation.evidence_text = item.get("source_text") or None
                relation.last_page_id = page.id
                stats["relations_updated"] += 1
        return stats

    def _aggregate_events(
        self, db: Session, user_id: int, page: DiaryPage, extracted: list[dict[str, Any]]
    ) -> dict[str, int]:
        stats = {"events_created": 0}
        for item in extracted:
            title = str(item.get("title", "")).strip()
            if not title:
                continue
            db.add(
                Event(
                    user_id=user_id,
                    diary_page_id=page.id,
                    title=title[:255],
                    description=item.get("description") or None,
                    time_text=(item.get("time_text") or None),
                    importance=self._clamp_confidence(item.get("importance")),
                    source_text=item.get("source_text") or None,
                )
            )
            stats["events_created"] += 1
        return stats

    async def _aggregate_habits(
        self, db: Session, user_id: int, page: DiaryPage, extracted: list[dict[str, Any]]
    ) -> dict[str, int]:
        stats = {"habits_created": 0, "habit_logs": 0}
        if not extracted:
            return stats

        existing = (
            db.query(Habit)
            .filter(Habit.user_id == user_id)
            .order_by(Habit.id.desc())
            .limit(200)
            .all()
        )
        matches = await self.resolve_candidates(
            "привычки",
            [str(item.get("title", "")) for item in extracted],
            [(habit.id, habit.title) for habit in existing],
        )
        habits_by_id = {habit.id: habit for habit in existing}

        for idx, item in enumerate(extracted):
            title = str(item.get("title", "")).strip()
            if not title:
                continue
            habit = habits_by_id.get(matches.get(idx) or -1)
            if habit is None:
                habit = Habit(user_id=user_id, title=title[:255])
                db.add(habit)
                db.flush()
                stats["habits_created"] += 1
            db.add(
                HabitLog(
                    habit_id=habit.id,
                    diary_page_id=page.id,
                    note=item.get("note") or None,
                    source_text=item.get("source_text") or None,
                )
            )
            stats["habit_logs"] += 1
        return stats

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    async def process_diary_page(self, db: Session, user_id: int, page: DiaryPage) -> dict[str, Any]:
        """Runs extraction -> resolution -> aggregation for one diary page."""
        summary: dict[str, Any] = {"errors": []}
        content = page.content or ""
        if not content.strip():
            return summary

        # Каждый этап независим: сбой одного не прерывает остальные (удобно для отладки).
        try:
            extracted_goals = await self.extract_goals(content)
            logger.info("page_id=%s extracted goals: %s", page.id, extracted_goals)
            summary.update(await self._aggregate_goals(db, user_id, page, extracted_goals))
        except Exception:
            logger.exception("Goal extraction failed for page_id=%s", page.id)
            summary["errors"].append("goals")

        try:
            extracted_relations = await self.extract_relations(content)
            logger.info("page_id=%s extracted relations: %s", page.id, extracted_relations)
            summary.update(await self._aggregate_relations(db, user_id, page, extracted_relations))
        except Exception:
            logger.exception("Relation extraction failed for page_id=%s", page.id)
            summary["errors"].append("relations")

        try:
            extracted_events = await self.extract_events(content)
            logger.info("page_id=%s extracted events: %s", page.id, extracted_events)
            summary.update(self._aggregate_events(db, user_id, page, extracted_events))
        except Exception:
            logger.exception("Event extraction failed for page_id=%s", page.id)
            summary["errors"].append("events")

        try:
            extracted_habits = await self.extract_habits(content)
            logger.info("page_id=%s extracted habits: %s", page.id, extracted_habits)
            summary.update(await self._aggregate_habits(db, user_id, page, extracted_habits))
        except Exception:
            logger.exception("Habit extraction failed for page_id=%s", page.id)
            summary["errors"].append("habits")

        db.commit()
        return summary
