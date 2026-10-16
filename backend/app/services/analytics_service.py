"""Аналитика по знаниям: инсайты, вопросы, идеи, выводы (ТЗ, раздел 10).

Агрегат строится по перспективе выбранного объекта, ограничен сверху (архив
растёт, а окно модели — нет), а каждый пункт обязан ссылаться на существующие
записи: ссылки проверяются на сервере, и пункт без валидных ссылок отбрасывается.
Так «интересный вывод» не может оказаться выдумкой, которую нечем подтвердить.

Кэш живёт в ``knowledge_insights`` и привязан к ``knowledge_state.version``:
после любого успешного извлечения версия растёт, и кэш становится ``stale``.

**Привычки выводятся здесь, а не при извлечении.** По одной странице
регулярность не установить, поэтому конвейер привычек не ищет; аналитика видит
все накопленные факты и называет повторяющееся. Код не верит ей на слово:
привычка обязана ссылаться либо на факт, где текст прямо говорит о регулярности
(``recurrence``), либо на факты из двух разных источников (``_habit_supported``).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from ..adapters.cloud_llm import StructuredLLM, knowledge_adapter, knowledge_model_label
from ..config import settings
from ..models import (
    Entity,
    EntityRelation,
    Fact,
    Goal,
    GoalProgress,
    KnowledgeInsight,
    KnowledgeState,
)

logger = logging.getLogger(__name__)

#: сколько записей каждого вида попадает в промпт (защита от переполнения окна)
AGGREGATE_LIMITS = {
    "facts": 200,
    "goals": 50,
    "goal_progress": 100,
    "relations": 50,
    "entities": 50,
}

_ITEM_SCHEMA: dict[str, Any] = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "confidence": {"type": "number"},
            "evidence": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "kind": {"type": "string"},
                        "id": {"type": "integer"},
                    },
                    "required": ["kind", "id"],
                },
            },
        },
        "required": ["text", "confidence", "evidence"],
    },
}

SECTIONS = ("insights", "habits", "questions", "ideas", "conclusions")

ANALYTICS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {section: _ITEM_SCHEMA for section in SECTIONS},
    "required": list(SECTIONS),
}

SECTION_TITLES = {
    "insights": "инсайты — нетривиальные закономерности в записях",
    "habits": (
        "привычки — регулярные занятия, которые видны по фактам: одно и то же "
        "действие на разных страницах или факт с пометкой «регулярно». Ссылайся "
        "на сами факты (kind=fact); одно упоминание без пометки — не привычка"
    ),
    "questions": "вопросы — что стоит выяснить; ответа в данных пока нет",
    "ideas": "идеи — что из этих закономерностей следует сделать",
    "conclusions": "выводы — короткие обобщения по темам",
}


def build_prompt(subject_name: str, digest: str) -> str:
    return (
        f"Ты аналитик личного архива. Ниже — структурированные знания об объекте "
        f"«{subject_name}», извлечённые из дневников и рукописей.\n\n"
        "Правила, которые нельзя нарушать:\n"
        "1. Не выдумывай факты: опирайся только на список ниже.\n"
        "2. Если данных мало для какого-то раздела — верни для него пустой массив. "
        "Пустой раздел лучше выдуманного.\n"
        "3. Каждый пункт обязан ссылаться на записи: evidence — массив объектов "
        "{kind, id}, где kind — одно из fact, goal, relation, entity, "
        "а id — идентификатор из списка. Пункт без ссылок будет отброшен.\n"
        "4. Пиши по-русски, коротко и по делу, без общих слов.\n\n"
        f"Что нужно вернуть:\n"
        + "\n".join(f"- {key}: {SECTION_TITLES[key]}" for key in SECTIONS)
        + f"\n\nЗнания:\n{digest}"
    )


class AnalyticsService:
    def __init__(self, db: Session, adapter: StructuredLLM | None = None) -> None:
        self.db = db
        self._owns_adapter = adapter is None
        self.adapter = adapter or knowledge_adapter()

    # ------------------------------------------------------------------
    # Чтение
    # ------------------------------------------------------------------

    def state_version(self, user_id: int) -> int:
        state = self.db.query(KnowledgeState).filter(KnowledgeState.user_id == user_id).first()
        return state.version if state is not None else 0

    def get_cached(self, user_id: int, subject_entity_id: int | None) -> dict[str, Any]:
        """Кэш аналитики вместе с признаком устаревания."""
        row = (
            self.db.query(KnowledgeInsight)
            .filter(
                KnowledgeInsight.user_id == user_id,
                KnowledgeInsight.subject_entity_id == subject_entity_id,
            )
            .first()
        )
        current = self.state_version(user_id)
        if row is None:
            return {
                "subject_entity_id": subject_entity_id,
                "version": current,
                "stale": True,
                "generated_at": None,
                "model": None,
                "payload": {key: [] for key in SECTIONS},
            }
        try:
            payload = json.loads(row.payload)
        except (json.JSONDecodeError, TypeError):
            payload = {}
        return {
            "subject_entity_id": subject_entity_id,
            "version": row.version,
            "stale": row.version != current,
            "generated_at": row.generated_at,
            "model": row.model,
            "payload": {key: payload.get(key, []) for key in SECTIONS},
        }

    # ------------------------------------------------------------------
    # Генерация
    # ------------------------------------------------------------------

    async def generate(self, user_id: int, subject_entity_id: int | None) -> dict[str, Any]:
        subject = self._subject(user_id, subject_entity_id)
        digest, index = self._digest(user_id, subject)
        if not index:
            empty = {key: [] for key in SECTIONS}
            self._save(user_id, subject_entity_id, empty)
            return self.get_cached(user_id, subject_entity_id)

        prompt = build_prompt(subject.canonical_name if subject else "я", digest)
        try:
            raw = await self.adapter.generate_structured(prompt, ANALYTICS_SCHEMA)
        finally:
            if self._owns_adapter and hasattr(self.adapter, "aclose"):
                await self.adapter.aclose()
        payload = self._validate(self._parse(raw), index)
        self._save(user_id, subject_entity_id, payload)
        return self.get_cached(user_id, subject_entity_id)

    def generate_sync(self, user_id: int, subject_entity_id: int | None) -> dict[str, Any]:
        """``generate`` для синхронного роута."""
        import asyncio

        return asyncio.run(self.generate(user_id, subject_entity_id))

    # ------------------------------------------------------------------
    # Агрегат
    # ------------------------------------------------------------------

    def _subject(self, user_id: int, subject_entity_id: int | None) -> Entity | None:
        if subject_entity_id is None:
            return None
        return (
            self.db.query(Entity)
            .filter(Entity.id == subject_entity_id, Entity.user_id == user_id)
            .first()
        )

    def _digest(self, user_id: int, subject: Entity | None) -> tuple[str, dict[str, Any]]:
        """Текстовый агрегат знаний и индекс «kind -> {id: подпись}».

        Второе — то, чем проверяются ссылки модели: если id нет в индексе,
        пункт отбрасывается, а не сохраняется с битой ссылкой.
        """
        subject_id = subject.id if subject else None
        lines: list[str] = []
        index: dict[str, dict[int, str]] = {key: {} for key in ("fact", "goal", "relation", "entity")}
        #: id факта -> (источник, сказано ли о регулярности): чем проверяются привычки
        self._fact_meta: dict[int, tuple[int, bool]] = {}

        def add(kind: str, record_id: int, label: str) -> None:
            index[kind][record_id] = label
            lines.append(f"[{kind}:{record_id}] {label}")

        # объект-перспектива и его окружение
        if subject is not None:
            add("entity", subject.id, f"{subject.canonical_name} (сам объект анализа, {subject.entity_type})")
        for edge in (
            self.db.query(EntityRelation)
            .filter(EntityRelation.user_id == user_id)
            .limit(AGGREGATE_LIMITS["relations"])
            .all()
        ):
            other = edge.to_entity
            if other is not None:
                add("relation", edge.id, f"{other.canonical_name} — {edge.relation_type}")
        for entity in (
            self.db.query(Entity)
            .filter(Entity.user_id == user_id, Entity.entity_type == "person")
            .limit(AGGREGATE_LIMITS["entities"])
            .all()
        ):
            add("entity", entity.id, f"человек: {entity.canonical_name}")

        facts_query = self.db.query(Fact).filter(Fact.user_id == user_id)
        if subject_id is not None:
            facts_query = facts_query.filter(
                (Fact.subject_entity_id == subject_id) | (Fact.subject_entity_id.is_(None))
            )
        for fact in facts_query.order_by(Fact.id.desc()).limit(AGGREGATE_LIMITS["facts"]).all():
            subject_name = fact.subject_entity.canonical_name if fact.subject_entity else "автор"
            obj = fact.object_entity.canonical_name if fact.object_entity else (fact.object_text or "")
            when = f" ({fact.time_text})" if fact.time_text else ""
            marks = []
            if fact.importance and fact.importance >= settings.knowledge_event_min_importance:
                marks.append(f"событие, важность {fact.importance:.1f}")
            if fact.recurrence:
                marks.append(f"регулярно: {fact.recurrence}")
            tail = f" [{'; '.join(marks)}]" if marks else ""
            add("fact", fact.id, f"{subject_name}: {fact.predicate} {obj}{when}{tail}".strip())
            self._fact_meta[fact.id] = (fact.source_id, bool(fact.recurrence))

        goals_query = self.db.query(Goal).filter(Goal.user_id == user_id)
        if subject_id is not None:
            goals_query = goals_query.filter(Goal.subject_entity_id == subject_id)
        goals = goals_query.order_by(Goal.updated_at.desc().nullslast(), Goal.id.desc()).limit(AGGREGATE_LIMITS["goals"]).all()
        for goal in goals:
            add("goal", goal.id, f"цель «{goal.title}» — статус {goal.status}")
        goal_ids = [goal.id for goal in goals]
        if goal_ids:
            progress = (
                self.db.query(GoalProgress)
                .filter(GoalProgress.goal_id.in_(goal_ids))
                .order_by(GoalProgress.id.desc())
                .limit(AGGREGATE_LIMITS["goal_progress"])
                .all()
            )
            for note in progress:
                add("goal", note.goal_id, f"цель: {note.progress_kind} — {note.note or ''}".strip())

        if not index["fact"] and not index["goal"]:
            # одни имена без записей — анализировать нечего
            return "", {}
        return "\n".join(lines), index

    # ------------------------------------------------------------------
    # Валидация и кэш
    # ------------------------------------------------------------------

    @staticmethod
    def _parse(raw: str) -> dict[str, Any]:
        try:
            payload = json.loads(raw) if raw else {}
        except (json.JSONDecodeError, TypeError):
            logger.warning("Analytics answer is not JSON: %r", str(raw)[:200])
            return {}
        return payload if isinstance(payload, dict) else {}

    def _validate(self, payload: dict[str, Any], index: dict[str, Any]) -> dict[str, list[dict]]:
        """Оставляет пункты с существующими ссылками; остальные отбрасывает."""
        result: dict[str, list[dict]] = {}
        dropped = 0
        for section in SECTIONS:
            items = payload.get(section)
            kept: list[dict] = []
            if isinstance(items, list):
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    text = str(item.get("text") or "").strip()
                    if not text:
                        continue
                    evidence = []
                    for link in item.get("evidence") or []:
                        if not isinstance(link, dict):
                            continue
                        kind = str(link.get("kind") or "").strip()
                        record_id = link.get("id")
                        if kind in index and isinstance(record_id, int) and record_id in index[kind]:
                            evidence.append(
                                {"kind": kind, "id": record_id, "label": index[kind][record_id][:200]}
                            )
                    if not evidence:
                        dropped += 1
                        continue
                    if section == "habits" and not self._habit_supported(evidence):
                        dropped += 1
                        continue
                    kept.append(
                        {
                            "text": text[:1000],
                            "confidence": self._clamp(item.get("confidence")),
                            "evidence": evidence,
                        }
                    )
            result[section] = kept
        if dropped:
            logger.info("Analytics: dropped %s items without valid evidence", dropped)
        return result

    def _habit_supported(self, evidence: list[dict]) -> bool:
        """Привычка — это повтор: факт с «регулярно» или факты двух источников."""
        meta = getattr(self, "_fact_meta", {})
        facts = [meta[link["id"]] for link in evidence if link["kind"] == "fact" and link["id"] in meta]
        if any(recurrent for _, recurrent in facts):
            return True
        return len({source_id for source_id, _ in facts}) >= 2

    @staticmethod
    def _clamp(value: Any) -> float | None:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return None

    def _save(self, user_id: int, subject_entity_id: int | None, payload: dict[str, list[dict]]) -> None:
        version = self.state_version(user_id)
        row = (
            self.db.query(KnowledgeInsight)
            .filter(
                KnowledgeInsight.user_id == user_id,
                KnowledgeInsight.subject_entity_id == subject_entity_id,
            )
            .first()
        )
        text = json.dumps(payload, ensure_ascii=False)
        if row is None:
            row = KnowledgeInsight(
                user_id=user_id,
                subject_entity_id=subject_entity_id,
                version=version,
                payload=text,
                model=knowledge_model_label(),
            )
            self.db.add(row)
        else:
            row.version = version
            row.payload = text
            row.model = knowledge_model_label()
            row.generated_at = datetime.now(timezone.utc)
        self.db.commit()
