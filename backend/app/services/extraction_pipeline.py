"""Конвейер извлечения знаний из одного источника (этап 4 ТЗ, раздел 7).

Вызовы A–E, дедупликация (эмбеддинги отбирают кандидатов, LLM решает),
идемпотентная запись знаний с офсетами в тексте источника.

Как это устроено и почему так:

* **Каждый вызов независим.** Четыре обращения к модели (сущности, упоминания,
  факты, связи) идут последовательно, и падение одного не отменяет остальные:
  текст, из которого не извлеклись связи, всё равно даёт факты. Ошибки
  собираются в статистику прогона.
* **События и цели — это факты.** Событие — факт с важностью (``importance``),
  цель — факт с ``goal_title`` (только дневник). Отдельных вызовов нет: у факта
  уже есть разрешение субъекта, подтверждения и дедуп. Привычки не извлекаются:
  по одной странице регулярность не установить, их выводит аналитика по
  накопленным фактам (``recurrence`` — её сырьё).
* **Резолюция батчами.** Сопоставление «новое ↔ существующее» — самый дорогой
  шаг (он видит все головы пользователя), поэтому реляционные типы проходят его
  один раз на прогон, а не по элементу.
* **Эмбеддинг только отбирает кандидатов.** Замеры (ТЗ 4.5) показывают, что
  несвязанные русские фразы дают ~0.81, а два разных дня рождения одного ребёнка
  — 0.955. Поэтому порог решает лишь «показывать ли кандидата модели», а слияние
  подтверждает LLM. Если эмбеддинги недоступны, сужение пропускается и работает
  резолюция по последним N записям.
* **Офсеты проверяются.** Цитата обязана быть подстрокой текста источника:
  выдуманную цитату отбрасываем вместе с фактом, а не сохраняем с офсетом
  «примерно там». Если цитата найдена не там, где её назвала модель, берём
  первое вхождение.
* **Транзакцией владеет вызывающий.** ``KnowledgeSourceService`` уже очистил
  знание прошлого прогона в этой же транзакции; конвейер только пишет и делает
  ``flush``. Никаких ``commit`` внутри — иначе падение на середине оставило бы
  страницу наполовину извлечённой.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

from sqlalchemy.orm import Session

from ..config import settings
from ..models import (
    Chunk,
    Entity,
    EntityMention,
    EntityRelation,
    Fact,
    FactConflict,
    FactSource,
    Goal,
    GoalProgress,
    KnowledgeSource,
    User,
)
from ..adapters.cloud_llm import StructuredLLM, knowledge_adapter
from . import extraction_prompts as prompts
from .embedding_service import EmbeddingError, EmbeddingService, get_embedding_service
from .predicate_kinds import (
    DATE_VALUED_PREDICATES,
    PLACE_VALUED_PREDICATES,
    is_functional as is_functional_predicate,
    normalize_predicate_kind,
)
from .kinship import aggregate_confirmations
from .relation_kinds import (
    anchor_display_name,
    is_functional,
    normalize_relation,
    relation_label,
)
from .relation_store import (
    apply_aggregate,
    canonical_fields,
    confirmation_payload,
    confirmation_rows,
    upsert_confirmation,
)
from .source_text import chunk_text

logger = logging.getLogger(__name__)

#: реляционные типы, которые проходят резолюцию батчем
RESOLUTION_KINDS = ("entities", "goals")

#: сколько символов цитаты показывать в логах
LOG_QUOTE = 80


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _normalize(value: str | None) -> str:
    return " ".join((value or "").split()).lower()


def _clamp(value: Any, default: float | None = None) -> float | None:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _quote_and_offset(text: str, quote: str, hint: int | None = None) -> tuple[str, int, int] | None:
    """Finds the quote in the source text; None when it is not there verbatim.

    A manuscript page is a stack of lines, so the model reads a sentence that
    wraps across a line break as one continuous string and returns it that way
    («…дружили с ними. Мы…» while the page has «…дружили с ними.\nМы…»). Two
    passes handle that:

    1. the quote as it came back;
    2. the quote with every whitespace run collapsed, matched against the page
       with the same collapsing — the returned span is then widened to whole
       lines, so what the UI highlights is real page text, not a fragment of it.

    A model that *paraphrases* still gets the item dropped: an offset pointing
    at something else is worse than a missing fact, because the UI promises to
    show «the real text».
    """
    if not quote:
        return None
    if hint is not None and 0 <= hint <= len(text) - len(quote) and text.startswith(quote, hint):
        return quote, hint, hint + len(quote)
    start = text.find(quote)
    if start >= 0:
        return quote, start, start + len(quote)
    return _quote_with_collapsed_whitespace(text, quote)


def _collapse(value: str) -> tuple[str, list[int]]:
    """Squeezes whitespace runs; returns the text and offset -> original index."""
    collapsed: list[str] = []
    mapping: list[int] = []
    in_space = False
    for index, char in enumerate(value):
        if char.isspace():
            if in_space:
                continue
            collapsed.append(" ")
            mapping.append(index)
            in_space = True
        else:
            collapsed.append(char)
            mapping.append(index)
            in_space = False
    return "".join(collapsed), mapping


def _quote_with_collapsed_whitespace(text: str, quote: str) -> tuple[str, int, int] | None:
    collapsed_text, mapping = _collapse(text)
    collapsed_quote, _ = _collapse(quote)
    position = collapsed_text.find(collapsed_quote)
    if position < 0:
        return None
    start = mapping[position]
    end = mapping[min(position + len(collapsed_quote) - 1, len(mapping) - 1)] + 1
    return text[start:end], start, end


def _sentence_for(text: str, start: int, end: int, limit: int = 400) -> str:
    """The sentence around a span, for the mention's context."""
    left = max(text.rfind("\n", 0, start), text.rfind(". ", 0, start), text.rfind("! ", 0, start), text.rfind("? ", 0, start))
    right_candidates = [position for position in (text.find("\n", end), text.find(". ", end), text.find("! ", end), text.find("? ", end)) if position >= 0]
    right = min(right_candidates) if right_candidates else len(text)
    return text[left + 1 : min(right + 1, left + 1 + limit)].strip()


def _digits(text: str | None) -> set[str]:
    """Numbers in a time expression: what makes two similar events different."""
    return set(re.findall(r"\d{2,4}", text or ""))


#: Слова, которые сами по себе не имя: местоимения и указательные обороты.
#: Модель иногда возвращает их сущностями, а в базе от такого «объекта» только
#: мусор: у него нет имени, и в списке персон он выглядит случайным словом.
#:
#: Родственные обращения («мама», «отец», «бабушка») здесь НЕ перечислены
#: намеренно: в дневнике это обычное имя человека — «мама приехала» пишут и
#: когда у неё есть имя, и когда его не называют вовсе. Отсекаются только
#: обороты, которые не могут быть обращением: «мой отец», «мои родители».
_PRONOUNS = {
    "я", "мы", "вы", "ты", "он", "она", "они", "оно",
    "мой", "моя", "мои", "моё", "наш", "наша", "наши", "свой", "своя", "свои",
    "это", "этот", "эта", "эти", "тот", "та", "те", "этот человек", "эта женщина", "этот мужчина",
    "человек", "люди", "мужчина", "женщина", "девушка", "парень", "кто-то", "никто",
}
_POSSESSIVE_PREFIXES = (
    "мой ", "моя ", "мои ", "моё ", "наш ", "наша ", "наши ", "его ", "её ", "их ",
)


#: Нарицательные слова, которые модель иногда называет «человеком». Список не
#: претендует на полноту: он закрывает самые частые случаи из живых прогонов
#: («мох», «дела», «дом», «звание»), а остальное отсекает правило ниже.
_COMMON_NOUNS = {
    "дом", "дома", "мох", "дела", "дело", "собрания", "собрание", "звание",
    "работа", "работе", "лес", "лесу", "половина", "половину", "часть", "место",
    "время", "день", "годы", "город", "деревня", "улица", "школа", "жизнь",
    "письмо", "книга", "фотография", "память", "история", "война", "фронт",
    "тратуары", "тратцары", "мост", "озеро", "сад", "огород", "дача",
}


def _looks_like_a_common_noun(name: str) -> bool:
    """Одно нарицательное слово без заглавной буквы — не имя человека.

    Имена в дневнике пишут с заглавной («Маша», «Баба Галя», «Анна Трухина»),
    а «мох» и «дела» — нет. Односложные обращения вроде «мамы» и «отца» под это
    правило не попадают: они не в списке и обычно идут как обращение.
    """
    value = _clean(name)
    if not value:
        return True
    first = value[0]
    if first.isupper():  # «Мох» как имя — редкость, но пусть остаётся
        return False
    words = value.split()
    return len(words) == 1 and _normalize(value) in _COMMON_NOUNS


def _is_nameless_person(name: str) -> bool:
    """True для «мой отец», «мы», «этот человек» — их нельзя завести объектом.

    «мой отец Сергей» уже имя и остаётся: отсекаются только описания, за
    которыми имени нет, потому что обращаться к такому объекту в интерфейсе
    нечем — в списке персон он выглядит как случайное слово из текста.
    """
    value = _normalize(name)
    if not value:
        return True
    if value in _PRONOUNS:
        return True
    for prefix in _POSSESSIVE_PREFIXES:
        if value.startswith(prefix):
            rest = value[len(prefix) :].strip()
            # «мой отец» — описание, «мой отец Сергей» — имя
            return len(rest.split()) < 2
    return False


def _digits(text: str | None) -> set[str]:
    """Numbers in a time expression: what makes two similar events different."""
    return set(re.findall(r"\d{2,4}", text or ""))


def _fact_value(
    predicate_kind: str,
    *,
    time_text: str | None,
    object_text: str | None,
    object_name: str | None,
    time_iso: str | None = None,
) -> str:
    """Сравнимое значение функционального предиката.

    Для рождения и смерти это дата (набор чисел: «20 августа 1910» -> 20 1910),
    для места — нормализованная строка. Приведённая моделью дата ``time_iso``
    («1910-08-20») даёт те же числа, что и текст, поэтому «1910» и «20 августа
    1910» сходятся, а «1910» и «1911» — нет. Пустое значение сравнивать не с
    чем, и конфликтом оно не считается.
    """
    if predicate_kind in DATE_VALUED_PREDICATES:
        iso = _clean(time_iso)
        if iso:
            return f"iso:{iso}"
        # числовая сортировка: «20 1910», а не «1910 20» — значение читаемо
        return "text:" + " ".join(sorted(_digits(time_text), key=lambda digit: int(digit)))
    if predicate_kind in PLACE_VALUED_PREDICATES:
        return _normalize(object_name or object_text)
    return _normalize(time_text or object_text or object_name)


def _dates_differ(
    left_text: str | None, left_iso: str | None, right_text: str | None, right_iso: str | None
) -> bool:
    """Две даты факта точно разные (у одной из сторон даты нет — не разные).

    Приведённые даты сравниваются терпимо к точности («1910» и «1910-08-20» —
    одна дата), иначе сравниваются числа из текста.
    """
    a, b = _clean(left_iso), _clean(right_iso)
    if a and b:
        return not (a.startswith(b) or b.startswith(a))
    left_digits, right_digits = _digits(left_text), _digits(right_text)
    if left_digits and right_digits:
        return left_digits != right_digits
    return False


def _name_words(value: str | None) -> list[str]:
    return [word for word in re.split(r"[^\w]+", _normalize(value)) if word]


def _word_matches(a: str, b: str) -> bool:
    """Одно слово имени: равенство или общий корень («Виталии» ~ «Виталий»).

    Корень — общий префикс не короче 4 букв и без хвоста длиннее 2 букв, поэтому
    падежи и описки рукописи совпадают, а «Лена» с «Еленой» — нет.
    """
    if a == b:
        return True
    common = len(os.path.commonprefix([a, b]))
    return common >= 4 and common >= min(len(a), len(b)) - 2


def _name_matches(hint: str | None, name: str | None) -> bool:
    """Каждое слово подсказки находится в имени: «Павел» ~ «Павел Фредорович»."""
    hint_words, name_words = _name_words(hint), _name_words(name)
    if not hint_words or not name_words:
        return False
    return all(any(_word_matches(h, w) for w in name_words) for h in hint_words)


def _entity_names(entity: Entity) -> list[str]:
    try:
        aliases = json.loads(entity.aliases_json) if entity.aliases_json else []
    except (json.JSONDecodeError, TypeError):
        aliases = []
    return [entity.canonical_name] + [str(alias) for alias in aliases if alias]


def _is_goal(item: dict[str, Any]) -> bool:
    """Факт о цели: есть название цели и статус не «none»."""
    return bool(_clean(item.get("goal_title"))) and _normalize(item.get("goal_status")) != "none"


def _goal_items(facts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in facts if _is_goal(item)]


def _date_body(value: str) -> str:
    """Значение даты без пометки источника: ``iso:1910-08-20`` -> ``1910-08-20``."""
    _prefix, _sep, body = value.partition(":")
    return body if _sep else value


def _values_conflict(predicate_kind: str, existing: str, incoming: str) -> bool:
    """Разошлись ли два значения одного функционального предиката.

    Даты сравниваются терпимо к точности, но не вперемешку: приведённые моделью
    даты (``iso:``) сравниваются по префиксу — «1910» и «1910-08-20» это одно
    рождение, а «1910-08-20» и «1910-08-21» — разные. Если приведённой даты нет,
    сравниваются наборы чисел из текста («1910» против «20 августа 1910»). Даты
    разной природы (одна приведена, другая нет) не сравниваются — конфликта нет.
    """
    if predicate_kind in DATE_VALUED_PREDICATES:
        left, right = existing, incoming
        if not left or not right:
            return False
        if left.startswith("iso:") and right.startswith("iso:"):
            a, b = _date_body(left), _date_body(right)
            return not (a.startswith(b) or b.startswith(a))
        if left.startswith("iso:") != right.startswith("iso:"):
            return False
        left_set, right_set = set(_date_body(left).split()), set(_date_body(right).split())
        if not left_set or not right_set:
            return False
        return not (left_set <= right_set or right_set <= left_set)
    if not existing or not incoming:
        return False
    return existing != incoming


class ExtractionPipeline:
    """One run of extraction over one source."""

    def __init__(
        self,
        db: Session,
        source: KnowledgeSource,
        *,
        adapter: StructuredLLM | None = None,
        embeddings: EmbeddingService | None = None,
        service: Any | None = None,
    ) -> None:
        self.db = db
        self.source = source
        # Ollama по умолчанию; Claude/OpenAI/DeepSeek — по KNOWLEDGE_LLM_PROVIDER.
        # Свой адаптер конвейер закрывает сам: иначе HTTP-клиент облака закрывает
        # сборщик мусора уже после asyncio.run (воркер, скрипты) — «Event loop is closed»
        self._owns_adapter = adapter is None
        self.adapter = adapter or knowledge_adapter()
        self.embeddings = embeddings or get_embedding_service()
        #: KnowledgeSourceService — за автором-объектом и контекстом соседних страниц
        self.service = service
        self.text = source.text or ""
        self.author = self._author()
        self.author_name = self.author.canonical_name if self.author else None
        self.context = ""
        self.stats: dict[str, Any] = {"source_id": source.id, "source_type": source.source_type, "errors": []}
        self._mentions = 0
        #: рёбра, созданные в этом прогоне: дубль до flush база ещё не видит
        self._run_edges: dict[tuple[int, int, str], EntityRelation] = {}
        #: значения функциональных предикатов этого прогона:
        #: (субъект, предикат) -> (значение, факт)
        self._run_fact_values: dict[tuple[int, str], tuple[str, Fact]] = {}

    # ------------------------------------------------------------------
    # Вход
    # ------------------------------------------------------------------

    def _author(self) -> Entity | None:
        if self.service is not None:
            return self.service.author_entity(self.source)
        if self.source.author_entity_id is None:
            return None
        return self.db.query(Entity).filter(Entity.id == self.source.author_entity_id).first()

    def _context(self) -> str:
        if self.service is None:
            return ""
        return self.service.context_block(self.source)

    # ------------------------------------------------------------------
    # Запуск
    # ------------------------------------------------------------------

    async def run(self) -> dict[str, Any]:
        try:
            return await self._run()
        finally:
            if self._owns_adapter and hasattr(self.adapter, "aclose"):
                await self.adapter.aclose()

    async def _run(self) -> dict[str, Any]:
        if not self.text.strip():
            self.stats["skipped"] = "empty source text"
            return self.stats
        if self.author is None:
            # без автора некуда относить связи и факты
            raise ValueError("у источника нет автора-объекта")

        self.context = self._context()
        raw = await self._extract_all()
        self._counts("extracted", raw)

        await self._resolve(raw)
        self._write(raw)
        self.stats["mentions_kept"] = self._mentions
        await self._write_chunks()
        return self.stats

    def _counts(self, prefix: str, raw: dict[str, list[dict[str, Any]]]) -> None:
        for kind, items in raw.items():
            self.stats[f"{prefix}_{kind}"] = len(items)

    @property
    def with_goals(self) -> bool:
        """Цели извлекаются только из дневника: в рукописях архива их почти нет."""
        return self.source.source_type == "diary"

    async def _extract_all(self) -> dict[str, list[dict[str, Any]]]:
        """Четыре вызова; каждый сам по себе, сбой одного не отменяет остальные.

        Промпты начинаются с одного и того же блока документа, поэтому Ollama
        переиспользует KV-кэш текста страницы между вызовами.
        """
        raw: dict[str, list[dict[str, Any]]] = {
            "entities": [],
            "mentions": [],
            "facts": [],
            "relations": [],
        }
        author = self.author_name

        raw["entities"] = await self._call(
            "entities", prompts.ENTITIES_SCHEMA, lambda: prompts.entities_prompt(self.text, self.context, author)
        )
        names = [_clean(item.get("name")) for item in raw["entities"] if _clean(item.get("name"))]
        known = [entity.canonical_name for entity in self._known_people(limit=40)]
        for name in known:
            if name not in names:
                names.append(name)

        raw["mentions"] = await self._call(
            "mentions",
            prompts.MENTIONS_SCHEMA,
            lambda: prompts.mentions_prompt(self.text, self.context, names, author),
        )
        # граф известных близких: модель ссылается на уже заведённого человека
        # по id, а не заводит его заново под другим написанием имени
        graph = prompts.graph_block(self._author_relatives())
        raw["facts"] = await self._call(
            "facts",
            prompts.facts_schema(with_goals=self.with_goals),
            lambda: prompts.facts_prompt(self.text, self.context, author, graph, with_goals=self.with_goals),
        )
        raw["relations"] = await self._call(
            "relations",
            prompts.RELATIONS_SCHEMA,
            lambda: prompts.relations_prompt(self.text, self.context, author, known, graph),
        )
        return raw

    async def _call(self, kind: str, schema: dict, prompt_factory) -> list[dict[str, Any]]:
        try:
            raw = await self.adapter.generate_structured(prompt_factory(), schema)
            items = self._parse(raw, kind)
            logger.info("source=%s %s: %s items", self.source.id, kind, len(items))
            return items
        except Exception as exc:  # noqa: BLE001 - сбой вызова не роняет прогон
            logger.exception("Extraction call %s failed for source %s", kind, self.source.id)
            self.stats["errors"].append(f"{kind}: {exc}")
            return []

    @staticmethod
    def _parse(raw: str, kind: str) -> list[dict[str, Any]]:
        if not raw:
            return []
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            logger.warning("Unparsable %s payload: %r", kind, str(raw)[:200])
            return []
        items = payload.get(kind) if isinstance(payload, dict) else None
        if not isinstance(items, list):
            return []
        return [item for item in items if isinstance(item, dict)]

    # ------------------------------------------------------------------
    # Знакомые записи
    # ------------------------------------------------------------------

    def _author_relatives(self) -> list[tuple[int, str, str]]:
        """Близкие автора из графа: ``(id, имя, подпись отношения)``."""
        if self.author is None:
            return []
        rows = (
            self.db.query(EntityRelation, Entity)
            .join(Entity, Entity.id == EntityRelation.to_entity_id)
            .filter(
                EntityRelation.user_id == self.source.user_id,
                EntityRelation.from_entity_id == self.author.id,
            )
            .order_by(EntityRelation.id)
            .limit(50)
            .all()
        )
        return [(entity.id, entity.canonical_name, edge.relation_type) for edge, entity in rows]

    def _entity_by_id(self, entity_id: int) -> Entity | None:
        return (
            self.db.query(Entity)
            .filter(Entity.id == entity_id, Entity.user_id == self.source.user_id)
            .first()
        )

    def _known_people(self, limit: int) -> list[Entity]:
        """Люди с именами: якорные сущности («Отец автора») сюда не попадают.

        Якорь разрешается структурно — по паре (якорь, отношение), а не по
        имени, поэтому в списке «уже известных людей» ему делать нечего: его
        отображаемое имя не встречается в тексте и только шумит в промпте.
        """
        return (
            self.db.query(Entity)
            .filter(
                Entity.user_id == self.source.user_id,
                Entity.entity_type == "person",
                Entity.anchor_entity_id.is_(None),
            )
            .order_by(Entity.last_seen_at.desc().nullslast(), Entity.id.desc())
            .limit(limit)
            .all()
        )

    def _existing(self, kind: str, limit: int = prompts.EXISTING_LIMIT) -> list[tuple[int, str]]:
        if kind != "goals":
            return [(entity.id, entity.canonical_name) for entity in self._known_people(limit)]
        rows = (
            self.db.query(Goal)
            .filter(Goal.user_id == self.source.user_id)
            .order_by(Goal.updated_at.desc())
            .limit(limit)
            .all()
        )
        return [(row.id, row.title) for row in rows]

    def _existing_by_id(self, kind: str, ids: Iterable[int]) -> dict[int, Any]:
        ids = [record_id for record_id in ids if record_id and record_id > 0]
        if not ids:
            return {}
        model = {"entities": Entity, "goals": Goal}[kind]
        rows = self.db.query(model).filter(model.id.in_(ids)).all()
        return {row.id: row for row in rows}

    # ------------------------------------------------------------------
    # Резолюция: эмбеддинги сужают, LLM решает
    # ------------------------------------------------------------------

    async def _resolve(self, raw: dict[str, list[dict[str, Any]]]) -> None:
        await self._resolve_entities(raw["entities"])
        # цель — факт с названием цели: резолвер решает, та ли это цель, что уже есть
        if self.with_goals:
            await self._resolve_labels(
                "goals", _goal_items(raw["facts"]), lambda item: _clean(item.get("goal_title"))
            )
        await self._resolve_relatives(raw["facts"])

    async def _resolve_relatives(self, facts: list[dict[str, Any]]) -> None:
        """Роль с несколькими кандидатами (два брата) — решает резолвер, одним вызовом.

        Якорная сущность одна на роль только для «одного отца»: братьев,
        друзей, внуков может быть много. Если на роль уже есть несколько
        человек, выбор между ними — вопрос смысла, а не структуры, и его решает
        та же LLM-резолюция, что и для имён. Все такие роли страницы уходят в
        резолвер одним вызовом, а одинаковые упоминания («брат Павел» в пяти
        фактах) — одним кандидатом: раньше это был отдельный вызов на факт.
        Не выбрала — в записи создастся новая якорная сущность (слияние
        остаётся пользователю).
        """
        if self.author is None:
            return
        groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
        candidates_of: dict[tuple[str, str], list[Entity]] = {}
        for item in facts:
            if _normalize(item.get("subject_kind")) != "relative":
                continue
            if _normalize(item.get("subject_of")) not in ("", "self"):
                continue  # вложенный якорь разрешается структурно при записи
            kind = normalize_relation(item.get("subject_relation"))
            if kind not in candidates_of:
                candidates_of[kind] = self._relative_candidates(kind)
            candidates = candidates_of[kind]
            if len(candidates) == 1:
                item["_relative_id"] = candidates[0].id
                continue
            if len(candidates) < 2:
                continue
            label = _clean(item.get("subject_name")) or f"мой {relation_label(kind)}"
            groups.setdefault((kind, label), []).append(item)
        if not groups:
            return

        keys = list(groups)
        existing: dict[int, str] = {}
        neighbours: dict[int, list[tuple[int, str, float]]] = {}
        for index, (kind, _label) in enumerate(keys):
            neighbours[index] = [(entity.id, entity.canonical_name, 1.0) for entity in candidates_of[kind]]
            for entity in candidates_of[kind]:
                existing[entity.id] = entity.canonical_name
        matches = await self._ask_resolver(
            "entities",
            [(index, f"{label} ({relation_label(kind)} автора)") for index, (kind, label) in enumerate(keys)],
            list(existing.items()),
            neighbours,
        )
        for index, key in enumerate(keys):
            chosen = matches.get(index)
            # резолвер видит соседей всех кандидатов: выбор вне своей роли не принимаем
            allowed = {entity_id for entity_id, _, _ in neighbours[index]}
            for item in groups[key]:
                if chosen in allowed:
                    item["_relative_id"] = chosen
                    self._count("relative_resolved")
                else:
                    self._count("relative_unresolved")

    def _relative_candidates(self, kind: str) -> list[Entity]:
        """Кто из уже известных людей может быть этим родственником автора."""
        if self.author is None:
            return []
        found: dict[int, Entity] = {}
        anchors = (
            self.db.query(Entity)
            .filter(
                Entity.user_id == self.source.user_id,
                Entity.anchor_entity_id == self.author.id,
                Entity.anchor_relation == kind,
            )
            .all()
        )
        for entity in anchors:
            found[entity.id] = entity
        edges = (
            self.db.query(EntityRelation)
            .filter(
                EntityRelation.user_id == self.source.user_id,
                EntityRelation.from_entity_id == self.author.id,
                EntityRelation.relation_kind == kind,
            )
            .all()
        )
        for edge in edges:
            entity = self.db.query(Entity).filter(Entity.id == edge.to_entity_id).first()
            if entity is not None:
                found[entity.id] = entity
        return list(found.values())

    async def _resolve_labels(
        self,
        kind: str,
        items: list[dict[str, Any]],
        label_of,
        *,
        describe=None,
        existing_described: dict[int, str] | None = None,
    ) -> None:
        """Кандидаты отбираются по ``label_of``, а резолвер видит ``describe``.

        Эмбеддинг считается по голому имени (описание сдвинуло бы косинус), но
        решать «тот же ли человек» по одному имени нельзя: «Лена» (дочь
        Виталия) и «Елена Юрьевна» (мама автора) — разные люди.
        """
        if not items:
            return
        existing = self._existing(kind)
        existing_ids = {record_id for record_id, _ in existing}
        for item in items:
            item["_match_id"] = -1

        neighbours = await self._embedding_candidates(
            kind,
            [label_of(item) for item in items],
            existing,
        )
        pending_indexes = [index for index in range(len(items)) if index in neighbours]
        if not pending_indexes:
            self.stats[f"{kind}_no_candidates"] = len(items)
            return

        # модель видит только тех, кто прошёл порог: дорогой вызов не тратится
        # на заведомо разные записи, но и решает не косинус, а она
        for index in pending_indexes:
            items[index]["_neighbours"] = neighbours[index]

        shown = describe or label_of
        shown_existing = (
            [(record_id, existing_described.get(record_id, label)) for record_id, label in existing]
            if existing_described
            else existing
        )
        matches = await self._ask_resolver(
            kind,
            [(index, shown(items[index])) for index in pending_indexes],
            shown_existing,
            neighbours,
        )
        hits = 0
        for index, matched_id in matches.items():
            if matched_id in existing_ids:
                items[index]["_match_id"] = matched_id
                hits += 1
        self.stats[f"{kind}_matched"] = hits

    async def _resolve_entities(self, items: list[dict[str, Any]]) -> None:
        """Сущности: кандидаты и запросы идут одним батчем, ключ — имя.

        Резолвер видит имя вместе с описанием: у нового — что о нём сказано на
        странице, у известного — его описание в базе.
        """

        def described(name: str, description: str | None) -> str:
            description = _clean(description)
            return f"{name} — {description}" if description else name

        existing_described = {
            entity.id: described(entity.canonical_name, entity.description)
            for entity in self._known_people(prompts.EXISTING_LIMIT)
        }
        await self._resolve_labels(
            "entities",
            items,
            lambda item: _clean(item.get("name")),
            describe=lambda item: described(_clean(item.get("name")), item.get("description")),
            existing_described=existing_described,
        )

    async def _embedding_candidates(
        self,
        kind: str,
        labels: list[str],
        existing: list[tuple[int, str]],
    ) -> dict[int, list[tuple[int, str, float]]]:
        """top-k соседей по косинусу для каждого кандидата, если он выше порога."""
        if not existing or not labels:
            return {}
        try:
            vectors = await self.embeddings.embed_texts(
                labels + [label for _, label in existing], prompt_name=None
            )
        except EmbeddingError as exc:
            logger.warning("Embedding prefilter unavailable (%s); falling back to the model", exc)
            self.stats.setdefault("embedding_fallbacks", []).append(kind)
            return {index: [(record_id, label, 1.0) for record_id, label in existing] for index in range(len(labels))}

        candidate_vectors = vectors[: len(labels)]
        existing_vectors = vectors[len(labels) :]
        threshold = settings.knowledge_dedup_min_similarity
        top_k = max(1, settings.knowledge_dedup_candidates)
        result: dict[int, list[tuple[int, str, float]]] = {}
        for index, vector in enumerate(candidate_vectors):
            scored = [
                (existing[position][0], existing[position][1], EmbeddingService.cosine_similarity(vector, other))
                for position, other in enumerate(existing_vectors)
            ]
            scored.sort(key=lambda item: item[2], reverse=True)
            best = [item for item in scored[:top_k] if item[2] >= threshold]
            if best:
                result[index] = best
        self.stats[f"{kind}_candidates"] = len(result)
        return result

    async def _ask_resolver(
        self,
        kind: str,
        candidates: list[tuple[int, str]],
        existing: list[tuple[int, str]],
        neighbours: dict[int, list[tuple[int, str, float]]],
    ) -> dict[int, int]:
        """LLM-резолюция; кандидат видит только своих соседей и их id."""
        if not candidates:
            return {}
        visible_ids: list[int] = []
        for index, _ in candidates:
            for record_id, _, _ in neighbours.get(index, []):
                if record_id not in visible_ids:
                    visible_ids.append(record_id)
        if not visible_ids:
            return {}
        labels = {record_id: label for record_id, label in existing}
        prompt = prompts.resolution_prompt(
            kind,
            [label for _, label in candidates],
            [(record_id, labels.get(record_id, "")) for record_id in visible_ids],
        )
        try:
            raw = await self.adapter.generate_structured(prompt, prompts.RESOLUTION_SCHEMA)
        except Exception as exc:  # noqa: BLE001 - без резолюции просто создадим новое
            logger.exception("Resolution call for %s failed", kind)
            self.stats["errors"].append(f"resolution_{kind}: {exc}")
            return {}

        payload = self._parse(raw, "matches")
        valid = set(visible_ids)
        result: dict[int, int] = {}
        for item in payload:
            index = item.get("candidate_index")
            matched_id = item.get("matched_id")
            if not isinstance(index, int) or index < 0 or index >= len(candidates):
                continue
            if isinstance(matched_id, int) and matched_id in valid:
                result[candidates[index][0]] = matched_id
        return result

    # ------------------------------------------------------------------
    # Запись (идемпотентно: включая повторный запуск по тому же источнику)
    # ------------------------------------------------------------------

    def _write(self, raw: dict[str, list[dict[str, Any]]]) -> None:
        entities = self._write_entities(raw["entities"])
        self._write_mentions(raw["mentions"], entities)
        known = {**entities, **{entity.canonical_name: entity for entity in self._known_people(200)}}
        self._write_facts(raw["facts"], known)
        self._write_relations(raw["relations"], known)
        self.db.flush()

    def _get_or_create_entity(
        self,
        name: str,
        *,
        entity_type: str = "person",
        description: str | None = None,
        aliases: Sequence[str] = (),
    ) -> Entity | None:
        name = _clean(name)
        if not name:
            return None
        normalized = _normalize(name)[:255]
        entity = (
            self.db.query(Entity)
            .filter_by(user_id=self.source.user_id, entity_type=entity_type, normalized_name=normalized)
            .first()
        )
        now = _now()
        if entity is None:
            entity = Entity(
                user_id=self.source.user_id,
                entity_type=entity_type,
                canonical_name=name[:255],
                normalized_name=normalized,
                description=(_clean(description) or None),
                first_seen_at=now,
                last_seen_at=now,
            )
            self.db.add(entity)
            self.db.flush()
            self.stats["entities_created"] = self.stats.get("entities_created", 0) + 1
        else:
            entity.last_seen_at = now
            if description and not entity.description:
                entity.description = _clean(description)
        if aliases:
            self._merge_aliases(entity, aliases)
        return entity

    def _merge_aliases(self, entity: Entity, aliases: Sequence[str]) -> None:
        """Слияние персон пополняет алиасы, а не создаёт вторую запись."""
        try:
            current = json.loads(entity.aliases_json) if entity.aliases_json else []
        except (json.JSONDecodeError, TypeError):
            current = []
        if not isinstance(current, list):
            current = []
        changed = False
        for alias in aliases:
            alias = _clean(alias)
            if alias and alias not in current and _normalize(alias) != entity.normalized_name:
                current.append(alias[:255])
                changed = True
        if changed:
            entity.aliases_json = json.dumps(current, ensure_ascii=False)
            self.stats["aliases_added"] = self.stats.get("aliases_added", 0) + 1

    def _write_entities(self, items: list[dict[str, Any]]) -> dict[str, Entity]:
        """Имена из вызова A -> сущности; ключ — и название, и нормализованная форма.

        Счётчик ``entities_matched`` здесь обнуляется: резолюция уже посчитала
        совпадения (``_resolve_labels``), а эта запись только закрепляет их.
        """
        result: dict[str, Entity] = {}
        self.stats["entities_matched"] = 0
        for item in items:
            name = _clean(item.get("name"))
            entity_type = _clean(item.get("entity_type")) or "other"
            aliases = item.get("aliases") if isinstance(item.get("aliases"), list) else []
            matched_id = item.get("_match_id", -1)
            entity = None
            if matched_id and matched_id > 0:
                entity = self.db.query(Entity).filter(Entity.id == matched_id).first()
            if entity is not None:
                entity.last_seen_at = _now()
                if entity_type != "other" and entity.entity_type != entity_type:
                    # тип от модели точнее эвристики «person по умолчанию»
                    entity.entity_type = entity_type
                if item.get("description") and not entity.description:
                    entity.description = _clean(item.get("description"))
                self._merge_aliases(entity, aliases)
                self.stats["entities_matched"] = self.stats.get("entities_matched", 0) + 1
            else:
                if entity_type == "person" and (
                    _is_nameless_person(name) or _looks_like_a_common_noun(name)
                ):
                    # «мой отец», «мы»: имени нет, обращаться в интерфейсе нечем
                    self.stats["entities_skipped_nameless"] = (
                        self.stats.get("entities_skipped_nameless", 0) + 1
                    )
                    continue
                entity = self._get_or_create_entity(
                    name, entity_type=entity_type, description=item.get("description"), aliases=aliases
                )
                # новый объект уже посчитан как created; matched к нему не относится
            if entity is None:
                continue
            for key in (name, entity.canonical_name, _normalize(name), entity.normalized_name):
                if key:
                    result[key] = entity
        return result

    def _resolve_entity(
        self, known: dict[str, Entity], name: str, *, create: bool = False
    ) -> Entity | None:
        """Сущность по имени: найденная в прогоне, в базе или (если разрешено) новая.

        Создавать объект из любого упомянутого слова нельзя: вызовы фактов и
        связей возвращают в полях имён что угодно — «фамилию», «иногда»,
        «Председателем», «дом и двор», — и каждое такое слово становилось
        «персоной» в базе. Поэтому объект создаётся только там, где модель прямо
        называет сущность (вызов A), а факты, связи и упоминания лишь находят
        уже известный объект.
        """
        name = _clean(name)
        if not name:
            return None
        entity = known.get(name) or known.get(_normalize(name))
        if entity is not None:
            return entity
        if not create:
            self.stats["entities_unresolved"] = self.stats.get("entities_unresolved", 0) + 1
            return None
        if _is_nameless_person(name) or _looks_like_a_common_noun(name):
            return None
        return self._get_or_create_entity(name)

    def _write_mentions(self, items: list[dict[str, Any]], entities: dict[str, Entity]) -> None:
        """Упоминания принимаются только для сущностей, найденных вызовом A.

        Вызов упоминаний охотно придумывает имена, которых в списке сущностей нет
        («Председателем», «рано», «играть», «дом и двор»): он видит текст целиком
        и решает, что это тоже сущность. Создавать из такого «персону» нельзя —
        именно так в базе появлялись объекты, которые потом висели в списке на
        вкладке «Знания». Поэтому имя обязано уже существовать: либо в объектах
        прогона, либо в базе, либо это осмысленное имя (не нарицательное слово).
        """
        for item in items:
            entity_name = _clean(item.get("entity_name"))
            mention_text = _clean(item.get("mention_text"))
            if not entity_name or not mention_text:
                continue
            entity = entities.get(entity_name) or entities.get(_normalize(entity_name))
            if entity is None:
                self.stats["mentions_unknown_entity"] = (
                    self.stats.get("mentions_unknown_entity", 0) + 1
                )
                continue
            found = _quote_and_offset(self.text, mention_text)
            if found is None:
                self.stats.setdefault("mentions_dropped", 0)
                self.stats["mentions_dropped"] += 1
                continue
            quote, start, end = found
            self.db.add(
                EntityMention(
                    entity_id=entity.id,
                    source_id=self.source.id,
                    mention_text=quote[:255],
                    sentence_text=_clean(item.get("sentence_text")) or _sentence_for(self.text, start, end),
                    start_offset=start,
                    end_offset=end,
                    confidence=_clamp(item.get("confidence")),
                )
            )
            self._mentions += 1

    # ------------------------------------------------------------------
    # Действующие лица: явный субъект и якорные сущности (ТЗ, раздел 7.3)
    # ------------------------------------------------------------------

    def _count(self, key: str, amount: int = 1) -> None:
        self.stats[key] = self.stats.get(key, 0) + amount

    def _resolve_subject(
        self, item: dict[str, Any], known: dict[str, Entity]
    ) -> tuple[Entity | None, str | None]:
        """Кого касается утверждение. ``None`` — факт сохранять нельзя.

        Подстановки автора по умолчанию здесь нет намеренно: пустой или
        неизвестный субъект — ошибка валидации, а не «наверное, автор». Именно
        эта подстановка превращала «Отец мой … родился» в факт о самом авторе.

        Возвращает ``(сущность, источник)``; источник попадает в
        ``facts.subject_source`` и служит метрикой дрейфа.
        """
        # Явный id из графа сильнее любого имени: модель выбрала человека,
        # которого мы ей показали
        kind = _normalize(item.get("subject_kind"))
        subject_id = item.get("subject_id")
        if isinstance(subject_id, int) and not isinstance(subject_id, bool) and subject_id > 0:
            entity = self._entity_by_id(subject_id)
            if entity is not None and self._subject_id_agrees(item, kind, entity, known):
                self._count("facts_subject_graph")
                return entity, "graph"
            if entity is not None:
                # id противоречит тому, что модель сама сказала о субъекте
                # («Лена, дочь Виталия» с id мамы автора): решает структура
                self._count("facts_subject_id_rejected")
        if kind == "self":
            if self.author is None:
                return None, None
            self._count("facts_subject_self")
            return self.author, "self"
        if kind == "named":
            entity = self._resolve_entity(known, item.get("subject_name"))
            if entity is None:
                self._count("facts_subject_unresolved")
                return None, None
            if self.author is not None and entity.id == self.author.id:
                self._count("facts_subject_self")
                return self.author, "self"
            self._count("facts_subject_named")
            return entity, "named"
        if kind == "relative":
            entity = self._resolve_relative(
                known,
                item.get("subject_relation"),
                item.get("subject_of"),
                item.get("subject_name"),
                preferred_id=item.get("_relative_id"),
            )
            if entity is None:
                self._count("facts_subject_unresolved")
                return None, None
            self._count("facts_subject_relative")
            return entity, "relative"
        # unknown, пусто или мусор: без субъекта факт не сохраняется
        self._count("facts_subject_invalid")
        return None, None

    def _subject_id_agrees(
        self, item: dict[str, Any], kind: str, entity: Entity, known: dict[str, Entity]
    ) -> bool:
        """Совпадает ли выбранный по id человек с описанием субъекта.

        Для родственника — родство с якорем (``_kin_consistent``); для
        названного по имени — имя (каноническое или алиас). Без описания id
        принимается как есть: противоречить нечему.
        """
        name = _clean(item.get("subject_name"))
        if kind == "relative":
            anchor = self._relative_anchor(known, item.get("subject_of"))
            if anchor is None:
                return False
            relation = normalize_relation(item.get("subject_relation"))
            if not self._kin_consistent(anchor, entity, relation, name):
                return False
            if relation != "other":
                self._ensure_relation(anchor, entity, relation)
            return True
        if kind == "named" and name:
            return any(_name_matches(name, value) for value in _entity_names(entity))
        return True

    def _relative_anchor(self, known: dict[str, Entity], of: Any) -> Entity | None:
        """К кому относится родство: автор, вложенный родственник или человек по имени.

        Модель пишет в ``subject_of`` и ключ родства («father» — отец автора), и
        имя («Виталий»). Имя раньше читалось как вид родства «other», и «дочь
        Виталия» становилась родственницей безымянного «другого» автора.
        """
        value = _normalize(of)
        if value in ("", "self", "автор", "я"):
            return self.author
        if normalize_relation(of) != "other":
            return self._resolve_relative(known, value, "self", "")
        return self._resolve_entity(known, _clean(of))

    def _kin_consistent(self, anchor: Entity, entity: Entity, kind: str, name_hint: str | None) -> bool:
        """Может ли ``entity`` быть родственником ``anchor`` вида ``kind``.

        Да — если это уже записано (якорная сущность этой роли или ребро той же
        канонической пары) или если модель назвала имя и оно совпадает с
        каноническим именем человека. Алиасы здесь не в счёт: резолвер мог
        приписать «Лену» Елене Юрьевне в этом же прогоне, и проверка по алиасу
        подтвердила бы собственную ошибку.
        """
        if entity.id == anchor.id:
            return False
        if entity.anchor_entity_id == anchor.id and entity.anchor_relation == kind:
            return True
        if kind != "other":
            fields = canonical_fields(
                kind=kind, generations=None, via_marriage=None, bond=None,
                from_id=anchor.id, to_id=entity.id,
            )
            key = (fields["canon_from"], fields["canon_to"], fields["canon_key"])
            if key in self._run_edges:
                return True
            linked = (
                self.db.query(EntityRelation.id)
                .filter(
                    EntityRelation.user_id == self.source.user_id,
                    EntityRelation.canon_from == fields["canon_from"],
                    EntityRelation.canon_to == fields["canon_to"],
                    EntityRelation.canon_key == fields["canon_key"],
                )
                .first()
            )
            if linked is not None:
                return True
        return bool(name_hint) and _name_matches(name_hint, entity.canonical_name)

    def _resolve_relative(
        self,
        known: dict[str, Entity],
        relation: Any,
        of: Any,
        name_hint: Any,
        *,
        preferred_id: Any = None,
    ) -> Entity | None:
        """Якорная сущность родственника, разрешённая структурно (ТЗ, пункт 3).

        Порядок: названное имя → якорная сущность с той же идентичностью
        (якорь, отношение) → единственное ребро этого отношения от якоря →
        новая якорная сущность. Угадывания по словам здесь нет: язык уже свёл
        «батю» и «my old man» к ключу ``father``.
        """
        if self.author is None:
            return None
        kind = normalize_relation(relation)
        anchor = self._relative_anchor(known, of)
        if anchor is None:
            return None
        anchor_id = anchor.id

        # 0) выбор резолвера: кандидатов на роль было несколько, и модель выбрала
        #    одного из них (кандидаты уже связаны с якорем этой ролью)
        if isinstance(preferred_id, int) and not isinstance(preferred_id, bool) and preferred_id > 0:
            entity = self._entity_by_id(preferred_id)
            if entity is not None and entity.id != anchor_id:
                self._ensure_relation(anchor, entity, kind)
                return entity

        # 1) названное имя: «Федосив Александр Михайлович» уже есть объектом —
        #    но только если родство с ним не противоречит записанному
        hint = _clean(name_hint)
        if hint:
            entity = self._resolve_entity(known, hint)
            if entity is not None and self._kin_consistent(anchor, entity, kind, hint):
                self._ensure_relation(anchor, entity, kind)
                return entity
            if entity is not None:
                self._count("relative_name_rejected")
            # имя есть, но известный носитель — другой человек: заводим носителя
            # этого имени («Лена» ≠ «Елена Юрьевна»), а не безымянный якорь
            if not _is_nameless_person(hint) and not _looks_like_a_common_noun(hint):
                namesake = (
                    self.db.query(Entity)
                    .filter_by(user_id=self.source.user_id, entity_type="person", normalized_name=_normalize(hint)[:255])
                    .first()
                )
                if namesake is None or self._kin_consistent(anchor, namesake, kind, hint):
                    person = namesake or self._get_or_create_entity(hint)
                    if person is not None and person.id != anchor_id:
                        self._ensure_relation(anchor, person, kind)
                        return person

        # 2) якорная сущность с той же идентичностью (автор, father)
        anchors = (
            self.db.query(Entity)
            .filter(
                Entity.user_id == self.source.user_id,
                Entity.anchor_entity_id == anchor_id,
                Entity.anchor_relation == kind,
            )
            .order_by(Entity.id)
            .all()
        )
        if len(anchors) == 1:
            return anchors[0]

        # 3) единственное ребро нужного отношения: человек был назван раньше
        if not anchors:
            edges = (
                self.db.query(EntityRelation)
                .filter(
                    EntityRelation.user_id == self.source.user_id,
                    EntityRelation.from_entity_id == anchor_id,
                    EntityRelation.relation_kind == kind,
                )
                .order_by(EntityRelation.id)
                .all()
            )
            if len(edges) == 1:
                entity = (
                    self.db.query(Entity).filter(Entity.id == edges[0].to_entity_id).first()
                )
                if entity is not None and entity.id != anchor_id:
                    return entity

        # 4) кандидатов нет — создаём якорь; кандидатов несколько — создаём
        #    новый, а не угадываем: слияние решит пользователь или резолвер
        if anchors:
            self._count("relative_ambiguous")
            if is_functional(kind):
                # у человека не может быть двух отцов: это уже грязные данные
                self._count("functional_relation_ambiguous")
        entity = self._create_anchor(anchor, kind)
        self._ensure_relation(anchor, entity, kind)
        return entity

    def _create_anchor(self, anchor: Entity, kind: str) -> Entity:
        """Безымянный родственник становится объектом: «Отец автора»."""
        author_id = self.author.id if self.author else None
        name = anchor_display_name(kind, anchor.canonical_name, of_self=anchor.id == author_id)
        now = _now()
        entity = Entity(
            user_id=self.source.user_id,
            entity_type="person",
            canonical_name=name[:255],
            normalized_name=_normalize(name)[:255],
            anchor_entity_id=anchor.id,
            anchor_relation=kind,
            first_seen_at=now,
            last_seen_at=now,
        )
        self.db.add(entity)
        self.db.flush()
        self._count("anchors_created")
        return entity

    def _ensure_relation(
        self,
        anchor: Entity,
        person: Entity,
        kind: str,
        *,
        free_label: str | None = None,
        confidence: float | None = None,
        evidence: str | None = None,
        start: int | None = None,
        end: int | None = None,
        via_marriage: bool | None = None,
        generations: int | None = None,
    ) -> EntityRelation | None:
        """Ребро anchor → person: каноническая пара, подтверждение, пересчёт.

        Дедуп и уникальность считаются по канонической паре и токену вида, а не
        по направлению источника: «A → B, отец» и «B → A, ребёнок» — один факт, а
        «A → B, отец» и «B → A, отец» — разные (родительский цикл). Знак и число
        поколений считает код по таблице видов, модель их не присылает.
        """
        if person is None or anchor is None or person.id == anchor.id:
            return None
        label = relation_label(kind, free_label)[:64]
        fields = canonical_fields(
            kind=kind,
            generations=generations,
            via_marriage=via_marriage,
            bond=None,
            from_id=anchor.id,
            to_id=person.id,
            relation_type=label,
        )
        key = (fields["canon_from"], fields["canon_to"], fields["canon_key"])
        edge = self._run_edges.get(key)
        if edge is None:
            edge = (
                self.db.query(EntityRelation)
                .filter_by(
                    user_id=self.source.user_id,
                    canon_from=fields["canon_from"],
                    canon_to=fields["canon_to"],
                    canon_key=fields["canon_key"],
                )
                .first()
            )
        payload = confirmation_payload(
            edge,
            self.source.id,
            {**fields, "relation_type": label, "confidence": confidence},
            quote=evidence,
            start=start,
            end=end,
        )
        if edge is None:
            edge = EntityRelation(
                user_id=self.source.user_id,
                from_entity_id=anchor.id,
                to_entity_id=person.id,
                relation_type=label,
                relation_kind=fields["relation_kind"],
                generations=fields["generations"],
                via_marriage=fields["via_marriage"],
                bond=fields["bond"],
                canon_from=fields["canon_from"],
                canon_to=fields["canon_to"],
                canon_key=fields["canon_key"],
            )
            self.db.add(edge)
            # id нужен, чтобы записать подтверждение и увидеть его при пересчёте
            self.db.flush()
            self._count("relations_created")
        else:
            self._count("relations_merged")
        # уверенность и цитата ребра — из всех подтверждений, включая это
        rows = confirmation_rows(self.db, edge.id, skip_source=self.source.id)
        upsert_confirmation(self.db, edge, self.source.id, payload)
        rows.append(payload)
        aggregate = aggregate_confirmations(rows)
        if aggregate is not None:
            apply_aggregate(edge, aggregate)
        self._run_edges[key] = edge
        return edge

    def _write_facts(self, items: list[dict[str, Any]], known: dict[str, Entity]) -> None:
        for item in items:
            source_text = _clean(item.get("source_text"))
            found = _quote_and_offset(self.text, source_text)
            if found is None:
                self.stats["facts_dropped"] = self.stats.get("facts_dropped", 0) + 1
                continue
            quote, start, end = found
            subject, subject_source = self._resolve_subject(item, known)
            if subject is None:
                continue
            # цель пишется до дедупа и конфликтов факта: повтор того же намерения
            # на другой странице — это новая отметка прогресса, а не дубль
            if self.with_goals and _is_goal(item):
                self._write_goal(item, subject, quote)
            obj = self._resolve_entity(known, item.get("object_name"))
            predicate = _clean(item.get("predicate"))[:100]
            if not predicate:
                continue
            predicate_kind = normalize_predicate_kind(item.get("predicate_kind"), predicate)
            object_text = _clean(item.get("object_text")) or None
            time_text = _clean(item.get("time_text")) or None
            time_iso = _clean(item.get("time_iso")) or None
            importance = _clamp(item.get("importance")) or None  # 0 — не событие
            recurrence = _clean(item.get("recurrence"))[:64] or None
            # Инвариант функционального предиката стоит ДО дедупа: иначе две
            # разные даты рождения выглядят «одним фактом» по (субъект, предикат,
            # объект) — и вторая молча затирает первую. У человека одно рождение,
            # поэтому расхождение значений уходит «на проверку».
            if is_functional_predicate(predicate_kind) and subject is not None:
                incoming_value = _fact_value(
                    predicate_kind,
                    time_text=time_text,
                    object_text=object_text,
                    object_name=obj.canonical_name if obj else None,
                    time_iso=time_iso,
                )
                if not incoming_value:
                    # пустое значение не создаёт второй факт: иначе у человека
                    # окажется две смерти — с датой и без. Оно дополняет известное.
                    known_fact = self._known_functional_fact(subject, predicate_kind)
                    if known_fact is not None:
                        known_fact.confidence = max(
                            known_fact.confidence or 0.0, _clamp(item.get("confidence"), 0.0) or 0.0
                        )
                        self._remember_quote(
                            known_fact, quote, start, end, _clamp(item.get("confidence"))
                        )
                        self.stats["facts_merged"] = self.stats.get("facts_merged", 0) + 1
                        continue
                conflict = self._functional_conflict(subject, predicate_kind, incoming_value)
                if conflict is not None:
                    existing_fact, existing_value = conflict
                    self._record_conflict(
                        subject, predicate_kind, incoming_value, existing_value,
                        existing_fact, quote, time_text, _clamp(item.get("confidence")),
                    )
                    continue
            # дедупликация фактов без эмбеддингов: тот же субъект, предикат и
            # объект — и не разные даты: «день рождения сына» 2023 и 2024 — два
            # события, а не одно (раньше это стерёг отдельный вызов событий)
            duplicate = next(
                (
                    row
                    for row in self.db.query(Fact)
                    .filter(
                        Fact.user_id == self.source.user_id,
                        Fact.subject_entity_id == (subject.id if subject else None),
                        Fact.predicate == predicate,
                        Fact.object_entity_id == (obj.id if obj else None),
                        Fact.object_text == object_text,
                    )
                    .order_by(Fact.id)
                    .all()
                    if not _dates_differ(row.time_text, row.time_iso, time_text, time_iso)
                ),
                None,
            )
            if duplicate is not None:
                duplicate.confidence = max(duplicate.confidence or 0.0, _clamp(item.get("confidence"), 0.0) or 0.0)
                duplicate.subject_source = subject_source or duplicate.subject_source
                duplicate.time_iso = duplicate.time_iso or time_iso
                if importance is not None:
                    duplicate.importance = max(duplicate.importance or 0.0, importance)
                duplicate.recurrence = duplicate.recurrence or recurrence
                # Основная цитата не перезаписывается: прежнее подтверждение
                # остаётся, а новое добавляется отдельной строкой-подтверждением.
                self._remember_quote(duplicate, quote, start, end, _clamp(item.get("confidence")))
                self.stats["facts_merged"] = self.stats.get("facts_merged", 0) + 1
                continue
            fact = Fact(
                user_id=self.source.user_id,
                source_id=self.source.id,
                subject_entity_id=subject.id if subject else None,
                subject_source=subject_source,
                predicate=predicate,
                predicate_kind=predicate_kind,
                object_entity_id=obj.id if obj else None,
                object_text=object_text,
                source_text=quote,
                start_offset=start,
                end_offset=end,
                time_text=time_text,
                time_iso=time_iso,
                importance=importance,
                recurrence=recurrence,
                confidence=_clamp(item.get("confidence")),
            )
            self.db.add(fact)
            fact.quotes.append(
                FactSource(
                    source_id=self.source.id,
                    source_text=quote,
                    start_offset=start,
                    end_offset=end,
                    confidence=_clamp(item.get("confidence")),
                )
            )
            self._count("fact_quotes")
            if is_functional_predicate(predicate_kind) and subject is not None:
                # flush, чтобы у факта был id: на него сошлётся конфликт, если
                # второе значение придёт позже в этом же прогоне
                self.db.flush()
                self._run_fact_values[(subject.id, predicate_kind)] = (incoming_value, fact)
            self.stats["facts_created"] = self.stats.get("facts_created", 0) + 1

    def _remember_quote(
        self,
        fact: Fact,
        quote: str,
        start: int | None,
        end: int | None,
        confidence: float | None,
    ) -> None:
        """Добавляет подтверждение факта в этом источнике, не теряя прежних.

        Один факт — несколько страниц, и каждая со своей цитатой. Одна строка на
        источник: повторный прогон того же источника обновляет её, а не плодит.
        """
        for row in fact.quotes:
            if row.source_id == self.source.id:
                row.source_text = quote
                row.start_offset = start
                row.end_offset = end
                row.confidence = max(row.confidence or 0.0, confidence or 0.0)
                return
        fact.quotes.append(
            FactSource(
                source_id=self.source.id,
                source_text=quote,
                start_offset=start,
                end_offset=end,
                confidence=confidence,
            )
        )
        self._count("fact_quotes")

    def _known_functional_fact(self, subject: Entity, predicate_kind: str) -> Fact | None:
        """Первый факт этого функционального предиката: сохранённый или из прогона."""
        row = (
            self.db.query(Fact)
            .filter(
                Fact.user_id == self.source.user_id,
                Fact.subject_entity_id == subject.id,
                Fact.predicate_kind == predicate_kind,
            )
            .order_by(Fact.id)
            .first()
        )
        if row is not None:
            return row
        remembered = self._run_fact_values.get((subject.id, predicate_kind))
        return remembered[1] if remembered is not None else None

    def _functional_conflict(
        self, subject: Entity, predicate_kind: str, incoming: str
    ) -> tuple[Fact, str] | None:
        """Расхождение с уже сохранённым (или созданным в этом прогоне) значением."""
        existing = (
            self.db.query(Fact)
            .filter(
                Fact.user_id == self.source.user_id,
                Fact.subject_entity_id == subject.id,
                Fact.predicate_kind == predicate_kind,
            )
            .order_by(Fact.id)
            .all()
        )
        for row in existing:
            value = _fact_value(
                predicate_kind,
                time_text=row.time_text,
                object_text=row.object_text,
                object_name=row.object_entity.canonical_name if row.object_entity else None,
                time_iso=row.time_iso,
            )
            if _values_conflict(predicate_kind, value, incoming):
                return row, value
        remembered = self._run_fact_values.get((subject.id, predicate_kind))
        if remembered is not None and _values_conflict(predicate_kind, remembered[0], incoming):
            return remembered[1], remembered[0]
        return None

    def _record_conflict(
        self,
        subject: Entity,
        predicate_kind: str,
        incoming_value: str,
        existing_value: str,
        existing_fact: Fact | None,
        quote: str,
        time_text: str | None,
        confidence: float | None,
    ) -> None:
        """Записывает конфликт функционального предиката «на проверку»."""
        is_date = predicate_kind in DATE_VALUED_PREDICATES
        existing_display = _date_body(existing_value) if is_date else existing_value
        incoming_display = _date_body(incoming_value) if is_date else incoming_value
        self.db.add(
            FactConflict(
                user_id=self.source.user_id,
                source_id=self.source.id,
                subject_entity_id=subject.id,
                predicate_kind=predicate_kind,
                existing_fact_id=existing_fact.id if existing_fact is not None else None,
                existing_value=(existing_display or "")[:255] or None,
                incoming_value=(incoming_display or "")[:255] or None,
                source_text=quote,
                time_text=time_text,
                confidence=confidence,
            )
        )
        self._count("facts_conflicts")

    def _write_relations(self, items: list[dict[str, Any]], known: dict[str, Entity]) -> None:
        # Модель возвращает одно и то же ребро несколько раз (в одной странице
        # «отец» приходит дважды). Запросы к базе такого дубля не видят — строка
        # ещё не вставлена, — и flush падает на уникальности. Поэтому дубли
        # отсекаются в памяти прогона (`_run_edges`) по каноническому ключу.
        for item in items:
            person_name = _clean(item.get("person_name"))
            if not person_name:
                continue
            kind = normalize_relation(item.get("relation_kind"))
            found = _quote_and_offset(self.text, _clean(item.get("source_text")))
            evidence, start, end = found if found else (None, None, None)
            person = self._resolve_entity(known, person_name)
            if person is None or self.author is None or person.id == self.author.id:
                continue
            self._ensure_relation(
                self.author,
                person,
                kind,
                free_label=item.get("relation_label"),
                confidence=_clamp(item.get("confidence")),
                evidence=evidence,
                start=start,
                end=end,
                via_marriage=bool(item.get("via_marriage")),
                generations=item.get("generations"),
            )

    def _write_goal(self, item: dict[str, Any], subject: Entity, quote: str) -> None:
        """Цель из факта: новая или найденная резолвером, плюс отметка прогресса.

        Субъект цели — субъект факта, а не всегда автор страницы: «мама решила
        выучить английский» — цель мамы.
        """
        title = _clean(item.get("goal_title"))
        status_update = _normalize(item.get("goal_status"))
        if status_update not in prompts.GOAL_STATUSES or status_update == "none":
            status_update = "mentioned"
        note = _clean(item.get("object_text")) or _clean(item.get("predicate")) or None
        matched_id = item.get("_match_id", -1)
        goal = (
            self.db.query(Goal).filter(Goal.id == matched_id, Goal.user_id == self.source.user_id).first()
            if matched_id and matched_id > 0
            else None
        )
        if goal is None:
            goal = Goal(
                user_id=self.source.user_id,
                subject_entity_id=subject.id,
                title=title[:255],
                description=note,
                status="active",
                created_from_source_id=self.source.id,
            )
            self.db.add(goal)
            self.db.flush()
            # резолвер того же прогона мог не видеть только что созданную цель:
            # второй факт о ней в этой же странице должен попасть в неё же
            item["_match_id"] = goal.id
            self.stats["goals_created"] = self.stats.get("goals_created", 0) + 1
        else:
            self.stats["goals_merged"] = self.stats.get("goals_merged", 0) + 1
        if status_update in ("completed", "abandoned"):
            goal.status = status_update
        elif status_update in ("started", "progress"):
            goal.status = "active"
        goal.updated_at = _now()
        self.db.add(
            GoalProgress(
                goal_id=goal.id,
                source_id=self.source.id,
                progress_kind=status_update,
                note=note,
                source_text=quote,
                confidence=_clamp(item.get("confidence")),
            )
        )
        self.stats["goal_progress"] = self.stats.get("goal_progress", 0) + 1

    # ------------------------------------------------------------------
    # Чанки для RAG (эмбеддинги считаются здесь же и переиспользуются)
    # ------------------------------------------------------------------

    async def _write_chunks(self) -> None:
        spans = chunk_text(self.text, settings.rag_chunk_max_chars, settings.rag_chunk_overlap_chars)
        if not spans:
            return
        chunks = [
            Chunk(
                user_id=self.source.user_id,
                source_id=self.source.id,
                chunk_index=index,
                text=self.text[start:end],
                start_offset=start,
                end_offset=end,
            )
            for index, (start, end) in enumerate(spans)
        ]
        self.db.add_all(chunks)
        self.db.flush()
        self.stats["chunks_created"] = len(chunks)

        try:
            vectors = await self.embeddings.embed_texts(
                [chunk.text for chunk in chunks], prompt_name="Document"
            )
        except EmbeddingError as exc:
            # чанки без вектора переживут: их досчитают при первой индексации
            logger.warning("Chunks of source %s left without embeddings: %s", self.source.id, exc)
            self.stats["chunks_without_embeddings"] = len(chunks)
            return
        for chunk, vector in zip(chunks, vectors):
            chunk.embedding = list(vector)


def _default_embedding_service() -> EmbeddingService:
    return get_embedding_service()


async def run_extraction(db: Session, source: KnowledgeSource) -> dict[str, Any]:
    """Точка входа конвейера: читает источник и пишет знание из него.

    Вызывается ``KnowledgeSourceService`` внутри своей транзакции: очистка
    прошлого прогона уже сделана, а результат закоммитит вызывающий.
    """
    from .knowledge_source_service import KnowledgeSourceService

    pipeline = ExtractionPipeline(db, source, service=KnowledgeSourceService(db))
    return await pipeline.run()
