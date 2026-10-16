"""Структурный слой родства: пара (up, down) и флаги — без знаковых шагов.

Модель на извлечении называет отношение словом из закрытого списка, а структуру
считает этот модуль. Знак при «мой дед» и «Баба Галя → внук» легко перепутать,
поэтому на границе с LLM знакового числа нет вообще: код берёт пару (up, down)
из таблицы, а шаг выводит как ``down − up``.

**Что такое (up, down).** Для ребра «A → B, вид»: ``up`` — сколько поколений от
A вверх до общего предка, ``down`` — сколько от него вниз до B. Тогда

    отец (1, 0) → −1      брат (1, 1) → 0        дядя (2, 1) → −1
    дед  (2, 0) → −2      двоюродный (2, 2) → 0  племянник (1, 2) → +1
    внук (0, 2) → +2

Пара покрывает и прямые линии, и боковые; знаковый ``step`` не покрывал бы
дядю и двоюродного.

**Свойство́ — это флаг, а не отдельные виды.** ``via_marriage`` не меняет шаг
кровной части: тесть (1, 0) + брак → −1, зять (0, 1) + брак → +1, шурин и жена
брата (1, 1) + брак → 0. Поэтому ``parent_in_law`` / ``child_in_law`` как виды не
нужны. Два брачных звена в пару не ложатся (сват, свояк) — они остаются
отдельными видами с нулевым шагом и собственными токенами ключа.

**Флаги.** ``via_marriage`` — отношение проходит через брак; ``bond`` —
``blood | half | step | adoptive | ritual``. ``half`` отличает сводного брата от
родного, ``step``/``adoptive`` остаются в слоях (поколение у них надёжное).
Только ``ritual`` (крёстные, крестники, кум и кума) не определяет поколение:
такие рёбра не участвуют в расчёте уровней и **не соединяют компоненты** —
человек, иначе недостижимый, попадает в «уровень не определён», а не получает
ноль по умолчанию.

**В базе пара (up, down) не хранится.** Источник истины — ``relation_kind``
(закрытый список) плюс ``generations``; пара выводится из таблицы. Иначе первая
же поправка Δ потребовала бы миграции данных.

**Лексикона здесь нет.** Модуль работает только с ключами закрытого словаря
(``relation_kinds``); сведение слов «тесть»/«батя» к виду — задача модели.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

BOND_BLOOD = "blood"
BOND_HALF = "half"
BOND_STEP = "step"
BOND_ADOPTIVE = "adoptive"
BOND_RITUAL = "ritual"

#: глубокие степени («прадед и дальше») модель уточняет числом поколений.
#: Вне диапазона значение не принимается: 0 или 40 — это ошибка, а не глубина.
GENERATIONS_MIN = 3
GENERATIONS_MAX = 6


@dataclass(frozen=True)
class Kin:
    """Положение человека B относительно автора ребра A."""

    up: int
    down: int
    via_marriage: bool = False
    bond: str = BOND_BLOOD

    @property
    def step(self) -> int:
        """Сколько поколений от A до B: отрицательное — B выше."""
        return self.down - self.up


def _kin(up: int, down: int, *, marriage: bool = False, bond: str = BOND_BLOOD) -> Kin:
    return Kin(up=up, down=down, via_marriage=marriage, bond=bond)


#: вид -> (up, down, via_marriage, bond). Вида нет в таблице — либо ритуальная
#: связь, либо не родня вовсе (см. RITUAL_KINDS / NOT_KIN ниже).
KIN_TABLE: dict[str, Kin] = {
    # прямые линии
    "father": _kin(1, 0),
    "mother": _kin(1, 0),
    "parent": _kin(1, 0),
    "son": _kin(0, 1),
    "daughter": _kin(0, 1),
    "child": _kin(0, 1),
    "grandfather": _kin(2, 0),
    "grandmother": _kin(2, 0),
    "grandparent": _kin(2, 0),
    "grandchild": _kin(0, 2),
    "great_grandfather": _kin(3, 0),
    "great_grandmother": _kin(3, 0),
    "great_grandparent": _kin(3, 0),
    "great_grandchild": _kin(0, 3),
    # братья и сёстры: сводный отличается флагом, а не парой
    "brother": _kin(1, 1),
    "sister": _kin(1, 1),
    "sibling": _kin(1, 1),
    "half_brother": _kin(1, 1, bond=BOND_HALF),
    "half_sister": _kin(1, 1, bond=BOND_HALF),
    "stepbrother": _kin(1, 1, marriage=True, bond=BOND_STEP),
    "stepsister": _kin(1, 1, marriage=True, bond=BOND_STEP),
    # боковые линии
    "uncle": _kin(2, 1),
    "aunt": _kin(2, 1),
    "nephew": _kin(1, 2),
    "niece": _kin(1, 2),
    "cousin": _kin(2, 2),
    # брак
    "husband": _kin(0, 0, marriage=True),
    "wife": _kin(0, 0, marriage=True),
    "spouse": _kin(0, 0, marriage=True),
    # отчим/мачеха и пасынки: поколение надёжное, поэтому в слоях
    "stepfather": _kin(1, 0, marriage=True, bond=BOND_STEP),
    "stepmother": _kin(1, 0, marriage=True, bond=BOND_STEP),
    "stepson": _kin(0, 1, marriage=True, bond=BOND_STEP),
    "stepdaughter": _kin(0, 1, marriage=True, bond=BOND_STEP),
    # приёмные дети и приёмные родители
    "adoptive_parent": _kin(1, 0, bond=BOND_ADOPTIVE),
    "adopted_child": _kin(0, 1, bond=BOND_ADOPTIVE),
    # два брачных звена в пару не ложатся — отдельные виды с нулевым шагом
    "co_parent_in_law": _kin(0, 0, marriage=True),   # сват, сватья
    "svojak": _kin(0, 0, marriage=True),            # свояк
}

#: виды без дельты: поколение не определено самим отношением
RITUAL_KINDS: frozenset[str] = frozenset({
    "godfather", "godmother", "godparent", "godchild", "co_godparent",
})
AMBIGUOUS_KINDS: frozenset[str] = frozenset({"in_law_unknown"})

#: не родня: в древо не попадает, но в графе видна
NOT_KIN: frozenset[str] = frozenset({
    "friend", "colleague", "boss", "subordinate", "neighbour", "acquaintance",
    "classmate", "teacher", "student", "doctor", "trainer", "other",
})

#: составные виды: пара (0, 0) + брак у них одна, поэтому ключ уникальности
#: различает их отдельными токенами — иначе супруг, сват и свояк склеятся
COMPOSITE_TOKENS: dict[str, str] = {
    "husband": "pair:spouse",
    "wife": "pair:spouse",
    "spouse": "pair:spouse",
    "co_parent_in_law": "pair:co_parent_in_law",
    "svojak": "pair:svojak",
}

#: направление канонической пары. ``to_senior`` — старший это ``to`` (отец),
#: ``from_senior`` — старший это ``from`` (ребёнок), ``sym`` — пара без
#: направления (супруг, друг), ``as_is`` — направление источника (начальник,
#: врач). Ключ строится по канонической паре, поэтому «A → B, отец» и
#: «B → A, отец» — разные факты, а не один: это ошибка направления, и она должна
#: попасть в диагностику, а не молча склеиться.
ORIENT_TO_SENIOR = "to_senior"
ORIENT_FROM_SENIOR = "from_senior"
ORIENT_SYM = "sym"
ORIENT_AS_IS = "as_is"

#: асимметричные роли с обратной парой: без общего токена «A → B, начальник» и
#: «B → A, подчинённый» остались бы двумя линиями и завысили бы степень.
#: Токен общий на пару, ориентация — по роли (старший вперёд).
_REVERSE_ROLES: dict[str, tuple[str, str]] = {
    "boss": ("role:boss", ORIENT_TO_SENIOR),
    "subordinate": ("role:boss", ORIENT_FROM_SENIOR),
    "teacher": ("role:teacher", ORIENT_TO_SENIOR),
    "student": ("role:teacher", ORIENT_FROM_SENIOR),
    "godfather": ("pair:godparent", ORIENT_TO_SENIOR),
    "godmother": ("pair:godparent", ORIENT_TO_SENIOR),
    "godparent": ("pair:godparent", ORIENT_TO_SENIOR),
    "godchild": ("pair:godparent", ORIENT_FROM_SENIOR),
}

#: нулевой шаг, симметричный по смыслу
_SYMMETRIC_ZERO: frozenset[str] = frozenset({
    "husband", "wife", "spouse", "brother", "sister", "sibling",
    "half_brother", "half_sister", "stepbrother", "stepsister", "cousin",
    "co_parent_in_law", "svojak",
    "friend", "neighbour", "acquaintance", "classmate", "colleague",
})

#: виды, у которых число поколений уточняется моделью
_ANCESTOR_LINEAL: frozenset[str] = frozenset({
    "parent", "father", "mother", "grandparent", "grandfather", "grandmother",
    "great_grandparent", "great_grandfather", "great_grandmother",
})
_DESCENDANT_LINEAL: frozenset[str] = frozenset({
    "child", "son", "daughter", "grandchild", "great_grandchild",
})
_SYMMETRIC_DEEP: frozenset[str] = frozenset({"cousin"})


def generations_ok(value: Any) -> bool:
    """Число поколений от модели: только осмысленная глубина, без 0 и 40."""
    if isinstance(value, bool):
        return False
    try:
        depth = int(value)
    except (TypeError, ValueError):
        return False
    return GENERATIONS_MIN <= depth <= GENERATIONS_MAX


def kin_of(
    relation_kind: str | None,
    generations: int = 0,
    *,
    via_marriage: bool | None = None,
    bond: str | None = None,
) -> Kin | None:
    """Пара (up, down) и флаги для вида; ``None`` — поколение не определено.

    ``generations`` — неотрицательное число поколений от модели для глубоких
    степеней (прадед -> 4); знак здесь не участвует. Значение вне диапазона
    игнорируется: лучше взять таблицу, чем получить «40 поколений».

    ``via_marriage`` и ``bond`` приходят от извлечения отдельными полями — именно
    так выражается свойство́: «тесть» это вид ``parent`` с флагом брака, а не
    отдельный канонический вид ``parent_in_law``.
    """
    kind = (relation_kind or "").strip().lower()
    if not kind:
        return None
    if kind in RITUAL_KINDS or kind in AMBIGUOUS_KINDS:
        return None
    base = KIN_TABLE.get(kind)
    if base is None:
        return None
    depth = int(generations or 0) if generations_ok(generations) else 0
    if depth > 1:
        if kind in _ANCESTOR_LINEAL:
            base = Kin(up=depth, down=0, via_marriage=base.via_marriage, bond=base.bond)
        elif kind in _DESCENDANT_LINEAL:
            base = Kin(up=0, down=depth, via_marriage=base.via_marriage, bond=base.bond)
        elif kind in _SYMMETRIC_DEEP:
            base = Kin(up=depth, down=depth, via_marriage=base.via_marriage, bond=base.bond)
    # bond вид подразумевает сам (отчим — step, сводный — half): код перекрывает
    # ответ модели, иначе токен и колонка разойдутся. Брак вид не подразумевает
    # для кровных видов — его добавляет извлечение («тесть» = parent + брак).
    return Kin(
        up=base.up,
        down=base.down,
        via_marriage=base.via_marriage or bool(via_marriage),
        bond=base.bond,
    )


def is_kin(relation_kind: str | None) -> bool:
    """Родня ли это вообще (ритуальная связь и «не выбрал» тоже сюда входят)."""
    kind = (relation_kind or "").strip().lower()
    return kind in KIN_TABLE or kind in RITUAL_KINDS or kind in AMBIGUOUS_KINDS


def is_ritual(relation_kind: str | None) -> bool:
    return (relation_kind or "").strip().lower() in RITUAL_KINDS


def is_ambiguous(relation_kind: str | None) -> bool:
    return (relation_kind or "").strip().lower() in AMBIGUOUS_KINDS


def pair_key(left: int, right: int) -> tuple[int, int]:
    """Пара без направления — для симметричных видов."""
    return (left, right) if left <= right else (right, left)


def orientation_of(
    relation_kind: str | None,
    generations: int = 0,
    *,
    via_marriage: bool | None = None,
    bond: str | None = None,
) -> str:
    """Как ориентировать каноническую пару для этого вида."""
    kind = (relation_kind or "").strip().lower()
    reverse = _REVERSE_ROLES.get(kind)
    if reverse is not None:
        return reverse[1]
    kin = kin_of(kind, generations, via_marriage=via_marriage, bond=bond)
    if kin is None:
        # ритуальное, неопределённое и не-родня: направление источника
        return ORIENT_AS_IS
    if kin.step < 0:
        return ORIENT_TO_SENIOR
    if kin.step > 0:
        return ORIENT_FROM_SENIOR
    return ORIENT_SYM if kind in _SYMMETRIC_ZERO else ORIENT_AS_IS


def canon_pair(
    relation_kind: str | None,
    from_id: int,
    to_id: int,
    generations: int = 0,
    *,
    via_marriage: bool | None = None,
    bond: str | None = None,
) -> tuple[int, int]:
    """Каноническая пара для ключа уникальности: старший вперёд, симметричное по id.

    Ключ строится по этой паре, а не по паре без направления: иначе «A → B, отец»
    и «B → A, отец» (родительский цикл) и «A → B, отец» с «A → B, ребёнок»
    получили бы один ключ и слабейшее ребро молча исчезло бы.
    """
    orientation = orientation_of(
        relation_kind, generations, via_marriage=via_marriage, bond=bond
    )
    if orientation == ORIENT_TO_SENIOR:
        return (to_id, from_id)
    if orientation in (ORIENT_FROM_SENIOR, ORIENT_AS_IS):
        return (from_id, to_id)
    return pair_key(from_id, to_id)


def other_label_hash(relation_type: str | None) -> str:
    """Хеш свободной подписи для вида ``other``: разные подписи — разные факты."""
    text = " ".join((relation_type or "").split()).lower()
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


def canon_token(
    relation_kind: str | None,
    generations: int = 0,
    *,
    via_marriage: bool | None = None,
    bond: str | None = None,
    relation_type: str | None = None,
) -> str:
    """Токен канонической формы для ключа уникальности.

    Родственные виды — ``shape:{меньшее}:{большее}:{брак}:{bond}``: тесть
    (0:1:1) и отчим (0:1:1:step) дают разные токены, а «отец» и «сын» с двух
    сторон — один. Составные виды (супруг, сват, свояк) получают собственные
    токены, иначе их пара (0, 0) + брак не отличима от супруга. Всё остальное —
    ``kind:{вид}``; для ``other`` к ключу добавляется хеш подписи, иначе два
    разных факта одной пары склеились бы. NULL-дельта в уникальный ключ не
    попадает вовсе — NULL-ы в нём различны.
    """
    kind = (relation_kind or "").strip().lower()
    if kind in COMPOSITE_TOKENS:
        return COMPOSITE_TOKENS[kind]
    reverse = _REVERSE_ROLES.get(kind)
    if reverse is not None:
        return reverse[0]
    kin = kin_of(kind, generations, via_marriage=via_marriage, bond=bond)
    if kin is None:
        if kind in ("", "other"):
            return f"kind:other:{other_label_hash(relation_type)}"
        return f"kind:{kind}"
    low, high = sorted((kin.up, kin.down))
    return f"shape:{low}:{high}:{int(kin.via_marriage)}:{kin.bond}"


@dataclass
class CanonicalRelation:
    """Одно ребро на факт: каноническая пара, токен и все подтверждения.

    Направление источника сохраняется (``from_id``/``to_id``), а для уникальности
    считается каноническая пара: старший вперёд у линий, по id у симметричных, по
    роли у обратных пар и как у источника у остальных. Поэтому «A → B, отец» и
    «B → A, отец» — разные факты (родительский цикл), а «A → B, отец» и
    «B → A, ребёнок» — один.
    """

    from_id: int
    to_id: int
    canon_from: int
    canon_to: int
    canon_key: str
    kind: str
    relation_type: str | None = None
    generations: int | None = None
    via_marriage: bool = False
    bond: str = BOND_BLOOD
    up: int | None = None
    down: int | None = None
    confidence: float | None = None
    sources: tuple[int, ...] = ()
    edge_id: int | None = None
    #: все виды, которыми факт назван в источниках (диагностика расхождения)
    kinds: tuple[str, ...] = ()

    @property
    def delta(self) -> int | None:
        """``level(to) − level(from)``; ``None`` — поколение не определено."""
        if self.up is None or self.down is None:
            return None
        return self.down - self.up

    @property
    def blood_parent_child(self) -> tuple[int, int] | None:
        """(родитель, ребёнок) для кровного ребра в одну ступень, иначе None."""
        if self.delta is None or self.via_marriage or self.bond != BOND_BLOOD:
            return None
        if self.up == 1 and self.down == 0:
            return self.to_id, self.from_id
        if self.up == 0 and self.down == 1:
            return self.from_id, self.to_id
        return None


def _raw_priority(edge: Mapping[str, Any]) -> tuple:
    """Порядок рёбер: увереннее и подтверждённее — раньше (меньше — лучше)."""
    confidence = edge.get("confidence")
    sources = edge.get("sources") or ()
    return (
        -(float(confidence) if confidence is not None else 0.0),
        -len(sources),
        int(edge.get("id") or 0),
        int(edge.get("from") or 0),
        int(edge.get("to") or 0),
    )


def _confidence_of(edge: Mapping[str, Any]) -> float | None:
    value = edge.get("confidence")
    return float(value) if value is not None else None


def _sources_of(edge: Mapping[str, Any]) -> tuple[int, ...]:
    """Источники ребра или одного подтверждения (у подтверждения — ``source_id``)."""
    raw = edge.get("sources")
    if raw is None and edge.get("source_id") is not None:
        return (int(edge["source_id"]),)
    return tuple(sorted({int(s) for s in (raw or ()) if s is not None}))


def canonicalize_relations(edges: Iterable[Mapping[str, Any]]) -> list[CanonicalRelation]:
    """Сворачивает рёбра к одному факту: каноническая пара плюс токен.

    Работает со **всеми** видами, а не только с роднёй: обратные роли
    (начальник/подчинённый, учитель/ученик, крёстный/крестник) обязаны
    склеиться, иначе между людьми останутся две линии и двойная степень.
    Представителем остаётся самое уверенное ребро — вместе со своим видом и
    направлением, поэтому пол («отец» ≠ «мать») не подменяется, а расхождение
    видов видно в ``kinds``.
    """
    merged: dict[tuple[int, int, str], tuple[tuple, CanonicalRelation]] = {}
    for edge in edges:
        kind = (edge.get("relation_kind") or "").strip().lower()
        generations = edge.get("generations")
        depth = int(generations) if generations_ok(generations) else 0
        kin = kin_of(
            kind, depth, via_marriage=edge.get("via_marriage"), bond=edge.get("bond")
        )
        from_id, to_id = int(edge["from"]), int(edge["to"])
        canon_from, canon_to = canon_pair(
            kind,
            from_id,
            to_id,
            depth,
            via_marriage=edge.get("via_marriage"),
            bond=edge.get("bond"),
        )
        token = canon_token(
            kind,
            depth,
            via_marriage=edge.get("via_marriage"),
            bond=edge.get("bond"),
            relation_type=edge.get("relation_type"),
        )
        item = CanonicalRelation(
            from_id=from_id,
            to_id=to_id,
            canon_from=canon_from,
            canon_to=canon_to,
            canon_key=token,
            kind=kind or "other",
            relation_type=edge.get("relation_type"),
            generations=int(generations) if generations_ok(generations) else None,
            via_marriage=kin.via_marriage if kin else bool(edge.get("via_marriage")),
            bond=kin.bond if kin else (edge.get("bond") or BOND_BLOOD),
            up=kin.up if kin else None,
            down=kin.down if kin else None,
            confidence=_confidence_of(edge),
            sources=_sources_of(edge),
            edge_id=edge.get("id"),
            kinds=(kind or "other",),
        )
        key = (canon_from, canon_to, token)
        priority = _raw_priority(edge)
        existing = merged.get(key)
        if existing is None:
            merged[key] = (priority, item)
            continue
        best_priority, best = existing
        sources = tuple(sorted(set(best.sources) | set(item.sources)))
        confidence = max(best.confidence or 0.0, item.confidence or 0.0) or None
        kinds = tuple(sorted(set(best.kinds) | set(item.kinds)))
        if priority < best_priority:
            item.sources, item.confidence, item.kinds = sources, confidence, kinds
            merged[key] = (priority, item)
        else:
            best.sources, best.confidence, best.kinds = sources, confidence, kinds
    return sorted(
        (item for _priority, item in merged.values()),
        key=lambda e: (e.canon_from, e.canon_to, e.canon_key),
    )


def canonicalize_kin_edges(
    edges: Iterable[Mapping[str, Any]],
) -> list[CanonicalRelation]:
    """Только родственные рёбра с определённым поколением — вход для уровней."""
    return [item for item in canonicalize_relations(edges) if item.delta is not None]


def aggregate_confirmations(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any] | None:
    """Одно ребро из его подтверждений: представитель и совокупная уверенность.

    Нужен дважды: при записи (слияние источников в одно ребро) и после очистки
    источника — строка обязана отражать **оставшиеся** подтверждения, иначе
    уверенность и цитата устаревают. ``None`` означает, что подтверждений не
    осталось: ребро удаляется.
    """
    items = list(rows)
    if not items:
        return None
    best = min(items, key=_raw_priority)
    gathered: set[int] = set()
    for row in items:
        gathered.update(_sources_of(row))
    sources = sorted(gathered)
    kinds: list[str] = []
    for row in items:
        kind = (row.get("relation_kind") or "").strip().lower()
        if kind and kind not in kinds:
            kinds.append(kind)
    confidence = max((float(r.get("confidence") or 0.0) for r in items), default=0.0)
    return {
        "relation_kind": (best.get("relation_kind") or "").strip().lower() or None,
        "relation_type": best.get("relation_type"),
        #: источник-представитель: он остаётся «основным» для интерфейса
        "primary_source_id": best.get("source_id"),
        "generations": best.get("generations"),
        "via_marriage": best.get("via_marriage"),
        "bond": best.get("bond"),
        "confidence": confidence or None,
        "sources": sources,
        "kinds": kinds,
        "evidence_text": best.get("evidence_text"),
        "start_offset": best.get("start_offset"),
        "end_offset": best.get("end_offset"),
    }


@dataclass
class LevelResult:
    """Уровни поколений, конфликты данных и «уровень не определён»."""

    #: entity_id -> уровень (0 — низший в своей компоненте). None — не определён.
    levels: dict[int, int | None] = field(default_factory=dict)
    #: entity_id -> id компоненты родства (для раскладки и для «вне древа»)
    components: dict[int, int] = field(default_factory=dict)
    #: нарушенная разность, два кровных родителя, пара «якорная + именованная»
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    #: недостижимые по родству: причина ritual | ambiguous | not_kin | isolated
    unplaced: list[dict[str, Any]] = field(default_factory=list)
    #: рёбра, ссылающиеся на узлы вне переданного набора: это битые данные
    dangling: list[dict[str, Any]] = field(default_factory=list)


def levels(
    node_ids: Sequence[int],
    edges: Iterable[Mapping[str, Any]],
    *,
    anchors: Iterable[int] = (),
) -> LevelResult:
    """Уровни как согласованная система, не зависящая от корня и порядка рёбер.

    По каждой компоненте родства строится остовный лес с приоритетом рёбер
    (уверенность, число источников, id), затем **все** рёбра проверяются на
    ``level(to) − level(from) == down − up``. Нарушившие уходят в ``conflicts``
    как данные — там же дубли ролей (два кровных отца) и пары «якорная и
    именованная», которые надо слить выше по конвейеру.

    Ритуальные рёбра и «не выбрал» не участвуют и **не соединяют компоненты**:
    недостижимый человек честно попадает в ``unplaced``, а не получает ноль.
    Ребро, ссылающееся на узел вне набора, не добавляет его молча — оно
    попадает в ``dangling``.
    """
    anchors_set = {int(a) for a in anchors}
    declared = {int(n) for n in node_ids}
    ritual_or_ambiguous: dict[int, str] = {}
    dangling: list[dict[str, Any]] = []
    usable: list[Mapping[str, Any]] = []
    for raw in edges:
        from_id, to_id = int(raw["from"]), int(raw["to"])
        if from_id not in declared or to_id not in declared:
            dangling.append(
                {
                    "kind": "dangling_endpoint",
                    "edge_id": raw.get("id"),
                    "from": from_id,
                    "to": to_id,
                    "missing": sorted({from_id, to_id} - declared),
                }
            )
            continue
        usable.append(raw)
        kind = (raw.get("relation_kind") or "").strip().lower()
        if is_ritual(kind):
            ritual_or_ambiguous[from_id] = "ritual"
            ritual_or_ambiguous[to_id] = "ritual"
        elif is_ambiguous(kind):
            ritual_or_ambiguous[from_id] = "ambiguous"
            ritual_or_ambiguous[to_id] = "ambiguous"

    all_relations = canonicalize_relations(usable)
    kin_relations = [item for item in all_relations if item.delta is not None]

    def priority(edge: CanonicalRelation) -> tuple:
        return (
            -(edge.confidence or 0.0),
            -len(edge.sources),
            edge.edge_id or 0,
            edge.from_id,
            edge.to_id,
        )

    ordered = sorted(kin_relations, key=priority)

    # остовный лес: соединяем компоненты самыми надёжными рёбрами
    parent: dict[int, int] = {node: node for node in sorted(declared)}

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    forest: list[CanonicalRelation] = []
    for edge in ordered:
        left, right = find(edge.from_id), find(edge.to_id)
        if left == right:
            continue
        parent[left] = right
        forest.append(edge)

    # обход дерева: level(to) = level(from) + delta
    adjacency: dict[int, list[tuple[int, int]]] = {}
    for edge in forest:
        adjacency.setdefault(edge.from_id, []).append((edge.to_id, edge.delta))
        adjacency.setdefault(edge.to_id, []).append((edge.from_id, -edge.delta))

    result = LevelResult(dangling=dangling)
    result.levels = {node: None for node in sorted(declared)}
    # уровень получают только те, кого соединяет родство: узел с одним лишь
    # ритуальным ребром не должен «по умолчанию» оказаться на нуле
    kin_nodes = {edge.from_id for edge in forest} | {edge.to_id for edge in forest}
    for node in sorted(kin_nodes):
        if result.levels[node] is not None:
            continue
        root = find(node)
        component: list[int] = []
        queue = [(node, 0)]
        result.levels[node] = 0
        while queue:
            current, level_value = queue.pop()
            component.append(current)
            result.components[current] = root
            for neighbour, delta in adjacency.get(current, []):
                if result.levels.get(neighbour) is not None:
                    continue
                result.levels[neighbour] = level_value + delta
                queue.append((neighbour, level_value + delta))
        lowest = min(result.levels[member] for member in component)
        for member in component:
            result.levels[member] -= lowest

    # проверка всех рёбер, включая не попавшие в лес
    for edge in ordered:
        left = result.levels.get(edge.from_id)
        right = result.levels.get(edge.to_id)
        if left is None or right is None:
            continue
        if right - left != edge.delta:
            result.conflicts.append(
                {
                    "kind": "level_mismatch",
                    "edge_id": edge.edge_id,
                    "from": edge.from_id,
                    "to": edge.to_id,
                    "relation_kind": edge.kind,
                    "expected_delta": edge.delta,
                    "actual_delta": right - left,
                }
            )

    _functional_parent_conflicts(result, kin_relations, anchors_set)
    _kind_disagreements(result, all_relations)

    for node in sorted(declared):
        if result.levels.get(node) is not None:
            continue
        reason = ritual_or_ambiguous.get(node)
        if reason is None:
            reason = "isolated"
        result.unplaced.append({"entity_id": node, "reason": reason})
    return result


def _kind_disagreements(result: LevelResult, canonical: Sequence[CanonicalRelation]) -> None:
    """Источники одного ребра назвали его по-разному («дядя» и «тётя»).

    После слияния в строке остаётся вид победителя, и без этой диагностики
    расхождение исчезло бы молча.
    """
    for edge in canonical:
        if len(edge.kinds) < 2:
            continue
        result.conflicts.append(
            {
                "kind": "kind_disagreement",
                "from": edge.from_id,
                "to": edge.to_id,
                "kinds": list(edge.kinds),
                "kept": edge.kind,
            }
        )


#: слот, который вид занимает у ребёнка. «Родитель» без пола конкурирует и с
#: отцом, и с матерью: это либо тот же человек, либо ошибка атрибуции.
_PARENT_SLOTS: dict[str, str] = {
    "father": "father",
    "mother": "mother",
    "parent": "parent",
}


def parent_slot(relation_kind: str | None) -> str | None:
    """Какой родительский слот занимает вид (father | mother | parent)."""
    return _PARENT_SLOTS.get((relation_kind or "").strip().lower())


def _functional_parent_conflicts(
    result: LevelResult,
    canonical: Sequence[CanonicalRelation],
    anchors: set[int],
) -> None:
    """Двое в одном родительском слоте — диагностика, а не значок на узле.

    Признак конфликта — переполненный слот, а не «больше одного родителя»: отец и
    мать это норма, а два отца или два «родителя» — дубль (якорная «Отец автора» и
    именованный), ошибка атрибуции (текст одного автора приписан другому) или
    отчим, записанный как отец. Слот читается по ``relation_kind`` (закрытый
    список), а не по свободному тексту, поэтому лексикона тут нет.

    ``parents`` — **весь** записанный набор кровных родителей: именно он и есть
    находка («у Alexander записаны Сергей Владимирович, Елена Юрьевна и Федосив
    Александр Михайлович»), а ``duplicated_slots`` показывает, что в нём
    противоречиво. Сужать список до переполненного слота нельзя — мать из находки
    пропала бы.
    """
    slots: dict[int, dict[str, set[int]]] = {}
    for edge in canonical:
        pair = edge.blood_parent_child
        slot = parent_slot(edge.kind)
        if pair is None or slot is None:
            continue
        parent_id, child_id = pair
        slots.setdefault(child_id, {}).setdefault(slot, set()).add(parent_id)
    for child, by_slot in sorted(slots.items()):
        # «родитель» без пола претендует на оба слота
        father_like = by_slot.get("father", set()) | by_slot.get("parent", set())
        mother_like = by_slot.get("mother", set()) | by_slot.get("parent", set())
        duplicated = {
            slot: sorted(occupants)
            for slot, occupants in (("father", father_like), ("mother", mother_like))
            if len(occupants) > 1
        }
        if not duplicated:
            continue
        all_parents = sorted({p for occupants in by_slot.values() for p in occupants})
        result.conflicts.append(
            {
                "kind": "functional_parent",
                "child": child,
                "parents": all_parents,
                "duplicated_slots": duplicated,
            }
        )
        for slot, occupants in duplicated.items():
            anchor_parents = sorted(p for p in occupants if p in anchors)
            named_parents = sorted(p for p in occupants if p not in anchors)
            if anchor_parents and named_parents:
                result.conflicts.append(
                    {
                        "kind": "anchor_named_pair",
                        "slot": slot,
                        "child": child,
                        "anchor": anchor_parents[0],
                        "named": named_parents[0],
                        "merge_candidates": [anchor_parents[0], named_parents[0]],
                    }
                )
