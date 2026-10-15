"""Закрытый словарь предикатов и инвариант функциональности.

Зачем: дедупликация по точной тройке (субъект, предикат, объект) работает,
только если предикат нормализован. «родился в 1910», «год рождения 1910» и
«born 1910» — это один предикат, и сводит его **модель** (классификация), а не
лексикон: языков и оборотов не перечесть.

Инвариант: часть предикатов функциональна — у человека одно рождение, одна
смерть, одно место рождения. Второе, отличающееся значение — не новый факт, а
**конфликт на проверку** (``fact_conflicts``). Именно так «Alexander родился в
1910» рядом с настоящей датой рождения ловится без единого слова про отца.

Здесь нет правил на обороты: только ключи, подписи для интерфейса и разбор
значения (дата/место) для сравнения.
"""
from __future__ import annotations

#: закрытый список предикатов; всё, что не помещается, — ``other``
PREDICATE_KINDS: tuple[str, ...] = (
    "birth", "birth_place", "death", "death_place",
    "full_name", "occupation", "residence", "education", "other",
)

PREDICATE_LABELS_RU: dict[str, str] = {
    "birth": "рождение",
    "birth_place": "место рождения",
    "death": "смерть",
    "death_place": "место смерти",
    "full_name": "полное имя",
    "occupation": "занятие",
    "residence": "место жительства",
    "education": "образование",
    "other": "другое",
}

#: у человека одно рождение, одна смерть, одно место рождения и смерти:
#: второе, отличающееся значение — конфликт, а не новый факт
FUNCTIONAL_PREDICATES: frozenset[str] = frozenset({
    "birth", "birth_place", "death", "death_place",
})

#: значение сравнивается как дата (набор чисел: 20 августа 1910 -> {20, 1910})
DATE_VALUED_PREDICATES: frozenset[str] = frozenset({"birth", "death"})

#: значение сравнивается как место (нормализованная строка)
PLACE_VALUED_PREDICATES: frozenset[str] = frozenset({"birth_place", "death_place"})

#: страховка, если модель вернула слово вместо ключа (основной путь — ключ)
PREDICATE_SURFACE_TO_KIND: dict[str, str] = {
    "родился": "birth", "родилась": "birth", "родились": "birth",
    "рождение": "birth", "родился в": "birth", "born": "birth", "birth": "birth",
    "место рождения": "birth_place", "родился в деревне": "birth_place",
    "birth_place": "birth_place",
    "умер": "death", "умерла": "death", "умерли": "death", "скончался": "death",
    "скончалась": "death", "смерть": "death", "died": "death", "death": "death",
    "место смерти": "death_place", "death_place": "death_place",
    "имя": "full_name", "полное имя": "full_name", "ф.и.о.": "full_name",
    "фио": "full_name", "full_name": "full_name",
    "работал": "occupation", "работала": "occupation", "профессия": "occupation",
    "occupation": "occupation",
    "жил": "residence", "жила": "residence", "проживал": "residence",
    "residence": "residence",
    "учился": "education", "училась": "education", "образование": "education",
    "education": "education",
}

PREDICATE_KIND_SET = frozenset(PREDICATE_KINDS)


def normalize_predicate_kind(value: str | None, predicate: str | None = None) -> str:
    """Ключ предиката: из поля модели или, как страховка, из её же слова."""
    for candidate in (value, predicate):
        text = " ".join((candidate or "").split()).strip().lower()
        if not text:
            continue
        if text in PREDICATE_KIND_SET:
            return text
        if text in PREDICATE_SURFACE_TO_KIND:
            return PREDICATE_SURFACE_TO_KIND[text]
        for word in text.replace("-", " ").split():
            if word in PREDICATE_SURFACE_TO_KIND:
                return PREDICATE_SURFACE_TO_KIND[word]
    return "other"


def is_functional(kind: str | None) -> bool:
    return normalize_predicate_kind(kind) in FUNCTIONAL_PREDICATES
