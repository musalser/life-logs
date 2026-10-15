"""Закрытый словарь отношений: язык понимает модель, структуру держим мы.

Проблема, которую решает модуль: «мой отец», «батя», «my old man» — это один и
тот же смысл, и сводить его к словарным правилам бессмысленно (языков, сленга и
оборотов не перечесть). Поэтому словарных правил здесь нет: модель сама
возвращает **ключ из закрытого списка** (``father``), а не свободное слово, и
конвейер работает только с ключами.

Что даёт закрытый список:

* якорная сущность безымянного родственника получает идентичность
  ``(anchor_entity_id, anchor_relation)`` — «отец автора» существует в графе как
  объект, даже если имени в тексте нет;
* резолюция relative становится структурной (по ребру графа), а не угадыванием
  по имени;
* дедупликация фактов и рёбер сравнивает ключи, а не написания.

Русские подписи (``RELATION_LABELS_RU``) — только для интерфейса; в базе
канонический вид отношения — ключ.
"""
from __future__ import annotations

#: закрытый список отношений. Модель обязана выбрать ключ отсюда; всё, что не
#: помещается, идёт в ``other`` со свободной подписью.
RELATION_KINDS: tuple[str, ...] = (
    # прямые родственники
    "father", "mother", "parent",
    "son", "daughter", "child",
    "brother", "sister", "sibling",
    "husband", "wife", "spouse",
    # старшее и младшее поколение
    "grandfather", "grandmother", "great_grandfather", "great_grandmother",
    "grandchild", "great_grandchild",
    "uncle", "aunt", "nephew", "niece", "cousin",
    # свойство́ (родня по браку)
    "father_in_law", "mother_in_law", "son_in_law", "daughter_in_law",
    "brother_in_law", "sister_in_law",
    "stepfather", "stepmother", "stepson", "stepdaughter",
    # социальные
    "friend", "colleague", "boss", "subordinate", "neighbour", "acquaintance",
    "classmate", "teacher", "student", "doctor", "trainer",
    "godparent", "godchild",
    # всё остальное — со свободной подписью
    "other",
)

#: подписи для интерфейса (русский). Ключ — канонический вид.
RELATION_LABELS_RU: dict[str, str] = {
    "father": "отец", "mother": "мать", "parent": "родитель",
    "son": "сын", "daughter": "дочь", "child": "ребёнок",
    "brother": "брат", "sister": "сестра", "sibling": "брат или сестра",
    "husband": "муж", "wife": "жена", "spouse": "супруг",
    "grandfather": "дедушка", "grandmother": "бабушка",
    "great_grandfather": "прадед", "great_grandmother": "прабабушка",
    "grandchild": "внук", "great_grandchild": "правнук",
    "uncle": "дядя", "aunt": "тётя", "nephew": "племянник",
    "niece": "племянница", "cousin": "двоюродный родственник",
    "father_in_law": "тесть или свёкор", "mother_in_law": "тёща или свекровь",
    "son_in_law": "зять", "daughter_in_law": "невестка",
    "brother_in_law": "свояк", "sister_in_law": "свояченица",
    "stepfather": "отчим", "stepmother": "мачеха",
    "stepson": "пасынок", "stepdaughter": "падчерица",
    "friend": "друг", "colleague": "коллега", "boss": "начальник",
    "subordinate": "подчинённый", "neighbour": "сосед",
    "acquaintance": "знакомый", "classmate": "одноклассник",
    "teacher": "учитель", "student": "ученик", "doctor": "врач",
    "trainer": "тренер", "godparent": "крёстный", "godchild": "крестник",
    "other": "родственник или близкий",
}

#: отношения, которых у человека не может быть больше одного. По ним резолюция
#: relative идёт по единственному ребру графа; для остальных (брат, друг) одного
#: ребра мало — нужен резолвер или новая якорная сущность.
FUNCTIONAL_RELATIONS: frozenset[str] = frozenset({
    "father", "mother", "husband", "wife", "spouse",
    "stepfather", "stepmother",
})

#: обратный разбор: русское написание (и частый сленг) -> канонический ключ.
#: Нужен для переноса уже сохранённых рёбер и как страховка, если модель всё же
#: вернула слово вместо ключа.
RELATION_SURFACE_TO_KIND: dict[str, str] = {
    "отец": "father", "папа": "father", "папаня": "father", "батя": "father",
    "тятя": "father", "папка": "father", "father": "father", "dad": "father",
    "мать": "mother", "мама": "mother", "мамка": "mother", "матерь": "mother",
    "мамочка": "mother", "mother": "mother", "mom": "mother", "mum": "mother",
    "родитель": "parent", "родители": "parent",
    "сын": "son", "сынок": "son", "son": "son",
    "дочь": "daughter", "дочка": "daughter", "daughter": "daughter",
    "ребёнок": "child", "ребенок": "child", "child": "child",
    "брат": "brother", "братик": "brother", "brother": "brother",
    "сестра": "sister", "сестрёнка": "sister", "сестренка": "sister",
    "sister": "sister",
    "муж": "husband", "супруг": "husband", "husband": "husband",
    "жена": "wife", "супруга": "wife", "wife": "wife",
    "дед": "grandfather", "дедушка": "grandfather", "дедуля": "grandfather",
    "grandfather": "grandfather", "grandpa": "grandfather",
    "баба": "grandmother", "бабушка": "grandmother", "бабуля": "grandmother",
    "grandmother": "grandmother", "grandma": "grandmother",
    "прадед": "great_grandfather", "прабабушка": "great_grandmother",
    "внук": "grandchild", "внучка": "grandchild", "grandchild": "grandchild",
    "дядя": "uncle", "дяденька": "uncle", "uncle": "uncle",
    "тётя": "aunt", "тетя": "aunt", "тётенька": "aunt", "aunt": "aunt",
    "племянник": "nephew", "nephew": "nephew",
    "племянница": "niece", "niece": "niece",
    "двоюродный": "cousin", "двоюродная": "cousin", "cousin": "cousin",
    "тесть": "father_in_law", "свёкор": "father_in_law", "свекор": "father_in_law",
    "тёща": "mother_in_law", "теща": "mother_in_law", "свекровь": "mother_in_law",
    "зять": "son_in_law", "невестка": "daughter_in_law", "сноха": "daughter_in_law",
    "отчим": "stepfather", "мачеха": "stepmother",
    "пасынок": "stepson", "падчерица": "stepdaughter",
    "друг": "friend", "подруга": "friend", "дружок": "friend", "friend": "friend",
    "коллега": "colleague", "сослуживец": "colleague", "colleague": "colleague",
    "начальник": "boss", "шеф": "boss", "boss": "boss",
    "подчинённый": "subordinate", "подчиненный": "subordinate",
    "сосед": "neighbour", "соседка": "neighbour", "neighbour": "neighbour",
    "знакомый": "acquaintance", "знакомая": "acquaintance",
    "одноклассник": "classmate", "одноклассница": "classmate",
    "учитель": "teacher", "учительница": "teacher", "teacher": "teacher",
    "ученик": "student", "ученица": "student",
    "врач": "doctor", "доктор": "doctor", "doctor": "doctor",
    "тренер": "trainer", "trainer": "trainer",
    "крёстный": "godparent", "крестный": "godparent",
    "крёстная": "godparent", "крестная": "godparent",
    "крестник": "godchild", "крестница": "godchild",
}

RELATION_KIND_SET = frozenset(RELATION_KINDS)


def normalize_relation(value: str | None) -> str:
    """Свободное слово или ключ -> канонический ключ из закрытого списка.

    Модель обычно возвращает ключ (``father``); если вернула слово
    («батя», «my old man»), пробуем разобрать его. Неизвестное — ``other``.
    """
    text = " ".join((value or "").split()).strip().lower()
    if not text:
        return "other"
    if text in RELATION_KIND_SET:
        return text
    if text in RELATION_SURFACE_TO_KIND:
        return RELATION_SURFACE_TO_KIND[text]
    # «мой отец», «отец мой» — берём первое знакомое слово
    for word in text.replace("-", " ").split():
        if word in RELATION_SURFACE_TO_KIND:
            return RELATION_SURFACE_TO_KIND[word]
    return "other"


def relation_label(kind: str | None, free_label: str | None = None) -> str:
    """Подпись для интерфейса: русское слово из словаря или свободная подпись."""
    kind = normalize_relation(kind)
    if kind == "other":
        cleaned = " ".join((free_label or "").split())
        if cleaned:
            return cleaned[:64]
    return RELATION_LABELS_RU.get(kind, kind)


def is_functional(kind: str | None) -> bool:
    """У человека не может быть двух отцов (или двух мужей) — резолюция по ребру."""
    return normalize_relation(kind) in FUNCTIONAL_RELATIONS


def anchor_display_name(kind: str, anchor_name: str | None, *, of_self: bool) -> str:
    """Отображаемое имя якорной сущности: «Отец автора».

    Для вложенного якоря («жена брата») к имени добавляется имя самого якоря:
    «Жена — Брат автора».
    """
    label = RELATION_LABELS_RU.get(normalize_relation(kind), "родственник")
    label = label[:1].upper() + label[1:]
    if of_self:
        return f"{label} автора"
    return f"{label} — {anchor_name}" if anchor_name else label
