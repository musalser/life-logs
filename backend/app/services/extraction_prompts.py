"""Промпты и JSON-схемы извлечения знаний (этап 4 ТЗ).

Промпты вынесены из сервиса: их читают чаще, чем код вокруг, а тесты проверяют
именно формулировки — «связь от лица автора текста» и «контекст соседних
страниц не извлекать» должны сломать тест, а не тихо испортить данные.

**Порядок частей важен для скорости.** Каждый промпт начинается с одного и того
же блока документа (правила, автор, контекст соседних страниц, текст страницы —
см. ``document_block``), и только потом идёт задача вызова. Ollama хранит
KV-кэш общего префикса между запросами к одной модели, поэтому длинный текст
страницы считается один раз на прогон, а не в каждом вызове. Всё, что
отличается между вызовами (задача, списки известных людей), стоит после текста.

**Вызовов три, а не семь.** События — это факты с важностью (``importance``),
цели — факты с ``goal_title`` (только для дневника), привычки не извлекаются
вовсе: по одной странице регулярность не установить, их выводит аналитика по
накопленным фактам (``recurrence`` — её сырьё).

Все схемы — плоские (только object/array/string/number/integer), потому что
Ollama передаёт их в грамматику llama.cpp: непривычные конструкции там
принимаются не всеми сборками.
"""
from __future__ import annotations

from typing import Any

from .predicate_kinds import PREDICATE_KINDS
from .relation_kinds import RELATION_KINDS

#: чем ограничено число источников в промптах резолюции и фактов
EXISTING_LIMIT = 200

#: словарь отношений, который видит модель: ключ закрытого списка, а не слово.
#: «Батя», «папаня» и «my old man» — это один ключ father, поэтому правило не
#: нужно ни на оборот, ни на язык — сведение делает модель.
RELATION_VOCABULARY = ", ".join(RELATION_KINDS)

#: словарь предикатов: «родился», «год рождения» и «born» — один ключ birth
PREDICATE_VOCABULARY = ", ".join(PREDICATE_KINDS)

#: правило субъекта факта: субъект называется явно, автор по умолчанию не
#: подставляется (см. ТЗ, раздел 7.3)
FACTS_SUBJECT_RULE = (
    "СУБЪЕКТ указывается явно, угадывать его нельзя, и подставлять автора по "
    "умолчанию запрещено — пустой subject_kind считается ошибкой:\n"
    "- subject_kind=self — утверждение о самом авторе текста («я родился…»);\n"
    "- subject_kind=named — человек назван по имени: subject_name — это имя («Пётр»);\n"
    "- subject_kind=relative — человек назван через родство: subject_relation — "
    "отношение (ключ из списка ниже), subject_of — к кому это отношение («self» — "
    "к автору). Если имя тоже названо, положи его в subject_name. Пример: «Отец мой "
    "Федосив Александр Михайлович родился…» — subject_kind=relative, "
    "subject_relation=father, subject_of=self, subject_name=Федосив Александр "
    "Михайлович;\n"
    "- subject_kind=unknown — о ком утверждение, непонятно. Такой факт не сохраняется.\n"
    "Если в предложении действует другой человек, а не автор, — это НЕ self.\n"
    "- subject_id: если субъект — человек из списка близких автора (ниже), верни "
    "его id; иначе 0. Это надёжнее имени: в тексте оно может быть написано иначе.\n\n"
    f"Отношения (subject_relation, subject_of): {RELATION_VOCABULARY}. "
    "«Батя», «папаня», «my old man» — это father; не подходит под список — other.\n\n"
)

#: блок «близкие автора» из графа: имя, отношение и id, чтобы модель ссылалась
#: на уже известного человека, а не заводила его заново
GRAPH_RULE = (
    "Близкие автора, уже известные в базе. Если человек из этого списка — верни "
    "его id в subject_id (или используй имя как оно записано):\n"
)


def graph_block(relatives: list[tuple[int, str, str]]) -> str:
    """``[(id, name, relation_label)]`` -> блок промпта (пусто, если близких нет)."""
    if not relatives:
        return ""
    lines = "\n".join(f"- {label} — {name} [{entity_id}]" for entity_id, name, label in relatives)
    return f"{GRAPH_RULE}{lines}\n\n"

ENTITIES_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "entity_type": {
                        "type": "string",
                        "enum": ["person", "organization", "location", "event", "object", "date", "other"],
                    },
                    "description": {"type": "string"},
                    "aliases": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["name", "entity_type", "description", "aliases"],
            },
        }
    },
    "required": ["entities"],
}

MENTIONS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "mentions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "entity_name": {"type": "string"},
                    "mention_text": {"type": "string"},
                    "sentence_text": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["entity_name", "mention_text", "sentence_text", "confidence"],
            },
        }
    },
    "required": ["mentions"],
}

#: статусы цели в факте; none — факт не о цели
GOAL_STATUSES = ("none", "started", "progress", "completed", "abandoned", "mentioned")

#: поля, общие для фактов любого источника
_FACT_PROPERTIES: dict[str, Any] = {
    #: субъект указывается явно, см. FACTS_SUBJECT_RULE
    "subject_kind": {
        "type": "string",
        "enum": ["self", "named", "relative", "unknown"],
    },
    "subject_name": {"type": "string"},
    "subject_relation": {"type": "string"},
    "subject_of": {"type": "string"},
    #: id человека из списка близких автора (0 — не из списка)
    "subject_id": {"type": "integer"},
    "predicate": {"type": "string"},
    #: канонический ключ предиката (birth, death, …) для дедупа
    "predicate_kind": {"type": "string", "enum": list(PREDICATE_KINDS)},
    "object_name": {"type": "string"},
    "object_text": {"type": "string"},
    "source_text": {"type": "string"},
    "time_text": {"type": "string"},
    #: та же дата, приведённая к ГГГГ[-ММ[-ДД]] — для сравнения
    "time_iso": {"type": "string"},
    #: событие — факт с важностью (0 — обычное утверждение)
    "importance": {"type": "number"},
    #: как текст говорит о регулярности; пусто — не сказано
    "recurrence": {"type": "string"},
    "confidence": {"type": "number"},
}

#: цель — тоже факт: название и статус продвижения (только дневник)
_GOAL_PROPERTIES: dict[str, Any] = {
    "goal_title": {"type": "string"},
    "goal_status": {"type": "string", "enum": list(GOAL_STATUSES)},
}


def facts_schema(with_goals: bool = False) -> dict[str, Any]:
    """Схема вызова фактов; с целями — для дневника, без — для рукописей."""
    properties = dict(_FACT_PROPERTIES)
    if with_goals:
        properties.update(_GOAL_PROPERTIES)
    return {
        "type": "object",
        "properties": {
            "facts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": properties,
                    "required": list(properties),
                },
            }
        },
        "required": ["facts"],
    }


FACTS_SCHEMA: dict[str, Any] = facts_schema(with_goals=False)
FACTS_WITH_GOALS_SCHEMA: dict[str, Any] = facts_schema(with_goals=True)

RELATIONS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "person_name": {"type": "string"},
                    #: ключ из закрытого словаря отношений (father, mother, …)
                    "relation_kind": {"type": "string"},
                    #: свободная подпись — только когда relation_kind = other
                    "relation_label": {"type": "string"},
                    #: связь проходит через брак (тесть, зять, шурин): знак
                    #: и число поколений считает код, а не модель
                    "via_marriage": {"type": "boolean"},
                    #: глубина для прадеда и дальше (3..6); иначе 0
                    "generations": {"type": "integer"},
                    "source_text": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": [
                    "person_name",
                    "relation_kind",
                    "relation_label",
                    "via_marriage",
                    "generations",
                    "source_text",
                    "confidence",
                ],
            },
        }
    },
    "required": ["relations"],
}

#: matched_id = -1 значит «ничего не подошло, создавай новое»
RESOLUTION_SCHEMA: dict[str, Any] = {
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

#: общее правило для всех вызовов: контекст соседних страниц не извлекаем
CONTEXT_RULE = (
    "Ниже может быть «Контекст соседних страниц» — он даётся только чтобы понять "
    "оборванное предложение. НИКОГДА не извлекай знания из этого контекста: "
    "извлекай только из основного текста.\n\n"
)

SOURCE_RULE = (
    "Если в задаче есть поле source_text — это точная цитата ИЗ ОСНОВНОГО ТЕКСТА, "
    "скопированная посимвольно, без пересказа и без правок. Если точной цитаты нет, "
    "элемент не возвращай.\n\n"
)

#: граница между общим блоком документа и задачей конкретного вызова
TASK_HEADER = "=== ЗАДАЧА ===\n"


def author_line(author_name: str | None) -> str:
    """Who wrote the text: the perspective every relation is read from."""
    if not author_name:
        return ""
    return (
        f"Автор этого текста: {author_name}. Все родственные и социальные связи "
        "указывай ОТ ЛИЦА АВТОРА: если автор пишет «моя мама», это мама автора, "
        "а не читателя.\n\n"
    )


def document_block(text: str, context: str, author_name: str | None) -> str:
    """Общий префикс всех вызовов: правила, автор, контекст и текст страницы.

    Он побайтно одинаков во всех вызовах прогона — на этом держится KV-кэш
    Ollama. Не вставляйте сюда ничего, что зависит от вызова.
    """
    return (
        CONTEXT_RULE
        + SOURCE_RULE
        + author_line(author_name)
        + f"{context}Основной текст:\n{text}\n\n"
        + TASK_HEADER
    )


def entities_prompt(text: str, context: str, author_name: str | None = None) -> str:
    return (
        document_block(text, context, author_name)
        + "Ты выделяешь ИМЕНОВАННЫЕ сущности из основного текста личного дневника "
        "(рукописного или обычного).\n\n"
        "Сущность — это конкретный человек, организация, место, событие или предмет, "
        "у которого есть имя или название: «Баба Галя», «Сочи», «завод Красный Октябрь», "
        "«Ивановы», «деревня Игумновцы».\n\n"
        "Для каждой сущности укажи:\n"
        "- name: каноническое имя в именительном падеже и в единственном числе, "
        "если речь об одном («Баба Галя», «Сочи», «завод»)\n"
        "- entity_type: person (человек) | organization (семья, завод, учреждение) | "
        "location (город, деревня, улица, дом) | event | object | date | other\n"
        "- description: одна короткая фраза, чем сущность является по тексту (или пустая строка)\n"
        "- aliases: другие написания того же (например «Галя», «бабушка»); пустой массив, если нет\n\n"
        "ЧЕГО НЕ ДЕЛАТЬ — это важнее всего остального:\n"
        "- НЕ выделяй обычные слова: «мох», «дом», «дела», «собрания», «звание», «ранее», «мы», "
        "«мои родители», «маленькую половину». Это не сущности, а слова текста.\n"
        "- НЕ выделяй указательные обороты и родственные описания без имени "
        "(«мой отец», «моя мама», «наши соседи»). Связь с таким человеком выражается "
        "в отдельном вызове (relation_type), а не сущностью.\n"
        "- НЕ выделяй время и числа сами по себе: «лет 7», «20 августа 1910 года» — это "
        "time_text факта, а не сущность.\n"
        "- НЕ выделяй себя словами «я», «мы», «мой».\n"
        "- Если у человека нет имени в тексте — не выделяй его вовсе.\n\n"
        "Пустой массив — правильный ответ для текста без имён. Лучше вернуть меньше "
        "сущностей, чем добавить в базу слово, которое именем не является."
    )


def mentions_prompt(
    text: str, context: str, entity_names: list[str], author_name: str | None = None
) -> str:
    names = ", ".join(entity_names) if entity_names else "—"
    return (
        document_block(text, context, author_name)
        + "Ты находишь в основном тексте упоминания уже известных сущностей.\n\n"
        f"Сущности: {names}\n\n"
        "Для каждого упоминания укажи:\n"
        "- entity_name: имя сущности из списка\n"
        "- mention_text: ТОЧНАЯ подстрока текста, как она написана в тексте\n"
        "- sentence_text: предложение, в котором стоит упоминание\n"
        "- confidence: уверенность 0.0-1.0"
    )


#: цели — только в дневнике: в чужих рукописях намерения почти не встречаются,
#: а лишние поля удлиняют каждый ответ модели
GOALS_RULE = (
    "Цель — тоже факт. Если факт — осознанное намерение чего-то достичь или "
    "продвижение к нему (выучить язык, похудеть, сменить работу, накопить деньги), "
    "заполни:\n"
    "- goal_title: краткое название цели (2-6 слов, именительный падеж); иначе пустая строка\n"
    "- goal_status: started (впервые поставил) | progress (есть продвижение) | "
    "completed (достиг) | abandoned (отказался) | mentioned (просто упомянул); "
    "none — факт не о цели\n"
    "Мимолётные желания и бытовые дела (помыть посуду) целями не считаются.\n\n"
)


def facts_prompt(
    text: str,
    context: str,
    author_name: str | None,
    graph: str = "",
    with_goals: bool = False,
) -> str:
    return (
        document_block(text, context, author_name)
        + "Ты извлекаешь факты из основного текста личного дневника.\n"
        "Факт — это утверждение о мире: кто что сделал, кем кому приходится, что где "
        "находится, что случилось.\n\n"
        + FACTS_SUBJECT_RULE
        + graph
        + "Для каждого факта укажи:\n"
        "- predicate: сказуемое одним-двумя словами в нижнем регистре («купил», «работает», «мама»)\n"
        f"- predicate_kind: ключ закрытого списка — {PREDICATE_VOCABULARY}. "
        "«Родился», «год рождения», «born» — это birth; «умер», «скончалась» — death. "
        "Не подходит ничего — other.\n"
        "- object_name: имя сущности-объекта, если это человек, место или организация (иначе пустая строка)\n"
        "- object_text: объект словами, если это не сущность («новую машину»)\n"
        "- source_text: точная цитата\n"
        "- time_text: когда это было, как написано в тексте (или пустая строка)\n"
        "- time_iso: та же дата в виде ГГГГ, ГГГГ-ММ или ГГГГ-ММ-ДД, если её можно "
        "определить («20 августа 1910 года» -> 1910-08-20); иначе пустая строка. "
        "Точность не выдумывай: «в 1910 году» — это 1910, а не 1910-01-01.\n"
        "- importance: насколько это событие в жизни субъекта, 0.0-1.0. Событие — "
        "переезд, свадьба, рождение, смерть, болезнь, смена работы, учёба, служба, "
        "путешествие, крупная покупка, важная встреча: заметное ~0.5, жизненно важное "
        "> 0.8. Обычное утверждение, черта характера, рутина (поел, лёг спать) — 0.\n"
        "- recurrence: если текст прямо говорит, что действие регулярное («каждое утро», "
        "«по выходным», «всегда», «часто»), — эти слова, как в тексте; иначе пустая "
        "строка. Из одного упоминания регулярность не выводи.\n"
        "- confidence: уверенность 0.0-1.0\n\n"
        + (GOALS_RULE if with_goals else "")
        + "Не выдумывай факты и не пересказывай общие рассуждения. Если фактов нет, верни пустой массив."
    )


def relations_prompt(
    text: str,
    context: str,
    author_name: str | None,
    known_people: list[str],
    graph: str = "",
) -> str:
    known = ", ".join(known_people) if known_people else "—"
    return (
        document_block(text, context, author_name)
        + "Ты определяешь круг общения автора текста: родственные, дружеские, "
        "рабочие и социальные связи.\n\n"
        f"Уже известные люди: {known}\n\n"
        + graph
        + "Для каждого человека, чью связь с АВТОРОМ можно определить, укажи:\n"
        "- person_name: имя человека, как в тексте (именительный падеж)\n"
        f"- relation_kind: кем этот человек приходится АВТОРУ — ключ закрытого "
        f"списка: {RELATION_VOCABULARY}. «Батя», «папаня», «my old man» — это father, "
        "«мама» — mother. Если ничего не подходит — other.\n"
        "- relation_label: свободная подпись, ТОЛЬКО когда relation_kind=other "
        "(в остальных случаях пустая строка)\n"
        "- via_marriage: true, если связь идёт через брак (тесть, зять, шурин, "
        "сват, свояк, отчим). Для кровной родни false. Знак и число поколений "
        "не указывай — это считает код\n"
        "- generations: только для прадеда и дальше — сколько поколений (3..6). "
        "В остальных случаях 0\n"
        "- source_text: точная цитата\n"
        "- confidence: уверенность 0.0-1.0\n\n"
        "Включай человека только если связь явно следует из текста И у человека есть имя \n"
        "или однозначное обращение («Баба Галя», «дядя Коля», «тренер Пётр»). Родственное \n"
        "описание без имени («мой отец», «соседи») здесь не годится: роль такого человека \n"
        "относительно автора выражает вызов фактов (subject_kind=relative), а не связь.\n"
        "Не выводи связь через цепочку: если сказано «мама Маши», это связь Маши, а не автора.\n"
        "Если автор пишет о себе или связи не определяются, верни пустой массив."
    )


def resolution_prompt(kind_label: str, candidates: list[str], existing: list[tuple[int, str]]) -> str:
    existing_lines = "\n".join(f"- id={record_id}: {title}" for record_id, title in existing)
    candidate_lines = "\n".join(f"- index={index}: {title}" for index, title in enumerate(candidates))
    return (
        f"Ты сопоставляешь новые упоминания ({kind_label}) с уже существующими записями "
        "пользователя.\n"
        "Одно и то же может называться по-разному («английский» и «выучить английский язык», "
        "«Маша» и «Мария»).\n\n"
        f"Существующие записи:\n{existing_lines}\n\n"
        f"Новые упоминания:\n{candidate_lines}\n\n"
        "Для каждого нового упоминания верни matched_id — id существующей записи, "
        "если это ровно то же самое, иначе -1.\n"
        "Сопоставляй только при уверенном совпадении по смыслу: сомневаешься — верни -1.\n"
        "Разные люди с похожими именами, разные годы и разные формулировки одной темы "
        "считаются разными записями, если в них есть различающая деталь."
    )
