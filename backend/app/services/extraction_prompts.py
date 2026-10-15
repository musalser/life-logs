"""Промпты и JSON-схемы извлечения знаний (этап 4 ТЗ, вызовы A–E).

Промпты вынесены из сервиса: их читают чаще, чем код вокруг, а тесты проверяют
именно формулировки — «связь от лица автора текста» и «контекст соседних
страниц не извлекать» должны сломать тест, а не тихо испортить данные.

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

FACTS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
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
                    "confidence": {"type": "number"},
                },
                "required": [
                    "subject_kind",
                    "subject_name",
                    "subject_relation",
                    "subject_of",
                    "subject_id",
                    "predicate",
                    "predicate_kind",
                    "object_name",
                    "object_text",
                    "source_text",
                    "time_text",
                    "time_iso",
                    "confidence",
                ],
            },
        }
    },
    "required": ["facts"],
}

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
                    "source_text": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": [
                    "person_name",
                    "relation_kind",
                    "relation_label",
                    "source_text",
                    "confidence",
                ],
            },
        }
    },
    "required": ["relations"],
}

GOALS_SCHEMA: dict[str, Any] = {
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

EVENTS_SCHEMA: dict[str, Any] = {
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

HABITS_SCHEMA: dict[str, Any] = {
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
    "source_text — точная цитата ИЗ ОСНОВНОГО ТЕКСТА, скопированная посимвольно, "
    "без пересказа и без правок. Если точной цитаты нет, элемент не возвращай.\n\n"
)


def author_line(author_name: str | None) -> str:
    """Who wrote the text: the perspective every relation is read from."""
    if not author_name:
        return ""
    return (
        f"Автор этого текста: {author_name}. Все родственные и социальные связи "
        "указывай ОТ ЛИЦА АВТОРА: если автор пишет «моя мама», это мама автора, "
        "а не читателя.\n\n"
    )


def entities_prompt(text: str, context: str) -> str:
    return (
        CONTEXT_RULE
        + "Ты выделяешь ИМЕНОВАННЫЕ сущности из текста личного дневника "
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
        "time_text факта или события, а не сущность.\n"
        "- НЕ выделяй себя словами «я», «мы», «мой».\n"
        "- Если у человека нет имени в тексте — не выделяй его вовсе.\n\n"
        "Пустой массив — правильный ответ для текста без имён. Лучше вернуть меньше "
        "сущностей, чем добавить в базу слово, которое именем не является.\n\n"
        f"{context}Основной текст:\n{text}"
    )


def mentions_prompt(text: str, context: str, entity_names: list[str]) -> str:
    names = ", ".join(entity_names) if entity_names else "—"
    return (
        CONTEXT_RULE
        + SOURCE_RULE
        + "Ты находишь в тексте упоминания уже известных сущностей.\n\n"
        f"Сущности: {names}\n\n"
        "Для каждого упоминания укажи:\n"
        "- entity_name: имя сущности из списка\n"
        "- mention_text: ТОЧНАЯ подстрока текста, как она написана в тексте\n"
        "- sentence_text: предложение, в котором стоит упоминание\n"
        "- confidence: уверенность 0.0-1.0\n\n"
        f"{context}Основной текст:\n{text}"
    )


def facts_prompt(
    text: str, context: str, author_name: str | None, graph: str = ""
) -> str:
    return (
        CONTEXT_RULE
        + SOURCE_RULE
        + author_line(author_name)
        + "Ты извлекаешь факты из текста личного дневника.\n"
        "Факт — это утверждение о мире: кто что сделал, кем кому приходится, что где находится.\n\n"
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
        "- confidence: уверенность 0.0-1.0\n\n"
        "Не выдумывай факты и не пересказывай общие рассуждения. Если фактов нет, верни пустой массив.\n\n"
        f"{context}Основной текст:\n{text}"
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
        CONTEXT_RULE
        + SOURCE_RULE
        + author_line(author_name)
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
        "- source_text: точная цитата\n"
        "- confidence: уверенность 0.0-1.0\n\n"
        "Включай человека только если связь явно следует из текста И у человека есть имя \n"
        "или однозначное обращение («Баба Галя», «дядя Коля», «тренер Пётр»). Родственное \n"
        "описание без имени («мой отец», «соседи») здесь не годится: роль такого человека \n"
        "относительно автора выражает вызов фактов (subject_kind=relative), а не связь.\n"
        "Не выводи связь через цепочку: если сказано «мама Маши», это связь Маши, а не автора.\n"
        "Если автор пишет о себе или связи не определяются, верни пустой массив.\n\n"
        f"{context}Основной текст:\n{text}"
    )


def goals_prompt(text: str, context: str, author_name: str | None) -> str:
    return (
        CONTEXT_RULE
        + SOURCE_RULE
        + author_line(author_name)
        + "Ты извлекаешь личные цели автора текста.\n"
        "Цель — осознанное намерение чего-то достичь (выучить язык, похудеть, "
        "сменить работу, накопить деньги).\n\n"
        "Для каждой цели укажи:\n"
        "- title: краткое название цели (2-6 слов, именительный падеж)\n"
        "- status_update: started (впервые поставил) | progress (есть продвижение) | "
        "completed (достиг) | abandoned (отказался) | mentioned (просто упомянул)\n"
        "- note: краткое описание продвижения или контекста\n"
        "- source_text: точная цитата\n"
        "- confidence: уверенность 0.0-1.0\n\n"
        "Не выдумывай цели: мимолётные желания и бытовые дела (помыть посуду) целями не считаются.\n"
        "Если целей нет, верни пустой массив.\n\n"
        f"{context}Основной текст:\n{text}"
    )


def events_prompt(text: str, context: str, author_name: str | None) -> str:
    return (
        CONTEXT_RULE
        + SOURCE_RULE
        + author_line(author_name)
        + "Ты извлекаешь важные события из текста.\n"
        "Важное событие — то, что заметно влияет на жизнь автора: переезд, смена работы, "
        "экзамен, свадьба, рождение, болезнь, путешествие, крупная покупка, встреча.\n\n"
        "Для каждого события укажи:\n"
        "- title: краткое название (2-6 слов)\n"
        "- description: описание в 1-2 предложениях\n"
        "- time_text: когда произошло, как написано в тексте (или пустая строка)\n"
        "- importance: важность 0.0-1.0 (рутина < 0.3, заметное ~0.5, жизненно важное > 0.8)\n"
        "- source_text: точная цитата\n\n"
        "Рутинные действия (поел, лёг спать) событиями не считаются. "
        "Если событий нет, верни пустой массив.\n\n"
        f"{context}Основной текст:\n{text}"
    )


def habits_prompt(text: str, context: str, author_name: str | None) -> str:
    return (
        CONTEXT_RULE
        + SOURCE_RULE
        + author_line(author_name)
        + "Ты извлекаешь регулярные привычки и активности из текста.\n"
        "Привычка — повторяющееся действие: бег, зарядка, чтение, медитация, "
        "изучение языка, прогулки.\n\n"
        "Для каждой привычки, выполнение (или пропуск) которой упомянуто, укажи:\n"
        "- title: название привычки (1-4 слова, именительный падеж)\n"
        "- note: что именно сделано (или почему пропущено)\n"
        "- source_text: точная цитата\n\n"
        "Только регулярные действия, не разовые дела. Если привычек нет, верни пустой массив.\n\n"
        f"{context}Основной текст:\n{text}"
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
