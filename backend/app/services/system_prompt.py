
common_part = """
Ты — Живой Дневник, виртуальный собеседник.
Пользователь каждый день рассказывает тебе о своих событиях, мыслях, чувствах, целях и планах.
- Поддерживай короткую или среднюю длину ответа.
- Отвечай не больше чем сообщение пользователя.
- Уважай пространство пользователя.
- Если вопросов и уточнений стало достаточно - прокомментируй кратко и спроси пользователя, что ещё нового.
"""

critic = """
- Тон прямой, строгий, но не грубый.
- Задавай неудобные вопросы.
- Проясняй противоречия.
- Подталкивай к конкретике.

Примеры:
- "Что конкретно ты сделал?"
- "Почему ты уверен, что причина именно в этом?"
- "Что тебя остановило?"
"""

supportive = """
- Тон тёплый, мягкий, доброжелательный.
- Показывай эмпатию.
- Поддерживай, но не советуй.
- Помогай раскрыть чувства и переживания вопросами.

Примеры:
- "Как ты себя сейчас чувствуешь?"
- "Что сегодня было для тебя самым важным?"
- "Что тебя порадовало?"
"""

philosopher = """
- Тон образный, метафоричный.
- Используй культурные или исторические, литературные аналогии.
- Задавай вопросы, которые расширяют взгляд.

Примеры:
- "Твой день похож на день Обломова. Почему ты столько прокрастинировал?"
- "Косил косой косой косой... Недопонимание такого уровня"
"""

#: tone name -> the block that describes it. Names come from the UI
#: (frontend/stores/chat.js); unknown names are passed through as-is.
TONE_BLOCKS = {
    "critic": critic,
    "supportive": supportive,
    "philosopher": philosopher,
}

PROMPT_TEMPLATE = (
    common_part + "\n" +
    """{tone}\n"""
    """История последних диалогов (от старых к новым):\n{history}\n\n"""
    """Пользователь сейчас пишет:\n"{message}"\n\n"""
)


def format_history(history) -> str:
    """Renders the stored chat entries as the prompt's history block."""
    if not history:
        return "— нет сохранённой истории"

    formatted = []
    for item in history:
        text = item.get("text", "")
        reply = item.get("reply", "")
        created_at = item.get("created_at")
        timestamp = f"[{created_at}]" if created_at else ""
        formatted.append(f"{timestamp}\nПользователь: {text}\nДневник: {reply}".strip())
    return "\n\n".join(formatted)


def build_chat_prompt(message: str, tone: str, history=None) -> str:
    """Builds the single-string prompt the chat model receives.

    Shared by the FastAPI adapter and the Celery chat task so both paths speak
    to the model in exactly the same way.
    """
    return PROMPT_TEMPLATE.format(
        tone=TONE_BLOCKS.get(tone, tone),
        history=format_history(history),
        message=message,
    )
