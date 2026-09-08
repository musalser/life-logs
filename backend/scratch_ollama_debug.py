import asyncio

import ollama

from app.config import settings

SCHEMA = {
    "type": "object",
    "properties": {
        "goals": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"title": {"type": "string"}},
                "required": ["title"],
            },
        }
    },
    "required": ["goals"],
}

PROMPT = "Выдели цели из текста: Хочу выучить английский к лету. Сегодня прошёл два урока."


async def main():
    c = ollama.AsyncClient(host=settings.ollama_url)

    print("=== format=SCHEMA, num_predict=2048 ===")
    r = await c.generate(
        model="gpt-oss:20b",
        prompt=PROMPT,
        stream=False,
        format=SCHEMA,
        options={"temperature": 0.1, "num_predict": 2048},
    )
    print("done_reason:", r.get("done_reason"))
    print("eval_count:", r.get("eval_count"))
    print("response:", repr(r.get("response"))[:400])
    print("thinking:", repr(r.get("thinking"))[:400])

    print("\n=== format=SCHEMA, think=low ===")
    try:
        r2 = await c.generate(
            model="gpt-oss:20b",
            prompt=PROMPT,
            stream=False,
            format=SCHEMA,
            think="low",
            options={"temperature": 0.1, "num_predict": 2048},
        )
        print("done_reason:", r2.get("done_reason"))
        print("eval_count:", r2.get("eval_count"))
        print("response:", repr(r2.get("response"))[:400])
        print("thinking:", repr(r2.get("thinking"))[:400])
    except Exception as e:
        print("think=low failed:", e)


asyncio.run(main())
