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

PROMPT = (
    "Выдели цели из текста и верни только валидный JSON вида "
    '{"goals": [{"title": "..."}]} без пояснений.\n\n'
    "Текст: Хочу выучить английский к лету. Сегодня прошёл два урока."
)


async def main():
    c = ollama.AsyncClient(host=settings.ollama_url)

    print("=== chat API, format=SCHEMA ===")
    try:
        r = await c.chat(
            model="gpt-oss:20b",
            messages=[{"role": "user", "content": PROMPT}],
            stream=False,
            format=SCHEMA,
            options={"temperature": 0.1, "num_predict": 2048},
        )
        print("done_reason:", r.get("done_reason"))
        print("eval_count:", r.get("eval_count"))
        print("content:", repr(r["message"].get("content"))[:400])
        print("thinking:", repr(r["message"].get("thinking"))[:200])
    except Exception as e:
        print("chat+format failed:", e)

    print("\n=== generate, no format (JSON via prompt) ===")
    r2 = await c.generate(
        model="gpt-oss:20b",
        prompt=PROMPT,
        stream=False,
        options={"temperature": 0.1, "num_predict": 2048},
    )
    print("done_reason:", r2.get("done_reason"))
    print("eval_count:", r2.get("eval_count"))
    print("response:", repr(r2.get("response"))[:400])
    print("thinking:", repr(r2.get("thinking"))[:200])


asyncio.run(main())
