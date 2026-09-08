
from pathlib import Path
import asyncio
import re
from typing import AsyncGenerator, Mapping, Sequence

from llama_cpp import Llama

from ..config import settings
from .system_prompt import common_part, critic, supportive, philosopher


class LlamaService:
    def __init__(self):
        model_path = Path(settings.llama_model_path)
        if not model_path.is_absolute():
            model_path = Path(__file__).resolve().parents[2] / model_path

        if not model_path.exists():
            raise FileNotFoundError(
                f"GGUF model not found: {model_path}. Set llama_model_path in .env"
            )

        self.llm = Llama(
            model_path=str(model_path),
            n_ctx=settings.llama_n_ctx,
            n_threads=settings.llama_n_threads,
            n_gpu_layers=-1,  # Все слои на GPU (если доступна CUDA/Metal)
        )
        
        # Прогрев модели при инициализации
        self._warmup_model()

    def _warmup_model(self):
        """Прогрев модели тестовым запросом для избежания задержек при первом использовании."""
        try:
            self.llm(
                "Hi",
                max_tokens=1024,
                temperature=0.2,
                top_p=0.7,
                top_k=30,
            )
            print("✓ LlamaService модель прогрета")
        except Exception as e:
            print(f"⚠ Прогрев модели не удался: {e}")

    def generate(self, prompt: str):
        result = self.llm(prompt, max_tokens=512)
        return result["choices"][0]["text"].strip()

    def generate_diary_title(self, content: str) -> str:
        # Fallback keeps endpoint working even if model output is noisy.
        fallback_title = self._fallback_title(content)
        prompt = (
            "/no_think Create a short diary page title in Russian using 2-5 words. "
            "Return title only, without quotes, punctuation, or explanation.\n\n"
            f"Diary page:\n{content}\n\nTitle:"
        )
        try:
            result = self.llm(
                prompt,
                max_tokens=32,  # Увеличено для лучшего результата
                temperature=0.5,  # Повышено для лучшей креативности на русском
                # Удалён stop=["\n"] - улучшает результаты на русском
            )
            raw_title = result["choices"][0]["text"].strip()
            
            if not raw_title:
                return fallback_title

            # Удаляем кавычки и лишние символы
            cleaned = " ".join(raw_title.replace('"', "").replace(".", "").split())
            
            # Берём первые 5 слов
            words = cleaned.split()[:5]
            if not words or not any(c.isalpha() for c in cleaned):
                return fallback_title
            
            title = " ".join(words)
            return title
        except Exception as e:
            print(f"[LLM Title] Error: {e}, using fallback")
            return fallback_title

    @staticmethod
    def _strip_thinking(text: str) -> str:
        """Удаляет блок <think>...</think> из ответа thinking-моделей (Qwen3 и др.)."""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    @staticmethod
    def _fallback_title(content: str) -> str:
        compact = " ".join((content or "").split())
        if not compact:
            return "Новая запись"
        words = compact.split()[:5]
        return " ".join(words)
    
    @staticmethod
    def _format_history(history: Sequence[Mapping[str, str]] | None) -> str:
        """Форматирует историю чата для промпта."""
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
    
    @staticmethod
    def _build_prompt(
        message: str,
        tone: str,
        history: Sequence[Mapping[str, str]] | None,
    ) -> str:
        """Строит полный промпт для модели."""
        history_block = LlamaService._format_history(history)
        
        if tone == "critic":
            tone_text = critic
        elif tone == "supportive":
            tone_text = supportive
        elif tone == "philosopher":
            tone_text = philosopher
        else:
            tone_text = tone
        
        prompt_template = (
            common_part + "\n" +
            f"{tone_text}\n"
            f"История последних диалогов (от старых к новым):\n{history_block}\n\n"
            f"Пользователь сейчас пишет:\n\"{message}\"\n\n"
        )
        return prompt_template
    
    async def generate_reply(
        self,
        message: str,
        tone: str,
        history: Sequence[Mapping[str, str]] | None = None,
    ) -> str:
        """Генерирует ответ (неблокирующий вызов)."""
        # prompt = self._build_prompt(message, tone, history)
        prompt = message  # TODO: убрать, когда промпт будет отлажен
        # Запускаем блокирующий вызов в отдельном потоке
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self.llm(
                prompt,
                max_tokens=2048,
                temperature=0.2,
                top_p=0.7,
                top_k=30,
            )
        )
        return self._strip_thinking(result["choices"][0]["text"])
    
    async def stream_reply(
        self,
        message: str,
        tone: str,
        history: Sequence[Mapping[str, str]] | None = None,
    ) -> AsyncGenerator[str, None]:
        """Генерирует ответ потоком (асинхронный генератор)."""
        # prompt = self._build_prompt(message, tone, history)
        prompt = message  # TODO: убрать, когда промпт будет отлажен
        loop = asyncio.get_event_loop()
        
        # Используем run_in_executor для вызова блокирующей llama функции
        result = await loop.run_in_executor(
            None,
            lambda: self.llm(
                prompt,
                max_tokens=2048,
                temperature=0.2,
                top_p=0.7,
                top_k=30,
                stream=False,
            )
        )
        
        text = self._strip_thinking(result["choices"][0]["text"])
        # Разбиваем на куски для имитации потока
        for word in text.split():
            yield word + " "
            await asyncio.sleep(0.01)  # Имитируем потоковый вывод
    
# TODO: 
# 🔹 очередь запросов (как в OpenAI API)
# 🔹 батчинг
# 🔹 кеширование ответов
# 🔹 RAG (подключить базу знаний)
# 🔹 сравнение с Ollama / vLLM