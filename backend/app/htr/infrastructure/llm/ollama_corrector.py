"""LLM text correction through the Ollama instance used by chat and knowledge.

The corrector is deliberately best-effort: if Ollama is down, the model is
still loading or the answer looks unusable, the raw HTR prediction is kept and
the failure is logged. Recognition must never fail because of the LLM step.
"""
from __future__ import annotations

import logging
from typing import Any

from ...domain.entities import CorrectionContext
from ...domain.interfaces import TextCorrector
from .prompt import build_correction_prompt, sanitize_correction

logger = logging.getLogger(__name__)

# Conservative generation settings: we want a mechanical fix, not creativity.
GENERATION_OPTIONS = {"temperature": 0.0, "top_p": 0.9}


class OllamaLineCorrector(TextCorrector):
    def __init__(
        self,
        model: str,
        host: str,
        timeout: float = 90.0,
        keep_alive: str = "10m",
        num_predict: int = 256,
    ):
        self.model = model
        self.host = host
        self.timeout = timeout
        self.keep_alive = keep_alive
        self.num_predict = num_predict
        self._client: Any = None

    @property
    def client(self):
        if self._client is None:
            import ollama  # imported lazily: the HTR module works without it

            self._client = ollama.Client(host=self.host, timeout=self.timeout)
        return self._client

    # ------------------------------------------------------------------

    def correct_line(
        self,
        text: str,
        context_lines: list[str],
        context: CorrectionContext,
    ) -> str | None:
        if not text.strip():
            return None
        prompt = build_correction_prompt(text, context_lines, context)
        try:
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                stream=False,
                keep_alive=self.keep_alive,
                options={**GENERATION_OPTIONS, "num_predict": self.num_predict},
            )
        except Exception as exc:  # transport, timeout, model not loaded, ...
            logger.warning(
                "HTR correction: Ollama request failed (%s: %s); keeping the raw prediction",
                type(exc).__name__, exc,
            )
            return None

        answer = self._answer_text(response)
        cleaned = sanitize_correction(answer, text)
        if cleaned is None and answer:
            logger.warning("HTR correction: unusable answer %r", str(answer)[:160])
        return cleaned

    @staticmethod
    def _answer_text(response: Any) -> str | None:
        if response is None:
            return None
        if isinstance(response, dict):
            return response.get("response")
        return getattr(response, "response", None)
