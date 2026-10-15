"""Поиск по архиву для чата (RAG, ТЗ 8.4).

Чанки индексировались промптом ``Document``, запрос ищется промптом
``SearchQuery`` — так устроена модель: у вопроса и у фрагмента текста разные
инструкции (ТЗ 4.5). Вектора лежат в pgvector, поэтому близость считает
Postgres, а не Python.

Если эмбеддинги недоступны или в архиве нет ни одного вектора, поиск возвращает
пустой список: чат обязан ответить, а не упасть (проверка «есть ли вообще
вектора» стоит до вызова модели).
"""
from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import text as sql_text
from sqlalchemy.orm import Session

from ..config import settings
from ..models import Chunk, KnowledgeSource
from .embedding_service import EmbeddingError, EmbeddingService, get_embedding_service

logger = logging.getLogger(__name__)

#: запрос на поиск: чанк, его источник и косинусное расстояние
_SEARCH_SQL = """
SELECT c.id, c.text, c.source_id, c.start_offset, c.end_offset,
       s.source_type, s.ref_id, s.title, s.author_entity_id,
       1 - (c.embedding <=> CAST(:vector AS vector)) AS similarity
FROM chunks c
JOIN knowledge_sources s ON s.id = c.source_id
WHERE c.user_id = :user_id
  AND c.embedding IS NOT NULL
  AND (
    :subject_entity_id IS NULL
    OR s.author_entity_id = :subject_entity_id
    OR s.author_entity_id IN (
        SELECT self_entity_id FROM users
        WHERE id = :user_id AND self_entity_id IS NOT NULL
    )
  )
ORDER BY c.embedding <=> CAST(:vector AS vector)
LIMIT :limit
"""


class RagService:
    def __init__(self, db: Session, embeddings: EmbeddingService | None = None) -> None:
        self.db = db
        self.embeddings = embeddings or get_embedding_service()

    def has_vectors(self, user_id: int) -> bool:
        """Cheap check before embedding a question nobody can answer."""
        return (
            self.db.query(Chunk.id)
            .filter(Chunk.user_id == user_id, Chunk.embedding.isnot(None))
            .first()
            is not None
        )

    async def search(
        self,
        user_id: int,
        query: str,
        *,
        subject_entity_id: int | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Top-k passages of the archive that answer a question."""
        question = (query or "").strip()
        if not question:
            return []
        if not self.has_vectors(user_id):
            logger.info("RAG: user %s has no embedded chunks yet", user_id)
            return []

        try:
            vector = await self.embeddings.embed_text(question, prompt_name="SearchQuery")
        except EmbeddingError as exc:
            logger.warning("RAG search skipped: %s", exc)
            return []

        rows = self.db.execute(
            sql_text(_SEARCH_SQL),
            {
                "user_id": user_id,
                "subject_entity_id": subject_entity_id,
                "vector": "[" + ",".join(f"{value:.6f}" for value in vector) + "]",
                "limit": limit or settings.rag_top_k,
            },
        ).all()

        results: list[dict[str, Any]] = []
        for row in rows:
            results.append(
                {
                    "chunk_id": row[0],
                    "text": row[1],
                    "source_id": row[2],
                    "start_offset": row[3],
                    "end_offset": row[4],
                    "source_type": row[5],
                    "ref_id": row[6],
                    "title": row[7] or (f"Стр. #{row[6]}" if row[5] == "htr" else "Запись дневника"),
                    "similarity": float(row[9]) if row[9] is not None else None,
                }
            )
        logger.info("RAG: %s chunks for a question of %s chars", len(results), len(question))
        return results

    def search_sync(self, user_id: int, query: str, **kwargs) -> list[dict[str, Any]]:
        import asyncio

        return asyncio.run(self.search(user_id, query, **kwargs))

    @staticmethod
    def context_block(chunks: list[dict[str, Any]]) -> str:
        """Passages as the prompt sees them: numbered, with their source named."""
        if not chunks:
            return ""
        lines = [
            "Контекст из архива (отвечай по нему и ссылайся на источники; "
            "если ответа в нём нет — скажи прямо):"
        ]
        for index, chunk in enumerate(chunks, start=1):
            label = chunk.get("title") or "источник"
            lines.append(f"[{index}] {label}: {chunk.get('text', '').strip()}")
        return "\n".join(lines)
