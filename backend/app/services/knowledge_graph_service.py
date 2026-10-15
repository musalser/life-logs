"""Граф связей: рёбра от перспективы объекта и пути до него (ТЗ, раздел 9).

Связь читается от лица автора текста, поэтому «мама» из дневника Бабы Гали —
её мама. Когда пользователь смотрит знания от своего объекта, родня из чужого
дневника не имеет прямого ребра, но до неё есть путь; UI показывает его
цепочкой («мама Бабы Гали»). Типы при этом не пересчитываются: «прабабушка» из
двух рёбер не выводится — так решено в ТЗ.
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Iterable

from sqlalchemy.orm import Session

from ..models import Entity, EntityRelation

logger = logging.getLogger(__name__)

#: насколько длинную цепочку ещё показываем
MAX_PATH_DEPTH = 3

#: не разворачиваемся в бесконечный поиск на больших архивах
MAX_EDGES = 5000


class KnowledgeGraphService:
    def __init__(self, db: Session, user_id: int) -> None:
        self.db = db
        self.user_id = user_id

    # ------------------------------------------------------------------
    # Рёбра
    # ------------------------------------------------------------------

    def edges(self, from_entity_id: int) -> list[EntityRelation]:
        """Прямые связи объекта: кем приходятся ему люди из его текстов."""
        return (
            self.db.query(EntityRelation)
            .filter(
                EntityRelation.user_id == self.user_id,
                EntityRelation.from_entity_id == from_entity_id,
            )
            .order_by(EntityRelation.relation_type, EntityRelation.id)
            .all()
        )

    def all_edges(self) -> list[EntityRelation]:
        return (
            self.db.query(EntityRelation)
            .filter(EntityRelation.user_id == self.user_id)
            .limit(MAX_EDGES)
            .all()
        )

    def people_mentioned(self, subject_entity_id: int, limit: int = 200) -> list[Entity]:
        """Персоны, на которые есть рёбра у объекта или путь до них."""
        reachable = self.reachable_map(subject_entity_id, limit=limit)
        if not reachable:
            return []
        return (
            self.db.query(Entity)
            .filter(Entity.id.in_(list(reachable)))
            .order_by(Entity.canonical_name)
            .all()
        )

    # ------------------------------------------------------------------
    # Пути
    # ------------------------------------------------------------------

    def paths(
        self, subject_entity_id: int, max_depth: int = MAX_PATH_DEPTH
    ) -> dict[int, list[dict]]:
        """BFS от объекта: кому он приходится кем через цепочку рёбер.

        Возвращает ``{entity_id: [{"entity_id", "name", "relation_type"}, ...]}``,
        где цепочка описывает связь выбранного объекта с персоной. Прямые рёбра
        (глубина 1) сюда не попадают: они и так известны вызывающему.
        """
        edges = self.all_edges()
        adjacency: dict[int, list[EntityRelation]] = {}
        for edge in edges:
            adjacency.setdefault(edge.from_entity_id, []).append(edge)

        names = self._names({edge.to_entity_id for edge in edges} | {subject_entity_id})
        paths: dict[int, list[dict]] = {}
        visited = {subject_entity_id}
        queue: deque[tuple[int, list[dict]]] = deque([(subject_entity_id, [])])
        while queue:
            current, chain = queue.popleft()
            if len(chain) >= max_depth:
                continue
            for edge in adjacency.get(current, []):
                if edge.to_entity_id in visited:
                    continue
                visited.add(edge.to_entity_id)
                step = {
                    "entity_id": edge.to_entity_id,
                    "name": names.get(edge.to_entity_id, "?"),
                    "relation_type": edge.relation_type,
                }
                next_chain = chain + [step]
                if len(next_chain) > 1:  # прямая связь — не путь
                    paths[edge.to_entity_id] = next_chain
                queue.append((edge.to_entity_id, next_chain))
        return paths

    def reachable_map(self, subject_entity_id: int, limit: int = 200) -> dict[int, list[dict]]:
        """Прямые связи + пути ≤ 3 рёбер: одна карта «кто и как связан»."""
        result: dict[int, list[dict]] = {
            edge.to_entity_id: [
                {
                    "entity_id": edge.to_entity_id,
                    "name": self._names({edge.to_entity_id}).get(edge.to_entity_id, "?"),
                    "relation_type": edge.relation_type,
                }
            ]
            for edge in self.edges(subject_entity_id)
        }
        paths = self.paths(subject_entity_id)
        for entity_id, chain in paths.items():
            result.setdefault(entity_id, chain)
        # ограничение на размер ответа: сначала прямые связи, потом ближние пути
        return dict(list(result.items())[:limit])

    def path_label(self, chain: Iterable[dict]) -> str:
        """Человеческая подпись цепочки: «мама Бабы Гали»."""
        steps = list(chain)
        if not steps:
            return ""
        label = steps[-1].get("name", "?")
        for step in reversed(steps[:-1]):
            label = f"{label} → {step.get('name', '?')}"
        return label

    # ------------------------------------------------------------------
    # Внутреннее
    # ------------------------------------------------------------------

    def _names(self, entity_ids: set[int]) -> dict[int, str]:
        if not entity_ids:
            return {}
        rows = (
            self.db.query(Entity.id, Entity.canonical_name)
            .filter(Entity.id.in_(list(entity_ids)))
            .all()
        )
        return {row[0]: row[1] for row in rows}
