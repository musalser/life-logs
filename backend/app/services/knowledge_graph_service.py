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

from ..models import Entity, EntityRelation, KnowledgeState
from .kinship import canonicalize_relations, levels

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


    # ------------------------------------------------------------------
    # Граф целиком (раздел 9): уровни, конфликты, «вне древа»
    # ------------------------------------------------------------------

    def relation_graph(self) -> dict:
        """Весь архив связей сразу — без выбранного объекта.

        Корень выбирает клиент, а уровни считаются внутри компоненты с постоянным
        сдвигом: смена корня — это вычитание его уровня, а не новый запрос. Узлы
        без родства (ритуальные, неоднозначные, одинокие) получают ``level=None``
        и в «Древо» не попадают — интерфейс показывает их отдельным списком, а не
        ставит на нулевой уровень по умолчанию.
        """
        persons = (
            self.db.query(Entity)
            .filter(Entity.user_id == self.user_id, Entity.entity_type == "person")
            .order_by(Entity.canonical_name)
            .all()
        )
        edges = self.all_edges()
        by_id = {person.id: person for person in persons}
        for edge in edges:
            for entity in (edge.from_entity, edge.to_entity):
                if entity is not None and entity.id not in by_id:
                    by_id[entity.id] = entity
        anchors = {entity.id for entity in by_id.values() if entity.anchor_entity_id is not None}

        rows = [
            {
                "id": edge.id,
                "from": edge.from_entity_id,
                "to": edge.to_entity_id,
                "relation_kind": edge.relation_kind,
                "generations": edge.generations,
                "via_marriage": edge.via_marriage,
                "bond": edge.bond,
                "confidence": edge.confidence,
                "relation_type": edge.relation_type,
                "sources": [row.source_id for row in edge.confirmations] or (
                    [edge.source_id] if edge.source_id else []
                ),
            }
            for edge in edges
        ]
        result = levels(sorted(by_id), rows, anchors=anchors)

        state = (
            self.db.query(KnowledgeState)
            .filter(KnowledgeState.user_id == self.user_id)
            .first()
        )
        canonical = canonicalize_relations(rows)
        return {
            "version": state.version if state is not None else 0,
            "nodes": [
                {
                    "id": entity.id,
                    "name": entity.canonical_name,
                    "entity_type": entity.entity_type,
                    "is_anchor": entity.id in anchors,
                    "anchor_of": entity.anchor_entity_id,
                    "anchor_relation": entity.anchor_relation,
                    "level": result.levels.get(entity.id),
                    "component": result.components.get(entity.id),
                }
                for entity in by_id.values()
            ],
            "edges": [
                {
                    "id": edge.edge_id,
                    "from_id": edge.from_id,
                    "to_id": edge.to_id,
                    "relation_kind": edge.kind,
                    "relation_type": edge.relation_type,
                    "delta": edge.delta,
                    "via_marriage": edge.via_marriage,
                    "bond": edge.bond,
                    "canon_key": edge.canon_key,
                    "confidence": edge.confidence,
                    "sources": list(edge.sources),
                }
                for edge in canonical
            ],
            "conflicts": result.conflicts,
            "unplaced": result.unplaced,
            "dangling": result.dangling,
        }
