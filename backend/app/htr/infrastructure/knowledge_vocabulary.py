"""Domain vocabulary of a user, taken from the knowledge base.

The LLM corrector gets proper nouns and terms a generic language model cannot
know: people, places and topics already extracted from the user's diary pages.
"""
from __future__ import annotations

from sqlalchemy.orm import Session

from ...models import Entity, Event, Goal, Habit


class SqlAlchemyKnowledgeVocabulary:
    def __init__(self, db: Session, limit_per_table: int = 200):
        self.db = db
        self.limit_per_table = limit_per_table

    def vocabulary(self, user_id: int) -> list[str]:
        terms: list[str] = []

        entities = (
            self.db.query(Entity.canonical_name)
            .filter(Entity.user_id == user_id)
            .order_by(Entity.last_seen_at.desc().nullslast())
            .limit(self.limit_per_table)
            .all()
        )
        terms.extend(row[0] for row in entities if row[0])

        # titles are phrases, but they still carry names and toponyms
        for model in (Event, Goal, Habit):
            rows = (
                self.db.query(model.title)
                .filter(model.user_id == user_id)
                .limit(self.limit_per_table)
                .all()
            )
            terms.extend(row[0] for row in rows if row[0])

        return terms
