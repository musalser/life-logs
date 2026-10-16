"""факты поглощают события: важность и регулярность у факта

* ``facts.importance`` — событие стало фактом с важностью: у него появляются
  субъект (а не всегда автор страницы), подтверждения из нескольких источников
  и дедупликация фактов;
* ``facts.recurrence`` — как текст говорит о регулярности; привычки выводит
  аналитика по накопленным фактам, а не отдельный вызов модели по одной странице;
* существующие строки ``events`` копируются в ``facts`` (с подтверждением в
  ``fact_sources``) и помечаются ``subject_source='event'``. Таблица ``events``
  не удаляется: откат удаляет перенесённые факты и возвращает схему как была.

Revision ID: 6f7a8b9c0d1e
Revises: 5e6f7a8b9c0d
Create Date: 2026-10-16 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import text


# revision identifiers, used by Alembic.
revision: str = "6f7a8b9c0d1e"
down_revision: Union[str, Sequence[str], None] = "5e6f7a8b9c0d"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

#: пометка перенесённых событий: по ней откат находит именно их
MIGRATED_MARK = "event"


def upgrade() -> None:
    op.add_column("facts", sa.Column("importance", sa.Float(), nullable=True))
    op.add_column("facts", sa.Column("recurrence", sa.String(length=64), nullable=True))
    op.create_index(op.f("ix_facts_importance"), "facts", ["importance"], unique=False)

    bind = op.get_bind()
    events = bind.execute(
        text(
            "SELECT id, user_id, source_id, subject_entity_id, title, description, time_text, "
            "importance, source_text, start_offset, end_offset FROM events ORDER BY id"
        )
    ).mappings().all()
    for event in events:
        quote = event["source_text"] or event["title"]
        fact_id = bind.execute(
            text(
                "INSERT INTO facts (user_id, source_id, subject_entity_id, subject_source, predicate, "
                "predicate_kind, object_text, source_text, start_offset, end_offset, time_text, "
                "importance) VALUES (:user_id, :source_id, :subject_id, :mark, :predicate, 'other', "
                ":object_text, :quote, :start, :end, :time_text, :importance) RETURNING id"
            ),
            {
                "user_id": event["user_id"],
                "source_id": event["source_id"],
                "subject_id": event["subject_entity_id"],
                "mark": MIGRATED_MARK,
                "predicate": (event["title"] or "событие")[:100],
                "object_text": event["description"],
                "quote": quote,
                "start": event["start_offset"],
                "end": event["end_offset"],
                "time_text": event["time_text"],
                # у старого события важность могла быть пустой: событием оно было
                # всё равно, поэтому ставим порог «заметное»
                "importance": event["importance"] if event["importance"] is not None else 0.5,
            },
        ).scalar_one()
        bind.execute(
            text(
                "INSERT INTO fact_sources (fact_id, source_id, source_text, start_offset, end_offset) "
                "VALUES (:fact_id, :source_id, :quote, :start, :end)"
            ),
            {
                "fact_id": fact_id,
                "source_id": event["source_id"],
                "quote": quote,
                "start": event["start_offset"],
                "end": event["end_offset"],
            },
        )


def downgrade() -> None:
    # fact_sources удаляются каскадом
    op.execute(text("DELETE FROM facts WHERE subject_source = :mark").bindparams(mark=MIGRATED_MARK))
    op.drop_index(op.f("ix_facts_importance"), table_name="facts")
    op.drop_column("facts", "recurrence")
    op.drop_column("facts", "importance")
