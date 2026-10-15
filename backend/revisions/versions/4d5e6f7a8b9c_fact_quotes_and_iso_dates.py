"""несколько цитат на факт и нормализованная дата

* ``fact_sources`` — по строке на каждое подтверждение факта (источник, цитата,
  офсеты, уверенность). Перезаписывать единственную цитату при дедупе нельзя:
  вместе с ней теряется прежнее подтверждение. Основной факт оставляет первую
  цитату у себя, все подтверждения живут здесь;
* ``facts.time_iso`` — та же дата, приведённая моделью к ГГГГ[-ММ[-ДД]]: «1910»
  и «20 августа 1910» сравниваются как одна дата, а не как две.

Данные: у уже сохранённых фактов создаётся по одной строке ``fact_sources`` из
их текущей цитаты — иначе панель знаний страницы перестала бы их видеть.

Revision ID: 4d5e6f7a8b9c
Revises: 3c4d5e6f7a8b
Create Date: 2026-10-15 16:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import text


# revision identifiers, used by Alembic.
revision: str = "4d5e6f7a8b9c"
down_revision: Union[str, Sequence[str], None] = "3c4d5e6f7a8b"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    bind = op.get_bind()

    op.add_column("facts", sa.Column("time_iso", sa.String(length=32), nullable=True))
    op.create_index(op.f("ix_facts_time_iso"), "facts", ["time_iso"], unique=False)

    op.create_table(
        "fact_sources",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("fact_id", sa.Integer(), nullable=False),
        sa.Column("source_id", sa.Integer(), nullable=False),
        sa.Column("source_text", sa.Text(), nullable=False),
        sa.Column("start_offset", sa.Integer(), nullable=True),
        sa.Column("end_offset", sa.Integer(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True),
        sa.ForeignKeyConstraint(["fact_id"], ["facts.id"], name="fk_fact_sources_fact", ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["source_id"], ["knowledge_sources.id"], name="fk_fact_sources_source", ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("fact_id", "source_id", name="uq_fact_source_fact_source"),
    )
    op.create_index(op.f("ix_fact_sources_id"), "fact_sources", ["id"], unique=False)
    op.create_index(op.f("ix_fact_sources_fact_id"), "fact_sources", ["fact_id"], unique=False)
    op.create_index(op.f("ix_fact_sources_source_id"), "fact_sources", ["source_id"], unique=False)

    created = bind.execute(
        text(
            "insert into fact_sources (fact_id, source_id, source_text, start_offset, end_offset, confidence) "
            "select id, source_id, source_text, start_offset, end_offset, confidence from facts "
            "where source_id is not null"
        )
    ).rowcount
    print(f"[fact quotes migration] fact_sources created for {created} facts")


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f("ix_fact_sources_source_id"), table_name="fact_sources")
    op.drop_index(op.f("ix_fact_sources_fact_id"), table_name="fact_sources")
    op.drop_index(op.f("ix_fact_sources_id"), table_name="fact_sources")
    op.drop_table("fact_sources")

    op.drop_index(op.f("ix_facts_time_iso"), table_name="facts")
    op.drop_column("facts", "time_iso")
