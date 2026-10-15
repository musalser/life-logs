"""функциональные предикаты: канонический ключ предиката и конфликты на проверку

* ``facts.predicate_kind`` — ключ закрытого словаря (birth, death, …): по нему
  дедупликация сравнивает предикаты, а не написания;
* ``fact_conflicts`` — второе, отличающееся значение функционального предиката
  (у человека одно рождение и одна смерть). Это не новый факт и не молчаливая
  перезапись, а запись на проверку пользователю.

Данные: у уже сохранённых фактов ``predicate_kind`` выводится из подписи
замороженной картой — чтобы существующая база сразу попадала под инвариант.

Revision ID: 3c4d5e6f7a8b
Revises: 2b3c4d5e6f7a
Create Date: 2026-10-15 15:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import text


# revision identifiers, used by Alembic.
revision: str = "3c4d5e6f7a8b"
down_revision: Union[str, Sequence[str], None] = "2b3c4d5e6f7a"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


#: замороженная копия разбора подписей предикатов
_SURFACE_TO_KIND: dict[str, str] = {
    "родился": "birth", "родилась": "birth", "родились": "birth",
    "рождение": "birth", "born": "birth", "birth": "birth",
    "место рождения": "birth_place", "birth_place": "birth_place",
    "умер": "death", "умерла": "death", "умерли": "death", "скончался": "death",
    "скончалась": "death", "смерть": "death", "died": "death", "death": "death",
    "место смерти": "death_place", "death_place": "death_place",
    "имя": "full_name", "полное имя": "full_name", "фио": "full_name",
    "full_name": "full_name",
    "работал": "occupation", "работала": "occupation", "профессия": "occupation",
    "occupation": "occupation",
    "жил": "residence", "жила": "residence", "проживал": "residence",
    "residence": "residence",
    "учился": "education", "училась": "education", "образование": "education",
    "education": "education",
}


def _kind(value: str | None) -> str:
    text = " ".join((value or "").split()).strip().lower()
    if not text:
        return "other"
    if text in _SURFACE_TO_KIND:
        return _SURFACE_TO_KIND[text]
    for word in text.replace("-", " ").split():
        if word in _SURFACE_TO_KIND:
            return _SURFACE_TO_KIND[word]
    return "other"


def _backfill_predicate_kinds(bind) -> int:
    rows = bind.execute(text("select id, predicate from facts order by id")).all()
    for fact_id, predicate in rows:
        bind.execute(
            text("update facts set predicate_kind = :kind where id = :fact_id"),
            {"kind": _kind(predicate), "fact_id": fact_id},
        )
    return len(rows)


def upgrade() -> None:
    """Upgrade schema."""
    bind = op.get_bind()

    op.add_column("facts", sa.Column("predicate_kind", sa.String(length=32), nullable=True))
    op.create_index(op.f("ix_facts_predicate_kind"), "facts", ["predicate_kind"], unique=False)
    converted = _backfill_predicate_kinds(bind)
    print(f"[functional predicates migration] predicate_kind backfilled for {converted} facts")

    op.create_table(
        "fact_conflicts",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("source_id", sa.Integer(), nullable=True),
        sa.Column("subject_entity_id", sa.Integer(), nullable=True),
        sa.Column("predicate_kind", sa.String(length=32), nullable=False),
        sa.Column("existing_fact_id", sa.Integer(), nullable=True),
        sa.Column("existing_value", sa.String(length=255), nullable=True),
        sa.Column("incoming_value", sa.String(length=255), nullable=True),
        sa.Column("source_text", sa.Text(), nullable=True),
        sa.Column("time_text", sa.String(length=255), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("resolution", sa.String(length=24), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True),
        sa.ForeignKeyConstraint(["existing_fact_id"], ["facts.id"], name="fk_fact_conflicts_fact", ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["source_id"], ["knowledge_sources.id"], name="fk_fact_conflicts_source", ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["subject_entity_id"], ["entities.id"], name="fk_fact_conflicts_subject", ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], name="fk_fact_conflicts_user", ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_fact_conflicts_id"), "fact_conflicts", ["id"], unique=False)
    op.create_index(op.f("ix_fact_conflicts_user_id"), "fact_conflicts", ["user_id"], unique=False)
    op.create_index(op.f("ix_fact_conflicts_source_id"), "fact_conflicts", ["source_id"], unique=False)
    op.create_index(op.f("ix_fact_conflicts_subject_entity_id"), "fact_conflicts", ["subject_entity_id"], unique=False)
    op.create_index(op.f("ix_fact_conflicts_predicate_kind"), "fact_conflicts", ["predicate_kind"], unique=False)
    op.create_index(op.f("ix_fact_conflicts_existing_fact_id"), "fact_conflicts", ["existing_fact_id"], unique=False)
    op.create_index(op.f("ix_fact_conflicts_created_at"), "fact_conflicts", ["created_at"], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    for name in (
        "ix_fact_conflicts_created_at",
        "ix_fact_conflicts_existing_fact_id",
        "ix_fact_conflicts_predicate_kind",
        "ix_fact_conflicts_subject_entity_id",
        "ix_fact_conflicts_source_id",
        "ix_fact_conflicts_user_id",
        "ix_fact_conflicts_id",
    ):
        op.drop_index(op.f(name), table_name="fact_conflicts")
    op.drop_table("fact_conflicts")

    op.drop_index(op.f("ix_facts_predicate_kind"), table_name="facts")
    op.drop_column("facts", "predicate_kind")
