"""let the user remove a word from their own vocabulary

The dictionary check treats the words of the author's *confirmed* pages as
known, which is what keeps their names and dialect words from being flagged.
The same rule lets a once-confirmed typo masquerade as a real word on every
other page. This table stores the words the user explicitly removed, so the
vocabulary snapshot can subtract them and the word is highlighted again.

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-09-25 23:40:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "b2c3d4e5f6a7"
down_revision: Union[str, Sequence[str], None] = "a1b2c3d4e5f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "htr_lexicon_ignores",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("author_id", sa.Integer(), nullable=False),
        sa.Column("word", sa.String(length=255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.ForeignKeyConstraint(["author_id"], ["htr_authors.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("author_id", "word", name="uq_htr_lexicon_ignore_author_word"),
    )
    op.create_index(
        "ix_htr_lexicon_ignores_author_id", "htr_lexicon_ignores", ["author_id"]
    )


def downgrade() -> None:
    op.drop_index("ix_htr_lexicon_ignores_author_id", table_name="htr_lexicon_ignores")
    op.drop_table("htr_lexicon_ignores")
