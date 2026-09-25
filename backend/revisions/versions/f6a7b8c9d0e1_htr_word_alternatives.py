"""keep the alternative readings of every recognized word

The beam search already produces an N-best list per line; the variants that
differ in one word are the natural "what else could this word be" list the
editor shows when the user clicks a word. They are written once, at recognition
time, because they come from the CTC matrix, which is not stored.

Only what the beam considered ends up here — a word the acoustic model never
produced (a rare name, a dialect spelling) cannot appear. The dictionary check
and the LLM proposals stay the path for those.

Revision ID: f6a7b8c9d0e1
Revises: e5f6a7b8c9d0
Create Date: 2026-09-25 20:40:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "f6a7b8c9d0e1"
down_revision: Union[str, Sequence[str], None] = "e5f6a7b8c9d0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("htr_words", sa.Column("alternatives", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("htr_words", "alternatives")
