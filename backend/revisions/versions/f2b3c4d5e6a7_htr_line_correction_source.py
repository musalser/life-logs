"""add corrected_by to htr_lines

The LLM corrector fills `corrected_text` on its own, so the UI has to be able to
tell an automatic correction from a human one.

Revision ID: f2b3c4d5e6a7
Revises: e7f8a9b0c1d2
Create Date: 2026-09-23 14:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "f2b3c4d5e6a7"
down_revision: Union[str, Sequence[str], None] = "e7f8a9b0c1d2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("htr_lines", sa.Column("corrected_by", sa.String(length=16), nullable=True))


def downgrade() -> None:
    op.drop_column("htr_lines", "corrected_by")
