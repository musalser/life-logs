"""add polygons to htr_lines and htr_words

Baseline segmentation returns curved outlines; the axis-aligned bbox stays as a
cheap envelope, but the shape that is actually drawn (and shown to the user) is
the polygon.

Revision ID: e7f8a9b0c1d2
Revises: d4e5f6a7b8c9
Create Date: 2026-09-23 00:40:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "e7f8a9b0c1d2"
down_revision: Union[str, Sequence[str], None] = "d4e5f6a7b8c9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("htr_lines", sa.Column("polygon", sa.Text(), nullable=True))
    op.add_column("htr_words", sa.Column("polygon", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("htr_words", "polygon")
    op.drop_column("htr_lines", "polygon")
