"""log accepted and dismissed model proposals

The correction loop needs feedback: which proposals were taken, which were
rejected, and whether the dictionary verdict predicted the human decision.

Revision ID: e5f6a7b8c9d0
Revises: c3d4e5f6a7b8
Create Date: 2026-09-24 00:45:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "e5f6a7b8c9d0"
down_revision: Union[str, Sequence[str], None] = "c3d4e5f6a7b8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "htr_suggestion_events",
        sa.Column("id", sa.Integer(), primary_key=True, index=True),
        sa.Column("page_id", sa.Integer(), sa.ForeignKey("htr_pages.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("line_id", sa.Integer(), sa.ForeignKey("htr_lines.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("action", sa.String(length=24), nullable=False, index=True),
        sa.Column("suggested_text", sa.Text(), nullable=True),
        sa.Column("original_text", sa.Text(), nullable=True),
        sa.Column("change_before", sa.Text(), nullable=True),
        sa.Column("change_after", sa.Text(), nullable=True),
        sa.Column("in_lexicon", sa.Boolean(), nullable=True),
        sa.Column("suggested_by", sa.String(length=32), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), index=True),
    )


def downgrade() -> None:
    op.drop_table("htr_suggestion_events")
