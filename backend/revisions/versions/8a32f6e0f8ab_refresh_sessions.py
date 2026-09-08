"""add refresh sessions

Revision ID: 8a32f6e0f8ab
Revises: e54ece0090cd
Create Date: 2026-09-08 13:10:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "8a32f6e0f8ab"
down_revision: Union[str, Sequence[str], None] = "e54ece0090cd"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "refresh_sessions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("username", sa.String(), nullable=False),
        sa.Column("family_id", sa.String(length=64), nullable=False),
        sa.Column("token_jti", sa.String(length=64), nullable=False),
        sa.Column("token_hash", sa.String(length=128), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("issued_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True),
        sa.Column("rotated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("reuse_detected_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("replaced_by_jti", sa.String(length=64), nullable=True),
        sa.Column("user_agent", sa.String(length=255), nullable=True),
        sa.Column("ip_address", sa.String(length=64), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_refresh_sessions_id"), "refresh_sessions", ["id"], unique=False)
    op.create_index(op.f("ix_refresh_sessions_user_id"), "refresh_sessions", ["user_id"], unique=False)
    op.create_index(op.f("ix_refresh_sessions_username"), "refresh_sessions", ["username"], unique=False)
    op.create_index(op.f("ix_refresh_sessions_family_id"), "refresh_sessions", ["family_id"], unique=False)
    op.create_index(op.f("ix_refresh_sessions_token_jti"), "refresh_sessions", ["token_jti"], unique=True)
    op.create_index(op.f("ix_refresh_sessions_token_hash"), "refresh_sessions", ["token_hash"], unique=True)
    op.create_index(op.f("ix_refresh_sessions_expires_at"), "refresh_sessions", ["expires_at"], unique=False)
    op.create_index(op.f("ix_refresh_sessions_replaced_by_jti"), "refresh_sessions", ["replaced_by_jti"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_refresh_sessions_replaced_by_jti"), table_name="refresh_sessions")
    op.drop_index(op.f("ix_refresh_sessions_expires_at"), table_name="refresh_sessions")
    op.drop_index(op.f("ix_refresh_sessions_token_hash"), table_name="refresh_sessions")
    op.drop_index(op.f("ix_refresh_sessions_token_jti"), table_name="refresh_sessions")
    op.drop_index(op.f("ix_refresh_sessions_family_id"), table_name="refresh_sessions")
    op.drop_index(op.f("ix_refresh_sessions_username"), table_name="refresh_sessions")
    op.drop_index(op.f("ix_refresh_sessions_user_id"), table_name="refresh_sessions")
    op.drop_index(op.f("ix_refresh_sessions_id"), table_name="refresh_sessions")
    op.drop_table("refresh_sessions")
