"""add HTR tables: htr_authors, htr_pages, htr_lines, htr_words, htr_model_versions, htr_training_runs

Revision ID: d4e5f6a7b8c9
Revises: b7c1d94e2f10
Create Date: 2026-09-18 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "d4e5f6a7b8c9"
down_revision: Union[str, Sequence[str], None] = "b7c1d94e2f10"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "htr_authors",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_htr_authors_id"), "htr_authors", ["id"], unique=False)
    op.create_index(op.f("ix_htr_authors_user_id"), "htr_authors", ["user_id"], unique=False)

    op.create_table(
        "htr_model_versions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("author_id", sa.Integer(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("base_model_id", sa.String(length=255), nullable=False),
        sa.Column("file_path", sa.String(length=1024), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="TRAINING"),
        sa.Column("dataset_hash", sa.String(length=64), nullable=True),
        sa.Column("metrics", sa.Text(), nullable=True),
        sa.Column("training_config", sa.Text(), nullable=True),
        sa.Column("environment", sa.Text(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True),
        sa.Column("activated_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["author_id"], ["htr_authors.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("author_id", "version", name="uq_htr_model_author_version"),
    )
    op.create_index(op.f("ix_htr_model_versions_id"), "htr_model_versions", ["id"], unique=False)
    op.create_index(op.f("ix_htr_model_versions_author_id"), "htr_model_versions", ["author_id"], unique=False)
    op.create_index(op.f("ix_htr_model_versions_status"), "htr_model_versions", ["status"], unique=False)

    op.create_table(
        "htr_pages",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("author_id", sa.Integer(), nullable=False),
        sa.Column("file_path", sa.String(length=1024), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="UPLOADED"),
        sa.Column("width", sa.Integer(), nullable=True),
        sa.Column("height", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True),
        sa.Column("confirmed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("recognition_model_version_id", sa.Integer(), nullable=True),
        sa.Column("prediction_cer", sa.Float(), nullable=True),
        sa.Column("prediction_wer", sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["author_id"], ["htr_authors.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["recognition_model_version_id"], ["htr_model_versions.id"], ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_htr_pages_id"), "htr_pages", ["id"], unique=False)
    op.create_index(op.f("ix_htr_pages_user_id"), "htr_pages", ["user_id"], unique=False)
    op.create_index(op.f("ix_htr_pages_author_id"), "htr_pages", ["author_id"], unique=False)
    op.create_index(op.f("ix_htr_pages_status"), "htr_pages", ["status"], unique=False)

    op.create_table(
        "htr_lines",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("page_id", sa.Integer(), nullable=False),
        sa.Column("order_index", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("x1", sa.Integer(), nullable=False),
        sa.Column("y1", sa.Integer(), nullable=False),
        sa.Column("x2", sa.Integer(), nullable=False),
        sa.Column("y2", sa.Integer(), nullable=False),
        sa.Column("predicted_text", sa.Text(), nullable=True),
        sa.Column("corrected_text", sa.Text(), nullable=True),
        sa.Column("words_stale", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.ForeignKeyConstraint(["page_id"], ["htr_pages.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_htr_lines_id"), "htr_lines", ["id"], unique=False)
    op.create_index(op.f("ix_htr_lines_page_id"), "htr_lines", ["page_id"], unique=False)

    op.create_table(
        "htr_words",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("line_id", sa.Integer(), nullable=False),
        sa.Column("order_index", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("x1", sa.Integer(), nullable=False),
        sa.Column("y1", sa.Integer(), nullable=False),
        sa.Column("x2", sa.Integer(), nullable=False),
        sa.Column("y2", sa.Integer(), nullable=False),
        sa.Column("predicted_text", sa.Text(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("corrected_text", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["line_id"], ["htr_lines.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_htr_words_id"), "htr_words", ["id"], unique=False)
    op.create_index(op.f("ix_htr_words_line_id"), "htr_words", ["line_id"], unique=False)

    op.create_table(
        "htr_training_runs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("author_id", sa.Integer(), nullable=False),
        sa.Column("model_version_id", sa.Integer(), nullable=True),
        sa.Column("dataset_hash", sa.String(length=64), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="RUNNING"),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("metrics", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(["author_id"], ["htr_authors.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["model_version_id"], ["htr_model_versions.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_htr_training_runs_id"), "htr_training_runs", ["id"], unique=False)
    op.create_index(op.f("ix_htr_training_runs_author_id"), "htr_training_runs", ["author_id"], unique=False)
    op.create_index(op.f("ix_htr_training_runs_status"), "htr_training_runs", ["status"], unique=False)


def downgrade() -> None:
    op.drop_table("htr_training_runs")
    op.drop_table("htr_words")
    op.drop_table("htr_lines")
    op.drop_table("htr_pages")
    op.drop_table("htr_model_versions")
    op.drop_table("htr_authors")
