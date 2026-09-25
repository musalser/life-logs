"""keep the uploaded file name of a page and let the user order the list

The sidebar used to show "Стр. #<id>" and to sort by id, which says nothing
about the manuscript and cannot be changed. Pages now remember the name of the
uploaded file (plus the client-side path when a folder upload provided one, so
equal names stay distinguishable) and carry an explicit order_index that
drag-and-drop rewrites.

Existing rows are numbered per author in their current visible order (newest
first) so the list does not jump when this migration lands.

Revision ID: a1b2c3d4e5f6
Revises: f6a7b8c9d0e1
Create Date: 2026-09-25 22:55:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, Sequence[str], None] = "f6a7b8c9d0e1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("htr_pages", sa.Column("file_name", sa.String(length=1024), nullable=True))
    op.add_column("htr_pages", sa.Column("source_path", sa.String(length=2048), nullable=True))
    op.add_column(
        "htr_pages",
        sa.Column("order_index", sa.Integer(), nullable=False, server_default="0"),
    )
    op.create_index("ix_htr_pages_order_index", "htr_pages", ["order_index"])

    # number the existing pages per author in the order the UI showed them
    # (id desc, newest first); no-op on an empty table
    bind = op.get_bind()
    rows = bind.execute(
        sa.text("SELECT id, author_id FROM htr_pages ORDER BY author_id, id DESC")
    ).fetchall()
    counters: dict[int, int] = {}
    for page_id, author_id in rows:
        index = counters.get(author_id, 0)
        bind.execute(
            sa.text("UPDATE htr_pages SET order_index = :index WHERE id = :page_id"),
            {"index": index, "page_id": page_id},
        )
        counters[author_id] = index + 1


def downgrade() -> None:
    op.drop_index("ix_htr_pages_order_index", table_name="htr_pages")
    op.drop_column("htr_pages", "order_index")
    op.drop_column("htr_pages", "source_path")
    op.drop_column("htr_pages", "file_name")
