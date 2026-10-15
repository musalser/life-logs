"""unified library: user-configured folders for diary and manuscript pages

The «Записи» tab replaces the separate «Дневник» and «Рукописи» tabs. Its page
list is one tree: author objects at the root, then folders the user builds and
pages. This migration adds the storage for that:

* a ``folders`` table — folders owned by an author object (``owner_entity_id``)
  and nested through ``parent_id``;
* a ``folder_id`` column on both ``diary_pages`` and ``htr_pages`` (NULL = the
  author's root level);
* a data backfill that turns the old client-side upload path of a manuscript
  page (``source_path``, e.g. ``diary/1975/scan.jpg``) into a nested folder
  chain under its author, so the structure the reader already had is kept.

Revision ID: 1a2b3c4d5e6f
Revises: 70d54e2ac8e3
Create Date: 2026-10-15 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import text


# revision identifiers, used by Alembic.
revision: str = "1a2b3c4d5e6f"
down_revision: Union[str, Sequence[str], None] = "70d54e2ac8e3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _backfill_folders(bind) -> int:
    """Turns each manuscript page's ``source_path`` into a nested folder chain.

    Only the directory part of the path is used, and one folder chain is built
    per author object, so two authors may each have a ``diary/1975`` of their
    own. Idempotent by construction: the same path yields the same chain.
    """
    pages = bind.execute(
        text(
            "select p.id, p.user_id, a.entity_id, p.source_path "
            "from htr_pages p join htr_authors a on a.id = p.author_id "
            "where p.source_path is not null and p.source_path <> ''"
        )
    ).all()

    cache: dict[tuple, int] = {}
    for page_id, user_id, entity_id, source_path in pages:
        if not entity_id:
            continue
        path = (source_path or "").replace("\\", "/")
        cut = path.rfind("/")
        if cut <= 0:
            continue
        parent_id = None
        for segment in path[:cut].split("/"):
            segment = segment.strip()
            if not segment:
                continue
            key = (user_id, entity_id, parent_id, segment)
            folder_id = cache.get(key)
            if folder_id is None:
                folder_id = bind.execute(
                    text(
                        "insert into folders (user_id, owner_entity_id, name, parent_id) "
                        "values (:user_id, :entity_id, :name, :parent_id) returning id"
                    ),
                    {
                        "user_id": user_id,
                        "entity_id": entity_id,
                        "name": segment[:255],
                        "parent_id": parent_id,
                    },
                ).scalar_one()
                cache[key] = folder_id
            parent_id = folder_id
        bind.execute(
            text("update htr_pages set folder_id = :folder_id where id = :page_id"),
            {"folder_id": parent_id, "page_id": page_id},
        )
    return len(cache)


def upgrade() -> None:
    """Upgrade schema."""
    bind = op.get_bind()

    op.create_table(
        "folders",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("owner_entity_id", sa.Integer(), nullable=True),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("parent_id", sa.Integer(), nullable=True),
        sa.Column("order_index", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True),
        sa.ForeignKeyConstraint(["owner_entity_id"], ["entities.id"], name="fk_folders_owner", ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["parent_id"], ["folders.id"], name="fk_folders_parent", ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], name="fk_folders_user", ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_folders_id"), "folders", ["id"], unique=False)
    op.create_index(op.f("ix_folders_user_id"), "folders", ["user_id"], unique=False)
    op.create_index(op.f("ix_folders_owner_entity_id"), "folders", ["owner_entity_id"], unique=False)
    op.create_index(op.f("ix_folders_parent_id"), "folders", ["parent_id"], unique=False)

    op.add_column("diary_pages", sa.Column("folder_id", sa.Integer(), nullable=True))
    op.add_column("htr_pages", sa.Column("folder_id", sa.Integer(), nullable=True))

    created = _backfill_folders(bind)
    print(f"[library migration] folders created from source_path: {created}")

    op.create_index(op.f("ix_diary_pages_folder_id"), "diary_pages", ["folder_id"], unique=False)
    op.create_foreign_key(
        "fk_diary_pages_folder", "diary_pages", "folders", ["folder_id"], ["id"], ondelete="SET NULL"
    )
    op.create_index(op.f("ix_htr_pages_folder_id"), "htr_pages", ["folder_id"], unique=False)
    op.create_foreign_key(
        "fk_htr_pages_folder", "htr_pages", "folders", ["folder_id"], ["id"], ondelete="SET NULL"
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint("fk_htr_pages_folder", "htr_pages", type_="foreignkey")
    op.drop_index(op.f("ix_htr_pages_folder_id"), table_name="htr_pages")
    op.drop_column("htr_pages", "folder_id")

    op.drop_constraint("fk_diary_pages_folder", "diary_pages", type_="foreignkey")
    op.drop_index(op.f("ix_diary_pages_folder_id"), table_name="diary_pages")
    op.drop_column("diary_pages", "folder_id")

    op.drop_index(op.f("ix_folders_parent_id"), table_name="folders")
    op.drop_index(op.f("ix_folders_owner_entity_id"), table_name="folders")
    op.drop_index(op.f("ix_folders_user_id"), table_name="folders")
    op.drop_index(op.f("ix_folders_id"), table_name="folders")
    op.drop_table("folders")
