"""knowledge sources, perspective objects and graph edges

Schema of the knowledge base (spec section 5) plus the data migration that turns
the old single-perspective knowledge into it:

* every account gets a person-object ("Я"), and users.self_entity_id points at it;
* a manuscript author is bound to an object, so relations read from their pages
  are attributed to *them* ("её мама" is the author's mother, not the reader's);
* every diary page and every confirmed manuscript page becomes a knowledge_source
  (its text, its text hash, and — for manuscripts — the line map that turns an
  offset back into a line);
* children (mentions, events, progress, habit logs) keep their connection to the
  page through the new source_id;
* the old entity_relations rows become graph edges from the account's own object.

Data steps run in Python on purpose: assembling a manuscript page's text and its
line offsets in SQL would be unreadable, and the pipeline of stage 3 does the
same thing in app code (the copy is frozen here the way migrations should be).

Revision ID: 70d54e2ac8e3
Revises: b2c3d4e5f6a7
Create Date: 2026-10-09 22:35:56.805491

"""
import hashlib
import json
from datetime import datetime, timezone
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import text
from pgvector.sqlalchemy import Vector


# revision identifiers, used by Alembic.
revision: str = "70d54e2ac8e3"
down_revision: Union[str, Sequence[str], None] = "b2c3d4e5f6a7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

SOURCE_DIARY = "diary"
SOURCE_HTR = "htr"
STATUS_PENDING = "pending"

#: names that stand for "the account owner" when a manuscript author is bound
SELF_NAMES = ("me", "я", "i")


# ---------------------------------------------------------------------------
# Helpers of the data migration
# ---------------------------------------------------------------------------


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _normalize(value: str | None) -> str:
    return " ".join((value or "").split()).lower()[:255]


def _sha256(value: str) -> str:
    return hashlib.sha256((value or "").encode("utf-8")).hexdigest()


def _assemble_lines(rows: list[tuple[int, str | None]]) -> tuple[str, str]:
    """Turns ordered (line_id, text) pairs into page text plus a line map.

    The map carries global offsets into that text, which is how a fact later
    points at "the real text": offset -> line -> the image region. Lines are
    joined with a newline, and the newline counts towards the offsets.
    """
    parts: list[str] = []
    line_map: list[dict] = []
    offset = 0
    for index, (line_id, raw) in enumerate(rows):
        value = raw or ""
        if index:
            parts.append("\n")
            offset += 1
        start = offset
        parts.append(value)
        offset += len(value)
        line_map.append({"line_id": line_id, "start": start, "end": offset})
    return "".join(parts), json.dumps(line_map, ensure_ascii=False)


def _find_entity(bind, user_id: int, normalized: str) -> int | None:
    row = bind.execute(
        text(
            "select id from entities where user_id = :user_id "
            "and entity_type = 'person' and normalized_name = :name"
        ),
        {"user_id": user_id, "name": normalized},
    ).first()
    return row[0] if row else None


def _entity_for_name(bind, user_id: int, name: str) -> int:
    """The person object with that name, created if it does not exist yet."""
    normalized = _normalize(name)
    existing = _find_entity(bind, user_id, normalized)
    if existing is not None:
        return existing
    now = _now()
    return bind.execute(
        text(
            "insert into entities (user_id, entity_type, canonical_name, normalized_name, "
            "first_seen_at, last_seen_at, created_at) "
            "values (:user_id, 'person', :canonical, :normalized, :now, :now, :now) "
            "returning id"
        ),
        {
            "user_id": user_id,
            "canonical": (name or "")[:255],
            "normalized": normalized,
            "now": now,
        },
    ).scalar_one()


def _migrate_self_entities(bind) -> int:
    """Gives every account its own object and points users.self_entity_id at it."""
    users = bind.execute(text("select id, username, name from users order by id")).all()
    created = 0
    for user_id, username, name in users:
        entity_id = None
        for candidate in SELF_NAMES + (_normalize(name), _normalize(username)):
            normalized = _normalize(candidate)
            if not normalized:
                continue
            entity_id = _find_entity(bind, user_id, normalized)
            if entity_id is not None:
                break
        if entity_id is None:
            entity_id = _entity_for_name(bind, user_id, name or username or "Я")
        bind.execute(
            text("update users set self_entity_id = :entity_id where id = :user_id"),
            {"entity_id": entity_id, "user_id": user_id},
        )
        created += 1
    return created


def _migrate_authors(bind) -> int:
    """Binds a manuscript author to an object: "me" is the account's own object."""
    users = {
        row[0]: (row[1], row[2])
        for row in bind.execute(text("select id, username, name, self_entity_id from users")).all()
    }
    bound = 0
    authors = bind.execute(text("select id, user_id, name from htr_authors order by id")).all()
    for author_id, user_id, author_name in authors:
        normalized = _normalize(author_name)
        username, name = users.get(user_id, (None, None))
        self_entity_id = bind.execute(
            text("select self_entity_id from users where id = :user_id"), {"user_id": user_id}
        ).scalar()
        if normalized in SELF_NAMES or normalized in (_normalize(name), _normalize(username)):
            entity_id = self_entity_id
        else:
            entity_id = _entity_for_name(bind, user_id, author_name)
        bind.execute(
            text("update htr_authors set entity_id = :entity_id where id = :author_id"),
            {"entity_id": entity_id, "author_id": author_id},
        )
        bound += 1
    return bound


def _migrate_sources(bind) -> tuple[int, int]:
    """Creates a source per diary page and per *confirmed* manuscript page."""
    sources: dict[tuple[str, int], int] = {}

    diary_pages = bind.execute(
        text("select id, user_id, title, content from diary_pages order by id")
    ).all()
    for page_id, user_id, title, content in diary_pages:
        body = content or ""
        self_entity_id = bind.execute(
            text("select self_entity_id from users where id = :user_id"), {"user_id": user_id}
        ).scalar()
        source_id = bind.execute(
            text(
                "insert into knowledge_sources (user_id, source_type, ref_id, author_entity_id, "
                "title, text, text_hash, line_map, extraction_status) "
                "values (:user_id, :source_type, :ref_id, :author_entity_id, :title, :text, "
                ":text_hash, null, :status) returning id"
            ),
            {
                "user_id": user_id,
                "source_type": SOURCE_DIARY,
                "ref_id": page_id,
                "author_entity_id": self_entity_id,
                "title": (title or "")[:255] or None,
                "text": body,
                "text_hash": _sha256(body),
                "status": STATUS_PENDING,
            },
        ).scalar_one()
        sources[(SOURCE_DIARY, page_id)] = source_id

    htr_pages = bind.execute(
        text(
            "select p.id, p.user_id, p.file_name, p.author_id, a.entity_id "
            "from htr_pages p join htr_authors a on a.id = p.author_id "
            "where p.status = 'CONFIRMED' order by p.id"
        )
    ).all()
    for page_id, user_id, file_name, author_id, author_entity_id in htr_pages:
        lines = bind.execute(
            text(
                "select id, coalesce(nullif(corrected_text, ''), predicted_text) "
                "from htr_lines where page_id = :page_id order by order_index, id"
            ),
            {"page_id": page_id},
        ).all()
        body, line_map = _assemble_lines([(row[0], row[1]) for row in lines])
        source_id = bind.execute(
            text(
                "insert into knowledge_sources (user_id, source_type, ref_id, author_entity_id, "
                "title, text, text_hash, line_map, extraction_status) "
                "values (:user_id, :source_type, :ref_id, :author_entity_id, :title, :text, "
                ":text_hash, :line_map, :status) returning id"
            ),
            {
                "user_id": user_id,
                "source_type": SOURCE_HTR,
                "ref_id": page_id,
                "author_entity_id": author_entity_id,
                "title": (file_name or f"Стр. #{page_id}")[:255],
                "text": body,
                "text_hash": _sha256(body),
                "line_map": line_map,
                "status": STATUS_PENDING,
            },
        ).scalar_one()
        sources[(SOURCE_HTR, page_id)] = source_id

    return len(diary_pages), len(htr_pages)


def _redirect_children(bind) -> None:
    """Points the old diary_page_id references at the new sources."""
    bind.execute(
        text(
            "update entity_mentions m set source_id = s.id "
            "from knowledge_sources s "
            "where s.source_type = 'diary' and s.ref_id = m.diary_page_id and m.source_id is null"
        )
    )
    for table in ("facts", "goal_progress", "events", "habit_logs"):
        bind.execute(
            text(
                f"update {table} c set source_id = s.id "
                "from knowledge_sources s "
                "where s.source_type = 'diary' and s.ref_id = c.diary_page_id and c.source_id is null"
            )
        )
    bind.execute(
        text(
            "update goals g set created_from_source_id = s.id "
            "from knowledge_sources s "
            "where s.source_type = 'diary' and s.ref_id = g.created_from_page_id "
            "and g.created_from_source_id is null"
        )
    )


def _migrate_relations(bind) -> tuple[int, int]:
    """Old single-perspective relations become edges from the account's object.

    The rows are converted in place (not copied): the legacy table has a NOT NULL
    ``entity_id`` and the new edge columns must be filled on the very same rows,
    so that later tightening them cannot trip over leftovers with a NULL `from`.
    An old row whose perspective does not exist (or which pointed at the object
    itself) carries no edge and is removed.
    """
    rows = bind.execute(
        text("select id, user_id, entity_id from entity_relations order by id")
    ).all()
    converted = 0
    dropped = 0
    for edge_id, user_id, to_entity_id in rows:
        from_entity_id = bind.execute(
            text("select self_entity_id from users where id = :user_id"), {"user_id": user_id}
        ).scalar()
        if from_entity_id is None or from_entity_id == to_entity_id:
            bind.execute(text("delete from entity_relations where id = :edge_id"), {"edge_id": edge_id})
            dropped += 1
            continue
        bind.execute(
            text(
                "update entity_relations set from_entity_id = :from_entity_id, "
                "to_entity_id = :to_entity_id where id = :edge_id"
            ),
            {"from_entity_id": from_entity_id, "to_entity_id": to_entity_id, "edge_id": edge_id},
        )
        converted += 1
    return converted, dropped


def _assert_no_orphans(bind) -> None:
    """Refuses to tighten NOT NULL while a child still has no source."""
    problems = []
    for table in ("entity_mentions", "facts", "goal_progress", "events", "habit_logs"):
        missing = bind.execute(
            text(f"select count(*) from {table} where source_id is null")
        ).scalar()
        if missing:
            problems.append(f"{table}={missing}")
    if problems:
        raise RuntimeError(
            "knowledge migration: rows without a source cannot be migrated ("
            + ", ".join(problems)
            + "); their diary page is gone, delete them manually and re-run"
        )


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def upgrade() -> None:
    """Upgrade schema."""
    bind = op.get_bind()
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # --- new tables -------------------------------------------------------
    op.create_table(
        "knowledge_sources",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("source_type", sa.String(length=16), nullable=False),
        sa.Column("ref_id", sa.Integer(), nullable=False),
        sa.Column("author_entity_id", sa.Integer(), nullable=True),
        sa.Column("title", sa.String(length=255), nullable=True),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("text_hash", sa.String(length=64), nullable=True),
        sa.Column("line_map", sa.Text(), nullable=True),
        sa.Column("extraction_status", sa.String(length=16), nullable=False),
        sa.Column("last_extracted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True),
        sa.ForeignKeyConstraint(["author_entity_id"], ["entities.id"], name="fk_knowledge_sources_author", ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], name="fk_knowledge_sources_user", ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", "source_type", "ref_id", name="uq_knowledge_source_user_ref"),
    )
    op.create_index(op.f("ix_knowledge_sources_id"), "knowledge_sources", ["id"], unique=False)
    op.create_index(op.f("ix_knowledge_sources_user_id"), "knowledge_sources", ["user_id"], unique=False)
    op.create_index(op.f("ix_knowledge_sources_source_type"), "knowledge_sources", ["source_type"], unique=False)
    op.create_index(op.f("ix_knowledge_sources_ref_id"), "knowledge_sources", ["ref_id"], unique=False)
    op.create_index(op.f("ix_knowledge_sources_author_entity_id"), "knowledge_sources", ["author_entity_id"], unique=False)
    op.create_index(op.f("ix_knowledge_sources_text_hash"), "knowledge_sources", ["text_hash"], unique=False)
    op.create_index(op.f("ix_knowledge_sources_extraction_status"), "knowledge_sources", ["extraction_status"], unique=False)

    op.create_table(
        "extraction_runs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("source_id", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("stats", sa.Text(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("model", sa.String(length=64), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["source_id"], ["knowledge_sources.id"], name="fk_extraction_runs_source", ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_extraction_runs_id"), "extraction_runs", ["id"], unique=False)
    op.create_index(op.f("ix_extraction_runs_source_id"), "extraction_runs", ["source_id"], unique=False)
    op.create_index(op.f("ix_extraction_runs_status"), "extraction_runs", ["status"], unique=False)

    op.create_table(
        "chunks",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("source_id", sa.Integer(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("start_offset", sa.Integer(), nullable=True),
        sa.Column("end_offset", sa.Integer(), nullable=True),
        sa.Column("embedding", Vector(768), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True),
        sa.ForeignKeyConstraint(["source_id"], ["knowledge_sources.id"], name="fk_chunks_source", ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], name="fk_chunks_user", ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_chunks_id"), "chunks", ["id"], unique=False)
    op.create_index(op.f("ix_chunks_user_id"), "chunks", ["user_id"], unique=False)
    op.create_index(op.f("ix_chunks_source_id"), "chunks", ["source_id"], unique=False)

    op.create_table(
        "knowledge_insights",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("subject_entity_id", sa.Integer(), nullable=True),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("payload", sa.Text(), nullable=False),
        sa.Column("model", sa.String(length=64), nullable=True),
        sa.Column("generated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True),
        sa.ForeignKeyConstraint(["subject_entity_id"], ["entities.id"], name="fk_knowledge_insights_subject", ondelete="SET NULL"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], name="fk_knowledge_insights_user", ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", "subject_entity_id", name="uq_knowledge_insight_subject"),
    )
    op.create_index(op.f("ix_knowledge_insights_id"), "knowledge_insights", ["id"], unique=False)
    op.create_index(op.f("ix_knowledge_insights_user_id"), "knowledge_insights", ["user_id"], unique=False)
    op.create_index(op.f("ix_knowledge_insights_subject_entity_id"), "knowledge_insights", ["subject_entity_id"], unique=False)

    op.create_table(
        "knowledge_state",
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], name="fk_knowledge_state_user", ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("user_id"),
    )
    op.execute("INSERT INTO knowledge_state (user_id, version) SELECT id, 0 FROM users")

    # --- new columns on existing tables (nullable until the data is moved) --
    op.add_column("users", sa.Column("self_entity_id", sa.Integer(), nullable=True))
    op.create_unique_constraint("uq_users_self_entity", "users", ["self_entity_id"])
    op.create_foreign_key(
        "fk_users_self_entity", "users", "entities", ["self_entity_id"], ["id"], ondelete="SET NULL"
    )

    op.add_column("entities", sa.Column("aliases_json", sa.Text(), nullable=True))
    op.add_column("entities", sa.Column("embedding", Vector(768), nullable=True))
    op.add_column(
        "entities",
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True),
    )

    op.add_column("entries", sa.Column("context_json", sa.Text(), nullable=True))

    op.add_column("entity_mentions", sa.Column("source_id", sa.Integer(), nullable=True))
    op.add_column("facts", sa.Column("source_id", sa.Integer(), nullable=True))
    op.add_column("facts", sa.Column("start_offset", sa.Integer(), nullable=True))
    op.add_column("facts", sa.Column("end_offset", sa.Integer(), nullable=True))

    op.add_column("entity_relations", sa.Column("from_entity_id", sa.Integer(), nullable=True))
    op.add_column("entity_relations", sa.Column("to_entity_id", sa.Integer(), nullable=True))
    op.add_column("entity_relations", sa.Column("source_id", sa.Integer(), nullable=True))
    op.add_column("entity_relations", sa.Column("start_offset", sa.Integer(), nullable=True))
    op.add_column("entity_relations", sa.Column("end_offset", sa.Integer(), nullable=True))

    op.add_column("goals", sa.Column("subject_entity_id", sa.Integer(), nullable=True))
    op.add_column("goals", sa.Column("created_from_source_id", sa.Integer(), nullable=True))
    op.add_column("goals", sa.Column("embedding", Vector(768), nullable=True))

    op.add_column("goal_progress", sa.Column("source_id", sa.Integer(), nullable=True))
    op.add_column("events", sa.Column("source_id", sa.Integer(), nullable=True))
    op.add_column("events", sa.Column("subject_entity_id", sa.Integer(), nullable=True))
    op.add_column("events", sa.Column("start_offset", sa.Integer(), nullable=True))
    op.add_column("events", sa.Column("end_offset", sa.Integer(), nullable=True))
    op.add_column("events", sa.Column("embedding", Vector(768), nullable=True))

    op.add_column("habits", sa.Column("subject_entity_id", sa.Integer(), nullable=True))
    op.add_column("habits", sa.Column("embedding", Vector(768), nullable=True))

    op.add_column("habit_logs", sa.Column("source_id", sa.Integer(), nullable=True))
    op.add_column("htr_authors", sa.Column("entity_id", sa.Integer(), nullable=True))

    # --- move the data ----------------------------------------------------
    users = _migrate_self_entities(bind)
    authors = _migrate_authors(bind)
    diary_sources, htr_sources = _migrate_sources(bind)
    _redirect_children(bind)
    edges, dropped_edges = _migrate_relations(bind)
    _assert_no_orphans(bind)
    print(
        f"[knowledge migration] self-objects={users} authors={authors} "
        f"diary sources={diary_sources} manuscript sources={htr_sources} "
        f"edges={edges} (dropped {dropped_edges})"
    )

    # --- tighten the moved columns ---------------------------------------
    for table in ("entity_mentions", "facts", "goal_progress", "events", "habit_logs"):
        op.alter_column(table, "source_id", existing_type=sa.Integer(), nullable=False)
    op.alter_column("entity_relations", "from_entity_id", existing_type=sa.Integer(), nullable=False)
    op.alter_column("entity_relations", "to_entity_id", existing_type=sa.Integer(), nullable=False)

    # --- drop what the sources replaced ----------------------------------
    for table in ("entity_mentions", "facts", "goal_progress", "events", "habit_logs"):
        op.drop_index(op.f(f"ix_{table}_diary_page_id"), table_name=table)
        op.drop_constraint(f"{table}_diary_page_id_fkey", table, type_="foreignkey")
        op.drop_column(table, "diary_page_id")

    op.drop_index(op.f("ix_entity_relations_entity_id"), table_name="entity_relations")
    op.drop_constraint("uq_entity_relation_user_entity", "entity_relations", type_="unique")
    op.drop_constraint("entity_relations_entity_id_fkey", "entity_relations", type_="foreignkey")
    op.drop_constraint("entity_relations_last_page_id_fkey", "entity_relations", type_="foreignkey")
    op.drop_column("entity_relations", "last_page_id")
    op.drop_column("entity_relations", "entity_id")

    op.drop_constraint("goals_created_from_page_id_fkey", "goals", type_="foreignkey")
    op.drop_column("goals", "created_from_page_id")

    # --- indexes and foreign keys of the new columns ----------------------
    op.create_index(op.f("ix_entity_mentions_source_id"), "entity_mentions", ["source_id"], unique=False)
    op.create_foreign_key(
        "fk_entity_mentions_source", "entity_mentions", "knowledge_sources", ["source_id"], ["id"], ondelete="CASCADE"
    )
    op.create_index(op.f("ix_facts_source_id"), "facts", ["source_id"], unique=False)
    op.create_foreign_key("fk_facts_source", "facts", "knowledge_sources", ["source_id"], ["id"], ondelete="CASCADE")
    op.create_index(op.f("ix_goal_progress_source_id"), "goal_progress", ["source_id"], unique=False)
    op.create_foreign_key(
        "fk_goal_progress_source", "goal_progress", "knowledge_sources", ["source_id"], ["id"], ondelete="CASCADE"
    )
    op.create_index(op.f("ix_events_source_id"), "events", ["source_id"], unique=False)
    op.create_index(op.f("ix_events_subject_entity_id"), "events", ["subject_entity_id"], unique=False)
    op.create_foreign_key("fk_events_source", "events", "knowledge_sources", ["source_id"], ["id"], ondelete="CASCADE")
    op.create_foreign_key(
        "fk_events_subject", "events", "entities", ["subject_entity_id"], ["id"], ondelete="SET NULL"
    )
    op.create_index(op.f("ix_habit_logs_source_id"), "habit_logs", ["source_id"], unique=False)
    op.create_foreign_key(
        "fk_habit_logs_source", "habit_logs", "knowledge_sources", ["source_id"], ["id"], ondelete="CASCADE"
    )
    op.create_index(op.f("ix_entity_relations_from_entity_id"), "entity_relations", ["from_entity_id"], unique=False)
    op.create_index(op.f("ix_entity_relations_to_entity_id"), "entity_relations", ["to_entity_id"], unique=False)
    op.create_index(op.f("ix_entity_relations_source_id"), "entity_relations", ["source_id"], unique=False)
    op.create_foreign_key(
        "fk_entity_relations_from", "entity_relations", "entities", ["from_entity_id"], ["id"], ondelete="CASCADE"
    )
    op.create_foreign_key(
        "fk_entity_relations_to", "entity_relations", "entities", ["to_entity_id"], ["id"], ondelete="CASCADE"
    )
    op.create_foreign_key(
        "fk_entity_relations_source", "entity_relations", "knowledge_sources", ["source_id"], ["id"], ondelete="CASCADE"
    )
    op.create_unique_constraint(
        "uq_entity_relation_edge", "entity_relations", ["user_id", "from_entity_id", "to_entity_id", "relation_type"]
    )
    op.create_index(op.f("ix_goals_subject_entity_id"), "goals", ["subject_entity_id"], unique=False)
    op.create_foreign_key(
        "fk_goals_subject", "goals", "entities", ["subject_entity_id"], ["id"], ondelete="SET NULL"
    )
    op.create_foreign_key(
        "fk_goals_created_from_source", "goals", "knowledge_sources", ["created_from_source_id"], ["id"], ondelete="SET NULL"
    )
    op.create_index(op.f("ix_habits_subject_entity_id"), "habits", ["subject_entity_id"], unique=False)
    op.create_foreign_key(
        "fk_habits_subject", "habits", "entities", ["subject_entity_id"], ["id"], ondelete="SET NULL"
    )
    op.create_index(op.f("ix_htr_authors_entity_id"), "htr_authors", ["entity_id"], unique=False)
    op.create_foreign_key(
        "fk_htr_authors_entity", "htr_authors", "entities", ["entity_id"], ["id"], ondelete="SET NULL"
    )

    # pre-existing drift: the model declares an index the database never got
    op.create_index(op.f("ix_htr_lexicon_ignores_id"), "htr_lexicon_ignores", ["id"], unique=False)

    # --- vector indexes (after the data, so they are built once) ----------
    hnsw = dict(
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )
    op.create_index("ix_entities_embedding_hnsw", "entities", ["embedding"], unique=False, **hnsw)
    op.create_index("ix_goals_embedding_hnsw", "goals", ["embedding"], unique=False, **hnsw)
    op.create_index("ix_habits_embedding_hnsw", "habits", ["embedding"], unique=False, **hnsw)
    op.create_index("ix_events_embedding_hnsw", "events", ["embedding"], unique=False, **hnsw)
    op.create_index("ix_chunks_embedding_hnsw", "chunks", ["embedding"], unique=False, **hnsw)


def downgrade() -> None:
    """Downgrade schema.

    The data migration is only partly reversible: children get their diary page
    back (from the source), but an edge loses the perspective it was read from,
    because the old table had room for one perspective only.
    """
    op.drop_index("ix_chunks_embedding_hnsw", table_name="chunks")
    op.drop_index("ix_events_embedding_hnsw", table_name="events")
    op.drop_index("ix_habits_embedding_hnsw", table_name="habits")
    op.drop_index("ix_goals_embedding_hnsw", table_name="goals")
    op.drop_index("ix_entities_embedding_hnsw", table_name="entities")

    op.drop_index(op.f("ix_htr_lexicon_ignores_id"), table_name="htr_lexicon_ignores")

    op.drop_constraint("fk_htr_authors_entity", "htr_authors", type_="foreignkey")
    op.drop_index(op.f("ix_htr_authors_entity_id"), table_name="htr_authors")
    op.drop_column("htr_authors", "entity_id")
    op.drop_constraint("fk_habits_subject", "habits", type_="foreignkey")
    op.drop_index(op.f("ix_habits_subject_entity_id"), table_name="habits")
    op.drop_constraint("fk_goals_created_from_source", "goals", type_="foreignkey")
    op.drop_constraint("fk_goals_subject", "goals", type_="foreignkey")
    op.drop_index(op.f("ix_goals_subject_entity_id"), table_name="goals")
    op.drop_constraint("uq_entity_relation_edge", "entity_relations", type_="unique")
    op.drop_constraint("fk_entity_relations_source", "entity_relations", type_="foreignkey")
    op.drop_constraint("fk_entity_relations_to", "entity_relations", type_="foreignkey")
    op.drop_constraint("fk_entity_relations_from", "entity_relations", type_="foreignkey")
    op.drop_index(op.f("ix_entity_relations_source_id"), table_name="entity_relations")
    op.drop_index(op.f("ix_entity_relations_to_entity_id"), table_name="entity_relations")
    op.drop_index(op.f("ix_entity_relations_from_entity_id"), table_name="entity_relations")
    op.drop_constraint("fk_habit_logs_source", "habit_logs", type_="foreignkey")
    op.drop_index(op.f("ix_habit_logs_source_id"), table_name="habit_logs")
    op.drop_constraint("fk_events_subject", "events", type_="foreignkey")
    op.drop_constraint("fk_events_source", "events", type_="foreignkey")
    op.drop_index(op.f("ix_events_subject_entity_id"), table_name="events")
    op.drop_index(op.f("ix_events_source_id"), table_name="events")
    op.drop_constraint("fk_goal_progress_source", "goal_progress", type_="foreignkey")
    op.drop_index(op.f("ix_goal_progress_source_id"), table_name="goal_progress")
    op.drop_constraint("fk_facts_source", "facts", type_="foreignkey")
    op.drop_index(op.f("ix_facts_source_id"), table_name="facts")
    op.drop_constraint("fk_entity_mentions_source", "entity_mentions", type_="foreignkey")
    op.drop_index(op.f("ix_entity_mentions_source_id"), table_name="entity_mentions")

    # rebuild the old references and move the data back
    for table in ("entity_mentions", "facts", "goal_progress", "events", "habit_logs"):
        op.add_column(table, sa.Column("diary_page_id", sa.Integer(), nullable=True))
    op.execute(
        "update entity_mentions c set diary_page_id = s.ref_id "
        "from knowledge_sources s where s.id = c.source_id and s.source_type = 'diary'"
    )
    for table in ("facts", "goal_progress", "events", "habit_logs"):
        op.execute(
            f"update {table} c set diary_page_id = s.ref_id "
            "from knowledge_sources s where s.id = c.source_id and s.source_type = 'diary'"
        )
    op.execute(
        "delete from entity_mentions where diary_page_id is null"
    )
    for table in ("facts", "goal_progress", "events", "habit_logs"):
        op.execute(f"delete from {table} where diary_page_id is null")
    for table in ("entity_mentions", "facts", "goal_progress", "events", "habit_logs"):
        op.alter_column(table, "diary_page_id", existing_type=sa.Integer(), nullable=False)
        op.create_foreign_key(
            f"{table}_diary_page_id_fkey", table, "diary_pages", ["diary_page_id"], ["id"], ondelete="CASCADE"
        )
        op.create_index(op.f(f"ix_{table}_diary_page_id"), table, ["diary_page_id"], unique=False)

    op.add_column("goals", sa.Column("created_from_page_id", sa.Integer(), nullable=True))
    op.execute(
        "update goals g set created_from_page_id = s.ref_id "
        "from knowledge_sources s where s.id = g.created_from_source_id and s.source_type = 'diary'"
    )
    op.create_foreign_key(
        "goals_created_from_page_id_fkey", "goals", "diary_pages", ["created_from_page_id"], ["id"], ondelete="SET NULL"
    )

    op.add_column("entity_relations", sa.Column("entity_id", sa.Integer(), nullable=True))
    op.add_column("entity_relations", sa.Column("last_page_id", sa.Integer(), nullable=True))
    op.execute("delete from entity_relations where from_entity_id is null")
    op.execute(
        "update entity_relations e set entity_id = e.to_entity_id, last_page_id = s.ref_id "
        "from knowledge_sources s where s.id = e.source_id and s.source_type = 'diary'"
    )
    op.execute("update entity_relations set entity_id = to_entity_id where entity_id is null")
    op.execute("delete from entity_relations where entity_id is null")
    op.alter_column("entity_relations", "entity_id", existing_type=sa.Integer(), nullable=False)
    op.create_foreign_key(
        "entity_relations_entity_id_fkey", "entity_relations", "entities", ["entity_id"], ["id"], ondelete="CASCADE"
    )
    op.create_foreign_key(
        "entity_relations_last_page_id_fkey", "entity_relations", "diary_pages", ["last_page_id"], ["id"], ondelete="SET NULL"
    )
    op.create_index(op.f("ix_entity_relations_entity_id"), "entity_relations", ["entity_id"], unique=False)
    op.create_unique_constraint(
        "uq_entity_relation_user_entity", "entity_relations", ["user_id", "entity_id"]
    )
    op.drop_column("entity_relations", "end_offset")
    op.drop_column("entity_relations", "start_offset")
    op.drop_column("entity_relations", "source_id")
    op.drop_column("entity_relations", "to_entity_id")
    op.drop_column("entity_relations", "from_entity_id")

    op.drop_column("goals", "embedding")
    op.drop_column("goals", "created_from_source_id")
    op.drop_column("goals", "subject_entity_id")
    op.drop_column("goal_progress", "source_id")
    op.drop_column("facts", "end_offset")
    op.drop_column("facts", "start_offset")
    op.drop_column("facts", "source_id")
    op.drop_column("events", "embedding")
    op.drop_column("events", "end_offset")
    op.drop_column("events", "start_offset")
    op.drop_column("events", "subject_entity_id")
    op.drop_column("events", "source_id")
    op.drop_column("habits", "embedding")
    op.drop_column("habits", "subject_entity_id")
    op.drop_column("habit_logs", "source_id")
    op.drop_column("entity_mentions", "source_id")
    op.drop_column("entries", "context_json")
    op.drop_column("entities", "updated_at")
    op.drop_column("entities", "embedding")
    op.drop_column("entities", "aliases_json")

    op.drop_constraint("fk_users_self_entity", "users", type_="foreignkey")
    op.drop_constraint("uq_users_self_entity", "users", type_="unique")
    op.drop_column("users", "self_entity_id")

    op.drop_table("knowledge_state")
    op.drop_index(op.f("ix_knowledge_insights_subject_entity_id"), table_name="knowledge_insights")
    op.drop_index(op.f("ix_knowledge_insights_user_id"), table_name="knowledge_insights")
    op.drop_index(op.f("ix_knowledge_insights_id"), table_name="knowledge_insights")
    op.drop_table("knowledge_insights")
    op.drop_index(op.f("ix_chunks_source_id"), table_name="chunks")
    op.drop_index(op.f("ix_chunks_user_id"), table_name="chunks")
    op.drop_index(op.f("ix_chunks_id"), table_name="chunks")
    op.drop_table("chunks")
    op.drop_index(op.f("ix_extraction_runs_status"), table_name="extraction_runs")
    op.drop_index(op.f("ix_extraction_runs_source_id"), table_name="extraction_runs")
    op.drop_index(op.f("ix_extraction_runs_id"), table_name="extraction_runs")
    op.drop_table("extraction_runs")
    op.drop_index(op.f("ix_knowledge_sources_extraction_status"), table_name="knowledge_sources")
    op.drop_index(op.f("ix_knowledge_sources_text_hash"), table_name="knowledge_sources")
    op.drop_index(op.f("ix_knowledge_sources_author_entity_id"), table_name="knowledge_sources")
    op.drop_index(op.f("ix_knowledge_sources_ref_id"), table_name="knowledge_sources")
    op.drop_index(op.f("ix_knowledge_sources_source_type"), table_name="knowledge_sources")
    op.drop_index(op.f("ix_knowledge_sources_user_id"), table_name="knowledge_sources")
    op.drop_index(op.f("ix_knowledge_sources_id"), table_name="knowledge_sources")
    op.drop_table("knowledge_sources")
