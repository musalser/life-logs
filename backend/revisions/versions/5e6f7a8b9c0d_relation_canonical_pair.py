"""каноническая пара рёбер и подтверждения по источникам

Переход к жизненному циклу ребра по подтверждениям:

* ``entity_relations.canon_from`` / ``canon_to`` / ``canon_key`` — ключ
  уникальности по канонической паре, а не по направлению источника. Иначе
  «A → B, отец» и «B → A, отец» (родительский цикл) и «A → B, отец» с
  «A → B, ребёнок» получили бы один ключ, и слабейшее ребро молча исчезло бы —
  ровно то, что обязано попадать в диагностику;
* ``generations``, ``via_marriage``, ``bond`` — флаги вида: глубина для прадеда,
  брак для свойства́, сводные/приёмные/ритуальные связи;
* ``relation_sources`` — по строке на источник: перепрогон одной страницы
  удаляет только её подтверждения, ребро живёт, пока подтверждения есть.

Таблица видов **заморожена здесь**: правка Δ в app/services/kinship.py не должна
молча менять поведение старой миграции на чистой базе. Тест
``tests/test_relation_migration_tokens.py`` сверяет копию с кодом — его падение
означает, что нужна новая миграция, а не правка этой.

Revision ID: 5e6f7a8b9c0d
Revises: 4d5e6f7a8b9c
Create Date: 2026-10-15 18:00:00.000000

"""
import hashlib
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import text


# revision identifiers, used by Alembic.
revision: str = "5e6f7a8b9c0d"
down_revision: Union[str, Sequence[str], None] = "4d5e6f7a8b9c"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# ---------------------------------------------------------------------------
# Замороженная таблица видов (копия app/services/kinship.py на эту дату)
# ---------------------------------------------------------------------------

BOND_BLOOD = "blood"
BOND_HALF = "half"
BOND_STEP = "step"
BOND_ADOPTIVE = "adoptive"

ORIENT_TO_SENIOR = "to_senior"
ORIENT_FROM_SENIOR = "from_senior"
ORIENT_SYM = "sym"
ORIENT_AS_IS = "as_is"

KIN: dict[str, tuple[int, int, bool, str]] = {
    "father": (1, 0, False, BOND_BLOOD), "mother": (1, 0, False, BOND_BLOOD),
    "parent": (1, 0, False, BOND_BLOOD),
    "son": (0, 1, False, BOND_BLOOD), "daughter": (0, 1, False, BOND_BLOOD),
    "child": (0, 1, False, BOND_BLOOD),
    "grandfather": (2, 0, False, BOND_BLOOD), "grandmother": (2, 0, False, BOND_BLOOD),
    "grandparent": (2, 0, False, BOND_BLOOD), "grandchild": (0, 2, False, BOND_BLOOD),
    "great_grandfather": (3, 0, False, BOND_BLOOD),
    "great_grandmother": (3, 0, False, BOND_BLOOD),
    "great_grandparent": (3, 0, False, BOND_BLOOD),
    "great_grandchild": (0, 3, False, BOND_BLOOD),
    "brother": (1, 1, False, BOND_BLOOD), "sister": (1, 1, False, BOND_BLOOD),
    "sibling": (1, 1, False, BOND_BLOOD),
    "half_brother": (1, 1, False, BOND_HALF), "half_sister": (1, 1, False, BOND_HALF),
    "stepbrother": (1, 1, True, BOND_STEP), "stepsister": (1, 1, True, BOND_STEP),
    "uncle": (2, 1, False, BOND_BLOOD), "aunt": (2, 1, False, BOND_BLOOD),
    "nephew": (1, 2, False, BOND_BLOOD), "niece": (1, 2, False, BOND_BLOOD),
    "cousin": (2, 2, False, BOND_BLOOD),
    "husband": (0, 0, True, BOND_BLOOD), "wife": (0, 0, True, BOND_BLOOD),
    "spouse": (0, 0, True, BOND_BLOOD),
    "stepfather": (1, 0, True, BOND_STEP), "stepmother": (1, 0, True, BOND_STEP),
    "stepson": (0, 1, True, BOND_STEP), "stepdaughter": (0, 1, True, BOND_STEP),
    "adoptive_parent": (1, 0, False, BOND_ADOPTIVE),
    "adopted_child": (0, 1, False, BOND_ADOPTIVE),
    "co_parent_in_law": (0, 0, True, BOND_BLOOD), "svojak": (0, 0, True, BOND_BLOOD),
}

COMPOSITE_TOKENS: dict[str, str] = {
    "husband": "pair:spouse", "wife": "pair:spouse", "spouse": "pair:spouse",
    "co_parent_in_law": "pair:co_parent_in_law", "svojak": "pair:svojak",
}

REVERSE_ROLES: dict[str, tuple[str, str]] = {
    "boss": ("role:boss", ORIENT_TO_SENIOR),
    "subordinate": ("role:boss", ORIENT_FROM_SENIOR),
    "teacher": ("role:teacher", ORIENT_TO_SENIOR),
    "student": ("role:teacher", ORIENT_FROM_SENIOR),
    "godfather": ("pair:godparent", ORIENT_TO_SENIOR),
    "godmother": ("pair:godparent", ORIENT_TO_SENIOR),
    "godparent": ("pair:godparent", ORIENT_TO_SENIOR),
    "godchild": ("pair:godparent", ORIENT_FROM_SENIOR),
}

SYMMETRIC_ZERO = frozenset({
    "husband", "wife", "spouse", "brother", "sister", "sibling",
    "half_brother", "half_sister", "stepbrother", "stepsister", "cousin",
    "co_parent_in_law", "svojak",
    "friend", "neighbour", "acquaintance", "classmate", "colleague",
})

ANCESTOR_LINEAL = frozenset({
    "parent", "father", "mother", "grandparent", "grandfather", "grandmother",
    "great_grandparent", "great_grandfather", "great_grandmother",
})
DESCENDANT_LINEAL = frozenset({
    "child", "son", "daughter", "grandchild", "great_grandchild",
})
SYMMETRIC_DEEP = frozenset({"cousin"})

GENERATIONS_MIN = 3
GENERATIONS_MAX = 6


def _generations_ok(value) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    try:
        depth = int(value)
    except (TypeError, ValueError):
        return False
    return GENERATIONS_MIN <= depth <= GENERATIONS_MAX


def _kin_of(kind: str, generations=0) -> tuple[int, int, bool, str] | None:
    base = KIN.get(kind)
    if base is None:
        return None
    depth = int(generations) if _generations_ok(generations) else 0
    if depth > 1:
        up, down, marriage, bond = base
        if kind in ANCESTOR_LINEAL:
            return (depth, 0, marriage, bond)
        if kind in DESCENDANT_LINEAL:
            return (0, depth, marriage, bond)
        if kind in SYMMETRIC_DEEP:
            return (depth, depth, marriage, bond)
    return base


def frozen_token(kind: str | None, generations=0, relation_type: str | None = None) -> str:
    """Токен канонической формы — та же формула, что в app/services/kinship.py."""
    kind = (kind or "").strip().lower()
    if kind in COMPOSITE_TOKENS:
        return COMPOSITE_TOKENS[kind]
    reverse = REVERSE_ROLES.get(kind)
    if reverse is not None:
        return reverse[0]
    kin = _kin_of(kind, generations)
    if kin is None:
        if kind in ("", "other"):
            label = " ".join((relation_type or "").split()).lower()
            return f"kind:other:{hashlib.md5(label.encode('utf-8')).hexdigest()[:12]}"
        return f"kind:{kind}"
    up, down, marriage, bond = kin
    low, high = sorted((up, down))
    return f"shape:{low}:{high}:{int(marriage)}:{bond}"


def frozen_canon_pair(kind: str | None, from_id: int, to_id: int, generations=0) -> tuple[int, int]:
    """Каноническая пара: старший вперёд, симметричное по id, остальное как есть."""
    kind = (kind or "").strip().lower()
    reverse = REVERSE_ROLES.get(kind)
    if reverse is not None:
        orientation = reverse[1]
    else:
        kin = _kin_of(kind, generations)
        if kin is None:
            orientation = ORIENT_AS_IS
        elif kin[1] - kin[0] < 0:
            orientation = ORIENT_TO_SENIOR
        elif kin[1] - kin[0] > 0:
            orientation = ORIENT_FROM_SENIOR
        else:
            orientation = ORIENT_SYM if kind in SYMMETRIC_ZERO else ORIENT_AS_IS
    if orientation == ORIENT_TO_SENIOR:
        return (to_id, from_id)
    if orientation in (ORIENT_FROM_SENIOR, ORIENT_AS_IS):
        return (from_id, to_id)
    return (from_id, to_id) if from_id <= to_id else (to_id, from_id)


def _backfill_canonical(bind) -> int:
    rows = bind.execute(
        text(
            "select id, from_entity_id, to_entity_id, relation_kind, relation_type, generations "
            "from entity_relations order by id"
        )
    ).all()
    for row in rows:
        canon_from, canon_to = frozen_canon_pair(
            row.relation_kind, row.from_entity_id, row.to_entity_id, row.generations
        )
        bind.execute(
            text(
                "update entity_relations set canon_from = :cf, canon_to = :ct, canon_key = :key "
                "where id = :edge_id"
            ),
            {
                "cf": canon_from,
                "ct": canon_to,
                "key": frozen_token(row.relation_kind, row.generations, row.relation_type),
                "edge_id": row.id,
            },
        )
    return len(rows)


def _merge_unconfirmed(bind) -> int:
    """Склеивает рёбра, совпавшие по каноническому ключу, до создания ограничения.

    Подтверждения «проигравших» переезжают к победителю: иначе вместе с ребром
    пропал бы и источник, который его подтверждал.
    """
    bind.execute(
        text(
            """
            WITH ranked AS (
                SELECT id,
                       first_value(id) OVER w AS winner_id,
                       row_number()     OVER w AS rn
                  FROM entity_relations
                WINDOW w AS (PARTITION BY user_id, canon_from, canon_to, canon_key
                             ORDER BY coalesce(confidence, 0) DESC, id)
            ), mapping AS (
                SELECT id AS loser_id, winner_id FROM ranked WHERE rn > 1
            )
            UPDATE relation_sources rs
               SET relation_id = m.winner_id
              FROM mapping m
             WHERE rs.relation_id = m.loser_id
               AND NOT EXISTS (
                     SELECT 1 FROM relation_sources w
                      WHERE w.relation_id = m.winner_id AND w.source_id = rs.source_id)
            """
        )
    )
    result = bind.execute(
        text(
            """
            WITH ranked AS (
                SELECT id, row_number() OVER (
                           PARTITION BY user_id, canon_from, canon_to, canon_key
                           ORDER BY coalesce(confidence, 0) DESC, id) AS rn
                  FROM entity_relations
            )
            DELETE FROM entity_relations WHERE id IN (SELECT id FROM ranked WHERE rn > 1)
            """
        )
    )
    return result.rowcount or 0


def upgrade() -> None:
    """Upgrade schema."""
    bind = op.get_bind()

    op.add_column("entity_relations", sa.Column("generations", sa.Integer(), nullable=True))
    op.add_column("entity_relations", sa.Column("via_marriage", sa.Boolean(), nullable=True))
    op.add_column("entity_relations", sa.Column("bond", sa.String(length=16), nullable=True))
    op.add_column("entity_relations", sa.Column("canon_from", sa.Integer(), nullable=True))
    op.add_column("entity_relations", sa.Column("canon_to", sa.Integer(), nullable=True))
    # отдельных индексов по canon_* нет: селективность низкая, а канонические
    # поля ищутся вместе с user_id — это префикс уникального ограничения
    op.add_column("entity_relations", sa.Column("canon_key", sa.String(length=64), nullable=True))

    converted = _backfill_canonical(bind)
    print(f"[relation canonical pair] canonical pair filled for {converted} edges")

    op.alter_column("entity_relations", "canon_from", nullable=False)
    op.alter_column("entity_relations", "canon_to", nullable=False)
    op.alter_column("entity_relations", "canon_key", nullable=False)

    op.create_table(
        "relation_sources",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("relation_id", sa.Integer(), nullable=False),
        sa.Column("source_id", sa.Integer(), nullable=False),
        sa.Column("relation_kind", sa.String(length=32), nullable=True),
        sa.Column("relation_type", sa.String(length=64), nullable=True),
        sa.Column("generations", sa.Integer(), nullable=True),
        sa.Column("via_marriage", sa.Boolean(), nullable=True),
        sa.Column("bond", sa.String(length=16), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("evidence_text", sa.Text(), nullable=True),
        sa.Column("start_offset", sa.Integer(), nullable=True),
        sa.Column("end_offset", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=True),
        sa.ForeignKeyConstraint(["relation_id"], ["entity_relations.id"], name="fk_relation_sources_relation", ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["source_id"], ["knowledge_sources.id"], name="fk_relation_sources_source", ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("relation_id", "source_id", name="uq_relation_source_relation_source"),
    )
    op.create_index(op.f("ix_relation_sources_id"), "relation_sources", ["id"], unique=False)
    op.create_index(op.f("ix_relation_sources_relation_id"), "relation_sources", ["relation_id"], unique=False)
    op.create_index(op.f("ix_relation_sources_source_id"), "relation_sources", ["source_id"], unique=False)

    confirmations = bind.execute(
        text(
            "insert into relation_sources (relation_id, source_id, relation_kind, relation_type, "
            "generations, via_marriage, bond, confidence, evidence_text, start_offset, end_offset) "
            "select id, source_id, relation_kind, relation_type, generations, via_marriage, bond, "
            "       confidence, evidence_text, start_offset, end_offset "
            "  from entity_relations where source_id is not null "
            "on conflict (relation_id, source_id) do nothing"
        )
    ).rowcount
    print(f"[relation canonical pair] relation_sources filled with {confirmations} confirmations")

    merged = _merge_unconfirmed(bind)
    print(f"[relation canonical pair] merged duplicates: {merged}")

    op.drop_constraint("uq_entity_relation_edge", "entity_relations", type_="unique")
    op.create_unique_constraint(
        "uq_entity_relation_canonical",
        "entity_relations",
        ["user_id", "canon_from", "canon_to", "canon_key"],
    )
    op.create_check_constraint(
        "ck_entity_relations_bond",
        "entity_relations",
        "bond is null or bond in ('blood','half','step','adoptive','ritual')",
    )
    op.create_check_constraint(
        "ck_entity_relations_generations",
        "entity_relations",
        "generations is null or (generations between 3 and 6)",
    )


def downgrade() -> None:
    """Downgrade schema.

    Оговорки: удалённые при слиянии «проигравшие» рёбра не возвращаются, а старое
    ограничение по ``relation_type`` может не встать, если после апгрейда
    накопились дубли, которых оно не допускает.
    """
    op.drop_constraint("ck_entity_relations_generations", "entity_relations", type_="check")
    op.drop_constraint("ck_entity_relations_bond", "entity_relations", type_="check")
    op.drop_constraint("uq_entity_relation_canonical", "entity_relations", type_="unique")
    op.create_unique_constraint(
        "uq_entity_relation_edge",
        "entity_relations",
        ["user_id", "from_entity_id", "to_entity_id", "relation_type"],
    )

    op.drop_index(op.f("ix_relation_sources_source_id"), table_name="relation_sources")
    op.drop_index(op.f("ix_relation_sources_relation_id"), table_name="relation_sources")
    op.drop_index(op.f("ix_relation_sources_id"), table_name="relation_sources")
    op.drop_table("relation_sources")

    op.drop_column("entity_relations", "canon_key")
    op.drop_column("entity_relations", "canon_to")
    op.drop_column("entity_relations", "canon_from")
    op.drop_column("entity_relations", "bond")
    op.drop_column("entity_relations", "via_marriage")
    op.drop_column("entity_relations", "generations")
