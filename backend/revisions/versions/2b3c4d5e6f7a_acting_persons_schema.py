"""структурная схема действующих лиц: якорные сущности и канонический вид отношений

Переход от словарных правил к схеме, которую понимает модель:

* ``entities.anchor_entity_id`` + ``entities.anchor_relation`` — безымянный
  родственник («мой отец») живёт в графе как объект с идентичностью
  (автор, father) и отображаемым именем «Отец автора»;
* ``entity_relations.relation_kind`` — канонический ключ закрытого словаря
  (father, mother, …) рядом с человеческой подписью ``relation_type``;
* ``facts.subject_source`` — откуда взят субъект (self/named/relative/graph),
  чтобы доля self была метрикой дрейфа.

Данные: у уже сохранённых рёбер ``relation_kind`` выводится из подписи
замороженной картой. Якорных сущностей до этой миграции не существовало
(безымянных родственников отсеивали), поэтому их переносить нечего.

Revision ID: 2b3c4d5e6f7a
Revises: 1a2b3c4d5e6f
Create Date: 2026-10-15 14:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import text


# revision identifiers, used by Alembic.
revision: str = "2b3c4d5e6f7a"
down_revision: Union[str, Sequence[str], None] = "1a2b3c4d5e6f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


#: замороженная копия разбора подписей (app/services/relation_kinds.py) — так
#: миграция не меняется, когда словарь в коде дополняют
_SURFACE_TO_KIND: dict[str, str] = {
    "отец": "father", "папа": "father", "папаня": "father", "батя": "father",
    "тятя": "father", "папка": "father", "father": "father", "dad": "father",
    "мать": "mother", "мама": "mother", "мамка": "mother", "матерь": "mother",
    "mother": "mother", "mom": "mother", "mum": "mother",
    "родитель": "parent", "родители": "parent",
    "сын": "son", "сынок": "son", "son": "son",
    "дочь": "daughter", "дочка": "daughter", "daughter": "daughter",
    "ребёнок": "child", "ребенок": "child", "child": "child",
    "брат": "brother", "братик": "brother", "brother": "brother",
    "сестра": "sister", "сестрёнка": "sister", "сестренка": "sister",
    "sister": "sister",
    "муж": "husband", "супруг": "husband", "husband": "husband",
    "жена": "wife", "супруга": "wife", "wife": "wife",
    "дед": "grandfather", "дедушка": "grandfather", "grandfather": "grandfather",
    "баба": "grandmother", "бабушка": "grandmother", "grandmother": "grandmother",
    "прадед": "great_grandfather", "прабабушка": "great_grandmother",
    "внук": "grandchild", "внучка": "grandchild", "grandchild": "grandchild",
    "дядя": "uncle", "uncle": "uncle",
    "тётя": "aunt", "тетя": "aunt", "aunt": "aunt",
    "племянник": "nephew", "племянница": "niece",
    "двоюродный": "cousin", "двоюродная": "cousin", "cousin": "cousin",
    "тесть": "father_in_law", "свёкор": "father_in_law", "свекор": "father_in_law",
    "тёща": "mother_in_law", "теща": "mother_in_law", "свекровь": "mother_in_law",
    "зять": "son_in_law", "невестка": "daughter_in_law", "сноха": "daughter_in_law",
    "отчим": "stepfather", "мачеха": "stepmother",
    "пасынок": "stepson", "падчерица": "stepdaughter",
    "друг": "friend", "подруга": "friend", "friend": "friend",
    "коллега": "colleague", "сослуживец": "colleague", "colleague": "colleague",
    "начальник": "boss", "шеф": "boss", "boss": "boss",
    "подчинённый": "subordinate", "подчиненный": "subordinate",
    "сосед": "neighbour", "соседка": "neighbour", "neighbour": "neighbour",
    "знакомый": "acquaintance", "знакомая": "acquaintance",
    "одноклассник": "classmate", "одноклассница": "classmate",
    "учитель": "teacher", "учительница": "teacher", "teacher": "teacher",
    "ученик": "student", "ученица": "student",
    "врач": "doctor", "доктор": "doctor", "doctor": "doctor",
    "тренер": "trainer", "trainer": "trainer",
    "крёстный": "godparent", "крестный": "godparent",
    "крёстная": "godparent", "крестная": "godparent",
    "крестник": "godchild", "крестница": "godchild",
}


def _kind(value: str | None) -> str:
    text = " ".join((value or "").split()).strip().lower()
    if not text:
        return "other"
    if text in _SURFACE_TO_KIND:
        return _SURFACE_TO_KIND[text]
    for word in text.replace("-", " ").split():
        if word in _SURFACE_TO_KIND:
            return _SURFACE_TO_KIND[word]
    return "other"


def _backfill_relation_kinds(bind) -> int:
    rows = bind.execute(
        text("select id, relation_type from entity_relations order by id")
    ).all()
    for edge_id, relation_type in rows:
        bind.execute(
            text("update entity_relations set relation_kind = :kind where id = :edge_id"),
            {"kind": _kind(relation_type), "edge_id": edge_id},
        )
    return len(rows)


def upgrade() -> None:
    """Upgrade schema."""
    bind = op.get_bind()

    op.add_column("entities", sa.Column("anchor_entity_id", sa.Integer(), nullable=True))
    op.add_column("entities", sa.Column("anchor_relation", sa.String(length=32), nullable=True))
    op.create_index(op.f("ix_entities_anchor_entity_id"), "entities", ["anchor_entity_id"], unique=False)
    op.create_foreign_key(
        "fk_entities_anchor", "entities", "entities", ["anchor_entity_id"], ["id"], ondelete="SET NULL"
    )

    op.add_column("entity_relations", sa.Column("relation_kind", sa.String(length=32), nullable=True))
    op.create_index(
        op.f("ix_entity_relations_relation_kind"), "entity_relations", ["relation_kind"], unique=False
    )

    op.add_column("facts", sa.Column("subject_source", sa.String(length=16), nullable=True))
    op.create_index(op.f("ix_facts_subject_source"), "facts", ["subject_source"], unique=False)

    converted = _backfill_relation_kinds(bind)
    print(f"[acting persons migration] relation_kind backfilled for {converted} edges")


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f("ix_facts_subject_source"), table_name="facts")
    op.drop_column("facts", "subject_source")

    op.drop_index(op.f("ix_entity_relations_relation_kind"), table_name="entity_relations")
    op.drop_column("entity_relations", "relation_kind")

    op.drop_constraint("fk_entities_anchor", "entities", type_="foreignkey")
    op.drop_index(op.f("ix_entities_anchor_entity_id"), table_name="entities")
    op.drop_column("entities", "anchor_relation")
    op.drop_column("entities", "anchor_entity_id")
