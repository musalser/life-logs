"""Соответствие замороженной карты миграции и таблицы видов в коде.

Миграция намеренно держит копию таблицы: правка Δ в ``kinship.py`` не должна
молча менять поведение старой миграции на чистой базе. Этот тест сверяет копию с
кодом; его падение означает, что нужна **новая** миграция, а не правка старой.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

from app.services import kinship

MIGRATION_PATH = (
    Path(__file__).resolve().parents[1]
    / "revisions/versions/5e6f7a8b9c0d_relation_canonical_pair.py"
)


def _load_migration():
    spec = importlib.util.spec_from_file_location("relation_canonical_pair", MIGRATION_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


frozen = _load_migration()


def test_frozen_kin_table_matches_the_code():
    code = {
        kind: (kin.up, kin.down, kin.via_marriage, kin.bond)
        for kind, kin in kinship.KIN_TABLE.items()
    }
    assert frozen.KIN == code


def test_frozen_specials_match_the_code():
    assert frozen.COMPOSITE_TOKENS == kinship.COMPOSITE_TOKENS
    assert frozen.REVERSE_ROLES == kinship._REVERSE_ROLES
    assert frozen.SYMMETRIC_ZERO == kinship._SYMMETRIC_ZERO
    assert frozen.ANCESTOR_LINEAL == kinship._ANCESTOR_LINEAL
    assert frozen.DESCENDANT_LINEAL == kinship._DESCENDANT_LINEAL
    assert frozen.SYMMETRIC_DEEP == kinship._SYMMETRIC_DEEP
    assert (frozen.GENERATIONS_MIN, frozen.GENERATIONS_MAX) == (
        kinship.GENERATIONS_MIN,
        kinship.GENERATIONS_MAX,
    )


def test_every_known_kind_is_covered_by_the_frozen_table():
    """Ни один вид не должен получить kind:… «по недосмотру»."""
    known = (
        set(kinship.KIN_TABLE)
        | set(kinship.COMPOSITE_TOKENS)
        | set(kinship._REVERSE_ROLES)
        | set(kinship.RITUAL_KINDS)
        | set(kinship.AMBIGUOUS_KINDS)
        | set(kinship.NOT_KIN)
    )
    for kind in known:
        if kind in kinship.KIN_TABLE or kind in kinship.COMPOSITE_TOKENS:
            assert kind in frozen.KIN or kind in frozen.COMPOSITE_TOKENS, kind
        token = frozen.frozen_token(kind)
        assert token
    # социальные и ритуальные виды ключуются по виду, и это осознанно
    assert frozen.frozen_token("neighbour") == "kind:neighbour"
    assert frozen.frozen_token("in_law_unknown") == "kind:in_law_unknown"


def test_frozen_tokens_match_the_code_for_every_kind():
    labels = {"other": "кум", None: "сватья"}
    kinds = (
        sorted(kinship.KIN_TABLE)
        + sorted(kinship.COMPOSITE_TOKENS)
        + sorted(kinship._REVERSE_ROLES)
        + sorted(kinship.RITUAL_KINDS)
        + sorted(kinship.AMBIGUOUS_KINDS)
        + sorted(kinship.NOT_KIN)
        + ["other", None, ""]
    )
    for kind in kinds:
        for generations in (0, 4):
            label = labels.get(kind)
            assert frozen.frozen_token(kind, generations, label) == kinship.canon_token(
                kind, generations, relation_type=label
            ), (kind, generations)


def test_frozen_canonical_pair_matches_the_code():
    kinds = sorted(kinship.KIN_TABLE) + sorted(kinship._REVERSE_ROLES) + ["friend", "other", None]
    for kind in kinds:
        for from_id, to_id in ((10, 20), (20, 10)):
            assert frozen.frozen_canon_pair(kind, from_id, to_id) == kinship.canon_pair(
                kind, from_id, to_id
            ), (kind, from_id, to_id)
