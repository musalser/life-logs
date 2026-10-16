"""Тесты структурного слоя родства: пары (up, down), канонизация, уровни.

Модуль чистый — ни БД, ни Vue. Проверяются ровно те свойства, ради которых он
и вводился: знак не приходит от модели, свойство́ считается флагом брака, ключ
уникальности не склеивает разные факты и не пропускает дубли, а неоднозначные и
ритуальные связи не получают нулевой уровень по умолчанию.
"""
from __future__ import annotations

import random

import pytest

from app.services.kinship import (
    aggregate_confirmations,
    canonicalize_relations,
    canon_pair,
    canon_token,
    canonicalize_kin_edges,
    generations_ok,
    is_kin,
    kin_of,
    levels,
    other_label_hash,
    pair_key,
)


def edge(
    edge_id, frm, to, kind, *, confidence=0.9, sources=(1,), generations=0,
    via_marriage=False, bond=None, relation_type=None,
):
    """Ребро читается как «B — это вид A»: ``edge(1, 10, 20, "father")`` — 20 отец 10."""
    return {
        "id": edge_id,
        "from": frm,
        "to": to,
        "relation_kind": kind,
        "relation_type": relation_type,
        "generations": generations,
        "via_marriage": via_marriage,
        "bond": bond,
        "confidence": confidence,
        "sources": list(sources),
    }


# ---------------------------------------------------------------------------
# Таблица (up, down): знакового шага на границе с моделью нет
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kind,up,down,step",
    [
        ("father", 1, 0, -1),
        ("mother", 1, 0, -1),
        ("parent", 1, 0, -1),
        ("grandfather", 2, 0, -2),
        ("grandmother", 2, 0, -2),
        ("grandparent", 2, 0, -2),
        ("grandchild", 0, 2, 2),
        ("great_grandchild", 0, 3, 3),
        ("brother", 1, 1, 0),
        ("sister", 1, 1, 0),
        ("cousin", 2, 2, 0),
        ("uncle", 2, 1, -1),
        ("aunt", 2, 1, -1),
        ("nephew", 1, 2, 1),
        ("niece", 1, 2, 1),
    ],
)
def test_blood_pairs(kind, up, down, step):
    kin = kin_of(kind)
    assert kin is not None
    assert (kin.up, kin.down) == (up, down)
    assert kin.step == step
    assert kin.via_marriage is False
    assert kin.bond == "blood"


@pytest.mark.parametrize("kind", ["husband", "wife", "spouse"])
def test_spouse_is_the_same_level_through_marriage(kind):
    kin = kin_of(kind)
    assert kin is not None
    assert (kin.up, kin.down) == (0, 0)
    assert kin.step == 0
    assert kin.via_marriage is True


@pytest.mark.parametrize("kind,step", [("parent", -1), ("child", 1), ("sibling", 0)])
def test_in_law_is_the_blood_part_plus_a_flag(kind, step):
    """Свойство́ — флаг брака, а не отдельные виды parent_in_law/child_in_law."""
    kin = kin_of(kind, via_marriage=True)
    assert kin is not None
    assert kin.step == step
    assert kin.via_marriage is True


@pytest.mark.parametrize("kind", ["co_parent_in_law", "svojak"])
def test_two_marriage_links_are_their_own_kinds(kind):
    """Сват/сватья и свояк в пару (up, down) не ложатся — отдельные виды, шаг 0."""
    kin = kin_of(kind)
    assert kin is not None
    assert kin.step == 0
    assert kin.via_marriage is True


def test_half_sibling_differs_from_a_full_one_by_the_bond():
    assert kin_of("brother").bond == "blood"
    assert kin_of("half_brother").bond == "half"


def test_step_and_adoptive_stay_in_layers():
    """У отчима и пасынка поколение надёжное — они не «уровень не определён»."""
    assert kin_of("stepfather").step == -1
    assert kin_of("stepfather").bond == "step"
    assert kin_of("stepson").step == 1
    assert kin_of("adoptive_parent").step == -1
    assert kin_of("adopted_child").bond == "adoptive"


def test_ritual_and_ambiguous_have_no_step():
    for kind in ("godfather", "godmother", "godchild", "co_godparent", "in_law_unknown"):
        assert kin_of(kind) is None
        assert is_kin(kind) is True
    assert kin_of("friend") is None
    assert is_kin("friend") is False


@pytest.mark.parametrize("value,valid", [(3, True), (4, True), (6, True), (0, False), (2, False), (40, False), ("x", False), (None, False)])
def test_generations_domain(value, valid):
    assert generations_ok(value) is valid


def test_deep_lineal_uses_generations_and_rejects_nonsense():
    """«Прадед» — неотрицательное число поколений; 0 и 40 не принимаются."""
    kin = kin_of("great_grandfather", generations=4)
    assert (kin.up, kin.down) == (4, 0)
    assert kin.step == -4
    descendant = kin_of("great_grandchild", generations=5)
    assert (descendant.up, descendant.down) == (0, 5)
    deep_cousin = kin_of("cousin", generations=3)
    assert (deep_cousin.up, deep_cousin.down) == (3, 3)
    # вне диапазона — берём таблицу, а не «40 поколений»
    assert kin_of("great_grandfather", generations=40).step == -3
    assert kin_of("great_grandfather", generations=0).step == -3


# ---------------------------------------------------------------------------
# Ключ уникальности
# ---------------------------------------------------------------------------


def test_canon_token_shapes_and_specials():
    assert canon_token("father") == "shape:0:1:0:blood"
    assert canon_token("child") == "shape:0:1:0:blood"
    assert canon_token("parent", via_marriage=True) == "shape:0:1:1:blood"
    assert canon_token("half_brother") == "shape:1:1:0:half"
    assert canon_token("stepfather") == "shape:0:1:1:step"
    assert canon_token("adoptive_parent") == "shape:0:1:0:adoptive"
    assert canon_token("brother") == "shape:1:1:0:blood"
    assert canon_token("uncle") == "shape:1:2:0:blood"
    assert canon_token("godfather") == "pair:godparent"
    assert canon_token("in_law_unknown") == "kind:in_law_unknown"
    assert canon_token("neighbour") == "kind:neighbour"
    assert canon_token("colleague") == "kind:colleague"
    assert canon_token("neighbour") != canon_token("colleague")


def test_composite_tokens_are_distinct_from_spouse():
    assert canon_token("husband") == canon_token("wife") == canon_token("spouse") == "pair:spouse"
    assert canon_token("co_parent_in_law") == "pair:co_parent_in_law"
    assert canon_token("svojak") == "pair:svojak"
    assert len({canon_token("spouse"), canon_token("co_parent_in_law"), canon_token("svojak")}) == 3


def test_other_relies_on_the_hashed_label():
    """Разные свободные подписи под «other» — разные факты, а не один."""
    uncle_like = canon_token("other", relation_type="кум")
    aunt_like = canon_token("other", relation_type="сватья")
    assert uncle_like.startswith("kind:other:")
    assert uncle_like != aunt_like
    assert canon_token(None, relation_type="кум") == uncle_like
    assert other_label_hash("  Кум ") == other_label_hash("кум")


@pytest.mark.parametrize(
    "kind,frm,to,expected",
    [
        ("father", 10, 20, (20, 10)),      # старший — отец
        ("child", 10, 20, (10, 20)),       # старший — автор
        ("uncle", 10, 20, (20, 10)),       # дядя старше
        ("nephew", 10, 20, (10, 20)),      # племянник младше
        ("husband", 10, 20, (10, 20)),     # симметричное — по id
        ("husband", 20, 10, (10, 20)),
        ("doctor", 10, 20, (10, 20)),      # несимметричное — как у источника
        ("doctor", 20, 10, (20, 10)),
    ],
)
def test_canon_pair_orientation(kind, frm, to, expected):
    assert canon_pair(kind, frm, to) == expected


def test_pair_key_ignores_direction():
    assert pair_key(20, 10) == pair_key(10, 20) == (10, 20)


def test_composite_kinds_do_not_collide_in_canonicalization():
    """Супруг, сват и свояк структурно (0, 0) + брак, но это три разных факта."""
    canonical = canonicalize_kin_edges(
        [
            edge(1, 10, 20, "spouse"),
            edge(2, 10, 20, "co_parent_in_law"),
            edge(3, 10, 20, "svojak"),
        ]
    )
    assert len(canonical) == 3
    assert {c.kind for c in canonical} == {"spouse", "co_parent_in_law", "svojak"}


# ---------------------------------------------------------------------------
# Канонизация рёбер
# ---------------------------------------------------------------------------


def test_father_and_child_from_two_authors_are_one_edge():
    """«A → B, отец» и «B → A, ребёнок» — один факт, а не две линии."""
    canonical = canonicalize_kin_edges(
        [
            edge(1, 10, 20, "father", confidence=0.9, sources=(1,)),
            edge(2, 20, 10, "child", confidence=0.7, sources=(2,)),
        ]
    )
    assert len(canonical) == 1
    result = canonical[0]
    assert result.sources == (1, 2)
    assert result.confidence == 0.9
    assert result.delta == -1  # «20 — отец 10»


def test_gender_of_the_parent_survives_canonicalization():
    """Пол не теряется: представителем остаётся «мать», а не обезличенный «ребёнок»."""
    canonical = canonicalize_kin_edges(
        [
            edge(1, 20, 10, "daughter", confidence=0.6),  # 10 — дочь 20
            edge(2, 10, 20, "mother", confidence=0.9),    # 20 — мать 10
            edge(3, 20, 10, "child", confidence=0.5),     # 10 — ребёнок 20
        ]
    )
    assert len(canonical) == 1
    # представителем остаётся самое уверенное ребро, поэтому пол родителя цел
    assert canonical[0].kind == "mother"
    kin = kin_of(canonical[0].kind)
    assert (kin.up, kin.down) == (1, 0)
    assert canonical[0].from_id == 10 and canonical[0].to_id == 20
    # все три точки зрения описывают одну пару «20 старше 10»
    assert canon_pair("mother", 10, 20) == canon_pair("child", 20, 10) == (20, 10)


def test_symmetric_kinds_collapse_in_both_directions():
    """«A → B, муж» и «B → A, жена» — один факт: пара без направления."""
    canonical = canonicalize_kin_edges(
        [edge(1, 10, 20, "husband"), edge(2, 20, 10, "wife")]
    )
    assert len(canonical) == 1
    assert canonical[0].kind in {"husband", "wife"}


def test_full_and_half_siblings_are_two_facts():
    canonical = canonicalize_kin_edges(
        [edge(1, 10, 20, "brother"), edge(2, 10, 20, "half_brother")]
    )
    assert len(canonical) == 2
    assert {c.bond for c in canonical} == {"blood", "half"}


def test_blood_parent_and_in_law_parent_stay_apart():
    canonical = canonicalize_kin_edges(
        [edge(1, 10, 20, "parent"), edge(2, 10, 20, "parent", via_marriage=True)]
    )
    assert len(canonical) == 2
    assert {c.via_marriage for c in canonical} == {False, True}


def test_step_father_and_father_are_two_facts():
    canonical = canonicalize_kin_edges(
        [edge(1, 10, 20, "father"), edge(2, 10, 30, "stepfather")]
    )
    assert len(canonical) == 2
    assert {c.bond for c in canonical} == {"blood", "step"}


def test_social_edges_are_not_kinship():
    assert canonicalize_kin_edges([edge(1, 10, 20, "neighbour"), edge(2, 10, 30, "friend")]) == []


def test_parent_cycle_is_two_facts_and_a_conflict():
    """«A → B, отец» и «B → A, отец» — разные факты: это цикл, а не дубль.

    Пара без направления склеила бы их и молча удалила слабейшее ребро — ровно
    тот случай, который обязан попасть в диагностику.
    """
    edges = [
        edge(1, 10, 20, "father", confidence=0.9),   # 20 — отец 10
        edge(2, 20, 10, "father", confidence=0.4),   # 10 — отец 20
    ]
    canonical = canonicalize_kin_edges(edges)
    assert len(canonical) == 2
    assert canon_pair("father", 10, 20) == (20, 10)
    assert canon_pair("father", 20, 10) == (10, 20)
    result = levels([10, 20], edges)
    assert len([c for c in result.conflicts if c["kind"] == "level_mismatch"]) == 1


def test_same_pair_father_and_child_are_two_facts():
    """«A → B, отец» и «A → B, ребёнок» — противоречие, а не один факт."""
    edges = [
        edge(1, 10, 20, "father", confidence=0.9),
        edge(2, 10, 20, "child", confidence=0.4),
    ]
    canonical = canonicalize_kin_edges(edges)
    assert len(canonical) == 2
    assert {c.kind for c in canonical} == {"father", "child"}
    result = levels([10, 20], edges)
    assert len([c for c in result.conflicts if c["kind"] == "level_mismatch"]) == 1


@pytest.mark.parametrize(
    "forward,backward,token",
    [
        ("boss", "subordinate", "role:boss"),
        ("teacher", "student", "role:teacher"),
        ("godfather", "godchild", "pair:godparent"),
    ],
)
def test_reverse_roles_collapse_to_one_edge(forward, backward, token):
    """Начальник/подчинённый, учитель/ученик, крёстный/крестник — один факт.

    Без обратной пары это были бы две линии между теми же людьми и двойная
    степень у обоих.
    """
    assert canon_token(forward) == canon_token(backward) == token
    canonical = canonicalize_relations(
        [edge(1, 10, 20, forward), edge(2, 20, 10, backward)]
    )
    assert len(canonical) == 1
    assert canon_pair(forward, 10, 20) == canon_pair(backward, 20, 10)


def test_deep_degree_tokens_keep_the_depth():
    """Прадед и прапрадед одной пары — разные факты, а не одна строка."""
    assert canon_token("great_grandfather", generations=3) == "shape:0:3:0:blood"
    assert canon_token("great_grandfather", generations=4) == "shape:0:4:0:blood"
    assert canon_token("great_grandchild", generations=4) == "shape:0:4:0:blood"
    assert canon_token("cousin", generations=3) == "shape:3:3:0:blood"
    assert canon_token("great_grandfather", generations=4) != canon_token(
        "great_grandfather", generations=5
    )


def test_bond_comes_from_the_kind_and_overrides_the_model():
    """Отчим — это step по виду; модель не может назвать его кровным."""
    assert kin_of("stepfather", bond="blood").bond == "step"
    assert kin_of("half_brother", bond="blood").bond == "half"
    assert kin_of("adopted_child", bond="blood").bond == "adoptive"
    # а брак кровный вид не подразумевает: «тесть» — это parent + брак
    assert kin_of("parent", via_marriage=True).via_marriage is True


def test_kind_disagreement_is_reported():
    """Один факт, названный «дядей» и «тётей», не должен молча терять вид."""
    edges = [
        edge(1, 10, 20, "uncle", confidence=0.9),
        edge(2, 10, 20, "aunt", confidence=0.5),
    ]
    canonical = canonicalize_kin_edges(edges)
    assert len(canonical) == 1
    assert set(canonical[0].kinds) == {"uncle", "aunt"}
    result = levels([10, 20], edges)
    disagreement = [c for c in result.conflicts if c["kind"] == "kind_disagreement"]
    assert len(disagreement) == 1
    assert set(disagreement[0]["kinds"]) == {"uncle", "aunt"}


def test_aggregate_confirmations_picks_the_best_and_keeps_all_sources():
    """Строка ребра отражает оставшиеся подтверждения, а не первое из них."""
    rows = [
        edge(1, 10, 20, "father", confidence=0.4, sources=(7,)),
        edge(2, 10, 20, "father", confidence=0.95, sources=(8,), relation_type="папа"),
    ]
    aggregated = aggregate_confirmations(rows)
    assert aggregated["confidence"] == 0.95
    assert aggregated["relation_type"] == "папа"
    assert aggregated["sources"] == [7, 8]
    assert aggregate_confirmations([]) is None


def test_asymmetric_social_reverse_is_not_the_same_fact():
    """«A → B, врач» и «B → A, врач» — разные факты, а не один."""
    canonical = canonicalize_kin_edges(
        [edge(1, 10, 20, "doctor"), edge(2, 20, 10, "doctor")]
    )
    assert canonical == []  # врач — не родня, в слои не идёт
    first = canon_token("doctor")
    assert canon_pair("doctor", 10, 20) != canon_pair("doctor", 20, 10)
    assert first == "kind:doctor"


# ---------------------------------------------------------------------------
# Уровни: согласованная система, а не BFS от корня
# ---------------------------------------------------------------------------


def test_grandfather_without_the_intermediate_parent():
    """Дед и отец: разница −2 и −1 от человека 10, промежуточное звено не нужно."""
    edges = [
        edge(1, 10, 20, "grandfather"),   # 20 — дед человека 10
        edge(2, 10, 30, "father"),        # 30 — отец человека 10
    ]
    result = levels([10, 20, 30], edges)
    assert result.conflicts == []
    assert result.levels[20] + 2 == result.levels[10]
    assert result.levels[30] + 1 == result.levels[10]
    assert result.components[10] == result.components[20] == result.components[30]


def _levels_of(edges, nodes=(10, 20, 30, 40)):
    result = levels(list(nodes), edges)
    return {node: result.levels[node] for node in nodes}


def test_levels_do_not_depend_on_root_or_edge_order():
    edges = [
        edge(1, 10, 20, "father"),
        edge(2, 10, 30, "father"),
        edge(3, 20, 40, "brother"),
        edge(4, 30, 20, "wife"),
    ]
    first = _levels_of(edges)
    assert _levels_of(list(reversed(edges))) == first
    reordered = levels([40, 30, 20, 10], edges)
    assert {n: reordered.levels[n] for n in (10, 20, 30, 40)} == first


def test_random_consistent_trees_have_no_level_conflicts():
    """Свойство: на случайных согласованных деревьях конфликтов уровней нет."""
    rng = random.Random(20261015)
    nodes = list(range(1, 15))
    for _attempt in range(25):
        edges: list[dict] = []
        tree_edges: list[tuple[int, int, int]] = []  # (родитель, ребёнок, шаг)
        for index, node in enumerate(nodes[1:], start=1):
            parent = nodes[rng.randrange(0, index)]
            step = rng.choice([1, 1, 1, 2])
            tree_edges.append((parent, node, step))
        reference: dict[int, int] = {nodes[0]: 0}
        for edge_id, (parent, child, step) in enumerate(tree_edges, start=1):
            # ребро читается «B — это вид A», поэтому уровень ребёнка всегда ниже
            # родителя, независимо от того, с чьей стороны записано ребро
            if rng.random() < 0.5:
                kind = {1: "child", 2: "grandchild"}[step]
                edges.append(edge(edge_id, parent, child, kind, confidence=rng.random()))
            else:
                kind = {1: "father", 2: "grandfather"}[step]
                edges.append(edge(edge_id, child, parent, kind, confidence=rng.random()))
            reference[child] = reference[parent] + step
        result = levels(nodes, edges)
        assert [c for c in result.conflicts if c["kind"] == "level_mismatch"] == []
        offset = result.levels[nodes[0]] - reference[nodes[0]]
        for node in nodes:
            assert result.levels[node] == reference[node] + offset


def test_only_the_corrupted_low_confidence_edge_is_flagged():
    edges = [
        edge(1, 10, 20, "father", confidence=0.9),
        edge(2, 20, 30, "father", confidence=0.8),
        # 10 и 30 на одном уровне, а ребро утверждает шаг 1 — и оно самое слабое
        edge(3, 10, 30, "child", confidence=0.05),
    ]
    result = levels([10, 20, 30], edges)
    mismatches = [c for c in result.conflicts if c["kind"] == "level_mismatch"]
    assert len(mismatches) == 1
    assert mismatches[0]["edge_id"] == 3


def test_in_law_edge_does_not_create_a_false_conflict():
    """«Жена → её отец» — это −1, а не 0: ложного конфликта быть не должно."""
    edges = [
        edge(1, 10, 30, "wife"),
        edge(2, 30, 20, "parent", via_marriage=True, confidence=0.8),
    ]
    result = levels([10, 20, 30], edges)
    assert [c for c in result.conflicts if c["kind"] == "level_mismatch"] == []
    assert result.levels[20] + 1 == result.levels[10]


def test_ritual_edge_does_not_connect_and_gives_no_level():
    result = levels([10, 20], [edge(1, 10, 20, "godfather")])
    assert result.levels[10] is None
    assert result.levels[20] is None
    assert {item["entity_id"]: item["reason"] for item in result.unplaced} == {
        10: "ritual",
        20: "ritual",
    }


def test_ambiguous_in_law_is_unplaced_not_zero():
    result = levels([10, 20], [edge(1, 10, 20, "in_law_unknown")])
    assert result.levels[20] is None
    assert result.unplaced[0]["reason"] == "ambiguous"


def test_ritual_node_keeps_its_level_from_other_edges():
    edges = [
        edge(1, 10, 20, "father"),
        edge(2, 20, 30, "godfather"),
    ]
    result = levels([10, 20, 30], edges)
    assert result.levels[20] is not None
    assert result.levels[10] - result.levels[20] == 1
    assert result.levels[30] is None
    assert {item["entity_id"] for item in result.unplaced} == {30}


def test_isolated_person_is_unplaced():
    result = levels([10, 20, 30], [edge(1, 10, 30, "father")])
    reasons = {item["entity_id"]: item["reason"] for item in result.unplaced}
    assert reasons == {20: "isolated"}


def test_dangling_endpoint_is_reported_not_silently_added():
    """Ребро на узел вне набора — битые данные: диагностика, а не тихое добавление."""
    result = levels([10, 20], [edge(1, 10, 30, "father")])
    assert 30 not in result.levels
    assert len(result.dangling) == 1
    assert result.dangling[0]["kind"] == "dangling_endpoint"
    assert result.dangling[0]["missing"] == [30]


# ---------------------------------------------------------------------------
# Диагностика данных
# ---------------------------------------------------------------------------


def test_two_blood_fathers_are_reported_and_anchor_named_pair_is_flagged():
    """Два кровных отца — диагноз; якорная и именованная рядом — кандидат на слияние."""
    edges = [
        edge(1, 10, 20, "father", confidence=0.9),
        edge(2, 10, 30, "father", confidence=0.6),
    ]
    result = levels([10, 20, 30], edges, anchors={30})
    kinds = [c["kind"] for c in result.conflicts]
    assert "functional_parent" in kinds
    assert "anchor_named_pair" in kinds
    pair = [c for c in result.conflicts if c["kind"] == "anchor_named_pair"][0]
    assert pair["merge_candidates"] == [30, 20]
    assert pair["child"] == 10
    report = [c for c in result.conflicts if c["kind"] == "functional_parent"][0]
    assert report["parents"] == [20, 30]
    assert report["duplicated_slots"] == {"father": [20, 30]}


def test_father_and_mother_are_not_a_functional_conflict():
    """Отец и мать — норма, а не «два родителя»: слот сравнивается по роли."""
    edges = [
        edge(1, 10, 20, "father"),
        edge(2, 10, 30, "mother"),
    ]
    result = levels([10, 20, 30], edges)
    assert [c for c in result.conflicts if c["kind"] == "functional_parent"] == []


def test_genderless_parent_competes_with_both_slots():
    """«Родитель» без пола конкурирует и с отцом, и с матерью."""
    edges = [
        edge(1, 10, 20, "father"),
        edge(2, 10, 30, "parent"),
    ]
    result = levels([10, 20, 30], edges, anchors={30})
    # слот матери занят одним «родителем» — конфликта там нет
    reports = [c for c in result.conflicts if c["kind"] == "functional_parent"]
    assert [list(r["duplicated_slots"]) for r in reports] == [["father"]]
    assert len([c for c in result.conflicts if c["kind"] == "anchor_named_pair"]) == 1


def test_genderless_parent_conflicts_with_both_slots_when_both_are_taken():
    edges = [
        edge(1, 10, 20, "father"),
        edge(2, 10, 30, "mother"),
        edge(3, 10, 40, "parent"),
    ]
    result = levels([10, 20, 30, 40], edges)
    reports = [c for c in result.conflicts if c["kind"] == "functional_parent"]
    assert [sorted(r["duplicated_slots"]) for r in reports] == [["father", "mother"]]
    # в находке остаётся весь записанный набор родителей, а не только спорный слот
    assert reports[0]["parents"] == [20, 30, 40]


def test_two_fathers_and_a_mother_keep_the_whole_recorded_set():
    """Находка — весь записанный набор родителей, а слот показывает противоречие.

    Ровно случай из живых данных: у Alexander записаны два отца и мать. Мать не
    пропадает из списка только потому, что спорен слот отца.
    """
    edges = [
        edge(1, 10, 20, "father", confidence=0.9),
        edge(2, 10, 30, "mother", confidence=0.9),
        edge(3, 10, 40, "father", confidence=0.6),
    ]
    result = levels([10, 20, 30, 40], edges)
    reports = [c for c in result.conflicts if c["kind"] == "functional_parent"]
    assert len(reports) == 1
    assert reports[0]["parents"] == [20, 30, 40]
    assert reports[0]["duplicated_slots"] == {"father": [20, 40]}


def test_father_and_stepfather_are_not_a_functional_conflict():
    edges = [
        edge(1, 10, 20, "father"),
        edge(2, 10, 30, "stepfather"),
    ]
    result = levels([10, 20, 30], edges)
    assert [c for c in result.conflicts if c["kind"] == "functional_parent"] == []


def test_same_fact_from_two_authors_does_not_double_the_parents():
    """Один отец, описанный дважды, — не «два отца»."""
    edges = [
        edge(1, 10, 20, "father", confidence=0.9),
        edge(2, 20, 10, "child", confidence=0.4),
    ]
    result = levels([10, 20], edges)
    assert [c for c in result.conflicts if c["kind"] == "functional_parent"] == []
