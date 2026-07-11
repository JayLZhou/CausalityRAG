from causalityrag.graph_cut import (
    _maximum_weight_closure_with_node_weights_scipy,
    decompose_source_target_densest_subgraphs,
    minimum_support_cut,
    solve_source_target_densest_subgraph,
    solve_source_target_supermodular_ratio,
    solve_source_target_vertex_cut,
    solve_weighted_densest_subgraph,
    top_budgeted_cut_sets,
)
from causalityrag.rules import TypedRuleLibrary


def test_top_budgeted_cut_sets_returns_ranked_distinct_candidates() -> None:
    units = [
        {"unit_id": f"u{i}", "support": support}
        for i, support in enumerate([5.0, 4.0, 3.0, 2.0])
    ]
    cuts = top_budgeted_cut_sets(units, 2, limit=3)
    assert [[unit["unit_id"] for unit in cut] for cut in cuts] == [
        ["u0", "u1"],
        ["u0", "u2"],
        ["u1", "u2"],
    ]


def test_explicit_ner_type_takes_priority_over_component_pool_guess() -> None:
    library = TypedRuleLibrary({
        "type_pool": {"ORG": ["American Media"], "NORP": ["Canadian"]},
        "role_pool": {},
    })
    assert library.infer_type("American", "NORP") == "NORP"


def test_pool_component_does_not_turn_content_word_into_entity() -> None:
    library = TypedRuleLibrary({
        "type_pool": {"PERSON": ["Someone Plays", "A Clever Boy"]},
        "role_pool": {},
    })
    assert library.infer_type("plays", "CONTENT") == ""
    assert library.infer_type("boy", "CONTENT") == ""


def test_entity_replacement_preserves_same_type_and_name_slot() -> None:
    library = TypedRuleLibrary({
        "type_pool": {"PERSON": ["Anthony Rapp", "Mary Smith", "John Brown"]},
        "role_pool": {},
    })
    result = library.replacement_for_unit({
        "text": "Rapp",
        "type": "PERSON",
        "entity_token_index": 1,
        "entity_token_count": 2,
    })
    assert result["ok"]
    assert result["new"] in {"Smith", "Brown"}
    assert result["policy"] == "same_type_entity_slot"


def test_minimum_support_cut_hits_requested_mass_with_fewest_unit_cost_edits() -> None:
    units = [
        {"unit_id": f"u{i}", "support": support}
        for i, support in enumerate([6.0, 3.0, 1.0])
    ]
    cut = minimum_support_cut(units, remaining_tau=0.1)
    assert [unit["unit_id"] for unit in cut] == ["u0", "u1"]


def test_weighted_densest_subgraph_prefers_a_dense_triangle() -> None:
    units = [{"unit_id": unit, "support": 0.0} for unit in ["a", "b", "c", "d"]]
    interactions = {("a", "b"): 2.0, ("b", "c"): 2.0, ("a", "c"): 2.0, ("c", "d"): 0.1}
    result = solve_weighted_densest_subgraph(units, interactions)
    assert set(result["selected_ids"]) == {"a", "b", "c"}
    assert result["density"] == 2.0


def test_source_target_vertex_cut_selects_only_the_chunk_bottleneck() -> None:
    units = [
        {"unit_id": "a", "cost": 1.0},
        {"unit_id": "b", "cost": 1.0},
        {"unit_id": "bottleneck", "cost": 1.0},
    ]
    result = solve_source_target_vertex_cut(
        units,
        {"a": 0.7, "b": 0.5},
        {("a", "bottleneck"): 0.3, ("b", "bottleneck"): 0.4},
        {"bottleneck": 0.9},
    )

    assert result["status"] == "optimal"
    assert result["selected_ids"] == ["bottleneck"]
    assert result["cut_cost"] == 1.0


def test_source_target_densest_uses_fixed_terminal_edges_without_selecting_terminals() -> None:
    units = [{"unit_id": unit} for unit in ["a", "b", "n1", "n2", "n3"]]
    result = solve_source_target_densest_subgraph(
        units,
        {"a": 1.0, "b": 1.0},
        {
            ("a", "b"): 0.2,
            ("n1", "n2"): 0.6,
            ("n1", "n3"): 0.6,
            ("n2", "n3"): 0.6,
        },
        {"a": 1.0, "b": 1.0},
    )

    assert result["selected_ids"] == ["a", "b"]
    assert abs(result["density"] - 2.1) < 1e-8


def test_source_target_densest_decomposition_peels_disjoint_solutions() -> None:
    units = [{"unit_id": unit} for unit in ["a", "b", "c"]]
    rounds = decompose_source_target_densest_subgraphs(
        units,
        {"a": 2.0, "b": 1.0, "c": 0.5},
        {},
        {"a": 2.0, "b": 1.0, "c": 0.5},
    )

    assert [round_["selected_ids"] for round_ in rounds] == [["a"], ["b"], ["c"]]


def test_supermodular_ratio_uses_no_cardinality_budget() -> None:
    units = [{"unit_id": unit} for unit in ["a", "b", "c"]]
    result = solve_source_target_supermodular_ratio(
        units,
        {"a": 1.0, "b": 1.0, "c": 0.0},
        {("a", "b"): 4.0, ("b", "c"): 0.1},
        {},
    )

    assert result["selected_ids"] == ["a", "b"]
    assert abs(result["density"] - 3.0) < 1e-8
    assert result["iterations"] >= 1


def test_scipy_closure_matches_expected_positive_pair_solution() -> None:
    selected, value = _maximum_weight_closure_with_node_weights_scipy(
        {"a": 1.0, "b": 1.0, "c": 0.0},
        [("a", "b", 4.0), ("b", "c", 0.1)],
        2.9,
    )
    assert selected == {"a", "b"}
    assert abs(value - 0.2) < 1e-8
