from causalityrag.flow_contract import (
    breakpoint_price_cuts,
    require_complete_graph_domain,
    solve_price_cut,
)


def test_fixed_price_flow_contract_returns_a_token_gated_cut():
    units = [{"unit_id": "u1"}, {"unit_id": "u2"}]
    result = solve_price_cut(
        units,
        {"u1": 2.0},
        {("u1", "u2"): 2.0},
        {"u2": 2.0},
        token_price=0.5,
        edge_capacity_mode="unit-plus-normalized",
    )

    assert result["status"] == "optimal"
    assert result["selected_ids"]
    assert result["n_selected"] == len(result["selected_ids"])


def test_selection_is_independent_of_replacement_pool_availability():
    units = [{"unit_id": "first"}, {"unit_id": "second"}]
    result = solve_price_cut(
        units,
        {"first": 2.0},
        {("first", "second"): 2.0},
        {"second": 2.0},
        token_price=0.5,
        edge_capacity_mode="unit-plus-normalized",
    )

    assert result["status"] == "optimal"
    assert result["selected_ids"]
    assert set(result["selected_ids"]).issubset({"first", "second"})


def test_graph_domain_cannot_be_pre_filtered_by_replacement_availability():
    graph = {
        "contribution_graph": {
            "token_nodes": ["fixed", "editable"],
        }
    }
    units = [{"unit_id": "fixed"}, {"unit_id": "editable"}]
    require_complete_graph_domain(graph, units)

    graph["contribution_graph"]["token_nodes"] = ["editable"]
    try:
        require_complete_graph_domain(graph, units)
    except ValueError as error:
        assert "graph and token cache differ" in str(error)
    else:
        raise AssertionError("replacement-filtered graph domain was accepted")


def test_breakpoint_frontier_recovers_every_extreme_cardinality():
    units = [{"unit_id": "high"}, {"unit_id": "low"}]
    result = breakpoint_price_cuts(
        units,
        {"high": 3.0, "low": 1.0},
        {},
        {"high": 3.0, "low": 1.0},
        edge_capacity_mode="unit-plus-normalized",
    )

    assert result["status"] == "ok"
    assert [candidate["n_selected"] for candidate in result["candidates"]] == [1, 2]
    assert result["diagnostics"]["frontier_complete"]
    assert result["diagnostics"]["supported_extreme_points_including_empty"] == 3
