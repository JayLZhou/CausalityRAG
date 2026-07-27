from causalityrag.graph_cut import project_cached_units_source_target_graph
from exp.run_contribution_aware_flow_contract_attack import (
    breakpoint_price_cuts,
    solve_price_cut,
)


def test_projection_preserves_source_interaction_and_target_contributions():
    units = [
        {
            "unit_id": "u1",
            "chunk_id": "c1",
            "chunk_char_start": 0,
            "chunk_char_end": 5,
        },
        {
            "unit_id": "u2",
            "chunk_id": "c1",
            "chunk_char_start": 6,
            "chunk_char_end": 10,
        },
    ]
    graph = {
        "context_token_supports": [
            {"position": 3, "chunk_id": "c1", "chunk_char_start": 0, "chunk_char_end": 5},
            {"position": 4, "chunk_id": "c1", "chunk_char_start": 6, "chunk_char_end": 10},
        ],
        "graph": {
            "token_partitions": {"query": [1]},
            "target_positions": [8],
            "edges": [
                {"kind": "attention_ov_write", "src_position": 3, "dst_position": 1, "contribution": 2.0},
                {"kind": "attention_ov_write", "src_position": 3, "dst_position": 4, "contribution": 1.0},
                {"kind": "attention_ov_write", "src_position": 4, "dst_position": 7, "contribution": 3.0},
            ],
        },
    }

    source, interactions, target, metadata = project_cached_units_source_target_graph(graph, units)

    assert source == {"u1": 2.0}
    assert interactions == {("u1", "u2"): 1.0}
    assert target == {"u2": 3.0}
    assert not metadata["path_shortcut_fallback"]


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
