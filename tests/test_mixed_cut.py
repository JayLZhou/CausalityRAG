from causalityrag.mixed_cut import (
    augment_with_unary_support,
    build_projected_token_contribution_network,
    build_raw_contribution_network,
    direct_support_source_prizes,
    remaining_support_flow,
    restrict_group_editable_units,
    search_mixed_cut_threshold,
    solve_bicriteria_flow_interdiction,
    solve_fixed_mixed_cut,
    solve_mixed_cut,
    solve_prize_collecting_separator,
    sweep_mixed_cuts,
    sweep_prize_collecting_separators,
)


def test_mixed_cut_uses_shared_flow_structure_beyond_unary_scores():
    units = [
        _unit("a", 0, 1),
        _unit("b", 1, 2),
        _unit("c", 2, 3),
    ]
    row = {
        "context_token_supports": [
            _support(0, 0, 1),
            _support(1, 1, 2),
            _support(2, 2, 3),
        ],
        "graph": {
            "edge_weight_semantics": (
                "signed local target-logit contribution: target gradient dotted "
                "with the actual residual, attention OV, or MLP output write"
            ),
            "target_objective": "mean raw logit over clean-answer tokens",
            "edges": [
                _edge("s0:t0", "x", 0, 1, 0, 5.0),
                _edge("s0:t1", "x", 0, 1, 1, 5.0),
                _edge("x", "answer_target", 1, 2, 3, 5.0),
                _edge("s0:t2", "answer_target", 0, 2, 2, 4.0),
            ],
        },
    }
    network = build_raw_contribution_network(row, units)
    assert network.status == "ok"
    assert abs(remaining_support_flow(network, set()) - 9.0) < 1e-9

    result = solve_mixed_cut(network, 3.0)
    assert result["selected_ids"] == ["c"]
    assert abs(remaining_support_flow(network, {"c"}) - 5.0) < 1e-9
    assert abs(remaining_support_flow(network, {"a"}) - 9.0) < 1e-9


def test_projected_network_propagates_graph_construction_failure():
    network = build_projected_token_contribution_network(
        {
            "status": "no_context_to_answer_path",
            "graph": {"edges": []},
        },
        [],
    )

    assert network.status == "no_context_to_answer_path"
    assert network.diagnostics["graph_status"] == "no_context_to_answer_path"


def test_fixed_lambda_wrapper_returns_one_exact_supported_cut():
    units = [_unit("a", 0, 1), _unit("b", 1, 2), _unit("c", 2, 3)]
    row = {
        "context_token_supports": [
            _support(0, 0, 1),
            _support(1, 1, 2),
            _support(2, 2, 3),
        ],
        "graph": {
            "edge_weight_semantics": (
                "signed local target-logit contribution: target gradient dotted "
                "with the actual residual, attention OV, or MLP output write"
            ),
            "target_objective": "mean raw logit over clean-answer tokens",
            "edges": [
                _edge("s0:t0", "x", 0, 1, 0, 5.0),
                _edge("s0:t1", "x", 0, 1, 1, 5.0),
                _edge("x", "answer_target", 1, 2, 3, 5.0),
                _edge("s0:t2", "answer_target", 0, 2, 2, 4.0),
            ],
        },
    }
    result = solve_fixed_mixed_cut(
        build_raw_contribution_network(row, units),
        token_cost=3.0,
    )
    assert result["status"] == "ok"
    assert result["diagnostics"]["mincut_calls"] == 1
    assert len(result["candidates"]) == 1
    assert result["candidates"][0]["selected_ids"] == ["c"]
    assert result["candidates"][0]["lambda"] == 3.0


def test_parametric_sweep_returns_non_unary_candidate():
    units = [_unit("a", 0, 1), _unit("b", 1, 2), _unit("c", 2, 3)]
    row = {
        "context_token_supports": [
            _support(0, 0, 1),
            _support(1, 1, 2),
            _support(2, 2, 3),
        ],
        "graph": {
            "edge_weight_semantics": (
                "signed local target-logit contribution: target gradient dotted "
                "with the actual residual, attention OV, or MLP output write"
            ),
            "target_objective": "mean raw logit over clean-answer tokens",
            "edges": [
                _edge("s0:t0", "x", 0, 1, 0, 5.0),
                _edge("s0:t1", "x", 0, 1, 1, 5.0),
                _edge("x", "answer_target", 1, 2, 3, 5.0),
                _edge("s0:t2", "answer_target", 0, 2, 2, 4.0),
            ],
        },
    }
    sweep = sweep_mixed_cuts(
        build_raw_contribution_network(row, units),
        points=80,
    )
    assert sweep["status"] == "ok"
    candidate = next(row for row in sweep["candidates"] if row["selected_ids"] == ["c"])
    assert candidate["differs_from_unary"]
    assert candidate["flow_improvement_over_unary"] > 0


def test_binary_lambda_finds_largest_supported_threshold_cut():
    units = [_unit("a", 0, 1), _unit("b", 1, 2)]
    row = {
        "context_token_supports": [
            _support(0, 0, 1),
            _support(1, 1, 2),
        ],
        "graph": {
            "edge_weight_semantics": (
                "signed local target-logit contribution: target gradient dotted "
                "with the actual residual, attention OV, or MLP output write"
            ),
            "target_objective": "mean raw logit over clean-answer tokens",
            "edges": [
                _edge("s0:t0", "answer_target", 0, 1, 0, 0.6),
                _edge("s0:t1", "answer_target", 0, 1, 1, 0.4),
            ],
        },
    }

    result = search_mixed_cut_threshold(
        build_projected_token_contribution_network(row, units),
        beta=0.5,
        iterations=40,
    )

    assert result["status"] == "ok"
    assert result["strict_candidate"]["selected_ids"] == ["a"]
    assert abs(result["strict_candidate"]["remaining_support_fraction"] - 0.4) < 1e-9
    assert abs(result["diagnostics"]["lambda_feasible_lower"] - 0.6) < 1e-8
    assert result["diagnostics"]["residual_cut_monotone_over_ascending_lambda"]


def test_normalized_graph_is_rejected():
    row = {
        "context_token_supports": [],
        "graph": {
            "edge_weight_semantics": "positive target-logit contribution, row-normalized",
            "edges": [],
        },
    }
    network = build_raw_contribution_network(row, [])
    assert network.status == "not_raw_direct_activation_graph"


def test_projected_token_graph_makes_internal_token_editable_once():
    units = [
        _unit("a", 0, 1),
        _unit("b", 1, 2),
        _unit("bridge", 2, 3),
    ]
    row = {
        "context_token_supports": [
            _support(0, 0, 1),
            _support(1, 1, 2),
            _support(2, 2, 3),
        ],
        "graph": {
            "edge_weight_semantics": (
                "signed local target-logit contribution: target gradient dotted "
                "with the actual residual, attention OV, or MLP output write"
            ),
            "target_objective": "mean raw logit over clean-answer tokens",
            "edges": [
                _edge("s0:t0", "s1:t2", 0, 1, 0, 4.0, dst_position=2),
                _edge("s0:t1", "s1:t2", 0, 1, 1, 4.0, dst_position=2),
                _edge("s1:t2", "answer_target", 1, 2, 2, 4.0),
            ],
        },
    }
    network = build_projected_token_contribution_network(row, units)
    assert network.status == "ok"
    result = solve_mixed_cut(network, 2.0)
    assert result["selected_ids"] == ["bridge"]
    assert abs(remaining_support_flow(network, {"bridge"})) < 1e-9


def test_prize_separator_prefers_shared_bottleneck_over_unary_root():
    units = [
        _unit("a", 0, 1),
        _unit("b", 1, 2),
        _unit("bridge", 2, 3),
    ]
    row = {
        "context_token_supports": [
            {**_support(0, 0, 1), "support": 3.0},
            {**_support(1, 1, 2), "support": 3.0},
            {**_support(2, 2, 3), "support": 0.0},
        ],
        "graph": {
            "edge_weight_semantics": (
                "signed local target-logit contribution: target gradient dotted "
                "with the actual residual, attention OV, or MLP output write"
            ),
            "target_objective": "mean raw logit over clean-answer tokens",
            "edges": [
                _edge("s0:t0", "s1:t2", 0, 1, 0, 4.0, dst_position=2),
                _edge("s0:t1", "s1:t2", 0, 1, 1, 4.0, dst_position=2),
                _edge("s1:t2", "answer_target", 1, 2, 2, 4.0),
            ],
        },
    }
    network = build_projected_token_contribution_network(row, units)
    prizes = direct_support_source_prizes(row, units, network)
    result = solve_prize_collecting_separator(network, prizes, token_cost=4.0)

    assert result["selected_ids"] == ["bridge"]
    assert abs(result["covered_source_prize"] - 6.0) < 1e-9
    assert abs(result["objective_value"] - 4.0) < 1e-9
    assert result["cut_identity_error"] < 1e-9

    sweep = sweep_prize_collecting_separators(network, prizes, points=80)
    candidate = next(
        candidate
        for candidate in sweep["candidates"]
        if candidate["selected_ids"] == ["bridge"]
    )
    assert candidate["unary_matched_ids"] == ["a"]
    assert candidate["coverage_improvement_over_unary"] == 3.0


def test_backward_conserved_capacities_preserve_unit_target_mass():
    units = [_unit("a", 0, 1), _unit("b", 1, 2)]
    row = {
        "context_token_supports": [
            _support(0, 0, 1),
            _support(1, 1, 2),
        ],
        "graph": {
            "edge_weight_semantics": (
                "signed local target-logit contribution: target gradient dotted "
                "with the actual residual, attention OV, or MLP output write"
            ),
            "target_objective": "mean raw logit over clean-answer tokens",
            "edges": [
                _edge("s0:t0", "x", 0, 1, 0, 100.0),
                _edge("s0:t1", "x", 0, 1, 1, 1.0),
                _edge("x", "answer_target", 1, 2, 2, 0.001),
            ],
        },
    }

    network = build_projected_token_contribution_network(
        row,
        units,
        capacity_mode="backward-conserved",
    )

    assert network.status == "ok"
    assert abs(remaining_support_flow(network, set()) - 1.0) < 1e-9
    assert abs(network.diagnostics["conservation_terminal_mass"] - 1.0) < 1e-9
    assert network.diagnostics["maximum_internal_conservation_error"] < 1e-9
    assert abs(remaining_support_flow(network, {"a"}) - 1.0 / 101.0) < 1e-9


def test_backward_conserved_input_root_gates_are_source_faithful():
    units = [_unit("a", 0, 1), _unit("b", 1, 2)]
    row = {
        "context_token_supports": [
            _support(0, 0, 1),
            _support(1, 1, 2),
        ],
        "graph": {
            "edge_weight_semantics": (
                "signed local target-logit contribution: target gradient dotted "
                "with the actual residual, attention OV, or MLP output write"
            ),
            "target_objective": "mean raw logit over clean-answer tokens",
            "edges": [
                _edge("s0:t0", "x", 0, 1, 0, 100.0),
                _edge("s0:t1", "x", 0, 1, 1, 1.0),
                _edge("x", "answer_target", 1, 2, 2, 0.001),
            ],
        },
    }

    network = build_raw_contribution_network(
        row,
        units,
        capacity_mode="backward-conserved",
    )

    assert network.status == "ok"
    assert network.gate_scope == "input_roots"
    assert network.diagnostics["projection"] == "input_root_gates"
    assert abs(remaining_support_flow(network, set()) - 1.0) < 1e-9
    assert abs(remaining_support_flow(network, {"a"}) - 1.0 / 101.0) < 1e-9


def test_integer_k_guessing_returns_theorem_aligned_candidate():
    units = [_unit("a", 0, 1), _unit("b", 1, 2), _unit("c", 2, 3)]
    row = {
        "context_token_supports": [
            _support(0, 0, 1),
            _support(1, 1, 2),
            _support(2, 2, 3),
        ],
        "graph": {
            "edge_weight_semantics": (
                "signed local target-logit contribution: target gradient dotted "
                "with the actual residual, attention OV, or MLP output write"
            ),
            "target_objective": "mean raw logit over clean-answer tokens",
            "edges": [
                _edge("s0:t0", "answer_target", 0, 1, 0, 4.0),
                _edge("s0:t1", "answer_target", 0, 1, 1, 3.0),
                _edge("s0:t2", "answer_target", 0, 1, 2, 2.0),
            ],
        },
    }
    network = build_raw_contribution_network(row, units)
    result = solve_bicriteria_flow_interdiction(
        network,
        beta=1.0 / 3.0,
        eta=0.5,
        max_k_guess=3,
    )

    assert result["status"] == "ok"
    assert result["diagnostics"]["mincut_calls"] == 3
    assert result["diagnostics"]["cardinality_factor"] == 3.0
    assert result["diagnostics"]["residual_flow_factor"] == 1.5
    assert "unary_order" not in result
    assert result["strict_candidate"] is not None
    assert result["bicriteria_candidate"] is not None
    assert "unary_matched_ids" not in result["bicriteria_candidate"]
    assert result["strict_candidate"]["remaining_support_flow"] <= 3.0 + 1e-9
    assert result["bicriteria_candidate"]["remaining_support_flow"] <= 4.5 + 1e-9


def test_geometric_k_guessing_reports_gamma_adjusted_guarantee():
    units = [_unit("a", 0, 1), _unit("b", 1, 2), _unit("c", 2, 3)]
    row = {
        "context_token_supports": [
            _support(0, 0, 1),
            _support(1, 1, 2),
            _support(2, 2, 3),
        ],
        "graph": {
            "edge_weight_semantics": (
                "signed local target-logit contribution: target gradient dotted "
                "with the actual residual, attention OV, or MLP output write"
            ),
            "target_objective": "mean raw logit over clean-answer tokens",
            "edges": [
                _edge("s0:t0", "answer_target", 0, 1, 0, 4.0),
                _edge("s0:t1", "answer_target", 0, 1, 1, 3.0),
                _edge("s0:t2", "answer_target", 0, 1, 2, 2.0),
            ],
        },
    }
    result = solve_bicriteria_flow_interdiction(
        build_raw_contribution_network(row, units),
        beta=1.0 / 3.0,
        eta=1.0,
        max_k_guess=3,
        gamma=1.0,
    )

    assert result["status"] == "ok"
    assert result["diagnostics"]["guess_scales"] == [1.0, 2.0, 3.0]
    assert result["diagnostics"]["mincut_calls"] == 3
    assert result["diagnostics"]["cardinality_factor"] == 3.0
    assert result["diagnostics"]["residual_flow_factor"] == 2.0
    assert result["bicriteria_candidate"] is not None


def test_projected_flow_reduction_handles_multiple_selected_tokens():
    units = [_unit("a", 0, 1), _unit("b", 1, 2), _unit("c", 2, 3)]
    row = {
        "context_token_supports": [
            _support(0, 0, 1),
            _support(1, 1, 2),
            _support(2, 2, 3),
        ],
        "graph": {
            "edge_weight_semantics": (
                "signed local target-logit contribution: target gradient dotted "
                "with the actual residual, attention OV, or MLP output write"
            ),
            "target_objective": "mean raw logit over clean-answer tokens",
            "edges": [
                _edge("s0:t0", "answer_target", 0, 1, 0, 5.0),
                _edge("s0:t1", "answer_target", 0, 1, 1, 3.0),
                _edge("s0:t2", "answer_target", 0, 1, 2, 1.0),
            ],
        },
    }
    network = build_projected_token_contribution_network(row, units)

    assert abs(remaining_support_flow(network, {"a"}) - 4.0) < 1e-9
    assert abs(remaining_support_flow(network, {"a", "b"}) - 1.0) < 1e-9


def test_projected_mixed_cut_keeps_registry_excluded_gates_immutable():
    units = [_unit("a", 0, 1), _unit("b", 1, 2)]
    row = {
        "context_token_supports": [
            _support(0, 0, 1),
            _support(1, 1, 2),
        ],
        "graph": {
            "edge_weight_semantics": (
                "signed local target-logit contribution: target gradient dotted "
                "with the actual residual, attention OV, or MLP output write"
            ),
            "target_objective": "mean raw logit over clean-answer tokens",
            "edges": [
                _edge("s0:t0", "answer_target", 0, 1, 0, 5.0),
                _edge("s0:t1", "answer_target", 0, 1, 1, 1.0),
            ],
        },
    }
    network = build_projected_token_contribution_network(row, units)
    restricted = restrict_group_editable_units(network, {"b"})

    result = solve_mixed_cut(restricted, 0.1)
    sweep = sweep_mixed_cuts(restricted, points=8)

    assert result["selected_ids"] == ["b"]
    assert abs(remaining_support_flow(restricted, {"b"}) - 5.0) < 1e-9
    assert sweep["unary_order"] == ["b"]
    assert all(
        set(candidate["selected_ids"]) <= {"b"}
        and set(candidate["unary_matched_ids"]) <= {"b"}
        for candidate in sweep["candidates"]
    )


def test_projected_hybrid_uses_one_shared_gate_per_token():
    units = [_unit("a", 0, 1), _unit("b", 1, 2)]
    row = {
        "context_token_supports": [
            _support(0, 0, 1),
            _support(1, 1, 2),
        ],
        "graph": {
            "edge_weight_semantics": (
                "signed local target-logit contribution: target gradient dotted "
                "with the actual residual, attention OV, or MLP output write"
            ),
            "target_objective": "mean raw logit over clean-answer tokens",
            "edges": [
                _edge("s0:t0", "answer_target", 0, 1, 0, 3.0),
                _edge("s0:t1", "answer_target", 0, 1, 1, 1.0),
            ],
        },
    }
    projected = build_projected_token_contribution_network(row, units)
    hybrid = augment_with_unary_support(
        projected,
        {"a": 1.0, "b": 3.0},
        graph_weight=0.5,
    )
    sweep = sweep_mixed_cuts(hybrid, points=8)

    assert hybrid.diagnostics["projection"] == (
        "projected_token_graph_with_shared_unary_gate"
    )
    assert hybrid.diagnostics["maximum_active_group_rank"] == 1
    assert set(hybrid.token_nodes_by_unit) == {"a", "b"}
    assert set(hybrid.selection_unit_by_gate) == {"a", "b"}
    assert all(
        not unit_id.startswith("unary-gate::")
        for candidate in sweep["candidates"]
        for unit_id in candidate["selected_ids"]
    )


def _unit(unit_id, start, end):
    return {
        "unit_id": unit_id,
        "chunk_id": "chunk",
        "chunk_char_start": start,
        "chunk_char_end": end,
        "text": unit_id,
    }


def _support(position, start, end):
    return {
        "position": position,
        "chunk_id": "chunk",
        "chunk_char_start": start,
        "chunk_char_end": end,
    }


def _edge(
    src,
    dst,
    src_layer,
    dst_layer,
    src_position,
    capacity,
    *,
    dst_position=3,
):
    return {
        "src": src,
        "dst": dst,
        "src_layer": src_layer,
        "dst_layer": dst_layer,
        "src_position": src_position,
        "dst_position": dst_position,
        "contribution": capacity,
    }
