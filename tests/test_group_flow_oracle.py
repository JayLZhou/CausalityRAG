from causalityrag.group_flow_oracle import solve_exact_group_flow_blocker
from causalityrag.mixed_cut import (
    RawContributionNetwork,
    restrict_group_editable_units,
)
from scripts.analyze_group_flow_oracle import summarize


def test_exact_group_oracle_charges_one_label_for_multiple_layer_copies():
    network = _parallel_copy_network()

    result = solve_exact_group_flow_blocker(network, flow_threshold=5.0)

    assert result["status"] == "optimal"
    assert result["selected_ids"] == ["a"]
    assert result["n_selected"] == 1
    assert abs(result["remaining_support_flow"] - 5.0) < 1e-8


def test_exact_group_oracle_respects_immutable_registry_gates():
    network = restrict_group_editable_units(_parallel_copy_network(), {"b"})

    result = solve_exact_group_flow_blocker(network, flow_threshold=5.0)

    assert result["status"] == "infeasible"
    assert result["selected_ids"] == []


def test_oracle_summary_separates_feasible_and_proven_infeasible_rows():
    feasible = {
        "algorithm": {"n_selected": 2},
        "exact_strict": {"optimal": True, "status": "optimal", "elapsed_seconds": 1.0},
        "exact_relaxed": {
            "optimal": True,
            "status": "optimal",
            "n_selected": 2,
            "elapsed_seconds": 0.5,
        },
        "same_threshold_size_ratio": 1.0,
        "max_k_guess": 10,
        "theorem_applicable": True,
        "residual_bound_holds": True,
        "cardinality_bound_holds": True,
        "editable_labels": 8,
        "maximum_group_rank": 3,
    }
    infeasible = {
        "algorithm": None,
        "exact_strict": {"optimal": False, "status": "infeasible", "elapsed_seconds": 0.1},
        "exact_relaxed": {"optimal": False, "status": "infeasible", "elapsed_seconds": 0.1},
        "same_threshold_size_ratio": None,
        "max_k_guess": 10,
        "theorem_applicable": False,
        "residual_bound_holds": False,
        "cardinality_bound_holds": None,
        "editable_labels": 4,
        "maximum_group_rank": 2,
    }

    result = summarize([feasible, infeasible])

    assert result["feasible_both_oracles_optimal"] == 1
    assert result["proven_infeasible"] == 1
    assert result["algorithm_same_threshold_optimal"] == 1
    assert result["theorem_residual_bound_violations"] == 0
    assert result["theorem_cardinality_bound_violations"] == 0


def _parallel_copy_network() -> RawContributionNetwork:
    nodes = frozenset({"a0", "a1", "b0", "answer_target"})
    gates = {
        "copy::a0": "a0",
        "copy::a1": "a1",
        "copy::b0": "b0",
    }
    return RawContributionNetwork(
        status="ok",
        nodes=nodes,
        edges=(
            ("a0", "answer_target", 3.0),
            ("a1", "answer_target", 3.0),
            ("b0", "answer_target", 5.0),
        ),
        roots_by_unit={gate: (node,) for gate, node in gates.items()},
        token_nodes_by_unit=gates,
        selection_unit_by_gate={
            "copy::a0": "a",
            "copy::a1": "a",
            "copy::b0": "b",
        },
        gate_scope="contracted_token_nodes",
        unit_scores={gate: 0.0 for gate in gates},
        sink="answer_target",
        diagnostics={},
    )
