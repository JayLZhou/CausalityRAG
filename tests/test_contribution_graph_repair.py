from causalityrag.contribution_graph import (
    ensure_executable_source_target_path,
    positive_source_target_path_exists,
)


def test_disconnected_executable_graph_gets_one_bottleneck_bridge():
    source, interactions, target, diagnostics = (
        ensure_executable_source_target_path(
            {"source-token", "target-token"},
            {"source-token": 0.8, "excluded": 9.0},
            {},
            {"target-token": 0.3},
        )
    )

    assert diagnostics["applied"]
    assert diagnostics["reason"] == "disconnected_executable_positive_subgraph"
    assert diagnostics["added_source_edges"] == []
    assert diagnostics["added_target_edges"] == []
    assert diagnostics["added_interaction_edges"] == [
        {
            "source": "source-token",
            "target": "target-token",
            "capacity": 0.3,
        }
    ]
    assert positive_source_target_path_exists(source, interactions, target)


def test_connectivity_repair_adds_missing_endpoint_edges():
    source, interactions, target, diagnostics = (
        ensure_executable_source_target_path(
            {"a", "b"},
            {},
            {("a", "b"): 0.4},
            {},
        )
    )

    assert diagnostics["applied"]
    assert diagnostics["added_source_edges"]
    assert diagnostics["added_target_edges"]
    assert positive_source_target_path_exists(source, interactions, target)
