from causalityrag.attribution_graph import (
    DirectActivationAttributionGraphBuilder,
    NativeMLPAttributionGraphBuilder,
    _overlaps,
)


def test_overlaps_uses_nonempty_intersection() -> None:
    assert _overlaps(2, 5, 4, 8)
    assert not _overlaps(2, 4, 4, 8)
    assert not _overlaps(0, 0, 0, 8)


def test_query_context_answer_partitions_and_region_mass_are_explicit() -> None:
    tokens = [
        {"position": 0, "region": "prompt"},
        {"position": 1, "region": "context"},
        {"position": 2, "region": "query"},
        {"position": 3, "region": "answer"},
    ]
    edges = [
        {"src_position": 1, "dst_position": 2, "contribution": 0.25},
        {"src_position": 2, "dst_position": 3, "contribution": 0.75},
    ]

    assert NativeMLPAttributionGraphBuilder._token_partitions(tokens) == {
        "query": [2],
        "context": [1],
        "answer": [3],
        "prompt": [0],
    }
    assert NativeMLPAttributionGraphBuilder._region_edge_mass(edges, tokens) == {
        "context->query": 0.25,
        "query->answer": 0.75,
    }


def test_direct_edge_keeps_signed_and_support_weights_separate() -> None:
    positive = DirectActivationAttributionGraphBuilder._direct_edge(
        0, 2, 1, 5, "attention_ov_write", 0.75, 0
    )
    negative = DirectActivationAttributionGraphBuilder._direct_edge(
        1, 5, 2, 5, "mlp_output_write", -0.25, 0
    )

    assert positive["signed_contribution"] == 0.75
    assert positive["contribution"] == 0.75
    assert positive["negative_contribution"] == 0.0
    assert negative["signed_contribution"] == -0.25
    assert negative["contribution"] == 0.0
    assert negative["negative_contribution"] == 0.25
    assert negative["relevance"] == 0.25


def test_direct_nodes_preserve_layer_token_stages_and_answer_sink() -> None:
    token_meta = [
        {"position": 0, "region": "context", "text": "A"},
        {"position": 1, "region": "answer", "text": "B"},
    ]
    edge = DirectActivationAttributionGraphBuilder._direct_edge(
        0, 0, 1, 1, "attention_ov_write", 0.5, 0
    )
    output_edge = DirectActivationAttributionGraphBuilder._direct_edge(
        4, 0, 5, 1, "answer_logit", 0.4, 1
    )
    output_edge["dst"] = "answer_target"

    nodes = DirectActivationAttributionGraphBuilder._direct_nodes(
        [edge, output_edge], token_meta, layer_count=2, target_positions=[1]
    )
    node_ids = {node["node_id"] for node in nodes}

    assert {"s0:t0", "s1:t1", "s4:t0", "answer_target"} <= node_ids
    assert next(node for node in nodes if node["node_id"] == "answer_target")["region"] == "answer"


def test_direct_pruning_always_keeps_answer_logit_edges() -> None:
    weak_sink = DirectActivationAttributionGraphBuilder._direct_edge(
        4, 0, 5, 1, "answer_logit", 0.01, 1
    )
    strong_edges = [
        DirectActivationAttributionGraphBuilder._direct_edge(
            0, index, 1, index, "attention_ov_write", float(index + 1), 0
        )
        for index in range(4)
    ]

    kept = DirectActivationAttributionGraphBuilder._prune_direct_edges(
        strong_edges + [weak_sink], max_edges=3
    )

    assert len(kept) == 3
    assert weak_sink in kept
    assert sum(edge["kind"] == "answer_logit" for edge in kept) == 1


def test_closed_flow_diagnostics_preserve_mass_and_expose_background() -> None:
    token_meta = [
        {"position": 0, "region": "context"},
        {"position": 1, "region": "answer"},
    ]
    context_edge = DirectActivationAttributionGraphBuilder._closed_flow_edge(
        0, 0, 1, 1, "attention_ov_write", 3.0, 0.6, 0
    )
    background_edge = (
        DirectActivationAttributionGraphBuilder._closed_background_edge(
            1, 1, 0.4, 0, "attention_beam"
        )
    )
    answer_edge = DirectActivationAttributionGraphBuilder._closed_flow_edge(
        1, 1, 2, 1, "answer_logit", 2.0, 1.0, 0
    )
    answer_edge["dst"] = "answer_target"

    diagnostics = DirectActivationAttributionGraphBuilder._closed_flow_diagnostics(
        [context_edge, background_edge, answer_edge],
        token_meta,
    )

    assert abs(diagnostics["sink_inflow"] - 1.0) < 1e-12
    assert abs(diagnostics["source_total_flow"] - 1.0) < 1e-12
    assert abs(diagnostics["background_flow"] - 0.4) < 1e-12
    assert diagnostics["input_region_flow"] == {"context": 0.6}
    assert diagnostics["maximum_internal_conservation_error"] < 1e-12


def test_region_mass_labels_closed_flow_background_explicitly() -> None:
    tokens = [{"position": 0, "region": "context"}]
    edge = DirectActivationAttributionGraphBuilder._closed_background_edge(
        1, 0, 0.25, 0, "attention_beam"
    )

    assert DirectActivationAttributionGraphBuilder._region_edge_mass(
        [edge], tokens
    ) == {"background->context": 0.25}


def test_absorbing_flow_removes_background_edges_and_records_mass() -> None:
    context_edge = DirectActivationAttributionGraphBuilder._closed_flow_edge(
        0, 0, 1, 1, "attention_ov_write", 3.0, 0.6, 0
    )
    background_edge = (
        DirectActivationAttributionGraphBuilder._closed_background_edge(
            1, 1, 0.4, 0, "attention_beam"
        )
    )

    retained, diagnostics = (
        DirectActivationAttributionGraphBuilder._absorbing_flow_subgraph(
            [context_edge, background_edge],
            {"background_flow": 0.4},
        )
    )

    assert retained == [context_edge]
    assert diagnostics["absorbed_background_mass"] == 0.4
    assert diagnostics["removed_background_edges"] == 1
    assert diagnostics["retained_background_flow"] == 0.0
