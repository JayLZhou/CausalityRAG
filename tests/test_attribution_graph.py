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
