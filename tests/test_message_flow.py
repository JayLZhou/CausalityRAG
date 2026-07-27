from causalityrag.message_flow import (
    _ClosedMessageFlowExtractor,
    _MessageTraceRuntime,
    _overlaps,
)
from scripts.build_contribution_graph import answer_from_result_row


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

    assert _MessageTraceRuntime._token_partitions(tokens) == {
        "query": [2],
        "context": [1],
        "answer": [3],
        "prompt": [0],
    }
    assert _MessageTraceRuntime._region_edge_mass(edges, tokens) == {
        "context->query": 0.25,
        "query->answer": 0.75,
    }


def test_contribution_nodes_preserve_token_stages_and_answer_sink() -> None:
    token_meta = [
        {"position": 0, "region": "context", "text": "A"},
        {"position": 1, "region": "answer", "text": "B"},
    ]
    edge = {
        "src": "s0:t0",
        "dst": "s1:t1",
        "src_layer": 0,
        "dst_layer": 1,
        "src_position": 0,
        "dst_position": 1,
    }
    output_edge = {
        "src": "s4:t0",
        "dst": "answer_target",
        "src_layer": 4,
        "dst_layer": 5,
        "src_position": 0,
        "dst_position": 1,
    }

    nodes = _ClosedMessageFlowExtractor._contribution_nodes(
        [edge, output_edge], token_meta, layer_count=2, target_positions=[1]
    )
    node_ids = {node["node_id"] for node in nodes}

    assert {"s0:t0", "s1:t1", "s4:t0", "answer_target"} <= node_ids
    assert (
        next(node for node in nodes if node["node_id"] == "answer_target")["region"]
        == "answer"
    )


def test_closed_flow_diagnostics_preserve_mass_and_expose_background() -> None:
    token_meta = [
        {"position": 0, "region": "context"},
        {"position": 1, "region": "answer"},
    ]
    context_edge = _ClosedMessageFlowExtractor._closed_flow_edge(
        0, 0, 1, 1, "attention_ov_write", 3.0, 0.6, 0
    )
    background_edge = _ClosedMessageFlowExtractor._closed_background_edge(
        1, 1, 0.4, 0, "attention_beam"
    )
    answer_edge = _ClosedMessageFlowExtractor._closed_flow_edge(
        1, 1, 2, 1, "answer_logit", 2.0, 1.0, 0
    )
    answer_edge["dst"] = "answer_target"

    diagnostics = _ClosedMessageFlowExtractor._closed_flow_diagnostics(
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
    edge = _ClosedMessageFlowExtractor._closed_background_edge(
        1, 0, 0.25, 0, "attention_beam"
    )

    assert _ClosedMessageFlowExtractor._region_edge_mass(
        [edge], tokens
    ) == {"background->context": 0.25}


def test_default_context_handling_does_not_truncate_retrieved_text() -> None:
    builder = _ClosedMessageFlowExtractor.__new__(
        _ClosedMessageFlowExtractor
    )
    builder.max_context_tokens = 0
    contexts = [
        {"chunk_id": "a", "text": "one two three"},
        {"chunk_id": "b", "text": "four five six"},
    ]

    prepared = builder._truncate_contexts(contexts)

    assert prepared == contexts
    assert prepared is not contexts
    assert all(left is not right for left, right in zip(prepared, contexts))


def test_answer_objective_seed_is_uniform_and_has_unit_mass() -> None:
    edges = _ClosedMessageFlowExtractor._answer_objective_edges(
        [4, 7, 9],
        final_stage=56,
        final_layer=27,
        sink_position=10,
        target_positions=[5, 8, 10],
    )

    assert len(edges) == 3
    assert {edge["kind"] for edge in edges} == {"answer_objective"}
    assert {edge["dst"] for edge in edges} == {"answer_target"}
    assert abs(sum(edge["contribution"] for edge in edges) - 1.0) < 1e-12
    assert all(abs(edge["contribution"] - 1 / 3) < 1e-12 for edge in edges)


def test_contribution_graph_requires_a_context_to_answer_path() -> None:
    token_meta = [
        {"position": 0, "region": "context", "text": "Paris"},
        {"position": 1, "region": "query", "text": "?"},
        {"position": 2, "region": "answer", "text": "Paris"},
    ]
    answer_edge = _ClosedMessageFlowExtractor._closed_flow_edge(
        2, 1, 3, 2, "answer_objective", 1.0, 1.0, 0
    )
    answer_edge["dst"] = "answer_target"

    assert (
        _ClosedMessageFlowExtractor._contribution_graph_status(
            [answer_edge],
            token_meta,
            require_context_path=True,
        )
        == "no_context_input_flow"
    )

    context_edge = _ClosedMessageFlowExtractor._closed_flow_edge(
        0, 0, 2, 1, "attention_ov_write", 0.5, 1.0, 0
    )
    assert (
        _ClosedMessageFlowExtractor._contribution_graph_status(
            [context_edge, answer_edge],
            token_meta,
            require_context_path=True,
        )
        == "ok"
    )


def test_empty_contribution_graph_never_reports_ok() -> None:
    assert (
        _ClosedMessageFlowExtractor._contribution_graph_status(
            [],
            [],
            require_context_path=True,
        )
        == "empty_contribution_graph"
    )


def test_empty_frozen_reader_answer_is_preserved_as_abstention_target() -> None:
    assert answer_from_result_row({"clean_answer": ""}) == ""
