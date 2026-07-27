from causalityrag.contribution_graph import (
    ContributionGraphBuilder,
    _contract_token_labels,
    contribution_graph_edges,
)


def test_builds_contribution_graph_from_closed_message_flow() -> None:
    message_flow = {
        "status": "ok",
        "context_token_supports": [
            {
                "position": 10,
                "chunk_id": "c1",
                "chunk_char_start": 0,
                "chunk_char_end": 5,
            },
            {
                "position": 11,
                "chunk_id": "c1",
                "chunk_char_start": 6,
                "chunk_char_end": 10,
            },
        ],
        "graph": {
            "token_partitions": {"query": [20]},
            "target_positions": [30],
            "edges": [
                {
                    "src": "prompt",
                    "dst": "s1:t10",
                    "src_position": 0,
                    "dst_position": 10,
                    "kind": "mlp_output_write",
                    "contribution": 0.2,
                },
                {
                    "src": "s1:t10",
                    "dst": "s2:t20",
                    "src_position": 10,
                    "dst_position": 20,
                    "kind": "attention_ov_write",
                    "contribution": 0.3,
                },
                {
                    "src": "s2:t10",
                    "dst": "s3:t11",
                    "src_position": 10,
                    "dst_position": 11,
                    "kind": "attention_ov_write",
                    "contribution": 0.4,
                },
                {
                    "src": "s2:t10",
                    "dst": "s3:t10",
                    "src_position": 10,
                    "dst_position": 10,
                    "kind": "attention_residual",
                    "contribution": 0.8,
                },
                {
                    "src": "s4:t11",
                    "dst": "s5:t29",
                    "src_position": 11,
                    "dst_position": 29,
                    "kind": "attention_ov_write",
                    "contribution": 0.6,
                },
                {
                    "src": "background::s1:t10",
                    "dst": "s1:t10",
                    "src_position": -1,
                    "dst_position": 10,
                    "kind": "background_mlp",
                    "contribution": 9.0,
                },
            ],
        },
    }

    class FakeExtractor:
        def extract(self, *args, **kwargs):
            return message_flow

    builder = ContributionGraphBuilder.__new__(ContributionGraphBuilder)
    builder._extractor = FakeExtractor()
    graph_row = builder.build(
        {
            "id": "q1",
            "question": "q",
            "answer": "a",
            "retrieved": [
                {"chunk_id": "c1", "text": "Alpha beta", "rank": 1},
            ],
        },
        "answer",
        k=1,
    )
    source, interactions, target = contribution_graph_edges(graph_row)
    diagnostics = graph_row["contribution_graph"]["diagnostics"]

    assert source == {"token:c1:0:5": 0.5}
    assert interactions == {
        ("token:c1:0:5", "token:c1:6:10"): 0.4,
    }
    assert target == {"token:c1:6:10": 0.6}
    assert graph_row["contribution_graph"]["token_nodes"] == [
        "token:c1:0:5",
        "token:c1:6:10",
    ]
    assert diagnostics["raw_edge_counts"] == {
        "fixed_source": 1,
        "query_anchor": 1,
        "interaction": 1,
        "target": 1,
        "contracted_internal": 1,
    }
    assert diagnostics["raw_edge_mass"]["contracted_internal"] == 0.8


def test_support_shortcut_handles_a_flow_without_direct_source_or_target_edges() -> None:
    message_flow = {
        "context_token_supports": [{
            "position": 10,
            "chunk_id": "c1",
            "chunk_char_start": 0,
            "chunk_char_end": 5,
            "support": 0.7,
        }],
        "graph": {
            "token_partitions": {"query": []},
            "target_positions": [],
            "edges": [],
        },
    }
    units = [{
        "unit_id": "token:c1:0:5",
        "chunk_id": "c1",
        "chunk_char_start": 0,
        "chunk_char_end": 5,
    }]

    source, interactions, target, diagnostics = _contract_token_labels(
        message_flow,
        units,
    )

    assert source == {"token:c1:0:5": 0.7}
    assert interactions == {}
    assert target == {"token:c1:0:5": 0.7}
    assert diagnostics["path_shortcut_fallback"]
