from causalityrag.contribution_graph import contract_message_flow_row


def test_contracts_a_saved_message_flow_without_loading_a_model() -> None:
    record = {
        "id": "q1",
        "question": "Where?",
        "answer": "Paris",
        "retrieved": [{"chunk_id": "c1", "text": "Paris wins", "rank": 1}],
    }
    units = [
        {
            "unit_id": "token:c1:0:5",
            "chunk_id": "c1",
            "chunk_char_start": 0,
            "chunk_char_end": 5,
        },
        {
            "unit_id": "token:c1:6:10",
            "chunk_id": "c1",
            "chunk_char_start": 6,
            "chunk_char_end": 10,
        },
    ]
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
            "token_partitions": {"query": []},
            "target_positions": [20],
            "edges": [
                {
                    "src": "prompt",
                    "dst": "s1:t10",
                    "src_position": 0,
                    "dst_position": 10,
                    "kind": "attention_ov_write",
                    "contribution": 0.4,
                },
                {
                    "src": "s1:t10",
                    "dst": "s2:t11",
                    "src_position": 10,
                    "dst_position": 11,
                    "kind": "attention_ov_write",
                    "contribution": 0.3,
                },
                {
                    "src": "s2:t11",
                    "dst": "s3:t19",
                    "src_position": 11,
                    "dst_position": 19,
                    "kind": "answer_objective",
                    "contribution": 0.2,
                },
            ],
        },
    }

    row = contract_message_flow_row(
        record,
        "Paris",
        message_flow,
        token_units=units,
        k=1,
    )

    assert row["status"] == "ok"
    graph = row["contribution_graph"]
    assert graph["token_nodes"] == ["token:c1:0:5", "token:c1:6:10"]
    assert graph["source_edges"] == [
        {"token_id": "token:c1:0:5", "capacity": 0.4}
    ]
    assert graph["interaction_edges"] == [{
        "source": "token:c1:0:5",
        "target": "token:c1:6:10",
        "capacity": 0.3,
    }]
    assert graph["target_edges"] == [
        {"token_id": "token:c1:6:10", "capacity": 0.2}
    ]
