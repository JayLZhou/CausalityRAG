import json

from scripts.repair_disconnected_contribution_graphs import (
    load_target_ids,
    repair_graph_row,
    repair_selected_graph_row,
)


def test_repair_graph_row_appends_auditable_bridge():
    row = {
        "id": "q1",
        "status": "ok",
        "contribution_graph": {
            "token_nodes": ["a", "b", "excluded"],
            "source_edges": [
                {"token_id": "a", "capacity": 0.8},
                {"token_id": "excluded", "capacity": 9.0},
            ],
            "interaction_edges": [],
            "target_edges": [{"token_id": "b", "capacity": 0.3}],
            "diagnostics": {},
        },
    }

    repaired, changed = repair_graph_row(row, {"a", "b"})

    assert changed
    assert repaired["status"] == "ok"
    assert repaired["status_before_connectivity_repair"] == "ok"
    assert repaired["contribution_graph"]["interaction_edges"] == [
        {"source": "a", "target": "b", "capacity": 0.3}
    ]
    assert repaired["connectivity_repair"]["applied"]


def test_repair_graph_row_keeps_connected_graph_unchanged():
    row = {
        "id": "q1",
        "status": "ok",
        "contribution_graph": {
            "token_nodes": ["a"],
            "source_edges": [{"token_id": "a", "capacity": 0.8}],
            "interaction_edges": [],
            "target_edges": [{"token_id": "a", "capacity": 0.3}],
        },
    }

    repaired, changed = repair_graph_row(row, {"a"})

    assert not changed
    assert repaired == row


def test_repair_selected_graph_row_does_not_touch_unselected_disconnect():
    row = {
        "id": "q1",
        "contribution_graph": {
            "token_nodes": ["a", "b"],
            "source_edges": [{"token_id": "a", "capacity": 0.8}],
            "interaction_edges": [],
            "target_edges": [{"token_id": "b", "capacity": 0.3}],
        },
    }

    repaired, changed = repair_selected_graph_row(
        row,
        {"a", "b"},
        {"some-other-query"},
    )

    assert not changed
    assert repaired is row


def test_repair_selected_graph_row_marks_connected_target_without_new_edges():
    row = {
        "id": "q1",
        "contribution_graph": {
            "token_nodes": ["a"],
            "source_edges": [{"token_id": "a", "capacity": 0.8}],
            "interaction_edges": [],
            "target_edges": [{"token_id": "a", "capacity": 0.3}],
        },
    }

    processed, changed = repair_selected_graph_row(row, {"a"}, {"q1"})

    assert not changed
    assert processed["contribution_graph"] == row["contribution_graph"]
    assert processed["connectivity_repair"] == {
        "applied": False,
        "targeted": True,
        "reason": "already_connected_target_graph",
        "policy": "strongest_endpoint_bottleneck_bridge",
        "added_source_edges": [],
        "added_interaction_edges": [],
        "added_target_edges": [],
    }


def test_load_target_ids_uses_only_explicit_no_frontier_status(tmp_path):
    results = tmp_path / "results.jsonl"
    rows = [
        {"id": "ok", "evaluation_status": "success"},
        {"id": "target", "evaluation_status": "no_frontier_candidate"},
        {"id": "invalid", "evaluation_status": "invalid_clean_answer"},
    ]
    results.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    assert load_target_ids(
        results,
        target_status="no_frontier_candidate",
        expected_rows=3,
    ) == ["target"]
