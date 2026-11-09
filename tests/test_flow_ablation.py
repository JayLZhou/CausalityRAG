import copy

import pytest

from causalityrag.flow_ablation import rank_graph_tokens


def graph():
    return {
        "id": "q1", "status": "ok",
        "contribution_graph": {
            "token_nodes": ["a", "bridge", "b", "dead"],
            "source_edges": [{"token_id": "a", "capacity": 2}],
            "interaction_edges": [
                {"source": "a", "target": "bridge", "capacity": 2},
                {"source": "bridge", "target": "b", "capacity": 2},
                {"source": "dead", "target": "bridge", "capacity": 9},
            ],
            "target_edges": [{"token_id": "b", "capacity": 2}],
        },
    }


def test_direct_score_counts_outgoing_not_source_injection():
    result = rank_graph_tokens(graph(), {"a", "b", "dead"}, method="direct_score")
    assert result["token_scores"] == {"a": 2, "b": 2, "dead": 9}
    assert result["ranked_ids"] == ["dead", "a", "b"]
    assert result["max_flow_calls"] == 0


def test_singleton_flow_retains_noneditable_transit_and_does_not_mutate():
    row = graph()
    before = copy.deepcopy(row)
    result = rank_graph_tokens(row, {"a", "b", "dead"}, method="individual_flow")
    assert row == before
    assert result["initial_flow"] > 0
    assert result["token_scores"]["a"] == pytest.approx(result["initial_flow"])
    assert result["token_scores"]["b"] == pytest.approx(result["initial_flow"])
    assert result["token_scores"]["dead"] == 0
    assert result["ranked_ids"] == ["a", "b", "dead"]
    assert result["max_flow_calls"] == 4
    assert result["reader_calls"] == 0
    assert result["verification_status"] == "not_run"


@pytest.mark.parametrize("method", ["direct_score", "individual_flow"])
def test_empty_eligible_domain(method):
    result = rank_graph_tokens(graph(), set(), method=method)
    assert result["ranked_ids"] == []
    assert result["status"] == "no_eligible_tokens"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1])
def test_bad_capacity_rejected(bad):
    row = graph()
    row["contribution_graph"]["target_edges"][0]["capacity"] = bad
    with pytest.raises(ValueError, match="capacities"):
        rank_graph_tokens(row, {"a"}, method="direct_score")


def test_invalid_graph_and_domain_rejected():
    row = graph()
    with pytest.raises(ValueError, match="missing"):
        rank_graph_tokens(row, {"unknown"}, method="individual_flow")
    row["status"] = "failed"
    with pytest.raises(ValueError, match="successful"):
        rank_graph_tokens(row, {"a"}, method="direct_score")


def test_duplicate_edge_rejected():
    row = graph()
    row["contribution_graph"]["source_edges"] *= 2
    with pytest.raises(ValueError, match="duplicate"):
        rank_graph_tokens(row, {"a"}, method="direct_score")


def test_explicit_editable_projection_matches_legacy_flow_domain():
    result = rank_graph_tokens(graph(), {"a", "b", "dead"}, method="individual_flow",
                              graph_domain="editable_projection")
    assert result["initial_flow"] == 0
    assert result["flow_domain_token_count"] == 3
    assert result["token_scores"] == {"a": 0, "b": 0, "dead": 0}
