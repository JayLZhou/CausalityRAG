from types import SimpleNamespace

from exp.select_topk_graph_tokens import select_topk_candidate


def test_topk_graph_token_baseline_ranks_local_scores(monkeypatch) -> None:
    network = SimpleNamespace(
        status="ok",
        selection_unit_by_gate={"a": "a", "b": "b", "c": "c"},
        unit_scores={"a": 1.0, "b": 3.0, "c": 2.0},
    )
    monkeypatch.setattr(
        "exp.select_topk_graph_tokens.remaining_support_flow",
        lambda _network, selected: 10.0 - len(selected),
    )

    candidate = select_topk_candidate(
        network,
        {
            "a": {"text": "A"},
            "b": {"text": "B"},
            "c": {"text": "C"},
        },
        top_tokens=2,
    )

    assert candidate["selected_ids"] == ["b", "c"]
    assert candidate["selected_scores"] == [3.0, 2.0]
    assert candidate["n_selected"] == 2
