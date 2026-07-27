from exp.run_gradient_attribution_baselines import ranked_candidate


def test_ranked_candidate_uses_all_positive_context_tokens() -> None:
    units = [
        {"unit_id": "a", "text": "alpha"},
        {"unit_id": "b", "text": "beta"},
        {"unit_id": "c", "text": "gamma"},
    ]

    candidate = ranked_candidate(
        units,
        {"a": 0.2, "b": 0.9, "c": -1.0},
        top_k=2,
    )

    assert candidate is not None
    assert candidate["selected_ids"] == ["b", "a"]
    assert candidate["n_selected"] == 2


def test_ranked_candidate_is_deterministic_on_ties() -> None:
    units = [
        {"unit_id": "z", "text": "zulu"},
        {"unit_id": "a", "text": "alpha"},
    ]

    candidate = ranked_candidate(
        units,
        {"z": 1.0, "a": 1.0},
        top_k=2,
    )

    assert candidate is not None
    assert candidate["selected_ids"] == ["a", "z"]
