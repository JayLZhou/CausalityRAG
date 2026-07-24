from exp.score_mirage_tokens import ranked_candidate


def test_ranked_candidate_uses_mirage_scores_and_allowed_domain() -> None:
    units = [
        {"unit_id": "a", "text": "A"},
        {"unit_id": "b", "text": "B"},
        {"unit_id": "c", "text": "C"},
    ]

    candidate = ranked_candidate(
        units,
        {"a": 1.0, "b": 3.0, "c": 2.0},
        top_tokens=2,
        allowed_ids={"a", "c"},
    )

    assert candidate["selected_ids"] == ["c", "a"]
    assert candidate["selected_scores"] == [2.0, 1.0]
