from scripts.evaluate_matched_budget_baselines import completed_ranked_ids


def test_unscored_eligible_tokens_are_stable_lowest_rank_fallbacks() -> None:
    row = {"ranked_ids": ["u3", "u1"]}
    assert completed_ranked_ids(row, ["u1", "u2", "u3", "u4"]) == [
        "u3",
        "u1",
        "u2",
        "u4",
    ]
