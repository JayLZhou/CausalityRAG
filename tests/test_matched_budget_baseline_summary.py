import pytest

from scripts.evaluate_matched_budget_baselines import summarize


def test_token_summary_includes_failures_and_implicit_zero_budgets() -> None:
    rows = [
        {
            "id": "success",
            "matched_token_budget": 1,
            "methods": {
                "attention": {
                    "reader_called": True,
                    "verified_flip": True,
                    "n_modified_tokens": 1,
                }
            },
        },
        {
            "id": "failed",
            "matched_token_budget": 3,
            "methods": {
                "attention": {
                    "reader_called": False,
                    "verified_flip": False,
                    "n_modified_tokens": 3,
                }
            },
        },
        {
            "id": "zero-budget-omitted",
            "matched_token_budget": 0,
            "methods": {},
        },
        {
            "id": "zero-budget-empty",
            "matched_token_budget": 0,
            "methods": {"attention": {}},
        },
    ]

    method = summarize(rows)["methods"]["attention"]

    assert method["token_count_queries"] == 4
    assert method["zero_token_queries"] == 2
    assert method["mean_modified_tokens"] == 1.0


def test_token_summary_rejects_missing_positive_budget_count() -> None:
    rows = [
        {
            "id": "missing",
            "matched_token_budget": 2,
            "methods": {"attention": {}},
        }
    ]

    with pytest.raises(ValueError, match="positive-budget query missing"):
        summarize(rows)
