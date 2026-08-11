from scripts.prepare_targeted_table3_resume import (
    merge_by_target,
    validate_reflow_budgets,
    validate_resumed_baselines,
)


def test_validate_reflow_and_baseline_matched_budget():
    reflow = [{
        "id": "q1",
        "n_modified_tokens": 2,
        "selected_ids": ["u1", "u2"],
        "reader_calls": 1,
    }]
    resumed = [{
        "id": "q1",
        "matched_token_budget": 2,
        "reader_calls": 2,
        "methods": {
            "attention": {
                "selected_ids": ["u2", "u3"],
                "n_modified_tokens": 2,
                "reader_called": True,
            },
            "random_seed0": {
                "selected_ids": ["u4", "u5"],
                "n_modified_tokens": 2,
                "reader_called": True,
            },
        },
    }]

    validate_reflow_budgets(reflow, ["q1"])
    validate_resumed_baselines(
        resumed,
        reflow,
        ["q1"],
        {"attention", "random_seed0"},
    )


def test_merge_by_target_preserves_non_target_rows():
    untouched = {"id": "q2", "value": {"nested": [1, 2, 3]}}
    old = [{"id": "q1", "value": "old"}, untouched]
    resumed = [{"id": "q1", "value": "new"}]

    merged = merge_by_target(old, resumed, ["q1"])

    assert merged == [resumed[0], untouched]
    assert merged[1] is untouched


def test_validate_resumed_baselines_rejects_unmatched_token_count():
    reflow = [{
        "id": "q1",
        "n_modified_tokens": 2,
        "selected_ids": ["u1", "u2"],
        "reader_calls": 1,
    }]
    resumed = [{
        "id": "q1",
        "matched_token_budget": 2,
        "reader_calls": 1,
        "methods": {
            "attention": {
                "selected_ids": ["u1"],
                "n_modified_tokens": 1,
                "reader_called": True,
            },
        },
    }]

    try:
        validate_resumed_baselines(
            resumed,
            reflow,
            ["q1"],
            {"attention"},
        )
    except ValueError as exc:
        assert "token count mismatch" in str(exc)
    else:
        raise AssertionError("unmatched baseline budget was accepted")
