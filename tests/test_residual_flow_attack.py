from scripts.run_residual_flow_attack import (
    budget_candidate,
    summarize,
    threshold_candidate,
)


def test_threshold_candidate_uses_smallest_feasible_residual_cut():
    candidates = [
        {
            "n_selected": 0,
            "remaining_support_fraction": 1.0,
            "selected_ids": [],
        },
        {
            "n_selected": 2,
            "remaining_support_fraction": 0.7,
            "selected_ids": ["a", "b"],
        },
        {
            "n_selected": 3,
            "remaining_support_fraction": 0.1,
            "selected_ids": ["a", "b", "c"],
        },
    ]

    selected = threshold_candidate(candidates, 0.8, 10)

    assert selected is candidates[1]


def test_threshold_candidate_rejects_oversized_or_insufficient_cuts():
    candidates = [
        {
            "n_selected": 2,
            "remaining_support_fraction": 0.95,
            "selected_ids": ["a", "b"],
        },
        {
            "n_selected": 11,
            "remaining_support_fraction": 0.0,
            "selected_ids": [str(index) for index in range(11)],
        },
    ]

    assert threshold_candidate(candidates, 0.9, 10) is None


def test_budget_candidate_minimizes_flow_then_size_under_budget():
    candidates = [
        {
            "n_selected": 1,
            "remaining_support_fraction": 0.6,
            "selected_ids": ["a"],
        },
        {
            "n_selected": 2,
            "remaining_support_fraction": 0.2,
            "selected_ids": ["b", "c"],
        },
        {
            "n_selected": 3,
            "remaining_support_fraction": 0.0,
            "selected_ids": ["d", "e", "f"],
        },
    ]

    assert budget_candidate(candidates, 2) is candidates[1]


def test_budget_candidate_requires_a_nonempty_supported_set():
    candidates = [{
        "n_selected": 0,
        "remaining_support_fraction": 1.0,
        "selected_ids": [],
    }]

    assert budget_candidate(candidates, 2) is None


def test_summary_counts_no_candidate_as_failure_in_overall_rate():
    rows = [
        {
            "selection_mode": "budget",
            "token_budget": 1,
            "remaining_flow_threshold": 0.25,
            "methods": {
                "residual_flow": {
                    "status": "ok",
                    "flip": True,
                    "n_selected": 1,
                    "selected_ids": ["a"],
                },
                "unary_matched": {
                    "status": "ok",
                    "flip": False,
                    "n_selected": 1,
                    "selected_ids": ["b"],
                },
            },
        },
        {
            "selection_mode": "budget",
            "token_budget": 1,
            "remaining_flow_threshold": 0.25,
            "methods": {
                "residual_flow": {
                    "status": "no_candidate_under_selection_rule",
                    "flip": False,
                    "n_selected": 0,
                    "selected_ids": [],
                },
                "unary_matched": {
                    "status": "no_candidate_under_selection_rule",
                    "flip": False,
                    "n_selected": 0,
                    "selected_ids": [],
                },
            },
        },
    ]

    result = summarize(rows)
    assert result["residual_flow"]["flip_rate"] == 1.0
    assert result["residual_flow"]["overall_flip_rate"] == 0.5
    assert result["residual_flow"]["candidate_coverage"] == 0.5
    assert result["paired"]["flip_rate_difference"] == 1.0
    assert result["paired"]["paired_bootstrap_95_ci"] == [1.0, 1.0]
    assert result["paired"]["mcnemar_exact_two_sided_p"] == 1.0
