"""Tests for query-level residual-flow calibration utilities."""

from causalityrag.flow_calibration import (
    beta_statistics,
    choose_calibrated_beta,
    minimum_zero_nonflip_samples,
    select_threshold_candidate,
    stable_calibration_split,
)


def _candidate(ids, fraction, changed):
    return {
        "selected_ids": ids,
        "n_selected": len(ids),
        "remaining_support_fraction": fraction,
        "answer_changed": changed,
    }


def test_threshold_selection_uses_smallest_eligible_candidate():
    candidate = select_threshold_candidate(
        [
            _candidate(["a", "b"], 0.2, True),
            _candidate(["a"], 0.3, False),
            _candidate(["a", "b", "c"], 0.1, True),
        ],
        0.3,
    )
    assert candidate["selected_ids"] == ["a"]


def test_beta_statistics_counts_missing_candidates_as_overall_failures():
    rows = [
        {"id": "q1", "candidates": [_candidate(["a"], 0.1, True)]},
        {"id": "q2", "candidates": [_candidate(["b"], 0.7, True)]},
    ]
    result = beta_statistics(rows, 0.2)
    assert result["candidate_queries"] == 1
    assert result["conditional_flip_rate"] == 1.0
    assert result["overall_flip_rate"] == 0.5


def test_calibration_picks_largest_beta_that_meets_bound():
    rows = [
        {"id": f"q{index}", "candidates": [_candidate(["a"], 0.1, True)]}
        for index in range(200)
    ]
    beta, summaries = choose_calibrated_beta(
        rows,
        [0.1, 0.2],
        epsilon=0.2,
        delta=0.05,
    )
    assert beta == 0.2
    assert all(summary["meets_target"] for summary in summaries)


def test_split_is_deterministic_by_identifier():
    first = stable_calibration_split("hotpot-42", fraction=0.2, seed=7)
    second = stable_calibration_split("hotpot-42", fraction=0.2, seed=7)
    assert first == second


def test_zero_nonflip_sample_requirement_uses_the_beta_family_size():
    assert minimum_zero_nonflip_samples(
        epsilon=0.1,
        delta=0.05,
        candidate_count=11,
    ) == 270
