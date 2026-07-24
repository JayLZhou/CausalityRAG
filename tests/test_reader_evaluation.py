"""Tests for reader evaluation of contribution-flow selections."""

from scripts.evaluate_reader import (
    run_reader_requests,
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

    selected = threshold_candidate(candidates, 0.8)

    assert selected is candidates[1]


def test_threshold_candidate_rejects_insufficient_cuts():
    candidates = [
        {
            "n_selected": 2,
            "remaining_support_fraction": 0.95,
            "selected_ids": ["a", "b"],
        },
    ]

    assert threshold_candidate(candidates, 0.9) is None


def test_summary_counts_no_candidate_as_failure_in_overall_rate():
    rows = [
        {
            "remaining_flow_threshold": 0.25,
            "clean_correct": True,
            "methods": {
                "residual_flow": {
                    "status": "ok",
                    "flip": True,
                    "n_selected": 1,
                    "selected_ids": ["a"],
                },
            },
        },
        {
            "remaining_flow_threshold": 0.25,
            "clean_correct": True,
            "methods": {
                "residual_flow": {
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
    assert result["query_scope"] == "clean_correct_only"
    assert result["clean_correct_queries"] == 2
    assert result["clean_incorrect_queries"] == 0


def test_reader_requests_preserve_input_order_under_concurrency():
    class FakeReader:
        @staticmethod
        def answer(question, contexts):
            return f"{question}:{contexts[0]['text']}"

    answers = run_reader_requests(
        FakeReader(),
        [
            ("q2", [{"text": "b"}]),
            ("q1", [{"text": "a"}]),
        ],
        workers=2,
    )

    assert answers == ["q2:b", "q1:a"]
