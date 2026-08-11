import pytest

from scripts.evaluate_reflow import candidate_rows, summarize, valid_clean_answer


def test_frontier_candidates_are_verified_smallest_first():
    row = {
        "frontier_candidates": [
            {
                "n_selected": 3,
                "selected_ids": ["a", "b", "c"],
                "remaining_support_flow": 0.0,
            },
            {
                "n_selected": 1,
                "selected_ids": ["a"],
                "remaining_support_flow": 2.0,
            },
            {
                "n_selected": 2,
                "selected_ids": ["a", "b"],
                "remaining_support_flow": 1.0,
            },
        ]
    }

    assert [
        candidate["n_selected"] for candidate in candidate_rows(row)
    ] == [1, 2, 3]


def test_invalid_clean_answers_are_not_counted_as_reader_failures():
    assert valid_clean_answer("Paris")
    assert not valid_clean_answer("")
    assert not valid_clean_answer("[]")
    assert not valid_clean_answer("no answer found")


def test_invalid_edited_short_answers_use_the_same_protocol_guard():
    assert not valid_clean_answer("")
    assert not valid_clean_answer("null")
    assert not valid_clean_answer("no answer")


def test_summary_excludes_invalid_reader_answers_from_answer_denominator():
    summary = summarize([
        {
            "eligible": True,
            "verified_flip": True,
            "clean_correct": True,
            "n_modified_tokens": 2,
            "reader_calls": 1,
            "evaluation_status": "verified_flip",
        },
        {
            "eligible": True,
            "verified_flip": False,
            "clean_correct": True,
            "n_modified_tokens": 5,
            "reader_calls": 1,
            "evaluation_status": (
                "protocol_violation_invalid_reader_answer"
            ),
        },
    ])

    assert summary["eligible_queries"] == 2
    assert summary["valid_answer_queries"] == 1
    assert summary["answer_denominator_queries"] == 1
    assert summary["answer_flip_rate"] == pytest.approx(1.0)
    assert summary["mean_modified_tokens_all_queries"] == pytest.approx(2.0)
