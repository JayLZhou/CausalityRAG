from causalityrag.reflow_results import (
    canonicalize_reflow_row,
    summarize_reflow_rows,
)


def test_failed_query_records_last_executed_intervention():
    row = canonicalize_reflow_row(
        {
            "clean_answer": "clean",
            "frontier_candidates": [{"selected_ids": ["a"]}],
            "attempts": [
                {
                    "selected_ids": [],
                    "selected_tokens": [],
                    "reader_called": False,
                    "answer_changed": False,
                },
                {
                    "selected_ids": ["b", "c"],
                    "selected_tokens": ["B", "C"],
                    "n_edits": 2,
                    "reader_called": True,
                    "answer_changed": False,
                    "edited_answer": "clean",
                },
            ],
        }
    )

    assert row["evaluation_status"] == "verified_no_flip"
    assert row["verified_failure"]
    assert row["reader_executed"]
    assert row["selected_ids"] == ["b", "c"]
    assert row["n_selected"] == 2
    assert row["n_modified_tokens"] == 2


def test_missing_replacement_is_not_a_verified_failure():
    row = canonicalize_reflow_row(
        {
            "clean_answer": "clean",
            "frontier_candidates": [{"selected_ids": ["a"]}],
            "attempts": [
                {
                    "selected_ids": [],
                    "reader_called": False,
                    "answer_changed": False,
                }
            ],
        }
    )

    assert row["evaluation_status"] == "unsupported_no_executable_intervention"
    assert not row["verified_failure"]
    assert not row["reader_executed"]


def test_mean_modified_tokens_includes_verified_failures():
    rows = [
        {
            "clean_answer": "a",
            "attempts": [
                {
                    "selected_ids": ["x"],
                    "n_edits": 1,
                    "reader_called": True,
                    "answer_changed": True,
                }
            ],
        },
        {
            "clean_answer": "b",
            "attempts": [
                {
                    "selected_ids": ["x", "y", "z"],
                    "n_edits": 3,
                    "reader_called": True,
                    "answer_changed": False,
                }
            ],
        },
    ]

    summary = summarize_reflow_rows(rows)

    assert summary["evaluated_queries"] == 2
    assert summary["verified_failures"] == 1
    assert summary["mean_modified_tokens"] == 2.0
    assert summary["evaluated_mean_modified_tokens"] == 2.0
    assert summary["failure_mean_tokens"] == 3.0
    assert summary["failure_min_tokens"] == 3
    assert summary["failure_max_tokens"] == 3
    assert summary["success_mean_tokens"] == 1.0
