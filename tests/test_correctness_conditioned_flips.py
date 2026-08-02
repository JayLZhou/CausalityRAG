from __future__ import annotations

import pytest

from scripts.evaluate_paraphrase_controls import summarize as summarize_control
from scripts.summarize_clean_anchor_flips import _summarize


def test_factual_correctness_flips_use_only_clean_correct_queries() -> None:
    rows = [
        (
            {"id": "correct", "clean_answer": "Paris", "gold_answer": "Paris"},
            {"reader_called": True, "edited_answer": "London"},
        ),
        (
            {"id": "wrong", "clean_answer": "Rome", "gold_answer": "Paris"},
            {"reader_called": True, "edited_answer": "London"},
        ),
        (
            {"id": "unexecuted", "clean_answer": "Paris", "gold_answer": "Paris"},
            {"reader_called": False, "edited_answer": ""},
        ),
    ]

    summary = _summarize(rows)

    assert summary["total_queries"] == 3
    assert summary["f1_clean_correct_queries"] == 2
    assert summary["em_clean_correct_queries"] == 2
    assert summary["acc_clean_correct_queries"] == 2
    assert summary["answer_flip_ratio"] == pytest.approx(2 / 3)
    assert summary["f1_flip_ratio"] == pytest.approx(1 / 2)
    assert summary["em_flip_ratio"] == pytest.approx(1 / 2)
    assert summary["acc_flip_ratio"] == pytest.approx(1 / 2)


def test_control_summary_recomputes_degradation_from_saved_answers() -> None:
    rows = [
        {
            "clean_answer": "Paris",
            "gold_answer": "Paris",
            "methods": {
                "reflow": {
                    "reader_called": True,
                    "edited_answer": "London",
                    "f1_flip": False,
                }
            },
        },
        {
            "clean_answer": "Rome",
            "gold_answer": "Paris",
            "methods": {
                "reflow": {
                    "reader_called": True,
                    "edited_answer": "London",
                    "f1_flip": True,
                }
            },
        },
    ]

    method = summarize_control(rows)["methods"]["reflow"]

    assert method["f1_clean_correct_queries"] == 1
    assert method["em_clean_correct_queries"] == 1
    assert method["acc_clean_correct_queries"] == 1
    assert method["f1_flip_rate_itt"] == 1.0
    assert method["em_flip_rate_itt"] == 1.0
    assert method["acc_flip_rate_itt"] == 1.0
