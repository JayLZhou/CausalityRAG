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


def test_quartz_factual_summary_uses_exact_choice_identity_for_accuracy() -> None:
    summary = _summarize(
        [
            (
                {"id": "q", "clean_answer": "more", "gold_answer": "more or"},
                {
                    "reader_called": True,
                    "status": "verified_no_flip",
                    "edited_answer": "more",
                },
            )
        ],
        reader_mode="quartz",
    )

    assert summary["acc_clean_correct_queries"] == 0
    assert summary["acc_flip_count"] == 0


def test_protocol_violation_is_not_a_valid_factual_answer() -> None:
    summary = _summarize(
        [
            (
                {"id": "q", "clean_answer": "harder", "gold_answer": "harder"},
                {
                    "reader_called": True,
                    "status": "protocol_violation_invalid_reader_answer",
                    "edited_answer": "probably harder",
                },
            )
        ],
        reader_mode="quartz",
    )

    assert summary["reader_executed_queries"] == 0
    assert summary["valid_answer_queries"] == 0
    assert summary["answer_flip_count"] == 0


def test_invalid_clean_answer_is_excluded_from_answer_denominator() -> None:
    summary = _summarize([
        (
            {
                "id": "valid",
                "clean_answer": "Paris",
                "gold_answer": "Paris",
                "evaluation_status": "verified_flip",
            },
            {
                "reader_called": True,
                "status": "verified_flip",
                "edited_answer": "London",
            },
        ),
        (
            {
                "id": "invalid",
                "clean_answer": "[]",
                "gold_answer": "Paris",
                "evaluation_status": "invalid_clean_answer",
            },
            {"reader_called": False, "edited_answer": ""},
        ),
    ])

    assert summary["total_queries"] == 2
    assert summary["answer_denominator_queries"] == 1
    assert summary["answer_flip_count"] == 1
    assert summary["answer_flip_ratio"] == 1.0


def test_popqa_alias_switch_is_not_a_correctness_flip() -> None:
    rows = [(
        {
            "id": "alias",
            "clean_answer": "Kopenhagen",
            "gold_answer": "Copenhagen",
            "gold_answers": ["Copenhagen", "Kopenhagen", "København"],
            "evaluation_status": "verified_flip",
        },
        {
            "reader_called": True,
            "status": "verified_flip",
            "edited_answer": "Copenhagen",
        },
    )]

    factual = _summarize(rows)
    assert factual["answer_flip_count"] == 1
    assert factual["f1_flip_count"] == 0
    assert factual["em_flip_count"] == 0
    assert factual["acc_flip_count"] == 0

    control = summarize_control([{
        "clean_answer": "Kopenhagen",
        "gold_answer": "Copenhagen",
        "gold_answers": ["Copenhagen", "Kopenhagen", "København"],
        "methods": {
            "reflow": {
                "reader_called": True,
                "status": "evaluated",
                "edited_answer": "Copenhagen",
                "answer_flip": True,
            }
        },
    }])["methods"]["reflow"]
    assert control["answer_flips"] == 1
    assert control["f1_flips"] == 0
    assert control["em_flips"] == 0
    assert control["acc_flips"] == 0


def test_quartz_control_summary_excludes_protocol_failures() -> None:
    rows = [
        {
            "clean_answer": "harder",
            "gold_answer": "harder",
            "methods": {
                "reflow": {
                    "reader_called": True,
                    "status": "protocol_violation_invalid_reader_answer",
                    "edited_answer": "probably harder",
                    "answer_flip": True,
                }
            },
        }
    ]

    method = summarize_control(rows, reader_mode="quartz")["methods"]["reflow"]

    assert method["executed_queries"] == 0
    assert method["valid_answer_queries"] == 0
    assert method["answer_flips"] == 0
