"""Tests for final local Hugging Face result verification."""

import json
import tempfile

from scripts.verify_hf_results import (
    answer_from_response,
    load_evaluation_methods,
    paired_method_summary,
)


def test_hf_verification_parses_json_answer():
    assert answer_from_response('{"answer":"Paris"}') == "Paris"


def test_hf_verification_names_budget_methods_and_ignores_failures():
    row = {
        "id": "q1",
        "selection_mode": "budget",
        "token_budget": 3,
        "methods": {
            "residual_flow": {"status": "ok", "edits": []},
            "unary_matched": {"status": "replacement_registry_missing"},
        },
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl") as output:
        output.write(json.dumps(row) + "\n")
        output.flush()
        methods = load_evaluation_methods([output.name])

    assert set(methods["q1"]) == {"budget_b3:residual_flow"}


def test_paired_summary_reports_size_matched_flip_rates():
    rows = [
        {
            "clean_exact": True,
            "methods": {
                "flow": {"flip": True},
                "unary": {"flip": False},
            },
        },
        {
            "clean_exact": True,
            "methods": {
                "flow": {"flip": True},
                "unary": {"flip": True},
            },
        },
    ]

    result = paired_method_summary(rows, "flow", "unary")

    assert result["left_flips"] == 2
    assert result["right_flips"] == 1
    assert result["left_flip_rate"] == 1.0
    assert result["right_flip_rate"] == 0.5
