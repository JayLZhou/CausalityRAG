from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.render_cross_model_sensitivity import ModelSpec, render_values, summarize_dataset


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_summarize_dataset_uses_all_query_pairs(tmp_path: Path) -> None:
    base = tmp_path / "hotpotqa"
    write_jsonl(
        base / "factual/results.jsonl",
        [
            {
                "id": "a",
                "eligible": True,
                "clean_answer": "alpha",
                "gold_answer": "alpha",
                "edited_answer": "wrong",
                "evaluation_status": "verified_flip",
                "verified_flip": True,
                "n_modified_tokens": 2,
            },
            {
                "id": "b",
                "eligible": True,
                "clean_answer": "beta",
                "gold_answer": "beta",
                "edited_answer": "beta",
                "evaluation_status": "verified_no_flip",
                "verified_flip": False,
                "n_modified_tokens": 4,
            },
            {
                "id": "no_op_control",
                "eligible": True,
                "clean_answer": "gamma",
                "gold_answer": "gamma",
                "edited_answer": "wrong",
                "evaluation_status": "verified_flip",
                "verified_flip": True,
                "n_modified_tokens": 6,
            },
        ],
    )
    write_jsonl(
        base / "synonym/results.jsonl",
        [
            {
                "id": "a",
                "methods": {
                    "reflow": {"status": "evaluated", "answer_flip": False, "f1_flip": False}
                },
            },
            {
                "id": "b",
                "methods": {
                    "reflow": {"status": "evaluated", "answer_flip": True, "f1_flip": True}
                },
            },
            {
                "id": "no_op_control",
                "methods": {"reflow": {"status": "missing_paraphrase"}},
            },
        ],
    )
    (base / "graph").mkdir(parents=True)
    (base / "graph/summary.json").write_text(
        json.dumps({"records": 3, "ok": 2, "avg_seconds": 1.25}), encoding="utf-8"
    )

    summary = summarize_dataset(
        ModelSpec("m", "Model", "M", tmp_path, "sweep"),
        "hotpotqa",
        retrieval_records=[
            {"id": "a", "answer": "alpha"},
            {"id": "b", "answer": "beta"},
            {"id": "no_op_control", "answer": "gamma"},
        ],
    )

    assert summary["paired_queries"] == 3
    assert summary["f1_clean_paired_queries"] == 3
    assert summary["ans_cfr"] == 1 / 3
    assert summary["f1_cfr"] == 1 / 3
    assert summary["mean_modified_tokens"] == 4.0
    assert summary["token_budget_queries"] == 3
    assert summary["graph_coverage"] == 2 / 3


def test_render_values_uses_dataset_level_results() -> None:
    datasets = [
        {
            "dataset": dataset,
            "ans_cfr": 0.8 - index * 0.1,
            "f1_cfr": 0.6 - index * 0.1,
        }
        for index, dataset in enumerate(("hotpotqa", "finqa", "triviaqa", "popqa"))
    ]
    output = render_values([
        {
            "key": "qwen",
            "short_label": "Q25",
            "datasets": datasets,
        }
    ])

    assert "\\CrossModelHotpotAnsCoordinates{(1,80.00)}" in output
    assert "\\CrossModelFinQAFOneCoordinates{(1,50.00)}" in output
    assert "\\CrossModelTriviaAnsCoordinates{(1,60.00)}" in output
    assert "\\CrossModelPopQAAnsCoordinates{(1,50.00)}" in output
    assert "CrossModelTokenCoordinates" not in output


def test_popqa_aliases_and_invalid_answers_define_denominators(
    tmp_path: Path,
) -> None:
    base = tmp_path / "popqa"
    write_jsonl(
        base / "factual/results.jsonl",
        [
            {
                "id": "valid-alias",
                "eligible": True,
                "clean_answer": "accepted alias",
                "gold_answer": "canonical",
                "gold_answers": ["canonical", "accepted alias"],
                "edited_answer": "wrong",
                "evaluation_status": "verified_flip",
                "reader_calls": 1,
                "n_modified_tokens": 2,
            },
            {
                "id": "invalid-factual",
                "eligible": True,
                "clean_answer": "canonical",
                "gold_answers": ["canonical"],
                "edited_answer": "",
                "evaluation_status": "protocol_violation_invalid_reader_answer",
                "reader_calls": 1,
                "n_modified_tokens": 3,
            },
            {
                "id": "invalid-control",
                "eligible": True,
                "clean_answer": "canonical",
                "gold_answers": ["canonical"],
                "edited_answer": "canonical",
                "evaluation_status": "verified_no_flip",
                "reader_calls": 1,
                "n_modified_tokens": 4,
            },
        ],
    )
    write_jsonl(
        base / "synonym/results.jsonl",
        [
            {
                "id": "valid-alias",
                "methods": {"reflow": {"status": "no_selected_tokens", "reader_called": False}},
            },
            {
                "id": "invalid-factual",
                "methods": {"reflow": {"status": "no_selected_tokens", "reader_called": False}},
            },
            {
                "id": "invalid-control",
                "methods": {
                    "reflow": {
                        "status": "invalid_empty_answer",
                        "reader_called": True,
                        "edited_answer": "",
                    }
                },
            },
        ],
    )
    (base / "graph").mkdir(parents=True)
    (base / "graph/summary.json").write_text(
        json.dumps({"records": 3, "ok": 3}), encoding="utf-8"
    )

    summary = summarize_dataset(
        ModelSpec("m", "Model", "M", tmp_path, "sweep"),
        "popqa",
        retrieval_records=[
            {
                "id": "valid-alias",
                "answer": "canonical",
                "answers": ["canonical", "accepted alias"],
            },
            {"id": "invalid-factual", "answer": "canonical"},
            {"id": "invalid-control", "answer": "canonical"},
        ],
    )

    assert summary["raw_paired_queries"] == 3
    assert summary["paired_queries"] == 1
    assert summary["excluded_invalid_answer_pairs"] == 2
    assert summary["f1_clean_paired_queries"] == 1
    assert summary["ans_cfr"] == 1.0
    assert summary["f1_cfr"] == 1.0
    assert summary["token_budget_queries"] == 3
    assert summary["mean_modified_tokens"] == 3.0


def test_popqa_alias_switch_counts_for_ans_but_not_f1(
    tmp_path: Path,
) -> None:
    base = tmp_path / "popqa"
    write_jsonl(
        base / "factual/results.jsonl",
        [
            {
                "id": "alias-switch",
                "eligible": True,
                "clean_answer": "Conservative Party politician",
                "gold_answer": "Conservative Party politician",
                "gold_answers": ["Conservative Party politician", "politician"],
                "edited_answer": "politician",
                "evaluation_status": "verified_flip",
                "reader_calls": 1,
                "n_modified_tokens": 2,
            }
        ],
    )
    write_jsonl(
        base / "synonym/results.jsonl",
        [
            {
                "id": "alias-switch",
                "methods": {
                    "reflow": {
                        "status": "evaluated",
                        "reader_called": True,
                        "edited_answer": "Conservative Party politician",
                    }
                },
            }
        ],
    )
    (base / "graph").mkdir(parents=True)
    (base / "graph/summary.json").write_text(
        json.dumps({"records": 1, "ok": 1}), encoding="utf-8"
    )

    summary = summarize_dataset(
        ModelSpec("m", "Model", "M", tmp_path, "sweep"),
        "popqa",
        retrieval_records=[{
            "id": "alias-switch",
            "answer": "Conservative Party politician",
            "answers": ["Conservative Party politician", "politician"],
        }],
    )

    assert summary["paired_queries"] == 1
    assert summary["f1_clean_paired_queries"] == 1
    assert summary["ans_delta_sum"] == 1
    assert summary["f1_delta_sum"] == 0
    assert summary["ans_cfr"] == 1.0
    assert summary["f1_cfr"] == 0.0


def test_quartz_summary_refuses_same_choice_string_flips(tmp_path: Path) -> None:
    base = tmp_path / "quartz"
    write_jsonl(
        base / "factual/results.jsonl",
        [
            {
                "id": "q1",
                "clean_answer": "A",
                "gold_answer": "harder",
                "edited_answer": "harder",
                "verified_flip": True,
                "reader_calls": 1,
                "attempts": [
                    {
                        "reader_called": True,
                        "edited_answer": "harder",
                        "answer_changed": True,
                    }
                ],
                "n_modified_tokens": 1,
            }
        ],
    )
    write_jsonl(
        base / "synonym/results.jsonl",
        [
            {
                "id": "q1",
                "clean_answer": "A",
                "gold_answer": "harder",
                "methods": {
                    "reflow": {
                        "status": "evaluated",
                        "reader_called": True,
                        "edited_answer": "harder",
                        "answer_flip": True,
                    }
                },
            }
        ],
    )
    (base / "graph").mkdir(parents=True)
    (base / "graph/summary.json").write_text(
        json.dumps({"records": 1, "ok": 1}), encoding="utf-8"
    )
    retrieval = [
        {
            "id": "q1",
            "question": "Question? Choices: (A) harder; (B) easier.",
            "answer": "harder",
        }
    ]

    with pytest.raises(ValueError, match="require choice-level repair"):
        summarize_dataset(
            ModelSpec("m", "Model", "M", tmp_path, "sweep"),
            "quartz",
            retrieval_records=retrieval,
            quartz_records=retrieval,
        )
