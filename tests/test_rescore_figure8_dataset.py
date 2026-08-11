import json
from pathlib import Path

import pytest

from scripts.rescore_figure8_dataset import (
    discover_sources,
    prepare_population,
    rescore_sources,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_rescore_excludes_invalid_answers_and_respects_aliases(
    tmp_path: Path,
) -> None:
    records = [
        {
            "id": "a",
            "question": "Genre?",
            "answer": "J-pop",
            "answer_aliases": ["Japanese pop"],
        },
        {"id": "b", "question": "Capital?", "answer": "Paris"},
    ]
    reflow = [
        {
            "id": "a",
            "clean_answer": "Japanese pop",
            "edited_answer": "wrong",
            "evaluation_status": "verified_flip",
            "reader_calls": 1,
            "n_modified_tokens": 2,
        },
        {
            "id": "b",
            "clean_answer": "Paris",
            "edited_answer": "",
            "evaluation_status": "protocol_violation_invalid_reader_answer",
            "reader_calls": 1,
            "n_modified_tokens": 4,
        },
    ]
    source = tmp_path / "audits/baseline_budget_to_reflow/exact_points_v2/p.jsonl"
    write_jsonl(
        source,
        [
            {
                "id": "a",
                "methods": {
                    "MIRAGE": {
                        "1": {
                            "status": "acc_flip",
                            "edited_answer": "J-pop",
                            "n_modified_tokens": 1,
                        }
                    }
                },
            },
            {
                "id": "b",
                "methods": {
                    "MIRAGE": {
                        "1": {
                            "status": "protocol_violation_invalid_reader_answer",
                            "edited_answer": "",
                            "n_modified_tokens": 1,
                        }
                    }
                },
            },
        ],
    )

    population, reflow_summary = prepare_population(
        records, reflow, dataset="popqa", reader_mode="short_answer"
    )
    curves, _ = rescore_sources(
        discover_sources(tmp_path, set(population)),
        population,
        {"a", "b", "excluded-outside-clean-accuracy"},
        dataset="popqa",
        reader_mode="short_answer",
    )

    assert reflow_summary["clean_acc_queries"] == 2
    assert reflow_summary["valid_answer_queries"] == 1
    assert reflow_summary["acc_fr"] == 1.0
    assert reflow_summary["mean_modified_tokens"] == 2.0
    assert curves["mirage"]["1"]["valid_answer_queries"] == 1
    assert curves["mirage"]["1"]["acc_fr"] == 0.0
    assert curves["mirage"]["1"]["mean_modified_tokens"] == 1.0


def test_rescore_refuses_missing_final_population_entries(tmp_path: Path) -> None:
    records = [
        {"id": "a", "question": "Capital?", "answer": "Paris"},
        {"id": "b", "question": "Capital?", "answer": "Rome"},
    ]
    reflow = [
        {
            "id": "a",
            "clean_answer": "Paris",
            "edited_answer": "wrong",
            "evaluation_status": "verified_flip",
            "reader_calls": 1,
            "n_modified_tokens": 1,
        },
        {
            "id": "b",
            "clean_answer": "Rome",
            "edited_answer": "wrong",
            "evaluation_status": "verified_flip",
            "reader_calls": 1,
            "n_modified_tokens": 1,
        },
    ]
    source = tmp_path / "audits/baseline_budget_to_reflow/exact_points_v2/p.jsonl"
    write_jsonl(
        source,
        [
            {
                "id": "a",
                "methods": {
                    "mirage": {
                        "1": {
                            "status": "acc_flip",
                            "edited_answer": "wrong",
                            "n_modified_tokens": 1,
                        }
                    }
                },
            },
            {"id": "b", "methods": {}},
        ],
    )
    population, _ = prepare_population(
        records, reflow, dataset="hotpotqa", reader_mode="short_answer"
    )

    with pytest.raises(ValueError, match="lacks mirage/1"):
        rescore_sources(
            discover_sources(tmp_path, set(population)),
            population,
            {"a", "b"},
            dataset="hotpotqa",
            reader_mode="short_answer",
        )

    curves, _ = rescore_sources(
        discover_sources(tmp_path, set(population)),
        population,
        {"a", "b"},
        dataset="hotpotqa",
        reader_mode="short_answer",
        allow_missing=True,
    )
    assert curves["mirage"]["1"]["missing_population_queries"] == 1
    assert curves["mirage"]["1"]["missing_population_ids"] == ["b"]


def test_discovery_checks_method_budget_coverage_not_just_row_ids(
    tmp_path: Path,
) -> None:
    partial = (
        tmp_path
        / "audits/baseline_budget_to_reflow/exact_points_v2/partial.jsonl"
    )
    complete = tmp_path / "analysis/budget_sweep/coarse_results.jsonl"
    entry = {
        "status": "acc_flip",
        "edited_answer": "wrong",
        "n_modified_tokens": 1,
    }
    write_jsonl(
        partial,
        [
            {"id": "a", "methods": {"mirage": {"1": entry}}},
            {"id": "b", "methods": {}},
        ],
    )
    write_jsonl(
        complete,
        [
            {"id": "a", "methods": {"mirage": {"1": entry}}},
            {"id": "b", "methods": {"mirage": {"1": entry}}},
        ],
    )

    selected = discover_sources(tmp_path, {"a", "b"})

    assert selected[("mirage", 1)] == complete
