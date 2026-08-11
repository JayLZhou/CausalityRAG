import json
from pathlib import Path

from scripts.materialize_figure8_seed import (
    materialize_seed,
    parse_budget_values,
    parse_points,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_parse_points_supports_ranges_and_lists() -> None:
    assert parse_budget_values("1,3-5") == [1, 3, 4, 5]
    assert parse_points(["mirage=1,3-4", "arc_jsd=2"]) == {
        ("mirage", 1),
        ("mirage", 3),
        ("mirage", 4),
        ("arc_jsd", 2),
    }


def test_materialize_seed_is_aligned_and_reports_exact_missing_ids(
    tmp_path: Path,
) -> None:
    source = tmp_path / "saved.jsonl"
    write_jsonl(
        source,
        [
            {
                "id": "a",
                "methods": {
                    "MIRAGE": {
                        "1": {"edited_answer": "x"},
                        "2": {"edited_answer": "y"},
                    }
                },
            },
            {
                "id": "b",
                "methods": {"MIRAGE": {"1": {"edited_answer": "z"}}},
            },
        ],
    )
    records = [{"id": "a"}, {"id": "b"}, {"id": "c"}]

    rows, missing, sources = materialize_seed(
        records,
        {
            ("mirage", 1): source,
            ("mirage", 2): source,
        },
    )

    assert [row["id"] for row in rows] == ["a", "b", "c"]
    assert rows[0]["methods"]["mirage"]["2"]["edited_answer"] == "y"
    assert missing[("mirage", 1)] == ["c"]
    assert missing[("mirage", 2)] == ["b", "c"]
    assert list(sources.values())[0]["method_budgets"] == [
        "mirage:1",
        "mirage:2",
    ]
