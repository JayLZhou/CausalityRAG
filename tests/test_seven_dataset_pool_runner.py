import json

from scripts.run_seven_dataset_pools import artifact_complete


def test_artifact_complete_requires_rows_and_matching_summary(tmp_path):
    rows = tmp_path / "rows.jsonl"
    summary = tmp_path / "summary.json"
    rows.write_text('{"id":"1"}\n{"id":"2"}\n', encoding="utf-8")
    summary.write_text(
        json.dumps({"queries": 2, "retrieval_top_k": 10}),
        encoding="utf-8",
    )

    assert artifact_complete(
        rows,
        summary,
        expected_rows=2,
        top_k_field="retrieval_top_k",
    )
    assert not artifact_complete(
        rows,
        summary,
        expected_rows=3,
        top_k_field="retrieval_top_k",
    )


def test_artifact_complete_rejects_stale_summary(tmp_path):
    rows = tmp_path / "rows.jsonl"
    summary = tmp_path / "summary.json"
    rows.write_text('{"id":"1"}\n', encoding="utf-8")
    summary.write_text(
        json.dumps({"queries": 1, "top_k": 5}),
        encoding="utf-8",
    )

    assert not artifact_complete(
        rows,
        summary,
        expected_rows=1,
        top_k_field="top_k",
    )
