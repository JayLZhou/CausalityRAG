import json
from pathlib import Path

from scripts.merge_contribution_graph_shards import merge


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_merge_restores_frozen_retrieval_order(tmp_path: Path) -> None:
    retrieval = tmp_path / "retrieval.jsonl"
    left = tmp_path / "left.jsonl"
    right = tmp_path / "right.jsonl"
    output = tmp_path / "graph.jsonl"
    summary = tmp_path / "summary.json"
    write_jsonl(retrieval, [{"id": "a"}, {"id": "b"}])
    write_jsonl(left, [{"id": "b", "status": "ok", "elapsed_seconds": 2}])
    write_jsonl(right, [{"id": "a", "status": "ok", "elapsed_seconds": 4}])

    report = merge(retrieval, [left, right], output, summary, n=2)

    assert [json.loads(line)["id"] for line in output.read_text().splitlines()] == ["a", "b"]
    assert report["ok"] == 2
    assert report["avg_seconds"] == 3.0
