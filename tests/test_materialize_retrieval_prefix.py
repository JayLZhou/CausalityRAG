import hashlib
import json
from pathlib import Path

from scripts.materialize_retrieval_prefix import materialize


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_materialize_filters_nested_retrieval_units_and_pool(tmp_path: Path) -> None:
    records = []
    unit_rows = []
    pool_rows = [{"schema": "metadata", "row_kind": "metadata"}]
    for query_index in range(2):
        contexts = [
            {
                "chunk_id": f"q{query_index}-c{rank}",
                "rank": rank,
                "text": f"query {query_index} context {rank}",
            }
            for rank in range(1, 4)
        ]
        records.append({
            "id": f"q{query_index}",
            "question": "question",
            "retrieved": contexts,
        })
        units = [
            {
                "unit_id": f"token:{context['chunk_id']}:0:5",
                "chunk_id": context["chunk_id"],
            }
            for context in contexts
        ]
        unit_rows.append({
            "id": f"q{query_index}",
            "top_k": 3,
            "context_sha256": {
                context["chunk_id"]: digest(context["text"])
                for context in contexts
            },
            "units": units,
            "sentences": [
                {"sentence_id": f"s-{index}", "chunk_id": context["chunk_id"]}
                for index, context in enumerate(contexts)
            ],
        })
        pool_rows.extend(
            {
                "row_kind": "position_candidates",
                "unit_id": unit["unit_id"],
                "candidates": [{"old": "old", "new": "new"}],
            }
            for unit in units
        )

    retrieval = tmp_path / "retrieval.jsonl"
    units = tmp_path / "units.jsonl"
    pool = tmp_path / "pool.jsonl"
    write_jsonl(retrieval, records)
    write_jsonl(units, unit_rows)
    write_jsonl(pool, pool_rows)

    manifest = materialize(
        retrieval,
        units,
        pool,
        tmp_path / "prefix",
        n=2,
        k=1,
    )

    prefix_records = [
        json.loads(line)
        for line in (tmp_path / "prefix/retrieval.jsonl").read_text().splitlines()
    ]
    prefix_units = [
        json.loads(line)
        for line in (tmp_path / "prefix/units.jsonl").read_text().splitlines()
    ]
    prefix_pool = [
        json.loads(line)
        for line in (tmp_path / "prefix/shared_pool.jsonl").read_text().splitlines()
    ]

    assert all(len(row["retrieved"]) == 1 for row in prefix_records)
    assert all(row["top_k"] == 1 for row in prefix_units)
    assert all(len(row["context_sha256"]) == 1 for row in prefix_units)
    assert all(len(row["units"]) == 1 for row in prefix_units)
    assert all(len(row["sentences"]) == 1 for row in prefix_units)
    assert len(prefix_pool) == 3
    assert manifest["queries"] == 2
    assert manifest["retrieval_depth"] == 1
    assert manifest["prefix_pool_positions"] == 2
    assert manifest["outputs"]["shared_pool_sha256"]

    reused = materialize(
        retrieval,
        units,
        pool,
        tmp_path / "prefix",
        n=2,
        k=1,
        reuse_existing=True,
    )
    assert reused == manifest
