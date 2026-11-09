import hashlib

import pytest

from causalityrag.token_units import units_from_cache_row


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_top10_cache_can_serve_top5_prefix() -> None:
    contexts = [
        {"chunk_id": f"c{i}", "text": f"text {i}", "rank": i}
        for i in range(1, 11)
    ]
    record = {"id": "q1", "retrieved": contexts}
    cache = {
        "id": "q1",
        "top_k": 10,
        "context_sha256": {
            row["chunk_id"]: _digest(row["text"]) for row in contexts
        },
        "units": [
            {"unit_id": f"u{i}", "chunk_id": f"c{i}"}
            for i in range(1, 11)
        ],
    }

    units = units_from_cache_row(record, cache, k=5)

    assert [unit["unit_id"] for unit in units] == [f"u{i}" for i in range(1, 6)]


def test_cache_rejects_prefix_larger_than_cached_range() -> None:
    record = {
        "id": "q1",
        "retrieved": [
            {"chunk_id": f"c{i}", "text": f"text {i}", "rank": i}
            for i in range(1, 11)
        ],
    }
    cache = {"id": "q1", "top_k": 5, "units": []}

    with pytest.raises(ValueError, match="top-k mismatch"):
        units_from_cache_row(record, cache, k=10)


def test_cache_validates_hashes_for_requested_prefix() -> None:
    record = {
        "id": "q1",
        "retrieved": [
            {"chunk_id": "c1", "text": "current", "rank": 1},
            {"chunk_id": "c2", "text": "second", "rank": 2},
        ],
    }
    cache = {
        "id": "q1",
        "top_k": 2,
        "context_sha256": {"c1": _digest("stale"), "c2": _digest("second")},
        "units": [],
    }

    with pytest.raises(ValueError, match="context hash mismatch"):
        units_from_cache_row(record, cache, k=1)
