import hashlib

from causalityrag.token_units import units_from_cache_row


def test_units_cache_checks_context_hashes():
    record = {
        "id": "q1",
        "retrieved": [{"chunk_id": "c1", "text": "alpha beta"}],
    }
    row = {
        "id": "q1",
        "top_k": 1,
        "context_sha256": {
            "c1": hashlib.sha256(b"alpha beta").hexdigest(),
        },
        "units": [{"unit_id": "u1"}],
    }

    assert units_from_cache_row(record, row, k=1) == [{"unit_id": "u1"}]

    row["context_sha256"]["c1"] = "bad"
    try:
        units_from_cache_row(record, row, k=1)
    except ValueError as error:
        assert "context hash mismatch" in str(error)
    else:
        raise AssertionError("a stale token-units cache must be rejected")
