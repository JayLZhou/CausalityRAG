import json

import pytest

from causalityrag.shared_replacement_pool import (
    POOL_SCHEMA,
    FrozenSharedReplacementPool,
    is_editable_unit,
    stable_shared_candidate,
    typed_pool_key,
)


def test_typed_key_is_stable_and_type_sensitive():
    unit = {
        "text": "Washington",
        "type": "GPE",
        "pos": "PROPN",
        "tag": "NNP",
        "morph": {"Number": "Sing"},
    }
    assert typed_pool_key(unit) == typed_pool_key(dict(unit))
    assert typed_pool_key(unit) != typed_pool_key({**unit, "type": "PERSON"})
    assert typed_pool_key(unit) != typed_pool_key({
        **unit,
        "entity_token_index": 1,
        "entity_token_count": 2,
    })


def test_frozen_pool_fails_closed_on_missing_unit(tmp_path):
    path = tmp_path / "pool.jsonl"
    row = {
        "schema": POOL_SCHEMA,
        "unit_id": "token:c1:0:5",
        "typed_key": "typed:key",
        "candidates": [{"new": "London"}],
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    pool = FrozenSharedReplacementPool(str(path))

    assert pool.require(["token:c1:0:5"])["token:c1:0:5"] == row
    with pytest.raises(KeyError):
        pool.require(["token:c1:6:10"])


def test_shared_candidate_does_not_depend_on_method_or_query():
    candidates = [{"new": "London"}, {"new": "Berlin"}, {"new": "Rome"}]
    first = stable_shared_candidate(
        candidates,
        unit_id="token:c1:0:5",
        seed=3,
    )
    second = stable_shared_candidate(
        candidates,
        unit_id="token:c1:0:5",
        seed=3,
    )
    assert first == second


def test_editable_domain_excludes_nonsemantic_types():
    assert is_editable_unit({"text": "Paris", "pos": "PROPN", "type": "GPE"})
    assert not is_editable_unit(
        {"text": "Other", "pos": "ADJ", "type": "STOPWORD"}
    )


def test_frozen_pool_tracks_common_excluded_positions(tmp_path):
    path = tmp_path / "pool.jsonl"
    rows = [
        {
            "schema": POOL_SCHEMA,
            "row_kind": "position_candidates",
            "unit_id": "token:c1:0:5",
            "candidates": [{"new": "London"}],
        },
        {
            "schema": POOL_SCHEMA,
            "row_kind": "excluded_position",
            "unit_id": "token:c1:6:7",
            "candidates": [],
            "reason": "no_legal_counterfactual",
        },
    ]
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    pool = FrozenSharedReplacementPool(str(path))

    assert pool.is_eligible("token:c1:0:5")
    assert pool.is_excluded("token:c1:6:7")
    assert not pool.is_eligible("token:c1:6:7")
