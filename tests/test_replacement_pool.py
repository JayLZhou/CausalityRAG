import json

from causalityrag.replacement_pool import ReplacementPool


def test_pool_persists_and_reuses_query_replacements(tmp_path):
    path = tmp_path / "replacements.jsonl"
    pool = ReplacementPool(str(path))
    cache = pool.cache_for("q1")
    cache["c0:0:5"] = {
        "ok": True,
        "old": "Paris",
        "new": "London",
        "policy": "online_llm_counterfactual",
    }
    pool.persist("q1", cache, source="test")

    reopened = ReplacementPool(str(path))
    assert reopened.cache_for("q1")["c0:0:5"]["new"] == "London"
    row = json.loads(path.read_text(encoding="utf-8").strip())
    assert row["generation_mode"] == "on_demand_after_selection"
    assert row["valid_tokens"] == 1


def test_pool_merges_new_items_without_losing_existing_rows(tmp_path):
    path = tmp_path / "replacements.jsonl"
    path.write_text(
        json.dumps({
            "id": "q0",
            "replacements": {"u0": {"ok": True, "new": "old"}},
        })
        + "\n",
        encoding="utf-8",
    )
    pool = ReplacementPool(str(path))
    pool.persist(
        "q1",
        {"u1": {"ok": False, "policy": "skip_after_selection"}},
        source="baseline",
    )

    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["id"] for row in rows] == ["q0", "q1"]
    assert rows[0]["replacements"]["u0"]["new"] == "old"
    assert rows[1]["invalid_tokens"] == 1
