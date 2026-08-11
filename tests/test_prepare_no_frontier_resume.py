from scripts.prepare_no_frontier_resume import merge_rows


def test_merge_rows_replaces_only_manifest_targets_and_restores_index():
    old_rows = [
        {"id": "keep", "index": 0, "value": "original"},
        {"id": "repair", "index": 1, "value": "old"},
        {"id": "also-keep", "index": 2, "value": "original"},
    ]
    resumed_rows = [{"id": "repair", "index": 0, "value": "new"}]
    manifest = {
        "target_ids": ["repair"],
        "original_index_by_id": {"keep": 0, "repair": 1, "also-keep": 2},
    }

    merged = merge_rows(old_rows, resumed_rows, manifest)

    assert merged[0] is old_rows[0]
    assert merged[1] == {"id": "repair", "index": 1, "value": "new"}
    assert merged[2] is old_rows[2]
