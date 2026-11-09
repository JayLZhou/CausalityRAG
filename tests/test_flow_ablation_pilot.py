import pytest

from scripts.run_flow_ablation_pilot import freeze_json, summarize


def test_summary_keeps_failed_queries_and_excludes_invalid_clean():
    records = {qid: {"id": qid, "answer": "Paris"} for qid in ("a", "b", "c")}
    rows = [
        {"id": "a", "clean_answer": "Paris", "reader_calls": 1, "methods": {
            "direct_score": {"reader_called": True, "edited_answer": "London", "status": "verified_flip"}}},
        {"id": "b", "clean_answer": "Paris", "reader_calls": 0, "methods": {}},
        {"id": "c", "clean_answer": "none", "reader_calls": 0, "methods": {}},
    ]
    result = summarize(rows, records, "hotpotqa")
    assert result["denominators"]["answer"] == 2
    assert result["rates"]["direct_score"]["answer"] == 0.5
    assert result["rates"]["individual_flow"]["answer"] == 0
    assert result["rates"]["direct_score"]["em"] == 0.5
    assert result["reader_calls"] == 1


def test_manifest_cannot_be_changed_on_resume(tmp_path):
    target = tmp_path / "manifest.json"
    freeze_json(target, {"ids": ["a"]})
    freeze_json(target, {"ids": ["a"]})
    with pytest.raises(ValueError, match="mismatch"):
        freeze_json(target, {"ids": ["b"]})
