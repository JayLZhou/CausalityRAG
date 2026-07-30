import json

from causalityrag.io import load_records


def test_load_jsonl_preserves_unicode_line_separators(tmp_path):
    path = tmp_path / "records.jsonl"
    records = [
        {"id": "first", "text": "medical\u0085abstract"},
        {"id": "second", "text": "line\u2028separator"},
    ]
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )

    assert load_records(path) == records
