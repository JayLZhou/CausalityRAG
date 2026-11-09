from scripts.merge_shared_replacement_seeds import merge_seed_rows


def test_merge_seed_rows_keeps_most_complete_validated_row(tmp_path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first.write_text(
        '{"typed_key":"k","candidates":[{"new":"x"}]}\n',
        encoding="utf-8",
    )
    second.write_text(
        '{"typed_key":"k","candidates":[{"new":"x"},{"new":"y"}]}\n'
        '{"typed_key":"z","candidates":[{"new":"q"}]}\n',
        encoding="utf-8",
    )

    rows = merge_seed_rows([first, second])

    assert [row["typed_key"] for row in rows] == ["k", "z"]
    assert len(rows[0]["candidates"]) == 2
