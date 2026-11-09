import json
from pathlib import Path

import pytest

from scripts.prepare_medqa_textbook_corpus import (
    EXPECTED_BOOKS,
    convert,
    normalize_row,
)


def test_normalize_row_preserves_medrag_identity_and_content() -> None:
    assert normalize_row(
        {"id": "Book_0", "title": "Book", "content": "Clinical text."},
        source_file="Book.jsonl",
        source_line=1,
    ) == {
        "id": "Book_0",
        "title": "Book",
        "text": "Clinical text.",
        "source_file": "Book.jsonl",
        "source_line": 1,
    }


def test_convert_rejects_incomplete_textbook_snapshot(tmp_path: Path) -> None:
    chunk_dir = tmp_path / "chunk"
    chunk_dir.mkdir()
    (chunk_dir / "one.jsonl").write_text(
        json.dumps({"id": "one", "title": "One", "content": "Text"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=str(EXPECTED_BOOKS)):
        convert(
            chunk_dir,
            tmp_path / "corpus.jsonl",
            tmp_path / "manifest.json",
            source_revision="test",
        )
