from causalityrag.revision import apply_token_replacements


def test_frozen_token_replacement_preserves_the_surrounding_context():
    record = {
        "id": "x",
        "question": "Question?",
        "retrieved": [
            {
                "chunk_id": "c1",
                "text": "ordinary wording remains",
                "rank": 1,
            }
        ],
    }
    unit = {
        "unit_id": "token:c1:9:16",
        "text": "wording",
        "chunk_id": "c1",
        "chunk_char_start": 9,
        "chunk_char_end": 16,
    }

    result = apply_token_replacements(
        record,
        [unit],
        {"token:c1:9:16": {"new": "language", "policy": "test"}},
        k=1,
    )

    assert result["n_edits"] == 1
    assert result["n_failed_edits"] == 0
    assert result["edited_contexts"][0]["text"] == "ordinary language remains"


def test_invalid_or_missing_replacement_fails_closed():
    record = {
        "id": "x",
        "question": "Question?",
        "retrieved": [{"chunk_id": "c1", "text": "Paris", "rank": 1}],
    }
    unit = {
        "unit_id": "token:c1:0:5",
        "text": "Paris",
        "chunk_id": "c1",
        "chunk_char_start": 0,
        "chunk_char_end": 5,
    }

    result = apply_token_replacements(record, [unit], {}, k=1)

    assert result["n_edits"] == 0
    assert result["n_failed_edits"] == 1
    assert result["edited_contexts"][0]["text"] == "Paris"


def test_paraphrase_control_can_replace_one_span_with_a_short_phrase():
    record = {
        "id": "x",
        "question": "Question?",
        "retrieved": [{"chunk_id": "c1", "text": "USA", "rank": 1}],
    }
    unit = {
        "unit_id": "token:c1:0:3",
        "text": "USA",
        "chunk_id": "c1",
        "chunk_char_start": 0,
        "chunk_char_end": 3,
    }

    result = apply_token_replacements(
        record,
        [unit],
        {"token:c1:0:3": {"new": "United States", "policy": "paraphrase"}},
        k=1,
        allow_whitespace=True,
    )

    assert result["n_edits"] == 1
    assert result["n_failed_edits"] == 0
    assert result["edited_contexts"][0]["text"] == "United States"


def test_paraphrase_control_can_use_an_orthographic_alias():
    record = {
        "id": "x",
        "question": "Question?",
        "retrieved": [{"chunk_id": "c1", "text": "McCoy", "rank": 1}],
    }
    unit = {
        "unit_id": "token:c1:0:5",
        "text": "McCoy",
        "chunk_id": "c1",
        "chunk_char_start": 0,
        "chunk_char_end": 5,
    }

    result = apply_token_replacements(
        record,
        [unit],
        {"token:c1:0:5": {"new": "MCCOY", "policy": "paraphrase"}},
        k=1,
        allow_case_only=True,
    )

    assert result["n_edits"] == 1
    assert result["n_failed_edits"] == 0
    assert result["edited_contexts"][0]["text"] == "MCCOY"
