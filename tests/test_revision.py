from causalityrag.revision import apply_token_deletions, apply_token_replacements, apply_typed_token_revisions
from causalityrag.rules import TypedRuleLibrary


def test_typed_revision_uses_explicit_contextual_entity_type():
    record = {
        "id": "x",
        "question": "Where was Ada born?",
        "answer": "London",
        "retrieved": [{"chunk_id": "c1", "text": "Ada was born in London.", "rank": 1}],
    }
    unit = {
        "unit_id": "tok:4",
        "text": "London",
        "type": "GPE",
        "chunk_id": "c1",
        "chunk_char_start": 16,
        "chunk_char_end": 22,
        "support": 10.0,
    }
    library = TypedRuleLibrary({"type_pool": {"GPE": ["London", "Paris", "Boston"]}, "role_pool": {}})

    result = apply_typed_token_revisions(record, [unit], library, k=1)

    assert result["n_edits"] == 1
    assert result["edits"][0]["new"] != "London"
    assert result["edited_contexts"][0]["text"] != record["retrieved"][0]["text"]


def test_typed_revision_number_shift():
    record = {
        "id": "x",
        "question": "When?",
        "answer": "1947",
        "retrieved": [{"chunk_id": "c1", "text": "It happened in 1947.", "rank": 1}],
    }
    unit = {
        "unit_id": "tok:3",
        "text": "1947",
        "type": "NUMBER",
        "chunk_id": "c1",
        "chunk_char_start": 15,
        "chunk_char_end": 19,
        "support": 10.0,
    }
    library = TypedRuleLibrary({"type_pool": {}, "role_pool": {}})

    result = apply_typed_token_revisions(record, [unit], library, k=1)

    assert result["n_edits"] == 1
    assert "1947" not in result["edited_contexts"][0]["text"]


def test_token_deletion_edits_an_arbitrary_word_without_a_type_rule():
    record = {
        "id": "x",
        "question": "Question?",
        "retrieved": [{"chunk_id": "c1", "text": "ordinary wording remains", "rank": 1}],
    }
    unit = {
        "unit_id": "token:c1:9:16",
        "text": "wording",
        "chunk_id": "c1",
        "chunk_char_start": 9,
        "chunk_char_end": 16,
    }

    result = apply_token_deletions(record, [unit], k=1)

    assert result["n_edits"] == 1
    assert result["edits"][0]["note"] == "delete"
    assert result["edited_contexts"][0]["text"] == "ordinary  remains"


def test_token_replacement_edits_an_arbitrary_word_without_deleting_it():
    record = {
        "id": "x",
        "question": "Question?",
        "retrieved": [{"chunk_id": "c1", "text": "ordinary wording remains", "rank": 1}],
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
    assert result["edited_contexts"][0]["text"] == "ordinary language remains"
