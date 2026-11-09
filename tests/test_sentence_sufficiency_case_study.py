from scripts.run_sentence_sufficiency_case_study import (
    ranked_sentence_ids,
    sentence_contexts,
    sentence_ids_for_units,
)


def test_sentence_ids_for_units_deduplicates_in_token_order():
    mapping = {"u1": "s2", "u2": "s1", "u3": "s2"}
    assert sentence_ids_for_units(["u1", "u2", "u3"], mapping) == ["s2", "s1"]


def test_ranked_sentence_ids_matches_distinct_sentence_budget():
    mapping = {"u1": "s1", "u2": "s1", "u3": "s2", "u4": "s3"}
    assert ranked_sentence_ids(["u1", "u2", "u3", "u4"], mapping, 2) == [
        "s1",
        "s2",
    ]


def test_sentence_contexts_restore_canonical_retrieval_order():
    units_row = {
        "sentences": [
            {
                "sentence_id": "s2",
                "chunk_id": "c2",
                "chunk_rank": 2,
                "sentence_index": 0,
                "text": "Second chunk.",
            },
            {
                "sentence_id": "s1",
                "chunk_id": "c1",
                "chunk_rank": 1,
                "sentence_index": 1,
                "text": "First chunk.",
            },
        ]
    }
    retrieval = {
        "id": "q",
        "retrieved": [
            {"chunk_id": "c1", "title": "One", "text": "Full one"},
            {"chunk_id": "c2", "title": "Two", "text": "Full two"},
        ],
    }
    contexts = sentence_contexts(["s2", "s1"], units_row, retrieval)
    assert [context["text"] for context in contexts] == [
        "First chunk.",
        "Second chunk.",
    ]
    assert [context["title"] for context in contexts] == ["One", "Two"]
