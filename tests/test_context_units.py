from causalityrag.token_units import (
    all_context_word_units,
    context_sentence_units,
)


def test_all_context_word_units_includes_stopwords_but_not_punctuation():
    record = {
        "retrieved": [
            {"chunk_id": "c1", "text": "Ada was here.", "rank": 1}
        ],
    }

    units = all_context_word_units(record, k=1)

    assert [unit["text"] for unit in units] == ["Ada", "was", "here"]


def test_context_sentence_units_assigns_every_word_to_a_sentence():
    record = {
        "retrieved": [
            {"chunk_id": "c1", "text": "Ada stayed. Bob left!", "rank": 1}
        ],
    }

    units, sentences = context_sentence_units(record, k=1)

    assert len(sentences) == 2
    assert len({unit["sentence_id"] for unit in units}) == 2
