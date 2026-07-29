from causalityrag.semantic_filter import is_lexical_paraphrase


def test_entity_types_are_not_rejected_as_lexical_paraphrases():
    assert not is_lexical_paraphrase("Vermont", "Maine", "PROPN", "GPE")
    assert not is_lexical_paraphrase("Paris", "London", "PROPN", "GPE")


def test_identical_values_are_always_rejected():
    assert is_lexical_paraphrase("film", "FILM", "NOUN", "CONTENT")
