from scripts.generate_shared_typed_replacement_pool import (
    _allow_contextual_single_letter_override,
    _is_format_preserving_numeric_counterfactual,
)


def test_format_preserving_cardinal_replacement_is_counterfactual():
    row = {"surface": "1.84-3.25", "type": "CARDINAL"}

    assert _is_format_preserving_numeric_counterfactual(row, "2.10-3.80")
    assert not _is_format_preserving_numeric_counterfactual(row, "2.1-3.8")
    assert not _is_format_preserving_numeric_counterfactual(row, "2.10")


def test_numeric_slot_in_multi_token_entity_is_counterfactual():
    row = {
        "surface": "450",
        "type": "PRODUCT",
        "entity_token_count": 2,
    }

    assert _is_format_preserving_numeric_counterfactual(row, "500")
    assert not _is_format_preserving_numeric_counterfactual(row, "500.0")
    assert not _is_format_preserving_numeric_counterfactual(
        {**row, "entity_token_count": 1},
        "500",
    )


def test_single_letter_proper_label_can_override_wordnet():
    row = {"surface": "B", "type": "PROPER"}

    assert _allow_contextual_single_letter_override(
        row,
        "C",
        "lexical_paraphrase",
    )
    assert not _allow_contextual_single_letter_override(
        {"surface": "film", "type": "NOUN"},
        "movie",
        "lexical_paraphrase",
    )
