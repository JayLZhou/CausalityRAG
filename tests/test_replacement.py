import spacy

from causalityrag.replacement import validate_contextual_replacement


def test_contextual_validator_rejects_pos_change_and_accepts_matching_verb():
    nlp = spacy.load("en_core_web_lg")
    context = "Christopher McCoy plays Sam Docherty."
    unit = {
        "text": "plays",
        "type": "CONTENT",
        "chunk_char_start": 18,
        "chunk_char_end": 23,
    }
    rejected = validate_contextual_replacement(
        unit, context, {"new": "college"}, nlp
    )
    accepted = validate_contextual_replacement(
        unit, context, {"new": "portrays"}, nlp
    )
    assert not rejected["valid"]
    assert rejected["reason"] == "pos_mismatch"
    assert accepted["valid"]


def test_contextual_validator_rejects_common_noun_to_proper_noun():
    nlp = spacy.load("en_core_web_lg")
    context = "Sam is a boy who moved away."
    unit = {
        "text": "boy",
        "type": "CONTENT",
        "chunk_char_start": 9,
        "chunk_char_end": 12,
    }
    result = validate_contextual_replacement(unit, context, {"new": "York"}, nlp)
    assert not result["valid"]
    assert result["reason"] == "pos_mismatch"
