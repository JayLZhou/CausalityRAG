import spacy

from causalityrag.replacement import (
    build_selected_replacements,
    generate_valid_replacement,
    validate_contextual_replacement,
)


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


def test_relaxed_fallback_keeps_all_chunk_tokens_endogenous():
    class EmptyLibrary:
        def replacement_for_unit(self, unit, context):
            return {"ok": False}

    class AuxiliaryEditor:
        def replace(self, *args, **kwargs):
            return {"ok": True, "old": "is", "new": "was", "policy": "test"}

    class StrictValidator:
        def validate(self, unit, context, replacement):
            return {"valid": False, "reason": "tag_mismatch"}

    result = generate_valid_replacement(
        {
            "text": "is",
            "type": "STOPWORD",
            "chunk_char_start": 3,
            "chunk_char_end": 5,
        },
        "It is blue.",
        EmptyLibrary(),
        AuxiliaryEditor(),
        StrictValidator(),
    )

    assert result["ok"]
    assert result["new"] == "was"
    assert result["validation"]["reason"] == "relaxed_all_tokens_endogenous"
    assert result["validation"]["strict_validation"]["reason"] == "tag_mismatch"


def test_selected_replacements_reuses_frozen_registry_entry():
    class FailingLibrary:
        def replacement_for_unit(self, unit, context):
            raise AssertionError("registry hit must not regenerate a replacement")

    selected = [{
        "unit_id": "c0:0:4",
        "chunk_id": "c0",
        "text": "Blue",
    }]
    frozen = {
        "c0:0:4": {
            "ok": True,
            "old": "Blue",
            "new": "Green",
            "policy": "frozen-test",
        }
    }

    replacements, rejected = build_selected_replacements(
        selected,
        [{"chunk_id": "c0", "text": "Blue sky."}],
        FailingLibrary(),
        None,
        None,
        frozen,
    )

    assert replacements == frozen
    assert rejected == []
