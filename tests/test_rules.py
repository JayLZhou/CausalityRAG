from causalityrag.rules import TypedRuleLibrary


def test_entity_slots_use_one_coherent_replacement_entity() -> None:
    library = TypedRuleLibrary({
        "type_pool": {"PERSON": ["Anthony Rapp", "Mary Smith", "John Brown"]},
        "role_pool": {},
    })
    replacements = tuple(
        library.replacement_for_unit({
            "text": token,
            "type": "PERSON",
            "entity_text": "Anthony Rapp",
            "entity_token_index": slot,
            "entity_token_count": 2,
        })["new"]
        for slot, token in enumerate(("Anthony", "Rapp"))
    )

    assert replacements in {("Mary", "Smith"), ("John", "Brown")}
