from scripts.generate_shared_typed_replacement_pool import (
    _contextual_unit,
    _generate_batch,
)


class FakeEditor:
    def __init__(self, label: str) -> None:
        self.label = label
        self.judge_called = False

    def generate_many(self, targets, *, max_candidates, attempt):
        return {target["unit_id"]: ["London"] for target in targets}

    def judge_many(self, targets, candidates):
        self.judge_called = True
        raise AssertionError("binary semantic judge must not be used")

    def classify_relations_many(self, targets, candidates):
        return {
            (target["unit_id"], candidate.casefold()): {"label": self.label}
            for target in targets
            for candidate in candidates.get(target["unit_id"], [])
        }


class FakeSpacy:
    def __init__(self, *, valid: bool = True) -> None:
        self.valid = valid

    def validate_many(self, proposals):
        reason = "ok" if self.valid else "pos_mismatch"
        return [{"valid": self.valid, "reason": reason} for _ in proposals]


def typed_row():
    return {
        "typed_key": "typed:paris",
        "surface": "Paris",
        "type": "GPE",
        "pos": "PROPN",
        "tag": "NNP",
        "morph": "",
        "examples": [
            {
                "marked_sentence": "The meeting was held in [[Paris]].",
                "entity_text": "Paris",
                "entity_token_index": 0,
            }
        ],
    }


def test_generation_accepts_only_counterfactual_relation():
    editor = FakeEditor("COUNTERFACTUAL")

    resolved, unresolved, calls = _generate_batch(
        [typed_row()],
        editor,
        FakeSpacy(),
        max_candidates=5,
        generation_rounds=1,
    )

    assert calls == 2
    assert not editor.judge_called
    assert not unresolved
    assert resolved[0]["candidates"][0]["new"] == "London"
    assert (
        resolved[0]["candidates"][0]["semantic_relation"]["label"]
        == "COUNTERFACTUAL"
    )


def test_generation_rejects_paraphrase_relation():
    editor = FakeEditor("PARAPHRASE")

    resolved, unresolved, _ = _generate_batch(
        [typed_row()],
        editor,
        FakeSpacy(),
        max_candidates=5,
        generation_rounds=1,
    )

    assert not resolved
    assert unresolved[0]["forbidden"] == ["London"]


def test_generation_carries_rejected_candidates_across_passes():
    editor = FakeEditor("PARAPHRASE")

    _, unresolved, _ = _generate_batch(
        [typed_row()],
        editor,
        FakeSpacy(),
        max_candidates=5,
        generation_rounds=1,
        initial_forbidden={"typed:paris": ["Berlin"]},
    )

    assert unresolved[0]["forbidden"] == ["Berlin", "London"]


def test_counterfactual_classifier_can_override_spacy_tag_error():
    editor = FakeEditor("COUNTERFACTUAL")

    resolved, unresolved, calls = _generate_batch(
        [typed_row()],
        editor,
        FakeSpacy(valid=False),
        max_candidates=5,
        generation_rounds=1,
    )

    assert calls == 2
    assert not unresolved
    candidate = resolved[0]["candidates"][0]
    assert candidate["new"] == "London"
    assert candidate["llm_grammar_override"] is True


def test_contextual_unit_uses_explicit_span_when_source_contains_brackets():
    row = typed_row()
    row["surface"] = "RMMA"
    row["examples"] = [{
        "marked_sentence": "activity [[[RMMA]]] was measured.",
        "sentence": "activity [RMMA] was measured.",
        "sentence_char_start": 10,
        "sentence_char_end": 14,
    }]

    unit, context = _contextual_unit(row)

    assert context == "activity [RMMA] was measured."
    assert context[unit["chunk_char_start"]:unit["chunk_char_end"]] == "RMMA"
