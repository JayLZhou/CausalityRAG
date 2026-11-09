from scripts.run_security_aligned_guards import (
    anonymized_verifier_prompt,
    reflow_ranking,
)


def _units():
    return {
        "units": [
            {"unit_id": "u-clean", "sentence_id": "s-clean"},
            {"unit_id": "u-poison", "sentence_id": "s-poison"},
            {"unit_id": "u-other", "sentence_id": "s-other"},
        ],
        "sentences": [
            {
                "sentence_id": "s-clean",
                "chunk_id": "clean:1",
                "chunk_rank": 1,
                "sentence_index": 0,
                "text": "Clean evidence.",
            },
            {
                "sentence_id": "s-poison",
                "chunk_id": "poisonedrag:q:0",
                "chunk_rank": 2,
                "sentence_index": 0,
                "text": "Conflicting evidence.",
            },
            {
                "sentence_id": "s-other",
                "chunk_id": "clean:2",
                "chunk_rank": 3,
                "sentence_index": 0,
                "text": "Other evidence.",
            },
        ],
    }


def test_reflow_ranking_has_a_fixed_budget_and_no_attack_label() -> None:
    frontier = {
        "frontier_candidates": [
            {"n_selected": 1, "selected_ids": ["u-poison"]},
            {"n_selected": 2, "selected_ids": ["u-poison", "u-clean"]},
        ]
    }

    assert reflow_ranking(_units(), frontier, 2) == ["s-poison", "s-clean"]


def test_verifier_prompt_hides_original_chunk_identifiers() -> None:
    contexts = [
        {"chunk_id": "clean:1", "text": "Clean evidence."},
        {"chunk_id": "poisonedrag:q:0", "text": "Conflicting evidence."},
        {"chunk_id": "clean:2", "text": "Other evidence."},
    ]

    prompt, _, aliases = anonymized_verifier_prompt(
        question="Which claim is correct?",
        contexts=contexts,
        units_row=_units(),
        selected_ids=["s-poison", "s-clean"],
    )

    assert "poisonedrag" not in prompt
    assert "clean:1" not in prompt
    assert "[E2]" in prompt
    assert aliases == {"S1": "s-poison", "S2": "s-clean"}
