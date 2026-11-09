from scripts.summarize_security_case_study import summarize_method
from scripts.run_security_guard_keyword import (
    random_inspection,
    reflow_inspection,
)


def test_asr_uses_all_clean_correct_queries_as_denominator() -> None:
    rows = [
        (
            {"id": "success", "clean_answer": "Paris", "gold_answer": "Paris"},
            {
                "reader_called": True,
                "edited_answer": "London",
                "n_modified_tokens": 1,
                "reader_calls": 1,
            },
        ),
        (
            {"id": "still-correct", "clean_answer": "Paris", "gold_answer": "Paris"},
            {
                "reader_called": True,
                "edited_answer": "Paris",
                "n_modified_tokens": 2,
                "reader_calls": 1,
            },
        ),
        (
            {"id": "unexecuted", "clean_answer": "Paris", "gold_answer": "Paris"},
            {"reader_called": False},
        ),
        (
            {"id": "clean-wrong", "clean_answer": "Rome", "gold_answer": "Paris"},
            {
                "reader_called": True,
                "edited_answer": "London",
                "n_modified_tokens": 1,
                "reader_calls": 1,
            },
        ),
    ]

    summary = summarize_method(rows, max_examples=2)

    assert summary["eligible_clean_correct_queries"] == 3
    assert summary["eligible_clean_exact_match_queries"] == 3
    assert summary["executed_queries"] == 2
    assert summary["attack_successes"] == 1
    assert summary["untargeted_asr"] == 1 / 3
    assert summary["untargeted_em_asr"] == 1 / 3
    assert summary["reader_calls"] == 2
    assert summary["mean_modified_tokens_on_executed"] == 1.5
    assert [row["id"] for row in summary["examples"]] == ["success"]


def test_reflow_guard_inspects_attacked_frontier_until_poison_is_found() -> None:
    units = {
        "units": [
            {"unit_id": "clean-token", "sentence_id": "clean-sentence"},
            {"unit_id": "poison-token", "sentence_id": "poison-sentence"},
        ],
        "sentences": [
            {"sentence_id": "clean-sentence", "chunk_id": "clean:1"},
            {
                "sentence_id": "poison-sentence",
                "chunk_id": "poisonedrag:query:0",
            },
        ],
    }
    frontier = {
        "frontier_candidates": [
            {
                "n_selected": 1,
                "selected_ids": ["clean-token"],
                "remaining_support_flow": 2.0,
            },
            {
                "n_selected": 2,
                "selected_ids": ["clean-token", "poison-token"],
                "remaining_support_flow": 0.0,
            },
        ]
    }

    result = reflow_inspection(units, frontier, "poisonedrag")

    assert result["detected_chunk"] == "poisonedrag:query:0"
    assert result["inspected_sentence_ids"] == [
        "clean-sentence",
        "poison-sentence",
    ]
    assert result["verified_sentences"] == 2


def test_random_guard_uses_exactly_the_reflow_sentence_budget() -> None:
    units = {
        "sentences": [
            {"sentence_id": f"s{index}", "chunk_id": f"clean:{index}"}
            for index in range(5)
        ]
    }

    result = random_inspection(
        units,
        "poisonedrag",
        3,
        identifier="query",
        seed=11,
    )

    assert result["verified_sentences"] == 3
    assert len(result["inspected_sentence_ids"]) == 3
