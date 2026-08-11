import pytest

from scripts.evaluate_fixed_budget_acc import (
    answer_is_correct,
    rescore_seeded_entry,
    summarize,
    valid_clean_answer,
)


def test_popqa_correctness_matches_any_gold_alias() -> None:
    assert answer_is_correct(
        "What is this genre?",
        "Japanese pop",
        ["J-pop", "Japanese pop", "jpop"],
        reader_mode="short_answer",
    )


def test_quartz_correctness_uses_choice_identity() -> None:
    question = "Which is harder? Choices: (A) steel; (B) butter."
    assert answer_is_correct(
        question,
        "A",
        ["steel"],
        reader_mode="quartz",
    )
    assert not answer_is_correct(
        question,
        "B",
        ["steel"],
        reader_mode="quartz",
    )


@pytest.mark.parametrize("answer", ["", "[]", "no answer"])
def test_empty_short_answers_are_protocol_invalid(answer: str) -> None:
    assert not valid_clean_answer(answer)


def test_seeded_popqa_answer_is_rescored_against_all_aliases() -> None:
    rescored = rescore_seeded_entry(
        {"edited_answer": "Japanese pop", "acc_flip": True},
        question="What is the genre?",
        accepted_gold_answers=["J-pop", "Japanese pop", "jpop"],
        reader_mode="short_answer",
    )
    assert rescored["status"] == "acc_preserved"
    assert not rescored["acc_flip"]


def test_seeded_quartz_answer_is_rescored_as_choice() -> None:
    question = "Which is harder? Choices: (A) steel; (B) butter."
    rescored = rescore_seeded_entry(
        {"edited_answer": "(A) steel", "acc_flip": True},
        question=question,
        accepted_gold_answers=["steel"],
        reader_mode="quartz",
    )
    assert rescored["edited_answer"] == "steel"
    assert rescored["status"] == "acc_preserved"
    assert not rescored["acc_flip"]


def test_summary_excludes_invalid_reader_answers_from_acc_denominator() -> None:
    summary = summarize(
        [
            {
                "clean_acc": True,
                "reader_calls": 1,
                "methods": {
                    "baseline": {
                        "1": {
                            "status": "acc_flip",
                            "edited_answer": "London",
                            "acc_flip": True,
                            "n_modified_tokens": 1,
                        }
                    }
                },
            },
            {
                "clean_acc": True,
                "reader_calls": 1,
                "methods": {
                    "baseline": {
                        "1": {
                            "status": (
                                "protocol_violation_invalid_reader_answer"
                            ),
                            "edited_answer": "",
                            "acc_flip": False,
                            "n_modified_tokens": 1,
                        }
                    }
                },
            },
        ],
        [1],
    )

    result = summary["methods"]["baseline"]["1"]
    assert result["clean_acc_queries"] == 2
    assert result["valid_answer_queries"] == 1
    assert result["protocol_violations"] == 1
    assert result["acc_cfr"] == pytest.approx(1.0)
    assert result["mean_modified_tokens"] == pytest.approx(1.0)
