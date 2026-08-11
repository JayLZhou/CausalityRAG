from __future__ import annotations

from causalityrag.evaluation_metrics import (
    answer_changed,
    clean_correct,
    correctness_lost,
    gold_accuracy,
    gold_answer_spec,
    gold_exact,
    gold_f1,
    valid_answer,
)


def test_answer_change_is_gold_free_even_for_alias_switches() -> None:
    spec = gold_answer_spec(
        {
            "answer": "Copenhagen",
            "answers": ["Copenhagen", "Kopenhagen", "København"],
        },
        "popqa",
    )

    assert answer_changed("Kopenhagen", "Copenhagen")
    assert gold_f1("Kopenhagen", spec) == 1.0
    assert gold_exact("Kopenhagen", spec)
    assert not correctness_lost("f1", "Kopenhagen", "Copenhagen", spec)
    assert not correctness_lost("em", "Kopenhagen", "Copenhagen", spec)


def test_popqa_and_triviaqa_accept_any_frozen_alias() -> None:
    row = {
        "answer": "Sunset Boulevard",
        "answers": ["Sunset Boulevard", "Sunset Blvd", "Sunset Blvd."],
    }
    for dataset in ("popqa", "triviaqa"):
        spec = gold_answer_spec(row, dataset)
        assert gold_accuracy("Sunset Blvd.", spec)
        assert gold_exact("Sunset Blvd.", spec)
        assert clean_correct("acc", "Sunset Blvd.", spec)


def test_timeqa_requires_every_answer_component() -> None:
    spec = gold_answer_spec(
        {
            "answer": "Port F.C; Thailand national football team",
            "answers": ["Port F.C", "Thailand national football team"],
        },
        "timeqa",
    )

    assert not gold_accuracy("Port F.C", spec)
    assert gold_accuracy("Port F.C; Thailand national football team", spec)
    assert not clean_correct("f1", "Port F.C", spec)
    assert clean_correct(
        "f1", "Port F.C; Thailand national football team", spec
    )
    assert correctness_lost(
        "acc",
        "Port F.C; Thailand national football team",
        "Port F.C",
        spec,
    )


def test_quartz_accuracy_uses_choice_identity_not_substring() -> None:
    spec = gold_answer_spec({"answer": "more or"}, "quartz")
    assert not gold_accuracy("more", spec, reader_mode="quartz")


def test_invalid_answer_protocol_is_shared() -> None:
    for value in ("", "[]", "{}", "null", "None", "no answer found"):
        assert not valid_answer(value)
    assert valid_answer("Paris")
