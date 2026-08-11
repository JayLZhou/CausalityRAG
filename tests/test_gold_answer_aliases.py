from causalityrag.io import gold_answers


def test_gold_answers_preserves_canonical_answer_and_distinct_aliases():
    assert gold_answers({
        "answer": "politician",
        "answers": ["politician", "political leader"],
        "answer_aliases": ["Political Leader", "political figure"],
    }) == ["politician", "political leader", "political figure"]
