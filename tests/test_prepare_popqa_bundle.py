from scripts.prepare_popqa_bundle import normalize_corpus, normalize_questions


def test_normalize_questions_preserves_aliases_and_uses_canonical_answer():
    rows = normalize_questions([{
        "question": "What is X's occupation?",
        "answers": ["politician", "political leader", "Politician", ""],
        "answer": "politician|political leader",
    }])

    assert rows == [{
        "id": "popqa-000000",
        "question": "What is X's occupation?",
        "answer": "politician",
        "answers": ["politician", "political leader"],
        "answer_aliases": ["political leader"],
        "answer_type": "entity",
    }]


def test_normalize_questions_falls_back_to_pipe_delimited_answer():
    rows = normalize_questions([{
        "question": "Where was X born?",
        "answer": "Paris|Paris, France",
    }])

    assert rows[0]["answer"] == "Paris"
    assert rows[0]["answers"] == ["Paris", "Paris, France"]


def test_normalize_corpus_maps_context_to_text():
    rows = normalize_corpus([{
        "id": 7,
        "title": "X",
        "context": "X is a person.",
    }])

    assert rows == [{
        "id": "popqa-doc-7",
        "title": "X",
        "text": "X is a person.",
    }]


def test_normalize_corpus_skips_empty_passages():
    assert normalize_corpus([{
        "id": 8,
        "title": "Category:British naturalists",
        "context": "",
    }]) == []
