from scripts.prepare_dataset_retrieval import frozen_queries, gold_titles


def test_frozen_queries_skip_empty_gold_and_preserve_source_order() -> None:
    records = [
        {"id": "bad", "question": "Q?", "answer": ""},
        {
            "id": "q1",
            "question": "First?",
            "answer": "A1",
            "support_title": ["Doc 1"],
        },
        {"_id": "q2", "question": "Second?", "answer": "A2"},
    ]

    rows = frozen_queries(records, 2)

    assert [row["id"] for row in rows] == ["q1", "q2"]
    assert rows[0]["gold_titles"] == ["Doc 1"]


def test_gold_titles_supports_multihop_formats() -> None:
    row = {
        "supporting_facts": [["Alpha", 0], ["Beta", 2]],
        "paragraphs": [
            {"title": "Gamma", "is_supporting": True},
            {"title": "Delta", "is_supporting": False},
        ],
    }

    assert gold_titles(row) == ["Alpha", "Beta", "Gamma"]
