import io
import json
from urllib.error import HTTPError

from scripts.prepare_dataset_retrieval import (
    frozen_queries,
    gold_titles,
    render_medqa_question,
    request_embedding_batch,
)


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


def test_medqa_queries_use_choice_letters_and_render_all_options() -> None:
    record = {
        "id": "m1",
        "question": "Which treatment is indicated?",
        "answer": "Treatment beta",
        "answer_idx": "B",
        "answer_type": "mcq",
        "options": {
            "A": "Treatment alpha",
            "B": "Treatment beta",
            "C": "Treatment gamma",
            "D": "Treatment delta",
        },
    }

    rows = frozen_queries([record], 1, dataset="medqa")

    assert rows[0]["answer"] == "B"
    assert rows[0]["answer_idx"] == "B"
    assert rows[0]["question"] == (
        "Which treatment is indicated?\n"
        "Options:\n"
        "(A) Treatment alpha\n"
        "(B) Treatment beta\n"
        "(C) Treatment gamma\n"
        "(D) Treatment delta"
    )


def test_medqa_query_rejects_incomplete_choice_sets() -> None:
    record = {
        "question": "Incomplete?",
        "answer_idx": "A",
        "options": {"A": "Only option"},
    }

    try:
        render_medqa_question(record)
    except ValueError as error:
        assert "A/B/C/D" in str(error)
    else:
        raise AssertionError("incomplete MedQA choices must fail closed")


def test_embedding_request_splits_server_rejected_batches(monkeypatch) -> None:
    calls = []

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.close()

    def fake_urlopen(request, timeout):
        payload = json.loads(request.data.decode("utf-8"))
        texts = payload["input"]
        calls.append(len(texts))
        if len(texts) > 2:
            raise HTTPError(request.full_url, 500, "too large", {}, None)
        return Response(json.dumps({
            "data": [
                {"index": index, "embedding": [float(len(text)), 1.0]}
                for index, text in enumerate(texts)
            ]
        }).encode("utf-8"))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    rows = request_embedding_batch(
        ["a", "bb", "ccc", "dddd"],
        base_url="http://embedding/v1",
        model="test",
    )

    assert calls == [4, 2, 2]
    assert rows == [[1.0, 1.0], [2.0, 1.0], [3.0, 1.0], [4.0, 1.0]]
