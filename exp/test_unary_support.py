from exp.build_unary_support import answer_from_response, summarize


def test_answer_from_response_reads_strict_json_answer():
    assert answer_from_response('{"answer": "Paris"}') == "Paris"


def test_unary_support_summary_aggregates_stage_counts():
    summary = summarize([
        {
            "clean_correct": True,
            "n_context_tokens": 3,
            "n_sentences": 2,
            "elapsed_seconds": 1.0,
            "method": "unary",
        },
        {
            "clean_correct": False,
            "n_context_tokens": 5,
            "n_sentences": 4,
            "elapsed_seconds": 3.0,
            "method": "unary",
        },
    ])

    assert summary["queries"] == 2
    assert summary["exact_clean_queries"] == 1
    assert summary["context_tokens"] == 8
    assert summary["sentences"] == 6
    assert summary["mean_seconds"] == 2.0
