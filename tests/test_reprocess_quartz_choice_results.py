from scripts.reprocess_quartz_choice_results import reprocess, reprocess_baselines


def test_reprocess_quartz_uses_choice_identity_and_marks_required_resume():
    retrieval = [
        {
            "id": "q1",
            "question": "Question? Choices: (A) harder; (B) easier.",
            "answer": "harder",
        },
        {
            "id": "q2",
            "question": "Question? Choices: (A) more; (B) less.",
            "answer": "more",
        },
    ]
    factual = [
        {
            "id": "q1",
            "clean_answer": "(A) harder",
            "gold_answer": "harder",
            "edited_answer": "A",
            "verified_flip": True,
            "evaluation_status": "verified_flip",
            "attempts": [
                {
                    "reader_called": True,
                    "selected_ids": ["u1"],
                    "n_edits": 1,
                    "edited_answer": "A",
                    "answer_changed": True,
                    "candidate_status": "verified_flip",
                }
            ],
        },
        {
            "id": "q2",
            "clean_answer": "A",
            "gold_answer": "more",
            "edited_answer": "B",
            "verified_flip": True,
            "evaluation_status": "verified_flip",
            "attempts": [
                {
                    "reader_called": True,
                    "selected_ids": ["u2"],
                    "n_edits": 1,
                    "edited_answer": "B",
                    "answer_changed": True,
                    "candidate_status": "verified_flip",
                }
            ],
        },
    ]
    control = [
        {
            "id": "q1",
            "clean_answer": "A",
            "gold_answer": "harder",
            "methods": {
                "reflow": {
                    "status": "evaluated",
                    "reader_called": True,
                    "edited_answer": "harder",
                    "answer_flip": True,
                }
            },
        },
        {
            "id": "q2",
            "clean_answer": "more",
            "gold_answer": "more",
            "methods": {
                "reflow": {
                    "status": "evaluated",
                    "reader_called": True,
                    "edited_answer": "B",
                    "answer_flip": True,
                }
            },
        },
    ]

    factual_out, control_out, report = reprocess(retrieval, factual, control)

    assert factual_out[0]["verified_flip"] is False
    assert factual_out[0]["requires_frontier_resume"] is True
    assert factual_out[0]["clean_answer"] == "harder"
    assert factual_out[0]["edited_answer"] == "harder"
    assert factual_out[1]["verified_flip"] is True
    assert factual_out[1]["edited_answer"] == "less"
    assert control_out[0]["methods"]["reflow"]["answer_flip"] is False
    assert control_out[0]["methods"]["reflow"]["quartz_postprocess_status"] == (
        "same_choice_false_positive"
    )
    assert control_out[1]["methods"]["reflow"]["answer_flip"] is True
    assert report["requires_frontier_resume"] == 1
    assert report["factual_statuses"] == {
        "same_choice_false_positive_requires_resume": 1,
        "verified_true_choice_flip": 1,
    }


def test_reprocess_quartz_never_counts_unresolved_output_as_a_flip():
    retrieval = [
        {
            "id": "q1",
            "question": "Question? Choices: (A) harder; (B) easier.",
            "answer": "harder",
        }
    ]
    factual = [
        {
            "id": "q1",
            "clean_answer": "harder",
            "gold_answer": "harder",
            "verified_flip": True,
            "attempts": [
                {
                    "reader_called": True,
                    "edited_answer": "probably harder",
                    "answer_changed": True,
                }
            ],
        }
    ]
    control = [
        {
            "id": "q1",
            "clean_answer": "harder",
            "gold_answer": "harder",
            "methods": {
                "reflow": {
                    "reader_called": True,
                    "edited_answer": "probably harder",
                    "answer_flip": True,
                }
            },
        }
    ]

    factual_out, control_out, report = reprocess(retrieval, factual, control)

    assert factual_out[0]["verified_flip"] is False
    assert factual_out[0]["evaluation_status"] == (
        "protocol_violation_invalid_reader_answer"
    )
    method = control_out[0]["methods"]["reflow"]
    assert method["answer_flip"] is False
    assert method["status"] == "protocol_violation_invalid_reader_answer"
    assert report["factual_statuses"] == {"unresolved_reader_answer": 1}


def test_reprocess_quartz_baselines_canonicalizes_each_method():
    retrieval = [
        {
            "id": "q1",
            "question": "Question? Choices: (A) harder; (B) easier.",
            "answer": "harder",
        }
    ]
    baselines = [
        {
            "id": "q1",
            "clean_answer": "A",
            "gold_answer": "harder",
            "methods": {
                "same": {
                    "reader_called": True,
                    "edited_answer": "harder",
                    "verified_flip": True,
                },
                "changed": {
                    "reader_called": True,
                    "edited_answer": "B",
                    "verified_flip": True,
                },
            },
        }
    ]

    output, statuses = reprocess_baselines(retrieval, baselines)

    assert output[0]["methods"]["same"]["verified_flip"] is False
    assert output[0]["methods"]["changed"]["verified_flip"] is True
    assert statuses == {
        "same_choice_false_positive": 1,
        "verified_true_choice_flip": 1,
    }
