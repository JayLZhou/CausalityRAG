from scripts.rescore_fixed_budget_results import rescore_rows


def test_rescore_rows_uses_popqa_aliases() -> None:
    records = [{
        "id": "q1",
        "question": "What is the genre?",
        "answer": "J-pop",
        "answer_aliases": ["Japanese pop", "jpop"],
    }]
    merged = {
        "q1": {
            "id": "q1",
            "clean_answer": "Japanese pop",
            "reader_calls": 1,
            "methods": {
                "mirage": {
                    "1": {
                        "status": "acc_flip",
                        "edited_answer": "jpop",
                        "acc_flip": True,
                        "n_modified_tokens": 1,
                    }
                }
            },
        }
    }

    rows, budgets = rescore_rows(records, merged, reader_mode="short_answer")

    assert budgets == [1]
    assert rows[0]["clean_acc"]
    assert not rows[0]["methods"]["mirage"]["1"]["acc_flip"]
