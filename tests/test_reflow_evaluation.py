from scripts.evaluate_reflow import candidate_rows, valid_clean_answer


def test_frontier_candidates_are_verified_smallest_first():
    row = {
        "frontier_candidates": [
            {
                "n_selected": 3,
                "selected_ids": ["a", "b", "c"],
                "remaining_support_flow": 0.0,
            },
            {
                "n_selected": 1,
                "selected_ids": ["a"],
                "remaining_support_flow": 2.0,
            },
            {
                "n_selected": 2,
                "selected_ids": ["a", "b"],
                "remaining_support_flow": 1.0,
            },
        ]
    }

    assert [
        candidate["n_selected"] for candidate in candidate_rows(row)
    ] == [1, 2, 3]


def test_invalid_clean_answers_are_not_counted_as_reader_failures():
    assert valid_clean_answer("Paris")
    assert not valid_clean_answer("")
    assert not valid_clean_answer("[]")
    assert not valid_clean_answer("no answer found")
