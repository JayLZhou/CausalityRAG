from scripts.run_prefix_exhaustive_comparison import (
    minimum_successful_subset,
    summarize,
)


def test_minimum_successful_subset_searches_by_cardinality() -> None:
    successful = {("a", "c"), ("a", "b", "c")}
    selected, found, calls = minimum_successful_subset(
        ("a", "b", "c"), lambda subset: subset in successful
    )
    assert found
    assert selected == ("a", "c")
    assert calls == 6


def test_minimum_successful_subset_returns_full_prefix_when_infeasible() -> None:
    selected, found, calls = minimum_successful_subset(
        ("a", "b", "c"), lambda subset: False
    )
    assert not found
    assert selected == ("a", "b", "c")
    assert calls == 7


def test_summary_subtracts_synonym_flip_rate() -> None:
    rows = [
        {
            "prefixes": {
                "1": {
                    "exhaustive": {
                        "n_modified_tokens": 1,
                        "factual_f1_flip": True,
                        "synonym_f1_flip": False,
                        "selection_reader_calls": 1,
                    },
                    "reflow": {
                        "n_modified_tokens": 1,
                        "factual_f1_flip": True,
                        "synonym_f1_flip": True,
                        "selection_reader_calls": 1,
                    },
                }
            }
        },
        {
            "prefixes": {
                "1": {
                    "exhaustive": {
                        "n_modified_tokens": 1,
                        "factual_f1_flip": False,
                        "synonym_f1_flip": False,
                        "selection_reader_calls": 1,
                    },
                    "reflow": {
                        "n_modified_tokens": 1,
                        "factual_f1_flip": True,
                        "synonym_f1_flip": False,
                        "selection_reader_calls": 1,
                    },
                }
            }
        },
    ]
    summary = summarize(rows, 1)
    assert summary["methods"]["exhaustive"][0]["f1_cfr"] == 0.5
    assert summary["methods"]["reflow"][0]["f1_cfr"] == 0.5
