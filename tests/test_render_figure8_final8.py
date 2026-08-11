import pytest

from scripts.render_figure8_final8 import select_method


def point(rate: float, *, clean: int = 10) -> dict:
    return {
        "clean_acc_queries": clean,
        "valid_answer_queries": clean,
        "protocol_violations": 0,
        "acc_cfr": rate,
        "mean_modified_tokens": 1.0,
    }


def test_select_method_requires_dense_budgets_before_first_crossing() -> None:
    with pytest.raises(ValueError, match="missing lower budgets"):
        select_method(
            {"1": point(0.1), "3": point(0.8)},
            target=0.7,
            clean_queries=10,
        )

    result = select_method(
        {"1": point(0.1), "2": point(0.6), "3": point(0.8)},
        target=0.7,
        clean_queries=10,
    )
    assert result["status"] == "matched"
    assert result["minimum_budget"] == 3


def test_select_method_requires_budget_128_before_nr() -> None:
    with pytest.raises(ValueError, match="only tested through budget 64"):
        select_method(
            {"1": point(0.1), "64": point(0.2)},
            target=0.7,
            clean_queries=10,
        )

    result = select_method(
        {"1": point(0.1), "128": point(0.2)},
        target=0.7,
        clean_queries=10,
    )
    assert result["status"] == "not_reached"
    assert result["max_tested_budget"] == 128
