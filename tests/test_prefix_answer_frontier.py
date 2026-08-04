from scripts.evaluate_prefix_answer_frontier import (
    exhaustive_verifications,
    reflow_verifications,
)


def test_exhaustive_verifications_include_full_success_cardinality() -> None:
    assert exhaustive_verifications(n=5, selected_size=2, found=True) == 5 + 10


def test_exhaustive_verifications_charge_full_domain_when_unsuccessful() -> None:
    assert exhaustive_verifications(n=5, selected_size=5, found=False) == 31


def test_reflow_verifications_stop_at_selected_breakpoint() -> None:
    candidates = [("a",), ("a", "b"), ("a", "b", "c")]
    assert reflow_verifications(candidates, ("b", "a")) == 2


def test_reflow_verifications_charge_fallback_when_frontier_is_empty() -> None:
    assert reflow_verifications([], ("a", "b")) == 1
