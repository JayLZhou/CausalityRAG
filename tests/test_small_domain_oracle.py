from scripts.run_small_domain_oracle import clean_is_eligible, intervention_succeeds


def test_f1_cfr_conditions_on_clean_correctness() -> None:
    assert clean_is_eligible("f1-cfr", "Paris", "Paris")
    assert clean_is_eligible("f1-cfr", "The Paris", "Paris")
    assert not clean_is_eligible("f1-cfr", "London", "Paris")
    assert not clean_is_eligible("f1-cfr", "Paris", "")


def test_f1_cfr_requires_loss_of_full_f1() -> None:
    assert not intervention_succeeds("f1-cfr", "Paris", "The Paris", "Paris")
    assert intervention_succeeds("f1-cfr", "Paris", "London", "Paris")


def test_answer_change_remains_available_for_reproduction() -> None:
    assert not intervention_succeeds("answer-change", "The Paris", "Paris", "unused")
    assert intervention_succeeds("answer-change", "Paris", "London", "unused")
