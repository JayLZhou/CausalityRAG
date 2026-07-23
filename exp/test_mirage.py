from exp.mirage import context_sensitive_indices


def test_context_sensitive_indices_use_population_standard_deviation():
    selected, threshold = context_sensitive_indices([0.0, 0.0, 0.0, 4.0])
    assert selected == [3]
    assert abs(threshold - (1.0 + 3.0 ** 0.5)) < 1e-9


def test_context_sensitive_indices_keep_single_response_token():
    selected, threshold = context_sensitive_indices([2.5])
    assert selected == [0]
    assert threshold == 2.5
