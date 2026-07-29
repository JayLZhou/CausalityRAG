from exp.mirage import context_sensitive_indices


def test_context_sensitive_indices_uses_mean_plus_population_std():
    selected, threshold = context_sensitive_indices([0.0, 0.0, 0.0, 4.0])

    assert selected == [3]
    assert threshold > 2.0


def test_context_sensitive_indices_handles_empty_input():
    assert context_sensitive_indices([]) == ([], 0.0)
