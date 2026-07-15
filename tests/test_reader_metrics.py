from causalityrag.reader import answer_token_f1, answers_exact_match, answers_match


def test_exact_match_does_not_use_substring_containment():
    assert answers_match("Paris", "Paris, Texas")
    assert not answers_exact_match("Paris", "Paris, Texas")
    assert answers_exact_match("The United States", "United States")


def test_answer_token_f1_uses_normalized_token_overlap():
    assert answer_token_f1("The United States", "United States") == 1.0
    assert answer_token_f1("Paris", "Paris Texas") == 2.0 / 3.0
    assert answer_token_f1("Paris", "London") == 0.0
