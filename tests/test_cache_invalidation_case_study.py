from scripts.summarize_cache_invalidation_case_study import covered_by_signature


def test_cache_signature_matches_exact_token_positions() -> None:
    ranked_tokens = ["sentence-1:token-1", "sentence-2:token-1"]

    assert covered_by_signature({"sentence-1:token-1"}, ranked_tokens, 1)
    assert not covered_by_signature({"sentence-1:token-2"}, ranked_tokens, 1)
    assert covered_by_signature({"sentence-2:token-1"}, ranked_tokens, 2)
