from scripts.summarize_cache_invalidation_case_study import (
    covered_by_signature,
    project_units_to_sentences,
)


def test_cache_signature_matches_exact_token_positions() -> None:
    ranked_tokens = ["sentence-1:token-1", "sentence-2:token-1"]

    assert covered_by_signature({"sentence-1:token-1"}, ranked_tokens, 1)
    assert not covered_by_signature({"sentence-1:token-2"}, ranked_tokens, 1)
    assert covered_by_signature({"sentence-2:token-1"}, ranked_tokens, 2)


def test_sentence_projection_preserves_token_rank_and_deduplicates() -> None:
    ranked_tokens = ["u1", "u2", "u3"]
    sentence_by_token = {"u1": "s1", "u2": "s1", "u3": "s2"}

    ranked_sentences = project_units_to_sentences(ranked_tokens, sentence_by_token)

    assert ranked_sentences == ["s1", "s2"]
    assert covered_by_signature({"s1"}, ranked_sentences, 1)
