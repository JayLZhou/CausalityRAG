from scripts.build_replacement_registry import registry_candidate_ids


def test_registry_includes_native_size_matched_unary_tokens() -> None:
    gate = {
        "unary_order": ["u0", "u1", "u2", "u3", "u4"],
        "bicriteria_candidate": {
            "n_selected": 7,
            "selected_ids": ["flow-native"],
            "unary_matched_ids": ["unary-native-6", "unary-native-7"],
        },
        "budget_candidates": [{
            "budget": 3,
            "selected_ids": ["flow-budget"],
            "unary_matched_ids": ["unary-budget"],
        }],
    }

    candidate_ids = registry_candidate_ids(
        gate,
        max_budget=5,
        max_native_tokens=10,
    )

    assert "flow-native" in candidate_ids
    assert "unary-native-6" in candidate_ids
    assert "unary-native-7" in candidate_ids
    assert "flow-budget" in candidate_ids
    assert "unary-budget" in candidate_ids
