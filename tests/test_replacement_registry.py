from scripts.build_replacement_registry import registry_candidate_ids


def test_registry_includes_native_size_matched_unary_tokens() -> None:
    gate = {
        "bicriteria_candidate": {
            "n_selected": 7,
            "selected_ids": ["flow-native"],
            "unary_matched_ids": ["unary-native-6", "unary-native-7"],
        },
    }

    candidate_ids = registry_candidate_ids(gate)

    assert "flow-native" in candidate_ids
    assert "unary-native-6" in candidate_ids
    assert "unary-native-7" in candidate_ids
