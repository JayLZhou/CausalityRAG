from scripts.build_replacement_registry import (
    existing_flow_candidate_ids,
    registry_candidate_ids,
)


def test_registry_includes_only_flow_candidates() -> None:
    gate = {
        "strict_candidate": {
            "n_selected": 2,
            "selected_ids": ["flow-strict"],
            "unary_matched_ids": ["unary-strict"],
        },
        "bicriteria_candidate": {
            "n_selected": 7,
            "selected_ids": ["flow-native"],
            "unary_matched_ids": ["unary-native-6", "unary-native-7"],
        },
    }

    candidate_ids = registry_candidate_ids(gate)

    assert candidate_ids == {"flow-strict", "flow-native"}


def test_registry_accumulates_only_explicit_pure_flow_history() -> None:
    assert existing_flow_candidate_ids(
        {
            "candidate_source": "contribution_flow",
            "candidate_ids": ["old-flow"],
        }
    ) == {"old-flow"}
    assert existing_flow_candidate_ids(
        {"candidate_ids": ["historical-untyped-candidate"]}
    ) == set()
