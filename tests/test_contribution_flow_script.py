from scripts.solve_contribution_flow import summarize


def test_flow_summary_reports_registry_fixed_point():
    summary = summarize([
        {
            "status": "ok",
            "diagnostics": {
                "distinct_nonempty_candidate_sets": 0,
                "solver": "geometric_k_guessing_mincut",
                "strict_candidate_found": False,
                "bicriteria_candidate_found": False,
                "mincut_calls": 1,
            },
            "candidates": [],
            "elapsed_seconds": 0.1,
            "replacement_registry": "registry.jsonl",
            "registry_candidate_misses": [],
        }
    ])

    assert summary["registry_fixed_point"]
    assert summary["evaluated_candidate_registry_misses"] == 0
