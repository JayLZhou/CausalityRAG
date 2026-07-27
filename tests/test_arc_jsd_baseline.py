from exp.arc_jsd import solve_arc_jsd_ratio


def test_arc_jsd_ratio_keeps_positive_pair_synergy():
    result = solve_arc_jsd_ratio(
        [{"unit_id": "a"}, {"unit_id": "b"}, {"unit_id": "c"}],
        {"a": 1.0, "b": 1.0, "c": 0.0},
        {("a", "b"): 4.0, ("b", "c"): 0.1},
    )

    assert result["status"] == "optimal"
    assert result["selected_ids"] == ["a", "b"]
    assert result["density"] == 3.0
