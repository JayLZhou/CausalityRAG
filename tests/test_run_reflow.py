import pytest

from exp.run_reflow import EMPTY_EXECUTABLE_FRONTIER, frontier_status


def test_empty_executable_frontier_is_strict_by_default():
    with pytest.raises(RuntimeError, match="nonempty executable domain"):
        frontier_status(
            "q1",
            [{"unit_id": "u1"}],
            {"status": "ok", "candidates": []},
            record_empty_frontier=False,
        )


def test_empty_executable_frontier_can_be_recorded_for_targeted_repair():
    assert frontier_status(
        "q1",
        [{"unit_id": "u1"}],
        {"status": "ok", "candidates": []},
        record_empty_frontier=True,
    ) == EMPTY_EXECUTABLE_FRONTIER


def test_nonempty_frontier_preserves_solver_status():
    assert frontier_status(
        "q1",
        [{"unit_id": "u1"}],
        {"status": "ok", "candidates": [{"selected_ids": ["u1"]}]},
        record_empty_frontier=False,
    ) == "ok"
