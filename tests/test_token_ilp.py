from causalityrag.ilp import Unit, solve_budgeted_support, solve_min_cost_cover
from causalityrag.token_units import build_token_units, run_token_ilp_record


def test_min_cost_cover_with_dp_fallback():
    units = [Unit("a", support=4, cost=1), Unit("b", support=6, cost=2), Unit("c", support=9, cost=3)]

    result = solve_min_cost_cover(units, target_support=10, solver="dp")

    assert result.status == "optimal"
    assert result.objective == 3
    assert result.support_removed >= 10


def test_budgeted_support_with_dp_fallback():
    units = [Unit("a", support=5, cost=3), Unit("b", support=4, cost=2), Unit("c", support=3, cost=1)]

    result = solve_budgeted_support(units, budget=3, solver="dp")

    assert result.status == "optimal"
    assert set(result.selected_ids) == {"b", "c"}
    assert result.support_removed == 7


def test_token_units_keep_offsets_without_answer_matching():
    record = {
        "id": "ex1",
        "question": "Where was Ada born?",
        "answer": "London",
        "retrieved": [{"chunk_id": "c1", "text": "Ada Lovelace was born in London.", "rank": 1}],
    }

    units = build_token_units(record, k=1)
    london = next(unit for unit in units if unit.text == "London")
    different_answer = {**record, "answer": "Paris"}
    london_with_different_answer = next(
        unit for unit in build_token_units(different_answer, k=1) if unit.text == "London"
    )

    assert london.chunk_id == "c1"
    assert london.chunk_char_start == record["retrieved"][0]["text"].index("London")
    assert london.unit_type == "PROPER"
    assert "answer_match" not in london.sources
    assert london.support == london_with_different_answer.support


def test_run_token_ilp_record_outputs_selected_units():
    record = {
        "id": "ex2",
        "question": "Who created Sherlock Holmes?",
        "answer": "Arthur Conan Doyle",
        "retrieved": [
            {"chunk_id": "c1", "text": "Sherlock Holmes was created by Arthur Conan Doyle.", "rank": 1},
            {"chunk_id": "c2", "text": "The detective is known for observation.", "rank": 2},
        ],
    }

    row = run_token_ilp_record(record, k=2, solver="dp", tau_graph=0.8)

    assert row["status"] == "optimal"
    assert row["selected_cut"]
    assert row["selected_units"]
    assert row["context_ids"] == ["c1", "c2"]
