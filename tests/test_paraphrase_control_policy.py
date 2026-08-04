from causalityrag.paraphrase_control import excluded_from_paraphrase_control
from scripts.build_paraphrase_control_pool import equivalent_rendering
from scripts.evaluate_paraphrase_controls import evaluate_selected


class _Reader:
    def __init__(self) -> None:
        self.calls = 0

    def answer(self, question: str, contexts: list[dict]) -> str:
        self.calls += 1
        return contexts[0]["text"]


def test_numeric_and_date_units_are_excluded_from_paraphrase_control() -> None:
    assert excluded_from_paraphrase_control({"text": "0.060", "type": "NUMBER"})
    assert excluded_from_paraphrase_control({"text": "1998", "type": "DATE"})
    assert excluded_from_paraphrase_control({"text": "12%", "type": "CONTENT"})
    assert not excluded_from_paraphrase_control({"text": "physician", "type": "CONTENT"})


def test_fallback_never_rewrites_numeric_surface() -> None:
    assert equivalent_rendering("0.060") is None
    assert equivalent_rendering("1998") is None
    assert equivalent_rendering("physician") == "PHYSICIAN"


def test_numeric_selection_stays_unchanged_in_control() -> None:
    reader = _Reader()
    result = evaluate_selected(
        record={
            "question": "q",
            "retrieved": [{"chunk_id": "c1", "rank": 1, "text": "1998"}],
        },
        units={
            "n": {
                "unit_id": "n",
                "text": "1998",
                "type": "DATE",
                "chunk_id": "c1",
                "chunk_char_start": 0,
                "chunk_char_end": 4,
            }
        },
        selected=["n"],
        clean_answer="yes",
        gold_answer="yes",
        pool={},
        reader=reader,
        seed=0,
        k=1,
    )

    assert result["status"] == "no_paraphrase_eligible_tokens"
    assert result["excluded_numeric_date_ids"] == ["n"]
    assert result["n_modified_tokens"] == 0
    assert reader.calls == 0


def test_mixed_selection_only_paraphrases_non_numeric_token() -> None:
    reader = _Reader()
    result = evaluate_selected(
        record={
            "question": "q",
            "retrieved": [
                {"chunk_id": "c1", "rank": 1, "text": "1998 physician"}
            ],
        },
        units={
            "n": {
                "unit_id": "n",
                "text": "1998",
                "type": "DATE",
                "chunk_id": "c1",
                "chunk_char_start": 0,
                "chunk_char_end": 4,
            },
            "w": {
                "unit_id": "w",
                "text": "physician",
                "type": "CONTENT",
                "chunk_id": "c1",
                "chunk_char_start": 5,
                "chunk_char_end": 14,
            },
        },
        selected=["n", "w"],
        clean_answer="yes",
        gold_answer="yes",
        pool={"w": {"candidates": [{"new": "doctor", "policy": "test"}]}},
        reader=reader,
        seed=0,
        k=1,
    )

    assert result["status"] == "evaluated"
    assert result["excluded_numeric_date_ids"] == ["n"]
    assert result["control_selected_ids"] == ["w"]
    assert result["n_modified_tokens"] == 1
    assert result["edited_answer"] == "1998 doctor"
    assert reader.calls == 1
