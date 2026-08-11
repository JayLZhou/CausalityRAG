from concurrent.futures import Future

from causalityrag.reader import ReaderProtocolError
from scripts.generate_reader_targets import reader_outcome


def completed_future(*, result=None, error=None):
    future = Future()
    if error is None:
        future.set_result(result)
    else:
        future.set_exception(error)
    return future


def test_reader_outcome_keeps_a_valid_choice():
    assert reader_outcome(completed_future(result="B")) == {
        "answer": "B",
        "status": "ok",
        "raw_reader_answer": "",
        "protocol_error": "",
    }


def test_reader_outcome_records_protocol_violation_without_guessing():
    raw = "The answer is C because the patient has..."
    error = ReaderProtocolError(f"invalid MedQA choice: {raw!r}")
    outcome = reader_outcome(completed_future(error=error))

    assert outcome["answer"] == ""
    assert outcome["status"] == "protocol_violation"
    assert raw in outcome["raw_reader_answer"]
    assert outcome["protocol_error"] == str(error)


def test_reader_outcome_records_empty_abstention():
    assert reader_outcome(completed_future(result=""))["status"] == (
        "reader_abstention_empty_answer"
    )
