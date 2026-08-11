import json
from types import SimpleNamespace

import pytest

from causalityrag.reader import (
    ReaderClient,
    ReaderProtocolError,
    answer_token_f1,
    answers_exact_match,
    answers_match,
    canonicalize_quartz_answer,
    model_context_window,
    parse_quartz_choices,
    quartz_choice_label,
    reader_completion_text,
    reader_prompt,
)
from causalityrag.io import gold_answers
from scripts.run_table3_reader_stage import reader_mode_for_dataset


def test_exact_match_does_not_use_substring_containment():
    assert answers_match("Paris", "Paris, Texas")
    assert not answers_exact_match("Paris", "Paris, Texas")
    assert answers_exact_match("The United States", "United States")


def test_exact_match_is_reflexive_for_punctuation_answers():
    assert answers_exact_match("...", "...")
    assert answers_exact_match("[]", "[]")


def test_answer_token_f1_uses_normalized_token_overlap():
    assert answer_token_f1("The United States", "United States") == 1.0
    assert answer_token_f1("Paris", "Paris Texas") == 2.0 / 3.0
    assert answer_token_f1("Paris", "London") == 0.0


def test_reader_client_does_not_send_an_output_token_cap(monkeypatch):
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"choices":[{"message":{"content":"{\\"answer\\":\\"Paris\\"}"}}]}'

    def fake_urlopen(request, timeout):
        captured.update(json.loads(request.data.decode("utf-8")))
        assert timeout == 120
        return FakeResponse()

    monkeypatch.setattr(
        "causalityrag.reader.urllib.request.urlopen",
        fake_urlopen,
    )

    answer = ReaderClient(base_url="http://reader.test/v1").answer(
        "What is the capital of France?",
        [{"chunk_id": "c0", "text": "Paris is the capital of France."}],
    )

    assert answer == "Paris"
    assert "max_tokens" not in captured
    assert "max_completion_tokens" not in captured


def test_reader_client_sends_an_explicit_output_token_cap(monkeypatch):
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"choices":[{"message":{"content":"{\\"answer\\":\\"Paris\\"}"}}]}'

    def fake_urlopen(request, timeout):
        captured.update(json.loads(request.data.decode("utf-8")))
        return FakeResponse()

    monkeypatch.setattr(
        "causalityrag.reader.urllib.request.urlopen",
        fake_urlopen,
    )

    answer = ReaderClient(
        base_url="http://reader.test/v1",
        max_tokens=128,
    ).answer(
        "What is the capital of France?",
        [{"chunk_id": "c0", "text": "Paris is the capital of France."}],
    )

    assert answer == "Paris"
    assert captured["max_tokens"] == 128


def test_medqa_reader_uses_prompt_only_and_returns_a_letter(monkeypatch):
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"choices":[{"message":{"content":"B"}}]}'

    def fake_urlopen(request, timeout):
        captured.update(json.loads(request.data.decode("utf-8")))
        return FakeResponse()

    monkeypatch.setattr(
        "causalityrag.reader.urllib.request.urlopen",
        fake_urlopen,
    )

    answer = ReaderClient(
        base_url="http://reader.test/v1",
        reader_mode="medqa",
    ).answer(
        "Question?\nOptions:\n(A) One\n(B) Two\n(C) Three\n(D) Four",
        [{"chunk_id": "c0", "text": "Evidence."}],
    )

    assert answer == "B"
    assert "guided_choice" not in captured
    assert "guided_decoding_backend" not in captured
    assert "exactly one letter" in captured["messages"][1]["content"]


def test_medqa_reader_rejects_nonchoice_output(monkeypatch):
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"choices":[{"message":{"content":"The answer is B."}}]}'

    monkeypatch.setattr(
        "causalityrag.reader.urllib.request.urlopen",
        lambda request, timeout: FakeResponse(),
    )

    reader = ReaderClient(
        base_url="http://reader.test/v1",
        reader_mode="medqa",
    )
    with pytest.raises(ReaderProtocolError):
        reader.answer(
            "Question?\nOptions:\n(A) One\n(B) Two\n(C) Three\n(D) Four",
            [{"chunk_id": "c0", "text": "Evidence."}],
        )


def test_medqa_teacher_forced_completion_matches_live_letter_protocol():
    assert reader_completion_text("b", reader_mode="medqa") == "B"
    with pytest.raises(ReaderProtocolError):
        reader_completion_text("The answer is B", reader_mode="medqa")


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("A", "A"),
        ("(A)", "A"),
        ("A) harder", "A"),
        ('{"answer": "B"}', "B"),
        ("The answer is B.", "B"),
        ("easier", "B"),
        ('"harder"', "A"),
    ],
)
def test_quartz_choice_postprocessing_maps_unambiguous_answers(answer, expected):
    question = (
        "A 100 pound box is compared with a 10 pound box. "
        "Choices: (A) harder; (B) easier."
    )
    assert quartz_choice_label(question, answer) == expected


def test_quartz_choice_parser_handles_parenthetical_prose_and_punctuation():
    question = (
        "A(n) change was observed. Choices: "
        "(A) Helium, with its two protons; "
        "(B) Plutonium, with its 94 protons?."
    )
    assert parse_quartz_choices(question) == {
        "A": "Helium, with its two protons",
        "B": "Plutonium, with its 94 protons",
    }
    assert canonicalize_quartz_answer(question, "B") == (
        "Plutonium, with its 94 protons"
    )
    assert quartz_choice_label(question, "Plutonium") == "B"


@pytest.mark.parametrize(
    "answer",
    [
        "A) easier",
        "A 100 pound box would be harder to shove.",
        "probably harder",
        "C",
        "",
    ],
)
def test_quartz_choice_postprocessing_rejects_ambiguous_or_invalid_answers(answer):
    question = "Question? Choices: (A) harder; (B) easier."
    with pytest.raises(ReaderProtocolError):
        quartz_choice_label(question, answer)


def test_quartz_reader_canonicalizes_without_guided_choice(monkeypatch):
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"choices":[{"message":{"content":"{\\"answer\\":\\"A\\"}"}}]}'

    def fake_urlopen(request, timeout):
        captured.update(json.loads(request.data.decode("utf-8")))
        return FakeResponse()

    monkeypatch.setattr(
        "causalityrag.reader.urllib.request.urlopen",
        fake_urlopen,
    )
    answer = ReaderClient(
        base_url="http://reader.test/v1",
        reader_mode="quartz",
    ).answer(
        "Question? Choices: (A) harder; (B) easier.",
        [{"chunk_id": "c0", "text": "Evidence."}],
    )
    assert answer == "harder"
    assert "guided_choice" not in captured


def test_default_teacher_forced_completion_remains_json():
    assert reader_completion_text(
        "Paris", reader_mode="short_answer"
    ) == '{"answer": "Paris"}'


def test_pubmedqa_uses_minimal_label_prompt_and_plain_completion():
    system, user = reader_prompt(
        "Does the intervention help?",
        [{"chunk_id": "c0", "text": "The trial reports no benefit."}],
        reader_mode="pubmedqa",
    )
    assert system == "Answer the question using only the provided passages."
    assert "exactly one label: yes, no, or maybe" in user
    assert "JSON" not in system + user
    assert reader_completion_text("YES", reader_mode="pubmedqa") == "yes"
    with pytest.raises(ReaderProtocolError):
        reader_completion_text("yes because", reader_mode="pubmedqa")


def test_gold_answers_preserves_canonical_answer_and_distinct_aliases():
    assert gold_answers({
        "answer": "politician",
        "answers": ["politician", "political leader"],
        "answer_aliases": ["Political Leader", "political figure"],
    }) == ["politician", "political leader", "political figure"]


@pytest.mark.parametrize("dataset", ["pubmedqa", "medqa", "quartz"])
def test_table3_reader_stage_preserves_categorical_reader_mode(dataset):
    assert reader_mode_for_dataset(dataset) == dataset


def test_model_context_window_uses_the_declared_architecture_limit():
    assert model_context_window(
        SimpleNamespace(max_position_embeddings=32768)
    ) == 32768
