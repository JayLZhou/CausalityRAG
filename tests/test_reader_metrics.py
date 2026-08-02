import json
from types import SimpleNamespace

from causalityrag.reader import (
    ReaderClient,
    answer_token_f1,
    answers_exact_match,
    answers_match,
    model_context_window,
)


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


def test_model_context_window_uses_the_declared_architecture_limit():
    assert model_context_window(
        SimpleNamespace(max_position_embeddings=32768)
    ) == 32768
