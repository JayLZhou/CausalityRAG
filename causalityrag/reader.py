"""vLLM reader calls and experimental local-HF reader utilities."""

from __future__ import annotations

import json
import os
import re
import string
import unicodedata
import urllib.request
from collections import Counter
from typing import Sequence


READ_SYSTEM = "Answer the question using ONLY the passages. Give the shortest answer span. Output JSON only."
READ_USER = """Passages:
{passages}

Question: {question}
Return STRICT JSON: {{"answer": "..."}}"""
PUBMEDQA_CHOICES = ("yes", "no", "maybe")
PUBMEDQA_SYSTEM = "Answer the question using only the provided passages."
PUBMEDQA_USER = """Passages:
{passages}

Question: {question}
Answer with exactly one label: yes, no, or maybe. Do not explain."""
QUARTZ_SYSTEM = (
    "You are answering a QuaRTz qualitative reasoning multiple-choice question. "
    "Use only the provided passage and question. Return exactly one of the two "
    "listed answer choices, with no explanation. Output JSON only."
)
QUARTZ_USER = """Passage:
{passages}

Question and choices:
{question}
Return STRICT JSON containing exactly the selected choice text:
{{"answer": "<choice>"}}"""
MEDQA_CHOICES = ("A", "B", "C", "D")
QUARTZ_CHOICES = ("A", "B")
MEDQA_SYSTEM = (
    "You are answering a MedQA-USMLE multiple-choice question. "
    "Use only the provided passages. Return exactly one answer-choice letter: "
    "A, B, C, or D. Do not provide an explanation or punctuation."
)
MEDQA_USER = """Passages:
{passages}

Question and choices:
{question}
Return exactly one letter: A, B, C, or D."""


class ReaderProtocolError(ValueError):
    """Raised when a constrained reader returns an invalid answer."""


def reader_prompt(
    question: str,
    contexts: list[dict],
    *,
    reader_mode: str,
) -> tuple[str, str]:
    """Return the exact system/user prompt for one reader protocol."""

    mode = reader_mode.strip().lower()
    passages = format_passages(contexts)
    if mode == "pubmedqa":
        return (
            PUBMEDQA_SYSTEM,
            PUBMEDQA_USER.format(question=question, passages=passages),
        )
    if mode == "quartz":
        return (
            QUARTZ_SYSTEM,
            QUARTZ_USER.format(question=question, passages=passages),
        )
    if mode == "medqa":
        return (
            MEDQA_SYSTEM,
            MEDQA_USER.format(question=question, passages=passages),
        )
    return (
        READ_SYSTEM,
        READ_USER.format(question=question, passages=passages),
    )


def reader_completion_text(
    answer: str,
    *,
    reader_mode: str,
    compact_json: bool = False,
) -> str:
    """Render the teacher-forced completion used by the live reader."""

    mode = reader_mode.strip().lower()
    if mode == "pubmedqa":
        label = str(answer).strip().casefold()
        if label not in PUBMEDQA_CHOICES:
            raise ReaderProtocolError(
                f"invalid PubMedQA teacher-forced label: {answer!r}"
            )
        return label
    if mode == "medqa":
        choice = str(answer).strip().upper()
        if choice not in MEDQA_CHOICES:
            raise ReaderProtocolError(
                f"invalid MedQA teacher-forced choice: {answer!r}"
            )
        return choice
    kwargs = {"ensure_ascii": False}
    if compact_json:
        kwargs["separators"] = (",", ":")
    return json.dumps({"answer": str(answer)}, **kwargs)


class ReaderClient:
    """Concurrent-safe client for the vLLM OpenAI-compatible endpoint."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: int = 120,
        max_tokens: int | None = None,
        reader_mode: str | None = None,
    ) -> None:
        self.base_url = (
            base_url
            or os.environ.get("CAUSALITYRAG_LLM_BASE_URL")
            or os.environ.get("YVETTE_LLM_BASE_URL")
            or "http://127.0.0.1:8000/v1"
        ).rstrip("/")
        self.model = (
            model
            or os.environ.get("CAUSALITYRAG_LLM_MODEL")
            or os.environ.get("YVETTE_LLM_MODEL")
            or "qwen2.5-7b"
        )
        self.timeout = timeout
        env_max_tokens = os.environ.get("CAUSALITYRAG_READER_MAX_TOKENS")
        self.max_tokens = (
            max_tokens
            if max_tokens is not None
            else int(env_max_tokens)
            if env_max_tokens
            else None
        )
        self.reader_mode = (
            reader_mode or os.environ.get("CAUSALITYRAG_READER_MODE", "short_answer")
        ).strip().lower()

    def answer(self, question: str, contexts: list[dict]) -> str:
        system, user = reader_prompt(
            question,
            contexts,
            reader_mode=self.reader_mode,
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
        }
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        request = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            data = json.loads(response.read().decode("utf-8"))
        content = data["choices"][0]["message"]["content"]
        parsed = parse_json_object(content)
        if isinstance(parsed, dict) and parsed.get("answer") is not None:
            answer = str(parsed["answer"]).strip()
        else:
            answer = str(content).strip()
        return canonicalize_reader_answer(
            question,
            answer,
            reader_mode=self.reader_mode,
        )


class LocalHFReader:
    """Greedy local Hugging Face reader retained for experimental cross-checks."""

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cuda",
        dtype: str = "bfloat16",
        attn_implementation: str = "sdpa",
    ) -> None:
        try:
            import torch
            from transformers import (
                AutoConfig,
                AutoModelForCausalLM,
                AutoTokenizer,
            )
        except ImportError as exc:  # pragma: no cover - GPU integration
            raise RuntimeError("LocalHFReader requires torch and transformers") from exc

        self.torch = torch
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        config = AutoConfig.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        if hasattr(config, "base_model_tp_plan"):
            config.base_model_tp_plan = None
        if hasattr(config, "base_model_pp_plan"):
            config.base_model_pp_plan = None
        self.max_context_tokens = model_context_window(config)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            torch_dtype=getattr(torch, dtype),
            trust_remote_code=True,
            attn_implementation=attn_implementation,
        ).to(device)
        self.model.eval()

    def generate_responses_batch(
        self,
        question: str,
        context_variants: Sequence[Sequence[dict]],
    ) -> list[str]:
        """Greedily generate until EOS or the model context window."""

        if not context_variants:
            return []
        torch = self.torch
        prompts = [
            self._prompt_ids(question, contexts) for contexts in context_variants
        ]
        max_prompt_length = max(len(prompt) for prompt in prompts)
        max_new_tokens = self.max_context_tokens - max_prompt_length
        if max_new_tokens <= 0:
            raise ValueError(
                f"reader prompt has {max_prompt_length} tokens, which reaches "
                f"the {self.max_context_tokens}-token model context window"
            )
        input_ids = torch.full(
            (len(prompts), max_prompt_length),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
            device=self.device,
        )
        attention_mask = torch.zeros_like(input_ids)
        for row, prompt in enumerate(prompts):
            start = max_prompt_length - len(prompt)
            input_ids[row, start:] = torch.tensor(
                prompt,
                dtype=torch.long,
                device=self.device,
            )
            attention_mask[row, start:] = 1
        with torch.inference_mode():
            generated = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        responses = []
        for sequence in generated[:, max_prompt_length:].tolist():
            if self.tokenizer.eos_token_id in sequence:
                sequence = sequence[: sequence.index(self.tokenizer.eos_token_id)]
            responses.append(
                self.tokenizer.decode(
                    sequence,
                    skip_special_tokens=True,
                ).strip()
            )
        return responses

    def _prompt_ids(
        self,
        question: str,
        contexts: Sequence[dict],
    ) -> list[int]:
        messages = [
            {"role": "system", "content": READ_SYSTEM},
            {
                "role": "user",
                "content": READ_USER.format(
                    question=question,
                    passages=format_passages(list(contexts)),
                ),
            },
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return self.tokenizer(
            prompt,
            add_special_tokens=False,
        )["input_ids"]


def model_context_window(config) -> int:
    """Return the finite context limit declared by a Hugging Face model config."""

    for field in ("max_position_embeddings", "n_positions", "seq_length"):
        value = getattr(config, field, None)
        if isinstance(value, int) and 0 < value < 1_000_000_000:
            return value
    raise ValueError("model config does not declare a finite context window")


def format_passages(contexts: list[dict]) -> str:
    return "\n\n".join(
        f"[{ctx.get('chunk_id', i)}] {ctx.get('text', '')}"
        for i, ctx in enumerate(contexts)
    )


def canonicalize_quartz_answer(question: str, answer: str) -> str:
    """Return the selected QuaRTz choice text or reject an invalid response."""

    choices = parse_quartz_choices(question)
    return choices[quartz_choice_label(question, answer)]


def canonicalize_reader_answer(
    question: str,
    answer: str,
    *,
    reader_mode: str,
) -> str:
    """Canonicalize an answer according to the active reader protocol."""

    mode = reader_mode.strip().lower()
    if mode == "pubmedqa":
        match = re.search(r"\b(yes|no|maybe)\b", str(answer).casefold())
        return match.group(1) if match else str(answer).strip()
    if mode == "quartz":
        return canonicalize_quartz_answer(question, answer)
    if mode == "medqa":
        choice = str(answer).strip().upper()
        if choice not in MEDQA_CHOICES:
            raise ReaderProtocolError(
                f"MedQA reader returned an invalid choice: {answer!r}"
            )
        return choice
    return str(answer).strip()


def reader_answers_exact_match(
    question: str,
    left: str,
    right: str,
    *,
    reader_mode: str,
) -> bool:
    """Compare answers after applying the dataset reader protocol."""

    mode = reader_mode.strip().lower()
    if mode == "quartz":
        return quartz_choice_label(question, left) == quartz_choice_label(
            question, right
        )
    return answers_exact_match(
        canonicalize_reader_answer(question, left, reader_mode=mode),
        canonicalize_reader_answer(question, right, reader_mode=mode),
    )


def reader_answers_match(
    question: str,
    left: str,
    right: str,
    *,
    reader_mode: str,
) -> bool:
    """Apply benchmark-style correctness after protocol canonicalization."""

    mode = reader_mode.strip().lower()
    if mode == "quartz":
        return quartz_choice_label(question, left) == quartz_choice_label(
            question, right
        )
    return answers_match(
        canonicalize_reader_answer(question, left, reader_mode=mode),
        canonicalize_reader_answer(question, right, reader_mode=mode),
    )


def reader_answer_token_f1(
    question: str,
    left: str,
    right: str,
    *,
    reader_mode: str,
) -> float:
    """Compute token F1 after protocol-specific canonicalization."""

    return answer_token_f1(
        canonicalize_reader_answer(question, left, reader_mode=reader_mode),
        canonicalize_reader_answer(question, right, reader_mode=reader_mode),
    )


def parse_quartz_choices(question: str) -> dict[str, str]:
    """Parse the two choices from a frozen QuaRTz question."""

    markers = list(re.finditer(r"\bchoices\s*:\s*", question, flags=re.I))
    if not markers:
        raise ReaderProtocolError("QuaRTz question does not contain 'Choices:'")
    tail = question[markers[-1].end() :]
    labels = list(re.finditer(r"\(\s*([AB])\s*\)", tail, flags=re.I))
    if [match.group(1).upper() for match in labels] != list(QUARTZ_CHOICES):
        raise ReaderProtocolError(
            "QuaRTz question must contain exactly the choices (A) and (B)"
        )

    first, second = labels
    choice_a = tail[first.end() : second.start()].strip()
    choice_b = tail[second.end() :].strip()
    choice_a = re.sub(r"\s*;\s*$", "", choice_a).strip()
    choice_b = re.sub(r"[.?!]+\s*$", "", choice_b).strip()
    choices = {"A": choice_a, "B": choice_b}
    if any(not text for text in choices.values()):
        raise ReaderProtocolError("QuaRTz question contains an empty choice")
    if len({_normalize_quartz_choice(text) for text in choices.values()}) != 2:
        raise ReaderProtocolError("QuaRTz question contains duplicate choices")
    return choices


def quartz_choice_label(question: str, answer: str) -> str:
    """Map one unambiguous QuaRTz response to its canonical A/B label."""

    choices = parse_quartz_choices(question)
    raw = str(answer or "").strip()
    parsed = parse_json_object(raw)
    if isinstance(parsed, dict) and parsed.get("answer") is not None:
        raw = str(parsed["answer"]).strip()
    raw = _strip_matching_quotes(raw)
    if not raw:
        raise ReaderProtocolError("QuaRTz reader returned an empty answer")

    label = _quartz_label_only(raw)
    if label is not None:
        return label

    labelled = _quartz_labelled_text(raw)
    if labelled is not None:
        label, rendered_text = labelled
        if (
            rendered_text
            and _normalize_quartz_choice(rendered_text)
            != _normalize_quartz_choice(choices[label])
        ):
            raise ReaderProtocolError(
                "QuaRTz answer label conflicts with its rendered choice text: "
                f"{answer!r}"
            )
        return label

    normalized = _normalize_quartz_choice(raw)
    matches = [
        label
        for label, text in choices.items()
        if normalized == _normalize_quartz_choice(text)
    ]
    if len(matches) == 1:
        return matches[0]

    partial_matches = [
        label
        for label, text in choices.items()
        if _quartz_contiguous_token_subset(
            normalized,
            _normalize_quartz_choice(text),
        )
    ]
    if len(partial_matches) == 1:
        return partial_matches[0]
    raise ReaderProtocolError(
        f"QuaRTz reader returned an unmappable answer: {answer!r}"
    )


def _normalize_quartz_choice(text: str) -> str:
    """Normalize choice rendering without deleting meaningful articles."""

    normalized = unicodedata.normalize("NFKC", str(text or "")).casefold()
    normalized = re.sub(r"[^\w]+", " ", normalized, flags=re.UNICODE)
    return " ".join(normalized.split())


def _strip_matching_quotes(text: str) -> str:
    value = text.strip()
    pairs = {'"': '"', "'": "'", "`": "`"}
    while len(value) >= 2 and pairs.get(value[0]) == value[-1]:
        value = value[1:-1].strip()
    return value


def _quartz_contiguous_token_subset(answer: str, choice: str) -> bool:
    """Accept a uniquely identifying truncated choice, never a verbose answer."""

    answer_tokens = answer.split()
    choice_tokens = choice.split()
    if not answer_tokens or len(answer_tokens) >= len(choice_tokens):
        return False
    width = len(answer_tokens)
    return any(
        choice_tokens[index : index + width] == answer_tokens
        for index in range(len(choice_tokens) - width + 1)
    )


def _quartz_label_only(text: str) -> str | None:
    match = re.fullmatch(
        r"\s*(?:\(\s*([AB])\s*\)|([AB]))\s*[.)]?\s*",
        text,
        flags=re.I,
    )
    if not match:
        return None
    return (match.group(1) or match.group(2)).upper()


def _quartz_labelled_text(text: str) -> tuple[str, str] | None:
    direct = re.fullmatch(
        r"\s*(?:\(\s*([AB])\s*\)|([AB])\s*[):.\-])\s*(.*?)\s*",
        text,
        flags=re.I | re.S,
    )
    if direct:
        return (direct.group(1) or direct.group(2)).upper(), direct.group(3)

    explicit = re.fullmatch(
        r"\s*(?:the\s+)?(?:answer|choice|option)\s*(?:is|:)\s*"
        r"(?:\(\s*([AB])\s*\)|([AB]))\s*[.)]?\s*",
        text,
        flags=re.I,
    )
    if explicit:
        return (explicit.group(1) or explicit.group(2)).upper(), ""
    return None


def parse_json_object(text: str):
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def normalize_answer(text: str) -> str:
    text = (text or "").lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def answers_match(a: str, b: str) -> bool:
    if (a or "").strip() == (b or "").strip() and (a or "").strip():
        return True
    na, nb = normalize_answer(a), normalize_answer(b)
    return bool(na and nb and (na == nb or na in nb or nb in na))


def answers_exact_match(a: str, b: str) -> bool:
    """Hotpot-style normalized exact match without substring containment."""

    left = (a or "").strip()
    right = (b or "").strip()
    if left == right:
        return bool(left)
    na, nb = normalize_answer(a), normalize_answer(b)
    return bool(na and nb and na == nb)


def answer_token_f1(a: str, b: str) -> float:
    """Token F1 after the same normalization used by exact match."""

    left = normalize_answer(a).split()
    right = normalize_answer(b).split()
    if not left or not right:
        return float(left == right)
    overlap = sum((Counter(left) & Counter(right)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(left)
    recall = overlap / len(right)
    return 2.0 * precision * recall / (precision + recall)
