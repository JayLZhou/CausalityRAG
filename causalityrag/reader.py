"""vLLM reader calls and experimental local-HF reader utilities."""

from __future__ import annotations

import json
import os
import re
import string
import urllib.request
from collections import Counter
from typing import Sequence


READ_SYSTEM = (
    "Answer the question using ONLY the passages. Give the shortest answer span. "
    "The answer must be non-empty; if uncertain, return the best-supported span "
    "from the passages. Output JSON only."
)
READ_USER = """Passages:
{passages}

Question: {question}
Return STRICT JSON: {{"answer": "..."}}"""


class ReaderClient:
    """Concurrent-safe client for the vLLM OpenAI-compatible endpoint."""

    def __init__(
        self, base_url: str | None = None, model: str | None = None, timeout: int = 120
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

    def answer(self, question: str, contexts: list[dict]) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": READ_SYSTEM},
                {
                    "role": "user",
                    "content": READ_USER.format(
                        question=question, passages=format_passages(contexts)
                    ),
                },
            ],
            "temperature": 0,
            "max_tokens": 96,
        }
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
            return str(parsed["answer"]).strip()
        return str(content).strip()


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
        *,
        max_new_tokens: int = 96,
    ) -> list[str]:
        """Greedily generate answers for context variants with left padding."""

        if not context_variants:
            return []
        torch = self.torch
        prompts = [
            self._prompt_ids(question, contexts) for contexts in context_variants
        ]
        max_prompt_length = max(len(prompt) for prompt in prompts)
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


def format_passages(contexts: list[dict]) -> str:
    return "\n\n".join(
        f"[{ctx.get('chunk_id', i)}] {ctx.get('text', '')}"
        for i, ctx in enumerate(contexts)
    )


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
