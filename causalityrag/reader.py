"""OpenAI-compatible reader calls for answer-change verification."""

from __future__ import annotations

import json
import os
import re
import string
import urllib.request


READ_SYSTEM = "Answer the question using ONLY the passages. Give the shortest answer span. Output JSON only."
READ_USER = """Passages:
{passages}

Question: {question}
Return STRICT JSON: {{"answer": "..."}}"""


class ReaderClient:
    def __init__(self, base_url: str | None = None, model: str | None = None, timeout: int = 120) -> None:
        self.base_url = (base_url or os.environ.get("CAUSALITYRAG_LLM_BASE_URL")
                         or os.environ.get("YVETTE_LLM_BASE_URL")
                         or "http://127.0.0.1:8000/v1").rstrip("/")
        self.model = model or os.environ.get("CAUSALITYRAG_LLM_MODEL") or os.environ.get("YVETTE_LLM_MODEL") or "qwen2.5-7b"
        self.timeout = timeout

    def answer(self, question: str, contexts: list[dict]) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": READ_SYSTEM},
                {"role": "user", "content": READ_USER.format(question=question, passages=format_passages(contexts))},
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


def format_passages(contexts: list[dict]) -> str:
    return "\n\n".join(f"[{ctx.get('chunk_id', i)}] {ctx.get('text', '')}" for i, ctx in enumerate(contexts))


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
