"""Input normalization for small JSON/JSONL RAG records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def load_records(path: str | Path) -> list[dict]:
    path = Path(path)
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    for key in ("records", "questions", "data", "examples", "items"):
        if isinstance(data.get(key), list):
            return data[key]
    return [data]


def record_id(record: dict, default: str = "") -> str:
    for key in ("id", "_id", "qid", "question_id"):
        value = record.get(key)
        if value is not None:
            return str(value)
    return default


def retrieved_contexts(record: dict) -> list[dict]:
    """Return normalized contexts as ``{chunk_id, text, title, rank}``.

    Handles the common shapes used by HotpotQA-style retrieval files:
    ``retrieved``/``ctxs``/``contexts`` as strings or dictionaries.
    """

    raw = None
    for key in ("retrieved", "contexts", "ctxs", "docs", "passages", "results", "top_k", "topk"):
        value = record.get(key)
        if isinstance(value, list):
            raw = value
            break
    if raw is None:
        text = record.get("context") or record.get("text") or record.get("passage")
        return [{"chunk_id": "context:0", "text": str(text), "title": "", "rank": 1}] if text else []

    out = []
    for i, item in enumerate(raw):
        rank = i + 1
        if isinstance(item, str):
            out.append({"chunk_id": f"context:{i}", "text": item, "title": "", "rank": rank})
            continue
        if not isinstance(item, dict):
            continue
        chunk_id = item.get("chunk_id") or item.get("id") or item.get("cid") or item.get("doc_id") or f"context:{i}"
        text = item.get("text") or item.get("content") or item.get("passage") or item.get("body") or ""
        out.append({
            "chunk_id": str(chunk_id),
            "text": str(text),
            "title": str(item.get("title", "")),
            "rank": int(item.get("rank", rank) or rank),
        })
    return sorted(out, key=lambda x: x["rank"])


def iter_records(path: str | Path, limit: int = 0) -> Iterable[dict]:
    for i, record in enumerate(load_records(path)):
        if limit and i >= limit:
            break
        yield record
