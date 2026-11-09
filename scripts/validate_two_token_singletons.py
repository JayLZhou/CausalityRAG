#!/usr/bin/env python3
"""Validate both singleton edits for one saved two-token ReFlow result."""

from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.io import load_records, record_id, retrieved_contexts
from causalityrag.reader import LocalHFReader, parse_json_object

try:
    from causalityrag.reader import canonicalize_reader_answer
except ImportError:  # Compatibility with older experiment-server checkout.
    def canonicalize_reader_answer(
        question: str, answer: str, *, reader_mode: str
    ) -> str:
        del question, reader_mode
        return str(answer).strip()


def apply_one(contexts: list[dict], edit: dict) -> list[dict]:
    result = deepcopy(contexts)
    for context in result:
        if str(context.get("chunk_id")) != str(edit["chunk_id"]):
            continue
        text = str(context["text"])
        start = int(edit["chunk_char_start"])
        end = int(edit["chunk_char_end"])
        if text[start:end] != str(edit["old"]):
            raise ValueError(
                f"edit mismatch for {edit['unit_id']}: "
                f"expected {edit['old']!r}, found {text[start:end]!r}"
            )
        context["text"] = text[:start] + str(edit["new"]) + text[end:]
        return result
    raise ValueError(f"missing chunk {edit['chunk_id']}")


def apply_many(contexts: list[dict], edits: list[dict]) -> list[dict]:
    result = deepcopy(contexts)
    by_chunk: dict[str, list[dict]] = {}
    for edit in edits:
        by_chunk.setdefault(str(edit["chunk_id"]), []).append(edit)
    for context in result:
        chunk_edits = by_chunk.pop(str(context.get("chunk_id")), [])
        text = str(context["text"])
        for edit in sorted(
            chunk_edits,
            key=lambda item: int(item["chunk_char_start"]),
            reverse=True,
        ):
            start = int(edit["chunk_char_start"])
            end = int(edit["chunk_char_end"])
            if text[start:end] != str(edit["old"]):
                raise ValueError(f"edit mismatch for {edit['unit_id']}")
            text = text[:start] + str(edit["new"]) + text[end:]
        context["text"] = text
    if by_chunk:
        raise ValueError(f"missing chunks {sorted(by_chunk)}")
    return result


def parsed_answer(question: str, response: str, reader_mode: str) -> str:
    parsed = parse_json_object(response)
    answer = (
        str(parsed["answer"]).strip()
        if isinstance(parsed, dict) and parsed.get("answer") is not None
        else response.strip()
    )
    return canonicalize_reader_answer(question, answer, reader_mode=reader_mode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--reflow", required=True)
    parser.add_argument("--id", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--reader-mode", default="short_answer")
    parser.add_argument("--only-unit-id", default="")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--include-joint", action="store_true")
    parser.add_argument("--joint-only", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    records = {record_id(row): row for row in load_records(args.input)}
    results = {record_id(row): row for row in load_records(args.reflow)}
    record = records[args.id]
    result = results[args.id]
    selected = set(map(str, result["selected_ids"]))
    attempts = [
        attempt
        for attempt in result.get("attempts", [])
        if set(map(str, attempt.get("selected_ids", []))) == selected
    ]
    if len(selected) != 2 or not attempts:
        raise ValueError("expected a saved two-token attempt")
    edits = attempts[-1]["edits"]
    if len(edits) != 2:
        raise ValueError("expected exactly two saved edits")
    if args.only_unit_id:
        edits = [
            edit for edit in edits if str(edit["unit_id"]) == args.only_unit_id
        ]
        if len(edits) != 1:
            raise ValueError(f"missing requested unit {args.only_unit_id}")
    overrides = dict(value.split("=", 1) for value in args.override)
    for index, edit in enumerate(edits):
        unit_id = str(edit["unit_id"])
        if unit_id in overrides:
            edit = edit.copy()
            edit["new"] = overrides[unit_id]
            edits[index] = edit

    question = str(record["question"])
    contexts = retrieved_contexts(record)[: args.k]
    variants = [apply_one(contexts, edit) for edit in edits]
    labels = [("singleton", edit) for edit in edits]
    if args.include_joint and len(edits) == 2:
        variants.append(apply_many(contexts, edits))
        labels.append(("joint", None))
    if args.joint_only:
        if len(edits) != 2:
            raise ValueError("--joint-only requires exactly two edits")
        variants = [apply_many(contexts, edits)]
        labels = [("joint", None)]
    reader = LocalHFReader(
        args.model_path,
        device=args.device,
        dtype=args.dtype,
        attn_implementation="sdpa",
    )
    raw = reader.generate_responses_batch(question, variants)
    rows = []
    for (kind, edit), response in zip(labels, raw):
        rows.append({
            "kind": kind,
            "unit_id": str(edit["unit_id"]) if edit else None,
            "old": str(edit["old"]) if edit else None,
            "new": str(edit["new"]) if edit else None,
            "raw_response": response,
            "answer": parsed_answer(question, response, args.reader_mode),
        })
    payload = {
        "id": args.id,
        "question": question,
        "clean_answer": result["clean_answer"],
        "joint_answer": result["edited_answer"],
        "singletons": rows,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    Path(args.out).write_text(rendered + "\n", encoding="utf-8")
    print(rendered, flush=True)


if __name__ == "__main__":
    main()
