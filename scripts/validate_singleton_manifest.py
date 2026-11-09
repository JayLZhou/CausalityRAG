#!/usr/bin/env python3
"""Validate selected singleton edits from several saved ReFlow cases."""

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
except ImportError:
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
            raise ValueError(f"edit mismatch for {edit['unit_id']}")
        context["text"] = text[:start] + str(edit["new"]) + text[end:]
        return result
    raise ValueError(f"missing chunk {edit['chunk_id']}")


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
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    cases = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    prepared = []
    for case in cases:
        records = {record_id(row): row for row in load_records(case["input"])}
        results = {record_id(row): row for row in load_records(case["reflow"])}
        record = records[case["id"]]
        result = results[case["id"]]
        selected = set(map(str, result["selected_ids"]))
        attempts = [
            attempt
            for attempt in result.get("attempts", [])
            if set(map(str, attempt.get("selected_ids", []))) == selected
        ]
        edit = next(
            edit
            for edit in attempts[-1]["edits"]
            if str(edit["unit_id"]) == str(case["unit_id"])
        )
        question = str(record["question"])
        contexts = retrieved_contexts(record)[: int(case.get("k", 5))]
        prepared.append((case, result, question, edit, apply_one(contexts, edit)))

    reader = LocalHFReader(
        args.model_path,
        device=args.device,
        dtype=args.dtype,
        attn_implementation="sdpa",
    )
    output = []
    for case, result, question, edit, contexts in prepared:
        raw = reader.generate_responses_batch(question, [contexts])[0]
        output.append({
            "id": case["id"],
            "question": question,
            "clean_answer": result["clean_answer"],
            "joint_answer": result["edited_answer"],
            "unit_id": str(edit["unit_id"]),
            "old": str(edit["old"]),
            "new": str(edit["new"]),
            "raw_response": raw,
            "answer": parsed_answer(
                question, raw, str(case.get("reader_mode", "short_answer"))
            ),
        })
    rendered = json.dumps(output, ensure_ascii=False, indent=2)
    Path(args.out).write_text(rendered + "\n", encoding="utf-8")
    print(rendered, flush=True)


if __name__ == "__main__":
    main()
