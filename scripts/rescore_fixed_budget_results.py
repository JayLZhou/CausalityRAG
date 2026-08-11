#!/usr/bin/env python3
"""Rescore and merge saved fixed-budget reader outputs without new LLM calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.io import gold_answers, load_records, record_id
from causalityrag.reader import ReaderProtocolError, canonicalize_reader_answer
from scripts.evaluate_fixed_budget_acc import (
    answer_is_correct,
    rescore_seeded_entry,
    summarize,
    valid_clean_answer,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def merge_result_rows(paths: list[Path], expected_ids: list[str]) -> dict[str, dict]:
    expected = set(expected_ids)
    merged: dict[str, dict] = {}
    for path in paths:
        rows = load_records(path)
        by_id = {record_id(row): row for row in rows}
        if len(rows) != len(expected_ids) or set(by_id) != expected:
            raise ValueError(
                f"fixed-budget ID mismatch in {path}: rows={len(rows)} "
                f"unique={len(by_id)} expected={len(expected_ids)}"
            )
        for identifier in expected_ids:
            source = by_id[identifier]
            target = merged.setdefault(
                identifier,
                {
                    "index": source.get("index", -1),
                    "id": identifier,
                    "clean_answer": source.get("clean_answer", ""),
                    "methods": {},
                    "reader_calls": 0,
                },
            )
            if str(target.get("clean_answer", "")) != str(
                source.get("clean_answer", "")
            ):
                raise ValueError(f"clean-answer mismatch for {identifier}")
            target["reader_calls"] += int(source.get("reader_calls", 0))
            for method, budgets in source.get("methods", {}).items():
                method_target = target["methods"].setdefault(method, {})
                for budget, entry in budgets.items():
                    if budget in method_target and method_target[budget] != entry:
                        raise ValueError(
                            f"conflicting duplicate {identifier}/{method}/{budget}"
                        )
                    method_target[budget] = dict(entry)
    return merged


def rescore_rows(
    records: list[dict],
    merged: dict[str, dict],
    *,
    reader_mode: str,
) -> tuple[list[dict], list[int]]:
    output = []
    budgets = set()
    for index, record in enumerate(records):
        identifier = record_id(record)
        source = merged[identifier]
        question = str(record.get("question", ""))
        clean_raw = str(source.get("clean_answer", ""))
        accepted_raw = gold_answers(record)
        try:
            clean = canonicalize_reader_answer(
                question,
                clean_raw,
                reader_mode=reader_mode,
            )
            accepted = [
                canonicalize_reader_answer(
                    question,
                    answer,
                    reader_mode=reader_mode,
                )
                for answer in accepted_raw
            ]
        except ReaderProtocolError as exc:
            output.append({
                **source,
                "index": index,
                "clean_acc": False,
                "gold_answer": str(record.get("answer", "")),
                "gold_answers": accepted_raw,
                "methods": {},
                "evaluation_status": (
                    "protocol_violation_invalid_clean_or_gold_answer"
                ),
                "protocol_error": str(exc),
            })
            continue
        clean_acc = bool(
            valid_clean_answer(clean)
            and accepted
            and answer_is_correct(
                question,
                clean,
                accepted,
                reader_mode=reader_mode,
            )
        )
        methods = {}
        if clean_acc:
            for method, method_rows in source.get("methods", {}).items():
                rescored = {}
                for budget, entry in method_rows.items():
                    budgets.add(int(budget))
                    if entry.get("edited_answer") or str(
                        entry.get("status", "")
                    ).startswith("protocol_violation_invalid_reader_answer"):
                        rescored[budget] = rescore_seeded_entry(
                            entry,
                            question=question,
                            accepted_gold_answers=accepted,
                            reader_mode=reader_mode,
                        )
                    else:
                        rescored[budget] = dict(entry)
                methods[method] = rescored
        output.append({
            **source,
            "index": index,
            "clean_answer": clean,
            "clean_answer_raw": clean_raw,
            "gold_answer": str(record.get("answer", "")),
            "gold_answers": accepted,
            "clean_acc": clean_acc,
            "methods": methods,
            "evaluation_status": (
                "rescored" if clean_acc else "outside_clean_acc_population"
            ),
        })
    return output, sorted(budgets)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--results", type=Path, action="append", required=True)
    parser.add_argument("--reader-mode", default="short_answer")
    parser.add_argument(
        "--clean-results",
        type=Path,
        help="Optional final result rows whose clean_answer overrides old sweep anchors.",
    )
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    args = parser.parse_args()

    records = load_records(args.input)[: args.n]
    expected_ids = [record_id(record) for record in records]
    if len(records) != args.n or len(set(expected_ids)) != args.n:
        raise ValueError("input does not contain the expected unique query IDs")
    merged = merge_result_rows(args.results, expected_ids)
    if args.clean_results is not None:
        clean_rows = load_records(args.clean_results)[: args.n]
        clean_by_id = {record_id(row): row for row in clean_rows}
        if len(clean_rows) != args.n or set(clean_by_id) != set(expected_ids):
            raise ValueError("clean-results ID alignment failure")
        for identifier in expected_ids:
            merged[identifier]["clean_answer"] = str(
                clean_by_id[identifier].get("clean_answer", "")
            )
    rows, budgets = rescore_rows(
        records,
        merged,
        reader_mode=args.reader_mode,
    )
    summary = summarize(rows, budgets)
    summary.update({
        "schema": "causalityrag.fixed_budget_rescore.v1",
        "metric": "Acc-CFR",
        "reader_mode": args.reader_mode,
        "new_reader_calls": 0,
        "input": {"path": str(args.input.resolve()), "sha256": sha256(args.input)},
        "source_results": [
            {"path": str(path.resolve()), "sha256": sha256(path)}
            for path in args.results
        ],
        "clean_results": (
            {
                "path": str(args.clean_results.resolve()),
                "sha256": sha256(args.clean_results),
            }
            if args.clean_results is not None
            else None
        ),
    })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
