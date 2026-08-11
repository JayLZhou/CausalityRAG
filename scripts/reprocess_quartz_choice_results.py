#!/usr/bin/env python3
"""Audit or reprocess QuaRTz results using canonical A/B choice identities."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.reader import (  # noqa: E402
    ReaderProtocolError,
    answer_token_f1,
    canonicalize_quartz_answer,
    quartz_choice_label,
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def write_jsonl(path: Path, rows: list[dict], *, inputs: tuple[Path, ...]) -> None:
    resolved = path.resolve()
    if any(resolved == source.resolve() for source in inputs):
        raise ValueError(f"refusing to overwrite an input artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def records_by_id(rows: list[dict], *, artifact: str) -> dict[str, dict]:
    result = {}
    for row in rows:
        query_id = str(row.get("id", ""))
        if not query_id or query_id in result:
            raise ValueError(f"{artifact} contains an empty or duplicate id: {query_id!r}")
        result[query_id] = row
    return result


def canonical_pair(question: str, clean: str, gold: str) -> tuple[str, str, str, str]:
    clean_label = quartz_choice_label(question, clean)
    gold_label = quartz_choice_label(question, gold)
    return (
        clean_label,
        gold_label,
        canonicalize_quartz_answer(question, clean),
        canonicalize_quartz_answer(question, gold),
    )


def reprocess_factual_row(record: dict, row: dict) -> tuple[dict, str]:
    result = dict(row)
    question = str(record.get("question", ""))
    clean_raw = str(row.get("clean_answer", ""))
    gold_raw = str(row.get("gold_answer", record.get("answer", "")))
    try:
        clean_label, gold_label, clean, gold = canonical_pair(
            question, clean_raw, gold_raw
        )
    except ReaderProtocolError as exc:
        result.update({
            "verified_flip": False,
            "evaluation_status": "protocol_violation_invalid_clean_or_gold_answer",
            "quartz_postprocess_status": "unresolved_clean_or_gold",
            "quartz_postprocess_error": str(exc),
        })
        return result, "unresolved_clean_or_gold"

    attempts = []
    true_attempt = None
    last_valid_attempt = None
    unresolved = False
    for source_attempt in row.get("attempts", []):
        attempt = dict(source_attempt)
        if attempt.get("reader_called"):
            edited_raw = str(attempt.get("edited_answer", ""))
            attempt["edited_answer_raw"] = edited_raw
            try:
                edited_label = quartz_choice_label(question, edited_raw)
                attempt["edited_answer"] = canonicalize_quartz_answer(
                    question, edited_raw
                )
                attempt["answer_changed"] = edited_label != clean_label
                attempt["candidate_status"] = (
                    "verified_flip"
                    if attempt["answer_changed"]
                    else "verified_no_flip"
                )
                last_valid_attempt = attempt
                if true_attempt is None and attempt["answer_changed"]:
                    true_attempt = attempt
            except ReaderProtocolError as exc:
                unresolved = True
                attempt.update({
                    "answer_changed": False,
                    "candidate_status": "protocol_violation_invalid_reader_answer",
                    "protocol_error": str(exc),
                })
        attempts.append(attempt)

    original_flip = bool(row.get("verified_flip"))
    selected = true_attempt or last_valid_attempt
    result.update({
        "clean_answer_raw": clean_raw,
        "gold_answer_raw": gold_raw,
        "clean_answer": clean,
        "gold_answer": gold,
        "clean_correct": clean_label == gold_label,
        "attempts": attempts,
        "verified_flip": true_attempt is not None,
    })
    if selected is not None:
        result.update({
            "selected_ids": list(selected.get("selected_ids", [])),
            "selected_tokens": list(selected.get("selected_tokens", [])),
            "n_modified_tokens": int(
                selected.get("n_edits", len(selected.get("selected_ids", [])))
            ),
            "edited_answer": str(selected.get("edited_answer", clean)),
        })
    else:
        result["edited_answer"] = clean

    if true_attempt is not None:
        status = "verified_true_choice_flip"
        result["evaluation_status"] = "verified_flip"
    elif unresolved:
        status = "unresolved_reader_answer"
        result["evaluation_status"] = "protocol_violation_invalid_reader_answer"
    elif original_flip:
        status = "same_choice_false_positive_requires_resume"
        result["evaluation_status"] = status
        result["requires_frontier_resume"] = True
    else:
        status = "verified_no_choice_flip"
        if str(result.get("evaluation_status", "")).startswith("verified_"):
            result["evaluation_status"] = "verified_no_flip"
    result["quartz_postprocess_status"] = status
    result["original_verified_flip"] = original_flip
    return result, status


def reprocess_control_method(
    question: str,
    clean_label: str,
    gold_label: str,
    clean: str,
    gold: str,
    method: dict,
) -> tuple[dict, str]:
    result = dict(method)
    if not method.get("reader_called"):
        return result, "not_called"
    edited_raw = str(method.get("edited_answer", ""))
    try:
        edited_label = quartz_choice_label(question, edited_raw)
        edited = canonicalize_quartz_answer(question, edited_raw)
    except ReaderProtocolError as exc:
        result.update({
            "status": "protocol_violation_invalid_reader_answer",
            "edited_answer_raw": edited_raw,
            "answer_flip": False,
            "f1_flip": False,
            "em_flip": False,
            "acc_flip": False,
            "quartz_postprocess_status": "unresolved_reader_answer",
            "protocol_error": str(exc),
        })
        return result, "unresolved_reader_answer"

    original_flip = bool(method.get("answer_flip"))
    answer_flip = edited_label != clean_label
    clean_f1 = answer_token_f1(clean, gold)
    edited_f1 = answer_token_f1(edited, gold)
    result.update({
        "status": "evaluated",
        "clean_answer": clean,
        "gold_answer": gold,
        "edited_answer_raw": edited_raw,
        "edited_answer": edited,
        "answer_flip": answer_flip,
        "f1_flip": edited_f1 < clean_f1,
        "em_flip": clean_label == gold_label and edited_label != gold_label,
        "acc_flip": clean_label == gold_label and edited_label != gold_label,
        "original_answer_flip": original_flip,
        "quartz_postprocess_status": (
            "verified_true_choice_flip" if answer_flip else "verified_no_choice_flip"
        ),
    })
    if original_flip and not answer_flip:
        result["quartz_postprocess_status"] = "same_choice_false_positive"
        return result, "same_choice_false_positive"
    return result, "verified_true_choice_flip" if answer_flip else "verified_no_choice_flip"


def reprocess_control_row(record: dict, row: dict) -> tuple[dict, Counter]:
    result = dict(row)
    question = str(record.get("question", ""))
    clean_raw = str(row.get("clean_answer", ""))
    gold_raw = str(row.get("gold_answer", record.get("answer", "")))
    statuses = Counter()
    try:
        clean_label, gold_label, clean, gold = canonical_pair(
            question, clean_raw, gold_raw
        )
    except ReaderProtocolError as exc:
        result.update({
            "quartz_postprocess_status": "unresolved_clean_or_gold",
            "quartz_postprocess_error": str(exc),
        })
        statuses["unresolved_clean_or_gold"] += 1
        return result, statuses

    methods = {}
    for name, method in row.get("methods", {}).items():
        methods[name], status = reprocess_control_method(
            question,
            clean_label,
            gold_label,
            clean,
            gold,
            method,
        )
        statuses[status] += 1
    result.update({
        "clean_answer_raw": clean_raw,
        "gold_answer_raw": gold_raw,
        "clean_answer": clean,
        "gold_answer": gold,
        "methods": methods,
    })
    return result, statuses


def reprocess_baseline_method(
    question: str,
    clean_label: str,
    clean: str,
    gold: str,
    method: dict,
) -> tuple[dict, str]:
    result = dict(method)
    if not method.get("reader_called"):
        return result, "not_called"
    edited_raw = str(method.get("edited_answer", ""))
    try:
        edited_label = quartz_choice_label(question, edited_raw)
        edited = canonicalize_quartz_answer(question, edited_raw)
    except ReaderProtocolError as exc:
        result.update({
            "status": "protocol_violation_invalid_reader_answer",
            "edited_answer_raw": edited_raw,
            "verified_flip": False,
            "quartz_postprocess_status": "unresolved_reader_answer",
            "protocol_error": str(exc),
        })
        return result, "unresolved_reader_answer"

    original_flip = bool(method.get("verified_flip"))
    verified_flip = edited_label != clean_label
    result.update({
        "status": "verified_flip" if verified_flip else "verified_no_flip",
        "clean_answer": clean,
        "gold_answer": gold,
        "edited_answer_raw": edited_raw,
        "edited_answer": edited,
        "verified_flip": verified_flip,
        "original_verified_flip": original_flip,
        "quartz_postprocess_status": (
            "verified_true_choice_flip"
            if verified_flip
            else "verified_no_choice_flip"
        ),
    })
    if original_flip and not verified_flip:
        result["quartz_postprocess_status"] = "same_choice_false_positive"
        return result, "same_choice_false_positive"
    return result, (
        "verified_true_choice_flip"
        if verified_flip
        else "verified_no_choice_flip"
    )


def reprocess_baseline_row(record: dict, row: dict) -> tuple[dict, Counter]:
    result = dict(row)
    question = str(record.get("question", ""))
    clean_raw = str(row.get("clean_answer", ""))
    gold_raw = str(row.get("gold_answer", record.get("answer", "")))
    statuses = Counter()
    try:
        clean_label, _, clean, gold = canonical_pair(
            question, clean_raw, gold_raw
        )
    except ReaderProtocolError as exc:
        result.update({
            "quartz_postprocess_status": "unresolved_clean_or_gold",
            "quartz_postprocess_error": str(exc),
        })
        statuses["unresolved_clean_or_gold"] += 1
        return result, statuses

    methods = {}
    for name, method in row.get("methods", {}).items():
        methods[name], status = reprocess_baseline_method(
            question,
            clean_label,
            clean,
            gold,
            method,
        )
        statuses[status] += 1
    result.update({
        "clean_answer_raw": clean_raw,
        "gold_answer_raw": gold_raw,
        "clean_answer": clean,
        "gold_answer": gold,
        "methods": methods,
    })
    return result, statuses


def reprocess_baselines(
    retrieval_rows: list[dict],
    baseline_rows: list[dict],
) -> tuple[list[dict], dict]:
    records = records_by_id(retrieval_rows, artifact="retrieval")
    baselines = records_by_id(baseline_rows, artifact="baselines")
    if baselines.keys() != records.keys():
        raise ValueError("retrieval and baseline query-id sets must match")
    statuses = Counter()
    output = []
    for query_id, record in records.items():
        row, row_statuses = reprocess_baseline_row(
            record, baselines[query_id]
        )
        statuses.update(row_statuses)
        output.append(row)
    return output, dict(sorted(statuses.items()))


def reprocess(
    retrieval_rows: list[dict],
    factual_rows: list[dict],
    control_rows: list[dict],
) -> tuple[list[dict], list[dict], dict]:
    records = records_by_id(retrieval_rows, artifact="retrieval")
    factual = records_by_id(factual_rows, artifact="factual")
    control = records_by_id(control_rows, artifact="control")
    if factual.keys() != records.keys() or control.keys() != records.keys():
        raise ValueError("retrieval, factual, and control query-id sets must match")

    factual_statuses = Counter()
    control_statuses = Counter()
    reprocessed_factual = []
    reprocessed_control = []
    for query_id in records:
        factual_row, factual_status = reprocess_factual_row(
            records[query_id], factual[query_id]
        )
        control_row, method_statuses = reprocess_control_row(
            records[query_id], control[query_id]
        )
        factual_statuses[factual_status] += 1
        control_statuses.update(method_statuses)
        reprocessed_factual.append(factual_row)
        reprocessed_control.append(control_row)

    report = {
        "schema": "causalityrag.quartz_choice_postprocess.v1",
        "queries": len(records),
        "factual_statuses": dict(sorted(factual_statuses.items())),
        "control_method_statuses": dict(sorted(control_statuses.items())),
        "requires_frontier_resume": factual_statuses[
            "same_choice_false_positive_requires_resume"
        ],
    }
    return reprocessed_factual, reprocessed_control, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--factual", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--baselines", type=Path)
    parser.add_argument("--factual-out", type=Path)
    parser.add_argument("--control-out", type=Path)
    parser.add_argument("--baselines-out", type=Path)
    parser.add_argument("--report-out", type=Path)
    args = parser.parse_args()

    factual, control, report = reprocess(
        read_jsonl(args.retrieval),
        read_jsonl(args.factual),
        read_jsonl(args.control),
    )
    inputs = (args.retrieval, args.factual, args.control)
    if bool(args.baselines) != bool(args.baselines_out):
        raise ValueError("--baselines and --baselines-out must be provided together")
    if args.baselines:
        baseline_rows, baseline_statuses = reprocess_baselines(
            read_jsonl(args.retrieval),
            read_jsonl(args.baselines),
        )
        report["baseline_method_statuses"] = baseline_statuses
        inputs = (*inputs, args.baselines)
    if args.factual_out:
        write_jsonl(args.factual_out, factual, inputs=inputs)
    if args.control_out:
        write_jsonl(args.control_out, control, inputs=inputs)
    if args.baselines_out:
        write_jsonl(args.baselines_out, baseline_rows, inputs=inputs)
    rendered = json.dumps(report, indent=2) + "\n"
    if args.report_out:
        if args.report_out.resolve() in {path.resolve() for path in inputs}:
            raise ValueError("refusing to overwrite an input artifact")
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
