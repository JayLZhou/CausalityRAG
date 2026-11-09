"""Summarize a minimum-token intervention as an untargeted RAG attack.

Attack success is deliberately stricter than response flip: an eligible query
must be answered correctly before intervention and incorrectly afterwards.
Queries without an executed edit count as attack failures in the ASR
denominator.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from typing import Any, Iterable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.reader import answer_token_f1, answers_exact_match, answers_match


def load_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def reflow_result(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "reader_called": int(row.get("reader_calls", 0)) > 0,
        "edited_answer": str(row.get("edited_answer", "")),
        "n_modified_tokens": int(
            row.get("n_modified_tokens", len(row.get("selected_ids", [])))
        ),
        "selected_tokens": list(row.get("selected_tokens", [])),
        "edits": list(row.get("edits", [])),
        "reader_calls": int(row.get("reader_calls", 0)),
    }


def summarize_method(
    rows: Iterable[tuple[dict[str, Any], dict[str, Any]]],
    *,
    max_examples: int,
) -> dict[str, Any]:
    eligible = 0
    em_eligible = 0
    executed = 0
    successes = 0
    em_successes = 0
    answer_flips = 0
    reader_calls = 0
    executed_sizes: list[int] = []
    success_sizes: list[int] = []
    f1_deltas: list[float] = []
    examples: list[dict[str, Any]] = []

    for parent, method in rows:
        clean = str(parent.get("clean_answer", ""))
        gold = str(parent.get("gold_answer", ""))
        if not clean or not gold or not answers_match(clean, gold):
            continue
        eligible += 1
        clean_exact = answers_exact_match(clean, gold)
        em_eligible += int(clean_exact)

        called = bool(method.get("reader_called", False))
        edited = str(method.get("edited_answer", method.get("answer", "")))
        if not called or not edited.strip():
            continue

        executed += 1
        reader_calls += int(method.get("reader_calls", 1))
        size = int(
            method.get("n_modified_tokens", len(method.get("selected_ids", [])))
        )
        executed_sizes.append(size)
        answer_flips += int(not answers_exact_match(edited, clean))
        em_successes += int(clean_exact and not answers_exact_match(edited, gold))
        f1_delta = answer_token_f1(edited, gold) - answer_token_f1(clean, gold)
        f1_deltas.append(f1_delta)

        if answers_match(edited, gold):
            continue
        successes += 1
        success_sizes.append(size)
        if len(examples) < max_examples:
            examples.append(
                {
                    "id": str(parent.get("id", "")),
                    "gold_answer": gold,
                    "clean_answer": clean,
                    "attacked_answer": edited,
                    "n_modified_tokens": size,
                    "selected_tokens": list(method.get("selected_tokens", [])),
                    "edits": list(method.get("edits", [])),
                }
            )

    return {
        "eligible_clean_correct_queries": eligible,
        "eligible_clean_exact_match_queries": em_eligible,
        "executed_queries": executed,
        "attack_successes": successes,
        "untargeted_asr": successes / max(1, eligible),
        "exact_match_attack_successes": em_successes,
        "untargeted_em_asr": em_successes / max(1, em_eligible),
        "answer_flip_count": answer_flips,
        "answer_flip_rate_on_eligible": answer_flips / max(1, eligible),
        "reader_calls": reader_calls,
        "mean_modified_tokens_on_executed": (
            statistics.fmean(executed_sizes) if executed_sizes else 0.0
        ),
        "mean_modified_tokens_on_success": (
            statistics.fmean(success_sizes) if success_sizes else 0.0
        ),
        "mean_gold_f1_delta": (
            statistics.fmean(f1_deltas) if f1_deltas else 0.0
        ),
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reflow", required=True)
    parser.add_argument("--baselines")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-examples", type=int, default=5)
    args = parser.parse_args()

    reflow_rows = load_jsonl(args.reflow)
    output: dict[str, Any] = {
        "metric_contract": {
            "eligible_query": "the frozen clean response matches the gold answer",
            "untargeted_attack_success": (
                "an eligible query has an executed intervention whose response "
                "does not match the gold answer"
            ),
            "denominator": (
                "all eligible clean-correct queries; an unexecuted intervention "
                "counts as attack failure"
            ),
        },
        "sources": {"reflow": os.path.abspath(args.reflow)},
        "methods": {
            "reflow": summarize_method(
                ((row, reflow_result(row)) for row in reflow_rows),
                max_examples=args.max_examples,
            )
        },
    }

    if args.baselines:
        baseline_rows = load_jsonl(args.baselines)
        baseline_by_id = {str(row.get("id", "")): row for row in baseline_rows}
        if len(baseline_by_id) != len(baseline_rows):
            raise ValueError("baseline rows must have unique ids")
        if any(str(row.get("id", "")) not in baseline_by_id for row in reflow_rows):
            raise ValueError("baseline results do not cover every ReFlow query")
        method_names = sorted(
            {
                method
                for row in baseline_rows
                for method in row.get("methods", {})
            }
        )
        for method_name in method_names:
            output["methods"][method_name] = summarize_method(
                (
                    (
                        row,
                        baseline_by_id[str(row.get("id", ""))]
                        .get("methods", {})
                        .get(method_name, {}),
                    )
                    for row in reflow_rows
                ),
                max_examples=args.max_examples,
            )
        output["sources"]["baselines"] = os.path.abspath(args.baselines)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
