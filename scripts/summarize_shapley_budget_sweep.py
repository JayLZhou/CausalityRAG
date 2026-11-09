#!/usr/bin/env python3
"""Merge Shapley budget shards and score Acc-FR with the unified protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causalityrag.evaluation_metrics import (
    clean_correct,
    gold_accuracy,
    gold_answer_spec,
    valid_answer,
)


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def table(path: Path) -> dict[str, dict]:
    rows = load(path)
    result = {str(row["id"]): row for row in rows}
    if len(rows) != 1000 or len(result) != 1000:
        raise ValueError(f"{path}: expected 1,000 unique queries")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--reflow", type=Path, required=True)
    parser.add_argument("--shards", type=Path, nargs="+", required=True)
    parser.add_argument("--out-rows", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    args = parser.parse_args()

    retrieval = table(args.retrieval)
    reflow = table(args.reflow)
    sweep_rows = []
    for path in args.shards:
        sweep_rows.extend(load(path))
    sweep = {str(row["id"]): row for row in sweep_rows}
    if len(sweep_rows) != 1000 or len(sweep) != 1000:
        raise ValueError("sweep shards must contain 1,000 unique queries")
    if not (set(retrieval) == set(reflow) == set(sweep)):
        raise ValueError("retrieval, ReFlow, and sweep IDs do not align")

    reader_mode = "quartz" if args.dataset == "quartz" else "short_answer"
    budgets = sorted({
        int(budget)
        for row in sweep.values()
        for budget in row.get("methods", {})
        .get("token_leverage_shap", {})
    })
    population = []
    for identifier, record in retrieval.items():
        clean = str(reflow[identifier].get("clean_answer", ""))
        spec = gold_answer_spec(record, args.dataset)
        if valid_answer(clean) and clean_correct(
            "acc", clean, spec, reader_mode=reader_mode
        ):
            population.append((identifier, spec))

    def score_entry(entry: dict, spec) -> tuple[bool, bool]:
        status = str(entry.get("status", ""))
        if status.startswith("protocol_violation"):
            return False, False
        answer = str(entry.get("edited_answer", ""))
        if not answer:
            return True, False
        if not valid_answer(answer):
            return False, False
        return True, not gold_accuracy(answer, spec, reader_mode=reader_mode)

    reflow_valid = reflow_flips = 0
    for identifier, spec in population:
        row = reflow[identifier]
        if str(row.get("evaluation_status", "")).startswith("protocol_violation"):
            continue
        answer = str(row.get("edited_answer", ""))
        if answer and not valid_answer(answer):
            continue
        reflow_valid += 1
        reflow_flips += int(
            bool(answer)
            and not gold_accuracy(answer, spec, reader_mode=reader_mode)
        )
    target = reflow_flips / max(1, reflow_valid)

    curve = {}
    for budget in budgets:
        valid = flips = token_sum = 0
        for identifier, spec in population:
            entry = sweep[identifier].get("methods", {}).get(
                "token_leverage_shap", {}
            ).get(str(budget), {})
            is_valid, is_flip = score_entry(entry, spec)
            if not is_valid:
                continue
            valid += 1
            flips += int(is_flip)
            token_sum += int(entry.get("n_modified_tokens", 0))
        curve[str(budget)] = {
            "valid_queries": valid,
            "acc_flip_count": flips,
            "acc_fr": flips / max(1, valid),
            "mean_modified_tokens": token_sum / max(1, valid),
        }

    first_match = next(
        (
            budget
            for budget in budgets
            if curve[str(budget)]["acc_fr"] >= target
        ),
        None,
    )
    args.out_rows.parent.mkdir(parents=True, exist_ok=True)
    with args.out_rows.open("w", encoding="utf-8") as output:
        for row in sorted(sweep.values(), key=lambda item: int(item["index"])):
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "dataset": args.dataset,
        "queries": 1000,
        "clean_acc_queries": len(population),
        "reflow": {
            "valid_queries": reflow_valid,
            "acc_flip_count": reflow_flips,
            "acc_fr": target,
        },
        "token_shapley": curve,
        "first_matching_tested_budget": first_match,
        "reaches_target_through_128": first_match is not None,
    }
    args.out_summary.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
