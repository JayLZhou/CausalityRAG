#!/usr/bin/env python3
"""Evaluate Acc-CFR at fixed token budgets on one frozen RAG workload."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.io import load_records, record_id
from causalityrag.reader import ReaderClient, answers_match
from causalityrag.revision import apply_token_replacements
from causalityrag.shared_replacement_pool import (
    FrozenSharedReplacementPool,
    file_sha256,
    stable_shared_candidate,
)
from causalityrag.token_units import units_from_cache_row
from scripts.evaluate_matched_budget_baselines import ranked_ids
from scripts.evaluate_reflow import candidate_rows, valid_clean_answer


def parse_score_specs(specs: list[str], n: int) -> dict[str, list[dict]]:
    tables: dict[str, list[dict]] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"invalid --scores value: {spec}")
        name, path = spec.split("=", 1)
        name = name.strip()
        if not name or name in tables:
            raise ValueError(f"duplicate or empty baseline name: {name}")
        tables[name] = load_records(path)[:n]
    return tables


def reflow_set(frontier_row: dict, budget: int, eligible: set[str]) -> list[str]:
    candidates = []
    for candidate in candidate_rows(frontier_row):
        selected = [str(value) for value in candidate.get("selected_ids", [])]
        if not selected or len(selected) > budget or not set(selected) <= eligible:
            continue
        try:
            residual = float(candidate.get("remaining_support_flow", math.inf))
        except (TypeError, ValueError):
            residual = math.inf
        candidates.append((residual, -len(selected), tuple(selected), selected))
    return min(candidates)[-1] if candidates else []


def evaluate_query(
    record: dict,
    units_row: dict,
    frontier_row: dict,
    score_rows: dict[str, dict],
    pool: FrozenSharedReplacementPool,
    reader: ReaderClient,
    *,
    budgets: list[int],
    replacement_seed: int,
    k: int,
) -> dict:
    identifier = record_id(record)
    clean_answer = str(frontier_row.get("clean_answer", ""))
    gold_answer = str(record.get("answer", ""))
    clean_acc = bool(
        valid_clean_answer(clean_answer)
        and gold_answer.strip()
        and answers_match(clean_answer, gold_answer)
    )
    result = {
        "index": int(frontier_row.get("index", -1)),
        "id": identifier,
        "clean_answer": clean_answer,
        "gold_answer": gold_answer,
        "clean_acc": clean_acc,
        "methods": {},
        "reader_calls": 0,
    }
    if not clean_acc:
        result["evaluation_status"] = "outside_clean_acc_population"
        return result

    units = units_from_cache_row(record, units_row, k=k)
    by_id = {str(unit["unit_id"]): unit for unit in units}
    eligible = {
        unit_id for unit_id in by_id if pool.is_eligible(unit_id)
    }
    method_sets: dict[str, dict[int, list[str]]] = {
        "reflow": {
            budget: reflow_set(frontier_row, budget, eligible)
            for budget in budgets
        }
    }
    for name, row in score_rows.items():
        ranking = [
            unit_id
            for unit_id in ranked_ids(row)
            if unit_id in eligible
        ]
        method_sets[name] = {
            budget: ranking[:budget] if len(ranking) >= budget else []
            for budget in budgets
        }

    answer_cache: dict[tuple[str, ...], tuple[str, list[dict]]] = {}
    reader_calls = 0
    for method, budget_sets in method_sets.items():
        method_results = {}
        for budget in budgets:
            selected = budget_sets[budget]
            if not selected:
                method_results[str(budget)] = {
                    "status": "no_candidate_under_budget",
                    "selected_ids": [],
                    "n_modified_tokens": 0,
                    "reader_called": False,
                    "acc_flip": False,
                }
                continue
            cache_key = tuple(sorted(selected))
            if cache_key in answer_cache:
                edited_answer, edits = answer_cache[cache_key]
                reader_called = False
            else:
                pool_rows = pool.require(selected)
                replacements = {
                    unit_id: stable_shared_candidate(
                        pool_rows[unit_id]["candidates"],
                        unit_id=unit_id,
                        seed=replacement_seed,
                    )
                    for unit_id in selected
                }
                revision = apply_token_replacements(
                    record,
                    [by_id[unit_id] for unit_id in selected],
                    replacements,
                    k=k,
                )
                if (
                    int(revision["n_failed_edits"]) > 0
                    or int(revision["n_edits"]) != len(selected)
                ):
                    method_results[str(budget)] = {
                        "status": "protocol_violation_failed_edit",
                        "selected_ids": selected,
                        "n_modified_tokens": 0,
                        "reader_called": False,
                        "acc_flip": False,
                        "edits": revision["edits"],
                    }
                    continue
                edited_answer = reader.answer(
                    str(record.get("question", "")),
                    revision["edited_contexts"],
                )
                edits = list(revision["edits"])
                answer_cache[cache_key] = (edited_answer, edits)
                reader_calls += 1
                reader_called = True
            acc_flip = not answers_match(edited_answer, gold_answer)
            method_results[str(budget)] = {
                "status": "acc_flip" if acc_flip else "acc_preserved",
                "selected_ids": selected,
                "selected_tokens": [
                    str(by_id[unit_id].get("text", "")) for unit_id in selected
                ],
                "n_modified_tokens": len(selected),
                "edited_answer": edited_answer,
                "reader_called": reader_called,
                "acc_flip": acc_flip,
                "edits": edits,
            }
        result["methods"][method] = method_results
    result["reader_calls"] = reader_calls
    result["evaluation_status"] = "evaluated"
    return result


def summarize(rows: list[dict], budgets: list[int]) -> dict:
    clean_acc_rows = [row for row in rows if row.get("clean_acc")]
    method_names = sorted({
        name for row in rows for name in row.get("methods", {})
    })
    methods = {}
    for method in method_names:
        budget_results = {}
        for budget in budgets:
            key = str(budget)
            entries = [
                row.get("methods", {}).get(method, {}).get(key, {})
                for row in clean_acc_rows
            ]
            executed = [entry for entry in entries if entry.get("edited_answer")]
            flips = sum(bool(entry.get("acc_flip")) for entry in entries)
            budget_results[key] = {
                "clean_acc_queries": len(clean_acc_rows),
                "executed_queries": len(executed),
                "acc_flip_count": flips,
                "acc_cfr": flips / max(1, len(clean_acc_rows)),
                "mean_modified_tokens": (
                    sum(int(entry.get("n_modified_tokens", 0)) for entry in entries)
                    / max(1, len(clean_acc_rows))
                ),
            }
        methods[method] = budget_results
    return {
        "queries": len(rows),
        "clean_acc_queries": len(clean_acc_rows),
        "reader_calls": sum(int(row.get("reader_calls", 0)) for row in rows),
        "budgets": budgets,
        "methods": methods,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--units-cache", required=True)
    parser.add_argument("--frontier", required=True)
    parser.add_argument("--scores", action="append", default=[])
    parser.add_argument("--shared-pool", required=True)
    parser.add_argument("--expected-pool-sha256", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--budgets", default="1,3,5,7,9")
    parser.add_argument("--replacement-seed", type=int, default=0)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--llm-base-url", default="")
    parser.add_argument("--llm-model", default="")
    args = parser.parse_args()

    budgets = sorted({int(value) for value in args.budgets.split(",")})
    if not budgets or budgets[0] <= 0:
        raise ValueError("budgets must contain positive integers")
    actual_sha = file_sha256(args.shared_pool)
    if actual_sha != args.expected_pool_sha256:
        raise ValueError(
            f"replacement pool fingerprint mismatch: {actual_sha} != "
            f"{args.expected_pool_sha256}"
        )

    records = load_records(args.input)[:args.n]
    units_rows = load_records(args.units_cache)[:args.n]
    frontier_rows = load_records(args.frontier)[:args.n]
    score_tables = parse_score_specs(args.scores, args.n)
    lengths = {
        len(records), len(units_rows), len(frontier_rows),
        *[len(rows) for rows in score_tables.values()],
    }
    if lengths != {len(records)}:
        raise ValueError(f"input lengths are not aligned: {sorted(lengths)}")
    for index, record in enumerate(records):
        identifier = record_id(record)
        aligned = {
            str(units_rows[index].get("id", "")),
            str(frontier_rows[index].get("id", "")),
            *[
                str(rows[index].get("id", ""))
                for rows in score_tables.values()
            ],
        }
        if aligned != {identifier}:
            raise ValueError(f"misaligned row {index}: {identifier} vs {aligned}")

    pool = FrozenSharedReplacementPool(args.shared_pool)
    reader = ReaderClient(
        base_url=args.llm_base_url or None,
        model=args.llm_model or None,
    )
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    expected_ids = {record_id(record) for record in records}
    existing = {}
    if output_path.is_file():
        for row in load_records(output_path):
            identifier = str(row.get("id", ""))
            if identifier in expected_ids:
                existing[identifier] = row
    lock = Lock()

    def run(index: int) -> dict:
        row = evaluate_query(
            records[index],
            units_rows[index],
            frontier_rows[index],
            {name: rows[index] for name, rows in score_tables.items()},
            pool,
            reader,
            budgets=budgets,
            replacement_seed=args.replacement_seed,
            k=args.k,
        )
        with lock:
            with output_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row

    pending = [
        index for index, record in enumerate(records)
        if record_id(record) not in existing
    ]
    completed_rows = list(existing.values())
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run, index) for index in pending]
        for completed, future in enumerate(as_completed(futures), start=1):
            completed_rows.append(future.result())
            if completed % 50 == 0 or completed == len(futures):
                partial = summarize(completed_rows, budgets)
                print(
                    f"[{completed}/{len(futures)} new; "
                    f"{len(completed_rows)}/{len(records)} total] "
                    f"clean_acc={partial['clean_acc_queries']} "
                    f"calls={partial['reader_calls']}",
                    flush=True,
                )

    completed_rows.sort(key=lambda row: int(row.get("index", -1)))
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in completed_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(output_path)
    summary = summarize(completed_rows, budgets)
    summary.update({
        "metric": "Acc-CFR",
        "budget_contract": (
            "at most b edits; ReFlow uses the minimum-residual supported set "
            "with cardinality <= b; ranking baselines use their first b "
            "eligible tokens"
        ),
        "shared_pool": os.path.abspath(args.shared_pool),
        "shared_pool_sha256": actual_sha,
        "replacement_seed": args.replacement_seed,
    })
    Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_out).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
