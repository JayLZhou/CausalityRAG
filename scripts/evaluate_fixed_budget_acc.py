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

from causalityrag.io import gold_answers, load_records, record_id
from causalityrag.reader import (
    ReaderClient,
    ReaderProtocolError,
    canonicalize_reader_answer,
    reader_answers_match,
)
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


def answer_is_correct(
    question: str,
    answer: str,
    accepted_answers: list[str],
    *,
    reader_mode: str,
) -> bool:
    return any(
        reader_answers_match(
            question,
            answer,
            accepted,
            reader_mode=reader_mode,
        )
        for accepted in accepted_answers
    )


def rescore_seeded_entry(
    entry: dict,
    *,
    question: str,
    accepted_gold_answers: list[str],
    reader_mode: str,
) -> dict:
    rescored = dict(entry)
    edited_answer_raw = str(entry.get("edited_answer", ""))
    try:
        edited_answer = canonicalize_reader_answer(
            question,
            edited_answer_raw,
            reader_mode=reader_mode,
        )
    except ReaderProtocolError as exc:
        return {
            **rescored,
            "edited_answer": edited_answer_raw,
            "status": "protocol_violation_invalid_reader_answer",
            "acc_flip": False,
            "protocol_error": str(exc),
            "rescored_from_seed": True,
        }
    if not valid_clean_answer(edited_answer):
        return {
            **rescored,
            "edited_answer": edited_answer,
            "status": "protocol_violation_invalid_reader_answer",
            "acc_flip": False,
            "protocol_error": "invalid empty or non-answer reader output",
            "rescored_from_seed": True,
        }
    acc_flip = not answer_is_correct(
        question,
        edited_answer,
        accepted_gold_answers,
        reader_mode=reader_mode,
    )
    return {
        **rescored,
        "edited_answer": edited_answer,
        "status": "acc_flip" if acc_flip else "acc_preserved",
        "acc_flip": acc_flip,
        "rescored_from_seed": True,
    }


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
    include_reflow: bool = True,
    seed_row: dict | None = None,
) -> dict:
    identifier = record_id(record)
    question = str(record.get("question", ""))
    reader_mode = getattr(reader, "reader_mode", "short_answer")
    clean_answer_raw = str(frontier_row.get("clean_answer", ""))
    accepted_gold_answers_raw = gold_answers(record)
    try:
        clean_answer = canonicalize_reader_answer(
            question,
            clean_answer_raw,
            reader_mode=reader_mode,
        )
        accepted_gold_answers = [
            canonicalize_reader_answer(
                question,
                answer,
                reader_mode=reader_mode,
            )
            for answer in accepted_gold_answers_raw
        ]
    except ReaderProtocolError as exc:
        return {
            "index": int(frontier_row.get("index", -1)),
            "id": identifier,
            "clean_answer": clean_answer_raw,
            "gold_answer": str(record.get("answer", "")),
            "gold_answers": accepted_gold_answers_raw,
            "clean_acc": False,
            "methods": {},
            "reader_calls": 0,
            "evaluation_status": (
                "protocol_violation_invalid_clean_or_gold_answer"
            ),
            "protocol_error": str(exc),
        }
    clean_acc = bool(
        valid_clean_answer(clean_answer)
        and accepted_gold_answers
        and answer_is_correct(
            question,
            clean_answer,
            accepted_gold_answers,
            reader_mode=reader_mode,
        )
    )
    result = {
        "index": int(frontier_row.get("index", -1)),
        "id": identifier,
        "clean_answer": clean_answer,
        "clean_answer_raw": clean_answer_raw,
        "gold_answer": str(record.get("answer", "")),
        "gold_answers": accepted_gold_answers,
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
    method_sets: dict[str, dict[int, list[str]]] = {}
    if include_reflow:
        method_sets["reflow"] = {
            budget: reflow_set(frontier_row, budget, eligible)
            for budget in budgets
        }
    for name, row in score_rows.items():
        ranking = [
            unit_id
            for unit_id in ranked_ids(row)
            if unit_id in eligible
        ]
        method_sets[name] = {
            budget: ranking[:budget]
            for budget in budgets
        }

    answer_cache: dict[tuple[str, ...], dict] = {}
    reader_calls = 0
    for method, budget_sets in method_sets.items():
        method_results = {}
        for budget in budgets:
            seeded = (
                seed_row or {}
            ).get("methods", {}).get(method, {}).get(str(budget))
            if seeded is not None and seeded.get("edited_answer"):
                method_results[str(budget)] = rescore_seeded_entry(
                    seeded,
                    question=question,
                    accepted_gold_answers=accepted_gold_answers,
                    reader_mode=reader_mode,
                )
                continue
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
                cached = answer_cache[cache_key]
                edited_answer = str(cached.get("edited_answer", ""))
                edits = list(cached.get("edits", []))
                protocol_error = str(cached.get("protocol_error", ""))
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
                edits = list(revision["edits"])
                reader_calls += 1
                reader_called = True
                try:
                    edited_answer = reader.answer(
                        question,
                        revision["edited_contexts"],
                    )
                    protocol_error = (
                        "invalid empty or non-answer reader output"
                        if not valid_clean_answer(edited_answer)
                        else ""
                    )
                except ReaderProtocolError as exc:
                    edited_answer = ""
                    protocol_error = str(exc)
                answer_cache[cache_key] = {
                    "edited_answer": edited_answer,
                    "edits": edits,
                    "protocol_error": protocol_error,
                }
            if protocol_error:
                method_results[str(budget)] = {
                    "status": "protocol_violation_invalid_reader_answer",
                    "selected_ids": selected,
                    "n_modified_tokens": len(selected),
                    "edited_answer": edited_answer,
                    "reader_called": reader_called,
                    "acc_flip": False,
                    "edits": edits,
                    "protocol_error": protocol_error,
                }
                continue
            acc_flip = not answer_is_correct(
                question,
                edited_answer,
                accepted_gold_answers,
                reader_mode=reader_mode,
            )
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
            invalid = [
                entry
                for entry in entries
                if str(entry.get("status", "")).startswith(
                    "protocol_violation"
                )
            ]
            valid = [entry for entry in entries if entry not in invalid]
            flips = sum(bool(entry.get("acc_flip")) for entry in entries)
            budget_results[key] = {
                "clean_acc_queries": len(clean_acc_rows),
                "executed_queries": len(executed),
                "valid_answer_queries": len(valid),
                "protocol_violations": len(invalid),
                "acc_flip_count": flips,
                "acc_cfr": flips / max(1, len(valid)),
                "mean_modified_tokens": (
                    sum(int(entry.get("n_modified_tokens", 0)) for entry in valid)
                    / max(1, len(valid))
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
    parser.add_argument("--seed-results", default="")
    parser.add_argument("--skip-reflow", action="store_true")
    parser.add_argument("--replacement-seed", type=int, default=0)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--llm-base-url", default="")
    parser.add_argument("--llm-model", default="")
    parser.add_argument("--reader-max-tokens", type=int, default=128)
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
        max_tokens=args.reader_max_tokens,
    )
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    expected_ids = {record_id(record) for record in records}
    seed_by_id = {}
    if args.seed_results:
        for row in load_records(args.seed_results):
            identifier = str(row.get("id", ""))
            if identifier in expected_ids:
                seed_by_id[identifier] = row
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
            include_reflow=not args.skip_reflow,
            seed_row=seed_by_id.get(record_id(records[index])),
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
            "with cardinality <= b; ranking baselines use up to their first b "
            "eligible tokens and use all available ranked tokens when fewer "
            "than b exist"
        ),
        "shared_pool": os.path.abspath(args.shared_pool),
        "shared_pool_sha256": actual_sha,
        "replacement_seed": args.replacement_seed,
        "seed_results": os.path.abspath(args.seed_results) if args.seed_results else "",
        "includes_reflow": not args.skip_reflow,
    })
    Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_out).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
