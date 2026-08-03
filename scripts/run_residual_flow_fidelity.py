#!/usr/bin/env python3
"""Run graph-independent, equal-budget residual-flow fidelity probes."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import random
import statistics
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.contribution_graph import contribution_graph_edges
from causalityrag.flow_contract import remaining_contribution_flow
from causalityrag.io import load_records, record_id
from causalityrag.reader import ReaderClient, answer_token_f1, answers_exact_match
from causalityrag.revision import apply_token_replacements
from causalityrag.shared_replacement_pool import (
    FrozenSharedReplacementPool,
    file_sha256,
    stable_shared_candidate,
)
from causalityrag.token_units import units_from_cache_row


def stable_int(*values: object) -> int:
    payload = "\0".join(str(value) for value in values).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and values[order[end]] == values[order[position]]:
            end += 1
        rank = (position + 1 + end) / 2.0
        for index in order[position:end]:
            ranks[index] = rank
        position = end
    return ranks


def pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2:
        return None
    mean_left = statistics.fmean(left)
    mean_right = statistics.fmean(right)
    numerator = sum(
        (a - mean_left) * (b - mean_right) for a, b in zip(left, right)
    )
    denominator = math.sqrt(
        sum((value - mean_left) ** 2 for value in left)
        * sum((value - mean_right) ** 2 for value in right)
    )
    return numerator / denominator if denominator > 0 else None


def auc(labels: list[bool], scores: list[float]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    ranks = average_ranks(scores)
    rank_sum = sum(rank for rank, label in zip(ranks, labels) if label)
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def wilson(successes: int, total: int) -> list[float]:
    if not total:
        return [0.0, 0.0]
    z = 1.959963984540054
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / total + z * z / (4 * total**2)) / denominator
    return [center - radius, center + radius]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--units-cache", required=True)
    parser.add_argument("--graphs", required=True)
    parser.add_argument("--shared-pool", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--n-queries", type=int, default=100)
    parser.add_argument("--budgets", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--llm-base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--llm-model", default="qwen2.5-7b")
    args = parser.parse_args()

    records = load_records(args.input)
    unit_rows = load_records(args.units_cache)
    graph_rows = load_records(args.graphs)
    if len({len(records), len(unit_rows), len(graph_rows)}) != 1:
        raise ValueError("input, units, and graph rows must be aligned")
    pool = FrozenSharedReplacementPool(args.shared_pool)
    prepared = []
    for record, unit_row, graph_row in zip(records, unit_rows, graph_rows):
        query_id = record_id(record)
        if query_id != str(unit_row.get("id", "")) or query_id != str(graph_row.get("id", "")):
            raise ValueError(f"misaligned query {query_id}")
        if graph_row.get("status") != "ok":
            continue
        units = units_from_cache_row(record, unit_row, k=args.k)
        eligible = [unit for unit in units if pool.is_eligible(str(unit["unit_id"]))]
        if len(eligible) < max(args.budgets):
            continue
        source, interactions, target = contribution_graph_edges(graph_row)
        initial = remaining_contribution_flow(
            eligible, source, interactions, target, removed_ids=frozenset()
        )
        if initial <= 1e-12:
            continue
        prepared.append((query_id, record, graph_row, eligible, source, interactions, target, initial))
    prepared.sort(key=lambda item: (stable_int(args.seed, item[0]), item[0]))
    prepared = prepared[: args.n_queries]
    if len(prepared) != args.n_queries:
        raise ValueError(f"only {len(prepared)} eligible queries")

    tasks = []
    for query_id, record, graph_row, units, source, interactions, target, initial in prepared:
        unit_ids = [str(unit["unit_id"]) for unit in units]
        for budget in args.budgets:
            seen = set()
            for trial in range(args.trials):
                salt = 0
                while True:
                    generator = random.Random(
                        stable_int(args.seed, query_id, budget, trial, salt)
                    )
                    selected = tuple(sorted(generator.sample(unit_ids, budget)))
                    if selected not in seen:
                        break
                    salt += 1
                seen.add(selected)
                residual = remaining_contribution_flow(
                    units, source, interactions, target, removed_ids=set(selected)
                ) / initial
                tasks.append((query_id, record, graph_row, units, budget, trial, selected, residual))

    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    completed = {}
    if target.exists():
        for row in load_records(str(target)):
            completed[str(row["task_id"])] = row
    reader = ReaderClient(base_url=args.llm_base_url, model=args.llm_model)
    lock = threading.Lock()

    def execute(task: tuple) -> dict:
        query_id, record, graph_row, units, budget, trial, selected, residual = task
        task_id = f"{query_id}|{budget}|{trial}"
        if task_id in completed:
            return completed[task_id]
        by_id = {str(unit["unit_id"]): unit for unit in units}
        pool_rows = pool.require(selected)
        replacements = {
            unit_id: stable_shared_candidate(
                pool_rows[unit_id]["candidates"], unit_id=unit_id, seed=0
            )
            for unit_id in selected
        }
        revision = apply_token_replacements(
            record, [by_id[unit_id] for unit_id in selected], replacements, k=args.k
        )
        if revision["n_failed_edits"] or revision["n_edits"] != budget:
            raise ValueError(f"failed frozen edit for {task_id}")
        clean = str(graph_row.get("clean_answer", graph_row.get("target_answer", "")))
        edited = reader.answer(str(record.get("question", "")), revision["edited_contexts"])
        row = {
            "task_id": task_id,
            "id": query_id,
            "budget": budget,
            "trial": trial,
            "selected_ids": list(selected),
            "residual_flow_ratio": residual,
            "clean_answer": clean,
            "edited_answer": edited,
            "answer_flip": not answers_exact_match(clean, edited),
            "answer_distance": 1.0 - answer_token_f1(clean, edited),
        }
        with lock:
            with target.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row

    rows = list(completed.values())
    pending = [task for task in tasks if f"{task[0]}|{task[4]}|{task[5]}" not in completed]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(execute, task) for task in pending]
        for index, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if index % 250 == 0:
                print(f"[fidelity] {index}/{len(futures)} new probes", flush=True)
    rows = [completed.get(str(row.get("task_id")), row) for row in rows]
    by_task = {str(row["task_id"]): row for row in rows}
    rows = sorted(by_task.values(), key=lambda row: (row["id"], row["budget"], row["trial"]))
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    groups = {}
    for row in rows:
        groups.setdefault((row["id"], row["budget"]), []).append(row)
    deciles = {index: [] for index in range(1, args.trials + 1)}
    correlations = []
    concordant = comparable = 0
    for group in groups.values():
        ordered = sorted(group, key=lambda row: (row["residual_flow_ratio"], row["trial"]))
        for index, row in enumerate(ordered, start=1):
            deciles[index].append(row)
        correlation = pearson(
            average_ranks([row["residual_flow_ratio"] for row in group]),
            average_ranks([row["answer_distance"] for row in group]),
        )
        if correlation is not None:
            correlations.append(correlation)
        for left, right in itertools.combinations(group, 2):
            flow_delta = left["residual_flow_ratio"] - right["residual_flow_ratio"]
            distance_delta = left["answer_distance"] - right["answer_distance"]
            if abs(flow_delta) <= 1e-12 or abs(distance_delta) <= 1e-12:
                continue
            comparable += 1
            concordant += int(flow_delta * distance_delta < 0)
    decile_rows = []
    for index, values in deciles.items():
        flips = sum(bool(row["answer_flip"]) for row in values)
        decile_rows.append({
            "decile": index,
            "count": len(values),
            "mean_residual_flow_ratio": statistics.fmean(
                row["residual_flow_ratio"] for row in values
            ),
            "answer_flip_rate": flips / len(values),
            "wilson_95": wilson(flips, len(values)),
            "mean_answer_distance": statistics.fmean(
                row["answer_distance"] for row in values
            ),
        })
    summary = {
        "schema": "causalityrag.residual_flow_fidelity.v1",
        "queries": len(prepared),
        "budgets": args.budgets,
        "trials_per_query_budget": args.trials,
        "reader_executions": len(rows),
        "selection": "uniform legal token sets sampled before residual flow is computed",
        "deciles_low_to_high_residual": decile_rows,
        "mean_within_query_budget_spearman_residual_vs_distance": statistics.fmean(correlations),
        "auroc_removed_flow_predicting_answer_flip": auc(
            [bool(row["answer_flip"]) for row in rows],
            [1.0 - float(row["residual_flow_ratio"]) for row in rows],
        ),
        "equal_budget_pairwise_concordance": concordant / max(1, comparable),
        "comparable_pairs": comparable,
        "source_sha256": {
            "input": file_sha256(args.input),
            "units": file_sha256(args.units_cache),
            "graphs": file_sha256(args.graphs),
            "shared_pool": file_sha256(args.shared_pool),
        },
    }
    Path(args.summary_out).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
