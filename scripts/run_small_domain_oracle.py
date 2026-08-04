#!/usr/bin/env python3
"""Compare ReFlow with a level-wise exhaustive oracle on frozen micro-domains."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import statistics
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.contribution_graph import contribution_graph_edges
from causalityrag.flow_contract import breakpoint_price_cuts, remaining_contribution_flow
from causalityrag.io import load_records, record_id
from causalityrag.reader import ReaderClient, answers_exact_match
from causalityrag.revision import apply_token_replacements
from causalityrag.shared_replacement_pool import (
    FrozenSharedReplacementPool,
    file_sha256,
    stable_shared_candidate,
)
from causalityrag.token_units import units_from_cache_row


def stable_key(seed: int, query_id: str) -> tuple[bytes, str]:
    digest = hashlib.sha256(f"{seed}\0{query_id}".encode("utf-8")).digest()
    return digest, query_id


def incident_scores(
    unit_ids: set[str],
    source: dict[str, float],
    interactions: dict[tuple[str, str], float],
    target: dict[str, float],
) -> dict[str, float]:
    scores = {unit_id: source.get(unit_id, 0.0) + target.get(unit_id, 0.0) for unit_id in unit_ids}
    for (left, right), weight in interactions.items():
        if left in scores:
            scores[left] += weight
        if right in scores:
            scores[right] += weight
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--units-cache", required=True)
    parser.add_argument("--graphs", required=True)
    parser.add_argument("--shared-pool", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--n-queries", type=int, default=100)
    parser.add_argument("--domain-size", type=int, default=8)
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
        all_units = units_from_cache_row(record, unit_row, k=args.k)
        by_id = {
            str(unit["unit_id"]): unit
            for unit in all_units
            if pool.is_eligible(str(unit["unit_id"]))
        }
        source, interactions, target = contribution_graph_edges(graph_row)
        scores = incident_scores(set(by_id), source, interactions, target)
        domain_ids = [
            unit_id
            for unit_id, score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))
            if score > 0.0
        ][: args.domain_size]
        if len(domain_ids) != args.domain_size:
            continue
        domain_units = [by_id[unit_id] for unit_id in domain_ids]
        initial = remaining_contribution_flow(
            domain_units, source, interactions, target, removed_ids=frozenset()
        )
        if initial <= 1e-12:
            continue
        frontier = breakpoint_price_cuts(domain_units, source, interactions, target)
        if not frontier.get("candidates"):
            continue
        prepared.append((query_id, record, graph_row, domain_units, source, interactions, target, frontier))
    prepared.sort(key=lambda item: stable_key(args.seed, item[0]))
    prepared = prepared[: args.n_queries]
    if len(prepared) != args.n_queries:
        raise ValueError(f"only {len(prepared)} valid oracle micro-instances")

    target_path = Path(args.out)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    completed = {
        str(row["id"]): row
        for row in (load_records(str(target_path)) if target_path.exists() else [])
    }
    reader = ReaderClient(base_url=args.llm_base_url, model=args.llm_model)
    write_lock = threading.Lock()

    def evaluate(item: tuple) -> dict:
        query_id, record, graph_row, domain_units, source, interactions, target, frontier = item
        if query_id in completed:
            return completed[query_id]
        by_id = {str(unit["unit_id"]): unit for unit in domain_units}
        clean = str(graph_row.get("clean_answer", graph_row.get("target_answer", "")))
        answer_cache = {}

        def execute_subset(selected: tuple[str, ...]) -> tuple[str, bool]:
            key = tuple(sorted(selected))
            if key in answer_cache:
                return answer_cache[key]
            pool_rows = pool.require(key)
            replacements = {
                unit_id: stable_shared_candidate(
                    pool_rows[unit_id]["candidates"], unit_id=unit_id, seed=0
                )
                for unit_id in key
            }
            revision = apply_token_replacements(
                record, [by_id[unit_id] for unit_id in key], replacements, k=args.k
            )
            if revision["n_failed_edits"] or revision["n_edits"] != len(key):
                raise ValueError(f"failed frozen edit for {query_id}: {key}")
            answer = reader.answer(str(record.get("question", "")), revision["edited_contexts"])
            result = (answer, not answers_exact_match(clean, answer))
            answer_cache[key] = result
            return result

        oracle_calls = 0
        optimum_sets = []
        for size in range(1, args.domain_size + 1):
            level = list(itertools.combinations(sorted(by_id), size))
            level_results = []
            for subset in level:
                level_results.append((subset, execute_subset(subset)[1]))
            oracle_calls += len(level)
            optimum_sets = [list(subset) for subset, changed in level_results if changed]
            if optimum_sets:
                break
        optimum_size = len(optimum_sets[0]) if optimum_sets else None

        reflow_calls = 0
        reflow_set = None
        reflow_flip = False
        candidates = sorted(
            frontier["candidates"],
            key=lambda candidate: (
                int(candidate.get("n_selected", 0)),
                float(candidate.get("residual_edge_cost", float("inf"))),
                tuple(candidate.get("selected_ids", [])),
            ),
        )
        for candidate in candidates:
            selected = tuple(sorted(str(value) for value in candidate["selected_ids"]))
            reflow_calls += 1
            if execute_subset(selected)[1]:
                reflow_set = list(selected)
                reflow_flip = True
                break
        if reflow_set is None and candidates:
            reflow_set = list(sorted(str(value) for value in candidates[-1]["selected_ids"]))
        reflow_size = len(reflow_set or [])
        row = {
            "id": query_id,
            "domain_ids": sorted(by_id),
            "domain_size": args.domain_size,
            "oracle_feasible": bool(optimum_sets),
            "optimum_size": optimum_size,
            "optimum_sets": optimum_sets,
            "reflow_flip": reflow_flip,
            "reflow_size": reflow_size,
            "reflow_set": reflow_set,
            "exact_match": bool(optimum_sets) and reflow_flip and reflow_size == optimum_size,
            "oracle_reader_calls": oracle_calls,
            "reflow_reader_calls": reflow_calls,
            "actual_unique_reader_calls": len(answer_cache),
            "frontier_candidates": len(candidates),
        }
        with write_lock:
            with target_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row

    rows = list(completed.values())
    pending = [item for item in prepared if item[0] not in completed]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(evaluate, item) for item in pending]
        for index, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if index % 10 == 0:
                print(f"[oracle] {index}/{len(futures)} new queries", flush=True)
    rows = sorted({str(row["id"]): row for row in rows}.values(), key=lambda row: row["id"])
    with target_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    feasible = [row for row in rows if row["oracle_feasible"]]
    comparable = [row for row in feasible if row["reflow_flip"]]
    summary = {
        "schema": "causalityrag.restricted_domain_oracle.v1",
        "queries": len(rows),
        "domain_size": args.domain_size,
        "domain_policy": "top positive incident-capacity eligible tokens; frozen before reader execution",
        "oracle_feasible_queries": len(feasible),
        "reflow_feasible_queries": sum(bool(row["reflow_flip"]) for row in rows),
        "comparable_queries": len(comparable),
        "exact_optimum_match_rate": sum(bool(row["exact_match"]) for row in feasible) / max(1, len(feasible)),
        "mean_optimum_size": statistics.fmean(
            row["optimum_size"] for row in comparable
        ) if comparable else None,
        "mean_reflow_size": statistics.fmean(
            row["reflow_size"] for row in comparable
        ) if comparable else None,
        "mean_additive_gap": statistics.fmean(
            row["reflow_size"] - row["optimum_size"] for row in comparable
        ) if comparable else None,
        "mean_size_ratio": statistics.fmean(
            row["reflow_size"] / row["optimum_size"] for row in comparable
        ) if comparable else None,
        "oracle_reader_calls": sum(row["oracle_reader_calls"] for row in rows),
        "reflow_reader_calls": sum(row["reflow_reader_calls"] for row in rows),
        "actual_unique_reader_calls": sum(row["actual_unique_reader_calls"] for row in rows),
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
