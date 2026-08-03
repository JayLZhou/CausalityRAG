#!/usr/bin/env python3
"""Reconstruct fidelity sampling and verify that a probe file used one seed."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.contribution_graph import contribution_graph_edges
from causalityrag.flow_contract import remaining_contribution_flow
from causalityrag.io import load_records, record_id
from causalityrag.shared_replacement_pool import FrozenSharedReplacementPool
from causalityrag.token_units import units_from_cache_row
from scripts.run_residual_flow_fidelity import stable_int


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--units-cache", required=True)
    parser.add_argument("--graphs", required=True)
    parser.add_argument("--shared-pool", required=True)
    parser.add_argument("--probes", required=True)
    parser.add_argument("--n-queries", type=int, default=100)
    parser.add_argument("--budgets", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=Path)
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
        if query_id != str(unit_row.get("id", "")) or query_id != str(
            graph_row.get("id", "")
        ):
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
        prepared.append(
            (query_id, eligible, source, interactions, target, initial)
        )
    prepared.sort(key=lambda item: (stable_int(args.seed, item[0]), item[0]))
    prepared = prepared[: args.n_queries]
    if len(prepared) != args.n_queries:
        raise ValueError(f"only {len(prepared)} eligible queries")

    actual_rows = load_records(args.probes)
    actual = {
        (str(row["id"]), int(row["budget"]), int(row["trial"])): row
        for row in actual_rows
    }
    expected_count = args.n_queries * len(args.budgets) * args.trials
    if len(actual) != expected_count:
        raise ValueError(f"expected {expected_count} unique probes, got {len(actual)}")

    mismatches = []
    for query_id, units, source, interactions, target, initial in prepared:
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
                    units,
                    source,
                    interactions,
                    target,
                    removed_ids=set(selected),
                ) / initial
                key = (query_id, budget, trial)
                row = actual.get(key)
                if row is None:
                    mismatches.append({"key": key, "reason": "missing"})
                elif tuple(row.get("selected_ids", [])) != selected:
                    mismatches.append({
                        "key": key,
                        "reason": "selected_ids",
                        "expected": selected,
                        "actual": row.get("selected_ids", []),
                    })
                elif not math.isclose(
                    float(row.get("residual_flow_ratio", float("nan"))),
                    residual,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    mismatches.append({
                        "key": key,
                        "reason": "residual_flow_ratio",
                    })
                if len(mismatches) >= 20:
                    break
            if len(mismatches) >= 20:
                break
        if len(mismatches) >= 20:
            break

    result = {
        "schema": "causalityrag.residual_flow_fidelity_seed_audit.v1",
        "seed": args.seed,
        "queries": args.n_queries,
        "expected_probes": expected_count,
        "matched": not mismatches,
        "first_mismatches": mismatches,
    }
    rendered = json.dumps(result, indent=2) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
