#!/usr/bin/env python3
"""Audit aligned k=5 inputs for the fixed-budget residual-flow experiment."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.contribution_graph import contribution_graph_edges
from causalityrag.flow_contract import remaining_contribution_flow
from causalityrag.io import load_records, record_id
from causalityrag.shared_replacement_pool import FrozenSharedReplacementPool
from causalityrag.token_units import units_from_cache_row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--budget", type=int, default=20)
    parser.add_argument("--minimum-queries", type=int, default=100)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    results = []
    for dataset in args.datasets:
        root = Path(args.root) / dataset
        records = load_records(str(root / "inputs/retrieval.jsonl"))
        unit_rows = load_records(str(root / "inputs/units.jsonl"))
        graph_rows = load_records(str(root / "graph/contribution_graph.jsonl"))
        ids = [record_id(row) for row in records]
        unit_ids = [str(row.get("id", "")) for row in unit_rows]
        graph_ids = [str(row.get("id", "")) for row in graph_rows]
        if len({len(records), len(unit_rows), len(graph_rows)}) != 1:
            raise ValueError(f"{dataset}: row-count mismatch")
        if len(ids) != len(set(ids)):
            raise ValueError(f"{dataset}: duplicate retrieval IDs")
        if ids != unit_ids or ids != graph_ids:
            raise ValueError(f"{dataset}: ordered ID alignment failure")

        pool = FrozenSharedReplacementPool(str(root / "inputs/shared_pool.jsonl"))
        graph_ok = eligible_queries = 0
        for record, unit_row, graph_row in zip(records, unit_rows, graph_rows):
            if graph_row.get("status") != "ok":
                continue
            graph_ok += 1
            units = units_from_cache_row(record, unit_row, k=5)
            eligible = [
                unit for unit in units if pool.is_eligible(str(unit["unit_id"]))
            ]
            if len(eligible) < args.budget:
                continue
            source, interactions, target = contribution_graph_edges(graph_row)
            initial = remaining_contribution_flow(
                eligible,
                source,
                interactions,
                target,
                removed_ids=frozenset(),
            )
            if initial > 1e-12:
                eligible_queries += 1
        if eligible_queries < args.minimum_queries:
            raise ValueError(
                f"{dataset}: only {eligible_queries} eligible queries for "
                f"budget {args.budget}"
            )
        result = {
            "dataset": dataset,
            "rows": len(records),
            "unique_ids": len(set(ids)),
            "ordered_ids_aligned": True,
            "graph_ok": graph_ok,
            "eligible_queries_b20": eligible_queries,
        }
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)

    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {
                "schema": "causalityrag.residual_flow_b20_input_audit.v1",
                "budget": args.budget,
                "minimum_queries": args.minimum_queries,
                "datasets": results,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

