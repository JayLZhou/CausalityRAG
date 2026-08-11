"""Construct the exact ReFlow frontier on a frozen editable token domain."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.contribution_graph import contribution_graph_edges
from causalityrag.flow_contract import (
    breakpoint_price_cuts,
    remaining_contribution_flow,
    require_complete_graph_domain,
)
from causalityrag.io import load_records, record_id
from causalityrag.shared_replacement_pool import (
    FrozenSharedReplacementPool,
    file_sha256,
)
from causalityrag.token_units import units_from_cache_row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--graphs", required=True)
    parser.add_argument("--units-cache", required=True)
    parser.add_argument("--shared-pool", required=True)
    parser.add_argument("--expected-pool-sha256", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    actual_pool_sha = file_sha256(args.shared_pool)
    if actual_pool_sha != args.expected_pool_sha256:
        raise ValueError(
            "replacement pool fingerprint mismatch: "
            f"expected {args.expected_pool_sha256}, got {actual_pool_sha}"
        )
    pool = FrozenSharedReplacementPool(args.shared_pool)
    records = load_records(args.input)[args.start : args.start + args.n]
    graphs = load_records(args.graphs)[args.start : args.start + len(records)]
    unit_rows = load_records(args.units_cache)[
        args.start : args.start + len(records)
    ]
    if len({len(records), len(graphs), len(unit_rows)}) != 1:
        raise ValueError("input, graph, and token-unit rows must be aligned")

    rows = []
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as output:
        for offset, (record, graph, unit_row) in enumerate(
            zip(records, graphs, unit_rows)
        ):
            started = time.monotonic()
            index = args.start + offset
            identifier = record_id(record)
            if (
                str(graph.get("id", "")) != identifier
                or str(unit_row.get("id", "")) != identifier
            ):
                raise ValueError(f"misaligned row {index}: {identifier}")
            all_units = units_from_cache_row(record, unit_row, k=args.k)
            require_complete_graph_domain(graph, all_units)
            units = [
                unit
                for unit in all_units
                if pool.is_eligible(str(unit["unit_id"]))
            ]
            source, interactions, target = contribution_graph_edges(graph)
            graph_repair = graph.get(
                "connectivity_repair",
                graph.get("contribution_graph", {})
                .get("diagnostics", {})
                .get("connectivity_repair", {
                    "applied": False,
                    "reason": "not_targeted_for_repair",
                }),
            )
            frontier = breakpoint_price_cuts(
                units,
                source,
                interactions,
                target,
            )
            if units and not frontier.get("candidates"):
                raise RuntimeError(
                    f"{identifier}: nonempty executable domain produced an "
                    "empty frontier"
                )
            initial_flow = remaining_contribution_flow(
                units,
                source,
                interactions,
                target,
                removed_ids=frozenset(),
            )
            candidates = []
            for candidate in frontier.get("candidates", []):
                selected_ids = [
                    str(unit_id)
                    for unit_id in candidate.get("selected_ids", [])
                ]
                residual = remaining_contribution_flow(
                    units,
                    source,
                    interactions,
                    target,
                    removed_ids=set(selected_ids),
                )
                candidates.append({
                    **candidate,
                    "selected_ids": selected_ids,
                    "remaining_support_flow": residual,
                    "remaining_support_fraction": (
                        residual / initial_flow
                        if initial_flow > 1e-12
                        else None
                    ),
                })
            row = {
                "index": index,
                "id": identifier,
                "method": "reflow_frontier",
                "clean_answer": str(
                    graph.get("clean_answer", graph.get("target_answer", ""))
                ),
                "capacity_normalization": "mean-positive-edge",
                "graph_token_units": len(all_units),
                "editable_token_units": len(units),
                "shared_pool_sha256": actual_pool_sha,
                "initial_support_flow": initial_flow,
                "n_frontier": len(candidates),
                "frontier_candidates": candidates,
                "frontier_diagnostics": {
                    **frontier.get("diagnostics", {}),
                    "graph_connectivity_repair": graph_repair,
                },
                "graph_connectivity_repair": graph_repair,
                "frontier_status": frontier.get("status", ""),
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
            rows.append(row)
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            print(
                f"[reflow] {len(rows)}/{len(records)} "
                f"frontier={len(candidates)} status={row['frontier_status']}",
                flush=True,
            )

    summary = {
        "queries": len(rows),
        "queries_with_frontier": sum(
            bool(row["frontier_candidates"]) for row in rows
        ),
        "mean_frontier_size": statistics.fmean(
            row["n_frontier"] for row in rows
        )
        if rows
        else 0.0,
        "mean_editable_tokens": statistics.fmean(
            row["editable_token_units"] for row in rows
        )
        if rows
        else 0.0,
        "capacity_normalization": "mean-positive-edge",
        "shared_pool": os.path.abspath(args.shared_pool),
        "shared_pool_sha256": actual_pool_sha,
        "out": os.path.abspath(args.out),
    }
    with open(args.summary_out, "w", encoding="utf-8") as output:
        json.dump(summary, output, indent=2, ensure_ascii=False)
        output.write("\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
