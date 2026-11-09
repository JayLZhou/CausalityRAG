"""Enumerate ReFlow candidates on attacked contexts without replacement pools."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.contribution_graph import (
    contribution_graph_edges,
    ensure_executable_source_target_path,
)
from causalityrag.flow_contract import (
    breakpoint_price_cuts,
    remaining_contribution_flow,
    require_complete_graph_domain,
)
from causalityrag.io import load_records, record_id
from causalityrag.token_units import units_from_cache_row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--graphs", required=True)
    parser.add_argument("--units", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    records = load_records(args.input)
    graphs = load_records(args.graphs)
    unit_rows = load_records(args.units)
    if len({len(records), len(graphs), len(unit_rows)}) != 1:
        raise ValueError("retrieval, graph, and token-unit rows must align")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(args.out, "w", encoding="utf-8") as output:
        for index, (record, graph, unit_row) in enumerate(
            zip(records, graphs, unit_rows)
        ):
            started = time.monotonic()
            identifier = record_id(record)
            if (
                str(graph.get("id", "")) != identifier
                or str(unit_row.get("id", "")) != identifier
            ):
                raise ValueError(f"misaligned row {index}: {identifier}")
            units = units_from_cache_row(record, unit_row, k=args.k)
            require_complete_graph_domain(graph, units)
            source, interactions, target = contribution_graph_edges(graph)
            source, interactions, target, repair = ensure_executable_source_target_path(
                {str(unit["unit_id"]) for unit in units},
                source,
                interactions,
                target,
            )
            frontier = breakpoint_price_cuts(
                units,
                source,
                interactions,
                target,
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
                selected_ids = [str(value) for value in candidate["selected_ids"]]
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
                        residual / initial_flow if initial_flow > 1e-12 else None
                    ),
                })
            row = {
                "index": index,
                "id": identifier,
                "method": "security_reflow_frontier",
                "attacked_answer": str(graph.get("clean_answer", "")),
                "graph_token_units": len(units),
                "initial_support_flow": initial_flow,
                "frontier_candidates": candidates,
                "frontier_status": str(frontier.get("status", "")),
                "frontier_diagnostics": frontier.get("diagnostics", {}),
                "connectivity_repair": repair,
                "elapsed_seconds": round(time.monotonic() - started, 4),
            }
            rows.append(row)
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(
                f"[security-frontier] {len(rows)}/{len(records)} "
                f"candidates={len(candidates)} repair={repair['applied']}",
                flush=True,
            )

    summary = {
        "queries": len(rows),
        "queries_with_candidates": sum(bool(row["frontier_candidates"]) for row in rows),
        "connectivity_repairs": sum(row["connectivity_repair"]["applied"] for row in rows),
        "mean_candidates": statistics.fmean(
            len(row["frontier_candidates"]) for row in rows
        ) if rows else 0.0,
        "mean_token_units": statistics.fmean(
            row["graph_token_units"] for row in rows
        ) if rows else 0.0,
    }
    with open(args.summary_out, "w", encoding="utf-8") as output:
        json.dump(summary, output, indent=2, ensure_ascii=False)
        output.write("\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
