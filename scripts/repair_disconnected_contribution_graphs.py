#!/usr/bin/env python3
"""Repair disconnected contribution graphs on a frozen executable domain."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.contribution_graph import (
    contribution_graph_edges,
    ensure_executable_source_target_path,
    positive_source_target_path_exists,
)
from causalityrag.io import iter_records, record_id
from causalityrag.shared_replacement_pool import file_sha256


DEFAULT_TARGET_STATUS = "no_frontier_candidate"


def load_eligible_ids(path: Path) -> set[str]:
    eligible = set()
    for row in iter_records(path):
        if (
            row.get("row_kind") != "excluded_position"
            and row.get("candidates")
        ):
            eligible.add(str(row.get("unit_id", "")))
    eligible.discard("")
    return eligible


def load_target_ids(
    path: Path,
    *,
    target_status: str,
    expected_rows: int,
) -> list[str]:
    rows = list(iter_records(path, limit=expected_rows))
    if len(rows) != expected_rows:
        raise ValueError(
            f"expected {expected_rows} result rows, got {len(rows)}"
        )
    identifiers = [record_id(row) for row in rows]
    if any(not identifier for identifier in identifiers):
        raise ValueError("every result row must have a nonempty ID")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("result IDs must be unique")
    return [
        identifier
        for identifier, row in zip(identifiers, rows)
        if str(row.get("evaluation_status", "")) == target_status
    ]


def repair_graph_row(row: dict, eligible_ids: set[str]) -> tuple[dict, bool]:
    graph = dict(row.get("contribution_graph", {}))
    domain = {
        str(unit_id)
        for unit_id in graph.get("token_nodes", [])
        if str(unit_id) in eligible_ids
    }
    source, interactions, target = contribution_graph_edges(row)
    repaired_source, repaired_interactions, repaired_target, diagnostics = (
        ensure_executable_source_target_path(
            domain,
            source,
            interactions,
            target,
        )
    )
    if domain and not positive_source_target_path_exists(
        repaired_source,
        repaired_interactions,
        repaired_target,
    ):
        raise RuntimeError(f"{record_id(row)}: graph remains disconnected")
    if not diagnostics["applied"]:
        return dict(row), False

    graph["source_edges"] = [
        *graph.get("source_edges", []),
        *diagnostics["added_source_edges"],
    ]
    graph["interaction_edges"] = [
        *graph.get("interaction_edges", []),
        *diagnostics["added_interaction_edges"],
    ]
    graph["target_edges"] = [
        *graph.get("target_edges", []),
        *diagnostics["added_target_edges"],
    ]
    graph_diagnostics = dict(graph.get("diagnostics", {}))
    graph_diagnostics["connectivity_repair"] = diagnostics
    graph["diagnostics"] = graph_diagnostics
    return {
        **row,
        "status_before_connectivity_repair": str(row.get("status", "")),
        "status": "ok",
        "connectivity_repair": diagnostics,
        "contribution_graph": graph,
    }, True


def repair_selected_graph_row(
    row: dict,
    eligible_ids: set[str],
    target_ids: set[str],
) -> tuple[dict, bool]:
    """Repair one graph only when its old result explicitly selected it."""

    if record_id(row) not in target_ids:
        return row, False
    repaired, changed = repair_graph_row(row, eligible_ids)
    if changed:
        diagnostics = {
            **repaired["connectivity_repair"],
            "targeted": True,
        }
        graph = dict(repaired["contribution_graph"])
        graph_diagnostics = dict(graph.get("diagnostics", {}))
        graph_diagnostics["connectivity_repair"] = diagnostics
        graph["diagnostics"] = graph_diagnostics
        return {
            **repaired,
            "connectivity_repair": diagnostics,
            "contribution_graph": graph,
        }, True
    return {
        **row,
        "connectivity_repair": {
            "applied": False,
            "targeted": True,
            "reason": "already_connected_target_graph",
            "policy": "strongest_endpoint_bottleneck_bridge",
            "added_source_edges": [],
            "added_interaction_edges": [],
            "added_target_edges": [],
        },
    }, False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphs", type=Path, required=True)
    parser.add_argument("--shared-pool", type=Path, required=True)
    parser.add_argument("--target-results", type=Path, required=True)
    parser.add_argument(
        "--target-status",
        default=DEFAULT_TARGET_STATUS,
        help="Only result rows with this evaluation_status may be repaired.",
    )
    parser.add_argument("--expected-pool-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--n", type=int, default=1000)
    args = parser.parse_args()

    actual_pool_sha = file_sha256(str(args.shared_pool))
    if actual_pool_sha != args.expected_pool_sha256:
        raise ValueError(
            "replacement pool fingerprint mismatch: "
            f"expected {args.expected_pool_sha256}, got {actual_pool_sha}"
        )
    eligible_ids = load_eligible_ids(args.shared_pool)
    target_id_order = load_target_ids(
        args.target_results,
        target_status=args.target_status,
        expected_rows=args.n,
    )
    target_ids = set(target_id_order)
    if not target_ids:
        raise ValueError(
            f"no result rows have evaluation_status={args.target_status!r}"
        )
    rows = []
    repaired_ids = []
    already_connected_target_ids = []
    seen_graph_ids = set()
    for row in iter_records(args.graphs, limit=args.n):
        identifier = record_id(row)
        if not identifier:
            raise ValueError("every graph row must have a nonempty ID")
        if identifier in seen_graph_ids:
            raise ValueError(f"duplicate graph ID: {identifier}")
        seen_graph_ids.add(identifier)
        repaired, changed = repair_selected_graph_row(
            row,
            eligible_ids,
            target_ids,
        )
        rows.append(repaired)
        if changed:
            repaired_ids.append(identifier)
        elif identifier in target_ids:
            already_connected_target_ids.append(identifier)
    if len(rows) != args.n:
        raise ValueError(f"expected {args.n} graphs, got {len(rows)}")
    missing_targets = target_ids.difference(seen_graph_ids)
    if missing_targets:
        raise ValueError(
            "target result IDs missing from graph rows: "
            + ", ".join(sorted(missing_targets))
        )
    processed_target_ids = set(repaired_ids).union(
        already_connected_target_ids
    )
    if processed_target_ids != target_ids:
        raise RuntimeError("not every targeted no-frontier graph was processed")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, args.out)
    summary = {
        "schema": "causalityrag.contribution_graph_connectivity_repair.v1",
        "queries": len(rows),
        "target_status": args.target_status,
        "target_queries": len(target_ids),
        "repaired_queries": len(repaired_ids),
        "repaired_ids": [
            identifier
            for identifier in target_id_order
            if identifier in set(repaired_ids)
        ],
        "already_connected_target_queries": len(
            already_connected_target_ids
        ),
        "already_connected_target_ids": [
            identifier
            for identifier in target_id_order
            if identifier in set(already_connected_target_ids)
        ],
        "repair_policy": "strongest_endpoint_bottleneck_bridge",
        "source_graphs": str(args.graphs.resolve()),
        "source_graphs_sha256": file_sha256(str(args.graphs)),
        "source_results": str(args.target_results.resolve()),
        "source_results_sha256": file_sha256(str(args.target_results)),
        "shared_pool": str(args.shared_pool.resolve()),
        "shared_pool_sha256": actual_pool_sha,
        "out": str(args.out.resolve()),
        "out_sha256": file_sha256(str(args.out)),
    }
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
