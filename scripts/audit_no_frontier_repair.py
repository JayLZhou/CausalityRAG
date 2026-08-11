#!/usr/bin/env python3
"""Audit that an isolated no-frontier repair changed only selected IDs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.io import load_records, record_id
from causalityrag.shared_replacement_pool import file_sha256


def by_id(path: Path) -> tuple[list[str], dict[str, dict]]:
    rows = load_records(path)
    identifiers = [record_id(row) for row in rows]
    if any(not identifier for identifier in identifiers):
        raise ValueError(f"{path}: all rows must have IDs")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"{path}: duplicate IDs")
    return identifiers, dict(zip(identifiers, rows))


def audit_dataset(path: Path) -> dict:
    graph_summary = json.loads(
        (path / "graphs.summary.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (path / "resume/manifest.json").read_text(encoding="utf-8")
    )
    target_ids = [str(value) for value in manifest["target_ids"]]
    target_set = set(target_ids)

    source_graph_path = Path(graph_summary["source_graphs"])
    repaired_graph_path = Path(graph_summary["out"])
    graph_order, source_graphs = by_id(source_graph_path)
    repaired_graph_order, repaired_graphs = by_id(repaired_graph_path)
    if repaired_graph_order != graph_order:
        raise ValueError(f"{path.name}: repaired graph order changed")
    changed_graph_ids = {
        identifier
        for identifier in graph_order
        if source_graphs[identifier] != repaired_graphs[identifier]
    }
    if changed_graph_ids != target_set:
        raise ValueError(
            f"{path.name}: graph changes do not equal target whitelist"
        )

    old_frontier_path = Path(manifest["sources"]["old_frontier"])
    merged_frontier_path = path / "frontier_1000.jsonl"
    frontier_order, old_frontiers = by_id(old_frontier_path)
    merged_frontier_order, merged_frontiers = by_id(merged_frontier_path)
    if merged_frontier_order != frontier_order:
        raise ValueError(f"{path.name}: merged frontier order changed")
    changed_frontier_ids = {
        identifier
        for identifier in frontier_order
        if old_frontiers[identifier] != merged_frontiers[identifier]
    }
    if changed_frontier_ids != target_set:
        raise ValueError(
            f"{path.name}: frontier changes do not equal target whitelist"
        )
    empty_targets = [
        identifier
        for identifier in target_ids
        if not merged_frontiers[identifier].get("frontier_candidates")
    ]
    if empty_targets:
        raise ValueError(f"{path.name}: repaired targets remain empty")

    old_result_path = Path(manifest["sources"]["old_results"])
    result_order, old_results = by_id(old_result_path)
    wrong_status = [
        identifier
        for identifier in target_ids
        if old_results[identifier].get("evaluation_status")
        != manifest["target_status"]
    ]
    if wrong_status:
        raise ValueError(f"{path.name}: target whitelist status mismatch")

    merged_result_path = path / "results_1000.jsonl"
    merged_result_order, merged_results = by_id(merged_result_path)
    if merged_result_order != result_order:
        raise ValueError(f"{path.name}: merged result order changed")
    changed_result_ids = {
        identifier
        for identifier in result_order
        if old_results[identifier] != merged_results[identifier]
    }
    if changed_result_ids != target_set:
        raise ValueError(
            f"{path.name}: result changes do not equal target whitelist"
        )
    no_call_targets = [
        identifier
        for identifier in target_ids
        if int(merged_results[identifier].get("reader_calls", 0)) <= 0
    ]
    if no_call_targets:
        raise ValueError(f"{path.name}: repaired targets remain uncalled")
    remaining_no_frontier = [
        identifier
        for identifier in target_ids
        if merged_results[identifier].get("evaluation_status")
        == manifest["target_status"]
    ]
    if remaining_no_frontier:
        raise ValueError(f"{path.name}: repaired targets remain no-frontier")

    return {
        "dataset": path.name,
        "queries": len(graph_order),
        "target_queries": len(target_ids),
        "edge_repaired_queries": int(graph_summary["repaired_queries"]),
        "already_connected_target_queries": int(
            graph_summary.get("already_connected_target_queries", 0)
        ),
        "changed_graph_queries": len(changed_graph_ids),
        "changed_frontier_queries": len(changed_frontier_ids),
        "changed_result_queries": len(changed_result_ids),
        "empty_target_frontiers": 0,
        "target_reader_calls": sum(
            int(merged_results[identifier].get("reader_calls", 0))
            for identifier in target_ids
        ),
        "target_verified_flips": sum(
            bool(merged_results[identifier].get("verified_flip"))
            for identifier in target_ids
        ),
        "target_protocol_violations": sum(
            str(merged_results[identifier].get("evaluation_status", ""))
            .startswith("protocol_violation")
            for identifier in target_ids
        ),
        "remaining_target_no_frontier": 0,
        "preserved_graph_queries": len(graph_order) - len(changed_graph_ids),
        "preserved_frontier_queries": len(frontier_order)
        - len(changed_frontier_ids),
        "preserved_result_queries": len(result_order) - len(changed_result_ids),
        "repaired_graph_sha256": file_sha256(str(repaired_graph_path)),
        "merged_frontier_sha256": file_sha256(str(merged_frontier_path)),
        "merged_result_sha256": file_sha256(str(merged_result_path)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    datasets = [audit_dataset(args.root / name) for name in args.datasets]
    summary = {
        "schema": "causalityrag.no_frontier_repair_audit.v1",
        "datasets": datasets,
        "target_queries": sum(row["target_queries"] for row in datasets),
        "edge_repaired_queries": sum(
            row["edge_repaired_queries"] for row in datasets
        ),
        "already_connected_target_queries": sum(
            row["already_connected_target_queries"] for row in datasets
        ),
        "changed_graph_queries": sum(
            row["changed_graph_queries"] for row in datasets
        ),
        "changed_frontier_queries": sum(
            row["changed_frontier_queries"] for row in datasets
        ),
        "changed_result_queries": sum(
            row["changed_result_queries"] for row in datasets
        ),
        "empty_target_frontiers": sum(
            row["empty_target_frontiers"] for row in datasets
        ),
        "target_reader_calls": sum(
            row["target_reader_calls"] for row in datasets
        ),
        "target_verified_flips": sum(
            row["target_verified_flips"] for row in datasets
        ),
        "target_protocol_violations": sum(
            row["target_protocol_violations"] for row in datasets
        ),
        "remaining_target_no_frontier": sum(
            row["remaining_target_no_frontier"] for row in datasets
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
