#!/usr/bin/env python3
"""Prepare and merge a resume limited to old no-frontier result rows."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.io import load_records, record_id
from causalityrag.shared_replacement_pool import file_sha256
from scripts.evaluate_reflow import summarize


TARGET_STATUS = "no_frontier_candidate"


def require_rows(path: Path, expected_rows: int) -> list[dict]:
    rows = load_records(path)
    if len(rows) != expected_rows:
        raise ValueError(f"{path}: expected {expected_rows} rows, got {len(rows)}")
    identifiers = [record_id(row) for row in rows]
    if any(not identifier for identifier in identifiers):
        raise ValueError(f"{path}: every row must have a nonempty ID")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"{path}: IDs must be unique")
    return rows


def atomic_write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(value, output, indent=2, ensure_ascii=False)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def indexed_rows(rows: list[dict]) -> dict[str, dict]:
    return {record_id(row): row for row in rows}


def command_prepare(args: argparse.Namespace) -> None:
    inputs = require_rows(args.input, args.n)
    units = require_rows(args.units_cache, args.n)
    graphs = require_rows(args.repaired_graphs, args.n)
    frontiers = require_rows(args.old_frontier, args.n)
    results = require_rows(args.old_results, args.n)
    expected_ids = [record_id(row) for row in inputs]
    for path, rows in (
        (args.units_cache, units),
        (args.repaired_graphs, graphs),
        (args.old_frontier, frontiers),
        (args.old_results, results),
    ):
        actual_ids = [record_id(row) for row in rows]
        if actual_ids != expected_ids:
            raise ValueError(f"{path}: row IDs are not aligned with input")

    target_ids = [
        record_id(row)
        for row in results
        if str(row.get("evaluation_status", "")) == args.target_status
    ]
    if not target_ids:
        raise ValueError(f"no result rows have status {args.target_status!r}")
    target_set = set(target_ids)
    graph_by_id = indexed_rows(graphs)
    for identifier in target_ids:
        repair = graph_by_id[identifier].get("connectivity_repair", {})
        if not repair.get("targeted"):
            raise ValueError(f"{identifier}: graph lacks target-selection evidence")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    selected_paths = {
        "input": args.out_dir / "input.jsonl",
        "units_cache": args.out_dir / "units.jsonl",
        "graphs": args.out_dir / "graphs.jsonl",
    }
    for name, rows in (
        ("input", inputs),
        ("units_cache", units),
        ("graphs", graphs),
    ):
        atomic_write_jsonl(
            selected_paths[name],
            [row for row in rows if record_id(row) in target_set],
        )

    index_by_id = {
        record_id(row): int(row.get("index", offset))
        for offset, row in enumerate(results)
        if record_id(row) in target_set
    }
    manifest = {
        "schema": "causalityrag.no_frontier_resume.v1",
        "target_status": args.target_status,
        "source_queries": args.n,
        "target_queries": len(target_ids),
        "target_ids": target_ids,
        "original_index_by_id": index_by_id,
        "sources": {
            "input": str(args.input.resolve()),
            "input_sha256": file_sha256(str(args.input)),
            "units_cache": str(args.units_cache.resolve()),
            "units_cache_sha256": file_sha256(str(args.units_cache)),
            "repaired_graphs": str(args.repaired_graphs.resolve()),
            "repaired_graphs_sha256": file_sha256(str(args.repaired_graphs)),
            "old_frontier": str(args.old_frontier.resolve()),
            "old_frontier_sha256": file_sha256(str(args.old_frontier)),
            "old_results": str(args.old_results.resolve()),
            "old_results_sha256": file_sha256(str(args.old_results)),
        },
        "selected": {
            name: str(path.resolve())
            for name, path in selected_paths.items()
        },
    }
    for name, path in selected_paths.items():
        manifest["selected"][f"{name}_sha256"] = file_sha256(str(path))
    manifest_path = args.out_dir / "manifest.json"
    atomic_write_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "causalityrag.no_frontier_resume.v1":
        raise ValueError(f"unsupported resume manifest: {path}")
    target_ids = [str(value) for value in manifest.get("target_ids", [])]
    if not target_ids or len(set(target_ids)) != len(target_ids):
        raise ValueError("resume manifest target IDs must be nonempty and unique")
    return manifest


def merge_rows(
    old_rows: list[dict],
    resumed_rows: list[dict],
    manifest: dict,
) -> list[dict]:
    target_ids = [str(value) for value in manifest["target_ids"]]
    resumed_ids = [record_id(row) for row in resumed_rows]
    if resumed_ids != target_ids:
        raise ValueError("resumed row IDs do not exactly match the target manifest")
    replacements = indexed_rows(resumed_rows)
    original_indexes = {
        str(identifier): int(index)
        for identifier, index in manifest["original_index_by_id"].items()
    }
    merged = []
    for row in old_rows:
        identifier = record_id(row)
        if identifier not in replacements:
            merged.append(row)
            continue
        merged.append({
            **replacements[identifier],
            "index": original_indexes[identifier],
        })
    return merged


def command_merge_frontier(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.manifest)
    expected_rows = int(manifest["source_queries"])
    old_rows = require_rows(args.old_frontier, expected_rows)
    resumed_rows = require_rows(
        args.resumed_frontier,
        int(manifest["target_queries"]),
    )
    for row in resumed_rows:
        if not row.get("frontier_candidates"):
            raise ValueError(f"{record_id(row)}: repaired frontier is still empty")
        if not row.get("graph_connectivity_repair", {}).get("targeted"):
            raise ValueError(f"{record_id(row)}: missing graph target evidence")
    merged = merge_rows(old_rows, resumed_rows, manifest)
    atomic_write_jsonl(args.out, merged)
    summary = {
        "schema": "causalityrag.no_frontier_frontier_merge.v1",
        "queries": len(merged),
        "replaced_queries": len(resumed_rows),
        "preserved_queries": len(merged) - len(resumed_rows),
        "remaining_empty_frontiers": sum(
            not row.get("frontier_candidates") for row in merged
        ),
        "out": str(args.out.resolve()),
        "out_sha256": file_sha256(str(args.out)),
    }
    atomic_write_json(args.summary_out, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def command_merge_results(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.manifest)
    expected_rows = int(manifest["source_queries"])
    old_rows = require_rows(args.old_results, expected_rows)
    resumed_rows = require_rows(
        args.resumed_results,
        int(manifest["target_queries"]),
    )
    for row in resumed_rows:
        if int(row.get("reader_calls", 0)) <= 0:
            raise ValueError(f"{record_id(row)}: repaired result made no reader call")
        if row.get("evaluation_status") == manifest["target_status"]:
            raise ValueError(f"{record_id(row)}: result remains no-frontier")
    merged = merge_rows(old_rows, resumed_rows, manifest)
    atomic_write_jsonl(args.out, merged)
    summary = {
        **summarize(merged),
        "schema": "causalityrag.no_frontier_result_merge.v1",
        "replaced_queries": len(resumed_rows),
        "preserved_queries": len(merged) - len(resumed_rows),
        "remaining_no_frontier": sum(
            row.get("evaluation_status") == manifest["target_status"]
            for row in merged
        ),
        "out": str(args.out.resolve()),
        "out_sha256": file_sha256(str(args.out)),
    }
    atomic_write_json(args.summary_out, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--input", type=Path, required=True)
    prepare.add_argument("--units-cache", type=Path, required=True)
    prepare.add_argument("--repaired-graphs", type=Path, required=True)
    prepare.add_argument("--old-frontier", type=Path, required=True)
    prepare.add_argument("--old-results", type=Path, required=True)
    prepare.add_argument("--out-dir", type=Path, required=True)
    prepare.add_argument("--n", type=int, default=1000)
    prepare.add_argument("--target-status", default=TARGET_STATUS)
    prepare.set_defaults(handler=command_prepare)

    merge_frontier = subparsers.add_parser("merge-frontier")
    merge_frontier.add_argument("--manifest", type=Path, required=True)
    merge_frontier.add_argument("--old-frontier", type=Path, required=True)
    merge_frontier.add_argument("--resumed-frontier", type=Path, required=True)
    merge_frontier.add_argument("--out", type=Path, required=True)
    merge_frontier.add_argument("--summary-out", type=Path, required=True)
    merge_frontier.set_defaults(handler=command_merge_frontier)

    merge_results = subparsers.add_parser("merge-results")
    merge_results.add_argument("--manifest", type=Path, required=True)
    merge_results.add_argument("--old-results", type=Path, required=True)
    merge_results.add_argument("--resumed-results", type=Path, required=True)
    merge_results.add_argument("--out", type=Path, required=True)
    merge_results.add_argument("--summary-out", type=Path, required=True)
    merge_results.set_defaults(handler=command_merge_results)

    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
