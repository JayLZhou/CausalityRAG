#!/usr/bin/env python3
"""Prepare, merge, and audit Table 3 rows for an explicit query whitelist."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.io import load_records, record_id
from causalityrag.shared_replacement_pool import file_sha256
from scripts.evaluate_matched_budget_baselines import summarize as summarize_baselines
from scripts.evaluate_paraphrase_controls import summarize as summarize_controls


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


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "causalityrag.no_frontier_resume.v1":
        raise ValueError(f"unsupported target manifest: {path}")
    target_ids = [str(value) for value in manifest.get("target_ids", [])]
    if not target_ids or len(set(target_ids)) != len(target_ids):
        raise ValueError("target IDs must be nonempty and unique")
    return manifest


def aligned_tables(
    input_path: Path,
    named_paths: dict[str, Path],
    expected_rows: int,
) -> tuple[list[str], dict[str, list[dict]]]:
    tables = {"input": require_rows(input_path, expected_rows)}
    tables.update({
        name: require_rows(path, expected_rows)
        for name, path in named_paths.items()
    })
    expected_ids = [record_id(row) for row in tables["input"]]
    for name, rows in tables.items():
        if [record_id(row) for row in rows] != expected_ids:
            raise ValueError(f"{name}: row IDs are not aligned with input")
    return expected_ids, tables


def selected_rows(rows: list[dict], target_ids: list[str]) -> list[dict]:
    by_id = {record_id(row): row for row in rows}
    missing = [identifier for identifier in target_ids if identifier not in by_id]
    if missing:
        raise ValueError(f"target IDs missing from table: {missing[:5]}")
    return [by_id[identifier] for identifier in target_ids]


def parse_named_paths(specs: list[str]) -> dict[str, Path]:
    result = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"invalid NAME=PATH value: {spec}")
        name, value = spec.split("=", 1)
        name = name.strip()
        if not name or name in result:
            raise ValueError(f"duplicate or empty table name: {name!r}")
        result[name] = Path(value)
    return result


def validate_reflow_budgets(rows: list[dict], target_ids: list[str]) -> None:
    selected = selected_rows(rows, target_ids)
    for row in selected:
        identifier = record_id(row)
        budget = int(row.get("n_modified_tokens", 0))
        if budget <= 0:
            raise ValueError(f"{identifier}: repaired ReFlow budget is not positive")
        if int(row.get("reader_calls", 0)) <= 0:
            raise ValueError(f"{identifier}: repaired ReFlow made no reader call")
        selected_ids = [str(value) for value in row.get("selected_ids", [])]
        if len(selected_ids) != budget or len(set(selected_ids)) != budget:
            raise ValueError(f"{identifier}: ReFlow terminal token count mismatch")


def command_prepare(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.manifest)
    target_ids = [str(value) for value in manifest["target_ids"]]
    score_paths = parse_named_paths(args.scores)
    named_paths = {
        "units": args.units_cache,
        "reflow": args.reflow_results,
        "old_baselines": args.old_baselines,
        **{f"score:{name}": path for name, path in score_paths.items()},
    }
    _, tables = aligned_tables(args.input, named_paths, args.n)
    validate_reflow_budgets(tables["reflow"], target_ids)

    outputs = {
        "input": args.out_dir / "input.jsonl",
        "units": args.out_dir / "units.jsonl",
        "reflow": args.out_dir / "reflow_results.jsonl",
        "old_baselines": args.out_dir / "old_baselines.jsonl",
    }
    for name, path in score_paths.items():
        outputs[f"score:{name}"] = args.out_dir / "scores" / f"{name}.jsonl"
    for name, path in outputs.items():
        atomic_write_jsonl(path, selected_rows(tables[name], target_ids))

    summary = {
        "schema": "causalityrag.targeted_table3_prepare.v1",
        "target_queries": len(target_ids),
        "target_ids": target_ids,
        "outputs": {
            name: {
                "path": str(path.resolve()),
                "sha256": file_sha256(str(path)),
            }
            for name, path in outputs.items()
        },
    }
    atomic_write_json(args.out_dir / "prepare.summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def merge_by_target(
    old_rows: list[dict],
    resumed_rows: list[dict],
    target_ids: list[str],
) -> list[dict]:
    if [record_id(row) for row in resumed_rows] != target_ids:
        raise ValueError("resumed rows do not exactly match target manifest order")
    replacements = {record_id(row): row for row in resumed_rows}
    return [replacements.get(record_id(row), row) for row in old_rows]


def baseline_method_names(rows: list[dict]) -> set[str]:
    return {
        str(name)
        for row in rows
        for name in row.get("methods", {})
        if str(name)
    }


def validate_resumed_baselines(
    resumed: list[dict],
    reflow: list[dict],
    target_ids: list[str],
    expected_methods: set[str],
) -> None:
    reflow_by_id = {record_id(row): row for row in reflow}
    for row in resumed:
        identifier = record_id(row)
        budget = int(reflow_by_id[identifier].get("n_modified_tokens", 0))
        if int(row.get("matched_token_budget", -1)) != budget:
            raise ValueError(f"{identifier}: baseline budget does not match ReFlow")
        methods = row.get("methods", {})
        if set(methods) != expected_methods:
            raise ValueError(f"{identifier}: baseline method set is incomplete")
        for name, method in methods.items():
            selected = [str(value) for value in method.get("selected_ids", [])]
            if int(method.get("n_modified_tokens", -1)) != budget:
                raise ValueError(f"{identifier}/{name}: token count mismatch")
            if len(selected) != budget or len(set(selected)) != budget:
                raise ValueError(f"{identifier}/{name}: selected IDs mismatch")
            if not method.get("reader_called"):
                raise ValueError(f"{identifier}/{name}: reader was not called")
        if int(row.get("reader_calls", -1)) != len(expected_methods):
            raise ValueError(f"{identifier}: baseline reader call count mismatch")
    if [record_id(row) for row in resumed] != target_ids:
        raise ValueError("baseline rows are not in target order")


def command_merge_baselines(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.manifest)
    target_ids = [str(value) for value in manifest["target_ids"]]
    old_rows = require_rows(args.old_baselines, args.n)
    resumed = require_rows(args.resumed_baselines, len(target_ids))
    reflow = require_rows(args.reflow_results, args.n)
    expected_methods = baseline_method_names(old_rows)
    if not expected_methods:
        raise ValueError("old baseline table contains no method names")
    validate_resumed_baselines(
        resumed,
        reflow,
        target_ids,
        expected_methods,
    )
    merged = merge_by_target(old_rows, resumed, target_ids)
    atomic_write_jsonl(args.out, merged)
    summary = {
        **summarize_baselines(merged),
        "schema": "causalityrag.targeted_baseline_merge.v1",
        "target_queries": len(target_ids),
        "preserved_queries": len(merged) - len(target_ids),
        "target_reader_calls": sum(
            int(row.get("reader_calls", 0)) for row in resumed
        ),
        "out": str(args.out.resolve()),
        "out_sha256": file_sha256(str(args.out)),
    }
    atomic_write_json(args.summary_out, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def command_merge_controls(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.manifest)
    target_ids = [str(value) for value in manifest["target_ids"]]
    old_rows = require_rows(args.old_controls, args.n)
    resumed = require_rows(args.resumed_controls, len(target_ids))
    reflow = require_rows(args.reflow_results, args.n)
    baselines = require_rows(args.baseline_results, args.n)
    reflow_by_id = {record_id(row): row for row in reflow}
    baseline_by_id = {record_id(row): row for row in baselines}
    expected_methods = baseline_method_names(baselines).union({"reflow"})
    for row in resumed:
        identifier = record_id(row)
        methods = row.get("methods", {})
        if set(methods) != expected_methods:
            raise ValueError(f"{identifier}: control method set is incomplete")
        expected_selections = {
            "reflow": [
                str(value)
                for value in reflow_by_id[identifier].get("selected_ids", [])
            ],
            **{
                name: [str(value) for value in method.get("selected_ids", [])]
                for name, method in baseline_by_id[identifier]
                .get("methods", {})
                .items()
            },
        }
        for name, selection in expected_selections.items():
            if methods[name].get("selected_ids", selection) != selection:
                raise ValueError(f"{identifier}/{name}: control selection mismatch")
    merged = merge_by_target(old_rows, resumed, target_ids)
    atomic_write_jsonl(args.out, merged)
    summary = {
        **summarize_controls(merged, reader_mode=args.reader_mode),
        "schema": "causalityrag.targeted_control_merge.v1",
        "target_queries": len(target_ids),
        "preserved_queries": len(merged) - len(target_ids),
        "out": str(args.out.resolve()),
        "out_sha256": file_sha256(str(args.out)),
    }
    atomic_write_json(args.summary_out, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def changed_ids(old_path: Path, new_path: Path, expected_rows: int) -> set[str]:
    old_rows = require_rows(old_path, expected_rows)
    new_rows = require_rows(new_path, expected_rows)
    if [record_id(row) for row in old_rows] != [record_id(row) for row in new_rows]:
        raise ValueError("old and new row orders differ")
    return {
        record_id(old)
        for old, new in zip(old_rows, new_rows)
        if old != new
    }


def command_audit(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.manifest)
    target_ids = [str(value) for value in manifest["target_ids"]]
    target_set = set(target_ids)
    baseline_changes = changed_ids(
        args.old_baselines,
        args.new_baselines,
        args.n,
    )
    control_changes = changed_ids(args.old_controls, args.new_controls, args.n)
    if baseline_changes != target_set:
        raise ValueError("baseline changes do not equal target whitelist")
    if control_changes != target_set:
        raise ValueError("control changes do not equal target whitelist")
    reflow = require_rows(args.reflow_results, args.n)
    baselines = require_rows(args.new_baselines, args.n)
    resumed = selected_rows(baselines, target_ids)
    validate_resumed_baselines(
        resumed,
        reflow,
        target_ids,
        baseline_method_names(baselines),
    )
    summary = {
        "schema": "causalityrag.targeted_table3_audit.v1",
        "target_queries": len(target_ids),
        "changed_baseline_queries": len(baseline_changes),
        "changed_control_queries": len(control_changes),
        "preserved_baseline_queries": args.n - len(baseline_changes),
        "preserved_control_queries": args.n - len(control_changes),
        "target_baseline_reader_calls": sum(
            int(row.get("reader_calls", 0)) for row in resumed
        ),
        "new_baselines_sha256": file_sha256(str(args.new_baselines)),
        "new_controls_sha256": file_sha256(str(args.new_controls)),
    }
    atomic_write_json(args.out, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--manifest", type=Path, required=True)
    prepare.add_argument("--input", type=Path, required=True)
    prepare.add_argument("--units-cache", type=Path, required=True)
    prepare.add_argument("--reflow-results", type=Path, required=True)
    prepare.add_argument("--old-baselines", type=Path, required=True)
    prepare.add_argument("--scores", action="append", default=[])
    prepare.add_argument("--out-dir", type=Path, required=True)
    prepare.add_argument("--n", type=int, default=1000)
    prepare.set_defaults(handler=command_prepare)

    merge_baselines = subparsers.add_parser("merge-baselines")
    merge_baselines.add_argument("--manifest", type=Path, required=True)
    merge_baselines.add_argument("--old-baselines", type=Path, required=True)
    merge_baselines.add_argument("--resumed-baselines", type=Path, required=True)
    merge_baselines.add_argument("--reflow-results", type=Path, required=True)
    merge_baselines.add_argument("--out", type=Path, required=True)
    merge_baselines.add_argument("--summary-out", type=Path, required=True)
    merge_baselines.add_argument("--n", type=int, default=1000)
    merge_baselines.set_defaults(handler=command_merge_baselines)

    merge_controls = subparsers.add_parser("merge-controls")
    merge_controls.add_argument("--manifest", type=Path, required=True)
    merge_controls.add_argument("--old-controls", type=Path, required=True)
    merge_controls.add_argument("--resumed-controls", type=Path, required=True)
    merge_controls.add_argument("--reflow-results", type=Path, required=True)
    merge_controls.add_argument("--baseline-results", type=Path, required=True)
    merge_controls.add_argument("--reader-mode", default="short_answer")
    merge_controls.add_argument("--out", type=Path, required=True)
    merge_controls.add_argument("--summary-out", type=Path, required=True)
    merge_controls.add_argument("--n", type=int, default=1000)
    merge_controls.set_defaults(handler=command_merge_controls)

    audit = subparsers.add_parser("audit")
    audit.add_argument("--manifest", type=Path, required=True)
    audit.add_argument("--old-baselines", type=Path, required=True)
    audit.add_argument("--new-baselines", type=Path, required=True)
    audit.add_argument("--old-controls", type=Path, required=True)
    audit.add_argument("--new-controls", type=Path, required=True)
    audit.add_argument("--reflow-results", type=Path, required=True)
    audit.add_argument("--out", type=Path, required=True)
    audit.add_argument("--n", type=int, default=1000)
    audit.set_defaults(handler=command_audit)

    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
