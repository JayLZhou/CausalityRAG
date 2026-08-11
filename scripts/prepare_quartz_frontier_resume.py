#!/usr/bin/env python3
"""Prepare and merge aligned QuaRTz factual-frontier repair shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def row_id(row: dict) -> str:
    return str(row.get("id", row.get("query_id", "")))


def unique_by_id(rows: list[dict], *, label: str) -> dict[str, dict]:
    result = {}
    for row in rows:
        identifier = row_id(row)
        if not identifier or identifier in result:
            raise ValueError(f"{label} has an empty or duplicate id: {identifier!r}")
        result[identifier] = row
    return result


def prepare(args: argparse.Namespace) -> None:
    factual = read_jsonl(args.postprocessed_factual)
    resume_ids = [
        row_id(row) for row in factual if row.get("requires_frontier_resume")
    ]
    if len(resume_ids) != len(set(resume_ids)):
        raise ValueError("postprocessed factual artifact has duplicate resume ids")

    sources = {
        "input": read_jsonl(args.input),
        "units": read_jsonl(args.units),
        "frontier": read_jsonl(args.frontier),
    }
    source_maps = {
        name: unique_by_id(rows, label=name) for name, rows in sources.items()
    }
    expected = set(source_maps["input"])
    for name, rows in source_maps.items():
        if set(rows) != expected:
            raise ValueError(f"{name} query-id set does not match input")
    missing = set(resume_ids).difference(expected)
    if missing:
        raise ValueError(f"resume ids are absent from aligned inputs: {sorted(missing)[:5]}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in source_maps.items():
        write_jsonl(
            args.out_dir / f"{name}.jsonl",
            [rows[identifier] for identifier in resume_ids],
        )
    report = {
        "schema": "causalityrag.quartz_frontier_resume.v1",
        "queries": len(resume_ids),
        "ids": resume_ids,
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


def merge(args: argparse.Namespace) -> None:
    original = read_jsonl(args.original_factual)
    resumed = read_jsonl(args.resumed_factual)
    original_by_id = unique_by_id(original, label="original factual")
    resumed_by_id = unique_by_id(resumed, label="resumed factual")
    expected_ids = {
        row_id(row)
        for row in read_jsonl(args.postprocessed_factual)
        if row.get("requires_frontier_resume")
    }
    if set(resumed_by_id) != expected_ids:
        raise ValueError(
            "resumed factual ids do not exactly match the required resume set"
        )
    merged = [
        resumed_by_id.get(row_id(row), original_by_id[row_id(row)])
        for row in original
    ]
    write_jsonl(args.out, merged)
    print(json.dumps({
        "schema": "causalityrag.quartz_frontier_merge.v1",
        "queries": len(merged),
        "replaced": len(resumed_by_id),
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--input", type=Path, required=True)
    prepare_parser.add_argument("--units", type=Path, required=True)
    prepare_parser.add_argument("--frontier", type=Path, required=True)
    prepare_parser.add_argument(
        "--postprocessed-factual", type=Path, required=True
    )
    prepare_parser.add_argument("--out-dir", type=Path, required=True)
    prepare_parser.set_defaults(func=prepare)

    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--original-factual", type=Path, required=True)
    merge_parser.add_argument("--resumed-factual", type=Path, required=True)
    merge_parser.add_argument(
        "--postprocessed-factual", type=Path, required=True
    )
    merge_parser.add_argument("--out", type=Path, required=True)
    merge_parser.set_defaults(func=merge)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
