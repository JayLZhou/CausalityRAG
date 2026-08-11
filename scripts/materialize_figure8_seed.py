#!/usr/bin/env python3
"""Materialize aligned Figure 8 seeds from saved per-budget reader outputs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.io import load_records, record_id
from scripts.rescore_figure8_dataset import (
    METHOD_ALIASES,
    METHODS,
    discover_sources,
    prepare_population,
    sha256,
)


def parse_budget_values(value: str) -> list[int]:
    budgets: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_raw, end_raw = part.split("-", 1)
            start, end = int(start_raw), int(end_raw)
            if start <= 0 or end < start:
                raise ValueError(f"invalid budget range: {part}")
            budgets.update(range(start, end + 1))
        else:
            budget = int(part)
            if budget <= 0:
                raise ValueError(f"invalid budget: {part}")
            budgets.add(budget)
    if not budgets:
        raise ValueError("at least one positive budget is required")
    return sorted(budgets)


def parse_points(specs: list[str]) -> set[tuple[str, int]]:
    points: set[tuple[str, int]] = set()
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"invalid --points value: {spec}")
        method, value = spec.split("=", 1)
        method = method.strip()
        if method not in METHODS:
            raise ValueError(f"unsupported Figure 8 method: {method}")
        points.update((method, budget) for budget in parse_budget_values(value))
    if not points:
        raise ValueError("at least one --points value is required")
    return points


def method_entries(row: dict, method: str) -> dict:
    methods = row.get("methods", {})
    return next(
        (methods[alias] for alias in METHOD_ALIASES[method] if alias in methods),
        {},
    )


def materialize_seed(
    records: list[dict],
    selected: dict[tuple[str, int], Path],
) -> tuple[list[dict], dict[tuple[str, int], list[str]], dict[str, dict]]:
    output = [
        {
            "index": index,
            "id": record_id(record),
            "methods": {},
        }
        for index, record in enumerate(records)
    ]
    output_by_id = {record_id(row): row for row in output}
    keys_by_path: dict[Path, set[tuple[str, int]]] = defaultdict(set)
    for key, path in selected.items():
        keys_by_path[path].add(key)

    sources = {}
    for path, keys in sorted(keys_by_path.items(), key=lambda item: str(item[0])):
        rows = load_records(path)
        for row in rows:
            identifier = record_id(row)
            if identifier not in output_by_id:
                continue
            target = output_by_id[identifier]
            for method, budget in keys:
                entry = method_entries(row, method).get(str(budget))
                if entry is None:
                    continue
                target.setdefault("methods", {}).setdefault(method, {})[
                    str(budget)
                ] = entry
        sources[str(path.resolve())] = {
            "sha256": sha256(path),
            "method_budgets": [
                f"{method}:{budget}" for method, budget in sorted(keys)
            ],
        }

    missing: dict[tuple[str, int], list[str]] = {}
    for method, budget in sorted(selected):
        identifiers = [
            record_id(row)
            for row in output
            if str(budget) not in row.get("methods", {}).get(method, {})
        ]
        if identifiers:
            missing[(method, budget)] = identifiers
    return output, missing, sources


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reflow-results", type=Path, required=True)
    parser.add_argument("--points", action="append", default=[])
    parser.add_argument("--allow-absent", action="store_true")
    parser.add_argument("--reader-mode", default="short_answer")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    args = parser.parse_args()

    requested = parse_points(args.points)
    records = load_records(args.input)[: args.n]
    reflow_rows = load_records(args.reflow_results)[: args.n]
    if len(records) != args.n or len(reflow_rows) != args.n:
        raise ValueError("expected exactly --n retrieval and ReFlow rows")
    identifiers = [record_id(row) for row in records]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("input IDs must be unique")
    population, _ = prepare_population(
        records, reflow_rows, reader_mode=args.reader_mode
    )
    discovered = discover_sources(args.dataset_root, set(population))
    absent = requested - discovered.keys()
    if absent and not args.allow_absent:
        raise ValueError(f"requested points have no saved source: {sorted(absent)}")
    selected = {key: discovered[key] for key in requested if key in discovered}
    rows, missing, sources = materialize_seed(records, selected)
    for key in absent:
        missing[key] = identifiers
    write_jsonl(args.out, rows)
    summary = {
        "schema": "causalityrag.figure8_seed.v1",
        "queries": len(rows),
        "clean_acc_queries": len(population),
        "requested_points": [
            f"{method}:{budget}" for method, budget in sorted(requested)
        ],
        "missing": [
            {
                "method": method,
                "budget": budget,
                "queries": len(missing[(method, budget)]),
                "ids": missing[(method, budget)],
            }
            for method, budget in sorted(missing)
        ],
        "absent_source_points": [
            f"{method}:{budget}" for method, budget in sorted(absent)
        ],
        "sources": sources,
        "out": str(args.out.resolve()),
        "out_sha256": sha256(args.out),
    }
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
