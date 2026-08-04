#!/usr/bin/env python3
"""Validate and aggregate the eight-dataset prefix-search audit."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.aggregate_prefix_exhaustive_comparison import macro_curve
from scripts.run_prefix_exhaustive_comparison import summarize


METHODS = ("exhaustive", "reflow")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def close(left: float, right: float) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1e-10, abs_tol=1e-12)


def validate_dataset(dataset: str, directory: Path, expected: int) -> tuple[dict, dict]:
    result_path = directory / "results.jsonl"
    summary_path = directory / "summary.json"
    pool_path = directory / "paraphrase_pool.jsonl"
    manifest_path = directory / "paraphrase_pool.manifest.json"
    unresolved_path = directory / "paraphrase_pool.unresolved.jsonl"
    rows = load_jsonl(result_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if len(rows) != expected or len({str(row["id"]) for row in rows}) != expected:
        raise ValueError(f"{dataset}: expected {expected} unique rows, found {len(rows)}")
    if summary.get("schema") != "causalityrag.prefix_exhaustive_comparison.v1":
        raise ValueError(f"{dataset}: unexpected summary schema")
    if int(summary["queries"]) != expected or int(summary["max_n"]) != 10:
        raise ValueError(f"{dataset}: inconsistent summary dimensions")
    if manifest.get("coverage") != 1.0 or int(manifest["unresolved_positions"]) != 0:
        raise ValueError(f"{dataset}: incomplete paraphrase pool")
    if unresolved_path.exists() and unresolved_path.stat().st_size:
        raise ValueError(f"{dataset}: unresolved paraphrase rows remain")
    pool_rows = load_jsonl(pool_path)
    pool_ids = {str(row["unit_id"]) for row in pool_rows if row.get("candidates")}
    if len(pool_ids) != int(manifest["covered_positions"]):
        raise ValueError(f"{dataset}: pool manifest does not match pool rows")

    for row in rows:
        ranked = tuple(map(str, row["ranked_ids"]))
        if len(ranked) != 10 or len(set(ranked)) != 10:
            raise ValueError(f"{dataset}/{row['id']}: invalid ranked token domain")
        if set(row["prefixes"]) != {str(n) for n in range(1, 11)}:
            raise ValueError(f"{dataset}/{row['id']}: missing prefix")
        for n in range(1, 11):
            domain = set(ranked[:n])
            methods = row["prefixes"][str(n)]
            if set(methods) != set(METHODS):
                raise ValueError(f"{dataset}/{row['id']}/n={n}: method mismatch")
            for method in METHODS:
                value = methods[method]
                selected = list(map(str, value["selected_ids"]))
                if len(selected) != len(set(selected)):
                    raise ValueError(f"{dataset}/{row['id']}/n={n}: duplicate selection")
                if not set(selected).issubset(domain):
                    raise ValueError(f"{dataset}/{row['id']}/n={n}: selection outside prefix")
                if int(value["n_modified_tokens"]) != len(selected):
                    raise ValueError(f"{dataset}/{row['id']}/n={n}: edit count mismatch")
                for key in ("factual_f1_flip", "synonym_f1_flip"):
                    if not isinstance(value[key], bool):
                        raise ValueError(f"{dataset}/{row['id']}/n={n}: {key} is not bool")
                if not set(selected).issubset(pool_ids):
                    raise ValueError(f"{dataset}/{row['id']}/n={n}: synonym pool miss")

    recomputed = summarize(rows, 10)
    for method in METHODS:
        stored_curve = summary["methods"][method]
        computed_curve = recomputed["methods"][method]
        for stored, computed in zip(stored_curve, computed_curve):
            if int(stored["n"]) != int(computed["n"]):
                raise ValueError(f"{dataset}/{method}: n mismatch")
            for key in (
                "mean_modified_tokens",
                "factual_f1_flip_rate",
                "synonym_f1_flip_rate",
                "f1_cfr",
                "mean_selection_reader_calls",
            ):
                if not close(stored[key], computed[key]):
                    raise ValueError(f"{dataset}/{method}/n={stored['n']}: {key} mismatch")
    return summary, {
        "queries": expected,
        "paraphrase_positions": len(pool_ids),
        "factual_reader_calls": int(summary["reader_calls"]["unique_factual"]),
        "synonym_reader_calls": int(summary["reader_calls"]["unique_synonym"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", required=True, help="NAME=DIR=COUNT")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    summaries: dict[str, dict] = {}
    validation: dict[str, dict] = {}
    for item in args.dataset:
        dataset, raw_dir, raw_count = item.split("=", 2)
        summaries[dataset], validation[dataset] = validate_dataset(
            dataset, Path(raw_dir), int(raw_count)
        )
    max_n = {int(summary["max_n"]) for summary in summaries.values()}.pop()
    output = {
        "schema": "causalityrag.prefix_exhaustive_comparison.aggregate.v1",
        "datasets": sorted(summaries),
        "query_counts": {name: validation[name]["queries"] for name in sorted(validation)},
        "total_queries": sum(item["queries"] for item in validation.values()),
        "max_n": max_n,
        "macro_average": {
            method: macro_curve(summaries, method, max_n) for method in METHODS
        },
        "per_dataset": summaries,
        "validation": validation,
    }
    Path(args.out).write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"validation": validation, "macro_average": output["macro_average"]}, indent=2))


if __name__ == "__main__":
    main()
