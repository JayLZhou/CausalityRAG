#!/usr/bin/env python3
"""Aggregate per-dataset F1-CFR prefix-frontier summaries."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


FIELDS = (
    "mean_modified_tokens",
    "factual_f1_flip_rate",
    "synonym_f1_flip_rate",
    "f1_cfr",
    "mean_independent_verifications",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", action="append", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    summaries = {}
    for value in args.summary:
        dataset, separator, raw_path = value.partition("=")
        if not separator:
            raise ValueError(f"invalid --summary value: {value}")
        summary = json.loads(Path(raw_path).read_text(encoding="utf-8"))
        if summary.get("schema") != "causalityrag.prefix_f1_frontier.v1":
            raise ValueError(f"unexpected schema: {raw_path}")
        summaries[dataset] = summary
    max_values = {int(summary["max_n"]) for summary in summaries.values()}
    if len(max_values) != 1:
        raise ValueError("dataset summaries use different max_n values")
    max_n = max_values.pop()
    macro = {}
    for method in ("exhaustive", "reflow"):
        curves = {
            dataset: {int(row["n"]): row for row in summary["methods"][method]}
            for dataset, summary in summaries.items()
        }
        macro[method] = [
            {
                "n": n,
                **{
                    field: statistics.fmean(curve[n][field] for curve in curves.values())
                    for field in FIELDS
                },
            }
            for n in range(1, max_n + 1)
        ]
    output = {
        "schema": "causalityrag.prefix_f1_frontier.aggregate.v1",
        "datasets": sorted(summaries),
        "query_counts": {
            dataset: int(summary["queries"])
            for dataset, summary in sorted(summaries.items())
        },
        "total_queries": sum(int(summary["queries"]) for summary in summaries.values()),
        "max_n": max_n,
        "macro_average": macro,
        "per_dataset": summaries,
    }
    Path(args.out).write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(macro, indent=2))


if __name__ == "__main__":
    main()
