#!/usr/bin/env python3
"""Aggregate per-dataset answer-frontier summaries into macro curves."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def macro_curve(summaries: dict[str, dict], method: str, max_n: int) -> list[dict]:
    curves = {
        dataset: {int(row["n"]): row for row in summary["methods"][method]}
        for dataset, summary in summaries.items()
    }
    fields = (
        "mean_modified_tokens",
        "factual_answer_flip_rate",
        "synonym_answer_flip_rate",
        "answer_cfr",
        "mean_independent_verifications",
    )
    return [
        {
            "n": n,
            **{
                field: statistics.fmean(curve[n][field] for curve in curves.values())
                for field in fields
            },
        }
        for n in range(1, max_n + 1)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary",
        action="append",
        required=True,
        help="Dataset and summary path in DATASET=PATH form.",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    summaries = {}
    paths = {}
    for value in args.summary:
        dataset, separator, raw_path = value.partition("=")
        if not separator or not dataset or not raw_path:
            raise ValueError(f"invalid --summary value: {value}")
        path = Path(raw_path)
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("schema") != "causalityrag.prefix_answer_frontier.v1":
            raise ValueError(f"unexpected schema in {path}")
        summaries[dataset] = summary
        paths[dataset] = str(path.resolve())
    max_values = {int(summary["max_n"]) for summary in summaries.values()}
    if len(max_values) != 1:
        raise ValueError("dataset summaries use different max_n values")
    max_n = max_values.pop()
    output = {
        "schema": "causalityrag.prefix_answer_frontier.aggregate.v1",
        "datasets": sorted(summaries),
        "query_counts": {
            dataset: int(summary["queries"])
            for dataset, summary in sorted(summaries.items())
        },
        "total_queries": sum(int(summary["queries"]) for summary in summaries.values()),
        "max_n": max_n,
        "macro_average": {
            method: macro_curve(summaries, method, max_n)
            for method in ("exhaustive", "reflow")
        },
        "per_dataset": summaries,
        "source_paths": paths,
    }
    Path(args.out).write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output["macro_average"], indent=2))


if __name__ == "__main__":
    main()
