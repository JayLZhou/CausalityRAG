#!/usr/bin/env python3
"""Report every final Table 3 baseline execution contract violation."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.io import load_records, record_id
from scripts.audit_table3_final8 import (
    BASELINE_ALIASES,
    DATASETS,
    RANDOM_METHODS,
    is_valid_clean,
)


def indexed(path: Path) -> dict[str, dict]:
    rows = load_records(path)
    result = {record_id(row): row for row in rows}
    if len(rows) != 1000 or len(result) != 1000:
        raise ValueError(f"{path}: expected 1,000 unique rows")
    return result


def canonical_methods(names: set[str]) -> dict[str, str]:
    result = {name: name for name in RANDOM_METHODS}
    for canonical, aliases in BASELINE_ALIASES.items():
        matches = names.intersection(aliases)
        if len(matches) != 1:
            raise ValueError(
                f"expected exactly one {canonical} alias, got {sorted(matches)}"
            )
        result[canonical] = next(iter(matches))
    return result


def audit_dataset(root: Path, dataset: str) -> dict:
    base = root / dataset
    reflow = indexed(base / "methods/reflow/results_top5_1000.jsonl")
    baseline = indexed(
        base / "audits/final_top10pool_k5/baselines_1000.jsonl"
    )
    all_names = {
        str(name)
        for row in baseline.values()
        for name in row.get("methods", {})
    }
    methods = canonical_methods(all_names)
    violations = []
    status_histograms = {name: Counter() for name in methods}
    valid_clean = 0
    for identifier, reflow_row in reflow.items():
        if not is_valid_clean(reflow_row):
            continue
        valid_clean += 1
        budget = int(reflow_row.get("n_modified_tokens", 0))
        baseline_row = baseline[identifier]
        row_methods = baseline_row.get("methods", {})
        for canonical, stored_name in methods.items():
            method = row_methods.get(stored_name)
            reasons = []
            if not isinstance(method, dict):
                status_histograms[canonical]["missing"] += 1
                reasons.append("missing_method")
                method = {}
            else:
                status_histograms[canonical][str(method.get("status", ""))] += 1
            selected = [str(value) for value in method.get("selected_ids", [])]
            if budget <= 0:
                reasons.append("nonpositive_reflow_budget")
            if int(method.get("matched_token_budget", -1)) != budget:
                reasons.append("matched_budget_mismatch")
            if int(method.get("n_modified_tokens", -1)) != budget:
                reasons.append("modified_token_count_mismatch")
            if len(selected) != budget:
                reasons.append("selection_length_mismatch")
            if len(set(selected)) != len(selected):
                reasons.append("duplicate_selected_ids")
            if not method.get("reader_called"):
                reasons.append("reader_not_called")
            if reasons:
                violations.append({
                    "id": identifier,
                    "method": canonical,
                    "stored_method": stored_name,
                    "budget": budget,
                    "status": str(method.get("status", "")),
                    "n_modified_tokens": int(
                        method.get("n_modified_tokens", -1)
                    ),
                    "selected_ids": selected,
                    "reasons": reasons,
                })
    return {
        "dataset": dataset,
        "valid_clean_queries": valid_clean,
        "violations": violations,
        "violation_query_methods": len(violations),
        "violation_queries": len({row["id"] for row in violations}),
        "status_histograms": {
            method: dict(sorted(histogram.items()))
            for method, histogram in status_histograms.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--datasets", nargs="*", default=list(DATASETS))
    args = parser.parse_args()

    unknown = sorted(set(args.datasets).difference(DATASETS))
    if unknown:
        parser.error(f"unsupported datasets: {unknown}")
    datasets = [audit_dataset(args.root, dataset) for dataset in args.datasets]
    payload = {
        "schema": "causalityrag.table3_baseline_execution_audit.v1",
        "datasets": datasets,
        "violation_query_methods": sum(
            row["violation_query_methods"] for row in datasets
        ),
        "violation_queries": sum(row["violation_queries"] for row in datasets),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
