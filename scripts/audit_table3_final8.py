#!/usr/bin/env python3
"""Strictly audit the final eight-dataset Table 3 artifact bundle."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.io import load_records, record_id
from causalityrag.shared_replacement_pool import file_sha256
from scripts.evaluate_reflow import valid_clean_answer


DATASETS = (
    "hotpotqa",
    "timeqa",
    "finqa",
    "musique",
    "quartz",
    "triviaqa",
    "2wiki",
    "popqa",
)
BASELINE_ALIASES = {
    "attention": ("attention", "attention_prompt_last"),
    "gradient_x_input": ("gradient_x_input",),
    "integrated_gradients": ("integrated_gradients",),
    "mirage": ("mirage",),
    "arc_jsd": ("arc_jsd",),
}
RANDOM_METHODS = tuple(f"random_seed{seed}" for seed in range(5))


def require_rows(path: Path, expected: int = 1000) -> list[dict]:
    rows = load_records(path)
    identifiers = [record_id(row) for row in rows]
    if len(rows) != expected:
        raise ValueError(f"{path}: expected {expected} rows, got {len(rows)}")
    if any(not identifier for identifier in identifiers):
        raise ValueError(f"{path}: every row must have an ID")
    if len(set(identifiers)) != expected:
        raise ValueError(f"{path}: IDs must be unique")
    return rows


def is_valid_clean(row: dict) -> bool:
    status = str(row.get("evaluation_status", ""))
    return (
        valid_clean_answer(str(row.get("clean_answer", "")))
        and status != "invalid_clean_answer"
        and not status.startswith("protocol_violation_invalid_clean")
    )


def method_set(rows: list[dict]) -> set[str]:
    return {
        str(name)
        for row in rows
        for name in row.get("methods", {})
        if str(name)
    }


def canonical_method_set(names: set[str]) -> set[str]:
    canonical = set(RANDOM_METHODS)
    for name, aliases in BASELINE_ALIASES.items():
        matches = names.intersection(aliases)
        if len(matches) != 1:
            raise ValueError(
                f"expected exactly one {name} alias, found {sorted(matches)}"
            )
        canonical.add(next(iter(matches)))
    if names != canonical:
        raise ValueError(
            f"baseline method set mismatch: expected={sorted(canonical)} "
            f"actual={sorted(names)}"
        )
    return canonical


def require_ratio(row: dict, count_key: str, ratio_key: str, denominator: int) -> None:
    count = int(row[count_key])
    ratio = float(row[ratio_key])
    expected = count / max(1, denominator)
    if not math.isclose(ratio, expected, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"{ratio_key} mismatch: stored={ratio} expected={expected}"
        )


def audit_dataset(root: Path, dataset: str) -> dict:
    base = root / dataset
    paths = {
        "reflow": base / "methods/reflow/results_top5_1000.jsonl",
        "baselines": base / "audits/final_top10pool_k5/baselines_1000.jsonl",
        "controls": base / "controls/paraphrase_results_top5_1000_v2.jsonl",
        "factual": base / "audits/final_top10pool_k5/factual_metrics_1000.json",
        "control_summary": base / "controls/paraphrase_results_top5_1000_v2.summary.json",
        "adjusted": base / "audits/final_top10pool_k5/table3_metrics_1000_v2.json",
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{dataset}/{name}: {path}")

    reflow = require_rows(paths["reflow"])
    baselines = require_rows(paths["baselines"])
    controls = require_rows(paths["controls"])
    expected_ids = [record_id(row) for row in reflow]
    for name, rows in (("baselines", baselines), ("controls", controls)):
        if [record_id(row) for row in rows] != expected_ids:
            raise ValueError(f"{dataset}/{name}: row IDs are not aligned")

    baseline_names = canonical_method_set(method_set(baselines))
    expected_control_names = baseline_names.union({"reflow"})
    if method_set(controls) != expected_control_names:
        raise ValueError(f"{dataset}: control method set is incomplete")

    valid_clean_ids = {
        record_id(row) for row in reflow if is_valid_clean(row)
    }
    invalid_clean_ids = set(expected_ids).difference(valid_clean_ids)
    reflow_by_id = {record_id(row): row for row in reflow}
    baseline_by_id = {record_id(row): row for row in baselines}
    control_by_id = {record_id(row): row for row in controls}
    for identifier in valid_clean_ids:
        reflow_row = reflow_by_id[identifier]
        budget = int(reflow_row.get("n_modified_tokens", 0))
        if budget <= 0:
            raise ValueError(f"{dataset}/{identifier}: nonpositive ReFlow budget")
        methods = baseline_by_id[identifier].get("methods", {})
        if set(methods) != baseline_names:
            raise ValueError(f"{dataset}/{identifier}: baseline methods incomplete")
        for name, method in methods.items():
            selected = [str(value) for value in method.get("selected_ids", [])]
            if int(method.get("matched_token_budget", budget)) != budget:
                raise ValueError(f"{dataset}/{identifier}/{name}: budget mismatch")
            if int(method.get("n_modified_tokens", -1)) != budget:
                raise ValueError(f"{dataset}/{identifier}/{name}: edit count mismatch")
            if len(selected) != budget or len(set(selected)) != budget:
                raise ValueError(f"{dataset}/{identifier}/{name}: selection mismatch")
            if not method.get("reader_called"):
                raise ValueError(f"{dataset}/{identifier}/{name}: reader not called")
        control_methods = control_by_id[identifier].get("methods", {})
        if set(control_methods) != expected_control_names:
            raise ValueError(f"{dataset}/{identifier}: control methods incomplete")

    factual = json.loads(paths["factual"].read_text(encoding="utf-8"))
    control_summary = json.loads(
        paths["control_summary"].read_text(encoding="utf-8")
    )
    expected_metric_names = expected_control_names
    if set(factual.get("methods", {})) != expected_metric_names:
        raise ValueError(f"{dataset}: factual metric method set mismatch")
    if set(control_summary.get("methods", {})) != expected_metric_names:
        raise ValueError(f"{dataset}: control metric method set mismatch")
    denominator = len(valid_clean_ids)
    for source_name, summary, count_key, ratio_key in (
        ("factual", factual, "answer_flip_count", "answer_flip_ratio"),
        ("control", control_summary, "answer_flips", "answer_flip_rate_itt"),
    ):
        for name, method in summary["methods"].items():
            if int(method.get("answer_denominator_queries", -1)) != denominator:
                raise ValueError(
                    f"{dataset}/{source_name}/{name}: Ans-FR denominator mismatch"
                )
            require_ratio(method, count_key, ratio_key, denominator)

    if dataset == "popqa":
        aliases = [
            row.get("gold_answers", [])
            for row in reflow
            if len(row.get("gold_answers", [])) > 1
        ]
        if not aliases:
            raise ValueError("PopQA results do not preserve multiple aliases")

    return {
        "dataset": dataset,
        "queries": len(expected_ids),
        "valid_clean_queries": len(valid_clean_ids),
        "invalid_clean_queries": len(invalid_clean_ids),
        "baseline_methods": sorted(baseline_names),
        "control_methods": sorted(expected_control_names),
        "verified_reflow_flips": sum(
            bool(row.get("verified_flip")) for row in reflow
        ),
        "shas": {
            name: file_sha256(str(path)) for name, path in paths.items()
        },
        "paths": {name: str(path.resolve()) for name, path in paths.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    datasets = [audit_dataset(args.root, dataset) for dataset in DATASETS]
    result = {
        "schema": "causalityrag.table3_final8_audit.v1",
        "dataset_order": list(DATASETS),
        "pubmedqa_in_main_table": False,
        "datasets": datasets,
        "queries": sum(row["queries"] for row in datasets),
        "valid_clean_queries": sum(
            row["valid_clean_queries"] for row in datasets
        ),
        "invalid_clean_queries": sum(
            row["invalid_clean_queries"] for row in datasets
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
