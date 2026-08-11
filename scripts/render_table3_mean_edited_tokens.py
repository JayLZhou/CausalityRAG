#!/usr/bin/env python3
"""Render the final Table 3 per-dataset matched token budgets."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.audit_table3_token_counts import (
    DATASETS,
    audit_dataset,
    first_existing,
    load_jsonl,
    reflow_policy_count,
    row_id,
)


DATASET_LABELS = {
    "hotpotqa": "HQA",
    "timeqa": "TQA",
    "finqa": "FQA",
    "musique": "MSQ",
    "quartz": "QTZ",
    "triviaqa": "TrQA",
    "2wiki": "2Wiki",
    "popqa": "PopQA",
}

DATASET_MACROS = {
    "hotpotqa": "HotpotQA",
    "timeqa": "TimeQA",
    "finqa": "FinQA",
    "musique": "MuSiQue",
    "quartz": "QuaRTz",
    "triviaqa": "TriviaQA",
    "2wiki": "TwoWiki",
    "popqa": "PopQA",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tex_coordinates(rows: list[dict[str, Any]], field: str) -> str:
    return " ".join(
        f"({row['label']},{float(row[field]):.3f})" for row in rows
    )


def dataset_payload(root: Path, dataset: str) -> dict[str, Any]:
    audit = audit_dataset(root, dataset)
    if audit["expected_queries"] != 1000:
        raise ValueError(f"{dataset}: expected 1000 frozen queries")
    alignment_errors = {
        key: audit[key]
        for key in (
            "duplicate_expected_ids",
            "reflow_duplicate_ids",
            "baseline_duplicate_ids",
            "reflow_missing_ids",
            "baseline_missing_ids",
            "reflow_extra_ids",
            "baseline_extra_ids",
        )
        if audit[key]
    }
    if alignment_errors:
        raise ValueError(f"{dataset}: ID alignment failed: {alignment_errors}")
    if not audit["all_methods_have_1000_values"]:
        raise ValueError(f"{dataset}: at least one Table 3 method lacks a token count")

    reflow_audit = audit["methods"]["reflow"]
    if reflow_audit["stored_count_mismatch_ids"]:
        raise ValueError(
            f"{dataset}: stored ReFlow token counts violate the terminal-attempt "
            f"contract for {len(reflow_audit['stored_count_mismatch_ids'])} queries"
        )
    common_mean = float(reflow_audit["mean"])
    for method, method_audit in audit["methods"].items():
        method_mean = float(method_audit["mean"])
        if abs(method_mean - common_mean) > 1e-12:
            raise ValueError(
                f"{dataset}: {method} mean token count {method_mean} does not "
                f"match ReFlow {common_mean}"
            )

    base = root / dataset
    reflow_path = first_existing((
        base / "methods/reflow/results_top5_1000.jsonl",
        base / "audits/final_top10pool_k5/reflow_1000_v2.jsonl",
    ))
    reflow_rows = load_jsonl(reflow_path)
    by_id = {row_id(row): row for row in reflow_rows}
    retrieval_rows = load_jsonl(base / "retrieval/top10_1000.jsonl")[:1000]
    expected_ids = [row_id(row) for row in retrieval_rows]

    valid_counts: list[int] = []
    invalid_clean = 0
    executed = 0
    verified_flips = 0
    for identifier in expected_ids:
        row = by_id[identifier]
        count, _ = reflow_policy_count(row)
        if count is None:
            raise ValueError(f"{dataset}/{identifier}: missing ReFlow token count")
        if row.get("eligible"):
            valid_counts.append(count)
        else:
            invalid_clean += 1
        executed += int(row.get("reader_calls", 0)) > 0
        verified_flips += bool(row.get("verified_flip"))

    if len(valid_counts) + invalid_clean != 1000:
        raise ValueError(f"{dataset}: valid/invalid clean partition is incomplete")
    return {
        "dataset": dataset,
        "label": DATASET_LABELS[dataset],
        "queries": 1000,
        "valid_clean_queries": len(valid_counts),
        "invalid_clean_queries": invalid_clean,
        "edited_token_sum_valid_clean": sum(valid_counts),
        "executed_queries": executed,
        "verified_flips": verified_flips,
        "mean_edited_tokens_all_1000": common_mean,
        "mean_edited_tokens_valid_clean": (
            statistics.fmean(valid_counts) if valid_counts else 0.0
        ),
        "median_edited_tokens_all_1000": reflow_audit["median"],
        "p95_edited_tokens_all_1000": reflow_audit["p95"],
        "max_edited_tokens_all_1000": reflow_audit["max"],
        "zero_budget_queries": reflow_audit["zero_queries"],
        "status_histogram": reflow_audit["status_histogram"],
        "reflow_results": str(reflow_path),
        "reflow_results_sha256": file_sha256(reflow_path),
        "baseline_results": audit["paths"]["baselines"],
        "baseline_results_sha256": file_sha256(Path(audit["paths"]["baselines"])),
    }


def build_payload(root: Path, datasets: list[str]) -> dict[str, Any]:
    unknown = sorted(set(datasets).difference(DATASETS))
    if unknown:
        raise ValueError(f"unsupported datasets: {unknown}")
    rows = [dataset_payload(root, dataset) for dataset in datasets]
    token_sum = sum(row["edited_token_sum_valid_clean"] for row in rows)
    valid_clean_queries = sum(row["valid_clean_queries"] for row in rows)
    return {
        "schema": "causalityrag.table3_mean_edited_tokens.v2",
        "table3_datasets": datasets,
        "population": "queries with a valid clean answer in each dataset",
        "token_count_contract": {
            "verified_flip": "actual token count at the verified-flip attempt",
            "failed_flip": "largest saved terminal attempted token count",
            "invalid_clean": "excluded from both numerator and denominator",
            "baselines": (
                "all Table 3 baselines use ReFlow's exact per-query terminal "
                "budget, so every method has the same dataset mean"
            ),
        },
        "diagnostic_population": (
            "all-1,000-query means retain invalid-clean rows as zero only for audit"
        ),
        "datasets": rows,
        "unweighted_dataset_macro_mean_valid_clean": statistics.fmean(
            row["mean_edited_tokens_valid_clean"] for row in rows
        ),
        "pooled_valid_clean_token_sum": token_sum,
        "pooled_valid_clean_queries": valid_clean_queries,
        "pooled_valid_clean_mean": token_sum / max(1, valid_clean_queries),
        "diagnostic_unweighted_dataset_macro_mean_all_1000": statistics.fmean(
            row["mean_edited_tokens_all_1000"] for row in rows
        ),
    }


def render_tex(payload: dict[str, Any]) -> str:
    rows = payload["datasets"]
    lines = [
        "% Generated by scripts/render_table3_mean_edited_tokens.py; do not edit.",
        "\\def\\TableThreeMeanEditedTokenCoordinates{" +
        tex_coordinates(rows, "mean_edited_tokens_valid_clean") + "}",
        "\\def\\TableThreeValidCleanMeanEditedTokenCoordinates{" +
        tex_coordinates(rows, "mean_edited_tokens_valid_clean") + "}",
        f"\\def\\TableThreeMeanEditedTokenMacro{{{payload['unweighted_dataset_macro_mean_valid_clean']:.3f}}}",
    ]
    lines.extend(
        f"\\def\\TableThree{DATASET_MACROS[row['dataset']]}MeanEditedTokens"
        f"{{{float(row['mean_edited_tokens_valid_clean']):.3f}}}"
        for row in rows
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-tex", type=Path, required=True)
    parser.add_argument("--datasets", nargs="*", default=list(DATASETS))
    args = parser.parse_args()

    payload = build_payload(args.root, args.datasets)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_tex.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    args.out_tex.write_text(render_tex(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
