#!/usr/bin/env python3
"""Audit per-query edited-token counts for the frozen Table 3 artifacts."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def row_id(row: dict[str, Any]) -> str:
    return str(row.get("id", ""))


def numeric_count(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def selected_count(row: dict[str, Any]) -> int | None:
    counts = [
        numeric_count(row.get("n_selected")),
        numeric_count(row.get("n_edits")),
        numeric_count(row.get("n_modified_tokens")),
    ]
    selected_ids = row.get("selected_ids")
    if isinstance(selected_ids, list):
        counts.append(len(selected_ids))
    present = [count for count in counts if count is not None]
    return max(present) if present else None


def reflow_policy_count(row: dict[str, Any]) -> tuple[int | None, dict[str, Any]]:
    attempts = row.get("attempts")
    if not isinstance(attempts, list):
        attempts = []
    attempt_counts = [selected_count(attempt) for attempt in attempts]
    present_attempts = [count for count in attempt_counts if count is not None]
    verified = [
        selected_count(attempt)
        for attempt in attempts
        if attempt.get("candidate_status") == "verified_flip"
    ]
    verified = [count for count in verified if count is not None]
    success = bool(row.get("verified_flip"))
    if success and verified:
        count = verified[-1]
        source = "verified_flip_attempt"
    elif present_attempts:
        count = max(present_attempts)
        source = "max_failed_attempt"
    else:
        fallback = selected_count(row)
        count = fallback if fallback is not None else 0
        source = "row_fallback" if fallback is not None else "explicit_zero_no_attempt"
    return count, {
        "source": source,
        "success": success,
        "attempts": len(attempts),
        "last_attempt_count": attempt_counts[-1] if attempt_counts else None,
        "max_attempt_count": max(present_attempts) if present_attempts else None,
        "last_is_max": (
            not present_attempts
            or attempt_counts[-1] == max(present_attempts)
        ),
        "stored_n_modified_tokens": numeric_count(row.get("n_modified_tokens")),
        "status": str(row.get("evaluation_status", "")),
    }


def baseline_policy_count(method: dict[str, Any]) -> tuple[int | None, dict[str, Any]]:
    count = numeric_count(method.get("n_modified_tokens"))
    source = "n_modified_tokens"
    if count is None:
        count = selected_count(method)
        source = "selected_ids_fallback"
    if count is None:
        budget = numeric_count(method.get("matched_token_budget"))
        count = budget
        source = "matched_budget_fallback" if budget is not None else "missing_token_count"
    return count, {
        "source": source,
        "success": bool(method.get("verified_flip")),
        "status": str(method.get("status", "")),
        "matched_token_budget": numeric_count(method.get("matched_token_budget")),
    }


def stats(values: list[float]) -> dict[str, Any]:
    ordered = sorted(values)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "values": len(values),
        "mean": statistics.fmean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "p95": ordered[p95_index] if ordered else None,
        "max": max(values) if values else None,
        "zero_queries": sum(value == 0 for value in values),
    }


def unique_rows(rows: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], list[str]]:
    counts = Counter(row_id(row) for row in rows)
    duplicates = sorted(identifier for identifier, count in counts.items() if count > 1)
    return {row_id(row): row for row in rows}, duplicates


def first_existing(paths: tuple[Path, ...]) -> Path:
    for path in paths:
        if path.is_file():
            return path
    raise FileNotFoundError("none of the expected artifacts exists: " + ", ".join(map(str, paths)))


def audit_dataset(root: Path, dataset: str) -> dict[str, Any]:
    base = root / dataset
    retrieval = load_jsonl(base / "retrieval/top10_1000.jsonl")[:1000]
    expected_ids = [row_id(row) for row in retrieval]
    reflow_path = first_existing((
        base / "methods/reflow/results_top5_1000.jsonl",
        base / "audits/final_top10pool_k5/reflow_1000_v2.jsonl",
    ))
    baseline_path = base / "audits/final_top10pool_k5/baselines_1000.jsonl"
    reflow_rows = load_jsonl(reflow_path)
    baseline_rows = load_jsonl(baseline_path)
    reflow_by_id, reflow_duplicates = unique_rows(reflow_rows)
    baseline_by_id, baseline_duplicates = unique_rows(baseline_rows)

    result: dict[str, Any] = {
        "dataset": dataset,
        "paths": {"reflow": str(reflow_path), "baselines": str(baseline_path)},
        "expected_queries": len(expected_ids),
        "duplicate_expected_ids": len(expected_ids) - len(set(expected_ids)),
        "reflow_rows": len(reflow_rows),
        "baseline_rows": len(baseline_rows),
        "reflow_duplicate_ids": reflow_duplicates,
        "baseline_duplicate_ids": baseline_duplicates,
        "reflow_missing_ids": [identifier for identifier in expected_ids if identifier not in reflow_by_id],
        "baseline_missing_ids": [identifier for identifier in expected_ids if identifier not in baseline_by_id],
        "reflow_extra_ids": sorted(set(reflow_by_id).difference(expected_ids)),
        "baseline_extra_ids": sorted(set(baseline_by_id).difference(expected_ids)),
        "methods": {},
    }

    reflow_values: list[float] = []
    reflow_null_ids: list[str] = []
    reflow_last_not_max: list[str] = []
    reflow_stored_mismatch: list[str] = []
    reflow_sources: Counter[str] = Counter()
    reflow_statuses: Counter[str] = Counter()
    for identifier in expected_ids:
        row = reflow_by_id.get(identifier)
        if row is None:
            reflow_null_ids.append(identifier)
            continue
        count, diagnostics = reflow_policy_count(row)
        if count is None:
            reflow_null_ids.append(identifier)
            continue
        reflow_values.append(float(count))
        reflow_sources[diagnostics["source"]] += 1
        reflow_statuses[diagnostics["status"]] += 1
        if not diagnostics["last_is_max"]:
            reflow_last_not_max.append(identifier)
        stored = diagnostics["stored_n_modified_tokens"]
        if stored is None or stored != count:
            reflow_stored_mismatch.append(identifier)
    result["methods"]["reflow"] = {
        **stats(reflow_values),
        "null_query_ids": reflow_null_ids,
        "count_sources": dict(sorted(reflow_sources.items())),
        "status_histogram": dict(sorted(reflow_statuses.items())),
        "last_attempt_not_max_ids": reflow_last_not_max,
        "stored_count_mismatch_ids": reflow_stored_mismatch,
    }

    canonical_values: dict[str, list[float]] = {
        method: [] for method in BASELINE_ALIASES
    }
    canonical_nulls: dict[str, list[str]] = {
        method: [] for method in BASELINE_ALIASES
    }
    canonical_statuses: dict[str, Counter[str]] = {
        method: Counter() for method in BASELINE_ALIASES
    }
    canonical_sources: dict[str, Counter[str]] = {
        method: Counter() for method in BASELINE_ALIASES
    }
    canonical_implicit_zeros: dict[str, list[str]] = {
        method: [] for method in BASELINE_ALIASES
    }
    random_values: list[float] = []
    random_per_seed: dict[str, list[float]] = {f"random_seed{seed}": [] for seed in range(5)}
    random_nulls: dict[str, list[str]] = {name: [] for name in random_per_seed}
    random_implicit_zeros: dict[str, list[str]] = {
        name: [] for name in random_per_seed
    }
    budget_mismatches: dict[str, list[str]] = {
        method: [] for method in (*BASELINE_ALIASES, *random_per_seed)
    }

    for identifier in expected_ids:
        row = baseline_by_id.get(identifier)
        if row is None:
            for method in canonical_nulls:
                canonical_nulls[method].append(identifier)
            for method in random_nulls:
                random_nulls[method].append(identifier)
            continue
        methods = row.get("methods")
        methods = methods if isinstance(methods, dict) else {}
        row_budget = numeric_count(row.get("matched_token_budget"))
        for canonical, aliases in BASELINE_ALIASES.items():
            method_name = next((name for name in aliases if name in methods), None)
            if method_name is None:
                if row_budget == 0:
                    canonical_values[canonical].append(0.0)
                    canonical_sources[canonical]["implicit_zero_budget"] += 1
                    canonical_statuses[canonical]["zero_budget_no_attempt"] += 1
                    canonical_implicit_zeros[canonical].append(identifier)
                else:
                    canonical_nulls[canonical].append(identifier)
                continue
            count, diagnostics = baseline_policy_count(methods[method_name])
            if count is None:
                if row_budget == 0:
                    canonical_values[canonical].append(0.0)
                    canonical_sources[canonical]["implicit_zero_budget"] += 1
                    canonical_statuses[canonical]["zero_budget_no_attempt"] += 1
                    canonical_implicit_zeros[canonical].append(identifier)
                else:
                    canonical_nulls[canonical].append(identifier)
                continue
            canonical_values[canonical].append(float(count))
            canonical_statuses[canonical][diagnostics["status"]] += 1
            canonical_sources[canonical][diagnostics["source"]] += 1
            if (
                row_budget is not None
                and diagnostics["matched_token_budget"] is not None
                and diagnostics["matched_token_budget"] != row_budget
            ):
                budget_mismatches[canonical].append(identifier)
        query_random: list[float] = []
        for method_name in random_per_seed:
            method = methods.get(method_name)
            if not isinstance(method, dict):
                if row_budget == 0:
                    random_per_seed[method_name].append(0.0)
                    query_random.append(0.0)
                    random_implicit_zeros[method_name].append(identifier)
                else:
                    random_nulls[method_name].append(identifier)
                continue
            count, diagnostics = baseline_policy_count(method)
            if count is None:
                if row_budget == 0:
                    random_per_seed[method_name].append(0.0)
                    query_random.append(0.0)
                    random_implicit_zeros[method_name].append(identifier)
                else:
                    random_nulls[method_name].append(identifier)
                continue
            value = float(count)
            random_per_seed[method_name].append(value)
            query_random.append(value)
            if (
                row_budget is not None
                and diagnostics["matched_token_budget"] is not None
                and diagnostics["matched_token_budget"] != row_budget
            ):
                budget_mismatches[method_name].append(identifier)
        if len(query_random) == 5:
            random_values.append(statistics.fmean(query_random))

    for method in BASELINE_ALIASES:
        result["methods"][method] = {
            **stats(canonical_values[method]),
            "null_query_ids": canonical_nulls[method],
            "count_sources": dict(sorted(canonical_sources[method].items())),
            "status_histogram": dict(sorted(canonical_statuses[method].items())),
            "implicit_zero_budget_ids": canonical_implicit_zeros[method],
            "matched_budget_mismatch_ids": budget_mismatches[method],
        }
    result["methods"]["random"] = {
        **stats(random_values),
        "null_query_ids": sorted(set().union(*map(set, random_nulls.values()))),
        "per_seed": {
            name: {
                **stats(values),
                "null_query_ids": random_nulls[name],
                "implicit_zero_budget_ids": random_implicit_zeros[name],
                "matched_budget_mismatch_ids": budget_mismatches[name],
            }
            for name, values in random_per_seed.items()
        },
    }
    result["all_methods_have_1000_values"] = all(
        method["values"] == 1000 and not method["null_query_ids"]
        for method in result["methods"].values()
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--datasets", nargs="*", default=list(DATASETS))
    args = parser.parse_args()
    unknown = sorted(set(args.datasets).difference(DATASETS))
    if unknown:
        parser.error(f"unsupported datasets: {unknown}")
    payload = {
        "schema": "causalityrag.table3_token_count_audit.v1",
        "token_count_contract": {
            "reflow_success": "selected-token count of the verified-flip attempt",
            "reflow_failure": "maximum selected-token count across saved attempts",
            "baseline": "saved n_modified_tokens, with selected IDs then matched budget as fallbacks",
            "random": "per-query mean across five independently audited seeds",
            "no_attempt": "explicit numeric zero; never null",
        },
        "datasets": {
            dataset: audit_dataset(args.root, dataset)
            for dataset in args.datasets
        },
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
