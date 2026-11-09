#!/usr/bin/env python3
"""Strictly audit and aggregate the eight-dataset b=20 fidelity run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


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
DISPLAY_NAMES = {
    "hotpotqa": "HQA",
    "timeqa": "TQA",
    "finqa": "FQA",
    "musique": "MSQ",
    "quartz": "QTZ",
    "triviaqa": "TrQA",
    "2wiki": "2Wiki",
    "popqa": "PQA",
}
SOURCE_PATHS = {
    "input": "inputs/retrieval.jsonl",
    "units": "inputs/units.jsonl",
    "graphs": "graph/contribution_graph.jsonl",
    "shared_pool": "inputs/shared_pool.jsonl",
}
T_CRITICAL_DF7_95 = 2.364624251


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
    return rows


def close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)


def audit_dataset(root: Path, dataset: str, budget: int) -> dict:
    source_root = root / "source" / "k5" / dataset
    output_root = root / "out" / dataset / "fidelity_b20"
    probes_path = output_root / "probes_1000.jsonl"
    summary_path = output_root / "summary.json"

    if not (output_root / "COMPLETE").is_file():
        raise ValueError(f"{dataset}: COMPLETE marker missing")
    rows = load_jsonl(probes_path)
    summary = load_json(summary_path)
    if len(rows) != 1000:
        raise ValueError(f"{dataset}: expected 1000 probes, found {len(rows)}")

    task_ids = [str(row.get("task_id", "")) for row in rows]
    if len(set(task_ids)) != 1000 or any(not value for value in task_ids):
        raise ValueError(f"{dataset}: task IDs are missing or duplicated")

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        query_id = str(row.get("id", ""))
        if not query_id:
            raise ValueError(f"{dataset}: empty query ID")
        grouped[query_id].append(row)
        if row.get("budget") != budget:
            raise ValueError(f"{dataset}: non-{budget} budget probe")
        if row.get("task_id") != f"{query_id}|{budget}|{row.get('trial')}":
            raise ValueError(f"{dataset}: malformed task ID")
        selected_ids = row.get("selected_ids")
        if not isinstance(selected_ids, list) or len(selected_ids) != budget:
            raise ValueError(f"{dataset}: selected set does not contain {budget} tokens")
        if len(set(selected_ids)) != budget:
            raise ValueError(f"{dataset}: duplicate token in selected set")
        valid_answer = row.get("valid_answer") is True
        protocol_error = row.get("protocol_error")
        if valid_answer and protocol_error not in ("", None):
            raise ValueError(f"{dataset}: valid answer has a protocol error")
        if not valid_answer and not protocol_error:
            raise ValueError(f"{dataset}: invalid answer lacks a failure status")
        if not isinstance(row.get("answer_flip"), bool):
            raise ValueError(f"{dataset}: non-boolean answer_flip")
        residual = row.get("residual_flow_ratio")
        if not isinstance(residual, (int, float)) or not math.isfinite(residual):
            raise ValueError(f"{dataset}: invalid residual-flow ratio")

    if len(grouped) != 100:
        raise ValueError(f"{dataset}: expected 100 query IDs, found {len(grouped)}")
    for query_id, query_rows in grouped.items():
        trials = sorted(int(row["trial"]) for row in query_rows)
        if trials != list(range(10)):
            raise ValueError(f"{dataset}/{query_id}: trials are not exactly 0..9")

    valid_reader_calls = sum(row.get("valid_answer") is True for row in rows)
    protocol_violations = len(rows) - valid_reader_calls
    expected_summary = {
        "queries": 100,
        "budgets": [budget],
        "trials_per_query_budget": 10,
        "reader_executions": 1000,
        "valid_answer_executions": valid_reader_calls,
        "protocol_violations": protocol_violations,
    }
    for key, expected in expected_summary.items():
        if summary.get(key) != expected:
            raise ValueError(
                f"{dataset}: summary {key}={summary.get(key)!r}, expected {expected!r}"
            )

    deciles = summary.get("deciles_low_to_high_residual", [])
    if len(deciles) != 10:
        raise ValueError(f"{dataset}: expected 10 summary deciles")
    ranked_rows: dict[int, list[dict]] = defaultdict(list)
    for query_rows in grouped.values():
        ordered = sorted(
            query_rows,
            key=lambda row: (float(row["residual_flow_ratio"]), int(row["trial"])),
        )
        for rank, row in enumerate(ordered, start=1):
            ranked_rows[rank].append(row)
    flip_rates = []
    residual_means = []
    distance_means = []
    for index, summary_row in enumerate(deciles):
        chunk = ranked_rows[index + 1]
        valid_chunk = [row for row in chunk if row.get("valid_answer") is True]
        actual_flip = mean(float(row["answer_flip"]) for row in valid_chunk)
        actual_residual = mean(float(row["residual_flow_ratio"]) for row in chunk)
        actual_distance = mean(float(row["answer_distance"]) for row in valid_chunk)
        if summary_row.get("decile") != index + 1:
            raise ValueError(f"{dataset}: decile numbering mismatch")
        if (
            summary_row.get("count") != 100
            or summary_row.get("valid_answer_count") != len(valid_chunk)
        ):
            raise ValueError(f"{dataset}: decile count mismatch")
        checks = (
            ("answer_flip_rate", actual_flip),
            ("mean_residual_flow_ratio", actual_residual),
            ("mean_answer_distance", actual_distance),
        )
        for key, actual in checks:
            if not close(float(summary_row[key]), actual):
                raise ValueError(f"{dataset}: independently recomputed {key} mismatch")
        flip_rates.append(actual_flip)
        residual_means.append(actual_residual)
        distance_means.append(actual_distance)

    actual_source_sha = {
        key: sha256(source_root / relative_path)
        for key, relative_path in SOURCE_PATHS.items()
    }
    if summary.get("source_sha256") != actual_source_sha:
        raise ValueError(f"{dataset}: source SHA mismatch")

    return {
        "display_name": DISPLAY_NAMES[dataset],
        "queries": 100,
        "reader_calls": 1000,
        "valid_reader_calls": valid_reader_calls,
        "protocol_violations": protocol_violations,
        "ranks": list(range(1, 11)),
        "answer_flip_rate": flip_rates,
        "mean_residual_flow_ratio": residual_means,
        "mean_answer_distance": distance_means,
        "mean_within_query_budget_spearman_residual_vs_distance": summary[
            "mean_within_query_budget_spearman_residual_vs_distance"
        ],
        "auroc_removed_flow_predicting_answer_flip": summary[
            "auroc_removed_flow_predicting_answer_flip"
        ],
        "equal_budget_pairwise_concordance": summary[
            "equal_budget_pairwise_concordance"
        ],
        "comparable_pairs": summary["comparable_pairs"],
        "source_sha256": actual_source_sha,
        "artifact_sha256": {
            "probes_1000": sha256(probes_path),
            "summary": sha256(summary_path),
        },
        "audit": {
            "complete_marker": True,
            "probe_rows": 1000,
            "unique_task_ids": 1000,
            "unique_query_ids": 100,
            "ten_trials_per_query": True,
            "selected_tokens_per_trial": budget,
            "invalid_outputs_excluded_from_metric_numerators_and_denominators": True,
            "summary_matches_independent_recomputation": True,
            "source_sha_matches_files": True,
        },
    }


def macro_statistics(datasets: dict[str, dict]) -> dict:
    curves = [datasets[name]["answer_flip_rate"] for name in DATASETS]
    macro = []
    lower = []
    upper = []
    for rank in range(10):
        values = [curve[rank] for curve in curves]
        center = mean(values)
        half_width = T_CRITICAL_DF7_95 * stdev(values) / math.sqrt(len(values))
        macro.append(center)
        lower.append(max(0.0, center - half_width))
        upper.append(min(1.0, center + half_width))
    return {
        "ranks": list(range(1, 11)),
        "answer_flip_rate": macro,
        "ci95_across_datasets_lower": lower,
        "ci95_across_datasets_upper": upper,
        "ci_method": "two-sided t interval across eight dataset-level rates (df=7)",
        "rank1_to_rank10_ratio": macro[0] / macro[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--budget", type=int, default=20)
    args = parser.parse_args()

    input_audit_path = args.root / "input_audit.json"
    input_audit = load_json(input_audit_path)
    input_rows = input_audit.get("datasets", [])
    if [row.get("dataset") for row in input_rows] != list(DATASETS):
        raise ValueError("input audit dataset order mismatch")
    for row in input_rows:
        if row.get("rows") != 1000 or row.get("unique_ids") != 1000:
            raise ValueError(f"{row.get('dataset')}: input row/ID audit failed")
        if row.get("ordered_ids_aligned") is not True:
            raise ValueError(f"{row.get('dataset')}: input alignment audit failed")

    datasets = {
        dataset: audit_dataset(args.root, dataset, args.budget)
        for dataset in DATASETS
    }
    valid_reader_calls = sum(row["valid_reader_calls"] for row in datasets.values())
    protocol_violations = sum(row["protocol_violations"] for row in datasets.values())
    output = {
        "schema": "causalityrag.residual_flow_b20_all8.v1",
        "settings": {
            "retrieval_depth": 5,
            "fixed_token_budget": args.budget,
            "queries_per_dataset": 100,
            "trials_per_query": 10,
            "reader_calls_per_dataset": 1000,
            "total_reader_calls": 8000,
            "selection": "uniform legal token sets sampled before residual flow is computed",
        },
        "dataset_order": list(DATASETS),
        "datasets": datasets,
        "macro_average": macro_statistics(datasets),
        "input_audit": input_rows,
        "input_audit_sha256": sha256(input_audit_path),
        "audit": {
            "datasets_complete": 8,
            "probe_rows": 8000,
            "unique_task_ids_within_each_dataset": True,
            "valid_reader_calls": valid_reader_calls,
            "protocol_violations": protocol_violations,
            "invalid_outputs_excluded_from_metric_numerators_and_denominators": True,
            "all_summary_metrics_independently_recomputed": True,
            "all_source_sha_verified": True,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output["audit"], sort_keys=True))
    print(json.dumps(output["macro_average"], sort_keys=True))


if __name__ == "__main__":
    main()
