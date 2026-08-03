#!/usr/bin/env python3
"""Refine the smallest baseline budget that matches ReFlow's Acc-CFR."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


METHOD_FILES = {
    "mirage": "mirage_top5_1000.jsonl",
    "arc_jsd": "arc_jsd_top5_1000.jsonl",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_points(
    release: Path,
    dataset_dir: Path,
    method: str,
    budgets: list[int],
    endpoint: str,
    workers: int,
    pool_path: str,
    pool_sha: str,
    paths: dict[str, Path],
    seed_path: Path | None,
) -> dict:
    out_dir = dataset_dir / "audits/baseline_budget_to_reflow/exact_points_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    budget_slug = f"k{budgets[0]}" if len(budgets) == 1 else (
        f"k{budgets[0]}_{budgets[-1]}"
    )
    rows_path = out_dir / f"{method}_{budget_slug}_1000.jsonl"
    summary_path = out_dir / f"{method}_{budget_slug}_1000.summary.json"
    if summary_path.is_file():
        summary = load_json(summary_path)
        if all(
            str(budget) in summary.get("methods", {}).get(method, {})
            for budget in budgets
        ):
            return summary

    command = [
        sys.executable,
        str(release / "scripts/evaluate_fixed_budget_acc.py"),
        "--input", str(paths["input"]),
        "--units-cache", str(paths["units"]),
        "--frontier", str(paths["frontier"]),
        "--scores", f"{method}={paths[method]}",
        "--shared-pool", pool_path,
        "--expected-pool-sha256", pool_sha,
        "--out", str(rows_path),
        "--summary-out", str(summary_path),
        "--budgets", ",".join(str(budget) for budget in budgets),
        "--skip-reflow",
        "--n", "1000",
        "--k", "5",
        "--workers", str(workers),
        "--llm-base-url", endpoint,
        "--llm-model", "qwen2.5-7b",
        "--reader-max-tokens", "128",
    ]
    if seed_path is not None and seed_path.is_file():
        command.extend(["--seed-results", str(seed_path)])
    subprocess.run(command, cwd=release, check=True)
    return load_json(summary_path)


def refine_dataset(
    release: Path,
    out_root: Path,
    dataset: str,
    endpoint: str,
    workers: int,
    max_budget: int,
    budget_block: int,
) -> None:
    dataset_dir = out_root / dataset
    seed_path = dataset_dir / "audits/fixed_budget_acc/summary_1000.json"
    factual_path = dataset_dir / "audits/final_top10pool_k5/factual_metrics_1000.json"
    merged = load_json(seed_path)
    prior_summaries = [merged]
    summary_root = dataset_dir / "audits/baseline_budget_to_reflow"
    prior_summaries.extend(
        load_json(summary_path)
        for summary_path in sorted(summary_root.rglob("summary*.json"))
    )
    prior_summaries.extend(
        load_json(summary_path)
        for summary_path in sorted(summary_root.rglob("*.summary.json"))
    )
    for method in METHOD_FILES:
        merged["methods"][method] = {}
    for prior in prior_summaries:
        if int(prior.get("clean_acc_queries", -1)) != int(
            merged["clean_acc_queries"]
        ):
            continue
        uses_available_prefix = "use all available" in str(
            prior.get("budget_contract", "")
        )
        for method in METHOD_FILES:
            for budget, point in prior.get("methods", {}).get(method, {}).items():
                if uses_available_prefix or int(point["executed_queries"]) == int(
                    point["clean_acc_queries"]
                ):
                    merged["methods"][method][budget] = point
    factual = load_json(factual_path)
    target_count = int(factual["methods"]["reflow"]["acc_flip_count"])
    pool_path = str(merged["shared_pool"])
    pool_sha = str(merged["shared_pool_sha256"])
    paths = {
        "input": dataset_dir / "retrieval/top10_1000.jsonl",
        "units": dataset_dir / "inputs/token_units_top10_1000.jsonl",
        "frontier": dataset_dir / "methods/reflow/frontier_top5_1000.jsonl",
        "mirage": dataset_dir / "methods/baselines/mirage_top5_1000.jsonl",
        "arc_jsd": dataset_dir / "methods/baselines/arc_jsd_top5_1000.jsonl",
    }
    if dataset == "hotpotqa":
        paths.update({
            "frontier": dataset_dir
            / "audits/final_top10pool_k5/reflow_frontier_1000.jsonl",
            "mirage": dataset_dir
            / "methods/baselines/mirage_gamma1_setting_q1000_scores.jsonl",
            "arc_jsd": dataset_dir
            / "methods/baselines/arc_jsd_projection_1000.jsonl",
        })
        pool_path = str(
            dataset_dir
            / "replacements/shared_pool_top10_v1/shared_pool_top10_final.jsonl"
        )
    seed_path = (
        dataset_dir
        / "audits/baseline_budget_to_reflow/reusable_v1_seed_1000.jsonl"
    )

    for method in METHOD_FILES:
        curve = merged["methods"][method]
        crossing = None
        for block_start in range(1, max_budget + 1, budget_block):
            block_end = min(max_budget, block_start + budget_block - 1)
            missing = [
                budget for budget in range(block_start, block_end + 1)
                if str(budget) not in curve
            ]
            if missing:
                summary = run_points(
                    release,
                    dataset_dir,
                    method,
                    missing,
                    endpoint,
                    workers,
                    pool_path,
                    pool_sha,
                    paths,
                    seed_path,
                )
                curve.update(summary["methods"][method])
            for budget in range(block_start, block_end + 1):
                point = curve[str(budget)]
                flips = int(point["acc_flip_count"])
                print(
                    f"MATCH_PROGRESS dataset={dataset} method={method} "
                    f"budget={budget} flips={flips}/{target_count}",
                    flush=True,
                )
                if flips >= target_count:
                    crossing = budget
                    break
            if crossing is not None:
                break

        if crossing is None:
            print(
                f"MATCH_NOT_REACHED dataset={dataset} method={method} "
                f"max_budget={max_budget}",
                flush=True,
            )
        else:
            print(
                f"MATCH_DONE dataset={dataset} method={method} "
                f"budget={crossing}",
                flush=True,
            )

    merged["budgets"] = sorted(
        {int(value) for method in METHOD_FILES for value in merged["methods"][method]}
    )
    merged["exact_integer_refinement"] = True
    merged_path = (
        dataset_dir
        / "audits/baseline_budget_to_reflow/summary_exact_v2_1000.json"
    )
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    merged_path.write_text(
        json.dumps(merged, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--max-budget", type=int, default=128)
    parser.add_argument("--budget-block", type=int, default=8)
    args = parser.parse_args()

    release = Path(args.release)
    out_root = Path(args.out_root)
    for dataset in args.datasets.split(","):
        refine_dataset(
            release,
            out_root,
            dataset.strip(),
            args.endpoint,
            args.workers,
            args.max_budget,
            args.budget_block,
        )


if __name__ == "__main__":
    main()
