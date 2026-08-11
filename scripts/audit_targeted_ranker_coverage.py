#!/usr/bin/env python3
"""Audit ranker coverage against repaired per-query ReFlow budgets."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.io import load_records, record_id
from causalityrag.shared_replacement_pool import FrozenSharedReplacementPool
from scripts.evaluate_matched_budget_baselines import ranked_ids


DATASETS = (
    "hotpotqa",
    "timeqa",
    "finqa",
    "musique",
    "quartz",
    "triviaqa",
    "2wiki",
)


def score_paths(base: Path, dataset: str) -> dict[str, Path]:
    paths = {
        "attention": base / "methods/baselines/attention_top5_1000.jsonl",
        "gradient_x_input": (
            base / "methods/baselines/gradient_x_input_top5_1000.jsonl"
        ),
        "integrated_gradients": (
            base / "methods/baselines/integrated_gradients_top5_1000.jsonl"
        ),
        "mirage": base / "methods/baselines/mirage_top5_1000.jsonl",
        "arc_jsd": base / "methods/baselines/arc_jsd_top5_1000.jsonl",
    }
    if dataset == "hotpotqa":
        paths.update({
            "attention": (
                base / "methods/baselines/attention_prompt_last_1000.jsonl"
            ),
            "mirage": (
                base
                / "methods/baselines/mirage_gamma1_setting_q1000_scores.jsonl"
            ),
            "arc_jsd": (
                base / "methods/baselines/arc_jsd_projection_1000.jsonl"
            ),
        })
    return paths


def pool_path(base: Path, dataset: str) -> Path:
    name = (
        "shared_pool_top10_final.jsonl"
        if dataset in {"hotpotqa", "triviaqa"}
        else "shared_pool.jsonl"
    )
    return base / "replacements/shared_pool_top10_v1" / name


def indexed(path: Path) -> dict[str, dict]:
    rows = load_records(path)
    result = {record_id(row): row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"{path}: duplicate IDs")
    return result


def audit_dataset(source_root: Path, repair_root: Path, dataset: str) -> dict:
    base = source_root / dataset
    repair = repair_root / dataset
    manifest = json.loads(
        (repair / "resume/manifest.json").read_text(encoding="utf-8")
    )
    target_ids = [str(value) for value in manifest["target_ids"]]
    reflow = indexed(repair / "results_1000.jsonl")
    pool = FrozenSharedReplacementPool(str(pool_path(base, dataset)))
    methods = {}
    for method, path in score_paths(base, dataset).items():
        scores = indexed(path)
        insufficient = []
        for identifier in target_ids:
            budget = int(reflow[identifier].get("n_modified_tokens", 0))
            eligible_ranking = [
                unit_id
                for unit_id in ranked_ids(scores[identifier])
                if pool.is_eligible(unit_id)
            ]
            if len(eligible_ranking) < budget:
                insufficient.append({
                    "id": identifier,
                    "budget": budget,
                    "eligible_ranked_tokens": len(eligible_ranking),
                    "score_status": str(scores[identifier].get("status", "")),
                })
        methods[method] = {
            "source": str(path.resolve()),
            "target_queries": len(target_ids),
            "covered_queries": len(target_ids) - len(insufficient),
            "insufficient_queries": insufficient,
        }
    return {
        "dataset": dataset,
        "target_queries": len(target_ids),
        "methods": methods,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--repair-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--datasets", nargs="*", default=list(DATASETS))
    args = parser.parse_args()

    rows = [
        audit_dataset(args.source_root, args.repair_root, dataset)
        for dataset in args.datasets
    ]
    result = {
        "schema": "causalityrag.targeted_ranker_coverage_audit.v1",
        "datasets": rows,
        "insufficient_query_methods": sum(
            len(method["insufficient_queries"])
            for row in rows
            for method in row["methods"].values()
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
