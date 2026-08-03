#!/usr/bin/env python3
"""Evaluate ReFlow intervention size as a query-level fragility signal."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.io import load_records, record_id
from causalityrag.reader import ReaderClient, answers_exact_match
from causalityrag.revision import apply_token_replacements
from causalityrag.shared_replacement_pool import (
    FrozenSharedReplacementPool,
    stable_shared_candidate,
)
from causalityrag.token_units import units_from_cache_row


FACTUAL_TYPES = {
    "CARDINAL", "DATE", "EVENT", "FAC", "GPE", "LANGUAGE", "LAW",
    "LOC", "MONEY", "NORP", "ORDINAL", "ORG", "PERCENT", "PERSON",
    "PRODUCT", "QUANTITY", "TIME", "WORK_OF_ART",
}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")


def stable_ids(unit_ids: list[str], *, seed: int, query_id: str) -> list[str]:
    return sorted(
        unit_ids,
        key=lambda unit_id: hashlib.sha256(
            f"{seed}\0{query_id}\0{unit_id}".encode("utf-8")
        ).digest(),
    )


def prepare(args: argparse.Namespace) -> None:
    retrieval = load_records(args.retrieval)
    units = load_records(args.units)
    reflow = load_records(args.reflow)
    if len({len(retrieval), len(units), len(reflow)}) != 1:
        raise ValueError("retrieval, units, and ReFlow rows must be aligned")
    pool = FrozenSharedReplacementPool(args.shared_pool)
    rows = []
    for record, unit_row, reflow_row in zip(retrieval, units, reflow):
        query_id = record_id(record)
        if query_id != record_id(unit_row) or query_id != record_id(reflow_row):
            raise ValueError(f"misaligned row for {query_id}")
        if not bool(reflow_row.get("clean_correct")):
            continue
        token_units = units_from_cache_row(record, unit_row, k=args.k)
        by_id = {str(unit["unit_id"]): unit for unit in token_units}
        eligible = [
            unit_id
            for unit_id, unit in by_id.items()
            if pool.is_eligible(unit_id)
            and str(unit.get("type", "")).upper() in FACTUAL_TYPES
        ]
        if len(eligible) < args.budget:
            continue
        for seed in range(args.seeds):
            selected_ids = stable_ids(
                eligible, seed=args.seed_offset + seed, query_id=query_id
            )[: args.budget]
            pool_rows = pool.require(selected_ids)
            replacements = {
                unit_id: stable_shared_candidate(
                    pool_rows[unit_id]["candidates"],
                    unit_id=unit_id,
                    seed=args.seed_offset + seed,
                )
                for unit_id in selected_ids
            }
            revision = apply_token_replacements(
                record,
                [by_id[unit_id] for unit_id in selected_ids],
                replacements,
                k=args.k,
            )
            if int(revision.get("n_edits", 0)) != len(selected_ids):
                raise RuntimeError(f"unexecutable stress edit for {query_id}")
            rows.append({
                "id": query_id,
                "seed": seed,
                "question": str(record.get("question", "")),
                "gold_answer": str(record.get("answer", "")),
                "clean_answer": str(reflow_row.get("clean_answer", "")),
                "reflow_size": int(reflow_row.get("n_modified_tokens", 0)),
                "reflow_verified": bool(reflow_row.get("verified_flip")),
                "selected_ids": selected_ids,
                "selected_tokens": [by_id[unit_id]["text"] for unit_id in selected_ids],
                "retrieved": revision["edited_contexts"],
            })
    write_jsonl(Path(args.out), rows)
    print(json.dumps({
        "queries": len({row["id"] for row in rows}),
        "stress_trials": len(rows),
        "seeds": args.seeds,
        "tokens_per_trial": args.budget,
    }, indent=2))


def evaluate(args: argparse.Namespace) -> None:
    rows = load_records(args.input)
    reader = ReaderClient(args.llm_base_url, args.llm_model)

    def run(row: dict) -> dict:
        answer = reader.answer(str(row["question"]), list(row["retrieved"]))
        return {
            key: value for key, value in row.items() if key != "retrieved"
        } | {
            "stress_answer": answer,
            "answer_changed": not answers_exact_match(answer, str(row["clean_answer"])),
            "correctness_flip": not answers_exact_match(answer, str(row["gold_answer"])),
        }

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run, row) for row in rows]
        for index, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if index % 100 == 0 or index == len(futures):
                print(f"[triage] {index}/{len(futures)}", flush=True)
    results.sort(key=lambda row: (str(row["id"]), int(row["seed"])))
    write_jsonl(Path(args.out), results)


def resilience_group(size: int, *, verified: bool) -> str:
    if not verified:
        return ">=5"
    if size <= 1:
        return "1"
    if size == 2:
        return "2"
    if size <= 4:
        return "3-4"
    return ">=5"


def auc(scores: list[float], labels: list[bool]) -> float | None:
    positives = [score for score, label in zip(scores, labels) if label]
    negatives = [score for score, label in zip(scores, labels) if not label]
    if not positives or not negatives:
        return None
    wins = sum(
        1.0 if positive > negative else 0.5 if positive == negative else 0.0
        for positive in positives
        for negative in negatives
    )
    return wins / (len(positives) * len(negatives))


def summarize(args: argparse.Namespace) -> None:
    rows = load_records(args.input)
    by_query: dict[str, list[dict]] = {}
    for row in rows:
        by_query.setdefault(str(row["id"]), []).append(row)
    grouped: dict[str, list[dict]] = {key: [] for key in ["1", "2", "3-4", ">=5"]}
    query_rows = []
    for query_id, trials in by_query.items():
        size = int(trials[0]["reflow_size"])
        verified = bool(trials[0]["reflow_verified"])
        group = resilience_group(size, verified=verified)
        grouped[group].extend(trials)
        query_rows.append({
            "id": query_id,
            "reflow_size": size,
            "reflow_verified": verified,
            "group": group,
            "any_correctness_flip": any(row["correctness_flip"] for row in trials),
            "trial_flip_rate": statistics.mean(
                float(row["correctness_flip"]) for row in trials
            ),
        })
    max_verified_size = max(
        row["reflow_size"] for row in query_rows if row["reflow_verified"]
    )
    risk_scores = [
        -float(row["reflow_size"])
        if row["reflow_verified"] else -float(max_verified_size + 1)
        for row in query_rows
    ]
    rng = random.Random(args.bootstrap_seed)
    group_bootstrap = {group: [] for group in grouped}
    auc_bootstrap = []
    for _ in range(args.bootstrap_samples):
        sample = [query_rows[rng.randrange(len(query_rows))] for _ in query_rows]
        for group in grouped:
            values = [row["trial_flip_rate"] for row in sample if row["group"] == group]
            if values:
                group_bootstrap[group].append(statistics.mean(values))
        sampled_scores = [
            -float(row["reflow_size"])
            if row["reflow_verified"] else -float(max_verified_size + 1)
            for row in sample
        ]
        sampled_auc = auc(
            sampled_scores,
            [bool(row["any_correctness_flip"]) for row in sample],
        )
        if sampled_auc is not None:
            auc_bootstrap.append(sampled_auc)

    def interval(values: list[float]) -> list[float]:
        ordered = sorted(values)
        return [
            ordered[int(0.025 * (len(ordered) - 1))],
            ordered[int(0.975 * (len(ordered) - 1))],
        ]

    summary = {
        "queries": len(query_rows),
        "stress_trials": len(rows),
        "groups": {
            group: {
                "queries": len({str(row["id"]) for row in trials}),
                "trials": len(trials),
                "correctness_flip_rate": (
                    statistics.mean(float(row["correctness_flip"]) for row in trials)
                    if trials else None
                ),
                "answer_change_rate": (
                    statistics.mean(float(row["answer_changed"]) for row in trials)
                    if trials else None
                ),
                "query_cluster_bootstrap_95ci": interval(group_bootstrap[group]),
            }
            for group, trials in grouped.items()
        },
        "query_level_auroc": auc(
            risk_scores,
            [bool(row["any_correctness_flip"]) for row in query_rows],
        ),
        "query_level_auroc_bootstrap_95ci": interval(auc_bootstrap),
        "overall_correctness_flip_rate": statistics.mean(
            float(row["correctness_flip"]) for row in rows
        ),
    }
    Path(args.out).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--retrieval", required=True)
    prepare_parser.add_argument("--units", required=True)
    prepare_parser.add_argument("--reflow", required=True)
    prepare_parser.add_argument("--shared-pool", required=True)
    prepare_parser.add_argument("--out", required=True)
    prepare_parser.add_argument("--k", type=int, default=5)
    prepare_parser.add_argument("--budget", type=int, default=3)
    prepare_parser.add_argument("--seeds", type=int, default=5)
    prepare_parser.add_argument("--seed-offset", type=int, default=101)
    prepare_parser.set_defaults(function=prepare)

    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--input", required=True)
    evaluate_parser.add_argument("--out", required=True)
    evaluate_parser.add_argument("--workers", type=int, default=48)
    evaluate_parser.add_argument("--llm-base-url", default="http://127.0.0.1:8000/v1")
    evaluate_parser.add_argument("--llm-model", default="qwen2.5-7b")
    evaluate_parser.set_defaults(function=evaluate)

    summary_parser = subparsers.add_parser("summarize")
    summary_parser.add_argument("--input", required=True)
    summary_parser.add_argument("--out", required=True)
    summary_parser.add_argument("--bootstrap-samples", type=int, default=5000)
    summary_parser.add_argument("--bootstrap-seed", type=int, default=20260803)
    summary_parser.set_defaults(function=summarize)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
