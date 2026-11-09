#!/usr/bin/env python3
"""Evaluate Token Shapley on the frozen sentence-sufficiency population."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.io import load_records, record_id
from causalityrag.reader import ReaderClient, answer_token_f1, answers_exact_match

from run_sentence_sufficiency_case_study import (
    build_examples,
    context_token_count,
    index_rows,
    percentile_interval,
    ranked_ids,
    ranked_sentence_ids,
    reader_answer,
    sentence_contexts,
)


METHOD = "Token Shapley"


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def index_rankings(paths: list[str]) -> dict[str, dict]:
    indexed: dict[str, dict] = {}
    for path in paths:
        rows = load_records(path)
        for row in rows:
            query_id = record_id(row)
            if not query_id:
                raise ValueError(f"missing ranking ID in {path}")
            if query_id in indexed:
                raise ValueError(f"duplicate Token Shapley ID: {query_id}")
            indexed[query_id] = row
    return indexed


def attach_token_shapley(
    examples: list[dict],
    *,
    retrieval_path: str,
    units_path: str,
    ranking_paths: list[str],
) -> dict[str, Any]:
    retrieval = index_rows(retrieval_path)
    units = index_rows(units_path)
    rankings = index_rankings(ranking_paths)
    failures: dict[str, list[str]] = {
        "missing_ranking": [],
        "ranking_status": [],
        "sentence_budget_shortfall": [],
        "empty_context": [],
    }

    for example in examples:
        query_id = example["id"]
        ranking = rankings.get(query_id)
        if ranking is None:
            failures["missing_ranking"].append(query_id)
            continue
        if ranking.get("status") != "ok":
            failures["ranking_status"].append(query_id)
            continue
        units_row = units[query_id]
        sentence_by_unit = {
            str(unit.get("unit_id", "")): str(unit.get("sentence_id", ""))
            for unit in units_row.get("units", [])
        }
        selected = ranked_sentence_ids(
            ranked_ids(ranking), sentence_by_unit, example["sentence_budget"]
        )
        if len(selected) != example["sentence_budget"]:
            failures["sentence_budget_shortfall"].append(query_id)
            continue
        contexts = sentence_contexts(selected, units_row, retrieval[query_id])
        if len(contexts) != example["sentence_budget"]:
            failures["empty_context"].append(query_id)
            continue
        example["token_shapley_sentence_ids"] = selected
        example["token_shapley_contexts"] = contexts

    failed = {reason: ids for reason, ids in failures.items() if ids}
    if failed:
        preview = "; ".join(
            f"{reason}={len(ids)} ({', '.join(ids[:3])})"
            for reason, ids in failed.items()
        )
        raise ValueError(f"Token Shapley cannot match the frozen population: {preview}")
    return {
        "ranking_rows": len(rankings),
        "aligned_queries": len(examples),
        "ranking_status_counts": {
            status: sum(1 for row in rankings.values() if row.get("status") == status)
            for status in sorted({str(row.get("status")) for row in rankings.values()})
        },
    }


def evaluate_example(example: dict, reader: ReaderClient) -> dict:
    contexts = example["token_shapley_contexts"]
    prediction = reader_answer(reader, example["question"], contexts)
    return {
        "id": example["id"],
        "index": example["index"],
        "question": example["question"],
        "gold_answer": example["gold_answer"],
        "sentence_budget": example["sentence_budget"],
        "selected_sentence_ids": example["token_shapley_sentence_ids"],
        "prediction": prediction,
        "metrics": {
            "em": float(answers_exact_match(prediction, example["gold_answer"])),
            "f1": answer_token_f1(prediction, example["gold_answer"]),
        },
        "context_stats": {
            "sentences": len(contexts),
            "tokens": context_token_count(contexts),
        },
    }


def summarize(
    rows: list[dict],
    *,
    population: dict,
    alignment: dict,
    input_manifest: dict,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict:
    rng = random.Random(bootstrap_seed)
    method: dict[str, Any] = {}
    for metric in ("em", "f1"):
        values = [float(row["metrics"][metric]) for row in rows]
        bootstraps = [
            statistics.fmean(values[rng.randrange(len(values))] for _ in values)
            for _ in range(bootstrap_samples)
        ]
        method[metric] = {
            "mean": statistics.fmean(values),
            "ci95": percentile_interval(bootstraps),
        }
    method["mean_context_tokens"] = statistics.fmean(
        float(row["context_stats"]["tokens"]) for row in rows
    )
    method["mean_sentences"] = statistics.fmean(
        float(row["context_stats"]["sentences"]) for row in rows
    )
    return {
        "schema": "causalityrag.sentence_sufficiency.ts.v1",
        "population": population,
        "alignment": alignment,
        "queries": len(rows),
        "method": METHOD,
        "metrics": method,
        "bootstrap": {
            "samples": bootstrap_samples,
            "seed": bootstrap_seed,
            "unit": "query",
            "interval": "percentile_95",
        },
        "inputs": input_manifest,
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval", required=True)
    parser.add_argument("--units", required=True)
    parser.add_argument("--reflow", required=True)
    parser.add_argument("--mirage", required=True)
    parser.add_argument("--arc-jsd", required=True)
    parser.add_argument("--token-shapley", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--llm-base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--llm-model", default="qwen2.5-7b")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260804)
    args = parser.parse_args()

    examples, population = build_examples(args)
    if not examples:
        raise ValueError("no aligned ReFlow-supported examples")
    alignment = attach_token_shapley(
        examples,
        retrieval_path=args.retrieval,
        units_path=args.units,
        ranking_paths=args.token_shapley,
    )
    manifest = {
        name: {"path": path, "sha256": sha256_file(path)}
        for name, path in {
            "retrieval": args.retrieval,
            "units": args.units,
            "reflow": args.reflow,
            "mirage": args.mirage,
            "arc_jsd": args.arc_jsd,
        }.items()
    }
    manifest["token_shapley"] = [
        {"path": path, "sha256": sha256_file(path)}
        for path in args.token_shapley
    ]
    if args.prepare_only:
        print(json.dumps({
            "population": population,
            "alignment": alignment,
            "inputs": manifest,
        }, ensure_ascii=False, indent=2))
        return

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = index_rows(str(output_path)) if output_path.exists() else {}
    expected_ids = {example["id"] for example in examples}
    unexpected = sorted(set(completed) - expected_ids)
    if unexpected:
        raise ValueError(f"output contains unexpected IDs: {unexpected[:3]}")
    pending = [example for example in examples if example["id"] not in completed]
    reader = ReaderClient(
        args.llm_base_url,
        args.llm_model,
        timeout=240,
        max_tokens=args.max_tokens,
    )
    write_lock = threading.Lock()
    failures = []

    def append(row: dict) -> None:
        with write_lock:
            with output_path.open("a", encoding="utf-8") as target:
                target.write(json.dumps(row, ensure_ascii=False) + "\n")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(evaluate_example, example, reader): example
            for example in pending
        }
        done = len(completed)
        for future in as_completed(futures):
            example = futures[future]
            try:
                row = future.result()
                completed[row["id"]] = row
                append(row)
            except Exception as exc:  # pragma: no cover - integration retries
                failures.append((example["id"], repr(exc)))
            done += 1
            if done % 50 == 0 or done == len(examples):
                print(
                    f"[sentence-sufficiency-ts] {done}/{len(examples)} "
                    f"failures={len(failures)}",
                    flush=True,
                )
    if failures:
        preview = "; ".join(f"{query_id}: {error}" for query_id, error in failures[:5])
        raise RuntimeError(f"{len(failures)} reader evaluations failed: {preview}")

    ordered = [completed[example["id"]] for example in examples]
    write_jsonl(output_path, ordered)
    result = summarize(
        ordered,
        population=population,
        alignment=alignment,
        input_manifest=manifest,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
