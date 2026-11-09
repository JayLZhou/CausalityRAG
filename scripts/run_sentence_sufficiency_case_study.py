#!/usr/bin/env python3
"""Evaluate whether attribution-selected sentences preserve QA accuracy."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.io import load_records, record_id, retrieved_contexts
from causalityrag.reader import (
    ReaderClient,
    answer_token_f1,
    answers_exact_match,
)


METHODS = ("Full", "ReFlow", "MIRAGE", "ARC-JSD")
TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def index_rows(path: str) -> dict[str, dict]:
    rows = load_records(path)
    indexed = {record_id(row): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"duplicate IDs in {path}")
    return indexed


def ranked_ids(row: dict) -> list[str]:
    raw = row.get("ranked_ids")
    if isinstance(raw, list):
        values = raw
    else:
        values = [
            unit_id
            for unit_id, _ in sorted(
                row.get("token_scores", {}).items(),
                key=lambda item: (-float(item[1]), str(item[0])),
            )
        ]
    result = []
    seen = set()
    for value in values:
        unit_id = str(value)
        if unit_id and unit_id not in seen:
            seen.add(unit_id)
            result.append(unit_id)
    return result


def sentence_ids_for_units(
    unit_ids: list[str],
    sentence_by_unit: dict[str, str],
) -> list[str]:
    result = []
    seen = set()
    for unit_id in unit_ids:
        sentence_id = sentence_by_unit.get(str(unit_id), "")
        if sentence_id and sentence_id not in seen:
            seen.add(sentence_id)
            result.append(sentence_id)
    return result


def ranked_sentence_ids(
    ranking: list[str],
    sentence_by_unit: dict[str, str],
    budget: int,
) -> list[str]:
    result = []
    seen = set()
    for unit_id in ranking:
        sentence_id = sentence_by_unit.get(str(unit_id), "")
        if not sentence_id or sentence_id in seen:
            continue
        seen.add(sentence_id)
        result.append(sentence_id)
        if len(result) >= budget:
            break
    return result


def sentence_contexts(
    sentence_ids: list[str],
    units_row: dict,
    retrieval_row: dict,
) -> list[dict[str, Any]]:
    wanted = set(sentence_ids)
    contexts = {
        str(context["chunk_id"]): context
        for context in retrieved_contexts(retrieval_row)[:5]
    }
    rows = [
        sentence
        for sentence in units_row.get("sentences", [])
        if str(sentence.get("sentence_id", "")) in wanted
    ]
    rows.sort(
        key=lambda row: (
            int(row.get("chunk_rank", 1 << 30)),
            int(row.get("sentence_index", 1 << 30)),
            str(row.get("sentence_id", "")),
        )
    )
    result = []
    for index, sentence in enumerate(rows, start=1):
        chunk_id = str(sentence.get("chunk_id", ""))
        source = contexts.get(chunk_id, {})
        text = str(sentence.get("text", "")).strip()
        if not text:
            continue
        result.append(
            {
                "chunk_id": f"sentence:{sentence.get('sentence_id', index)}",
                "title": str(source.get("title", "")),
                "text": text,
                "rank": index,
            }
        )
    return result


def context_token_count(contexts: list[dict]) -> int:
    return sum(len(TOKEN_RE.findall(str(context.get("text", "")))) for context in contexts)


def build_examples(args: argparse.Namespace) -> tuple[list[dict], dict]:
    retrieval_rows = load_records(args.retrieval)
    units = index_rows(args.units)
    reflow = index_rows(args.reflow)
    mirage = index_rows(args.mirage)
    arc_jsd = index_rows(args.arc_jsd)
    examples = []
    exclusions: dict[str, int] = {}

    def exclude(reason: str) -> None:
        exclusions[reason] = exclusions.get(reason, 0) + 1

    for retrieval_row in retrieval_rows:
        query_id = record_id(retrieval_row)
        if not query_id:
            exclude("missing_id")
            continue
        if any(query_id not in source for source in (units, reflow, mirage, arc_jsd)):
            exclude("missing_aligned_artifact")
            continue
        units_row = units[query_id]
        reflow_row = reflow[query_id]
        selected_ids = [str(value) for value in reflow_row.get("selected_ids", [])]
        if not selected_ids:
            exclude("reflow_no_selected_tokens")
            continue
        sentence_by_unit = {
            str(unit.get("unit_id", "")): str(unit.get("sentence_id", ""))
            for unit in units_row.get("units", [])
        }
        reflow_sentence_ids = sentence_ids_for_units(selected_ids, sentence_by_unit)
        if not reflow_sentence_ids:
            exclude("reflow_tokens_missing_sentences")
            continue
        budget = len(reflow_sentence_ids)
        method_sentence_ids = {
            "ReFlow": reflow_sentence_ids,
            "MIRAGE": ranked_sentence_ids(
                ranked_ids(mirage[query_id]), sentence_by_unit, budget
            ),
            "ARC-JSD": ranked_sentence_ids(
                ranked_ids(arc_jsd[query_id]), sentence_by_unit, budget
            ),
        }
        if any(len(values) != budget for values in method_sentence_ids.values()):
            exclude("baseline_sentence_budget_shortfall")
            continue
        contexts = {"Full": retrieved_contexts(retrieval_row)[:5]}
        contexts.update(
            {
                method: sentence_contexts(ids, units_row, retrieval_row)
                for method, ids in method_sentence_ids.items()
            }
        )
        if any(not contexts[method] for method in METHODS):
            exclude("empty_context")
            continue
        gold = str(reflow_row.get("gold_answer", retrieval_row.get("answer", "")))
        examples.append(
            {
                "id": query_id,
                "index": int(reflow_row.get("index", len(examples))),
                "question": str(retrieval_row.get("question", "")),
                "gold_answer": gold,
                "sentence_budget": budget,
                "selected_sentence_ids": method_sentence_ids,
                "contexts": contexts,
            }
        )
        if args.limit and len(examples) >= args.limit:
            break
    return examples, {
        "retrieval_rows": len(retrieval_rows),
        "included_queries": len(examples),
        "exclusions": exclusions,
    }


def reader_answer(reader: ReaderClient, question: str, contexts: list[dict]) -> str:
    last_error: Exception | None = None
    for attempt in range(6):
        try:
            return reader.answer(question, contexts)
        except Exception as exc:  # pragma: no cover - integration retries
            last_error = exc
            time.sleep(min(8.0, 0.5 * (2**attempt)))
    raise RuntimeError(f"reader request failed: {last_error}")


def evaluate_example(example: dict, reader: ReaderClient) -> dict:
    predictions = {}
    metrics = {}
    context_stats = {}
    for method in METHODS:
        contexts = example["contexts"][method]
        prediction = reader_answer(reader, example["question"], contexts)
        predictions[method] = prediction
        metrics[method] = {
            "em": float(answers_exact_match(prediction, example["gold_answer"])),
            "f1": answer_token_f1(prediction, example["gold_answer"]),
        }
        context_stats[method] = {
            "sentences": (
                len(contexts) if method != "Full" else None
            ),
            "tokens": context_token_count(contexts),
        }
    return {
        key: example[key]
        for key in (
            "id",
            "index",
            "question",
            "gold_answer",
            "sentence_budget",
            "selected_sentence_ids",
        )
    } | {
        "predictions": predictions,
        "metrics": metrics,
        "context_stats": context_stats,
    }


def percentile_interval(values: list[float]) -> list[float]:
    ordered = sorted(values)
    return [
        ordered[int(0.025 * (len(ordered) - 1))],
        ordered[int(0.975 * (len(ordered) - 1))],
    ]


def summarize(
    rows: list[dict],
    population: dict,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict:
    rng = random.Random(bootstrap_seed)
    summary: dict[str, Any] = {
        "schema": "causalityrag.sentence_sufficiency.v1",
        "population": population,
        "queries": len(rows),
        "methods": {},
        "paired_deltas_reflow_minus_baseline": {},
    }
    for method in METHODS:
        method_summary = {}
        for metric in ("em", "f1"):
            values = [float(row["metrics"][method][metric]) for row in rows]
            bootstraps = [
                statistics.fmean(values[rng.randrange(len(values))] for _ in values)
                for _ in range(bootstrap_samples)
            ]
            method_summary[metric] = {
                "mean": statistics.fmean(values),
                "ci95": percentile_interval(bootstraps),
            }
        method_summary["mean_context_tokens"] = statistics.fmean(
            float(row["context_stats"][method]["tokens"]) for row in rows
        )
        if method != "Full":
            method_summary["mean_sentences"] = statistics.fmean(
                float(row["context_stats"][method]["sentences"]) for row in rows
            )
        summary["methods"][method] = method_summary

    for baseline in ("MIRAGE", "ARC-JSD"):
        baseline_summary = {}
        for metric in ("em", "f1"):
            deltas = [
                float(row["metrics"]["ReFlow"][metric])
                - float(row["metrics"][baseline][metric])
                for row in rows
            ]
            bootstraps = [
                statistics.fmean(deltas[rng.randrange(len(deltas))] for _ in deltas)
                for _ in range(bootstrap_samples)
            ]
            baseline_summary[metric] = {
                "mean": statistics.fmean(deltas),
                "ci95": percentile_interval(bootstraps),
            }
        summary["paired_deltas_reflow_minus_baseline"][baseline] = baseline_summary
    return summary


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
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--llm-base-url", default="http://127.0.0.1:8101/v1")
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
    if args.prepare_only:
        print(json.dumps(population, ensure_ascii=False, indent=2))
        return
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = {}
    if output_path.exists():
        completed = index_rows(str(output_path))
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
            except Exception as exc:  # pragma: no cover - integration failures
                failures.append((example["id"], repr(exc)))
            done += 1
            if done % 50 == 0 or done == len(examples):
                print(
                    f"[sentence-sufficiency] {done}/{len(examples)} "
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
        population,
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
