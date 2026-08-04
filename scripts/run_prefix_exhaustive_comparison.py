#!/usr/bin/env python3
"""Compare exhaustive subset search and ReFlow over nested token prefixes."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import statistics
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.contribution_graph import contribution_graph_edges
from causalityrag.flow_contract import breakpoint_price_cuts
from causalityrag.io import load_records, record_id
from causalityrag.reader import ReaderClient, answer_token_f1
from causalityrag.revision import apply_token_replacements
from causalityrag.shared_replacement_pool import (
    FrozenSharedReplacementPool,
    file_sha256,
    stable_shared_candidate,
)
from causalityrag.token_units import units_from_cache_row


def incident_scores(
    unit_ids: set[str],
    source: dict[str, float],
    interactions: dict[tuple[str, str], float],
    target: dict[str, float],
) -> dict[str, float]:
    scores = {
        unit_id: source.get(unit_id, 0.0) + target.get(unit_id, 0.0)
        for unit_id in unit_ids
    }
    for (left, right), weight in interactions.items():
        if left in scores:
            scores[left] += weight
        if right in scores:
            scores[right] += weight
    return scores


def stable_candidate(candidates: list[dict], unit_id: str, seed: int) -> dict:
    material = f"{seed}\0{unit_id}".encode("utf-8")
    index = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    return dict(candidates[index % len(candidates)])


def minimum_successful_subset(
    prefix: tuple[str, ...],
    evaluate: Callable[[tuple[str, ...]], bool],
) -> tuple[tuple[str, ...], bool, int]:
    calls = 0
    for size in range(1, len(prefix) + 1):
        successful = []
        for subset in itertools.combinations(prefix, size):
            calls += 1
            if evaluate(tuple(sorted(subset))):
                successful.append(tuple(sorted(subset)))
        if successful:
            return min(successful), True, calls
    return tuple(sorted(prefix)), False, calls


def load_paraphrase_pool(path: str) -> dict[str, dict]:
    pool = {}
    for row in load_records(path):
        unit_id = str(row.get("unit_id", ""))
        if unit_id and row.get("candidates"):
            pool[unit_id] = row
    return pool


def summarize(rows: list[dict], max_n: int) -> dict:
    methods = {}
    for method in ("exhaustive", "reflow"):
        curve = []
        for n in range(1, max_n + 1):
            values = [row["prefixes"][str(n)][method] for row in rows]
            factual = statistics.fmean(float(value["factual_f1_flip"]) for value in values)
            control = statistics.fmean(float(value["synonym_f1_flip"]) for value in values)
            curve.append({
                "n": n,
                "queries": len(values),
                "mean_modified_tokens": statistics.fmean(
                    int(value["n_modified_tokens"]) for value in values
                ),
                "factual_f1_flip_rate": factual,
                "synonym_f1_flip_rate": control,
                "f1_cfr": factual - control,
                "mean_selection_reader_calls": statistics.fmean(
                    int(value["selection_reader_calls"]) for value in values
                ),
            })
        methods[method] = curve
    return {"queries": len(rows), "methods": methods}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--units-cache", required=True)
    parser.add_argument("--graphs", required=True)
    parser.add_argument("--shared-pool", required=True)
    parser.add_argument("--cohort-results", required=True)
    parser.add_argument("--paraphrase-pool", default="")
    parser.add_argument("--required-units-out", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--max-n", type=int, default=10)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--llm-base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--llm-model", default="qwen2.5-7b")
    args = parser.parse_args()

    records = load_records(args.input)
    unit_rows = load_records(args.units_cache)
    graph_rows = load_records(args.graphs)
    if len({len(records), len(unit_rows), len(graph_rows)}) != 1:
        raise ValueError("input, units, and graph rows must be aligned")
    cohort_ids = {str(row["id"]) for row in load_records(args.cohort_results)}
    counterfactual_pool = FrozenSharedReplacementPool(args.shared_pool)
    prepared = []
    for record, unit_row, graph_row in zip(records, unit_rows, graph_rows):
        query_id = record_id(record)
        if query_id not in cohort_ids:
            continue
        aligned = {str(unit_row.get("id", "")), str(graph_row.get("id", ""))}
        if aligned != {query_id}:
            raise ValueError(f"misaligned query {query_id}: {aligned}")
        clean = str(graph_row.get("clean_answer", graph_row.get("target_answer", "")))
        gold = str(record.get("answer", graph_row.get("gold_answer", "")))
        if answer_token_f1(clean, gold) < 1.0 - 1e-12:
            raise ValueError(f"cohort query is not clean-F1-correct: {query_id}")
        all_units = units_from_cache_row(record, unit_row, k=args.k)
        by_id = {
            str(unit["unit_id"]): unit
            for unit in all_units
            if counterfactual_pool.is_eligible(str(unit["unit_id"]))
        }
        source, interactions, target = contribution_graph_edges(graph_row)
        scores = incident_scores(set(by_id), source, interactions, target)
        ranked_ids = tuple(
            unit_id
            for unit_id, score in sorted(scores.items(), key=lambda item: (-item[1], item[0]))
            if score > 0.0
        )[: args.max_n]
        if len(ranked_ids) != args.max_n:
            raise ValueError(f"cohort query lacks {args.max_n} ranked tokens: {query_id}")
        prepared.append({
            "id": query_id,
            "record": record,
            "clean": clean,
            "gold": gold,
            "units": {unit_id: by_id[unit_id] for unit_id in ranked_ids},
            "ranked_ids": ranked_ids,
            "source": source,
            "interactions": interactions,
            "target": target,
        })
    if {item["id"] for item in prepared} != cohort_ids:
        missing = sorted(cohort_ids - {item["id"] for item in prepared})
        raise ValueError(f"missing {len(missing)} cohort queries: {missing[:3]}")

    required_path = Path(args.required_units_out) if args.required_units_out else None
    if required_path:
        required_path.parent.mkdir(parents=True, exist_ok=True)
        with required_path.open("w", encoding="utf-8") as handle:
            for item in sorted(prepared, key=lambda value: value["id"]):
                handle.write(json.dumps({
                    "id": item["id"],
                    "selected_ids": list(item["ranked_ids"]),
                }) + "\n")
    if args.prepare_only:
        print(json.dumps({
            "queries": len(prepared),
            "max_n": args.max_n,
            "required_token_positions": sum(len(item["ranked_ids"]) for item in prepared),
            "required_units_out": str(required_path) if required_path else None,
        }, indent=2))
        return
    if not args.paraphrase_pool:
        raise ValueError("--paraphrase-pool is required unless --prepare-only is used")
    paraphrase_pool = load_paraphrase_pool(args.paraphrase_pool)
    missing_paraphrases = sorted({
        unit_id
        for item in prepared
        for unit_id in item["ranked_ids"]
        if unit_id not in paraphrase_pool
    })
    if missing_paraphrases:
        raise ValueError(
            f"paraphrase pool misses {len(missing_paraphrases)} required positions"
        )

    target_path = Path(args.out)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    completed = {
        str(row["id"]): row
        for row in (load_records(str(target_path)) if target_path.exists() else [])
    }
    reader = ReaderClient(base_url=args.llm_base_url, model=args.llm_model)
    write_lock = threading.Lock()

    def evaluate_query(item: dict) -> dict:
        query_id = item["id"]
        if query_id in completed:
            return completed[query_id]
        record = item["record"]
        clean = item["clean"]
        gold = item["gold"]
        units = item["units"]
        ranked_ids = item["ranked_ids"]
        factual_cache: dict[tuple[str, ...], dict] = {}
        synonym_cache: dict[tuple[str, ...], dict] = {}

        def execute(selected: tuple[str, ...], *, synonym: bool) -> dict:
            key = tuple(sorted(selected))
            cache = synonym_cache if synonym else factual_cache
            if key in cache:
                return cache[key]
            if not key:
                result = {
                    "answer": clean,
                    "f1": 1.0,
                    "f1_flip": False,
                    "reader_called": False,
                }
                cache[key] = result
                return result
            if synonym:
                replacements = {
                    unit_id: stable_candidate(
                        paraphrase_pool[unit_id]["candidates"], unit_id, args.seed
                    )
                    for unit_id in key
                }
            else:
                pool_rows = counterfactual_pool.require(key)
                replacements = {
                    unit_id: stable_shared_candidate(
                        pool_rows[unit_id]["candidates"], unit_id=unit_id, seed=args.seed
                    )
                    for unit_id in key
                }
            revision = apply_token_replacements(
                record,
                [units[unit_id] for unit_id in key],
                replacements,
                k=args.k,
                allow_whitespace=synonym,
                allow_case_only=synonym,
            )
            if revision["n_failed_edits"] or revision["n_edits"] != len(key):
                raise ValueError(f"failed frozen edit for {query_id}: {key}")
            answer = reader.answer(str(record.get("question", "")), revision["edited_contexts"])
            score = answer_token_f1(answer, gold)
            result = {
                "answer": answer,
                "f1": score,
                "f1_flip": score < 1.0 - 1e-12,
                "reader_called": True,
            }
            cache[key] = result
            return result

        prefixes = {}
        for n in range(1, args.max_n + 1):
            prefix = ranked_ids[:n]
            before_exhaustive = len(factual_cache)
            exhaustive_set, exhaustive_flip, _ = minimum_successful_subset(
                prefix,
                lambda selected: bool(execute(selected, synonym=False)["f1_flip"]),
            )
            exhaustive_calls = len(factual_cache) - before_exhaustive

            domain_units = [units[unit_id] for unit_id in prefix]
            frontier = breakpoint_price_cuts(
                domain_units, item["source"], item["interactions"], item["target"]
            )
            candidates = sorted(
                frontier.get("candidates", []),
                key=lambda candidate: (
                    int(candidate.get("n_selected", 0)),
                    float(candidate.get("residual_edge_cost", float("inf"))),
                    tuple(candidate.get("selected_ids", [])),
                ),
            )
            before_reflow = len(factual_cache)
            reflow_set = tuple(sorted(prefix))
            reflow_flip = False
            reflow_fallback = not candidates
            for candidate in candidates:
                selected = tuple(sorted(str(value) for value in candidate["selected_ids"]))
                reflow_set = selected
                if execute(selected, synonym=False)["f1_flip"]:
                    reflow_flip = True
                    break
            if reflow_fallback:
                reflow_flip = bool(execute(reflow_set, synonym=False)["f1_flip"])
            reflow_calls = len(factual_cache) - before_reflow

            method_rows = {}
            for method, selected, factual_flip, calls in (
                ("exhaustive", exhaustive_set, exhaustive_flip, exhaustive_calls),
                ("reflow", reflow_set, reflow_flip, reflow_calls),
            ):
                factual = execute(selected, synonym=False)
                control = execute(selected, synonym=True)
                method_rows[method] = {
                    "selected_ids": list(selected),
                    "n_modified_tokens": len(selected),
                    "factual_f1": factual["f1"],
                    "factual_f1_flip": bool(factual["f1_flip"]),
                    "synonym_f1": control["f1"],
                    "synonym_f1_flip": bool(control["f1_flip"]),
                    "selection_reader_calls": calls,
                    "used_full_prefix_fallback": bool(
                        method == "reflow" and reflow_fallback
                    ),
                }
            prefixes[str(n)] = method_rows
        row = {
            "id": query_id,
            "ranked_ids": list(ranked_ids),
            "prefixes": prefixes,
            "unique_factual_reader_calls": sum(
                int(value["reader_called"]) for value in factual_cache.values()
            ),
            "unique_synonym_reader_calls": sum(
                int(value["reader_called"]) for value in synonym_cache.values()
            ),
        }
        with write_lock:
            with target_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row

    rows = list(completed.values())
    pending = [item for item in prepared if item["id"] not in completed]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(evaluate_query, item) for item in pending]
        for index, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if index % 10 == 0:
                print(f"[prefix] {index}/{len(futures)} new queries", flush=True)
    rows = sorted({str(row["id"]): row for row in rows}.values(), key=lambda row: row["id"])
    with target_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "schema": "causalityrag.prefix_exhaustive_comparison.v1",
        "queries": len(rows),
        "max_n": args.max_n,
        "domain_policy": "nested prefixes ranked by positive incident capacity",
        "selection_condition": "clean token-F1 equals 1; factual edited token-F1 is below 1",
        "f1_cfr": "factual F1 flip rate minus same-position synonym F1 flip rate",
        **summarize(rows, args.max_n),
        "reader_calls": {
            "unique_factual": sum(row["unique_factual_reader_calls"] for row in rows),
            "unique_synonym": sum(row["unique_synonym_reader_calls"] for row in rows),
        },
        "source_sha256": {
            "input": file_sha256(args.input),
            "units": file_sha256(args.units_cache),
            "graphs": file_sha256(args.graphs),
            "shared_pool": file_sha256(args.shared_pool),
            "cohort": file_sha256(args.cohort_results),
            "paraphrase_pool": file_sha256(args.paraphrase_pool),
        },
    }
    Path(args.summary_out).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
