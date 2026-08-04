#!/usr/bin/env python3
"""Evaluate answer flips and independent search effort for prefix candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.contribution_graph import contribution_graph_edges
from causalityrag.flow_contract import breakpoint_price_cuts
from causalityrag.io import load_records, record_id
from causalityrag.reader import ReaderClient, answers_exact_match
from causalityrag.revision import apply_token_replacements
from causalityrag.shared_replacement_pool import (
    FrozenSharedReplacementPool,
    file_sha256,
    stable_shared_candidate,
)
from causalityrag.token_units import units_from_cache_row


def load_paraphrase_pool(path: str) -> dict[str, dict]:
    return {
        str(row["unit_id"]): row
        for row in load_records(path)
        if row.get("unit_id") and row.get("candidates")
    }


def stable_candidate(candidates: list[dict], unit_id: str, seed: int) -> dict:
    material = f"{seed}\0{unit_id}".encode("utf-8")
    index = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    return dict(candidates[index % len(candidates)])


def exhaustive_verifications(n: int, selected_size: int, found: bool) -> int:
    """Return calls made by cardinality-ordered exhaustive search in isolation."""

    stop_size = selected_size if found else n
    return sum(math.comb(n, size) for size in range(1, stop_size + 1))


def sorted_frontier_candidates(frontier: dict) -> list[tuple[str, ...]]:
    candidates = sorted(
        frontier.get("candidates", []),
        key=lambda candidate: (
            int(candidate.get("n_selected", 0)),
            float(candidate.get("residual_edge_cost", float("inf"))),
            tuple(candidate.get("selected_ids", [])),
        ),
    )
    return [
        tuple(sorted(str(value) for value in candidate["selected_ids"]))
        for candidate in candidates
    ]


def reflow_verifications(
    candidates: list[tuple[str, ...]], selected: tuple[str, ...]
) -> int:
    """Count independently verified breakpoint sets through the returned set."""

    if not candidates:
        return 1
    try:
        return candidates.index(tuple(sorted(selected))) + 1
    except ValueError as error:
        raise ValueError(f"selected set is absent from frontier: {selected}") from error


def summarize(rows: list[dict], max_n: int) -> dict:
    methods = {}
    for method in ("exhaustive", "reflow"):
        curve = []
        for n in range(1, max_n + 1):
            values = [row["prefixes"][str(n)][method] for row in rows]
            factual = statistics.fmean(bool(value["factual_answer_flip"]) for value in values)
            control = statistics.fmean(bool(value["synonym_answer_flip"]) for value in values)
            curve.append({
                "n": n,
                "queries": len(values),
                "mean_modified_tokens": statistics.fmean(
                    int(value["n_modified_tokens"]) for value in values
                ),
                "factual_answer_flip_rate": factual,
                "synonym_answer_flip_rate": control,
                "answer_cfr": factual - control,
                "mean_independent_verifications": statistics.fmean(
                    int(value["independent_verifications"]) for value in values
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
    parser.add_argument("--prefix-results", required=True)
    parser.add_argument("--paraphrase-pool", required=True)
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
    prefix_rows = {str(row["id"]): row for row in load_records(args.prefix_results)}
    if len({len(records), len(unit_rows), len(graph_rows)}) != 1:
        raise ValueError("input, units, and graph rows must be aligned")

    counterfactual_pool = FrozenSharedReplacementPool(args.shared_pool)
    paraphrase_pool = load_paraphrase_pool(args.paraphrase_pool)
    prepared = []
    for record, unit_row, graph_row in zip(records, unit_rows, graph_rows):
        query_id = record_id(record)
        if query_id not in prefix_rows:
            continue
        aligned = {str(unit_row.get("id", "")), str(graph_row.get("id", ""))}
        if aligned != {query_id}:
            raise ValueError(f"misaligned query {query_id}: {aligned}")
        prefix_row = prefix_rows[query_id]
        ranked_ids = tuple(str(value) for value in prefix_row["ranked_ids"])
        all_units = units_from_cache_row(record, unit_row, k=args.k)
        by_id = {str(unit["unit_id"]): unit for unit in all_units}
        missing_units = sorted(set(ranked_ids) - set(by_id))
        if missing_units:
            raise ValueError(f"missing units for {query_id}: {missing_units[:3]}")
        counterfactual_pool.require(ranked_ids)
        missing_paraphrases = sorted(set(ranked_ids) - set(paraphrase_pool))
        if missing_paraphrases:
            raise ValueError(
                f"missing paraphrases for {query_id}: {missing_paraphrases[:3]}"
            )
        source, interactions, target = contribution_graph_edges(graph_row)
        prepared.append({
            "id": query_id,
            "record": record,
            "clean": str(graph_row.get("clean_answer", graph_row.get("target_answer", ""))),
            "units": {unit_id: by_id[unit_id] for unit_id in ranked_ids},
            "ranked_ids": ranked_ids,
            "prefix_row": prefix_row,
            "source": source,
            "interactions": interactions,
            "target": target,
        })
    if {item["id"] for item in prepared} != set(prefix_rows):
        missing = sorted(set(prefix_rows) - {item["id"] for item in prepared})
        raise ValueError(f"missing {len(missing)} prefix queries: {missing[:3]}")

    if args.prepare_only:
        counts = {"exhaustive": [], "reflow": []}
        for item in prepared:
            for n in range(1, args.max_n + 1):
                original = item["prefix_row"]["prefixes"][str(n)]
                exhaustive = original["exhaustive"]
                counts["exhaustive"].append(
                    exhaustive_verifications(
                        n,
                        len(exhaustive["selected_ids"]),
                        bool(exhaustive["factual_f1_flip"]),
                    )
                )
                prefix = item["ranked_ids"][:n]
                candidates = sorted_frontier_candidates(
                    breakpoint_price_cuts(
                        [item["units"][unit_id] for unit_id in prefix],
                        item["source"],
                        item["interactions"],
                        item["target"],
                    )
                )
                selected = tuple(sorted(original["reflow"]["selected_ids"]))
                counts["reflow"].append(reflow_verifications(candidates, selected))
        print(json.dumps({
            "queries": len(prepared),
            "max_n": args.max_n,
            "mean_independent_verifications": {
                method: statistics.fmean(values) for method, values in counts.items()
            },
        }, indent=2))
        return

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
        units = item["units"]
        ranked_ids = item["ranked_ids"]
        prefix_row = item["prefix_row"]
        answer_cache: dict[tuple[bool, tuple[str, ...]], str] = {}

        def execute(selected: tuple[str, ...], *, synonym: bool) -> str:
            key = (synonym, tuple(sorted(selected)))
            if key in answer_cache:
                return answer_cache[key]
            if synonym:
                replacements = {
                    unit_id: stable_candidate(
                        paraphrase_pool[unit_id]["candidates"], unit_id, args.seed
                    )
                    for unit_id in key[1]
                }
            else:
                pool_rows = counterfactual_pool.require(key[1])
                replacements = {
                    unit_id: stable_shared_candidate(
                        pool_rows[unit_id]["candidates"], unit_id=unit_id, seed=args.seed
                    )
                    for unit_id in key[1]
                }
            revision = apply_token_replacements(
                record,
                [units[unit_id] for unit_id in key[1]],
                replacements,
                k=args.k,
                allow_whitespace=synonym,
                allow_case_only=synonym,
            )
            if revision["n_failed_edits"] or revision["n_edits"] != len(key[1]):
                raise ValueError(f"failed frozen edit for {query_id}: {key[1]}")
            answer = reader.answer(str(record.get("question", "")), revision["edited_contexts"])
            answer_cache[key] = answer
            return answer

        prefixes = {}
        for n in range(1, args.max_n + 1):
            prefix = ranked_ids[:n]
            domain_units = [units[unit_id] for unit_id in prefix]
            candidates = sorted_frontier_candidates(
                breakpoint_price_cuts(
                    domain_units,
                    item["source"],
                    item["interactions"],
                    item["target"],
                )
            )
            method_rows = {}
            for method in ("exhaustive", "reflow"):
                original = prefix_row["prefixes"][str(n)][method]
                selected = tuple(sorted(str(value) for value in original["selected_ids"]))
                factual_answer = execute(selected, synonym=False)
                synonym_answer = execute(selected, synonym=True)
                factual_flip = not answers_exact_match(clean, factual_answer)
                synonym_flip = not answers_exact_match(clean, synonym_answer)
                if method == "exhaustive":
                    verifications = exhaustive_verifications(
                        n,
                        len(selected),
                        bool(original["factual_f1_flip"]),
                    )
                else:
                    verifications = reflow_verifications(candidates, selected)
                method_rows[method] = {
                    "selected_ids": list(selected),
                    "n_modified_tokens": len(selected),
                    "factual_answer": factual_answer,
                    "synonym_answer": synonym_answer,
                    "factual_answer_flip": factual_flip,
                    "synonym_answer_flip": synonym_flip,
                    "independent_verifications": verifications,
                }
            prefixes[str(n)] = method_rows
        row = {
            "id": query_id,
            "ranked_ids": list(ranked_ids),
            "prefixes": prefixes,
            "unique_reader_calls": len(answer_cache),
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
                print(f"[answer-frontier] {index}/{len(futures)} new queries", flush=True)
    rows = sorted({str(row["id"]): row for row in rows}.values(), key=lambda row: row["id"])
    with target_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "schema": "causalityrag.prefix_answer_frontier.v1",
        "queries": len(rows),
        "max_n": args.max_n,
        "answer_equivalence": "Hotpot-style normalized exact match",
        "answer_cfr": "factual answer-flip rate minus same-position synonym answer-flip rate",
        "enumeration_accounting": "independent logical reader verifications without cross-method cache sharing",
        **summarize(rows, args.max_n),
        "reader_calls": sum(int(row["unique_reader_calls"]) for row in rows),
        "source_sha256": {
            "input": file_sha256(args.input),
            "units": file_sha256(args.units_cache),
            "graphs": file_sha256(args.graphs),
            "shared_pool": file_sha256(args.shared_pool),
            "prefix_results": file_sha256(args.prefix_results),
            "paraphrase_pool": file_sha256(args.paraphrase_pool),
        },
    }
    Path(args.summary_out).write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
