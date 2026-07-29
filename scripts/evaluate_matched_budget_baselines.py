"""Evaluate ranking baselines with ReFlow's token budget and frozen pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.io import load_records, record_id
from causalityrag.reader import ReaderClient, answers_exact_match
from causalityrag.revision import apply_token_replacements
from causalityrag.shared_replacement_pool import (
    FrozenSharedReplacementPool,
    file_sha256,
    stable_shared_candidate,
)
from causalityrag.token_units import units_from_cache_row
from scripts.evaluate_reflow import valid_clean_answer


def _safe_score(value: object) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return -math.inf
    return score if math.isfinite(score) else -math.inf


def ranked_ids(row: dict) -> list[str]:
    if isinstance(row.get("ranked_ids"), list):
        raw = row["ranked_ids"]
    elif isinstance(row.get("token_scores"), dict):
        raw = [
            unit_id
            for unit_id, _ in sorted(
                row.get("token_scores", {}).items(),
                key=lambda item: (-_safe_score(item[1]), str(item[0])),
            )
        ]
    else:
        return []
    result = []
    seen = set()
    for unit_id in raw:
        normalized = str(unit_id)
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def random_ranked_ids(
    unit_ids: list[str],
    *,
    query_id: str,
    seed: int,
) -> list[str]:
    return sorted(
        unit_ids,
        key=lambda unit_id: (
            hashlib.sha256(
                f"{seed}\0{query_id}\0{unit_id}".encode("utf-8")
            ).digest(),
            unit_id,
        ),
    )


def _parse_score_specs(specs: list[str], n: int) -> dict[str, list[dict]]:
    tables = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"invalid --scores value: {spec}")
        name, path = spec.split("=", 1)
        name = name.strip()
        if not name or name in tables:
            raise ValueError(f"duplicate or empty baseline name: {name}")
        rows = load_records(path)[:n]
        tables[name] = rows
    return tables


def evaluate_query(
    record: dict,
    units_row: dict,
    reflow_row: dict,
    score_rows: dict[str, dict],
    pool: FrozenSharedReplacementPool,
    reader: ReaderClient,
    *,
    random_seeds: list[int],
    replacement_seed: int,
    k: int,
) -> dict:
    query_id = record_id(record)
    clean_answer = str(reflow_row.get("clean_answer", ""))
    gold_answer = str(record.get("answer", ""))
    budget = int(reflow_row.get("n_modified_tokens", 0))
    base = {
        "index": int(reflow_row.get("index", -1)),
        "id": query_id,
        "clean_answer": clean_answer,
        "gold_answer": gold_answer,
        "clean_correct": answers_exact_match(clean_answer, gold_answer),
        "matched_token_budget": budget,
        "replacement_seed": replacement_seed,
    }
    if not valid_clean_answer(clean_answer):
        return {
            **base,
            "methods": {},
            "reader_calls": 0,
            "evaluation_status": "invalid_clean_answer",
        }
    if budget <= 0:
        return {
            **base,
            "methods": {},
            "reader_calls": 0,
            "evaluation_status": "zero_reflow_budget",
        }

    units = units_from_cache_row(record, units_row, k=k)
    by_id = {str(unit["unit_id"]): unit for unit in units}
    eligible_ids = sorted(
        unit_id for unit_id in by_id if pool.is_eligible(unit_id)
    )
    rankings = {
        name: [
            unit_id
            for unit_id in ranked_ids(row)
            if unit_id in by_id and pool.is_eligible(unit_id)
        ]
        for name, row in score_rows.items()
    }
    rankings.update({
        f"random_seed{seed}": random_ranked_ids(
            eligible_ids,
            query_id=query_id,
            seed=seed,
        )
        for seed in random_seeds
    })

    methods = {}
    reader_calls = 0
    for method, ranking in rankings.items():
        if len(ranking) < budget:
            methods[method] = {
                "status": "insufficient_ranked_tokens",
                "matched_token_budget": budget,
                "selected_ids": [],
                "n_modified_tokens": 0,
                "verified_flip": False,
                "reader_called": False,
            }
            continue
        selected_ids = ranking[:budget]
        pool_rows = pool.require(selected_ids)
        replacements = {
            unit_id: stable_shared_candidate(
                pool_rows[unit_id]["candidates"],
                unit_id=unit_id,
                seed=replacement_seed,
            )
            for unit_id in selected_ids
        }
        revision = apply_token_replacements(
            record,
            [by_id[unit_id] for unit_id in selected_ids],
            replacements,
            k=k,
        )
        if (
            int(revision["n_failed_edits"]) > 0
            or int(revision["n_edits"]) != len(selected_ids)
        ):
            methods[method] = {
                "status": "protocol_violation_failed_edit",
                "matched_token_budget": budget,
                "selected_ids": selected_ids,
                "n_modified_tokens": 0,
                "edits": revision["edits"],
                "verified_flip": False,
                "reader_called": False,
            }
            continue
        edited_answer = reader.answer(
            str(record.get("question", "")),
            revision["edited_contexts"],
        )
        changed = not answers_exact_match(clean_answer, edited_answer)
        reader_calls += 1
        methods[method] = {
            "status": "verified_flip" if changed else "verified_no_flip",
            "matched_token_budget": budget,
            "selected_ids": selected_ids,
            "selected_tokens": [
                str(by_id[unit_id].get("text", "")) for unit_id in selected_ids
            ],
            "n_modified_tokens": int(revision["n_edits"]),
            "edits": revision["edits"],
            "edited_answer": edited_answer,
            "verified_flip": changed,
            "reader_called": True,
        }
    return {
        **base,
        "methods": methods,
        "reader_calls": reader_calls,
        "evaluation_status": "evaluated",
    }


def summarize(rows: list[dict]) -> dict:
    names = sorted({
        name for row in rows for name in row.get("methods", {})
    })
    methods = {}
    for name in names:
        results = [
            row["methods"][name]
            for row in rows
            if name in row.get("methods", {})
        ]
        executed = [row for row in results if row.get("reader_called")]
        flips = [row for row in executed if row.get("verified_flip")]
        methods[name] = {
            "available_queries": len(results),
            "executed_queries": len(executed),
            "verified_flips": len(flips),
            "flip_rate": len(flips) / max(1, len(executed)),
            "mean_modified_tokens": (
                sum(int(row.get("n_modified_tokens", 0)) for row in executed)
                / max(1, len(executed))
            ),
        }
    return {
        "queries": len(rows),
        "reader_calls": sum(int(row.get("reader_calls", 0)) for row in rows),
        "methods": methods,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--units-cache", required=True)
    parser.add_argument("--reflow-results", required=True)
    parser.add_argument(
        "--scores",
        action="append",
        default=[],
        metavar="METHOD=JSONL",
    )
    parser.add_argument("--shared-pool", required=True)
    parser.add_argument("--expected-pool-sha256", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--random-seeds", default="0,1,2,3,4")
    parser.add_argument("--replacement-seed", type=int, default=0)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--llm-base-url", default="")
    parser.add_argument("--llm-model", default="")
    args = parser.parse_args()

    actual_pool_sha = file_sha256(args.shared_pool)
    if actual_pool_sha != args.expected_pool_sha256:
        raise ValueError(
            "replacement pool fingerprint mismatch: "
            f"expected {args.expected_pool_sha256}, got {actual_pool_sha}"
        )
    pool = FrozenSharedReplacementPool(args.shared_pool)
    records = load_records(args.input)[: args.n]
    units_rows = load_records(args.units_cache)[: args.n]
    reflow_rows = load_records(args.reflow_results)[: args.n]
    tables = _parse_score_specs(args.scores, args.n)
    lengths = {
        len(records),
        len(units_rows),
        len(reflow_rows),
        *[len(rows) for rows in tables.values()],
    }
    if len(lengths) != 1:
        raise ValueError("all inputs must contain aligned query rows")
    random_seeds = [
        int(value) for value in args.random_seeds.split(",") if value.strip()
    ]
    reader = ReaderClient(
        base_url=args.llm_base_url or None,
        model=args.llm_model or None,
    )
    lock = Lock()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    def run(index: int) -> dict:
        identifier = record_id(records[index])
        aligned = {
            str(units_rows[index].get("id", "")),
            str(reflow_rows[index].get("id", "")),
            *[
                str(rows[index].get("id", ""))
                for rows in tables.values()
            ],
        }
        if aligned != {identifier}:
            raise ValueError(f"misaligned row {index}: {aligned}")
        row = evaluate_query(
            records[index],
            units_rows[index],
            reflow_rows[index],
            {name: rows[index] for name, rows in tables.items()},
            pool,
            reader,
            random_seeds=random_seeds,
            replacement_seed=args.replacement_seed,
            k=args.k,
        )
        with lock:
            with open(args.out, "a", encoding="utf-8") as output:
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run, index) for index in range(len(records))]
        for completed, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if completed % 100 == 0:
                print(f"[{completed}/{len(futures)}]", flush=True)

    rows.sort(key=lambda row: int(row.get("index", -1)))
    temporary = args.out + ".tmp"
    with open(temporary, "w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, args.out)
    summary = {
        **summarize(rows),
        "shared_pool": os.path.abspath(args.shared_pool),
        "shared_pool_sha256": actual_pool_sha,
        "replacement_seed": args.replacement_seed,
        "random_seeds": random_seeds,
        "score_files": {
            spec.split("=", 1)[0]: os.path.abspath(spec.split("=", 1)[1])
            for spec in args.scores
        },
    }
    with open(args.summary_out, "w", encoding="utf-8") as output:
        json.dump(summary, output, indent=2, ensure_ascii=False)
        output.write("\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
