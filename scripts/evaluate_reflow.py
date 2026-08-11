"""Verify a saved ReFlow frontier against one frozen replacement pool."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.io import gold_answers, load_records, record_id
from causalityrag.reader import (
    ReaderClient,
    ReaderProtocolError,
    canonicalize_reader_answer,
    reader_answers_exact_match,
)
from causalityrag.revision import apply_token_replacements
from causalityrag.shared_replacement_pool import (
    FrozenSharedReplacementPool,
    file_sha256,
    stable_shared_candidate,
)
from causalityrag.token_units import units_from_cache_row


def candidate_rows(frontier_row: dict) -> list[dict]:
    rows = [
        dict(candidate)
        for candidate in frontier_row.get("frontier_candidates", [])
        if candidate.get("selected_ids")
    ]
    rows.sort(
        key=lambda row: (
            int(row.get("n_selected", len(row.get("selected_ids", [])))),
            float(row.get("remaining_support_flow", float("inf"))),
            tuple(str(value) for value in row.get("selected_ids", [])),
        )
    )
    return rows


def valid_clean_answer(answer: str) -> bool:
    return str(answer).strip().casefold() not in {
        "",
        "[]",
        "{}",
        "null",
        "none",
        "no answer",
        "no answer found",
    }


def evaluate_query(
    record: dict,
    units_row: dict,
    frontier_row: dict,
    pool: FrozenSharedReplacementPool,
    reader: ReaderClient,
    *,
    replacement_seed: int,
    k: int,
) -> dict:
    started = time.monotonic()
    query_id = record_id(record)
    question = str(record.get("question", ""))
    reader_mode = getattr(reader, "reader_mode", "short_answer")
    units = units_from_cache_row(record, units_row, k=k)
    by_id = {str(unit["unit_id"]): unit for unit in units}
    clean_answer_raw = str(frontier_row.get("clean_answer", ""))
    gold_answer_raw = str(record.get("answer", ""))
    accepted_gold_answers_raw = gold_answers(record)
    try:
        clean_answer = canonicalize_reader_answer(
            question,
            clean_answer_raw,
            reader_mode=reader_mode,
        )
        gold_answer = canonicalize_reader_answer(
            question,
            gold_answer_raw,
            reader_mode=reader_mode,
        )
        accepted_gold_answers = [
            canonicalize_reader_answer(
                question,
                answer,
                reader_mode=reader_mode,
            )
            for answer in accepted_gold_answers_raw
        ]
    except ReaderProtocolError as exc:
        return {
            "index": int(frontier_row.get("index", -1)),
            "id": query_id,
            "method": "reflow",
            "clean_answer": clean_answer_raw,
            "gold_answer": gold_answer_raw,
            "gold_answers": accepted_gold_answers_raw,
            "clean_correct": False,
            "replacement_seed": replacement_seed,
            "eligible": False,
            "verified_flip": False,
            "n_modified_tokens": 0,
            "reader_calls": 0,
            "attempts": [],
            "evaluation_status": "protocol_violation_invalid_clean_or_gold_answer",
            "protocol_error": str(exc),
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
    base = {
        "index": int(frontier_row.get("index", -1)),
        "id": query_id,
        "method": "reflow",
        "clean_answer": clean_answer,
        "gold_answer": gold_answer,
        "gold_answers": accepted_gold_answers,
        "clean_answer_raw": clean_answer_raw,
        "gold_answer_raw": gold_answer_raw,
        "clean_correct": any(
            reader_answers_exact_match(
                question,
                clean_answer,
                accepted,
                reader_mode=reader_mode,
            )
            for accepted in accepted_gold_answers
        ),
        "replacement_seed": replacement_seed,
    }
    if not valid_clean_answer(clean_answer):
        return {
            **base,
            "eligible": False,
            "verified_flip": False,
            "n_modified_tokens": 0,
            "reader_calls": 0,
            "attempts": [],
            "evaluation_status": "invalid_clean_answer",
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }

    attempts = []
    verified = None
    for candidate in candidate_rows(frontier_row):
        selected_ids = [
            str(unit_id) for unit_id in candidate.get("selected_ids", [])
        ]
        unknown = [unit_id for unit_id in selected_ids if unit_id not in by_id]
        uncovered = [
            unit_id
            for unit_id in selected_ids
            if unit_id in by_id and not pool.is_eligible(unit_id)
        ]
        if unknown or uncovered:
            attempts.append({
                **candidate,
                "selected_ids": selected_ids,
                "reader_called": False,
                "answer_changed": False,
                "candidate_status": "protocol_violation_unexecutable_candidate",
                "unknown_ids": unknown,
                "uncovered_ids": uncovered,
            })
            continue

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
            attempts.append({
                **candidate,
                "selected_ids": selected_ids,
                "edits": revision["edits"],
                "reader_called": False,
                "answer_changed": False,
                "candidate_status": "protocol_violation_failed_edit",
            })
            continue
        try:
            edited_answer = reader.answer(
                question,
                revision["edited_contexts"],
            )
        except ReaderProtocolError as exc:
            attempts.append({
                **candidate,
                "selected_ids": selected_ids,
                "selected_tokens": [
                    str(by_id[unit_id].get("text", ""))
                    for unit_id in selected_ids
                ],
                "n_edits": int(revision["n_edits"]),
                "edits": revision["edits"],
                "reader_called": True,
                "answer_changed": False,
                "candidate_status": "protocol_violation_invalid_reader_answer",
                "protocol_error": str(exc),
            })
            break
        changed = not reader_answers_exact_match(
            question,
            clean_answer,
            edited_answer,
            reader_mode=reader_mode,
        )
        attempt = {
            **candidate,
            "selected_ids": selected_ids,
            "selected_tokens": [
                str(by_id[unit_id].get("text", "")) for unit_id in selected_ids
            ],
            "n_edits": int(revision["n_edits"]),
            "edits": revision["edits"],
            "edited_answer": edited_answer,
            "reader_called": True,
            "answer_changed": changed,
            "candidate_status": "verified_flip" if changed else "verified_no_flip",
        }
        attempts.append(attempt)
        if changed:
            verified = attempt
            break

    executed = [
        attempt for attempt in attempts if attempt.get("reader_called")
    ]
    final_attempt = verified or (executed[-1] if executed else None)
    has_protocol_violation = any(
        str(attempt.get("candidate_status", "")).startswith(
            "protocol_violation"
        )
        for attempt in attempts
    )
    has_invalid_reader_answer = any(
        attempt.get("candidate_status")
        == "protocol_violation_invalid_reader_answer"
        for attempt in attempts
    )
    return {
        **base,
        "eligible": True,
        "verified_flip": verified is not None,
        "selected_ids": list((final_attempt or {}).get("selected_ids", [])),
        "selected_tokens": list(
            (final_attempt or {}).get("selected_tokens", [])
        ),
        "n_modified_tokens": int((final_attempt or {}).get("n_edits", 0)),
        "edited_answer": str(
            (final_attempt or {}).get("edited_answer", clean_answer)
        ),
        "reader_calls": len(executed),
        "attempts": attempts,
        "evaluation_status": (
            "verified_flip"
            if verified is not None
            else (
                "protocol_violation_invalid_reader_answer"
                if has_invalid_reader_answer
                else (
                    "verified_no_flip"
                    if executed
                    else (
                        "protocol_violation_unexecutable_frontier"
                        if has_protocol_violation
                        else "no_frontier_candidate"
                    )
                )
            )
        ),
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def summarize(rows: list[dict]) -> dict:
    eligible = [row for row in rows if row.get("eligible")]
    executed = [row for row in eligible if int(row.get("reader_calls", 0)) > 0]
    flips = [row for row in eligible if row.get("verified_flip")]
    clean_correct = [row for row in eligible if row.get("clean_correct")]
    flip_sizes = [
        int(row.get("n_modified_tokens", 0)) for row in flips
    ]
    all_sizes = [
        int(row.get("n_modified_tokens", 0)) for row in rows
    ]
    statuses = sorted({
        str(row.get("evaluation_status", "")) for row in rows
    })
    return {
        "queries": len(rows),
        "eligible_queries": len(eligible),
        "answer_denominator_queries": len(eligible),
        "executed_queries": len(executed),
        "verified_flips": len(flips),
        "raw_flip_rate": len(flips) / max(1, len(rows)),
        "answer_flip_rate": len(flips) / max(1, len(eligible)),
        "eligible_flip_rate": len(flips) / max(1, len(eligible)),
        "executed_flip_rate": len(flips) / max(1, len(executed)),
        "clean_correct_queries": len(clean_correct),
        "clean_correct_flips": sum(
            bool(row.get("verified_flip")) for row in clean_correct
        ),
        "mean_modified_tokens_all_queries": statistics.fmean(all_sizes)
        if all_sizes
        else 0.0,
        "mean_modified_tokens_flips": statistics.fmean(flip_sizes)
        if flip_sizes
        else 0.0,
        "median_modified_tokens_flips": statistics.median(flip_sizes)
        if flip_sizes
        else 0.0,
        "reader_calls": sum(int(row.get("reader_calls", 0)) for row in rows),
        "protocol_violations": sum(
            str(row.get("evaluation_status", "")).startswith(
                "protocol_violation"
            )
            for row in rows
        ),
        "status_histogram": {
            status: sum(
                str(row.get("evaluation_status", "")) == status
                for row in rows
            )
            for status in statuses
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--units-cache", required=True)
    parser.add_argument("--frontier", required=True)
    parser.add_argument("--shared-pool", required=True)
    parser.add_argument("--expected-pool-sha256", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--replacement-seed", type=int, default=0)
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
    unit_rows = load_records(args.units_cache)[: args.n]
    frontier_rows = load_records(args.frontier)[: args.n]
    if len({len(records), len(unit_rows), len(frontier_rows)}) != 1:
        raise ValueError("input, units, and frontier files must be aligned")
    for record, unit_row, frontier_row in zip(
        records,
        unit_rows,
        frontier_rows,
    ):
        identifier = record_id(record)
        if (
            str(unit_row.get("id", "")) != identifier
            or str(frontier_row.get("id", "")) != identifier
        ):
            raise ValueError(f"misaligned row for {identifier}")

    reader = ReaderClient(
        base_url=args.llm_base_url or None,
        model=args.llm_model or None,
    )
    lock = Lock()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    def run(item: tuple[dict, dict, dict]) -> dict:
        row = evaluate_query(
            *item,
            pool,
            reader,
            replacement_seed=args.replacement_seed,
            k=args.k,
        )
        with lock:
            with open(args.out, "a", encoding="utf-8") as output:
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row

    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(run, item)
            for item in zip(records, unit_rows, frontier_rows)
        ]
        for completed, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if completed % 100 == 0:
                partial = summarize(rows)
                print(
                    f"[{completed}/{len(futures)}] "
                    f"flip={partial['verified_flips']} "
                    f"calls={partial['reader_calls']}",
                    flush=True,
                )

    rows.sort(key=lambda row: int(row.get("index", -1)))
    temporary = args.out + ".tmp"
    with open(temporary, "w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, args.out)
    summary = {
        **summarize(rows),
        "replacement_seed": args.replacement_seed,
        "shared_pool": os.path.abspath(args.shared_pool),
        "shared_pool_sha256": actual_pool_sha,
        "frontier": os.path.abspath(args.frontier),
    }
    with open(args.summary_out, "w", encoding="utf-8") as output:
        json.dump(summary, output, indent=2, ensure_ascii=False)
        output.write("\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
