"""Generate and freeze clean reader answers with a concurrent vLLM client."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import islice

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.io import gold_answers, iter_records, record_id, retrieved_contexts
from causalityrag.reader import (
    ReaderClient,
    ReaderProtocolError,
    answers_exact_match,
)


def answer_with_retry(
    reader: ReaderClient,
    question: str,
    contexts: list[dict],
    *,
    retries: int = 5,
) -> str:
    for attempt in range(retries + 1):
        try:
            return reader.answer(question, contexts)
        except (OSError, TimeoutError):
            if attempt == retries:
                raise
            time.sleep(min(2**attempt, 15))
    raise AssertionError("unreachable")


def reader_outcome(future) -> dict:
    """Return a frozen clean-target outcome without accepting protocol errors."""

    try:
        answer = future.result()
    except ReaderProtocolError as error:
        raw_answer = str(getattr(error, "raw_answer", "") or error)
        return {
            "answer": "",
            "status": "protocol_violation",
            "raw_reader_answer": raw_answer,
            "protocol_error": str(error),
        }
    return {
        "answer": answer,
        "status": "ok" if answer.strip() else "reader_abstention_empty_answer",
        "raw_reader_answer": "",
        "protocol_error": "",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", default="")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument(
        "--n",
        type=int,
        default=0,
        help="Number of records; zero reads every record after --start.",
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--base-url", default="")
    parser.add_argument("--served-model", default="")
    args = parser.parse_args()
    if (
        args.start < 0
        or args.n < 0
        or args.k <= 0
        or args.workers <= 0
        or args.retries < 0
    ):
        parser.error(
            "--start and --n must be non-negative; --k and --workers "
            "must be positive"
        )

    stop = None if args.n == 0 else args.start + args.n
    records = list(islice(iter_records(args.input), args.start, stop))
    identifiers = [record_id(record) for record in records]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("input contains duplicate record IDs")

    reader = ReaderClient(
        base_url=args.base_url or None,
        model=args.served_model or None,
    )
    outcomes = [None] * len(records)
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                answer_with_retry,
                reader,
                str(record.get("question", "")),
                retrieved_contexts(record)[: args.k],
                retries=args.retries,
            ): index
            for index, record in enumerate(records)
        }
        completed = 0
        for future in as_completed(futures):
            index = futures[future]
            outcomes[index] = reader_outcome(future)
            completed += 1
            print(
                f"[vllm-clean-targets] {completed}/{len(records)} "
                f"id={identifiers[index]}",
                flush=True,
            )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    rows = []
    with open(args.out, "w", encoding="utf-8") as output:
        for record, outcome in zip(records, outcomes):
            if outcome is None:
                raise RuntimeError("reader outcome collection is incomplete")
            answer = str(outcome["answer"])
            status = str(outcome["status"])
            accepted_gold_answers = gold_answers(record)
            row = {
                "id": record_id(record),
                "question": str(record.get("question", "")),
                "gold_answer": str(record.get("answer", "")),
                "gold_answers": accepted_gold_answers,
                "clean_answer": answer,
                "status": status,
                "raw_reader_answer": str(outcome["raw_reader_answer"]),
                "protocol_error": str(outcome["protocol_error"]),
                "clean_correct": any(
                    answers_exact_match(answer, gold)
                    for gold in accepted_gold_answers
                )
                if status == "ok"
                else False,
                "reader_backend": "vllm_openai_compatible",
                "served_model": reader.model,
                "retrieved_chunks": args.k,
                "decoding": "greedy",
            }
            rows.append(row)
            output.write(json.dumps(row, ensure_ascii=False) + "\n")

    elapsed = time.monotonic() - started
    summary = {
        "records": len(rows),
        "nonempty_targets": sum(row["status"] == "ok" for row in rows),
        "reader_abstentions": sum(
            row["status"] == "reader_abstention_empty_answer" for row in rows
        ),
        "protocol_violations": sum(
            row["status"] == "protocol_violation" for row in rows
        ),
        "clean_correct": sum(bool(row["clean_correct"]) for row in rows),
        "clean_exact_match": (
            sum(bool(row["clean_correct"]) for row in rows) / len(rows)
            if rows
            else None
        ),
        "reader_backend": "vllm_openai_compatible",
        "served_model": reader.model,
        "workers": args.workers,
        "retrieved_chunks": args.k,
        "elapsed_seconds": round(elapsed, 3),
        "queries_per_second": round(len(rows) / elapsed, 3) if elapsed else None,
        "out": args.out,
    }
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    print("[vllm-clean-targets summary]", rendered)
    if args.summary_out:
        os.makedirs(
            os.path.dirname(os.path.abspath(args.summary_out)),
            exist_ok=True,
        )
        with open(args.summary_out, "w", encoding="utf-8") as output:
            output.write(rendered + "\n")


if __name__ == "__main__":
    main()
