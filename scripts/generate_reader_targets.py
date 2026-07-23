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

from causalityrag.io import iter_records, record_id, retrieved_contexts
from causalityrag.reader import ReaderClient, answers_exact_match


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
    parser.add_argument("--base-url", default="")
    parser.add_argument("--served-model", default="")
    args = parser.parse_args()
    if args.start < 0 or args.n < 0 or args.k <= 0 or args.workers <= 0:
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
    answers = [""] * len(records)
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                reader.answer,
                str(record.get("question", "")),
                retrieved_contexts(record)[: args.k],
            ): index
            for index, record in enumerate(records)
        }
        completed = 0
        for future in as_completed(futures):
            index = futures[future]
            answers[index] = future.result()
            completed += 1
            print(
                f"[vllm-clean-targets] {completed}/{len(records)} "
                f"id={identifiers[index]}",
                flush=True,
            )

    empty_answers = [
        identifiers[index]
        for index, answer in enumerate(answers)
        if not answer.strip()
    ]
    if empty_answers:
        raise RuntimeError(
            "vLLM returned empty clean answers; refusing to freeze unusable "
            f"targets: {empty_answers[:10]}"
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    rows = []
    with open(args.out, "w", encoding="utf-8") as output:
        for record, answer in zip(records, answers):
            row = {
                "id": record_id(record),
                "question": str(record.get("question", "")),
                "gold_answer": str(record.get("answer", "")),
                "clean_answer": answer,
                "clean_correct": answers_exact_match(
                    answer,
                    str(record.get("answer", "")),
                ),
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
