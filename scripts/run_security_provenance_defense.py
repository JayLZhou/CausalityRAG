"""Evaluate a proactive defense using only clean ReFlow provenance sentences."""

from __future__ import annotations

import argparse
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
from causalityrag.reader import ReaderClient, answers_match
from scripts.run_security_aligned_guards import ATTACKS, RANDOM_SEEDS
from scripts.run_security_protected_evidence import (
    clean_sentence_ids,
    protected_contexts,
    random_clean_sentence_ids,
)
from scripts.run_security_pareto_attacks import reader_answer


def answer_from_sentences(
    *,
    question: str,
    gold: str,
    sentence_ids: list[str],
    units_row: dict[str, Any],
    reader: ReaderClient,
) -> dict[str, Any]:
    contexts = protected_contexts(sentence_ids, units_row)
    answer = reader_answer(reader, question, contexts) if contexts else "I don't know"
    return {
        "answer": answer,
        "correct": answers_match(answer, gold),
        "sentence_ids": sentence_ids,
        "sentences": len(contexts),
    }


def evaluate_one(
    *,
    source: dict[str, Any],
    reflow_row: dict[str, Any],
    units_row: dict[str, Any],
    reader: ReaderClient,
) -> dict[str, Any]:
    identifier = str(source.get("id", ""))
    question = str(source.get("question", ""))
    gold = str(source.get("gold_answer", ""))
    reflow_ids = clean_sentence_ids(reflow_row, units_row)
    result = {
        "id": identifier,
        "gold_answer": gold,
        "sentence_budget": len(reflow_ids),
        "reflow": answer_from_sentences(
            question=question,
            gold=gold,
            sentence_ids=reflow_ids,
            units_row=units_row,
            reader=reader,
        ),
        "random": {"seeds": []},
    }
    for seed in RANDOM_SEEDS:
        random_ids = random_clean_sentence_ids(
            units_row,
            len(reflow_ids),
            identifier=identifier,
            seed=seed,
        )
        result["random"]["seeds"].append(
            {
                "seed": seed,
                **answer_from_sentences(
                    question=question,
                    gold=gold,
                    sentence_ids=random_ids,
                    units_row=units_row,
                    reader=reader,
                ),
            }
        )
    return result


def summarize(
    rows: list[dict[str, Any]],
    native_keyword: list[dict[str, Any]],
    native_decoding: list[dict[str, Any]],
) -> dict[str, Any]:
    keyword = {str(row.get("id", "")): row for row in native_keyword}
    decoding = {str(row.get("id", "")): row for row in native_decoding}

    def reflow_values() -> list[int]:
        return [int(row["reflow"]["correct"]) for row in rows]

    def random_values() -> list[int]:
        return [
            int(value["correct"])
            for row in rows
            for value in row["random"]["seeds"]
        ]

    def native_values(method: str, attack: str | None = None) -> list[int]:
        selected_attacks = (attack,) if attack else ATTACKS
        if method == "keyword":
            return [
                int(keyword[str(row["id"])] ["attacks"][name]["robustrag_keyword"]["correct"])
                for row in rows
                for name in selected_attacks
            ]
        return [
            int(decoding[str(row["id"])] ["attacks"][name]["correct"])
            for row in rows
            for name in selected_attacks
        ]

    result: dict[str, Any] = {
        "queries": len(rows),
        "mean_sentences": statistics.fmean(row["sentence_budget"] for row in rows),
        "empty_reflow_queries": sum(not row["sentence_budget"] for row in rows),
        "methods": {
            "random": {"accuracy_under_attack": statistics.fmean(random_values())},
            "keyword": {"accuracy_under_attack": statistics.fmean(native_values("keyword"))},
            "decoding": {"accuracy_under_attack": statistics.fmean(native_values("decoding"))},
            "reflow": {"accuracy_under_attack": statistics.fmean(reflow_values())},
        },
        "attacks": {},
        "protocol": {
            "reflow": "answer using only clean sentences containing ReFlow-selected tokens",
            "random": "answer using the same number of random clean sentences; five fixed seeds",
            "keyword": "native RobustRAG secure keyword aggregation over the attacked top-10 context",
            "decoding": "native RobustRAG secure decoding over the attacked top-10 context",
            "oracle_access": "no defense receives the gold answer or poisoned-passage identity",
        },
    }
    for attack in ATTACKS:
        result["attacks"][attack] = {
            "random": {"accuracy_under_attack": result["methods"]["random"]["accuracy_under_attack"]},
            "keyword": {"accuracy_under_attack": statistics.fmean(native_values("keyword", attack))},
            "decoding": {"accuracy_under_attack": statistics.fmean(native_values("decoding", attack))},
            "reflow": {"accuracy_under_attack": result["methods"]["reflow"]["accuracy_under_attack"]},
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attacks", required=True)
    parser.add_argument("--clean-reflow", required=True)
    parser.add_argument("--clean-units", required=True)
    parser.add_argument("--native-keyword", required=True)
    parser.add_argument("--native-decoding", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    attacks = load_records(args.attacks)
    reflow = {record_id(row): row for row in load_records(args.clean_reflow)}
    units = {record_id(row): row for row in load_records(args.clean_units)}
    completed = (
        {str(row.get("id", "")): row for row in load_records(args.out)}
        if os.path.exists(args.out)
        else {}
    )
    pending = [row for row in attacks if str(row.get("id", "")) not in completed]
    reader = ReaderClient(base_url=args.base_url, model=args.model, timeout=240)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()

    def work(source: dict[str, Any]) -> dict[str, Any]:
        identifier = str(source.get("id", ""))
        return evaluate_one(
            source=source,
            reflow_row=reflow[identifier],
            units_row=units[identifier],
            reader=reader,
        )

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(work, source): source for source in pending}
        for future in as_completed(futures):
            row = future.result()
            with lock:
                completed[str(row["id"])] = row
                with open(args.out, "a", encoding="utf-8") as output:
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"[provenance-defense] {len(completed)}/{len(attacks)} {row['id']}", flush=True)

    ordered = [completed[str(row.get("id", ""))] for row in attacks]
    with open(args.out, "w", encoding="utf-8") as output:
        for row in ordered:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    result = summarize(
        ordered,
        load_records(args.native_keyword),
        load_records(args.native_decoding),
    )
    with open(args.summary_out, "w", encoding="utf-8") as output:
        json.dump(result, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
