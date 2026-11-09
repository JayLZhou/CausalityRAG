"""Compare simple non-oracle defenses on frozen attacked RAG contexts.

ReFlow-Guard removes the three sentences prioritized by ReFlow and reruns the
reader. Random-Guard removes three uniformly sampled sentences. RobustRAG's
native keyword aggregation and secure decoding are read from their frozen
outputs and reported without modification.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.io import load_records
from causalityrag.reader import ReaderClient, answers_match
from scripts.run_security_aligned_guards import ATTACKS, RANDOM_SEEDS
from scripts.run_security_pareto_attacks import reader_answer


def remove_selected_sentences(
    contexts: list[dict[str, Any]],
    units_row: dict[str, Any],
    selected_ids: list[str],
) -> list[dict[str, Any]]:
    sentence_by_id = {
        str(row.get("sentence_id", "")): row for row in units_row.get("sentences", [])
    }
    ranges: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for sentence_id in selected_ids:
        sentence = sentence_by_id.get(sentence_id)
        if sentence is None:
            continue
        ranges[str(sentence.get("chunk_id", ""))].append(
            (
                int(sentence.get("chunk_char_start", 0)),
                int(sentence.get("chunk_char_end", 0)),
            )
        )
    result = []
    for context in contexts:
        row = dict(context)
        text = str(row.get("text", ""))
        chunk_ranges = sorted(
            ranges.get(str(row.get("chunk_id", "")), []),
            reverse=True,
        )
        for start, end in chunk_ranges:
            text = text[: max(0, start)] + " " + text[max(0, end) :]
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            row["text"] = text
            result.append(row)
    return result


def answer_after_removal(
    *,
    question: str,
    gold: str,
    contexts: list[dict[str, Any]],
    units_row: dict[str, Any],
    selected_ids: list[str],
    reader: ReaderClient,
) -> dict[str, Any]:
    retained = remove_selected_sentences(contexts, units_row, selected_ids)
    answer = reader_answer(reader, question, retained) if retained else "I don't know"
    return {
        "selected_sentence_ids": selected_ids,
        "answer": answer,
        "correct": answers_match(answer, gold),
        "reader_calls": int(bool(retained)),
    }


def evaluate_one(
    *,
    selection_row: dict[str, Any],
    source: dict[str, Any],
    units_row: dict[str, Any],
    reader: ReaderClient,
) -> dict[str, Any]:
    identifier = str(selection_row.get("id", ""))
    attack_name = str(selection_row.get("attack", ""))
    question = str(source.get("question", ""))
    gold = str(source.get("gold_answer", ""))
    attack = source.get("methods", {}).get(attack_name, {})
    contexts = [dict(row) for row in attack.get("attacked_contexts", [])]
    attacked_answer = str(attack.get("attacked_answer", ""))
    result = {
        "id": identifier,
        "attack": attack_name,
        "gold_answer": gold,
        "sentence_budget": int(selection_row.get("sentence_budget", 0)),
        "no_defense": {
            "answer": attacked_answer,
            "correct": answers_match(attacked_answer, gold),
        },
        "reflow_guard": answer_after_removal(
            question=question,
            gold=gold,
            contexts=contexts,
            units_row=units_row,
            selected_ids=list(selection_row["reflow_guard"]["selected_sentence_ids"]),
            reader=reader,
        ),
        "random_guard": {"seeds": []},
    }
    for seed in RANDOM_SEEDS:
        selected = next(
            row for row in selection_row["random_guard"]["seeds"] if int(row["seed"]) == seed
        )
        result["random_guard"]["seeds"].append(
            {
                "seed": seed,
                **answer_after_removal(
                    question=question,
                    gold=gold,
                    contexts=contexts,
                    units_row=units_row,
                    selected_ids=list(selected["selected_sentence_ids"]),
                    reader=reader,
                ),
            }
        )
    return result


def merge_and_summarize(
    rows: list[dict[str, Any]],
    keyword_rows: list[dict[str, Any]],
    decoding_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    keyword = {str(row.get("id", "")): row for row in keyword_rows}
    decoding = {str(row.get("id", "")): row for row in decoding_rows}
    result: dict[str, Any] = {
        "queries": len({str(row.get("id", "")) for row in rows}),
        "attack_rows": len(rows),
        "sentence_budget": rows[0]["sentence_budget"] if rows else 0,
        "methods": {},
        "attacks": {},
        "protocol": {
            "reflow_guard": "remove the three ReFlow-prioritized sentences and regenerate",
            "random_guard": "remove three random sentences and regenerate; five fixed seeds",
            "keyword": "native RobustRAG secure keyword aggregation",
            "decoding": "native RobustRAG secure decoding aggregation",
            "oracle_access": "no method receives the gold answer, attack label, or poisoned-chunk identity",
        },
    }

    def values(selected: list[dict[str, Any]], method: str) -> list[int]:
        if method == "no_defense":
            return [int(row[method]["correct"]) for row in selected]
        if method == "reflow_guard":
            return [int(row[method]["correct"]) for row in selected]
        if method == "random_guard":
            return [
                int(value["correct"])
                for row in selected
                for value in row[method]["seeds"]
            ]
        if method == "keyword":
            return [
                int(keyword[str(row["id"])] ["attacks"][str(row["attack"])] ["robustrag_keyword"]["correct"])
                for row in selected
            ]
        return [
            int(decoding[str(row["id"])] ["attacks"][str(row["attack"])] ["correct"])
            for row in selected
        ]

    methods = ("no_defense", "random_guard", "keyword", "decoding", "reflow_guard")
    for method in methods:
        result["methods"][method] = {
            "attacked_accuracy": statistics.fmean(values(rows, method))
        }
    for attack_name in ATTACKS:
        selected = [row for row in rows if row.get("attack") == attack_name]
        result["attacks"][attack_name] = {
            method: {"attacked_accuracy": statistics.fmean(values(selected, method))}
            for method in methods
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selections", required=True)
    parser.add_argument("--attacks", required=True)
    parser.add_argument("--poisonedrag-units", required=True)
    parser.add_argument("--tparag-units", required=True)
    parser.add_argument("--keyword-results", required=True)
    parser.add_argument("--decoding-results", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    selections = load_records(args.selections)
    attacks = {str(row.get("id", "")): row for row in load_records(args.attacks)}
    units = {
        "poisonedrag": {str(row.get("id", "")): row for row in load_records(args.poisonedrag_units)},
        "tparag": {str(row.get("id", "")): row for row in load_records(args.tparag_units)},
    }
    completed = {
        (str(row.get("id", "")), str(row.get("attack", ""))): row
        for row in load_records(args.out)
    } if os.path.exists(args.out) else {}
    pending = [
        row for row in selections
        if (str(row.get("id", "")), str(row.get("attack", ""))) not in completed
    ]
    reader = ReaderClient(base_url=args.base_url, model=args.model, timeout=240)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()

    def work(row: dict[str, Any]) -> dict[str, Any]:
        identifier = str(row.get("id", ""))
        attack_name = str(row.get("attack", ""))
        return evaluate_one(
            selection_row=row,
            source=attacks[identifier],
            units_row=units[attack_name][identifier],
            reader=reader,
        )

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(work, row): row for row in pending}
        for future in as_completed(futures):
            row = future.result()
            key = (str(row["id"]), str(row["attack"]))
            with lock:
                completed[key] = row
                with open(args.out, "a", encoding="utf-8") as output:
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"[simple-guard] {len(completed)}/{len(selections)} {key}", flush=True)

    ordered = [
        completed[(str(row.get("id", "")), str(row.get("attack", "")))]
        for row in selections
    ]
    with open(args.out, "w", encoding="utf-8") as output:
        for row in ordered:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    result = merge_and_summarize(
        ordered,
        load_records(args.keyword_results),
        load_records(args.decoding_results),
    )
    with open(args.summary_out, "w", encoding="utf-8") as output:
        json.dump(result, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
