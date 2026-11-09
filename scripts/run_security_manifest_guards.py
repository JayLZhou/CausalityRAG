"""Evaluate aligned sentence rankings with a frozen corpus-integrity guard.

The clean vector-store manifest is built before any attack and contains each
indexed chunk identifier and a hash of its text.  At defense time, every
method may submit the same fixed number of sentences for verification.  A
selected chunk is rejected only when its identifier is absent from the frozen
manifest or its content hash has changed.  Attack labels are used solely for
post-hoc detection statistics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.io import load_records, record_id, retrieved_contexts
from causalityrag.reader import ReaderClient, answers_match
from scripts.run_security_aligned_guards import ATTACKS, RANDOM_SEEDS
from scripts.run_security_pareto_attacks import reader_answer


METHODS = ("no_defense", "random_guard", "keyword_guard", "decoding_guard", "reflow_guard")


def text_hash(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def clean_manifest(rows: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        result[record_id(row)] = {
            str(context.get("chunk_id", "")): text_hash(str(context.get("text", "")))
            for context in retrieved_contexts(row)
        }
    return result


def checked_chunks(
    selected_ids: list[str],
    sentence_by_id: dict[str, dict[str, Any]],
) -> set[str]:
    return {
        str(sentence_by_id[sentence_id].get("chunk_id", ""))
        for sentence_id in selected_ids
        if sentence_id in sentence_by_id
    }


def rejected_by_manifest(
    *,
    selected_ids: list[str],
    sentence_by_id: dict[str, dict[str, Any]],
    context_by_id: dict[str, dict[str, Any]],
    trusted: dict[str, str],
) -> set[str]:
    rejected = set()
    for chunk_id in checked_chunks(selected_ids, sentence_by_id):
        context = context_by_id.get(chunk_id)
        if context is None or trusted.get(chunk_id) != text_hash(str(context.get("text", ""))):
            rejected.add(chunk_id)
    return rejected


def defend_selection(
    *,
    question: str,
    gold: str,
    attacked_answer: str,
    contexts: list[dict[str, Any]],
    selected_ids: list[str],
    sentence_by_id: dict[str, dict[str, Any]],
    trusted: dict[str, str],
    attack_name: str,
    reader: ReaderClient,
) -> dict[str, Any]:
    context_by_id = {
        str(context.get("chunk_id", "")): context for context in contexts
    }
    rejected = rejected_by_manifest(
        selected_ids=selected_ids,
        sentence_by_id=sentence_by_id,
        context_by_id=context_by_id,
        trusted=trusted,
    )
    if rejected:
        retained = [
            dict(context)
            for context in contexts
            if str(context.get("chunk_id", "")) not in rejected
        ]
        answer = reader_answer(reader, question, retained) if retained else "I don't know"
        reader_calls = int(bool(retained))
    else:
        answer = attacked_answer
        reader_calls = 0
    poison_chunks = {
        chunk_id for chunk_id in context_by_id if chunk_id.startswith(attack_name + ":")
    }
    return {
        "selected_sentence_ids": selected_ids,
        "checked_chunk_count": len(checked_chunks(selected_ids, sentence_by_id)),
        "rejected_chunk_count": len(rejected),
        "poison_removed": bool(rejected & poison_chunks),
        "false_positive_chunk_count": len(rejected - poison_chunks),
        "answer": answer,
        "correct": answers_match(answer, gold),
        "reader_calls": reader_calls,
    }


def evaluate_one(
    *,
    selection_row: dict[str, Any],
    source: dict[str, Any],
    units_row: dict[str, Any],
    trusted: dict[str, str],
    reader: ReaderClient,
) -> dict[str, Any]:
    identifier = str(selection_row.get("id", ""))
    attack_name = str(selection_row.get("attack", ""))
    question = str(source.get("question", ""))
    gold = str(source.get("gold_answer", ""))
    attack = source.get("methods", {}).get(attack_name, {})
    contexts = [dict(row) for row in attack.get("attacked_contexts", [])]
    attacked_answer = str(attack.get("attacked_answer", ""))
    sentence_by_id = {
        str(row.get("sentence_id", "")): row for row in units_row.get("sentences", [])
    }
    result = {
        "id": identifier,
        "attack": attack_name,
        "gold_answer": gold,
        "sentence_budget": int(selection_row.get("sentence_budget", 0)),
        "no_defense": {
            "answer": attacked_answer,
            "correct": answers_match(attacked_answer, gold),
        },
    }
    for method in ("keyword_guard", "decoding_guard", "reflow_guard"):
        result[method] = defend_selection(
            question=question,
            gold=gold,
            attacked_answer=attacked_answer,
            contexts=contexts,
            selected_ids=list(selection_row[method]["selected_sentence_ids"]),
            sentence_by_id=sentence_by_id,
            trusted=trusted,
            attack_name=attack_name,
            reader=reader,
        )
    result["random_guard"] = {"seeds": []}
    for seed in RANDOM_SEEDS:
        source_seed = next(
            row for row in selection_row["random_guard"]["seeds"] if int(row["seed"]) == seed
        )
        result["random_guard"]["seeds"].append(
            {
                "seed": seed,
                **defend_selection(
                    question=question,
                    gold=gold,
                    attacked_answer=attacked_answer,
                    contexts=contexts,
                    selected_ids=list(source_seed["selected_sentence_ids"]),
                    sentence_by_id=sentence_by_id,
                    trusted=trusted,
                    attack_name=attack_name,
                    reader=reader,
                ),
            }
        )
    return result


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "queries": len({str(row.get("id", "")) for row in rows}),
        "attack_rows": len(rows),
        "sentence_budget": rows[0]["sentence_budget"] if rows else 0,
        "methods": {},
        "attacks": {},
        "protocol": {
            "ranking_budget": "exactly three sentences per query and attack",
            "verification": "chunk ID and SHA-256 text hash against the frozen pre-attack vector-store manifest",
            "response": "reject failed chunks and regenerate once from the retained context",
            "oracle_access": "none during ranking or verification; attack labels are used only for post-hoc diagnostics",
            "random_seeds": list(RANDOM_SEEDS),
        },
    }

    def method_values(selected_rows: list[dict[str, Any]], method: str) -> tuple[list[int], list[int], list[int]]:
        if method == "random_guard":
            values = [value for row in selected_rows for value in row[method]["seeds"]]
        else:
            values = [row[method] for row in selected_rows]
        correct = [int(value["correct"]) for value in values]
        removals = [int(value["poison_removed"]) for value in values] if method != "no_defense" else []
        false_positives = [int(value["false_positive_chunk_count"]) for value in values] if method != "no_defense" else []
        return correct, removals, false_positives

    for method in METHODS:
        correct, removals, false_positives = method_values(rows, method)
        item = {"attacked_accuracy": statistics.fmean(correct)}
        if removals:
            item["poison_removal_rate"] = statistics.fmean(removals)
            item["false_positive_chunks"] = statistics.fmean(false_positives)
        if method == "random_guard":
            seed_accuracy = []
            for seed in RANDOM_SEEDS:
                values = [
                    next(value for value in row[method]["seeds"] if value["seed"] == seed)
                    for row in rows
                ]
                seed_accuracy.append(statistics.fmean(int(value["correct"]) for value in values))
            item["attacked_accuracy_std"] = statistics.pstdev(seed_accuracy)
        result["methods"][method] = item

    for attack_name in ATTACKS:
        selected = [row for row in rows if row.get("attack") == attack_name]
        result["attacks"][attack_name] = {}
        for method in METHODS:
            correct, removals, false_positives = method_values(selected, method)
            item = {"attacked_accuracy": statistics.fmean(correct)}
            if removals:
                item["poison_removal_rate"] = statistics.fmean(removals)
                item["false_positive_chunks"] = statistics.fmean(false_positives)
            result["attacks"][attack_name][method] = item
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selections", required=True)
    parser.add_argument("--attacks", required=True)
    parser.add_argument("--clean-retrieval", required=True)
    parser.add_argument("--poisonedrag-units", required=True)
    parser.add_argument("--tparag-units", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    selections = load_records(args.selections)
    attacks = {str(row.get("id", "")): row for row in load_records(args.attacks)}
    manifest = clean_manifest(load_records(args.clean_retrieval))
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
            trusted=manifest[identifier],
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
                print(f"[manifest-defense] {len(completed)}/{len(selections)} {key}", flush=True)

    ordered = [
        completed[(str(row.get("id", "")), str(row.get("attack", "")))]
        for row in selections
    ]
    with open(args.out, "w", encoding="utf-8") as output:
        for row in ordered:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    result = summarize(ordered)
    with open(args.summary_out, "w", encoding="utf-8") as output:
        json.dump(result, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
