"""Apply blind verifier verdicts and regenerate defended RAG answers."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.io import load_records
from causalityrag.reader import ReaderClient, answers_match
from scripts.run_security_aligned_guards import ATTACKS, RANDOM_SEEDS
from scripts.run_security_pareto_attacks import reader_answer


METHODS = ("no_defense", "random_guard", "keyword_guard", "decoding_guard", "reflow_guard")


def apply_verdict(
    *,
    question: str,
    gold: str,
    attacked_answer: str,
    contexts: list[dict[str, Any]],
    verdict: dict[str, Any],
    sentence_by_id: dict[str, dict[str, Any]],
    attack_name: str,
    reader: ReaderClient,
) -> dict[str, Any]:
    flagged_chunks = {
        str(sentence_by_id[sentence_id].get("chunk_id", ""))
        for sentence_id in verdict.get("flagged_sentence_ids", [])
        if sentence_id in sentence_by_id
    }
    if flagged_chunks:
        retained = [
            dict(context)
            for context in contexts
            if str(context.get("chunk_id", "")) not in flagged_chunks
        ]
        answer = reader_answer(reader, question, retained) if retained else "I don't know"
        reader_calls = int(bool(retained))
    else:
        answer = attacked_answer
        reader_calls = 0
    poison_chunks = {
        str(context.get("chunk_id", ""))
        for context in contexts
        if str(context.get("chunk_id", "")).startswith(attack_name + ":")
    }
    return {
        **verdict,
        "flagged_chunk_count": len(flagged_chunks),
        "poison_removed": bool(flagged_chunks & poison_chunks),
        "false_positive_chunk_count": len(flagged_chunks - poison_chunks),
        "answer": answer,
        "correct": answers_match(answer, gold),
        "reader_calls": reader_calls,
    }


def evaluate_one(
    *,
    verdict_row: dict[str, Any],
    source: dict[str, Any],
    units_row: dict[str, Any],
    reader: ReaderClient,
) -> dict[str, Any]:
    identifier = str(verdict_row.get("id", ""))
    attack_name = str(verdict_row.get("attack", ""))
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
        "sentence_budget": int(verdict_row.get("sentence_budget", 0)),
        "no_defense": {
            "answer": attacked_answer,
            "correct": answers_match(attacked_answer, gold),
        },
    }
    for method in ("keyword_guard", "decoding_guard", "reflow_guard"):
        result[method] = apply_verdict(
            question=question,
            gold=gold,
            attacked_answer=attacked_answer,
            contexts=contexts,
            verdict=verdict_row[method],
            sentence_by_id=sentence_by_id,
            attack_name=attack_name,
            reader=reader,
        )
    result["random_guard"] = {"seeds": []}
    for seed in RANDOM_SEEDS:
        verdict = next(
            row for row in verdict_row["random_guard"]["seeds"] if int(row["seed"]) == seed
        )
        result["random_guard"]["seeds"].append(
            {
                "seed": seed,
                **apply_verdict(
                    question=question,
                    gold=gold,
                    attacked_answer=attacked_answer,
                    contexts=contexts,
                    verdict=verdict,
                    sentence_by_id=sentence_by_id,
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
            "verifier": "independent Mistral-Nemo-12B blind fact-checker",
            "verifier_inputs": "anonymized selected sentences and attacked retrieved evidence; no source IDs, attack labels, gold answer, or clean answer",
            "response": "remove chunks containing sentences flagged by the verifier and regenerate once with Qwen2.5-7B",
            "random_seeds": list(RANDOM_SEEDS),
        },
    }

    def values(selected_rows: list[dict[str, Any]], method: str) -> list[dict[str, Any]]:
        return (
            [value for row in selected_rows for value in row[method]["seeds"]]
            if method == "random_guard"
            else [row[method] for row in selected_rows]
        )

    for method in METHODS:
        method_rows = values(rows, method)
        item = {
            "attacked_accuracy": statistics.fmean(int(row["correct"]) for row in method_rows)
        }
        if method != "no_defense":
            item.update(
                {
                    "poison_removal_rate": statistics.fmean(int(row["poison_removed"]) for row in method_rows),
                    "false_positive_chunks": statistics.fmean(row["false_positive_chunk_count"] for row in method_rows),
                    "mean_flagged_chunks": statistics.fmean(row["flagged_chunk_count"] for row in method_rows),
                }
            )
        result["methods"][method] = item
    for attack_name in ATTACKS:
        selected = [row for row in rows if row.get("attack") == attack_name]
        result["attacks"][attack_name] = {}
        for method in METHODS:
            method_rows = values(selected, method)
            item = {
                "attacked_accuracy": statistics.fmean(int(row["correct"]) for row in method_rows)
            }
            if method != "no_defense":
                item.update(
                    {
                        "poison_removal_rate": statistics.fmean(int(row["poison_removed"]) for row in method_rows),
                        "false_positive_chunks": statistics.fmean(row["false_positive_chunk_count"] for row in method_rows),
                    }
                )
            result["attacks"][attack_name][method] = item
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verdicts", required=True)
    parser.add_argument("--attacks", required=True)
    parser.add_argument("--poisonedrag-units", required=True)
    parser.add_argument("--tparag-units", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    verdicts = load_records(args.verdicts)
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
        row for row in verdicts
        if (str(row.get("id", "")), str(row.get("attack", ""))) not in completed
    ]
    reader = ReaderClient(base_url=args.base_url, model=args.model, timeout=240)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()

    def work(row: dict[str, Any]) -> dict[str, Any]:
        identifier = str(row.get("id", ""))
        attack_name = str(row.get("attack", ""))
        return evaluate_one(
            verdict_row=row,
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
                print(f"[blind-defense] {len(completed)}/{len(verdicts)} {key}", flush=True)

    ordered = [
        completed[(str(row.get("id", "")), str(row.get("attack", "")))]
        for row in verdicts
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
