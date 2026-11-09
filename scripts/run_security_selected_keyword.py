"""Apply RobustRAG keyword aggregation to ReFlow- or random-selected sentences."""

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
from causalityrag.linguistics import SpacyAnnotationClient
from causalityrag.reader import ReaderClient, answers_match
from scripts.run_security_aligned_guards import ATTACKS, RANDOM_SEEDS
from scripts.run_security_guard_keyword import keyword_aggregation
from scripts.run_security_pareto_attacks import OpenAIClient


def selected_contexts(
    units_row: dict[str, Any],
    selected_ids: list[str],
) -> list[dict[str, Any]]:
    sentence_by_id = {
        str(row.get("sentence_id", "")): row for row in units_row.get("sentences", [])
    }
    result = []
    for index, sentence_id in enumerate(selected_ids, start=1):
        sentence = sentence_by_id.get(sentence_id)
        if sentence is None or not str(sentence.get("text", "")).strip():
            continue
        result.append(
            {
                "chunk_id": f"selected:{index}",
                "title": "Selected retrieved evidence",
                "text": str(sentence.get("text", "")),
                "rank": index,
            }
        )
    return result


def aggregate(
    *,
    question: str,
    gold: str,
    contexts: list[dict[str, Any]],
    reader: ReaderClient,
    chat: OpenAIClient,
    nlp: SpacyAnnotationClient,
    alpha: float,
    beta: float,
) -> dict[str, Any]:
    result = keyword_aggregation(
        question=question,
        contexts=contexts,
        reader=reader,
        chat=chat,
        nlp=nlp,
        alpha=alpha,
        beta=beta,
    )
    return {**result, "correct": answers_match(str(result["answer"]), gold)}


def evaluate_one(
    *,
    selection_row: dict[str, Any],
    source: dict[str, Any],
    units_row: dict[str, Any],
    reader: ReaderClient,
    chat: OpenAIClient,
    nlp: SpacyAnnotationClient,
    alpha: float,
    beta: float,
) -> dict[str, Any]:
    question = str(source.get("question", ""))
    gold = str(source.get("gold_answer", ""))
    result = {
        "id": str(selection_row.get("id", "")),
        "attack": str(selection_row.get("attack", "")),
        "sentence_budget": int(selection_row.get("sentence_budget", 0)),
    }
    result["reflow_keyword"] = aggregate(
        question=question,
        gold=gold,
        contexts=selected_contexts(
            units_row,
            list(selection_row["reflow_guard"]["selected_sentence_ids"]),
        ),
        reader=reader,
        chat=chat,
        nlp=nlp,
        alpha=alpha,
        beta=beta,
    )
    result["random_keyword"] = {"seeds": []}
    for seed in RANDOM_SEEDS:
        selected = next(
            row for row in selection_row["random_guard"]["seeds"] if int(row["seed"]) == seed
        )
        result["random_keyword"]["seeds"].append(
            {
                "seed": seed,
                **aggregate(
                    question=question,
                    gold=gold,
                    contexts=selected_contexts(
                        units_row,
                        list(selected["selected_sentence_ids"]),
                    ),
                    reader=reader,
                    chat=chat,
                    nlp=nlp,
                    alpha=alpha,
                    beta=beta,
                ),
            }
        )
    return result


def summarize(
    rows: list[dict[str, Any]],
    attacks: list[dict[str, Any]],
    native_keyword: list[dict[str, Any]],
    native_decoding: list[dict[str, Any]],
) -> dict[str, Any]:
    attack_by_id = {str(row.get("id", "")): row for row in attacks}
    keyword_by_id = {str(row.get("id", "")): row for row in native_keyword}
    decoding_by_id = {str(row.get("id", "")): row for row in native_decoding}
    methods = (
        "no_defense",
        "random_keyword",
        "keyword",
        "decoding",
        "reflow_keyword",
    )

    def values(selected: list[dict[str, Any]], method: str) -> list[int]:
        if method == "reflow_keyword":
            return [int(row[method]["correct"]) for row in selected]
        if method == "random_keyword":
            return [
                int(value["correct"])
                for row in selected
                for value in row[method]["seeds"]
            ]
        if method == "no_defense":
            return [
                int(
                    answers_match(
                        str(attack_by_id[str(row["id"])] ["methods"][str(row["attack"])] ["attacked_answer"]),
                        str(attack_by_id[str(row["id"])] ["gold_answer"]),
                    )
                )
                for row in selected
            ]
        if method == "keyword":
            return [
                int(keyword_by_id[str(row["id"])] ["attacks"][str(row["attack"])] ["robustrag_keyword"]["correct"])
                for row in selected
            ]
        return [
            int(decoding_by_id[str(row["id"])] ["attacks"][str(row["attack"])] ["correct"])
            for row in selected
        ]

    result: dict[str, Any] = {
        "queries": len({str(row.get("id", "")) for row in rows}),
        "attack_rows": len(rows),
        "sentence_budget": rows[0]["sentence_budget"] if rows else 0,
        "methods": {},
        "attacks": {},
        "protocol": {
            "reflow_keyword": "RobustRAG keyword aggregation over three ReFlow-selected sentences",
            "random_keyword": "the same aggregation over three random sentences; five fixed seeds",
            "keyword": "native RobustRAG keyword aggregation over all ten attacked chunks",
            "decoding": "native RobustRAG secure decoding over all ten attacked chunks",
            "oracle_access": "no method receives the gold answer, attack label, or poisoned-chunk identity",
        },
    }
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
    parser.add_argument("--native-keyword", required=True)
    parser.add_argument("--native-decoding", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--spacy-base-url", default="http://127.0.0.1:8021")
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=3.0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    selections = load_records(args.selections)
    attacks = load_records(args.attacks)
    attack_by_id = {str(row.get("id", "")): row for row in attacks}
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
    chat = OpenAIClient(args.base_url, args.model, timeout=240)
    nlp = SpacyAnnotationClient(args.spacy_base_url)
    if not nlp.health().get("ok"):
        raise RuntimeError("spaCy service is unhealthy")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()

    def work(row: dict[str, Any]) -> dict[str, Any]:
        identifier = str(row.get("id", ""))
        attack_name = str(row.get("attack", ""))
        return evaluate_one(
            selection_row=row,
            source=attack_by_id[identifier],
            units_row=units[attack_name][identifier],
            reader=reader,
            chat=chat,
            nlp=nlp,
            alpha=args.alpha,
            beta=args.beta,
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
                print(f"[selected-keyword] {len(completed)}/{len(selections)} {key}", flush=True)

    ordered = [
        completed[(str(row.get("id", "")), str(row.get("attack", "")))]
        for row in selections
    ]
    with open(args.out, "w", encoding="utf-8") as output:
        for row in ordered:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    result = summarize(
        ordered,
        attacks,
        load_records(args.native_keyword),
        load_records(args.native_decoding),
    )
    with open(args.summary_out, "w", encoding="utf-8") as output:
        json.dump(result, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
