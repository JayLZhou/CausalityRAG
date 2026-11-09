"""Evaluate pre-attack ReFlow evidence prioritization under RAG poisoning."""

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
from scripts.run_security_pareto_attacks import reader_answer


def clean_sentence_ids(
    reflow_row: dict[str, Any],
    units_row: dict[str, Any],
) -> list[str]:
    sentence_by_unit = {
        str(unit.get("unit_id", "")): str(unit.get("sentence_id", ""))
        for unit in units_row.get("units", [])
    }
    selected_ids = [str(value) for value in reflow_row.get("selected_ids", [])]
    if not selected_ids:
        attempts = [
            row for row in reflow_row.get("attempts", [])
            if row.get("selected_ids")
        ]
        executed = [row for row in attempts if row.get("reader_called")]
        source = executed[-1] if executed else attempts[0] if attempts else {}
        selected_ids = [str(value) for value in source.get("selected_ids", [])]
    result = []
    seen = set()
    for unit_id in selected_ids:
        sentence_id = sentence_by_unit.get(str(unit_id), "")
        if sentence_id and sentence_id not in seen:
            seen.add(sentence_id)
            result.append(sentence_id)
    return result


def protected_contexts(
    sentence_ids: list[str],
    units_row: dict[str, Any],
) -> list[dict[str, Any]]:
    wanted = set(sentence_ids)
    rows = [
        row for row in units_row.get("sentences", [])
        if str(row.get("sentence_id", "")) in wanted
    ]
    rows.sort(
        key=lambda row: (
            int(row.get("chunk_rank", 10**9)),
            int(row.get("sentence_index", 10**9)),
        )
    )
    return [
        {
            "chunk_id": f"TRUSTED-{index}",
            "text": str(row.get("text", "")),
            "rank": index,
        }
        for index, row in enumerate(rows, start=1)
        if str(row.get("text", "")).strip()
    ]


def random_clean_sentence_ids(
    units_row: dict[str, Any],
    budget: int,
    *,
    identifier: str,
    seed: int,
) -> list[str]:
    candidates = [
        str(row.get("sentence_id", ""))
        for row in units_row.get("sentences", [])
        if int(row.get("chunk_rank", 10**9)) <= 5
    ]
    random.Random(f"protected:{seed}:{identifier}").shuffle(candidates)
    return candidates[: min(budget, len(candidates))]


def anonymize_attacked(contexts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            **dict(context),
            "chunk_id": f"E{index}",
            "rank": index,
        }
        for index, context in enumerate(contexts, start=1)
    ]


def prioritized_answer(
    *,
    question: str,
    gold: str,
    protected: list[dict[str, Any]],
    attacked: list[dict[str, Any]],
    reader: ReaderClient,
) -> dict[str, Any]:
    answer = reader_answer(reader, question, protected + attacked)
    return {
        "answer": answer,
        "correct": answers_match(answer, gold),
        "protected_sentences": len(protected),
    }


def evaluate_one(
    *,
    source: dict[str, Any],
    attack_name: str,
    reflow_row: dict[str, Any],
    units_row: dict[str, Any],
    reader: ReaderClient,
) -> dict[str, Any]:
    identifier = str(source.get("id", ""))
    question = str(source.get("question", ""))
    gold = str(source.get("gold_answer", ""))
    attack = source.get("methods", {}).get(attack_name, {})
    attacked = anonymize_attacked([dict(row) for row in attack.get("attacked_contexts", [])])
    reflow_ids = clean_sentence_ids(reflow_row, units_row)
    result = {
        "id": identifier,
        "attack": attack_name,
        "gold_answer": gold,
        "sentence_budget": len(reflow_ids),
        "no_defense": prioritized_answer(
            question=question,
            gold=gold,
            protected=[],
            attacked=attacked,
            reader=reader,
        ),
        "reflow_protect": prioritized_answer(
            question=question,
            gold=gold,
            protected=protected_contexts(reflow_ids, units_row),
            attacked=attacked,
            reader=reader,
        ),
        "random_protect": {"seeds": []},
    }
    for seed in RANDOM_SEEDS:
        random_ids = random_clean_sentence_ids(
            units_row,
            len(reflow_ids),
            identifier=identifier,
            seed=seed,
        )
        result["random_protect"]["seeds"].append(
            {
                "seed": seed,
                **prioritized_answer(
                    question=question,
                    gold=gold,
                    protected=protected_contexts(random_ids, units_row),
                    attacked=attacked,
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
    methods = ("no_defense", "random_protect", "keyword", "decoding", "reflow_protect")

    def values(selected: list[dict[str, Any]], method: str) -> list[int]:
        if method in ("no_defense", "reflow_protect"):
            return [int(row[method]["correct"]) for row in selected]
        if method == "random_protect":
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

    result: dict[str, Any] = {
        "queries": len({str(row.get("id", "")) for row in rows}),
        "attack_rows": len(rows),
        "mean_protected_sentences": statistics.fmean(row["sentence_budget"] for row in rows),
        "methods": {},
        "attacks": {},
        "protocol": {
            "reflow_protect": "prepend the clean ReFlow-selected sentences saved before attack",
            "random_protect": "prepend the same number of random clean sentences; five fixed seeds",
            "keyword": "native RobustRAG secure keyword aggregation",
            "decoding": "native RobustRAG secure decoding",
            "oracle_access": "the defense never receives the gold answer, attack label, or poisoned-chunk identity",
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
    completed = {
        (str(row.get("id", "")), str(row.get("attack", ""))): row
        for row in load_records(args.out)
    } if os.path.exists(args.out) else {}
    pending = [
        (source, attack_name)
        for source in attacks
        for attack_name in ATTACKS
        if (str(source.get("id", "")), attack_name) not in completed
    ]
    reader = ReaderClient(base_url=args.base_url, model=args.model, timeout=240)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()

    def work(item: tuple[dict[str, Any], str]) -> dict[str, Any]:
        source, attack_name = item
        identifier = str(source.get("id", ""))
        return evaluate_one(
            source=source,
            attack_name=attack_name,
            reflow_row=reflow[identifier],
            units_row=units[identifier],
            reader=reader,
        )

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(work, item): item for item in pending}
        for future in as_completed(futures):
            row = future.result()
            key = (str(row["id"]), str(row["attack"]))
            with lock:
                completed[key] = row
                with open(args.out, "a", encoding="utf-8") as output:
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"[protected-evidence] {len(completed)}/{len(attacks) * len(ATTACKS)} {key}", flush=True)

    ordered = [
        completed[(str(source.get("id", "")), attack_name)]
        for source in attacks
        for attack_name in ATTACKS
    ]
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
