"""Evaluate explanation-guided and RobustRAG defenses on saved attacks."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.io import load_records
from causalityrag.reader import ReaderClient, answers_match, normalize_answer


ATTACKS = ("poisonedrag", "tparag")
RANDOM_SEEDS = (11, 23, 47, 89, 131)


def word_tokens(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", text or "", flags=re.UNICODE)


def reader_answer(
    reader: ReaderClient,
    question: str,
    contexts: list[dict[str, Any]],
) -> str:
    last_error: Exception | None = None
    for attempt in range(6):
        try:
            return reader.answer(question, contexts)
        except Exception as exc:  # pragma: no cover - integration retries
            last_error = exc
            time.sleep(min(8.0, 0.5 * (2**attempt)))
    raise RuntimeError(f"reader request failed: {last_error}")


def selected_unit_ids(reflow: dict[str, Any]) -> list[str]:
    selected = [str(value) for value in reflow.get("selected_ids", [])]
    if selected:
        return selected
    executed = [
        attempt
        for attempt in reflow.get("attempts", [])
        if attempt.get("reader_called") and attempt.get("selected_ids")
    ]
    if executed:
        return [str(value) for value in executed[-1]["selected_ids"]]
    candidates = [
        attempt
        for attempt in reflow.get("attempts", [])
        if attempt.get("selected_ids")
    ]
    return [str(value) for value in candidates[0].get("selected_ids", [])] if candidates else []


def reflow_verified_sentences(
    units_row: dict[str, Any],
    reflow: dict[str, Any],
) -> list[dict[str, Any]]:
    by_unit = {
        str(unit.get("unit_id", "")): unit for unit in units_row.get("units", [])
    }
    sentence_ids = {
        str(by_unit[unit_id].get("sentence_id", ""))
        for unit_id in selected_unit_ids(reflow)
        if unit_id in by_unit
    }
    return [
        dict(sentence)
        for sentence in units_row.get("sentences", [])
        if str(sentence.get("sentence_id", "")) in sentence_ids
    ]


def sentence_contexts(sentences: list[dict[str, Any]], prefix: str) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": f"{prefix}:{index}",
            "title": "Verified retrieved evidence",
            "text": str(sentence.get("text", "")),
            "rank": index + 1,
        }
        for index, sentence in enumerate(sentences)
        if str(sentence.get("text", "")).strip()
    ]


def majority_answer(answers: list[str]) -> str:
    normalized = [normalize_answer(answer) for answer in answers]
    counts = Counter(value for value in normalized if value)
    if not counts:
        return "I don't know"
    winner, frequency = counts.most_common(1)[0]
    if frequency < 2:
        return "I don't know"
    return answers[normalized.index(winner)]


def keyword_hints(answers: list[str]) -> list[str]:
    stop = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "in", "is", "it", "of", "on", "or", "that", "the", "to", "was",
        "were", "with", "i", "dont", "know",
    }
    per_answer = []
    for answer in answers:
        per_answer.append(
            {
                token
                for token in normalize_answer(answer).split()
                if token not in stop and len(token) > 1
            }
        )
    counts = Counter(token for values in per_answer for token in values)
    return sorted(token for token, count in counts.items() if count >= 2)


def robust_rag(
    question: str,
    contexts: list[dict[str, Any]],
    reader: ReaderClient,
) -> dict[str, Any]:
    isolated_answers = [reader_answer(reader, question, [context]) for context in contexts]
    voting = majority_answer(isolated_answers)
    hints = keyword_hints(isolated_answers)
    if hints:
        keyword = reader_answer(
            reader,
            question,
            [
                {
                    "chunk_id": "robustrag:keywords",
                    "text": "Consensus evidence keywords: " + ", ".join(hints),
                    "rank": 1,
                }
            ],
        )
        keyword_calls = 1
    else:
        keyword = "I don't know"
        keyword_calls = 0
    footprint = sum(len(word_tokens(str(context.get("text", "")))) for context in contexts)
    return {
        "isolated_answers": isolated_answers,
        "voting_answer": voting,
        "keyword_answer": keyword,
        "keyword_hints": hints,
        "defense_token_footprint": footprint,
        "reader_calls": len(contexts) + keyword_calls,
    }


def random_verified_sentences(
    all_sentences: list[dict[str, Any]],
    count: int,
    *,
    query_id: str,
    seed: int,
) -> list[dict[str, Any]]:
    if count <= 0 or not all_sentences:
        return []
    rng = random.Random(f"{seed}:{query_id}")
    indices = list(range(len(all_sentences)))
    rng.shuffle(indices)
    return [dict(all_sentences[index]) for index in indices[:count]]


def evaluate_query(
    attack_row: dict[str, Any],
    units_row: dict[str, Any],
    reflow: dict[str, Any],
    reader: ReaderClient,
) -> dict[str, Any]:
    identifier = str(attack_row.get("id", ""))
    question = str(attack_row.get("question", ""))
    gold = str(attack_row.get("gold_answer", ""))
    reflow_sentences = reflow_verified_sentences(units_row, reflow)
    reflow_contexts = sentence_contexts(reflow_sentences, "reflow-guard")
    reflow_answer = (
        reader_answer(reader, question, reflow_contexts)
        if reflow_contexts
        else "I don't know"
    )
    reflow_footprint = sum(
        len(word_tokens(str(sentence.get("text", ""))))
        for sentence in reflow_sentences
    )

    random_rows = []
    all_sentences = list(units_row.get("sentences", []))
    for seed in RANDOM_SEEDS:
        sentences = random_verified_sentences(
            all_sentences,
            len(reflow_sentences),
            query_id=identifier,
            seed=seed,
        )
        contexts = sentence_contexts(sentences, f"random-guard:{seed}")
        answer = reader_answer(reader, question, contexts) if contexts else "I don't know"
        random_rows.append(
            {
                "seed": seed,
                "answer": answer,
                "correct": answers_match(answer, gold),
                "defense_token_footprint": sum(
                    len(word_tokens(str(sentence.get("text", ""))))
                    for sentence in sentences
                ),
                "reader_calls": int(bool(contexts)),
            }
        )

    attacks = {}
    for attack_name in ATTACKS:
        attack = attack_row["methods"][attack_name]
        contexts = [dict(row) for row in attack.get("attacked_contexts", [])]
        robust = robust_rag(question, contexts, reader)
        no_defense_answer = str(attack.get("attacked_answer", ""))
        attacks[attack_name] = {
            "no_defense": {
                "answer": no_defense_answer,
                "correct": answers_match(no_defense_answer, gold),
                "defense_token_footprint": 0,
                "reader_calls": 0,
            },
            "reflow_guard": {
                "answer": reflow_answer,
                "correct": answers_match(reflow_answer, gold),
                "defense_token_footprint": reflow_footprint,
                "verified_sentence_ids": [
                    str(sentence.get("sentence_id", ""))
                    for sentence in reflow_sentences
                ],
                "reader_calls": int(bool(reflow_contexts)),
            },
            "random_guard": {
                "seed_results": random_rows,
                "correct": statistics.fmean(
                    int(row["correct"]) for row in random_rows
                ),
                "defense_token_footprint": statistics.fmean(
                    int(row["defense_token_footprint"]) for row in random_rows
                ),
                "reader_calls": sum(int(row["reader_calls"]) for row in random_rows),
            },
            "robustrag_voting": {
                "answer": robust["voting_answer"],
                "correct": answers_match(robust["voting_answer"], gold),
                "defense_token_footprint": robust["defense_token_footprint"],
                "reader_calls": len(contexts),
            },
            "robustrag_keyword": {
                "answer": robust["keyword_answer"],
                "correct": answers_match(robust["keyword_answer"], gold),
                "defense_token_footprint": robust["defense_token_footprint"],
                "reader_calls": robust["reader_calls"],
                "keyword_hints": robust["keyword_hints"],
            },
        }
    return {"id": identifier, "gold_answer": gold, "attacks": attacks}


def bootstrap_interval(values: list[float], *, seed: int = 0) -> list[float]:
    if not values:
        return [0.0, 0.0]
    rng = random.Random(seed)
    means = []
    for _ in range(10_000):
        means.append(statistics.fmean(rng.choice(values) for _ in values))
    means.sort()
    return [means[249], means[9749]]


def summarize(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    methods = (
        "no_defense",
        "reflow_guard",
        "random_guard",
        "robustrag_voting",
        "robustrag_keyword",
    )
    output: dict[str, Any] = {"queries": len(rows), "attacks": {}, "macro": {}}
    for attack_name in ATTACKS:
        output["attacks"][attack_name] = {}
        for method in methods:
            values = [float(row["attacks"][attack_name][method]["correct"]) for row in rows]
            footprints = [
                float(row["attacks"][attack_name][method]["defense_token_footprint"])
                for row in rows
            ]
            output["attacks"][attack_name][method] = {
                "attacked_accuracy": statistics.fmean(values),
                "attacked_accuracy_ci95": bootstrap_interval(values, seed=71),
                "mean_defense_token_footprint": statistics.fmean(footprints),
            }
    for method in methods:
        values = [
            float(row["attacks"][attack_name][method]["correct"])
            for row in rows
            for attack_name in ATTACKS
        ]
        footprints = [
            float(row["attacks"][attack_name][method]["defense_token_footprint"])
            for row in rows
            for attack_name in ATTACKS
        ]
        output["macro"][method] = {
            "attacked_accuracy": statistics.fmean(values),
            "attacked_accuracy_ci95": bootstrap_interval(values, seed=73),
            "mean_defense_token_footprint": statistics.fmean(footprints),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attacks", required=True)
    parser.add_argument("--units", required=True)
    parser.add_argument("--reflow", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--llm-base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--llm-model", default="qwen2.5-7b")
    args = parser.parse_args()

    attacks = load_records(args.attacks)
    units_by_id = {str(row.get("id", "")): row for row in load_records(args.units)}
    reflow_by_id = {str(row.get("id", "")): row for row in load_records(args.reflow)}
    completed = {}
    if os.path.exists(args.out):
        completed = {str(row.get("id", "")): row for row in load_records(args.out)}
    pending = [row for row in attacks if str(row.get("id", "")) not in completed]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    reader = ReaderClient(base_url=args.llm_base_url, model=args.llm_model)
    write_lock = threading.Lock()

    def execute(row: dict[str, Any]) -> dict[str, Any]:
        identifier = str(row.get("id", ""))
        if identifier not in units_by_id or identifier not in reflow_by_id:
            raise ValueError(f"missing aligned defense input for {identifier}")
        return evaluate_query(row, units_by_id[identifier], reflow_by_id[identifier], reader)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(execute, row): row for row in pending}
        errors = []
        for future in as_completed(futures):
            source = futures[future]
            try:
                row = future.result()
            except Exception as exc:  # pragma: no cover - integration resilience
                identifier = str(source.get("id", ""))
                errors.append((identifier, str(exc)))
                print(
                    f"[security-defenses] ERROR id={identifier}: {exc}",
                    flush=True,
                )
                continue
            with write_lock:
                with open(args.out, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            completed[str(row["id"])] = row
            partial = summarize(completed.values())
            print(
                f"[security-defenses] {len(completed)}/{len(attacks)} "
                + " ".join(
                    f"{name}={100*stats['attacked_accuracy']:.1f}%"
                    for name, stats in partial["macro"].items()
                ),
                flush=True,
            )

    if errors:
        raise RuntimeError(
            f"{len(errors)} defense queries failed; rerun to resume. "
            f"First failures: {errors[:3]}"
        )

    ordered = [completed[str(row["id"])] for row in attacks]
    with open(args.out, "w", encoding="utf-8") as handle:
        for row in ordered:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    result = summarize(ordered)
    result["protocol"] = {
        "attack_population": list(ATTACKS),
        "reflow_guard": "verify clean evidence sentences containing ReFlow tokens",
        "random_guard_seeds": list(RANDOM_SEEDS),
        "robustrag": "per-passage isolation with voting or keyword aggregation",
        "footprint": "tokens verified or isolated before defended generation",
    }
    with open(args.summary_out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
