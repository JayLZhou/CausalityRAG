"""Evaluate ReFlow-Guard, Random-Guard, and RobustRAG Keyword aggregation."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import statistics
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.linguistics import SpacyAnnotationClient
from causalityrag.reader import (
    ReaderClient,
    answers_match,
    canonicalize_reader_answer,
    parse_json_object,
)
from causalityrag.io import load_records
from scripts.run_security_pareto_attacks import OpenAIClient, reader_answer, word_tokens


ATTACKS = ("poisonedrag", "tparag")
RANDOM_SEEDS = (11, 23, 47, 89, 131)
IGNORE_POS = {
    "VERB", "INTJ", "ADP", "AUX", "CCONJ", "DET", "PART", "PRON",
    "SCONJ", "PUNCT", "SPACE",
}
PUNCTUATION = set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "for",
    "from", "has", "have", "in", "is", "it", "of", "on", "or", "that",
    "the", "to", "was", "were", "with", "i", "you", "he", "she", "they",
}
KEYWORD_SYSTEM = (
    "Answer the question using only the consensus evidence phrases. Give the "
    "shortest answer span. Output JSON only."
)
KEYWORD_USER = """Consensus evidence phrases: {hints}

Question: {question}
Return STRICT JSON: {{"answer": "..."}}"""
KEYWORD_RETRY_SYSTEM = (
    "Answer the question using only the consensus evidence phrases. Return "
    "only the shortest answer span, without JSON or explanation."
)
KEYWORD_RETRY_USER = """Consensus evidence phrases: {hints}

Question: {question}
Answer:"""


def is_attack_chunk(chunk_id: str, attack_name: str) -> bool:
    return str(chunk_id).startswith(attack_name + ":")


def sentence_order_from_candidate(
    candidate: dict[str, Any],
    unit_by_id: dict[str, dict[str, Any]],
) -> list[str]:
    seen = set()
    result = []
    for unit_id in candidate.get("selected_ids", []):
        sentence_id = str(unit_by_id.get(str(unit_id), {}).get("sentence_id", ""))
        if sentence_id and sentence_id not in seen:
            seen.add(sentence_id)
            result.append(sentence_id)
    return result


def reflow_inspection(
    units_row: dict[str, Any],
    frontier_row: dict[str, Any],
    attack_name: str,
) -> dict[str, Any]:
    unit_by_id = {
        str(unit.get("unit_id", "")): unit for unit in units_row.get("units", [])
    }
    sentence_by_id = {
        str(sentence.get("sentence_id", "")): sentence
        for sentence in units_row.get("sentences", [])
    }
    candidates = sorted(
        frontier_row.get("frontier_candidates", []),
        key=lambda row: (
            int(row.get("n_selected", len(row.get("selected_ids", [])))),
            float(row.get("remaining_support_flow", 0.0)),
        ),
    )
    inspected = []
    seen = set()
    detected_chunk = ""
    for candidate in candidates:
        for sentence_id in sentence_order_from_candidate(candidate, unit_by_id):
            if sentence_id in seen or sentence_id not in sentence_by_id:
                continue
            seen.add(sentence_id)
            sentence = sentence_by_id[sentence_id]
            inspected.append(sentence_id)
            chunk_id = str(sentence.get("chunk_id", ""))
            if is_attack_chunk(chunk_id, attack_name):
                detected_chunk = chunk_id
                break
        if detected_chunk:
            break
    return {
        "detected_chunk": detected_chunk,
        "inspected_sentence_ids": inspected,
        "verified_sentences": len(inspected),
        "frontier_candidates_examined": len(candidates),
    }


def random_inspection(
    units_row: dict[str, Any],
    attack_name: str,
    budget: int,
    *,
    identifier: str,
    seed: int,
) -> dict[str, Any]:
    sentences = list(units_row.get("sentences", []))
    rng = random.Random(f"{seed}:{identifier}:{attack_name}")
    rng.shuffle(sentences)
    inspected = sentences[: min(max(0, budget), len(sentences))]
    detected = next(
        (
            str(sentence.get("chunk_id", ""))
            for sentence in inspected
            if is_attack_chunk(str(sentence.get("chunk_id", "")), attack_name)
        ),
        "",
    )
    return {
        "detected_chunk": detected,
        "inspected_sentence_ids": [
            str(sentence.get("sentence_id", "")) for sentence in inspected
        ],
        "verified_sentences": len(inspected),
    }


def defended_answer(
    *,
    question: str,
    contexts: list[dict[str, Any]],
    attacked_answer: str,
    detected_chunk: str,
    reader: ReaderClient,
) -> tuple[str, int]:
    if not detected_chunk:
        return attacked_answer, 0
    retained = [
        dict(context)
        for context in contexts
        if str(context.get("chunk_id", "")) != detected_chunk
    ]
    if not retained:
        return "I don't know", 0
    return reader_answer(reader, question, retained), 1


def extracted_phrases(answer: str, nlp: SpacyAnnotationClient) -> set[str]:
    annotation = nlp.annotate(answer)
    phrases = {answer.strip()}
    pending: list[dict[str, Any]] = []

    def flush() -> None:
        if not pending:
            return
        lemmas = [str(token.get("lemma", "")).strip() for token in pending]
        phrase = " ".join(value for value in lemmas if value).strip()
        if phrase:
            phrases.add(phrase)
        phrases.update(value for value in lemmas if value)
        pending.clear()

    for token in annotation.get("tokens", []):
        if str(token.get("pos", "")) in IGNORE_POS:
            flush()
        else:
            pending.append(token)
    flush()
    return {phrase for phrase in phrases if phrase}


def keyword_aggregation(
    *,
    question: str,
    contexts: list[dict[str, Any]],
    reader: ReaderClient,
    chat: OpenAIClient,
    nlp: SpacyAnnotationClient,
    alpha: float,
    beta: float,
) -> dict[str, Any]:
    isolated = [reader_answer(reader, question, [context]) for context in contexts]
    retained = [answer for answer in isolated if "i don't" not in answer.casefold()]
    if not retained:
        return {
            "answer": "I don't know",
            "isolated_answers": isolated,
            "hints": [],
            "reader_calls": len(contexts),
        }
    counts: dict[str, int] = defaultdict(int)
    for answer in retained:
        for phrase in extracted_phrases(answer, nlp):
            counts[phrase] += 1
    threshold = min(beta, alpha * len(retained))
    kept = [
        phrase
        for phrase, count in counts.items()
        if count >= threshold
        and phrase not in PUNCTUATION
        and phrase.casefold() not in STOPWORDS
    ]
    kept.sort(key=lambda value: (len(value), value), reverse=True)
    if not kept:
        return {
            "answer": "I don't know",
            "isolated_answers": isolated,
            "hints": [],
            "reader_calls": len(contexts),
        }
    joined_hints = ", ".join(kept)
    raw = chat.chat(
        KEYWORD_SYSTEM,
        KEYWORD_USER.format(question=question, hints=joined_hints),
        temperature=0.0,
        max_tokens=128,
    )
    parsed = parse_json_object(raw)
    answer = (
        str(parsed.get("answer", "")).strip()
        if isinstance(parsed, dict)
        else raw.strip()
    )
    answer = canonicalize_reader_answer(
        question,
        answer,
        reader_mode="short_answer",
    )
    format_retry = not answer or "{" in answer or '"answer"' in answer
    if format_retry:
        answer = canonicalize_reader_answer(
            question,
            chat.chat(
                KEYWORD_RETRY_SYSTEM,
                KEYWORD_RETRY_USER.format(
                    question=question,
                    hints=joined_hints,
                ),
                temperature=0.0,
                max_tokens=64,
            ).strip(),
            reader_mode="short_answer",
        )
    return {
        "answer": answer or "I don't know",
        "isolated_answers": isolated,
        "hints": kept,
        "threshold": threshold,
        "format_retry": format_retry,
        "reader_calls": len(contexts) + 1 + int(format_retry),
    }


def evaluate_attack(
    *,
    identifier: str,
    question: str,
    gold: str,
    attack_name: str,
    attack: dict[str, Any],
    units_row: dict[str, Any],
    frontier_row: dict[str, Any],
    reader: ReaderClient,
    chat: OpenAIClient,
    nlp: SpacyAnnotationClient,
    alpha: float,
    beta: float,
) -> dict[str, Any]:
    contexts = [dict(row) for row in attack.get("attacked_contexts", [])]
    attacked_answer = str(attack.get("attacked_answer", ""))
    reflow = reflow_inspection(units_row, frontier_row, attack_name)
    reflow_answer, reflow_calls = defended_answer(
        question=question,
        contexts=contexts,
        attacked_answer=attacked_answer,
        detected_chunk=str(reflow["detected_chunk"]),
        reader=reader,
    )
    random_results = []
    for seed in RANDOM_SEEDS:
        inspection = random_inspection(
            units_row,
            attack_name,
            int(reflow["verified_sentences"]),
            identifier=identifier,
            seed=seed,
        )
        answer, calls = defended_answer(
            question=question,
            contexts=contexts,
            attacked_answer=attacked_answer,
            detected_chunk=str(inspection["detected_chunk"]),
            reader=reader,
        )
        random_results.append({
            "seed": seed,
            **inspection,
            "answer": answer,
            "correct": answers_match(answer, gold),
            "reader_calls": calls,
        })
    keyword = keyword_aggregation(
        question=question,
        contexts=contexts,
        reader=reader,
        chat=chat,
        nlp=nlp,
        alpha=alpha,
        beta=beta,
    )
    malicious_present = any(
        is_attack_chunk(str(context.get("chunk_id", "")), attack_name)
        for context in contexts
    )
    return {
        "malicious_passage_present": malicious_present,
        "no_defense": {
            "answer": attacked_answer,
            "correct": answers_match(attacked_answer, gold),
        },
        "reflow_guard": {
            **reflow,
            "detected": bool(reflow["detected_chunk"]),
            "answer": reflow_answer,
            "correct": answers_match(reflow_answer, gold),
            "reader_calls": reflow_calls,
        },
        "random_guard": {
            "seed_results": random_results,
            "detected": statistics.fmean(
                int(bool(row["detected_chunk"])) for row in random_results
            ),
            "correct": statistics.fmean(
                int(row["correct"]) for row in random_results
            ),
            "verified_sentences": int(reflow["verified_sentences"]),
            "reader_calls": sum(int(row["reader_calls"]) for row in random_results),
        },
        "robustrag_keyword": {
            **keyword,
            "correct": answers_match(str(keyword["answer"]), gold),
            "context_tokens": sum(
                len(word_tokens(str(context.get("text", "")))) for context in contexts
            ),
        },
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    methods = ("no_defense", "reflow_guard", "random_guard", "robustrag_keyword")
    result: dict[str, Any] = {"queries": len(rows), "attacks": {}, "macro": {}}
    for attack_name in ATTACKS:
        result["attacks"][attack_name] = {}
        for method in methods:
            values = [
                float(row["attacks"][attack_name][method]["correct"])
                for row in rows
            ]
            entry = {"attacked_accuracy": statistics.fmean(values)}
            if method in {"reflow_guard", "random_guard"}:
                detection_values = [
                    float(row["attacks"][attack_name][method]["detected"])
                    for row in rows
                    if row["attacks"][attack_name]["malicious_passage_present"]
                ]
                entry["detection_rate"] = (
                    statistics.fmean(detection_values)
                    if detection_values
                    else None
                )
                entry["mean_verified_sentences"] = statistics.fmean(
                    float(row["attacks"][attack_name][method]["verified_sentences"])
                    for row in rows
                )
            result["attacks"][attack_name][method] = entry
    for method in methods:
        result["macro"][method] = {
            "attacked_accuracy": statistics.fmean(
                float(row["attacks"][attack_name][method]["correct"])
                for row in rows
                for attack_name in ATTACKS
            )
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attacks", required=True)
    for attack_name in ATTACKS:
        parser.add_argument(f"--{attack_name}-units", required=True)
        parser.add_argument(f"--{attack_name}-frontier", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--llm-base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--llm-model", default="qwen2.5-7b")
    parser.add_argument("--spacy-base-url", default="http://127.0.0.1:8021")
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=3.0)
    args = parser.parse_args()

    attacks = load_records(args.attacks)
    units = {
        name: {
            str(row.get("id", "")): row
            for row in load_records(getattr(args, f"{name}_units"))
        }
        for name in ATTACKS
    }
    frontiers = {
        name: {
            str(row.get("id", "")): row
            for row in load_records(getattr(args, f"{name}_frontier"))
        }
        for name in ATTACKS
    }
    completed = (
        {str(row.get("id", "")): row for row in load_records(args.out)}
        if os.path.exists(args.out)
        else {}
    )
    pending = [row for row in attacks if str(row.get("id", "")) not in completed]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    reader = ReaderClient(base_url=args.llm_base_url, model=args.llm_model)
    chat = OpenAIClient(args.llm_base_url, args.llm_model)
    nlp = SpacyAnnotationClient(args.spacy_base_url)
    if not nlp.health().get("ok"):
        raise RuntimeError("spaCy service is unhealthy")
    write_lock = threading.Lock()

    def execute(row: dict[str, Any]) -> dict[str, Any]:
        identifier = str(row.get("id", ""))
        question = str(row.get("question", ""))
        gold = str(row.get("gold_answer", ""))
        return {
            "id": identifier,
            "gold_answer": gold,
            "attacks": {
                name: evaluate_attack(
                    identifier=identifier,
                    question=question,
                    gold=gold,
                    attack_name=name,
                    attack=dict(row["methods"][name]),
                    units_row=units[name][identifier],
                    frontier_row=frontiers[name][identifier],
                    reader=reader,
                    chat=chat,
                    nlp=nlp,
                    alpha=args.alpha,
                    beta=args.beta,
                )
                for name in ATTACKS
            },
        }

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(execute, row): row for row in pending}
        failures = []
        for future in as_completed(futures):
            identifier = str(futures[future].get("id", ""))
            try:
                row = future.result()
            except Exception as exc:  # pragma: no cover - integration retries
                failures.append((identifier, repr(exc)))
                print(f"[security-defense] failed {identifier}: {exc!r}", flush=True)
                continue
            with write_lock:
                with open(args.out, "a", encoding="utf-8") as output:
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
            completed[str(row["id"])] = row
            print(f"[security-defense] {len(completed)}/{len(attacks)}", flush=True)
        if failures:
            raise RuntimeError(f"{len(failures)} queries failed: {failures[:3]}")

    ordered = [completed[str(row.get("id", ""))] for row in attacks]
    with open(args.out, "w", encoding="utf-8") as output:
        for row in ordered:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    result = summarize(ordered)
    result["protocol"] = {
        "attacked_context_only": True,
        "maximum_injected_passages": 1,
        "reflow_verifier": "perfect provenance verifier",
        "random_seeds": list(RANDOM_SEEDS),
        "robustrag_keyword_alpha": args.alpha,
        "robustrag_keyword_beta": args.beta,
    }
    with open(args.summary_out, "w", encoding="utf-8") as output:
        json.dump(result, output, indent=2, ensure_ascii=False)
        output.write("\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
