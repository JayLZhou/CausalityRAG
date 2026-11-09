"""Rebuild frozen security attacks with top-10 retrieval and one injection.

The expensive candidate generation from the original attack run is reused,
but every candidate is rescored with the frozen reader.  Only the strongest
candidate is injected, the complete top-10 retrieval is reranked, and the
attacked answer is generated again.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.io import load_records, record_id, retrieved_contexts
from causalityrag.reader import ReaderClient, answers_exact_match, answers_match
from scripts.run_security_pareto_attacks import (
    EmbeddingClient,
    candidate_attack_score,
    cosine,
    rank_injected_passages,
    reader_answer,
    token_edit_distance,
    word_tokens,
)


ATTACKS = ("poisonedrag", "tparag")


def rebuild_method(
    *,
    attack_name: str,
    source: dict[str, Any],
    record: dict[str, Any],
    question: str,
    gold: str,
    clean: str,
    target: str,
    reader: ReaderClient,
    embedder: EmbeddingClient,
    retrieved_k: int,
) -> dict[str, Any]:
    candidates = [dict(row) for row in source.get("injected_passages", [])]
    if not candidates:
        raise ValueError(f"{record_id(record)}:{attack_name}: no saved candidates")
    for index, candidate in enumerate(candidates):
        candidate["source"] = attack_name
        candidate.setdefault(
            "chunk_id",
            f"{attack_name}:{record_id(record)}:{index}",
        )

    vectors = embedder.embed(
        [question] + [str(row.get("text", "")) for row in candidates]
    )
    scored = []
    for candidate, vector in zip(candidates, vectors[1:]):
        answer = reader_answer(reader, question, [candidate])
        similarity = cosine(vectors[0], vector)
        scored.append({
            "row": candidate,
            "standalone_answer": answer,
            "similarity": similarity,
            "score": candidate_attack_score(
                answer,
                target=target,
                gold=gold,
                similarity=similarity,
            ),
        })
    scored.sort(key=lambda item: item["score"], reverse=True)
    injected = dict(scored[0]["row"])
    original = [dict(row) for row in retrieved_contexts(record)[:retrieved_k]]
    attacked_contexts, retrieval_ranking = rank_injected_passages(
        question,
        original,
        [injected],
        embedder,
        k=retrieved_k,
    )
    attacked_answer = reader_answer(reader, question, attacked_contexts)

    result = {
        "target_answer": target,
        "attacked_answer": attacked_answer,
        "attacked_contexts": attacked_contexts,
        "injected_passages": [injected],
        "retrieved_injected_passages": sum(
            str(row.get("source", "")) == attack_name
            for row in attacked_contexts
        ),
        "attack_token_footprint": len(word_tokens(str(injected.get("text", "")))),
        "attacker_reader_calls": len(candidates),
        "final_reader_calls": 1,
        "retrieval_ranking": retrieval_ranking,
        "candidate_selection": [
            {
                "chunk_id": str(item["row"].get("chunk_id", "")),
                "standalone_answer": item["standalone_answer"],
                "similarity": item["similarity"],
                "score": list(item["score"]),
                "selected": index == 0,
            }
            for index, item in enumerate(scored)
        ],
        "answer_flip": not answers_exact_match(attacked_answer, clean),
        "untargeted_success": not answers_match(attacked_answer, gold),
        "targeted_success": answers_match(attacked_answer, target),
    }
    if attack_name == "tparag":
        parent_id = str(injected.get("chunk_id", "")).removeprefix("tparag:")
        parent = next(
            (
                row
                for row in original
                if str(row.get("chunk_id", "")) == parent_id
            ),
            None,
        )
        result["changed_tokens_inside_passages"] = (
            token_edit_distance(
                str(parent.get("text", "")),
                str(injected.get("text", "")),
            )
            if parent is not None
            else None
        )
    return result


def rebuild_query(
    source_row: dict[str, Any],
    record: dict[str, Any],
    reader: ReaderClient,
    embedder: EmbeddingClient,
    *,
    retrieved_k: int,
) -> dict[str, Any]:
    identifier = record_id(record)
    question = str(source_row.get("question", record.get("question", "")))
    gold = str(source_row.get("gold_answer", record.get("answer", "")))
    clean = str(source_row.get("clean_answer", ""))
    target = str(source_row.get("target_answer", ""))
    methods = source_row.get("methods", {})
    return {
        "id": identifier,
        "question": question,
        "gold_answer": gold,
        "clean_answer": clean,
        "target_answer": target,
        "methods": {
            attack_name: rebuild_method(
                attack_name=attack_name,
                source=dict(methods[attack_name]),
                record=record,
                question=question,
                gold=gold,
                clean=clean,
                target=target,
                reader=reader,
                embedder=embedder,
                retrieved_k=retrieved_k,
            )
            for attack_name in ATTACKS
        },
    }


def summarize(rows: list[dict[str, Any]], *, retrieved_k: int) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "queries": len(rows),
        "protocol": {
            "retrieved_chunks": retrieved_k,
            "maximum_injected_passages": 1,
            "candidate_source": "frozen original attack run",
        },
        "methods": {},
    }
    for attack_name in ATTACKS:
        values = [row["methods"][attack_name] for row in rows]
        summary["methods"][attack_name] = {
            "retrieved_injection_rate": statistics.fmean(
                int(row["retrieved_injected_passages"] > 0) for row in values
            ),
            "untargeted_asr": statistics.fmean(
                int(row["untargeted_success"]) for row in values
            ),
            "targeted_asr": statistics.fmean(
                int(row["targeted_success"]) for row in values
            ),
            "mean_attack_token_footprint": statistics.fmean(
                int(row["attack_token_footprint"]) for row in values
            ),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval", required=True)
    parser.add_argument("--source-attacks", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--retrieved-k", type=int, default=10)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--llm-base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--llm-model", default="qwen2.5-7b")
    parser.add_argument("--embed-base-url", default="http://127.0.0.1:8017/v1")
    parser.add_argument("--embed-model", default="Qwen3-Embedding-0.6B")
    args = parser.parse_args()

    retrieval_by_id = {
        record_id(row): row for row in load_records(args.retrieval)
    }
    source_rows = load_records(args.source_attacks)[: args.n]
    aligned = []
    for row in source_rows:
        identifier = str(row.get("id", ""))
        if identifier not in retrieval_by_id:
            raise ValueError(f"missing retrieval row for {identifier}")
        aligned.append((row, retrieval_by_id[identifier]))
    if len(aligned) != args.n:
        raise ValueError(f"expected {args.n} source rows, got {len(aligned)}")

    completed = (
        {str(row.get("id", "")): row for row in load_records(args.out)}
        if os.path.exists(args.out)
        else {}
    )
    pending = [pair for pair in aligned if str(pair[0].get("id", "")) not in completed]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    reader = ReaderClient(base_url=args.llm_base_url, model=args.llm_model)
    embedder = EmbeddingClient(args.embed_base_url, args.embed_model)
    write_lock = threading.Lock()

    def execute(pair: tuple[dict[str, Any], dict[str, Any]]) -> dict[str, Any]:
        return rebuild_query(
            pair[0],
            pair[1],
            reader,
            embedder,
            retrieved_k=args.retrieved_k,
        )

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(execute, pair): pair for pair in pending}
        failures = []
        for future in as_completed(futures):
            identifier = str(futures[future][0].get("id", ""))
            try:
                row = future.result()
            except Exception as exc:  # pragma: no cover - integration retries
                failures.append((identifier, repr(exc)))
                print(f"[security-k10-k1] failed {identifier}: {exc!r}", flush=True)
                continue
            with write_lock:
                with open(args.out, "a", encoding="utf-8") as output:
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
            completed[str(row["id"])] = row
            print(
                f"[security-k10-k1] {len(completed)}/{len(aligned)}",
                flush=True,
            )
        if failures:
            raise RuntimeError(f"{len(failures)} queries failed: {failures[:3]}")

    ordered = [completed[str(row.get("id", ""))] for row, _ in aligned]
    with open(args.out, "w", encoding="utf-8") as output:
        for row in ordered:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    result = summarize(ordered, retrieved_k=args.retrieved_k)
    with open(args.summary_out, "w", encoding="utf-8") as output:
        json.dump(result, output, indent=2, ensure_ascii=False)
        output.write("\n")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
