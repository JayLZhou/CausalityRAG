#!/usr/bin/env python3
"""Find visually clean two-token joint-flip cases in frozen Table 3 results."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import re
import string
from pathlib import Path


ARTICLES = {"a", "an", "the"}


def read_jsonl(path: Path):
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize(text: object) -> list[str]:
    value = str(text or "").lower()
    value = "".join(character for character in value if character not in string.punctuation)
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return value.split()


def perfect_f1(answer: object, gold: object) -> bool:
    left, right = normalize(answer), normalize(gold)
    if not left or not right:
        return left == right
    overlap = sum((Counter(left) & Counter(right)).values())
    if not overlap:
        return False
    precision = overlap / len(left)
    recall = overlap / len(right)
    return abs(2 * precision * recall / (precision + recall) - 1.0) < 1e-12


def normalized_text(value: object) -> str:
    return " ".join(normalize(value))


def benchmark_accuracy(answer: object, record: dict, dataset: str) -> bool:
    rendered = normalized_text(answer)
    if not rendered:
        return False
    canonical = str(record.get("answer", record.get("gold_answer", ""))).strip()
    listed = record.get("answers", record.get("gold_answers", []))
    alternatives = list(listed) if isinstance(listed, (list, tuple)) else [listed]
    if dataset == "timeqa":
        components = [normalized_text(item) for item in alternatives if str(item).strip()]
        if not components and canonical:
            components = [normalized_text(item) for item in canonical.split(";") if item.strip()]
        return bool(components) and all(
            component == rendered or component in rendered or rendered in component
            for component in components
        )
    aliases = record.get("answer_aliases", [])
    aliases = list(aliases) if isinstance(aliases, (list, tuple)) else [aliases]
    candidates = [canonical]
    if dataset in {"popqa", "triviaqa"} or record.get("gold_answers") is not None:
        candidates.extend(alternatives)
        candidates.extend(aliases)
    for candidate in candidates:
        expected = normalized_text(candidate)
        if not expected:
            continue
        if dataset == "quartz":
            if rendered == expected:
                return True
        elif rendered == expected or rendered in expected or expected in rendered:
            return True
    return False


def final_edits(row: dict) -> list[dict]:
    wanted = set(row.get("selected_ids") or [])
    for attempt in reversed(row.get("attempts") or []):
        edits = attempt.get("edits") or []
        ids = {edit.get("unit_id") for edit in edits if edit.get("ok")}
        if wanted and wanted.issubset(ids):
            return [edit for edit in edits if edit.get("unit_id") in wanted]
    return []


def sentence_for(text: str, start: int) -> str:
    spans = list(re.finditer(r"[^.!?]+[.!?]?", text))
    for match in spans:
        if match.start() <= start < match.end():
            return match.group(0).strip()
    return text.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-baseline-flips", action="store_true")
    parser.add_argument("--require-saved-singletons", action="store_true")
    parser.add_argument("--answer-change-only", action="store_true")
    parser.add_argument("--acc-loss", action="store_true")
    parser.add_argument("--require-different-evidence", action="store_true")
    args = parser.parse_args()

    datasets = ["hotpotqa", "timeqa", "finqa", "musique", "quartz", "triviaqa", "2wiki", "popqa"]
    output = []
    for dataset in datasets:
        base = args.root / dataset
        reflow_path = base / "methods/reflow/results_top5_1000.jsonl"
        baseline_path = base / "audits/final_top10pool_k5/baselines_1000.jsonl"
        query_path = base / "retrieval/queries_1000.jsonl"
        retrieval_path = base / "retrieval/top10_1000.jsonl"
        if not query_path.exists():
            query_path = retrieval_path
        if not all(path.exists() for path in [reflow_path, baseline_path, query_path, retrieval_path]):
            continue
        reflow = {str(r["id"]): r for r in read_jsonl(reflow_path)}
        baselines = {str(r["id"]): r for r in read_jsonl(baseline_path)}
        queries = {str(r["id"]): r for r in read_jsonl(query_path)}
        retrieval = {str(r["id"]): r for r in read_jsonl(retrieval_path)}
        for query_id, row in reflow.items():
            if not row.get("verified_flip") or row.get("n_modified_tokens") != 2:
                continue
            gold = row.get("gold_answer")
            comparison_answer = gold
            query_record = queries.get(query_id, {})
            if args.acc_loss:
                if not benchmark_accuracy(row.get("clean_answer"), query_record, dataset):
                    continue
                if benchmark_accuracy(row.get("edited_answer"), query_record, dataset):
                    continue
            elif args.answer_change_only:
                comparison_answer = row.get("clean_answer")
                if perfect_f1(row.get("edited_answer"), comparison_answer):
                    continue
            else:
                if not perfect_f1(row.get("clean_answer"), gold):
                    continue
                if perfect_f1(row.get("edited_answer"), gold):
                    continue
            edits = final_edits(row)
            if len(edits) != 2:
                continue
            old = [str(edit.get("old") or edit.get("token") or "") for edit in edits]
            new = [str(edit.get("new") or "") for edit in edits]
            if old[0].casefold() == old[1].casefold():
                continue
            if re.sub(r"s$", "", old[0].casefold()) == re.sub(r"s$", "", old[1].casefold()):
                continue
            baseline = baselines.get(query_id, {}).get("methods", {})
            required = ["arc_jsd", "mirage"]
            if any(name not in baseline for name in required):
                continue
            if not args.allow_baseline_flips:
                if args.acc_loss:
                    if any(
                        not benchmark_accuracy(
                            baseline[name].get("edited_answer"),
                            query_record,
                            dataset,
                        )
                        for name in required
                    ):
                        continue
                elif any(
                    not perfect_f1(
                        baseline[name].get("edited_answer"), comparison_answer
                    )
                    for name in required
                ):
                    continue
            retrieved = retrieval.get(query_id, {}).get("retrieved", [])[:5]
            chunks = {str(chunk.get("chunk_id")): chunk for chunk in retrieved}
            evidence = []
            for edit in edits:
                chunk = chunks.get(str(edit.get("chunk_id")), {})
                text = str(chunk.get("text") or "")
                evidence.append(
                    {
                        "chunk_id": edit.get("chunk_id"),
                        "title": chunk.get("title"),
                        "sentence": sentence_for(text, int(edit.get("chunk_char_start") or 0)),
                    }
                )
            if (
                args.require_different_evidence
                and evidence[0]["sentence"] == evidence[1]["sentence"]
            ):
                continue
            singleton_answers = {}
            for edit in edits:
                unit_id = str(edit.get("unit_id"))
                singleton_attempts = [
                    attempt
                    for attempt in row.get("attempts") or []
                    if set(map(str, attempt.get("selected_ids") or [])) == {unit_id}
                    and attempt.get("reader_called")
                ]
                singleton_answers[unit_id] = (
                    singleton_attempts[-1].get("edited_answer")
                    if singleton_attempts
                    else None
                )
            if args.require_saved_singletons and any(
                answer is None or not perfect_f1(answer, comparison_answer)
                for answer in singleton_answers.values()
            ):
                continue
            output.append(
                {
                    "dataset": dataset,
                    "id": query_id,
                    "question": queries.get(query_id, {}).get("question"),
                    "gold": gold,
                    "clean": row.get("clean_answer"),
                    "joint": row.get("edited_answer"),
                    "edits": [
                        {
                            "unit_id": edit.get("unit_id"),
                            "old": old[index],
                            "new": new[index],
                            **evidence[index],
                        }
                        for index, edit in enumerate(edits)
                    ],
                    "baselines": {
                        name: {
                            "tokens": baseline[name].get("selected_tokens"),
                            "answer": baseline[name].get("edited_answer"),
                        }
                        for name in required
                    },
                    "singletons": singleton_answers,
                }
            )
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print(json.dumps({"count": len(output), "output": str(args.output)}))


if __name__ == "__main__":
    main()
