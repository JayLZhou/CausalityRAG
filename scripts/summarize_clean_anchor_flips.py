"""Summarize HotpotQA counterfactual flips against clean reader answers.

Each answer is evaluated against the same dataset gold answer with the
HippoRAG/MRQA normalizer.  The clean reader answer is the per-query reference
level: a flip is a change in the corresponding gold-scored RAG metric, rather
than a raw string difference.  We report:

* Answer-Flip: normalized EM(edited, clean) = 0;
* F1-Flip: normalized token F1(edited, gold) != F1(clean, gold);
* EM-Flip: normalized EM(edited, gold) != EM(clean, gold);
* Acc-Flip: normalized-containment answer accuracy changes from clean to edited.

This script only reads saved JSONL outputs; it never calls the reader or the
replacement model.  It accepts one flat ReFlow output and one nested matched-
budget baseline output produced by ``evaluate_matched_budget_baselines.py``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Iterable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.reader import answer_token_f1, answers_exact_match, answers_match


def _rows(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _is_executed(method: dict[str, Any]) -> bool:
    return bool(method.get("reader_called", False))


def _metric_row(*, row: dict[str, Any], edited_answer: str) -> dict[str, Any] | None:
    clean_answer = str(row.get("clean_answer", ""))
    gold_answer = str(row.get("gold_answer", ""))
    if not clean_answer or not gold_answer or not edited_answer.strip():
        return None
    return {
        "id": str(row.get("id", "")),
        "clean_f1": answer_token_f1(clean_answer, gold_answer),
        "edited_f1": answer_token_f1(edited_answer, gold_answer),
        "clean_em": answers_exact_match(clean_answer, gold_answer),
        "edited_em": answers_exact_match(edited_answer, gold_answer),
        "clean_acc": answers_match(clean_answer, gold_answer),
        "edited_acc": answers_match(edited_answer, gold_answer),
    }


def _summarize(method_rows: Iterable[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
    total_queries = 0
    evaluated = 0
    valid_answers = 0
    scored = 0
    answer_flips = 0
    f1_flips = 0
    em_flips = 0
    acc_flips = 0
    mean_f1_delta = 0.0
    examples = []
    for parent, method in method_rows:
        total_queries += 1
        if not _is_executed(method):
            continue
        evaluated += 1
        clean_answer = str(parent.get("clean_answer", ""))
        edited_answer = str(method.get("edited_answer", method.get("answer", "")))
        if clean_answer and edited_answer.strip():
            answer_flips += int(not answers_exact_match(edited_answer, clean_answer))
            valid_answers += 1
        metrics = _metric_row(
            row=parent,
            edited_answer=edited_answer,
        )
        if metrics is None:
            continue
        scored += 1
        clean_f1 = float(metrics["clean_f1"])
        edited_f1 = float(metrics["edited_f1"])
        f1_flip = edited_f1 != clean_f1
        em_flip = bool(metrics["clean_em"]) != bool(metrics["edited_em"])
        acc_flip = bool(metrics["clean_acc"]) != bool(metrics["edited_acc"])
        f1_flips += int(f1_flip)
        em_flips += int(em_flip)
        acc_flips += int(acc_flip)
        mean_f1_delta += edited_f1 - clean_f1
        if len(examples) < 5 and (f1_flip or em_flip or acc_flip):
            examples.append({
                "id": metrics["id"],
                "clean_f1": clean_f1,
                "edited_f1": edited_f1,
                "clean_em": bool(metrics["clean_em"]),
                "edited_em": bool(metrics["edited_em"]),
                "clean_acc": bool(metrics["clean_acc"]),
                "edited_acc": bool(metrics["edited_acc"]),
            })
    denominator = max(1, total_queries)
    return {
        "total_queries": total_queries,
        "reader_executed_queries": evaluated,
        "valid_answer_queries": valid_answers,
        "gold_scored_queries": scored,
        "answer_flip_count": answer_flips,
        "answer_flip_ratio": answer_flips / denominator,
        "f1_flip_count": f1_flips,
        "f1_flip_ratio": f1_flips / denominator,
        "em_flip_count": em_flips,
        "em_flip_ratio": em_flips / denominator,
        "acc_flip_count": acc_flips,
        "acc_flip_ratio": acc_flips / denominator,
        "mean_token_f1_delta_edited_minus_clean": mean_f1_delta / max(1, scored),
        "example_metric_rows": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reflow", required=True)
    parser.add_argument("--baselines", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    reflow_rows = _rows(args.reflow)
    baseline_rows = _rows(args.baselines)
    baseline_by_id = {str(row.get("id", "")): row for row in baseline_rows}
    if len(baseline_by_id) != len(baseline_rows):
        raise ValueError("baseline rows must have unique ids")

    reflow_method_rows = []
    for row in reflow_rows:
        reflow_method_rows.append((
            row,
            {
                "reader_called": int(row.get("reader_calls", 0)) > 0,
                "edited_answer": row.get("edited_answer", ""),
            },
        ))

    method_names = sorted({
        method
        for row in baseline_rows
        for method in row.get("methods", {})
    })
    output: dict[str, Any] = {
        "metric_contract": {
            "population": "all 1,000 HotpotQA rows; no reader execution, no legal edit, or an empty answer contributes zero flips",
            "answer_flip": "normalized EM(edited, clean) = 0",
            "f1_flip": "HippoRAG-style normalized token F1(edited, gold) != token F1(clean, gold)",
            "em_flip": "normalized EM(edited, gold) != normalized EM(clean, gold)",
            "acc_flip": "normalized containment accuracy(edited, gold) != accuracy(clean, gold)",
        },
        "sources": {
            "reflow": os.path.abspath(args.reflow),
            "baselines": os.path.abspath(args.baselines),
        },
        "methods": {
            "reflow": _summarize(reflow_method_rows),
        },
    }
    for method_name in method_names:
        output["methods"][method_name] = _summarize(
            (row, row.get("methods", {}).get(method_name, {}))
            for row in baseline_rows
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
