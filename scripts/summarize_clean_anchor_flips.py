"""Summarize HotpotQA counterfactual flips against clean reader answers.

Each answer is evaluated against every accepted dataset gold alias with the
HippoRAG/MRQA normalizer.  Answer-Flip excludes invalid clean answers from its
denominator.  The gold-based metrics are correctness-conditioned: each metric
uses the queries that the clean reader answered correctly under that same
metric, and only a decrease counts as a flip.  We report:

* Answer-Flip: normalized EM(edited, clean) = 0;
* F1-CFlip: normalized token F1(edited, gold) < F1(clean, gold);
* EM-CFlip: clean exact match is lost after editing;
* Acc-CFlip: clean benchmark correctness is lost after editing.

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

from causalityrag.evaluation_metrics import (
    answer_changed,
    gold_accuracy,
    gold_answer_spec,
    gold_exact,
    gold_f1,
    valid_answer,
)
from scripts.evaluate_reflow import valid_clean_answer


def _rows(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _is_executed(method: dict[str, Any]) -> bool:
    return bool(method.get("reader_called", False)) and not str(
        method.get("status", "")
    ).startswith("protocol_violation")


def _invalid_method(method: dict[str, Any]) -> bool:
    status = str(method.get("status", ""))
    called = bool(method.get("reader_called", False))
    return status.startswith("protocol_violation") or (
        called
        and not valid_answer(
            method.get("edited_answer", method.get("answer", ""))
        )
    )


def _valid_clean_row(row: dict[str, Any]) -> bool:
    status = str(row.get("evaluation_status", ""))
    return (
        valid_clean_answer(str(row.get("clean_answer", "")))
        and not status.startswith("protocol_violation_invalid_clean")
        and status != "invalid_clean_answer"
    )


def _metric_row(
    *,
    row: dict[str, Any],
    edited_answer: str,
    dataset: str,
    reader_mode: str,
) -> dict[str, Any] | None:
    clean_answer = str(row.get("clean_answer", ""))
    spec = gold_answer_spec(row, dataset)
    if not _valid_clean_row(row) or not spec.is_valid or not edited_answer.strip():
        return None
    return {
        "id": str(row.get("id", "")),
        "clean_f1": gold_f1(clean_answer, spec),
        "edited_f1": gold_f1(edited_answer, spec),
        "clean_em": gold_exact(clean_answer, spec),
        "edited_em": gold_exact(edited_answer, spec),
        "clean_acc": gold_accuracy(clean_answer, spec, reader_mode=reader_mode),
        "edited_acc": gold_accuracy(edited_answer, spec, reader_mode=reader_mode),
    }


def _summarize(
    method_rows: Iterable[tuple[dict[str, Any], dict[str, Any]]],
    *,
    dataset: str = "",
    reader_mode: str = "short_answer",
) -> dict[str, Any]:
    total_queries = 0
    valid_clean_queries = 0
    answer_denominator_queries = 0
    evaluated = 0
    valid_answers = 0
    scored = 0
    f1_clean_correct_queries = 0
    em_clean_correct_queries = 0
    acc_clean_correct_queries = 0
    answer_flips = 0
    f1_flips = 0
    em_flips = 0
    acc_flips = 0
    mean_f1_delta = 0.0
    examples = []
    for parent, method in method_rows:
        total_queries += 1
        clean_answer = str(parent.get("clean_answer", ""))
        spec = gold_answer_spec(parent, dataset)
        valid_clean = _valid_clean_row(parent)
        valid_clean_queries += int(valid_clean)
        valid_pair = valid_clean and not _invalid_method(method)
        answer_denominator_queries += int(valid_pair)
        clean_f1_score = (
            gold_f1(clean_answer, spec)
            if valid_pair and spec.is_valid
            else 0.0
        )
        clean_em_score = bool(
            valid_pair
            and gold_exact(clean_answer, spec)
        )
        clean_acc_score = bool(
            valid_pair
            and gold_accuracy(clean_answer, spec, reader_mode=reader_mode)
        )
        f1_clean_correct = clean_f1_score >= 1.0 - 1e-12
        f1_clean_correct_queries += int(f1_clean_correct)
        em_clean_correct_queries += int(clean_em_score)
        acc_clean_correct_queries += int(clean_acc_score)
        if not _is_executed(method):
            continue
        evaluated += 1
        edited_answer = str(method.get("edited_answer", method.get("answer", "")))
        if valid_pair and edited_answer.strip():
            answer_flips += int(answer_changed(clean_answer, edited_answer))
            valid_answers += 1
        metrics = _metric_row(
            row=parent,
            edited_answer=edited_answer,
            dataset=dataset,
            reader_mode=reader_mode,
        )
        if metrics is None:
            continue
        scored += 1
        clean_f1 = float(metrics["clean_f1"])
        edited_f1 = float(metrics["edited_f1"])
        f1_flip = f1_clean_correct and edited_f1 < clean_f1
        em_flip = bool(metrics["clean_em"]) and not bool(metrics["edited_em"])
        acc_flip = bool(metrics["clean_acc"]) and not bool(metrics["edited_acc"])
        f1_flips += int(f1_flip)
        em_flips += int(em_flip)
        acc_flips += int(acc_flip)
        if f1_clean_correct:
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
    return {
        "total_queries": total_queries,
        "answer_denominator_queries": answer_denominator_queries,
        "reader_executed_queries": evaluated,
        "valid_answer_queries": valid_answers,
        "gold_scored_queries": scored,
        "f1_clean_correct_queries": f1_clean_correct_queries,
        "em_clean_correct_queries": em_clean_correct_queries,
        "acc_clean_correct_queries": acc_clean_correct_queries,
        "answer_flip_count": answer_flips,
        "answer_flip_ratio": answer_flips / max(1, answer_denominator_queries),
        "f1_flip_count": f1_flips,
        "f1_flip_ratio": f1_flips / max(1, f1_clean_correct_queries),
        "em_flip_count": em_flips,
        "em_flip_ratio": em_flips / max(1, em_clean_correct_queries),
        "acc_flip_count": acc_flips,
        "acc_flip_ratio": acc_flips / max(1, acc_clean_correct_queries),
        "mean_token_f1_delta_edited_minus_clean": (
            mean_f1_delta / max(1, f1_clean_correct_queries)
        ),
        "example_metric_rows": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reflow", required=True)
    parser.add_argument("--baselines", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dataset", default="")
    parser.add_argument(
        "--reader-mode",
        default=os.environ.get("CAUSALITYRAG_READER_MODE", "short_answer"),
    )
    args = parser.parse_args()
    reader_mode = args.reader_mode.strip().lower()

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
                "status": row.get("evaluation_status", ""),
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
            "answer_population": "queries with a valid clean answer and non-invalid factual output; invalid clean/edited answers are excluded, while valid-clean queries with no executed legal edit contribute zero flips",
            "correctness_population": "metric-specific clean-correct queries: clean F1=1 for F1, clean exact match for EM, and clean benchmark accuracy for Acc; invalid edits are excluded and legal unexecuted edits contribute zero",
            "answer_flip": "normalized EM(edited, clean) = 0",
            "f1_flip": "on the clean-correct population, normalized token F1(edited, gold) < token F1(clean, gold)",
            "em_flip": "on the clean-correct population, clean exact match is lost after editing",
            "acc_flip": "on the clean-correct population, normalized containment accuracy is lost after editing",
        },
        "sources": {
            "reflow": os.path.abspath(args.reflow),
            "baselines": os.path.abspath(args.baselines),
        },
        "methods": {
            "reflow": _summarize(
                reflow_method_rows,
                dataset=args.dataset,
                reader_mode=reader_mode,
            ),
        },
    }
    for method_name in method_names:
        output["methods"][method_name] = _summarize(
            (
                (row, row.get("methods", {}).get(method_name, {}))
                for row in baseline_rows
            ),
            dataset=args.dataset,
            reader_mode=reader_mode,
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
