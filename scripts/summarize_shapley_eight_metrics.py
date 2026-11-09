#!/usr/bin/env python3
"""Summarize the eight Table 3 metrics for one Shapley evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causalityrag.evaluation_metrics import (
    answer_changed,
    clean_correct,
    correctness_lost,
    gold_answer_spec,
    valid_answer,
)


METRICS = ("answer", "f1", "em", "acc")


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def by_id(path: Path) -> dict[str, dict]:
    rows = load(path)
    table = {str(row["id"]): row for row in rows}
    if len(rows) != 1000 or len(table) != 1000:
        raise ValueError(f"{path}: expected 1000 unique queries")
    return table


def outcome(method: dict) -> tuple[bool, bool, str]:
    status = str(method.get("status", ""))
    called = bool(method.get("reader_called", False))
    answer = str(method.get("edited_answer", ""))
    invalid = status.startswith("protocol_violation") or (
        called and not valid_answer(answer)
    )
    return called and not invalid, invalid, answer


def flip(
    metric: str,
    *,
    clean: str,
    method: dict,
    spec,
    reader_mode: str,
) -> bool:
    executed, _, answer = outcome(method)
    if not executed:
        return False
    if metric == "answer":
        return answer_changed(clean, answer)
    return correctness_lost(
        metric,
        clean,
        answer,
        spec,
        reader_mode=reader_mode,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--factual", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        required=True,
        choices=(
            "hotpotqa",
            "timeqa",
            "finqa",
            "musique",
            "quartz",
            "triviaqa",
            "2wiki",
            "popqa",
        ),
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    retrieval = by_id(args.retrieval)
    factual = by_id(args.factual)
    control = by_id(args.control)
    if not (set(retrieval) == set(factual) == set(control)):
        raise ValueError("retrieval, factual, and control IDs do not align")

    counters = {
        name: {metric: 0 for metric in METRICS}
        for name in (
            "fr_denominator",
            "fr_flips",
            "cfr_denominator",
            "cfr_factual_flips",
            "control_flips",
            "delta",
        )
    }
    diagnostics = {
        "factual_invalid": 0,
        "control_invalid": 0,
        "factual_unexecuted": 0,
        "control_unexecuted": 0,
    }
    for identifier, record in retrieval.items():
        factual_row = factual[identifier]
        control_row = control[identifier]
        clean = str(factual_row.get("clean_answer", ""))
        if not valid_answer(clean):
            continue
        spec = gold_answer_spec(record, args.dataset)
        reader_mode = "quartz" if args.dataset == "quartz" else "short_answer"
        factual_method = factual_row.get("methods", {}).get(
            "token_leverage_shap", {}
        )
        control_method = control_row.get("methods", {}).get(
            "token_leverage_shap", {}
        )
        factual_executed, factual_invalid, _ = outcome(factual_method)
        control_executed, control_invalid, _ = outcome(control_method)
        diagnostics["factual_invalid"] += int(factual_invalid)
        diagnostics["control_invalid"] += int(control_invalid)
        diagnostics["factual_unexecuted"] += int(
            not factual_executed and not factual_invalid
        )
        diagnostics["control_unexecuted"] += int(
            not control_executed and not control_invalid
        )
        populations = {
            "answer": True,
            **{
                metric: clean_correct(
                    metric,
                    clean,
                    spec,
                    reader_mode=reader_mode,
                )
                for metric in ("f1", "em", "acc")
            },
        }
        for metric, eligible in populations.items():
            if not eligible:
                continue
            factual_flip = flip(
                metric,
                clean=clean,
                method=factual_method,
                spec=spec,
                reader_mode=reader_mode,
            )
            control_flip = flip(
                metric,
                clean=clean,
                method=control_method,
                spec=spec,
                reader_mode=reader_mode,
            )
            if not factual_invalid:
                counters["fr_denominator"][metric] += 1
                counters["fr_flips"][metric] += int(factual_flip)
            if factual_invalid or control_invalid:
                continue
            counters["cfr_denominator"][metric] += 1
            counters["cfr_factual_flips"][metric] += int(factual_flip)
            counters["control_flips"][metric] += int(control_flip)
            counters["delta"][metric] += int(factual_flip) - int(control_flip)

    names = {
        "answer": ("Ans-FR", "Ans-CFR"),
        "f1": ("F1-FR", "F1-CFR"),
        "em": ("EM-FR", "EM-CFR"),
        "acc": ("Acc-FR", "Acc-CFR"),
    }
    summary = {
        "dataset": args.dataset,
        "queries": 1000,
        "diagnostics": diagnostics,
        "metrics": {},
    }
    for metric, (fr_name, cfr_name) in names.items():
        fr_denominator = counters["fr_denominator"][metric]
        cfr_denominator = counters["cfr_denominator"][metric]
        summary["metrics"][fr_name] = {
            "flips": counters["fr_flips"][metric],
            "denominator": fr_denominator,
            "rate_percent": 100 * counters["fr_flips"][metric] / fr_denominator,
        }
        summary["metrics"][cfr_name] = {
            "factual_flips": counters["cfr_factual_flips"][metric],
            "control_flips": counters["control_flips"][metric],
            "denominator": cfr_denominator,
            "rate_percent": 100 * counters["delta"][metric] / cfr_denominator,
        }
    args.out.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
