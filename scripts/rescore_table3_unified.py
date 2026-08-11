#!/usr/bin/env python3
"""Rescore all Table 3 methods with one paired benchmark-aware contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from causalityrag.evaluation_metrics import (
    answer_changed,
    clean_correct,
    correctness_lost,
    gold_answer_spec,
    valid_answer,
)
from scripts.audit_table3_token_counts import reflow_policy_count


DATASETS = (
    "hotpotqa",
    "timeqa",
    "finqa",
    "musique",
    "quartz",
    "triviaqa",
    "2wiki",
    "popqa",
)
CORRECTNESS_METRICS = ("f1", "em", "acc")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def row_id(row: dict[str, Any]) -> str:
    return str(row.get("id", row.get("_id", row.get("qid", ""))))


def canonical_method(name: str) -> str:
    return "attention" if name == "attention_prompt_last" else name


def reader_mode(dataset: str) -> str:
    return "quartz" if dataset == "quartz" else "short_answer"


def valid_clean_row(row: dict[str, Any]) -> bool:
    status = str(row.get("evaluation_status", ""))
    return (
        valid_answer(row.get("clean_answer", ""))
        and status != "invalid_clean_answer"
        and not status.startswith("protocol_violation_invalid_clean")
    )


@dataclass(frozen=True)
class Outcome:
    executed: bool
    invalid: bool
    answer: str
    status: str


def method_outcome(method: dict[str, Any], *, reflow: bool = False) -> Outcome:
    status = str(
        method.get("evaluation_status" if reflow else "status", "")
    )
    if reflow:
        called = int(method.get("reader_calls", 0)) > 0
    else:
        called = bool(method.get("reader_called", False))
    answer = str(method.get("edited_answer", method.get("answer", "")))
    invalid = status.startswith("protocol_violation") or (
        called and not valid_answer(answer)
    )
    return Outcome(
        executed=called and not invalid,
        invalid=invalid,
        answer=answer,
        status=status,
    )


def event(
    metric: str,
    *,
    clean_answer: str,
    outcome: Outcome,
    spec,
    mode: str,
) -> bool:
    if not outcome.executed:
        return False
    if metric == "answer":
        return answer_changed(clean_answer, outcome.answer)
    return correctness_lost(
        metric,
        clean_answer,
        outcome.answer,
        spec,
        reader_mode=mode,
    )


def empty_counter() -> dict[str, dict[str, int]]:
    return {
        scope: {metric: 0 for metric in ("answer", *CORRECTNESS_METRICS)}
        for scope in (
            "fr_denominator",
            "fr_flips",
            "cfr_denominator",
            "cfr_factual_flips",
            "cfr_control_flips",
            "cfr_delta_sum",
        )
    }


def artifact_paths(root: Path, dataset: str) -> dict[str, Path]:
    base = root / dataset
    return {
        "retrieval": base / "retrieval/top10_1000.jsonl",
        "reflow": base / "methods/reflow/results_top5_1000.jsonl",
        "baselines": base / "audits/final_top10pool_k5/baselines_1000.jsonl",
        "controls": base / "controls/paraphrase_results_top5_1000_v2.jsonl",
    }


def aligned_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows = load_jsonl(path)
    by_id = {row_id(row): row for row in rows}
    if len(rows) != 1000 or len(by_id) != 1000 or not all(by_id):
        raise ValueError(
            f"{path}: expected 1000 unique non-empty query IDs, "
            f"got rows={len(rows)} unique={len(by_id)}"
        )
    return by_id


def summarize_dataset(root: Path, dataset: str) -> dict[str, Any]:
    paths = artifact_paths(root, dataset)
    tables = {name: aligned_rows(path) for name, path in paths.items()}
    ids = set(tables["retrieval"])
    for name, table in tables.items():
        if set(table) != ids:
            raise ValueError(f"{dataset}/{name}: query IDs do not align")

    baseline_names = sorted({
        canonical_method(name)
        for row in tables["baselines"].values()
        for name in row.get("methods", {})
    })
    control_names = sorted({
        canonical_method(name)
        for row in tables["controls"].values()
        for name in row.get("methods", {})
    })
    method_names = ["reflow", *baseline_names]
    if set(method_names) != set(control_names):
        raise ValueError(
            f"{dataset}: factual/control methods differ: "
            f"factual={method_names} control={control_names}"
        )

    counters = {name: empty_counter() for name in method_names}
    diagnostics = {
        name: {
            "factual_invalid": 0,
            "control_invalid": 0,
            "factual_unexecuted": 0,
            "control_unexecuted": 0,
        }
        for name in method_names
    }
    token_counts = []
    invalid_clean_ids = []
    valid_clean_ids = []
    mode = reader_mode(dataset)

    for identifier in sorted(ids):
        record = tables["retrieval"][identifier]
        reflow_row = tables["reflow"][identifier]
        baseline_row = tables["baselines"][identifier]
        control_row = tables["controls"][identifier]
        spec = gold_answer_spec(record, dataset)
        clean_answer = str(reflow_row.get("clean_answer", ""))
        if not valid_clean_row(reflow_row) or not spec.is_valid:
            invalid_clean_ids.append(identifier)
            continue
        valid_clean_ids.append(identifier)
        token_count, _ = reflow_policy_count(reflow_row)
        if token_count is None:
            raise ValueError(f"{dataset}/{identifier}: missing terminal budget")
        token_counts.append(int(token_count))

        factual_methods = {
            "reflow": reflow_row,
            **{
                canonical_method(name): value
                for name, value in baseline_row.get("methods", {}).items()
            },
        }
        control_methods = {
            canonical_method(name): value
            for name, value in control_row.get("methods", {}).items()
        }
        if set(factual_methods) != set(method_names):
            raise ValueError(f"{dataset}/{identifier}: missing factual method")
        if set(control_methods) != set(method_names):
            raise ValueError(f"{dataset}/{identifier}: missing control method")

        for name in method_names:
            factual = method_outcome(
                factual_methods[name], reflow=name == "reflow"
            )
            control = method_outcome(control_methods[name])
            diagnostics[name]["factual_invalid"] += int(factual.invalid)
            diagnostics[name]["control_invalid"] += int(control.invalid)
            diagnostics[name]["factual_unexecuted"] += int(
                not factual.executed and not factual.invalid
            )
            diagnostics[name]["control_unexecuted"] += int(
                not control.executed and not control.invalid
            )

            metric_population = {
                "answer": True,
                **{
                    metric: clean_correct(
                        metric,
                        clean_answer,
                        spec,
                        reader_mode=mode,
                    )
                    for metric in CORRECTNESS_METRICS
                },
            }
            for metric, eligible in metric_population.items():
                if not eligible:
                    continue
                factual_flip = event(
                    metric,
                    clean_answer=clean_answer,
                    outcome=factual,
                    spec=spec,
                    mode=mode,
                )
                control_flip = event(
                    metric,
                    clean_answer=clean_answer,
                    outcome=control,
                    spec=spec,
                    mode=mode,
                )
                if not factual.invalid:
                    counters[name]["fr_denominator"][metric] += 1
                    counters[name]["fr_flips"][metric] += int(factual_flip)
                if factual.invalid or control.invalid:
                    continue
                counters[name]["cfr_denominator"][metric] += 1
                counters[name]["cfr_factual_flips"][metric] += int(
                    factual_flip
                )
                counters[name]["cfr_control_flips"][metric] += int(
                    control_flip
                )
                counters[name]["cfr_delta_sum"][metric] += (
                    int(factual_flip) - int(control_flip)
                )

    methods = {}
    for name in method_names:
        counter = counters[name]
        method = {"diagnostics": diagnostics[name], "fr": {}, "cfr": {}}
        for metric in ("answer", *CORRECTNESS_METRICS):
            factual_denominator = counter["fr_denominator"][metric]
            paired_denominator = counter["cfr_denominator"][metric]
            method["fr"][metric] = {
                "flips": counter["fr_flips"][metric],
                "denominator": factual_denominator,
                "rate": counter["fr_flips"][metric]
                / max(1, factual_denominator),
            }
            method["cfr"][metric] = {
                "factual_flips": counter["cfr_factual_flips"][metric],
                "control_flips": counter["cfr_control_flips"][metric],
                "delta_sum": counter["cfr_delta_sum"][metric],
                "denominator": paired_denominator,
                "rate": counter["cfr_delta_sum"][metric]
                / max(1, paired_denominator),
            }
        methods[name] = method

    quartiles = statistics.quantiles(token_counts, n=4, method="inclusive")
    return {
        "dataset": dataset,
        "queries": 1000,
        "valid_clean_queries": len(valid_clean_ids),
        "invalid_clean_queries": len(invalid_clean_ids),
        "invalid_clean_ids": invalid_clean_ids,
        "methods": methods,
        "terminal_budget": {
            "population": "valid clean queries",
            "queries": len(token_counts),
            "sum": sum(token_counts),
            "mean": statistics.fmean(token_counts),
            "min": min(token_counts),
            "q1": quartiles[0],
            "median": statistics.median(token_counts),
            "q3": quartiles[2],
            "max": max(token_counts),
        },
        "artifacts": {
            name: {"path": str(path.resolve()), "sha256": sha256(path)}
            for name, path in paths.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--datasets", nargs="*", default=list(DATASETS))
    args = parser.parse_args()

    unknown = sorted(set(args.datasets).difference(DATASETS))
    if unknown:
        raise ValueError(f"unsupported datasets: {unknown}")
    rows = [summarize_dataset(args.root, dataset) for dataset in args.datasets]
    result = {
        "schema": "causalityrag.table3.unified.v1",
        "metric_contract": {
            "answer": "gold-free normalized clean-versus-edited answer change",
            "correctness": (
                "gold-aware loss on each metric's clean-correct population; "
                "PopQA and TriviaQA accept any frozen alias, while TimeQA "
                "requires every frozen answer component"
            ),
            "invalid": (
                "invalid clean answers are excluded; invalid factual/control "
                "outputs are excluded from the applicable denominator; legal "
                "unexecuted interventions contribute zero"
            ),
            "cfr": (
                "mean per-query factual flip minus meaning-preserving control "
                "flip on valid paired outputs"
            ),
        },
        "dataset_order": list(args.datasets),
        "datasets": {row["dataset"]: row for row in rows},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
