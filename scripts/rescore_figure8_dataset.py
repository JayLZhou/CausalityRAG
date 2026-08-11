#!/usr/bin/env python3
"""Rescore saved Figure 8 baseline outputs against a final ReFlow anchor."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.evaluation_metrics import (
    gold_accuracy,
    gold_answer_spec,
    valid_answer,
)
from causalityrag.io import load_records, record_id
from causalityrag.reader import ReaderProtocolError, canonicalize_reader_answer
from scripts.audit_table3_token_counts import reflow_policy_count
from scripts.evaluate_reflow import valid_clean_answer


METHODS = ("mirage", "arc_jsd")
METHOD_ALIASES = {
    "mirage": ("mirage", "MIRAGE"),
    "arc_jsd": ("arc_jsd", "ARC-JSD"),
}
SOURCE_PATTERNS = (
    "audits/baseline_budget_to_reflow/final8_repair/*.jsonl",
    "audits/baseline_budget_to_reflow/exact_points_v2/*.jsonl",
    "audits/baseline_budget_to_reflow/exact_points/*.jsonl",
    "audits/baseline_budget_to_reflow/sequential_*/*.jsonl",
    "audits/baseline_budget_to_reflow/results_*.jsonl",
    "audits/fixed_budget_acc/results_1000.jsonl",
    "analysis/budget_sweep/coarse_results.jsonl",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def method_budget_ids(path: Path) -> dict[tuple[str, int], set[str]]:
    covered: dict[tuple[str, int], set[str]] = defaultdict(set)
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            identifier = record_id(row)
            for raw_method, budgets in row.get("methods", {}).items():
                method = next(
                    (
                        canonical
                        for canonical, aliases in METHOD_ALIASES.items()
                        if raw_method in aliases
                    ),
                    None,
                )
                if method is None:
                    continue
                for budget in budgets:
                    covered[(method, int(budget))].add(identifier)
    return covered


def discover_sources(
    dataset_root: Path,
    required_ids: set[str] | None = None,
) -> dict[tuple[str, int], Path]:
    candidates: dict[
        tuple[str, int], list[tuple[int, Path, set[str]]]
    ] = defaultdict(list)
    for priority, pattern in enumerate(SOURCE_PATTERNS):
        for path in sorted(dataset_root.glob(pattern)):
            for key, covered_ids in method_budget_ids(path).items():
                candidates[key].append((priority, path, covered_ids))
    if not candidates:
        raise FileNotFoundError(
            f"no saved Figure 8 result points found under {dataset_root}"
        )
    if required_ids is None:
        return {
            key: min(paths, key=lambda item: (item[0], str(item[1])))[1]
            for key, paths in candidates.items()
        }

    selected = {}
    for key, paths in candidates.items():
        complete = [
            (priority, path, covered_ids)
            for priority, path, covered_ids in paths
            if required_ids <= covered_ids
        ]
        if complete:
            selected[key] = min(
                complete,
                key=lambda item: (
                    item[0], -len(item[2]), str(item[1])
                ),
            )[1]
            continue
        selected[key] = max(
            paths,
            key=lambda item: (
                len(required_ids & item[2]),
                len(item[2]),
                -item[0],
            ),
        )[1]
    return selected


def canonical_answer(record: dict, answer: object, reader_mode: str) -> str:
    return canonicalize_reader_answer(
        str(record.get("question", "")),
        str(answer),
        reader_mode=reader_mode,
    )


def prepare_population(
    records: list[dict],
    reflow_rows: list[dict],
    *,
    dataset: str,
    reader_mode: str,
) -> tuple[dict[str, dict], dict]:
    records_by_id = {record_id(row): row for row in records}
    reflow_by_id = {record_id(row): row for row in reflow_rows}
    if len(records_by_id) != len(records) or records_by_id.keys() != reflow_by_id.keys():
        raise ValueError("retrieval/ReFlow IDs are not exactly aligned")

    population: dict[str, dict] = {}
    valid_reflow_rows = []
    reflow_flips = 0
    for identifier, record in records_by_id.items():
        row = reflow_by_id[identifier]
        try:
            clean = canonical_answer(
                record, row.get("clean_answer", ""), reader_mode
            )
        except ReaderProtocolError:
            continue
        spec = gold_answer_spec(record, dataset)
        if not valid_clean_answer(clean) or not spec.is_valid:
            continue
        if not gold_accuracy(clean, spec, reader_mode=reader_mode):
            continue
        population[identifier] = {
            "record": record,
            "gold_spec": spec,
            "clean_answer": clean,
        }

        status = str(row.get("evaluation_status", ""))
        called = int(row.get("reader_calls", 0)) > 0
        if status.startswith("protocol_violation"):
            continue
        edited = ""
        if called:
            try:
                edited = canonical_answer(
                    record, row.get("edited_answer", ""), reader_mode
                )
            except ReaderProtocolError:
                continue
            if not valid_answer(edited):
                continue
        valid_reflow_rows.append(row)
        reflow_flips += int(
            called
            and not gold_accuracy(edited, spec, reader_mode=reader_mode)
        )

    if not population or not valid_reflow_rows:
        raise ValueError("empty clean-accuracy or valid ReFlow population")
    token_counts = []
    for row in valid_reflow_rows:
        count, _ = reflow_policy_count(row)
        if count is None:
            raise ValueError(f"missing terminal token count for {record_id(row)}")
        token_counts.append(count)
    reflow = {
        "clean_acc_queries": len(population),
        "valid_answer_queries": len(valid_reflow_rows),
        "excluded_invalid_answers": len(population) - len(valid_reflow_rows),
        "acc_flip_count": reflow_flips,
        "acc_fr": reflow_flips / len(valid_reflow_rows),
        "edited_token_sum": sum(token_counts),
        "mean_modified_tokens": sum(token_counts) / len(token_counts),
    }
    return population, reflow


def rescore_entry(
    entry: dict,
    population_row: dict,
    *,
    dataset: str,
    reader_mode: str,
) -> tuple[bool, bool, int]:
    status = str(entry.get("status", ""))
    if status.startswith("protocol_violation"):
        return False, False, 0
    called = bool(entry.get("reader_called", bool(entry.get("edited_answer"))))
    edited_raw = str(entry.get("edited_answer", ""))
    if not edited_raw.strip():
        return (
            not called,
            False,
            int(entry.get("n_modified_tokens", 0)),
        )
    record = population_row["record"]
    try:
        edited = canonical_answer(record, edited_raw, reader_mode)
    except ReaderProtocolError:
        return False, False, 0
    if not valid_clean_answer(edited):
        return False, False, 0
    flip = not gold_accuracy(
        edited,
        population_row["gold_spec"],
        reader_mode=reader_mode,
    )
    return True, flip, int(entry.get("n_modified_tokens", 0))


def rescore_sources(
    selected: dict[tuple[str, int], Path],
    population: dict[str, dict],
    expected_ids: set[str],
    *,
    dataset: str,
    reader_mode: str,
    allow_missing: bool = False,
) -> tuple[dict[str, dict[str, dict]], dict[str, dict]]:
    keys_by_path: dict[Path, set[tuple[str, int]]] = defaultdict(set)
    for key, path in selected.items():
        keys_by_path[path].add(key)

    curves: dict[str, dict[str, dict]] = {method: {} for method in METHODS}
    sources = {}
    for path, keys in sorted(keys_by_path.items(), key=lambda item: str(item[0])):
        rows = load_records(path)
        by_id = {record_id(row): row for row in rows}
        if len(by_id) != len(rows) or not by_id.keys() <= expected_ids:
            raise ValueError(
                f"saved point ID mismatch: {path} rows={len(rows)} "
                f"unique={len(by_id)} expected={len(expected_ids)}"
            )
        missing_population = population.keys() - by_id.keys()
        if missing_population and not allow_missing:
            raise ValueError(
                f"{path} omits {len(missing_population)} final clean-accuracy "
                f"queries; examples={sorted(missing_population)[:3]}"
            )
        sources[str(path.resolve())] = {
            "sha256": sha256(path),
            "method_budgets": [
                f"{method}:{budget}" for method, budget in sorted(keys)
            ],
        }
        for method, budget in sorted(keys):
            valid = flips = token_sum = executed = 0
            missing_ids = []
            for identifier, population_row in population.items():
                row = by_id.get(identifier)
                if row is None:
                    missing_ids.append(identifier)
                    continue
                row_methods = row.get("methods", {})
                method_rows = next(
                    (
                        row_methods[alias]
                        for alias in METHOD_ALIASES[method]
                        if alias in row_methods
                    ),
                    {},
                )
                entry = (
                    method_rows.get(str(budget))
                )
                if entry is None:
                    missing_ids.append(identifier)
                    continue
                is_valid, flip, token_count = rescore_entry(
                    entry,
                    population_row,
                    dataset=dataset,
                    reader_mode=reader_mode,
                )
                if not is_valid:
                    continue
                valid += 1
                executed += int(bool(entry.get("edited_answer")))
                flips += int(flip)
                token_sum += token_count
            if missing_ids and not allow_missing:
                raise ValueError(
                    f"{path} lacks {method}/{budget} for {len(missing_ids)} "
                    "final clean-accuracy queries"
                )
            if not valid:
                raise ValueError(f"no valid answers for {method}/{budget}")
            curves[method][str(budget)] = {
                "clean_acc_queries": len(population),
                "valid_answer_queries": valid,
                "protocol_violations": len(population) - valid,
                "executed_queries": executed,
                "acc_flip_count": flips,
                "acc_fr": flips / valid,
                "edited_token_sum": token_sum,
                "mean_modified_tokens": token_sum / valid,
                "missing_population_queries": len(missing_ids),
                "missing_population_ids": sorted(missing_ids),
            }
    return curves, sources


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reflow-results", type=Path, required=True)
    parser.add_argument("--reader-mode", default="short_answer")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    records = load_records(args.input)[: args.n]
    reflow_rows = load_records(args.reflow_results)[: args.n]
    if len(records) != args.n or len(reflow_rows) != args.n:
        raise ValueError("expected exactly --n retrieval and ReFlow rows")
    population, reflow = prepare_population(
        records,
        reflow_rows,
        dataset=args.dataset,
        reader_mode=args.reader_mode,
    )
    selected = discover_sources(args.dataset_root, set(population))
    curves, sources = rescore_sources(
        selected,
        population,
        {record_id(row) for row in records},
        dataset=args.dataset,
        reader_mode=args.reader_mode,
        allow_missing=args.allow_incomplete,
    )
    missing_points = [
        {
            "method": method,
            "budget": int(budget),
            "missing_queries": point["missing_population_queries"],
            "missing_ids": point["missing_population_ids"],
        }
        for method, curve in curves.items()
        for budget, point in curve.items()
        if point["missing_population_queries"]
    ]
    result = {
        "schema": "causalityrag.figure8_dataset_rescore.v1",
        "dataset": args.dataset,
        "metric": "Acc-FR",
        "contract": (
            "clean-accuracy population; invalid edited answers are excluded "
            "from both numerator and denominator"
        ),
        "reflow": reflow,
        "methods": curves,
        "complete": not missing_points,
        "missing_points": missing_points,
        "selected_sources": sources,
        "input": {"path": str(args.input.resolve()), "sha256": sha256(args.input)},
        "reflow_results": {
            "path": str(args.reflow_results.resolve()),
            "sha256": sha256(args.reflow_results),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
