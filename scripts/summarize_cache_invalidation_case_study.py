#!/usr/bin/env python3
"""Evaluate answer-cache invalidation from token-attribution signatures."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.io import load_records, record_id


DEFAULT_BUDGETS = (1, 2, 3, 5, 8, 10)


def index_rows(path: str) -> dict[str, dict]:
    rows = load_records(path)
    indexed = {record_id(row): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"duplicate IDs in {path}")
    return indexed


def deduplicate_unit_ids(unit_ids: list[str]) -> list[str]:
    result = []
    seen = set()
    for unit_id in unit_ids:
        unit_id = str(unit_id)
        if unit_id and unit_id not in seen:
            seen.add(unit_id)
            result.append(unit_id)
    return result


def project_units_to_sentences(
    unit_ids: list[str], sentence_by_unit: dict[str, str]
) -> list[str]:
    return deduplicate_unit_ids(
        [sentence_by_unit.get(str(unit_id), "") for unit_id in unit_ids]
    )


def reflow_rank(row: dict) -> list[str]:
    candidates = sorted(
        row.get("frontier_candidates", []),
        key=lambda candidate: (
            int(candidate.get("n_selected", 1 << 30)),
            float(candidate.get("remaining_support_flow", float("inf"))),
        ),
    )
    result = []
    seen = set()
    for candidate in candidates:
        for unit_id in candidate.get("selected_ids", []):
            unit_id = str(unit_id)
            if unit_id not in seen:
                seen.add(unit_id)
                result.append(unit_id)
    return result


def score_rank(row: dict) -> list[str]:
    if "ranked_ids" in row:
        return [str(unit_id) for unit_id in row.get("ranked_ids", [])]
    scores = row.get("token_scores", {})
    return [
        str(unit_id)
        for unit_id, _ in sorted(
            scores.items(),
            key=lambda item: (-float(item[1]), str(item[0])),
        )
    ]


def stable_random_units(
    unit_ids: list[str], *, query_id: str, seed: int
) -> list[str]:
    return sorted(
        unit_ids,
        key=lambda unit_id: hashlib.sha256(
            f"{seed}\0{query_id}\0{unit_id}".encode("utf-8")
        ).digest(),
    )


def metrics(labels: list[bool], predictions: list[bool]) -> dict:
    true_positive = sum(label and prediction for label, prediction in zip(labels, predictions))
    false_positive = sum(
        not label and prediction for label, prediction in zip(labels, predictions)
    )
    false_negative = sum(
        label and not prediction for label, prediction in zip(labels, predictions)
    )
    true_negative = sum(
        not label and not prediction for label, prediction in zip(labels, predictions)
    )
    total = len(labels)
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "stale_recall": true_positive / max(1, true_positive + false_negative),
        "precision": true_positive / max(1, true_positive + false_positive),
        "invalidation_rate": (true_positive + false_positive) / max(1, total),
        "cache_reuse_rate": (true_negative + false_negative) / max(1, total),
        "stale_answer_rate": false_negative / max(1, total),
    }


def covered_by_signature(
    update_ids: set[str], ranked_ids: list[str], budget: int
) -> bool:
    return bool(update_ids & set(ranked_ids[:budget]))


def evaluate_curves(
    *,
    labels: list[bool],
    trials: list[dict],
    update_ids: list[set[str]],
    ranked_ids: dict[str, dict[str, list[str]]],
    budgets: list[int],
) -> dict[str, dict[str, dict]]:
    curves = {}
    for method, signatures in ranked_ids.items():
        curve = {}
        for budget in budgets:
            predictions = [
                covered_by_signature(
                    update_ids[index], signatures[record_id(row)], budget
                )
                for index, row in enumerate(trials)
            ]
            curve[str(budget)] = metrics(labels, predictions)
        curves[method] = curve
    return curves


def average_random_curves(
    random_curves: list[dict[str, dict]], budgets: list[int]
) -> dict[str, dict]:
    return {
        str(budget): {
            key: statistics.mean(curve[str(budget)][key] for curve in random_curves)
            for key in random_curves[0][str(budget)]
        }
        for budget in budgets
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stress-results", required=True)
    parser.add_argument("--units", required=True)
    parser.add_argument("--reflow-frontier", required=True)
    parser.add_argument("--mirage", required=True)
    parser.add_argument("--arc-jsd", required=True)
    parser.add_argument("--integrated-gradients", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=DEFAULT_BUDGETS)
    parser.add_argument("--random-seeds", type=int, default=5)
    args = parser.parse_args()

    trials = load_records(args.stress_results)
    units = index_rows(args.units)
    sources = {
        "reflow": index_rows(args.reflow_frontier),
        "mirage": index_rows(args.mirage),
        "arc_jsd": index_rows(args.arc_jsd),
        "integrated_gradients": index_rows(args.integrated_gradients),
    }
    query_ids = sorted({record_id(row) for row in trials})
    eligible_units_by_query = {
        query_id: [
            str(unit["unit_id"])
            for unit in units[query_id].get("units", [])
            if unit.get("unit_id") is not None
        ]
        for query_id in query_ids
    }
    sentence_by_query = {
        query_id: {
            str(unit["unit_id"]): str(unit.get("sentence_id", ""))
            for unit in units[query_id].get("units", [])
            if unit.get("unit_id") is not None
        }
        for query_id in query_ids
    }
    ranked_units: dict[str, dict[str, list[str]]] = {}
    for method, source in sources.items():
        ranked_units[method] = {}
        for query_id in query_ids:
            unit_ids = (
                reflow_rank(source[query_id])
                if method == "reflow"
                else score_rank(source[query_id])
            )
            eligible = set(eligible_units_by_query[query_id])
            ranked_units[method][query_id] = [
                unit_id
                for unit_id in deduplicate_unit_ids(unit_ids)
                if unit_id in eligible
            ]
    ranked_sentences = {
        method: {
            query_id: project_units_to_sentences(
                unit_ids, sentence_by_query[query_id]
            )
            for query_id, unit_ids in signatures.items()
        }
        for method, signatures in ranked_units.items()
    }

    labels = [bool(row.get("correctness_flip")) for row in trials]
    update_units = [
        {str(unit_id) for unit_id in row.get("selected_ids", [])}
        for row in trials
    ]
    update_sentences = [
        {
            sentence_by_query[record_id(row)].get(str(unit_id), "")
            for unit_id in row.get("selected_ids", [])
        }
        - {""}
        for row in trials
    ]

    token_curves = evaluate_curves(
        labels=labels,
        trials=trials,
        update_ids=update_units,
        ranked_ids=ranked_units,
        budgets=args.budgets,
    )
    sentence_curves = evaluate_curves(
        labels=labels,
        trials=trials,
        update_ids=update_sentences,
        ranked_ids=ranked_sentences,
        budgets=args.budgets,
    )

    random_token_curves = []
    random_sentence_curves = []
    for seed in range(args.random_seeds):
        token_signatures = {
            query_id: stable_random_units(
                eligible_units_by_query[query_id],
                query_id=query_id,
                seed=seed,
            )
            for query_id in query_ids
        }
        sentence_signatures = {
            query_id: stable_random_units(
                sorted(set(sentence_by_query[query_id].values()) - {""}),
                query_id=query_id,
                seed=seed,
            )
            for query_id in query_ids
        }
        random_token_curves.append(
            evaluate_curves(
                labels=labels,
                trials=trials,
                update_ids=update_units,
                ranked_ids={"random": token_signatures},
                budgets=args.budgets,
            )["random"]
        )
        random_sentence_curves.append(
            evaluate_curves(
                labels=labels,
                trials=trials,
                update_ids=update_sentences,
                ranked_ids={"random": sentence_signatures},
                budgets=args.budgets,
            )["random"]
        )
    token_curves["random_5seed_mean"] = average_random_curves(
        random_token_curves, args.budgets
    )
    sentence_curves["random_5seed_mean"] = average_random_curves(
        random_sentence_curves, args.budgets
    )

    results = {
        "retrieved_token_position": {"curves": token_curves},
        "evidence_sentence": {"curves": sentence_curves},
    }

    output = {
        "schema": "causalityrag.cache_invalidation_case_study.v3",
        "monitoring_units": list(results),
        "queries": len(query_ids),
        "update_trials": len(trials),
        "factual_tokens_per_update": len(trials[0].get("selected_ids", [])),
        "budgets": list(args.budgets),
        "stale_updates": sum(labels),
        "no_invalidation_stale_answer_rate": statistics.mean(float(value) for value in labels),
        "eager_invalidation": {
            "stale_recall": 1.0,
            "precision": statistics.mean(float(value) for value in labels),
            "invalidation_rate": 1.0,
            "cache_reuse_rate": 0.0,
            "stale_answer_rate": 0.0,
        },
        "results": results,
    }
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
