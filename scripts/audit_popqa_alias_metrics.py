#!/usr/bin/env python3
"""Audit PopQA correctness and CFR metrics against every gold alias."""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.io import load_records, record_id
from causalityrag.reader import answer_token_f1, answers_exact_match


def aliases(row: dict) -> list[str]:
    values = row.get("gold_answers") or [row.get("gold_answer", "")]
    return [str(value) for value in values if str(value).strip()]


def best_f1(answer: str, accepted: list[str]) -> float:
    return max(
        (answer_token_f1(answer, gold) for gold in accepted),
        default=0.0,
    )


def matching_aliases(answer: str, accepted: list[str]) -> list[int]:
    return [
        index
        for index, gold in enumerate(accepted)
        if answers_exact_match(answer, gold)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=1000)
    parser.add_argument("--include-ids", action="store_true")
    args = parser.parse_args()

    rows = load_records(args.results)
    if len(rows) != args.expected_rows:
        raise ValueError(
            f"expected {args.expected_rows} rows, got {len(rows)}"
        )
    identifiers = [record_id(row) for row in rows]
    if any(not identifier for identifier in identifiers):
        raise ValueError("every result row must have a nonempty ID")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("result IDs must be unique")

    clean_exact_population = 0
    clean_f1_population = 0
    clean_exact_lost = 0
    f1_decreased = 0
    retained_correct = 0
    retained_only_via_noncanonical_alias = 0
    canonical_only_false_errors = 0
    stored_clean_correct_mismatches = []
    clean_f1_population_ids = []
    f1_decreased_ids = []
    edited_f1_on_clean_correct = []

    for row in rows:
        accepted = aliases(row)
        clean = str(row.get("clean_answer", ""))
        edited = str(row.get("edited_answer", ""))
        clean_matches = matching_aliases(clean, accepted)
        edited_matches = matching_aliases(edited, accepted)
        clean_f1 = best_f1(clean, accepted)
        edited_f1 = best_f1(edited, accepted)
        clean_exact = bool(clean_matches)
        clean_f1_one = clean_f1 >= 1.0 - 1e-12

        if bool(row.get("clean_correct")) != clean_exact:
            stored_clean_correct_mismatches.append(record_id(row))
        clean_exact_population += int(clean_exact)
        clean_f1_population += int(clean_f1_one)
        if clean_f1_one:
            clean_f1_population_ids.append(record_id(row))
            edited_f1_on_clean_correct.append(edited_f1)
            decreased = edited_f1 < clean_f1 - 1e-12
            f1_decreased += int(decreased)
            if decreased:
                f1_decreased_ids.append(record_id(row))
        if clean_exact:
            clean_exact_lost += int(not edited_matches)
            retained_correct += int(bool(edited_matches))
            retained_only_via_noncanonical_alias += int(
                bool(edited_matches) and 0 not in edited_matches
            )
            canonical_only_false_errors += int(
                bool(edited_matches)
                and bool(accepted)
                and not answers_exact_match(edited, accepted[0])
            )

    status_histogram = collections.Counter(
        str(row.get("evaluation_status", "")) for row in rows
    )
    alias_count_histogram = collections.Counter(
        len(aliases(row)) for row in rows
    )
    audit = {
        "schema": "causalityrag.popqa_alias_metric_audit.v1",
        "results": str(args.results.resolve()),
        "queries": len(rows),
        "status_histogram": dict(sorted(status_histogram.items())),
        "alias_count_histogram": {
            str(key): value
            for key, value in sorted(alias_count_histogram.items())
        },
        "queries_with_multiple_aliases": sum(
            len(aliases(row)) > 1 for row in rows
        ),
        "clean_exact_any_alias": clean_exact_population,
        "clean_f1_one_any_alias": clean_f1_population,
        "stored_clean_correct_mismatches": stored_clean_correct_mismatches,
        "acc_cfr_numerator": clean_exact_lost,
        "acc_cfr_denominator": clean_exact_population,
        "acc_cfr": clean_exact_lost / max(1, clean_exact_population),
        "f1_cfr_numerator": f1_decreased,
        "f1_cfr_denominator": clean_f1_population,
        "f1_cfr": f1_decreased / max(1, clean_f1_population),
        "clean_exact_becomes_incorrect_any_alias": clean_exact_lost,
        "edited_remains_correct_any_alias": retained_correct,
        "edited_correct_only_via_noncanonical_alias": (
            retained_only_via_noncanonical_alias
        ),
        "canonical_only_would_false_count_as_incorrect": (
            canonical_only_false_errors
        ),
        "edited_f1_on_clean_correct": {
            "zero": sum(value <= 1e-12 for value in edited_f1_on_clean_correct),
            "partial": sum(
                1e-12 < value < 1.0 - 1e-12
                for value in edited_f1_on_clean_correct
            ),
            "one": sum(
                value >= 1.0 - 1e-12
                for value in edited_f1_on_clean_correct
            ),
            "mean": (
                sum(edited_f1_on_clean_correct)
                / max(1, len(edited_f1_on_clean_correct))
            ),
        },
    }
    if args.include_ids:
        audit["clean_f1_population_ids"] = clean_f1_population_ids
        audit["f1_decreased_ids"] = f1_decreased_ids
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
