"""Select Top-K valid tokens from cached MIRAGE token scores."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.io import iter_records, load_records, record_id
from causalityrag.token_units import units_from_context_row
from exp.score_mirage_tokens import ranked_candidate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--context-units", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", default="")
    parser.add_argument("--replacement-registry", default="")
    parser.add_argument(
        "--replacement-registry-policy",
        choices=("allow-only", "exclude-known-invalid"),
        default="allow-only",
    )
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--top-tokens", type=int, default=5)
    args = parser.parse_args()

    score_by_id = {
        str(row["id"]): row
        for row in load_records(args.scores)
    }
    units_by_id = {
        str(row["id"]): row
        for row in load_records(args.context_units)
    }
    registry_by_id = (
        {
            str(row["id"]): row
            for row in load_records(args.replacement_registry)
        }
        if args.replacement_registry
        else {}
    )
    rows = []
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as output:
        for index, record in enumerate(iter_records(args.input, args.n)):
            identifier = record_id(record)
            score_row = score_by_id.get(identifier)
            units_row = units_by_id.get(identifier)
            if score_row is None or units_row is None:
                raise ValueError(f"missing cached row for {identifier}")
            units = units_from_context_row(record, units_row, k=args.k)
            known_registry_ids: set[str] | None = None
            allowed_ids = None
            if registry_by_id:
                registry_row = registry_by_id.get(identifier)
                if registry_row is None:
                    raise ValueError(
                        f"missing replacement registry row for {identifier}"
                    )
                replacements = registry_row.get("replacements", {})
                invalid_ids = {
                    str(unit_id)
                    for unit_id in registry_row.get("invalid", {})
                }
                known_registry_ids = {
                    str(unit_id)
                    for unit_id in replacements
                } | invalid_ids
                if args.replacement_registry_policy == "allow-only":
                    allowed_ids = {
                        str(unit_id)
                        for unit_id, replacement in replacements.items()
                        if isinstance(replacement, dict)
                        and replacement.get("ok")
                    }
                else:
                    allowed_ids = {
                        str(unit["unit_id"])
                        for unit in units
                    } - invalid_ids
            candidate = ranked_candidate(
                units,
                score_row.get("token_scores", {}),
                top_tokens=args.top_tokens,
                allowed_ids=allowed_ids,
            )
            selected_ids = (
                set(candidate["selected_ids"])
                if candidate is not None
                else set()
            )
            misses = sorted(
                selected_ids - known_registry_ids
                if known_registry_ids is not None
                else ()
            )
            if score_row.get("status") == "reader_abstention_empty_answer":
                status = "reader_abstention_empty_answer"
            elif candidate is None:
                status = "no_positive_mirage_token"
            elif int(candidate["n_selected"]) < args.top_tokens:
                status = "insufficient_editable_tokens"
            else:
                status = "ok"
            row = {
                "index": index,
                "id": identifier,
                "question": str(record.get("question", "")),
                "status": status,
                "baseline": f"mirage_top{args.top_tokens}",
                "top_tokens": args.top_tokens,
                "replacement_registry": args.replacement_registry,
                "replacement_registry_policy": (
                    args.replacement_registry_policy
                ),
                "registry_candidate_misses": misses,
                "selection_candidate": candidate,
                "candidates": (
                    [candidate]
                    if candidate is not None
                    else []
                ),
            }
            rows.append(row)
            output.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = summarize(rows, args)
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    print("[mirage-selection summary]", rendered)
    if args.summary_out:
        with open(args.summary_out, "w", encoding="utf-8") as output:
            output.write(rendered + "\n")


def summarize(rows: list[dict], args) -> dict:
    counts = [
        int(row["selection_candidate"]["n_selected"])
        for row in rows
        if row.get("selection_candidate")
    ]
    misses = {
        (str(row["id"]), str(unit_id))
        for row in rows
        for unit_id in row.get("registry_candidate_misses", [])
    }
    histogram: dict[str, int] = {}
    for row in rows:
        status = str(row["status"])
        histogram[status] = histogram.get(status, 0) + 1
    return {
        "baseline": f"mirage_top{args.top_tokens}",
        "top_tokens": args.top_tokens,
        "queries": len(rows),
        "status_histogram": histogram,
        "queries_with_exact_topk": sum(
            count == args.top_tokens
            for count in counts
        ),
        "queries_with_any_candidate": len(counts),
        "mean_selected_tokens": (
            statistics.fmean(counts)
            if counts
            else None
        ),
        "replacement_registry": args.replacement_registry or None,
        "registry_fixed_point": (
            bool(args.replacement_registry)
            and not misses
        ),
        "evaluated_candidate_registry_misses": len(misses),
    }


if __name__ == "__main__":
    main()
