"""Experimental baseline: match attention selections to reference cardinality."""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.io import load_records, record_id  # noqa: E402
from causalityrag.revision import apply_token_replacements  # noqa: E402
from causalityrag.token_units import units_from_cache_row  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--reference-attacks", nargs="+", required=True)
    parser.add_argument("--reference-method", default="residual_flow")
    parser.add_argument("--reference-name", default="reflow")
    parser.add_argument("--attention", required=True)
    parser.add_argument("--units-cache", required=True)
    parser.add_argument("--replacement-registry", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--k", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = {record_id(row): row for row in load_records(args.input)}
    unit_rows = {str(row["id"]): row for row in load_records(args.units_cache)}
    registries = {
        str(row["id"]): row for row in load_records(args.replacement_registry)
    }
    attention_rows = {
        str(row["id"]): row for row in load_records(args.attention)
    }
    reference_rows = [
        row
        for path in args.reference_attacks
        for row in load_records(path)
    ]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as output:
        for reference_row in reference_rows:
            identifier = str(reference_row["id"])
            record = records[identifier]
            units = units_from_cache_row(
                record,
                unit_rows[identifier],
                k=args.k,
            )
            unit_by_id = {str(unit["unit_id"]): unit for unit in units}
            registry = registries[identifier].get("replacements", {})
            reference = reference_row.get("methods", {}).get(
                args.reference_method,
                {},
            )
            reference_ok = reference.get("status") == "ok"
            reference_ids = (
                [str(value) for value in reference.get("selected_ids", [])]
                if reference_ok
                else []
            )
            size = len(reference_ids)

            methods = {
                args.reference_name: materialize(
                    record,
                    reference_ids,
                    unit_by_id,
                    registry,
                    k=args.k,
                    metadata={
                        "source_status": reference.get("status"),
                        "candidate_available": reference_ok,
                    },
                )
            }
            attention = attention_rows[identifier]
            ranked_ids = [str(value) for value in attention.get("ranked_ids", [])]
            if len(ranked_ids) < size:
                methods["attention_matched"] = {
                    "status": "insufficient_attention_tokens",
                    "selected_ids": ranked_ids,
                    "n_selected": len(ranked_ids),
                    "required_tokens": size,
                }
            else:
                methods["attention_matched"] = materialize(
                    record,
                    ranked_ids[:size],
                    unit_by_id,
                    registry,
                    k=args.k,
                    metadata={
                        "matched_to": args.reference_name,
                        "candidate_available": reference_ok,
                    },
                )

            row = {
                "index": reference_row.get("index"),
                "id": identifier,
                "question": str(record.get("question", "")),
                "selection_mode": "matched",
                "remaining_flow_threshold": reference_row.get(
                    "remaining_flow_threshold"
                ),
                "methods": methods,
            }
            output.write(json.dumps(row, ensure_ascii=False) + "\n")


def materialize(
    record: dict,
    selected_ids: list[str],
    unit_by_id: dict[str, dict],
    registry: dict,
    *,
    k: int,
    metadata: dict,
) -> dict:
    selected = [unit_by_id[unit_id] for unit_id in selected_ids]
    revision = apply_token_replacements(
        record,
        selected,
        registry,
        k=k,
    )
    status = (
        "ok"
        if revision["n_edits"] == len(selected_ids)
        and revision["n_failed_edits"] == 0
        else "replacement_failed"
    )
    return {
        "status": status,
        "selected_ids": selected_ids,
        "selected_tokens": [str(unit.get("text", "")) for unit in selected],
        "n_selected": len(selected_ids),
        "edits": revision["edits"],
        **metadata,
    }


if __name__ == "__main__":
    main()
