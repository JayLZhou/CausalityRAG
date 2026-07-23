"""Experimental ablation: materialize lambda and attention-matched edits."""

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
    parser.add_argument("--gate", required=True)
    parser.add_argument("--attention", required=True)
    parser.add_argument("--units-cache", required=True)
    parser.add_argument("--replacement-registry", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--lambda-value", type=float, default=1.0)
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

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as output:
        for gate_row in load_records(args.gate):
            identifier = str(gate_row.get("id", ""))
            record = records[identifier]
            units = units_from_cache_row(
                record,
                unit_rows[identifier],
                k=args.k,
            )
            unit_by_id = {str(unit["unit_id"]): unit for unit in units}
            registry = registries[identifier].get("replacements", {})
            candidates = gate_row.get("candidates", [])
            candidate = candidates[0] if candidates else None
            methods = {}
            if candidate is not None:
                reflow_ids = [str(value) for value in candidate["selected_ids"]]
                methods["reflow_lambda1"] = materialize(
                    record,
                    reflow_ids,
                    unit_by_id,
                    registry,
                    k=args.k,
                    metadata={
                        "lambda": float(candidate.get("lambda", args.lambda_value)),
                        "objective_value": candidate.get("objective_value"),
                        "remaining_support_fraction": candidate.get(
                            "remaining_support_fraction"
                        ),
                    },
                )

                attention = attention_rows[identifier]
                ranked_ids = [str(value) for value in attention.get("ranked_ids", [])]
                ranked_scores = [float(value) for value in attention.get("ranked_scores", [])]
                size = len(reflow_ids)
                if len(ranked_ids) < size:
                    methods["attention_matched"] = {
                        "status": "insufficient_attention_tokens",
                        "selected_ids": ranked_ids,
                        "selected_tokens": [
                            str(unit_by_id[unit_id].get("text", ""))
                            for unit_id in ranked_ids
                        ],
                        "n_selected": len(ranked_ids),
                        "required_tokens": size,
                    }
                else:
                    attention_ids = ranked_ids[:size]
                    total_score = sum(ranked_scores)
                    retained_score = total_score - sum(ranked_scores[:size])
                    methods["attention_matched"] = materialize(
                        record,
                        attention_ids,
                        unit_by_id,
                        registry,
                        k=args.k,
                        metadata={
                            "matched_to": "reflow_lambda1",
                            "remaining_attention_fraction": (
                                retained_score / total_score if total_score else 0.0
                            ),
                        },
                    )

            row = {
                "index": gate_row.get("index"),
                "id": identifier,
                "question": str(record.get("question", "")),
                "selection_mode": "lambda",
                "remaining_flow_threshold": args.lambda_value,
                "lambda": args.lambda_value,
                "gate_status": gate_row.get("status"),
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
    missing = [unit_id for unit_id in selected_ids if unit_id not in unit_by_id]
    if missing:
        return {
            "status": "missing_token_units",
            "selected_ids": selected_ids,
            "n_selected": len(selected_ids),
            "missing_ids": missing,
            **metadata,
        }
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
