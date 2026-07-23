"""Experimental ablation: materialize frozen contribution-flow replacements."""

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
    parser.add_argument("--selections", nargs="+", required=True)
    parser.add_argument("--units-cache", required=True)
    parser.add_argument("--replacement-registry", required=True)
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--budgets", default="1,3,5")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--common-complete-only",
        action="store_true",
        help="retain only queries with every method and requested budget",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    budgets = sorted({int(value) for value in args.budgets.split(",") if value})
    records = {record_id(row): row for row in load_records(args.input)}
    units = {str(row["id"]): row for row in load_records(args.units_cache)}
    registries = {
        str(row["id"]): row for row in load_records(args.replacement_registry)
    }
    methods: dict[str, dict[str, dict]] = {}
    variants = set()
    for path in args.selections:
        for row in load_records(path):
            identifier = str(row["id"])
            variant = str(row["variant"])
            variants.add(variant)
            methods.setdefault(identifier, {})[variant] = row
    required_variants = sorted(variants)

    outputs = {}
    for budget in budgets:
        path = f"{args.out_prefix}_b{budget}.jsonl"
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        outputs[budget] = open(path, "w", encoding="utf-8")
    try:
        for identifier, by_variant in methods.items():
            if identifier not in records:
                continue
            if args.common_complete_only and any(
                variant not in by_variant
                or by_variant[variant].get("status") != "ok"
                or any(
                    str(budget) not in by_variant[variant].get("prefixes", {})
                    for budget in budgets
                )
                for variant in required_variants
            ):
                continue
            record = records[identifier]
            unit_row = units[identifier]
            registry = registries[identifier].get("replacements", {})
            token_units = units_from_cache_row(record, unit_row, k=args.k)
            unit_by_id = {
                str(unit["unit_id"]): unit for unit in token_units
            }
            for budget in budgets:
                method_rows = {}
                for variant in required_variants:
                    selection = by_variant.get(variant, {})
                    prefix = selection.get("prefixes", {}).get(str(budget))
                    if prefix is None:
                        continue
                    selected_ids = [str(value) for value in prefix["selected_ids"]]
                    selected = [unit_by_id[unit_id] for unit_id in selected_ids]
                    revision = apply_token_replacements(
                        record,
                        selected,
                        registry,
                        k=args.k,
                    )
                    status = (
                        "ok"
                        if revision["n_edits"] == len(selected_ids)
                        and revision["n_failed_edits"] == 0
                        else "replacement_failed"
                    )
                    method_rows[variant] = {
                        "status": status,
                        "selected_ids": selected_ids,
                        "selected_tokens": [
                            str(unit.get("text", "")) for unit in selected
                        ],
                        "n_selected": len(selected_ids),
                        "remaining_support_fraction": prefix[
                            "remaining_support_fraction"
                        ],
                        "edits": revision["edits"],
                    }
                if args.common_complete_only and len(method_rows) != len(
                    required_variants
                ):
                    continue
                output_row = {
                    "index": next(iter(by_variant.values())).get("index"),
                    "id": identifier,
                    "question": str(record.get("question", "")),
                    "selection_mode": "budget",
                    "token_budget": budget,
                    "methods": method_rows,
                }
                outputs[budget].write(
                    json.dumps(output_row, ensure_ascii=False) + "\n"
                )
    finally:
        for output in outputs.values():
            output.close()


if __name__ == "__main__":
    main()
