"""Experimental ablation: select from alternative contribution graphs."""

from __future__ import annotations

import argparse
from itertools import islice
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.io import iter_records, load_records, record_id  # noqa: E402
from causalityrag.mixed_cut import (  # noqa: E402
    build_projected_token_contribution_network,
    build_raw_contribution_network,
    remaining_support_flow,
    restrict_group_editable_units,
)
from causalityrag.token_units import units_from_cache_row  # noqa: E402


VARIANTS = {
    "raw_r": ("input-roots", "raw"),
    "conserved_f": ("input-roots", "backward-conserved"),
    "aggregated_c": ("layer-copy-token", "backward-conserved"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--graphs", required=True)
    parser.add_argument("--units-cache", required=True)
    parser.add_argument("--replacement-registry", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", default="")
    parser.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--budgets", default="1,3,5")
    return parser.parse_args()


def greedy_flow_prefixes(network, budgets: list[int]) -> tuple[dict, int]:
    """Greedily choose the token with largest exact residual-flow reduction."""

    maximum_budget = max(budgets)
    available = sorted(set(network.selection_unit_by_gate.values()))
    selected: set[str] = set()
    cache: dict[frozenset[str], float] = {}
    flow_calls = 0

    def measured(selection: frozenset[str]) -> float:
        nonlocal flow_calls
        if selection not in cache:
            cache[selection] = remaining_support_flow(network, selection)
            flow_calls += 1
        return cache[selection]

    initial_flow = measured(frozenset())
    prefixes = {}
    for step in range(1, maximum_budget + 1):
        candidates = [unit_id for unit_id in available if unit_id not in selected]
        if not candidates:
            break
        best_id, best_flow = min(
            (
                (unit_id, measured(frozenset(selected | {unit_id})))
                for unit_id in candidates
            ),
            key=lambda item: (
                item[1],
                -float(network.unit_scores.get(item[0], 0.0)),
                item[0],
            ),
        )
        selected.add(best_id)
        if step in budgets:
            prefixes[str(step)] = {
                "budget": step,
                "selected_ids": sorted(selected),
                "n_selected": len(selected),
                "remaining_support_flow": best_flow,
                "remaining_support_fraction": (
                    best_flow / initial_flow if initial_flow > 0 else 0.0
                ),
            }
    return {
        "initial_flow": initial_flow,
        "available_tokens": len(available),
        "prefixes": prefixes,
    }, flow_calls


def main() -> None:
    args = parse_args()
    budgets = sorted({int(value) for value in args.budgets.split(",") if value})
    if not budgets or budgets[0] <= 0:
        raise ValueError("budgets must contain positive integers")
    units_by_id = {
        str(row["id"]): row for row in load_records(args.units_cache)
    }
    registry_by_id = {
        str(row["id"]): row for row in load_records(args.replacement_registry)
    }
    projection, capacity_mode = VARIANTS[args.variant]
    records = islice(iter_records(args.input), args.start, args.start + args.n)
    graphs = islice(iter_records(args.graphs), args.start, args.start + args.n)

    rows = []
    total_flow_calls = 0
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as output:
        for offset, (record, graph_row) in enumerate(zip(records, graphs)):
            started = time.monotonic()
            index = args.start + offset
            identifier = record_id(record)
            if str(graph_row.get("id")) != identifier:
                raise ValueError(f"input/graph mismatch at row {index}")
            units_row = units_by_id.get(identifier)
            registry_row = registry_by_id.get(identifier)
            if units_row is None or registry_row is None:
                raise ValueError(f"missing cache/registry row for {identifier}")
            units = units_from_cache_row(record, units_row, k=args.k)
            by_id = {str(unit["unit_id"]): unit for unit in units}
            if projection == "layer-copy-token":
                network = build_projected_token_contribution_network(
                    graph_row,
                    units,
                    capacity_mode=capacity_mode,
                )
            else:
                network = build_raw_contribution_network(
                    graph_row,
                    units,
                    capacity_mode=capacity_mode,
                )
            valid_ids = {
                str(unit_id)
                for unit_id, replacement in registry_row.get(
                    "replacements",
                    {},
                ).items()
                if isinstance(replacement, dict) and replacement.get("ok")
            }
            network = restrict_group_editable_units(network, valid_ids)
            if network.status == "ok":
                selection, flow_calls = greedy_flow_prefixes(network, budgets)
                total_flow_calls += flow_calls
                prefixes = selection["prefixes"]
                for prefix in prefixes.values():
                    prefix["selected_tokens"] = [
                        str(by_id[unit_id].get("text", ""))
                        for unit_id in prefix["selected_ids"]
                    ]
                status = (
                    "ok"
                    if str(max(budgets)) in prefixes
                    else "insufficient_editable_tokens"
                )
            else:
                selection = {
                    "initial_flow": 0.0,
                    "available_tokens": 0,
                    "prefixes": {},
                }
                flow_calls = 0
                status = network.status
            row = {
                "index": index,
                "id": identifier,
                "question": str(record.get("question", "")),
                "variant": args.variant,
                "projection": projection,
                "capacity_mode": capacity_mode,
                "status": status,
                "network_status": network.status,
                "network_diagnostics": network.diagnostics,
                "flow_calls": flow_calls,
                "elapsed_seconds": time.monotonic() - started,
                **selection,
            }
            rows.append(row)
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            print(
                f"[flow-ablation] {offset + 1}/{args.n} "
                f"variant={args.variant} status={status} calls={flow_calls} "
                f"seconds={row['elapsed_seconds']:.3f}",
                flush=True,
            )

    summary = {
        "variant": args.variant,
        "queries": len(rows),
        "ok_queries": sum(row["status"] == "ok" for row in rows),
        "total_flow_calls": total_flow_calls,
        "total_seconds": sum(row["elapsed_seconds"] for row in rows),
        "mean_seconds": (
            sum(row["elapsed_seconds"] for row in rows) / len(rows)
            if rows
            else None
        ),
        "status_histogram": histogram(row["status"] for row in rows),
    }
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    print("[flow-ablation summary]", rendered)
    if args.summary_out:
        with open(args.summary_out, "w", encoding="utf-8") as output:
            output.write(rendered + "\n")


def histogram(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return counts


if __name__ == "__main__":
    main()
