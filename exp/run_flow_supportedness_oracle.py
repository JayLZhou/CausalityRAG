"""Exhaustively test supportedness on small projected contribution graphs.

This is a graph-only oracle.  It never invokes the reader: its purpose is to
measure whether the breakpoint frontier contains a minimum-cardinality set for
a residual-flow threshold on small induced token graphs.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.graph_cut import project_cached_units_source_target_graph  # noqa: E402
from causalityrag.io import load_records, record_id  # noqa: E402
from causalityrag.token_units import units_from_cache_row  # noqa: E402
from run_contribution_aware_flow_contract_attack import (  # noqa: E402
    breakpoint_price_cuts,
    remaining_contribution_flow,
)


def parse_betas(value: str) -> list[float]:
    betas = sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    if not betas or any(beta <= 0.0 or beta > 1.0 for beta in betas):
        raise ValueError("--betas must contain comma-separated values in (0, 1]")
    return betas


def restrict_active_units(
    units: list[dict],
    source: dict[str, float],
    interactions: dict[tuple[str, str], float],
    target: dict[str, float],
    *,
    max_units: int,
) -> list[dict]:
    """Choose a deterministic contribution-only induced pool for the oracle."""

    incident: dict[str, float] = defaultdict(float)
    for unit_id, weight in source.items():
        incident[str(unit_id)] += float(weight)
    for unit_id, weight in target.items():
        incident[str(unit_id)] += float(weight)
    for (left, right), weight in interactions.items():
        incident[str(left)] += float(weight)
        incident[str(right)] += float(weight)
    active = [unit for unit in units if str(unit["unit_id"]) in incident]
    active.sort(key=lambda unit: (-incident[str(unit["unit_id"])], str(unit["unit_id"])))
    return active[:max_units]


def threshold_oracle(
    unit_ids: list[str],
    residuals: dict[tuple[str, ...], float],
    *,
    threshold: float,
) -> tuple[int | None, set[tuple[str, ...]]]:
    """Return every cardinality-minimum subset satisfying a flow threshold."""

    best_size = None
    best: set[tuple[str, ...]] = set()
    for selected, residual in residuals.items():
        if residual > threshold + 1e-12:
            continue
        size = len(selected)
        if best_size is None or size < best_size:
            best_size = size
            best = {selected}
        elif size == best_size:
            best.add(selected)
    return best_size, best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--graphs", required=True)
    parser.add_argument("--units-cache", required=True)
    parser.add_argument("--replacement-registry", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--max-units", type=int, default=12)
    parser.add_argument("--betas", default="0.10,0.20,0.30,0.40,0.50")
    parser.add_argument(
        "--edge-capacity-mode",
        choices=("unit-plus-normalized", "normalized"),
        default="unit-plus-normalized",
    )
    args = parser.parse_args()
    if args.max_units <= 0 or args.max_units > 20:
        raise ValueError("--max-units must be in [1, 20]")

    records = load_records(args.input)[: args.n]
    graphs = load_records(args.graphs)[: args.n]
    unit_rows = {str(row["id"]): row for row in load_records(args.units_cache)}
    registries = {
        str(row["id"]): row.get("replacements", {})
        for row in load_records(args.replacement_registry)
    }
    if len(records) != len(graphs):
        raise ValueError("input and graph rows are misaligned")
    betas = parse_betas(args.betas)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    rows = []
    with open(args.out, "w", encoding="utf-8") as output:
        for index, (record, graph) in enumerate(zip(records, graphs)):
            identifier = record_id(record)
            if identifier != str(graph.get("id", "")):
                raise ValueError(f"unaligned graph at index {index}: {identifier}")
            valid_replacements = {
                str(unit_id)
                for unit_id, replacement in registries.get(identifier, {}).items()
                if isinstance(replacement, dict) and replacement.get("ok")
            }
            units = [
                unit
                for unit in units_from_cache_row(record, unit_rows[identifier], k=args.k)
                if str(unit["unit_id"]) in valid_replacements
            ]
            source, interactions, target, projection = project_cached_units_source_target_graph(graph, units)
            pool = restrict_active_units(
                units, source, interactions, target, max_units=args.max_units
            )
            pool_ids = {str(unit["unit_id"]) for unit in pool}
            source = {unit_id: value for unit_id, value in source.items() if unit_id in pool_ids}
            target = {unit_id: value for unit_id, value in target.items() if unit_id in pool_ids}
            interactions = {
                (left, right): value
                for (left, right), value in interactions.items()
                if left in pool_ids and right in pool_ids
            }
            initial = remaining_contribution_flow(
                pool, source, interactions, target,
                removed_ids=frozenset(),
                edge_capacity_mode=args.edge_capacity_mode,
            )
            if initial <= 1e-12:
                row = {
                    "index": index,
                    "id": identifier,
                    "status": "no_positive_induced_flow",
                    "pool_size": len(pool),
                    "projection": projection,
                }
                rows.append(row)
                output.write(json.dumps(row) + "\n")
                continue
            unit_ids = sorted(pool_ids)
            residuals: dict[tuple[str, ...], float] = {}
            for cardinality in range(len(unit_ids) + 1):
                for combination in itertools.combinations(unit_ids, cardinality):
                    residuals[combination] = remaining_contribution_flow(
                        pool, source, interactions, target,
                        removed_ids=set(combination),
                        edge_capacity_mode=args.edge_capacity_mode,
                    )
            frontier = breakpoint_price_cuts(
                pool, source, interactions, target,
                edge_capacity_mode=args.edge_capacity_mode,
            )
            frontier_residuals = {
                tuple(str(unit_id) for unit_id in candidate.get("selected_ids", [])): remaining_contribution_flow(
                    pool, source, interactions, target,
                    removed_ids=set(candidate.get("selected_ids", [])),
                    edge_capacity_mode=args.edge_capacity_mode,
                )
                for candidate in frontier.get("candidates", [])
            }
            beta_rows = []
            for beta in betas:
                oracle_size, oracle_sets = threshold_oracle(
                    unit_ids, residuals, threshold=beta * initial
                )
                feasible_frontier = [
                    selected for selected, residual in frontier_residuals.items()
                    if residual <= beta * initial + 1e-12
                ]
                frontier_size = min((len(selected) for selected in feasible_frontier), default=None)
                frontier_sets = {
                    selected for selected in feasible_frontier if len(selected) == frontier_size
                } if frontier_size is not None else set()
                beta_rows.append({
                    "beta": beta,
                    "oracle_size": oracle_size,
                    "frontier_size": frontier_size,
                    "oracle_supported": bool(oracle_sets & set(frontier_residuals)),
                    "frontier_matches_oracle_size": (
                        oracle_size is not None and frontier_size == oracle_size
                    ),
                    "oracle_sets": [list(selected) for selected in sorted(oracle_sets)],
                    "frontier_sets": [list(selected) for selected in sorted(frontier_sets)],
                })
            row = {
                "index": index,
                "id": identifier,
                "status": "ok",
                "projection": projection,
                "pool_size": len(pool),
                "initial_support_flow": initial,
                "subset_count": len(residuals),
                "frontier_size": len(frontier_residuals),
                "frontier_diagnostics": frontier.get("diagnostics", {}),
                "betas": beta_rows,
            }
            rows.append(row)
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            print(
                f"[supportedness-oracle] {index + 1}/{len(records)} "
                f"pool={len(pool)} subsets={len(residuals)} frontier={len(frontier_residuals)}",
                flush=True,
            )
    valid = [row for row in rows if row.get("status") == "ok"]
    summary = {
        "queries": len(rows),
        "valid_queries": len(valid),
        "max_units": args.max_units,
        "betas": betas,
        "by_beta": [
            {
                "beta": beta,
                "supported_optimum_coverage": sum(
                    bool(next(item for item in row["betas"] if item["beta"] == beta)["oracle_supported"])
                    for row in valid
                ) / max(1, len(valid)),
                "frontier_matches_oracle_size": sum(
                    bool(next(item for item in row["betas"] if item["beta"] == beta)["frontier_matches_oracle_size"])
                    for row in valid
                ) / max(1, len(valid)),
            }
            for beta in betas
        ],
    }
    with open(args.summary_out, "w", encoding="utf-8") as output:
        json.dump(summary, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
