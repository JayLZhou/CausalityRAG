"""Evaluate a price frontier for contribution-aware token-gated flow cuts."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.graph_cut import project_cached_units_source_target_graph
from causalityrag.io import record_id
from causalityrag.max_flow import Dinic
from causalityrag.reader import ReaderClient, answers_match
from causalityrag.revision import apply_token_replacements
from causalityrag.token_units import units_from_cache_row


def take_jsonl(path: str, start: int, count: int) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as source:
        for index, line in enumerate(source):
            if index < start:
                continue
            if len(rows) >= count:
                break
            if line.strip():
                rows.append(json.loads(line))
    return rows


def contribution_capacity(weight: float, mean_weight: float, mode: str) -> float:
    normalized = float(weight) / mean_weight
    if mode == "normalized":
        return normalized
    if mode == "unit-plus-normalized":
        return 1.0 + normalized
    raise ValueError(f"unknown edge capacity mode: {mode}")


def solve_price_cut(
    units: list[dict],
    source_edges: dict[str, float],
    interactions: dict[tuple[str, str], float],
    target_edges: dict[str, float],
    *,
    token_price: float,
    edge_capacity_mode: str,
) -> dict:
    unit_by_id = {str(unit["unit_id"]): unit for unit in units}
    if not unit_by_id:
        return {"status": "no_editable_units", "selected_ids": [], "n_selected": 0}

    valid_source = {
        str(unit_id): float(weight)
        for unit_id, weight in source_edges.items()
        if str(unit_id) in unit_by_id and float(weight) > 0
    }
    valid_target = {
        str(unit_id): float(weight)
        for unit_id, weight in target_edges.items()
        if str(unit_id) in unit_by_id and float(weight) > 0
    }
    valid_interactions = {
        (str(left), str(right)): float(weight)
        for (left, right), weight in interactions.items()
        if (
            str(left) in unit_by_id
            and str(right) in unit_by_id
            and str(left) != str(right)
            and float(weight) > 0
        )
    }
    if not valid_source or not valid_target:
        return {
            "status": "no_source_target_support",
            "selected_ids": [],
            "n_selected": 0,
            "source_units": len(valid_source),
            "target_units": len(valid_target),
            "interaction_edges": len(valid_interactions),
        }

    contribution_weights = [
        *valid_source.values(),
        *valid_target.values(),
        *valid_interactions.values(),
    ]
    mean_weight = sum(contribution_weights) / len(contribution_weights)
    if mean_weight <= 1e-12:
        return {"status": "no_positive_contribution", "selected_ids": [], "n_selected": 0}

    dinic = Dinic()
    source = dinic.node()
    target = dinic.node()
    split: dict[str, tuple[int, int]] = {}
    for unit_id in unit_by_id:
        node_in = dinic.node()
        node_out = dinic.node()
        dinic.add_edge(node_in, node_out, token_price)
        split[unit_id] = (node_in, node_out)

    contribution_arcs: list[tuple[int, int, float, str]] = []

    def add_arc(src: int, dst: int, weight: float, label: str) -> None:
        capacity = contribution_capacity(weight, mean_weight, edge_capacity_mode)
        dinic.add_edge(src, dst, capacity)
        contribution_arcs.append((src, dst, capacity, label))

    for unit_id, weight in valid_source.items():
        add_arc(source, split[unit_id][0], weight, f"source:{unit_id}")
    for (left, right), weight in valid_interactions.items():
        add_arc(split[left][1], split[right][0], weight, f"interaction:{left}:{right}")
    for unit_id, weight in valid_target.items():
        add_arc(split[unit_id][1], target, weight, f"target:{unit_id}")

    max_flow = dinic.max_flow(source, target)
    if max_flow <= 1e-9:
        return {
            "status": "no_source_target_path",
            "selected_ids": [],
            "n_selected": 0,
            "source_units": len(valid_source),
            "target_units": len(valid_target),
            "interaction_edges": len(valid_interactions),
        }
    reachable = dinic.reachable(source)
    selected_ids = sorted(
        unit_id
        for unit_id, (node_in, node_out) in split.items()
        if node_in in reachable and node_out not in reachable
    )
    residual_edge_cost = sum(
        capacity
        for src, dst, capacity, _ in contribution_arcs
        if src in reachable and dst not in reachable
    )
    objective = token_price * len(selected_ids) + residual_edge_cost
    return {
        "status": "optimal",
        "selected_ids": selected_ids,
        "n_selected": len(selected_ids),
        "lambda": float(token_price),
        "objective_value": float(objective),
        "token_cost": float(token_price * len(selected_ids)),
        "residual_edge_cost": float(residual_edge_cost),
        "cut_value": float(max_flow),
        "mean_edge_weight": float(mean_weight),
        "edge_capacity_mode": edge_capacity_mode,
        "source_units": len(valid_source),
        "target_units": len(valid_target),
        "interaction_edges": len(valid_interactions),
    }


def remaining_contribution_flow(
    units: list[dict],
    source_edges: dict[str, float],
    interactions: dict[tuple[str, str], float],
    target_edges: dict[str, float],
    *,
    removed_ids: set[str] | frozenset[str],
    edge_capacity_mode: str,
) -> float:
    """Return the residual source-to-answer max flow after closing token gates.

    This is deliberately separate from ``residual_edge_cost`` in a priced
    cut.  The latter is the finite-edge portion of one mixed cut, whereas this
    function computes the quantity used by the paper's flow-to-flip analysis:
    the maximum contribution flow left after a fixed token intervention.
    """

    unit_ids = {str(unit["unit_id"]) for unit in units}
    source = {
        str(unit_id): float(weight)
        for unit_id, weight in source_edges.items()
        if str(unit_id) in unit_ids and float(weight) > 0
    }
    target = {
        str(unit_id): float(weight)
        for unit_id, weight in target_edges.items()
        if str(unit_id) in unit_ids and float(weight) > 0
    }
    internal = {
        (str(left), str(right)): float(weight)
        for (left, right), weight in interactions.items()
        if (
            str(left) in unit_ids
            and str(right) in unit_ids
            and str(left) != str(right)
            and float(weight) > 0
        )
    }
    if not source or not target:
        return 0.0
    weights = [*source.values(), *target.values(), *internal.values()]
    mean_weight = sum(weights) / len(weights)
    if mean_weight <= 1e-12:
        return 0.0

    def capacity(weight: float) -> float:
        return contribution_capacity(weight, mean_weight, edge_capacity_mode)

    total_capacity = sum(capacity(weight) for weight in weights)
    open_gate = total_capacity + 1.0
    removed = {str(unit_id) for unit_id in removed_ids}
    dinic = Dinic()
    source_node = dinic.node()
    target_node = dinic.node()
    split = {
        unit_id: (dinic.node(), dinic.node())
        for unit_id in sorted(unit_ids)
    }
    for unit_id, (node_in, node_out) in split.items():
        dinic.add_edge(node_in, node_out, 0.0 if unit_id in removed else open_gate)
    for unit_id, weight in source.items():
        dinic.add_edge(source_node, split[unit_id][0], capacity(weight))
    for (left, right), weight in internal.items():
        dinic.add_edge(split[left][1], split[right][0], capacity(weight))
    for unit_id, weight in target.items():
        dinic.add_edge(split[unit_id][1], target_node, capacity(weight))
    return float(dinic.max_flow(source_node, target_node))


def sweep_price_cuts(
    units: list[dict],
    source: dict[str, float],
    interactions: dict[tuple[str, str], float],
    target: dict[str, float],
    *,
    points: int,
    dynamic_range: float,
    edge_capacity_mode: str,
) -> dict:
    weights = [
        *[float(value) for value in source.values() if float(value) > 0],
        *[float(value) for value in target.values() if float(value) > 0],
        *[float(value) for value in interactions.values() if float(value) > 0],
    ]
    if not weights:
        return {"status": "no_positive_contribution", "candidates": [], "diagnostics": {}}
    mean_weight = sum(weights) / len(weights)
    capacities = [contribution_capacity(weight, mean_weight, edge_capacity_mode) for weight in weights]
    high = max(sum(capacities), 1e-9)
    low = high / dynamic_range
    log_high = math.log(high)
    log_low = math.log(max(low, 1e-12))
    prices = [
        math.exp(log_high + (log_low - log_high) * index / (points - 1))
        for index in range(points)
    ]

    by_selection: dict[tuple[str, ...], dict] = {}
    for price in prices:
        result = solve_price_cut(
            units,
            source,
            interactions,
            target,
            token_price=price,
            edge_capacity_mode=edge_capacity_mode,
        )
        selection = tuple(result.get("selected_ids", []))
        if result.get("status") != "optimal" or not selection:
            continue
        if selection not in by_selection:
            by_selection[selection] = {
                **result,
                "lambda_max": price,
                "lambda_min": price,
            }
        else:
            by_selection[selection]["lambda_min"] = min(by_selection[selection]["lambda_min"], price)
            by_selection[selection]["lambda_max"] = max(by_selection[selection]["lambda_max"], price)
    candidates = list(by_selection.values())
    candidates.sort(
        key=lambda row: (
            int(row["n_selected"]),
            float(row["residual_edge_cost"]),
            tuple(row["selected_ids"]),
        )
    )
    return {
        "status": "ok",
        "candidates": candidates,
        "diagnostics": {
            "frontier_mode": "geometric_sweep",
            "lambda_points": points,
            "lambda_min": low,
            "lambda_max": high,
            "distinct_candidate_sets": len(candidates),
            "edge_capacity_mode": edge_capacity_mode,
        },
    }


def breakpoint_price_cuts(
    units: list[dict],
    source: dict[str, float],
    interactions: dict[tuple[str, str], float],
    target: dict[str, float],
    *,
    edge_capacity_mode: str,
    tolerance: float = 1e-9,
) -> dict:
    """Enumerate every extreme supported cardinality-flow point.

    The two analytic endpoints are the minimum-cardinality full token cut and
    the empty intervention. For endpoint solutions A and B, their objective
    lines intersect at exactly

        (Phi(B) - Phi(A)) / (|A| - |B|).

    An exact min-cut at this price either exposes a strict lower-hull vertex or
    certifies that A and B are adjacent extreme points. Collinear interior
    points are deliberately omitted because they are not frontier vertices.
    """

    weights = [
        *[float(value) for value in source.values() if float(value) > 0],
        *[float(value) for value in target.values() if float(value) > 0],
        *[float(value) for value in interactions.values() if float(value) > 0],
    ]
    if not weights:
        return {"status": "no_positive_contribution", "candidates": [], "diagnostics": {}}

    initial_flow = remaining_contribution_flow(
        units,
        source,
        interactions,
        target,
        removed_ids=frozenset(),
        edge_capacity_mode=edge_capacity_mode,
    )
    if initial_flow <= tolerance:
        return {"status": "no_source_target_path", "candidates": [], "diagnostics": {}}

    full_cut = _minimum_token_full_cut(
        units,
        source,
        interactions,
        target,
    )
    if full_cut.get("status") != "optimal" or not full_cut.get("selected_ids"):
        return {"status": full_cut.get("status", "no_full_token_cut"), "candidates": [], "diagnostics": {}}

    left = {
        **full_cut,
        "lambda": 0.0,
        "objective_value": 0.0,
        "token_cost": 0.0,
        "residual_edge_cost": 0.0,
        "cut_value": 0.0,
        "edge_capacity_mode": edge_capacity_mode,
    }
    right = {
        "status": "optimal",
        "selected_ids": [],
        "n_selected": 0,
        "lambda": math.inf,
        "objective_value": initial_flow,
        "token_cost": 0.0,
        "residual_edge_cost": initial_flow,
        "cut_value": initial_flow,
        "edge_capacity_mode": edge_capacity_mode,
    }

    cache: dict[float, dict] = {}
    demand_calls = 0

    def demand(price: float) -> dict:
        nonlocal demand_calls
        key = float(price)
        if key not in cache:
            demand_calls += 1
            cache[key] = solve_price_cut(
                units,
                source,
                interactions,
                target,
                token_price=max(0.0, float(price)),
                edge_capacity_mode=edge_capacity_mode,
            )
        return cache[key]

    def cardinality(result: dict) -> int:
        return int(result.get("n_selected", len(result.get("selected_ids", []))))

    def residual(result: dict) -> float:
        return float(result["residual_edge_cost"])

    def point_key(result: dict) -> tuple[int, float]:
        return cardinality(result), residual(result)

    hull: dict[tuple[int, float], dict] = {
        point_key(left): left,
        point_key(right): right,
    }
    pending = [(left, right)]
    certified_intervals = 0
    while pending:
        low_result, high_result = pending.pop()
        cost_gap = cardinality(low_result) - cardinality(high_result)
        residual_gap = residual(high_result) - residual(low_result)
        if cost_gap <= 0 or residual_gap <= tolerance:
            certified_intervals += 1
            continue
        boundary = residual_gap / cost_gap
        if boundary <= 0.0 or not math.isfinite(boundary):
            certified_intervals += 1
            continue
        middle = demand(boundary)
        middle_cost = cardinality(middle)
        middle_residual = residual(middle)
        middle_value = middle_residual + boundary * middle_cost
        endpoint_value = residual(low_result) + boundary * cardinality(low_result)
        strictly_below_chord = middle_value < endpoint_value - tolerance
        strictly_between = (
            cardinality(high_result) < middle_cost < cardinality(low_result)
            and residual(low_result) < middle_residual < residual(high_result)
        )
        if strictly_below_chord and strictly_between:
            key = point_key(middle)
            hull[key] = middle
            pending.append((low_result, middle))
            pending.append((middle, high_result))
        else:
            certified_intervals += 1

    frontier = sorted(
        hull.values(),
        key=lambda row: (-cardinality(row), residual(row), tuple(row.get("selected_ids", []))),
    )
    breakpoints = []
    for current, following in zip(frontier, frontier[1:]):
        breakpoints.append(
            (residual(following) - residual(current))
            / (cardinality(current) - cardinality(following))
        )
    for index, item in enumerate(frontier):
        item["lambda_min"] = 0.0 if index == 0 else breakpoints[index - 1]
        item["lambda_max"] = math.inf if index == len(frontier) - 1 else breakpoints[index]
        item["frontier_extreme"] = True

    candidates = [item for item in frontier if item.get("selected_ids")]
    candidates.sort(
        key=lambda row: (
            int(row["n_selected"]),
            float(row["residual_edge_cost"]),
            tuple(row["selected_ids"]),
        )
    )
    return {
        "status": "ok",
        "candidates": candidates,
        "diagnostics": {
            "frontier_mode": "exact_breakpoint_hull",
            "frontier_complete": True,
            "analytic_endpoints": True,
            "lambda_min": 0.0,
            "lambda_max": None,
            "supported_extreme_points_including_empty": len(frontier),
            "distinct_candidate_sets": len(candidates),
            "demand_calls": demand_calls,
            "total_maxflow_calls": demand_calls + 2,
            "certified_adjacent_intervals": certified_intervals,
            "edge_capacity_mode": edge_capacity_mode,
        },
    }


def _minimum_token_full_cut(
    units: list[dict],
    source_edges: dict[str, float],
    interactions: dict[tuple[str, str], float],
    target_edges: dict[str, float],
) -> dict:
    """Return a minimum-cardinality token set that removes all support flow."""

    unit_ids = {str(unit["unit_id"]) for unit in units}
    valid_source = {str(unit_id) for unit_id, weight in source_edges.items() if str(unit_id) in unit_ids and float(weight) > 0.0}
    valid_target = {str(unit_id) for unit_id, weight in target_edges.items() if str(unit_id) in unit_ids and float(weight) > 0.0}
    valid_interactions = {
        (str(left), str(right))
        for (left, right), weight in interactions.items()
        if str(left) in unit_ids and str(right) in unit_ids and str(left) != str(right) and float(weight) > 0.0
    }
    if not valid_source or not valid_target:
        return {"status": "no_source_target_support", "selected_ids": [], "n_selected": 0}

    uncuttable = float(len(unit_ids) + 1)
    dinic = Dinic()
    source = dinic.node()
    target = dinic.node()
    split = {unit_id: (dinic.node(), dinic.node()) for unit_id in sorted(unit_ids)}
    for node_in, node_out in split.values():
        dinic.add_edge(node_in, node_out, 1.0)
    for unit_id in valid_source:
        dinic.add_edge(source, split[unit_id][0], uncuttable)
    for left, right in valid_interactions:
        dinic.add_edge(split[left][1], split[right][0], uncuttable)
    for unit_id in valid_target:
        dinic.add_edge(split[unit_id][1], target, uncuttable)
    cut_value = dinic.max_flow(source, target)
    reachable = dinic.reachable(source)
    selected_ids = sorted(
        unit_id
        for unit_id, (node_in, node_out) in split.items()
        if node_in in reachable and node_out not in reachable
    )
    return {
        "status": "optimal" if selected_ids else "no_full_token_cut",
        "selected_ids": selected_ids,
        "n_selected": len(selected_ids),
        "minimum_full_cut_size": int(round(cut_value)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--graphs", required=True)
    parser.add_argument("--units-cache", required=True)
    parser.add_argument("--replacement-registry", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--lambda-points", type=int, default=64)
    parser.add_argument("--dynamic-range", type=float, default=1e6)
    parser.add_argument(
        "--frontier-mode",
        choices=("breakpoint", "sweep"),
        default="breakpoint",
    )
    parser.add_argument(
        "--edge-capacity-mode",
        choices=("unit-plus-normalized", "normalized"),
        default="unit-plus-normalized",
    )
    parser.add_argument("--max-verify", type=int, default=1_000_000)
    parser.add_argument(
        "--evaluate-all-frontier",
        action="store_true",
        help=(
            "evaluate every nonempty frontier candidate instead of stopping at "
            "the first reader flip; required for calibration experiments"
        ),
    )
    args = parser.parse_args()

    records = take_jsonl(args.input, args.start, args.n)
    graphs = take_jsonl(args.graphs, args.start, len(records))
    units_rows = take_jsonl(args.units_cache, args.start, len(records))
    registries = take_jsonl(args.replacement_registry, args.start, len(records))
    if len({len(records), len(graphs), len(units_rows), len(registries)}) != 1:
        raise ValueError("input, graph, units, and registry rows are misaligned")

    reader = ReaderClient()
    rows: list[dict] = []
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as output:
        for offset, (record, graph, units_row, registry) in enumerate(
            zip(records, graphs, units_rows, registries)
        ):
            index = args.start + offset
            started = time.monotonic()
            identifier = record_id(record)
            if str(graph.get("id", "")) != identifier:
                raise ValueError(f"unaligned graph at row {index}: {identifier}")
            replacements = {
                str(unit_id): value
                for unit_id, value in registry.get("replacements", {}).items()
                if isinstance(value, dict) and value.get("ok")
            }
            units = [
                unit
                for unit in units_from_cache_row(record, units_row, k=args.k)
                if str(unit["unit_id"]) in replacements
            ]
            by_id = {str(unit["unit_id"]): unit for unit in units}
            source, interactions, target, projection = project_cached_units_source_target_graph(
                graph, units
            )
            source = {unit_id: weight for unit_id, weight in source.items() if unit_id in by_id}
            target = {unit_id: weight for unit_id, weight in target.items() if unit_id in by_id}
            interactions = {
                (left, right): weight
                for (left, right), weight in interactions.items()
                if left in by_id and right in by_id
            }
            if args.frontier_mode == "breakpoint":
                frontier = breakpoint_price_cuts(
                    units,
                    source,
                    interactions,
                    target,
                    edge_capacity_mode=args.edge_capacity_mode,
                )
            else:
                frontier = sweep_price_cuts(
                    units,
                    source,
                    interactions,
                    target,
                    points=args.lambda_points,
                    dynamic_range=args.dynamic_range,
                    edge_capacity_mode=args.edge_capacity_mode,
                )
            initial_support_flow = remaining_contribution_flow(
                units,
                source,
                interactions,
                target,
                removed_ids=frozenset(),
                edge_capacity_mode=args.edge_capacity_mode,
            )
            frontier_candidates = []
            for candidate in frontier.get("candidates", []):
                selected_ids = [
                    str(unit_id)
                    for unit_id in candidate.get("selected_ids", [])
                    if str(unit_id) in by_id
                ]
                residual_support_flow = remaining_contribution_flow(
                    units,
                    source,
                    interactions,
                    target,
                    removed_ids=set(selected_ids),
                    edge_capacity_mode=args.edge_capacity_mode,
                )
                frontier_candidates.append({
                    **candidate,
                    "selected_ids": selected_ids,
                    "remaining_support_flow": residual_support_flow,
                    "remaining_support_fraction": (
                        residual_support_flow / initial_support_flow
                        if initial_support_flow > 1e-12
                        else None
                    ),
                })
            clean_answer = str(graph.get("clean_answer", graph.get("target_answer", "")))
            attempts: list[dict] = []
            verified = None
            for candidate in frontier_candidates[: args.max_verify]:
                selected_ids = [
                    str(unit_id)
                    for unit_id in candidate.get("selected_ids", [])
                    if str(unit_id) in by_id
                ]
                if not selected_ids:
                    continue
                selected = [by_id[unit_id] for unit_id in selected_ids]
                revision = apply_token_replacements(
                    record,
                    selected,
                    {unit_id: replacements[unit_id] for unit_id in selected_ids},
                    k=args.k,
                )
                edited_answer = reader.answer(
                    str(record.get("question", "")),
                    revision["edited_contexts"],
                )
                changed = not answers_match(clean_answer, edited_answer)
                attempt = {
                    **candidate,
                    "selected_ids": selected_ids,
                    "selected_tokens": [str(unit.get("text", "")) for unit in selected],
                    "edited_answer": edited_answer,
                    "answer_changed": changed,
                    "n_edits": revision["n_edits"],
                }
                attempts.append(attempt)
                if changed:
                    if verified is None:
                        verified = attempt
                    if not args.evaluate_all_frontier:
                        break
            row = {
                "index": index,
                "id": identifier,
                "method": "contribution_aware_flow_contract_frontier",
                "frontier_mode": args.frontier_mode,
                "edge_capacity_mode": args.edge_capacity_mode,
                "evaluate_all_frontier": args.evaluate_all_frontier,
                "clean_answer": clean_answer,
                "projection": projection,
                "n_frontier": len(frontier_candidates),
                "frontier_diagnostics": frontier.get("diagnostics", {}),
                "initial_support_flow": initial_support_flow,
                "frontier_candidates": frontier_candidates,
                "attempts": attempts,
                "verified_flip": verified is not None,
                "selected_ids": verified["selected_ids"] if verified else [],
                "selected_tokens": verified["selected_tokens"] if verified else [],
                "n_selected": len(verified["selected_ids"]) if verified else 0,
                "edited_answer": verified["edited_answer"] if verified else clean_answer,
                "reader_calls": len(attempts),
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
            rows.append(row)
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            print(
                f"[contrib-flow-contract] {len(rows)}/{len(records)} "
                f"frontier={row['n_frontier']} calls={row['reader_calls']} "
                f"flip={row['verified_flip']} tokens={row['n_selected']} "
                f"seconds={row['elapsed_seconds']}",
                flush=True,
            )

    flips = [row for row in rows if row["verified_flip"]]
    terminal_sizes = []
    cumulative_sizes = []
    for row in rows:
        sizes = [len(item.get("selected_ids", [])) for item in row["attempts"]]
        cumulative_sizes.append(sum(sizes))
        terminal_sizes.append(row["n_selected"] if row["verified_flip"] else (max(sizes) if sizes else 0))
    summary = {
        "queries": len(rows),
        "flips": len(flips),
        "overall_flip_rate": len(flips) / max(1, len(rows)),
        "success_mean_tokens": (
            statistics.fmean(row["n_selected"] for row in flips) if flips else 0.0
        ),
        "terminal_mean_tokens": (
            statistics.fmean(terminal_sizes) if terminal_sizes else 0.0
        ),
        "cumulative_attempted_tokens_mean": (
            statistics.fmean(cumulative_sizes) if cumulative_sizes else 0.0
        ),
        "mean_reader_calls": (
            statistics.fmean(row["reader_calls"] for row in rows) if rows else 0.0
        ),
        "mean_frontier": (
            statistics.fmean(row["n_frontier"] for row in rows) if rows else 0.0
        ),
        "edge_capacity_mode": args.edge_capacity_mode,
        "frontier_mode": args.frontier_mode,
        "evaluate_all_frontier": args.evaluate_all_frontier,
        "lambda_points": args.lambda_points,
        "out": args.out,
    }
    print("[contrib-flow-contract summary] " + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
