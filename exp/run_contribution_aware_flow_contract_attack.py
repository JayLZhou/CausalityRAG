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
    max_depth: int = 256,
) -> dict:
    weights = [
        *[float(value) for value in source.values() if float(value) > 0],
        *[float(value) for value in target.values() if float(value) > 0],
        *[float(value) for value in interactions.values() if float(value) > 0],
    ]
    if not weights:
        return {"status": "no_positive_contribution", "candidates": [], "diagnostics": {}}
    mean_weight = sum(weights) / len(weights)
    capacities = [
        contribution_capacity(weight, mean_weight, edge_capacity_mode)
        for weight in weights
    ]
    high = max(sum(capacities), 1e-9)
    cache: dict[float, dict] = {}
    demand_calls = 0

    def demand(price: float) -> dict:
        nonlocal demand_calls
        key = round(float(price), 12)
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

    def selection(result: dict) -> tuple[str, ...]:
        return tuple(result.get("selected_ids", []))

    # At very high token price the supported set is normally empty.  Increase
    # the endpoint until it is, because this endpoint anchors the contract
    # frontier from the zero-edit side.
    right_price = high
    right = demand(right_price)
    expansions = 0
    while selection(right) and expansions < 64:
        right_price *= 2.0
        right = demand(right_price)
        expansions += 1

    left_price = max(high / 1e6, 1e-9)
    left = demand(left_price)
    left_endpoint_expansions = 0
    while not selection(left) and left_price < right_price and left_endpoint_expansions < 64:
        left_price *= 10.0
        left = demand(left_price)
        left_endpoint_expansions += 1
    if not selection(left):
        return {
            "status": "no_nonempty_supported_set",
            "candidates": [],
            "diagnostics": {
                "frontier_mode": "breakpoint_recursion",
                "lambda_min": left_price,
                "lambda_max": right_price,
                "endpoint_expansions": expansions,
                "left_endpoint_expansions": left_endpoint_expansions,
                "distinct_candidate_sets": 0,
                "demand_calls": demand_calls,
                "edge_capacity_mode": edge_capacity_mode,
            },
        }
    by_selection: dict[tuple[str, ...], dict] = {}
    seen_intervals: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()

    def cost(result: dict) -> float:
        return float(result.get("n_selected", len(result.get("selected_ids", []))))

    def residual(result: dict) -> float:
        return float(result.get("residual_edge_cost", 0.0))

    def add(result: dict, price: float) -> None:
        chosen = selection(result)
        if not chosen:
            return
        previous = by_selection.get(chosen)
        if previous is None:
            by_selection[chosen] = {
                **result,
                "lambda_min": price,
                "lambda_max": price,
            }
        else:
            previous["lambda_min"] = min(previous["lambda_min"], price)
            previous["lambda_max"] = max(previous["lambda_max"], price)

    add(left, left_price)

    def recurse(low_result: dict, high_result: dict, depth: int) -> None:
        if depth > max_depth:
            return
        low_set = selection(low_result)
        high_set = selection(high_result)
        if low_set == high_set:
            return
        interval_key = (low_set, high_set)
        if interval_key in seen_intervals:
            return
        seen_intervals.add(interval_key)
        cost_gap = cost(low_result) - cost(high_result)
        residual_gap = residual(high_result) - residual(low_result)
        if residual_gap <= tolerance or cost_gap <= tolerance:
            add(high_result, float(high_result.get("lambda", 0.0)))
            return
        boundary = residual_gap / cost_gap
        if boundary <= tolerance or not math.isfinite(boundary):
            return
        probe = boundary * (1.0 + 1e-8) + 1e-12
        middle = demand(probe)
        mid_set = selection(middle)
        if mid_set == low_set or mid_set == high_set:
            add(high_result, boundary)
            return
        add(middle, probe)
        recurse(low_result, middle, depth + 1)
        recurse(middle, high_result, depth + 1)

    recurse(left, right, 0)
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
            "frontier_mode": "breakpoint_recursion",
            "lambda_min": left_price,
            "lambda_max": right_price,
            "endpoint_expansions": expansions,
            "left_endpoint_expansions": left_endpoint_expansions,
            "distinct_candidate_sets": len(candidates),
            "demand_calls": demand_calls,
            "edge_capacity_mode": edge_capacity_mode,
        },
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
            clean_answer = str(graph.get("clean_answer", graph.get("target_answer", "")))
            attempts: list[dict] = []
            verified = None
            for candidate in frontier.get("candidates", [])[: args.max_verify]:
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
                    verified = attempt
                    break
            row = {
                "index": index,
                "id": identifier,
                "method": "contribution_aware_flow_contract_frontier",
                "frontier_mode": args.frontier_mode,
                "edge_capacity_mode": args.edge_capacity_mode,
                "clean_answer": clean_answer,
                "projection": projection,
                "n_frontier": len(frontier.get("candidates", [])),
                "frontier_diagnostics": frontier.get("diagnostics", {}),
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
        "lambda_points": args.lambda_points,
        "out": args.out,
    }
    print("[contrib-flow-contract summary] " + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
