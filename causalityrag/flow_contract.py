"""Exact breakpoint frontier for contribution-aware token-gated flow cuts."""

from __future__ import annotations

import math

from causalityrag.max_flow import Dinic


def contribution_capacity(weight: float, mean_weight: float, mode: str) -> float:
    normalized = float(weight) / mean_weight
    if mode == "normalized":
        return normalized
    if mode == "unit-plus-normalized":
        return 1.0 + normalized
    raise ValueError(f"unknown edge capacity mode: {mode}")


def require_complete_graph_domain(graph_row: dict, units: list[dict]) -> None:
    graph_ids = {
        str(unit_id)
        for unit_id in graph_row.get("contribution_graph", {}).get(
            "token_nodes",
            [],
        )
    }
    context_ids = {str(unit["unit_id"]) for unit in units}
    if graph_ids != context_ids:
        raise ValueError(
            "contribution graph and token cache differ: "
            f"missing={sorted(context_ids - graph_ids)[:5]}, "
            f"unexpected={sorted(graph_ids - context_ids)[:5]}"
        )


def _positive_graph(
    units: list[dict],
    source_edges: dict[str, float],
    interactions: dict[tuple[str, str], float],
    target_edges: dict[str, float],
) -> tuple[
    set[str],
    dict[str, float],
    dict[tuple[str, str], float],
    dict[str, float],
]:
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
    return unit_ids, source, internal, target


def solve_price_cut(
    units: list[dict],
    source_edges: dict[str, float],
    interactions: dict[tuple[str, str], float],
    target_edges: dict[str, float],
    *,
    token_price: float,
    edge_capacity_mode: str = "unit-plus-normalized",
) -> dict:
    unit_ids, source_weights, internal, target_weights = _positive_graph(
        units,
        source_edges,
        interactions,
        target_edges,
    )
    if not unit_ids:
        return {"status": "no_token_units", "selected_ids": [], "n_selected": 0}
    if not source_weights or not target_weights:
        return {
            "status": "no_source_target_support",
            "selected_ids": [],
            "n_selected": 0,
        }
    weights = [
        *source_weights.values(),
        *internal.values(),
        *target_weights.values(),
    ]
    mean_weight = sum(weights) / len(weights)
    if mean_weight <= 1e-12:
        return {
            "status": "no_positive_contribution",
            "selected_ids": [],
            "n_selected": 0,
        }

    graph = Dinic()
    source = graph.node()
    target = graph.node()
    split = {
        unit_id: (graph.node(), graph.node())
        for unit_id in sorted(unit_ids)
    }
    for node_in, node_out in split.values():
        graph.add_edge(node_in, node_out, float(token_price))

    contribution_arcs = []

    def add(src: int, dst: int, weight: float) -> None:
        capacity = contribution_capacity(
            weight,
            mean_weight,
            edge_capacity_mode,
        )
        graph.add_edge(src, dst, capacity)
        contribution_arcs.append((src, dst, capacity))

    for unit_id, weight in source_weights.items():
        add(source, split[unit_id][0], weight)
    for (left, right), weight in internal.items():
        add(split[left][1], split[right][0], weight)
    for unit_id, weight in target_weights.items():
        add(split[unit_id][1], target, weight)

    cut_value = graph.max_flow(source, target)
    if cut_value <= 1e-9:
        return {
            "status": "no_source_target_path",
            "selected_ids": [],
            "n_selected": 0,
        }
    reachable = graph.reachable(source)
    selected = sorted(
        unit_id
        for unit_id, (node_in, node_out) in split.items()
        if node_in in reachable and node_out not in reachable
    )
    residual = sum(
        capacity
        for src, dst, capacity in contribution_arcs
        if src in reachable and dst not in reachable
    )
    return {
        "status": "optimal",
        "selected_ids": selected,
        "n_selected": len(selected),
        "lambda": float(token_price),
        "objective_value": float(token_price * len(selected) + residual),
        "token_cost": float(token_price * len(selected)),
        "residual_edge_cost": float(residual),
        "cut_value": float(cut_value),
        "mean_edge_weight": float(mean_weight),
        "edge_capacity_mode": edge_capacity_mode,
    }


def remaining_contribution_flow(
    units: list[dict],
    source_edges: dict[str, float],
    interactions: dict[tuple[str, str], float],
    target_edges: dict[str, float],
    *,
    removed_ids: set[str] | frozenset[str],
    edge_capacity_mode: str = "unit-plus-normalized",
) -> float:
    unit_ids, source_weights, internal, target_weights = _positive_graph(
        units,
        source_edges,
        interactions,
        target_edges,
    )
    if not source_weights or not target_weights:
        return 0.0
    weights = [
        *source_weights.values(),
        *internal.values(),
        *target_weights.values(),
    ]
    mean_weight = sum(weights) / len(weights)
    if mean_weight <= 1e-12:
        return 0.0

    def capacity(weight: float) -> float:
        return contribution_capacity(
            weight,
            mean_weight,
            edge_capacity_mode,
        )

    open_gate = sum(capacity(weight) for weight in weights) + 1.0
    removed = {str(unit_id) for unit_id in removed_ids}
    graph = Dinic()
    source = graph.node()
    target = graph.node()
    split = {
        unit_id: (graph.node(), graph.node())
        for unit_id in sorted(unit_ids)
    }
    for unit_id, (node_in, node_out) in split.items():
        graph.add_edge(
            node_in,
            node_out,
            0.0 if unit_id in removed else open_gate,
        )
    for unit_id, weight in source_weights.items():
        graph.add_edge(source, split[unit_id][0], capacity(weight))
    for (left, right), weight in internal.items():
        graph.add_edge(
            split[left][1],
            split[right][0],
            capacity(weight),
        )
    for unit_id, weight in target_weights.items():
        graph.add_edge(split[unit_id][1], target, capacity(weight))
    return float(graph.max_flow(source, target))


def _minimum_token_full_cut(
    units: list[dict],
    source_edges: dict[str, float],
    interactions: dict[tuple[str, str], float],
    target_edges: dict[str, float],
) -> dict:
    unit_ids, source_weights, internal, target_weights = _positive_graph(
        units,
        source_edges,
        interactions,
        target_edges,
    )
    if not source_weights or not target_weights:
        return {
            "status": "no_source_target_support",
            "selected_ids": [],
            "n_selected": 0,
        }
    graph = Dinic()
    source = graph.node()
    target = graph.node()
    split = {
        unit_id: (graph.node(), graph.node())
        for unit_id in sorted(unit_ids)
    }
    fixed = float(len(unit_ids) + 1)
    for node_in, node_out in split.values():
        graph.add_edge(node_in, node_out, 1.0)
    for unit_id in source_weights:
        graph.add_edge(source, split[unit_id][0], fixed)
    for left, right in internal:
        graph.add_edge(split[left][1], split[right][0], fixed)
    for unit_id in target_weights:
        graph.add_edge(split[unit_id][1], target, fixed)
    cut_value = graph.max_flow(source, target)
    reachable = graph.reachable(source)
    selected = sorted(
        unit_id
        for unit_id, (node_in, node_out) in split.items()
        if node_in in reachable and node_out not in reachable
    )
    if cut_value > len(selected) + 1e-9:
        return {
            "status": "no_full_selectable_cut",
            "selected_ids": [],
            "n_selected": 0,
        }
    return {
        "status": "optimal" if selected else "no_full_token_cut",
        "selected_ids": selected,
        "n_selected": len(selected),
        "residual_edge_cost": 0.0,
        "cut_value": 0.0,
    }


def breakpoint_price_cuts(
    units: list[dict],
    source: dict[str, float],
    interactions: dict[tuple[str, str], float],
    target: dict[str, float],
    *,
    edge_capacity_mode: str = "unit-plus-normalized",
    tolerance: float = 1e-9,
) -> dict:
    initial_flow = remaining_contribution_flow(
        units,
        source,
        interactions,
        target,
        removed_ids=frozenset(),
        edge_capacity_mode=edge_capacity_mode,
    )
    if initial_flow <= tolerance:
        return {
            "status": "no_source_target_path",
            "candidates": [],
            "diagnostics": {},
        }
    left = _minimum_token_full_cut(units, source, interactions, target)
    if left.get("status") != "optimal":
        return {
            "status": str(left.get("status", "no_supported_intervention")),
            "candidates": [],
            "diagnostics": {},
        }
    left = {
        **left,
        "lambda": 0.0,
        "objective_value": 0.0,
        "token_cost": 0.0,
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
    cache = {}
    demand_calls = 0

    def demand(price: float) -> dict:
        nonlocal demand_calls
        if price not in cache:
            demand_calls += 1
            cache[price] = solve_price_cut(
                units,
                source,
                interactions,
                target,
                token_price=price,
                edge_capacity_mode=edge_capacity_mode,
            )
        return cache[price]

    def cardinality(row: dict) -> int:
        return int(row.get("n_selected", len(row.get("selected_ids", []))))

    def residual(row: dict) -> float:
        return float(row["residual_edge_cost"])

    def key(row: dict) -> tuple[int, float]:
        return cardinality(row), residual(row)

    hull = {key(left): left, key(right): right}
    pending = [(left, right)]
    certified = 0
    while pending:
        low, high = pending.pop()
        size_gap = cardinality(low) - cardinality(high)
        flow_gap = residual(high) - residual(low)
        if size_gap <= 0 or flow_gap <= tolerance:
            certified += 1
            continue
        price = flow_gap / size_gap
        middle = demand(price)
        if middle.get("status") != "optimal":
            certified += 1
            continue
        middle_value = residual(middle) + price * cardinality(middle)
        endpoint_value = residual(low) + price * cardinality(low)
        strict_chord = middle_value < endpoint_value - tolerance
        strict_order = (
            cardinality(high) < cardinality(middle) < cardinality(low)
            and residual(low) < residual(middle) < residual(high)
        )
        if strict_chord and strict_order:
            hull[key(middle)] = middle
            pending.extend(((low, middle), (middle, high)))
        else:
            certified += 1

    frontier = sorted(
        hull.values(),
        key=lambda row: (
            -cardinality(row),
            residual(row),
            tuple(row.get("selected_ids", [])),
        ),
    )
    breakpoints = [
        (residual(following) - residual(current))
        / (cardinality(current) - cardinality(following))
        for current, following in zip(frontier, frontier[1:])
    ]
    for index, row in enumerate(frontier):
        row["lambda_min"] = 0.0 if index == 0 else breakpoints[index - 1]
        row["lambda_max"] = (
            math.inf if index == len(frontier) - 1 else breakpoints[index]
        )
        row["frontier_extreme"] = True
    candidates = [row for row in frontier if row.get("selected_ids")]
    candidates.sort(
        key=lambda row: (
            cardinality(row),
            residual(row),
            tuple(row["selected_ids"]),
        )
    )
    return {
        "status": "ok",
        "candidates": candidates,
        "diagnostics": {
            "frontier_mode": "exact_breakpoint_hull",
            "frontier_complete": True,
            "supported_extreme_points_including_empty": len(frontier),
            "distinct_candidate_sets": len(candidates),
            "demand_calls": demand_calls,
            "total_maxflow_calls": demand_calls + 2,
            "certified_adjacent_intervals": certified,
            "edge_capacity_mode": edge_capacity_mode,
        },
    }
