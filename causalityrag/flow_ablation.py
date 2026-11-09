"""Ranking ablations on a frozen contribution graph, without reader calls."""

from __future__ import annotations

import math

from causalityrag.contribution_graph import contribution_graph_edges
from causalityrag.flow_contract import remaining_contribution_flow


METHODS = ("direct_score", "individual_flow")


def rank_graph_tokens(
    graph_row: dict, eligible_ids: set[str], *, method: str,
    graph_domain: str = "all_frozen_graph_nodes",
) -> dict:
    if method not in METHODS:
        raise ValueError(f"unknown flow ablation: {method}")
    if graph_domain not in {"all_frozen_graph_nodes", "editable_projection"}:
        raise ValueError("unknown graph domain")
    if graph_row.get("status") != "ok":
        raise ValueError("ablation requires a successful frozen graph")
    graph = graph_row["contribution_graph"]
    nodes = [str(value) for value in graph["token_nodes"]]
    node_ids = set(nodes)
    if len(nodes) != len(node_ids) or any(not value for value in nodes):
        raise ValueError("graph token IDs must be nonempty and unique")
    if not eligible_ids <= node_ids:
        raise ValueError("eligible token missing from graph")
    for family in ("source_edges", "interaction_edges", "target_edges"):
        seen = set()
        for edge in graph.get(family, []):
            endpoints = (
                (str(edge["source"]), str(edge["target"]))
                if family == "interaction_edges" else (str(edge["token_id"]),)
            )
            capacity = float(edge["capacity"])
            if not math.isfinite(capacity) or capacity < 0:
                raise ValueError("graph capacities must be finite and nonnegative")
            if not set(endpoints) <= node_ids or endpoints in seen:
                raise ValueError("invalid or duplicate graph edge")
            seen.add(endpoints)
    source, interactions, target = contribution_graph_edges(graph_row)
    domain_nodes = nodes
    if graph_domain == "editable_projection":
        # Match the saved ReFlow runner's eligible-only flow domain, explicitly.
        domain_nodes = [unit_id for unit_id in nodes if unit_id in eligible_ids]
        source = {key: value for key, value in source.items() if key in eligible_ids}
        target = {key: value for key, value in target.items() if key in eligible_ids}
        interactions = {
            pair: value for pair, value in interactions.items()
            if set(pair) <= eligible_ids
        }
    scores = {unit_id: 0.0 for unit_id in sorted(eligible_ids)}
    initial_flow = None
    flow_calls = 0
    if method == "direct_score":
        # Source injection is incoming; count only outgoing token contributions.
        for (left, _), weight in interactions.items():
            if left in scores:
                scores[left] += weight
        for unit_id, weight in target.items():
            if unit_id in scores:
                scores[unit_id] += weight
    else:
        # Noneditable tokens remain transit nodes in the original graph.
        units = [{"unit_id": unit_id} for unit_id in domain_nodes]
        initial_flow = remaining_contribution_flow(
            units, source, interactions, target, removed_ids=frozenset()
        )
        flow_calls = 1
        for unit_id in scores:
            residual = remaining_contribution_flow(
                units, source, interactions, target, removed_ids={unit_id}
            )
            delta = initial_flow - residual
            if delta < -1e-8 * max(1.0, initial_flow):
                raise ValueError("removing a token unexpectedly increased flow")
            scores[unit_id] = max(0.0, delta)
            flow_calls += 1
    return {
        "id": str(graph_row["id"]),
        "method": method,
        "status": "ok" if scores else "no_eligible_tokens",
        "token_scores": scores,
        "ranked_ids": sorted(scores, key=lambda unit_id: (-scores[unit_id], unit_id)),
        "graph_token_count": len(nodes),
        "eligible_token_count": len(scores),
        "initial_flow": initial_flow,
        "max_flow_calls": flow_calls,
        "reader_calls": 0,
        "verification_status": "not_run",
        "graph_domain": graph_domain,
        "flow_domain_token_count": len(domain_nodes),
    }
