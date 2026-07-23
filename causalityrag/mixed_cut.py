"""Parametric mixed token/edge cuts on raw contribution DAGs."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from math import exp, isfinite, log

from .graph_cut import _position_to_unit
from .max_flow import Dinic


@dataclass(frozen=True)
class RawContributionNetwork:
    status: str
    nodes: frozenset[str]
    edges: tuple[tuple[str, str, float], ...]
    roots_by_unit: dict[str, tuple[str, ...]]
    token_nodes_by_unit: dict[str, str]
    selection_unit_by_gate: dict[str, str]
    gate_scope: str
    unit_scores: dict[str, float]
    sink: str
    diagnostics: dict


def build_raw_contribution_network(
    graph_row: dict,
    units: list[dict],
    *,
    minimum_capacity: float = 1e-12,
    capacity_mode: str = "raw",
) -> RawContributionNetwork:
    """Restrict a direct-activation graph to retrieval-to-answer paths."""

    if minimum_capacity < 0:
        raise ValueError("minimum_capacity must be non-negative")
    if capacity_mode not in {"raw", "backward-conserved"}:
        raise ValueError("capacity_mode must be raw or backward-conserved")
    graph = graph_row.get("graph", {})
    semantics = str(graph.get("edge_weight_semantics", ""))
    target_objective = str(graph.get("target_objective", ""))
    if "signed local target-logit contribution" not in semantics:
        return _empty_network(
            "not_raw_direct_activation_graph",
            {
                "edge_weight_semantics": semantics,
                "target_objective": target_objective,
            },
        )

    position_to_unit = _position_to_unit(graph_row, units)
    raw_edges: list[tuple[str, str, float]] = []
    all_nodes: set[str] = set()
    discarded_nonpositive = 0
    edge_capacities, capacity_diagnostics = _projected_edge_capacities(
        graph.get("edges", []),
        mode=capacity_mode,
        minimum_capacity=minimum_capacity,
    )
    for edge, capacity in zip(graph.get("edges", []), edge_capacities):
        if capacity <= minimum_capacity:
            discarded_nonpositive += 1
            continue
        src = str(edge["src"])
        dst = str(edge["dst"])
        raw_edges.append((src, dst, capacity))
        all_nodes.update((src, dst))

    sink = "answer_target"
    roots_by_unit: dict[str, set[str]] = defaultdict(set)
    for edge in graph.get("edges", []):
        if int(edge.get("src_layer", -1)) != 0:
            continue
        unit_id = position_to_unit.get(int(edge.get("src_position", -1)))
        if unit_id is not None:
            roots_by_unit[str(unit_id)].add(str(edge["src"]))

    if sink not in all_nodes:
        return _empty_network(
            "no_positive_answer_sink",
            {
                "edge_weight_semantics": semantics,
                "target_objective": target_objective,
                "positive_edges": len(raw_edges),
            },
        )
    if not roots_by_unit:
        return _empty_network(
            "no_editable_context_roots",
            {
                "edge_weight_semantics": semantics,
                "target_objective": target_objective,
                "positive_edges": len(raw_edges),
            },
        )

    forward: dict[str, set[str]] = defaultdict(set)
    reverse: dict[str, set[str]] = defaultdict(set)
    for src, dst, _ in raw_edges:
        forward[src].add(dst)
        reverse[dst].add(src)
    root_nodes = set().union(*roots_by_unit.values())
    reachable_from_roots = _reachable(root_nodes, forward)
    can_reach_sink = _reachable({sink}, reverse)
    active_nodes = reachable_from_roots & can_reach_sink
    active_edges = tuple(
        (src, dst, capacity)
        for src, dst, capacity in raw_edges
        if src in active_nodes and dst in active_nodes
    )
    active_roots_by_unit = {
        unit_id: tuple(sorted(nodes & active_nodes))
        for unit_id, nodes in roots_by_unit.items()
        if nodes & active_nodes
    }
    if not active_edges or not active_roots_by_unit:
        return _empty_network(
            "no_retrieval_to_answer_path",
            {
                "edge_weight_semantics": semantics,
                "target_objective": target_objective,
                "positive_edges": len(raw_edges),
                "retrieval_roots": len(root_nodes),
            },
        )

    outgoing_by_root: dict[str, float] = defaultdict(float)
    for src, _, capacity in active_edges:
        if src in root_nodes:
            outgoing_by_root[src] += capacity
    unit_scores = {
        unit_id: sum(outgoing_by_root[root] for root in roots)
        for unit_id, roots in active_roots_by_unit.items()
    }
    capacities = [capacity for _, _, capacity in active_edges]
    return RawContributionNetwork(
        status="ok",
        nodes=frozenset(active_nodes),
        edges=active_edges,
        roots_by_unit=active_roots_by_unit,
        token_nodes_by_unit={},
        selection_unit_by_gate={
            unit_id: unit_id for unit_id in active_roots_by_unit
        },
        gate_scope="input_roots",
        unit_scores=unit_scores,
        sink=sink,
        diagnostics={
            "edge_weight_semantics": semantics,
            "target_objective": target_objective,
            "capacity_field": (
                "contribution"
                if capacity_mode == "raw"
                else "backward_conserved_flow"
            ),
            "normalization": capacity_mode,
            "projection": "input_root_gates",
            "input_graph_edges": len(graph.get("edges", [])),
            "positive_edges": len(raw_edges),
            "active_edges": len(active_edges),
            "active_nodes": len(active_nodes),
            "editable_units": len(active_roots_by_unit),
            "editable_model_token_roots": sum(
                len(roots) for roots in active_roots_by_unit.values()
            ),
            "discarded_nonpositive_or_tiny_edges": discarded_nonpositive,
            "minimum_active_capacity": min(capacities),
            "maximum_active_capacity": max(capacities),
            "total_active_capacity": sum(capacities),
            **capacity_diagnostics,
        },
    )


def build_projected_token_contribution_network(
    graph_row: dict,
    units: list[dict],
    *,
    minimum_capacity: float = 1e-12,
    capacity_mode: str = "raw",
) -> RawContributionNetwork:
    """Contract every layered copy of a chunk word into one editable node."""

    if minimum_capacity < 0:
        raise ValueError("minimum_capacity must be non-negative")
    if capacity_mode not in {"raw", "backward-conserved"}:
        raise ValueError("capacity_mode must be raw or backward-conserved")
    graph = graph_row.get("graph", {})
    semantics = str(graph.get("edge_weight_semantics", ""))
    target_objective = str(graph.get("target_objective", ""))
    if "signed local target-logit contribution" not in semantics:
        return _empty_network(
            "not_raw_direct_activation_graph",
            {
                "edge_weight_semantics": semantics,
                "target_objective": target_objective,
                "projection": "layer_copy_token_contraction",
            },
        )

    position_to_unit = _position_to_unit(graph_row, units)
    projected_capacity: dict[tuple[str, str], float] = defaultdict(float)
    source_units: set[str] = set()
    token_node_by_unit: dict[str, str] = {}
    discarded_nonpositive = 0
    contracted_self_edges = 0
    edge_capacities, capacity_diagnostics = _projected_edge_capacities(
        graph.get("edges", []),
        mode=capacity_mode,
        minimum_capacity=minimum_capacity,
    )

    def projected_node(node_id: str, position: int) -> str:
        unit_id = position_to_unit.get(position)
        if unit_id is None or node_id == "answer_target":
            return node_id
        unit_id = str(unit_id)
        token_node_by_unit.setdefault(unit_id, f"token::{unit_id}")
        return token_node_by_unit[unit_id]

    for edge, capacity in zip(graph.get("edges", []), edge_capacities):
        if capacity <= minimum_capacity:
            discarded_nonpositive += 1
            continue
        src_id = str(edge["src"])
        dst_id = str(edge["dst"])
        src_position = int(edge.get("src_position", -1))
        dst_position = int(edge.get("dst_position", -1))
        src = projected_node(src_id, src_position)
        dst = (
            "answer_target"
            if dst_id == "answer_target"
            else projected_node(dst_id, dst_position)
        )
        src_unit = position_to_unit.get(src_position)
        if int(edge.get("src_layer", -1)) == 0 and src_unit is not None:
            source_units.add(str(src_unit))
        if src == dst:
            contracted_self_edges += 1
            continue
        projected_capacity[(src, dst)] += capacity

    projected_edges = [
        (src, dst, capacity)
        for (src, dst), capacity in projected_capacity.items()
        if capacity > minimum_capacity
    ]
    sink = "answer_target"
    roots_by_unit = {
        unit_id: (token_node_by_unit[unit_id],)
        for unit_id in source_units
        if unit_id in token_node_by_unit
    }
    if not roots_by_unit or not any(dst == sink for _, dst, _ in projected_edges):
        return _empty_network(
            "no_projected_retrieval_to_answer_terminals",
            {
                "edge_weight_semantics": semantics,
                "target_objective": target_objective,
                "projection": "layer_copy_token_contraction",
                "projected_edges": len(projected_edges),
            },
        )

    forward: dict[str, set[str]] = defaultdict(set)
    reverse: dict[str, set[str]] = defaultdict(set)
    for src, dst, _ in projected_edges:
        forward[src].add(dst)
        reverse[dst].add(src)
    root_nodes = {nodes[0] for nodes in roots_by_unit.values()}
    active_nodes = _reachable(root_nodes, forward) & _reachable({sink}, reverse)
    active_edges = tuple(
        (src, dst, capacity)
        for src, dst, capacity in projected_edges
        if src in active_nodes and dst in active_nodes
    )
    active_roots = {
        unit_id: nodes
        for unit_id, nodes in roots_by_unit.items()
        if nodes[0] in active_nodes
    }
    active_token_nodes = {
        unit_id: node
        for unit_id, node in token_node_by_unit.items()
        if node in active_nodes
    }
    if not active_edges or not active_roots or not active_token_nodes:
        return _empty_network(
            "no_projected_retrieval_to_answer_path",
            {
                "edge_weight_semantics": semantics,
                "target_objective": target_objective,
                "projection": "layer_copy_token_contraction",
                "projected_edges": len(projected_edges),
            },
        )

    outgoing: dict[str, float] = defaultdict(float)
    for src, _, capacity in active_edges:
        outgoing[src] += capacity
    unit_scores = {
        unit_id: outgoing.get(node, 0.0)
        for unit_id, node in active_token_nodes.items()
    }
    capacities = [capacity for _, _, capacity in active_edges]
    return RawContributionNetwork(
        status="ok",
        nodes=frozenset(active_nodes),
        edges=active_edges,
        roots_by_unit=active_roots,
        token_nodes_by_unit=active_token_nodes,
        selection_unit_by_gate={
            unit_id: unit_id for unit_id in active_token_nodes
        },
        gate_scope="contracted_token_nodes",
        unit_scores=unit_scores,
        sink=sink,
        diagnostics={
            "edge_weight_semantics": semantics,
            "target_objective": target_objective,
            "capacity_field": (
                "contribution"
                if capacity_mode == "raw"
                else "backward_conserved_flow"
            ),
            "normalization": capacity_mode,
            "projection": "layer_copy_token_contraction",
            "input_graph_edges": len(graph.get("edges", [])),
            "projected_parallel_edge_groups": len(projected_edges),
            "active_edges": len(active_edges),
            "active_nodes": len(active_nodes),
            "editable_units": len(active_token_nodes),
            "retrieval_source_units": len(active_roots),
            "contracted_self_edges": contracted_self_edges,
            "discarded_nonpositive_or_tiny_edges": discarded_nonpositive,
            "minimum_active_capacity": min(capacities),
            "maximum_active_capacity": max(capacities),
            "total_active_capacity": sum(capacities),
            **capacity_diagnostics,
        },
    )


def build_layered_copy_contribution_network(
    graph_row: dict,
    units: list[dict],
    *,
    minimum_capacity: float = 1e-12,
    capacity_mode: str = "backward-conserved",
    max_copies_per_unit: int | None = None,
) -> RawContributionNetwork:
    """Keep the layered DAG and expose each editable layer copy as one gate.

    Gates that share ``selection_unit_by_gate`` correspond to the same input
    chunk token.  They remain distinct in the flow network, avoiding the
    layer-skipping paths introduced by contracting all copies into one node.
    """

    if minimum_capacity < 0:
        raise ValueError("minimum_capacity must be non-negative")
    if capacity_mode not in {"raw", "backward-conserved"}:
        raise ValueError("capacity_mode must be raw or backward-conserved")
    if max_copies_per_unit is not None and max_copies_per_unit <= 0:
        raise ValueError("max_copies_per_unit must be positive when provided")
    graph = graph_row.get("graph", {})
    semantics = str(graph.get("edge_weight_semantics", ""))
    target_objective = str(graph.get("target_objective", ""))
    if "signed local target-logit contribution" not in semantics:
        return _empty_network(
            "not_raw_direct_activation_graph",
            {
                "edge_weight_semantics": semantics,
                "target_objective": target_objective,
                "projection": "layered_copy_group_relaxation",
            },
        )

    position_to_unit = _position_to_unit(graph_row, units)
    edge_capacities, capacity_diagnostics = _projected_edge_capacities(
        graph.get("edges", []),
        mode=capacity_mode,
        minimum_capacity=minimum_capacity,
    )
    raw_edges = []
    node_position: dict[str, int] = {}
    node_stage: dict[str, int] = {}
    root_nodes: set[str] = set()
    discarded_nonpositive = 0
    for edge, capacity in zip(graph.get("edges", []), edge_capacities):
        if capacity <= minimum_capacity:
            discarded_nonpositive += 1
            continue
        src = str(edge["src"])
        dst = str(edge["dst"])
        src_position = int(edge.get("src_position", -1))
        dst_position = int(edge.get("dst_position", -1))
        src_stage = int(edge.get("src_layer", -1))
        dst_stage = int(edge.get("dst_layer", -1))
        raw_edges.append((src, dst, capacity))
        node_position[src] = src_position
        node_stage[src] = src_stage
        if dst != "answer_target":
            node_position[dst] = dst_position
            node_stage[dst] = dst_stage
        if src_stage == 0 and src_position in position_to_unit:
            root_nodes.add(src)

    sink = "answer_target"
    if not root_nodes or not any(dst == sink for _, dst, _ in raw_edges):
        return _empty_network(
            "no_layered_retrieval_to_answer_terminals",
            {
                "edge_weight_semantics": semantics,
                "target_objective": target_objective,
                "projection": "layered_copy_group_relaxation",
                "positive_edges": len(raw_edges),
            },
        )

    forward: dict[str, set[str]] = defaultdict(set)
    reverse: dict[str, set[str]] = defaultdict(set)
    for src, dst, _ in raw_edges:
        forward[src].add(dst)
        reverse[dst].add(src)
    active_nodes = _reachable(root_nodes, forward) & _reachable({sink}, reverse)
    active_edges = tuple(
        (src, dst, capacity)
        for src, dst, capacity in raw_edges
        if src in active_nodes and dst in active_nodes
    )
    if not active_edges:
        return _empty_network(
            "no_layered_retrieval_to_answer_path",
            {
                "edge_weight_semantics": semantics,
                "target_objective": target_objective,
                "projection": "layered_copy_group_relaxation",
                "positive_edges": len(raw_edges),
            },
        )

    incoming: dict[str, float] = defaultdict(float)
    outgoing: dict[str, float] = defaultdict(float)
    for src, dst, capacity in active_edges:
        outgoing[src] += capacity
        incoming[dst] += capacity

    candidate_nodes_by_unit: dict[str, set[str]] = defaultdict(set)
    for node in active_nodes:
        position = node_position.get(node, -1)
        unit_id = position_to_unit.get(position)
        if unit_id is not None and node != sink:
            candidate_nodes_by_unit[str(unit_id)].add(node)
    root_nodes_by_unit: dict[str, set[str]] = defaultdict(set)
    for node in root_nodes & active_nodes:
        unit_id = position_to_unit.get(node_position.get(node, -1))
        if unit_id is not None:
            root_nodes_by_unit[str(unit_id)].add(node)

    selected_gate_nodes: set[str] = set()
    for unit_id, candidate_nodes in candidate_nodes_by_unit.items():
        mandatory = root_nodes_by_unit.get(unit_id, set())
        selected_gate_nodes.update(mandatory)
        optional = candidate_nodes - mandatory
        if max_copies_per_unit is None:
            selected_gate_nodes.update(optional)
            continue
        remaining_slots = max(0, max_copies_per_unit - len(mandatory))
        selected_gate_nodes.update(sorted(
            optional,
            key=lambda node: (
                -min(incoming.get(node, 0.0), outgoing.get(node, 0.0)),
                node_stage.get(node, -1),
                node,
            ),
        )[:remaining_slots])

    token_nodes_by_gate: dict[str, str] = {}
    selection_unit_by_gate: dict[str, str] = {}
    gate_by_node: dict[str, str] = {}
    for node in sorted(selected_gate_nodes):
        position = node_position.get(node, -1)
        unit_id = position_to_unit.get(position)
        if unit_id is None or node == sink:
            continue
        gate_id = f"copy::{node}"
        gate_by_node[node] = gate_id
        token_nodes_by_gate[gate_id] = node
        selection_unit_by_gate[gate_id] = str(unit_id)

    roots_by_gate = {
        gate_by_node[node]: (node,)
        for node in root_nodes & active_nodes
        if node in gate_by_node
    }
    if not token_nodes_by_gate or not roots_by_gate:
        return _empty_network(
            "no_editable_layer_copies",
            {
                "edge_weight_semantics": semantics,
                "target_objective": target_objective,
                "projection": "layered_copy_group_relaxation",
            },
        )

    gate_scores = {
        gate_id: outgoing.get(node, 0.0)
        for gate_id, node in token_nodes_by_gate.items()
    }
    copies_per_unit: dict[str, int] = defaultdict(int)
    for unit_id in selection_unit_by_gate.values():
        copies_per_unit[unit_id] += 1
    uncapped_copies_per_unit = {
        unit_id: len(nodes)
        for unit_id, nodes in candidate_nodes_by_unit.items()
    }
    all_candidate_nodes = set().union(*candidate_nodes_by_unit.values())
    excluded_gate_nodes = all_candidate_nodes - selected_gate_nodes
    uncapped_copy_throughput = sum(
        min(incoming.get(node, 0.0), outgoing.get(node, 0.0))
        for node in all_candidate_nodes
    )
    excluded_copy_throughput = sum(
        min(incoming.get(node, 0.0), outgoing.get(node, 0.0))
        for node in excluded_gate_nodes
    )
    capacities = [capacity for _, _, capacity in active_edges]
    return RawContributionNetwork(
        status="ok",
        nodes=frozenset(active_nodes),
        edges=active_edges,
        roots_by_unit=roots_by_gate,
        token_nodes_by_unit=token_nodes_by_gate,
        selection_unit_by_gate=selection_unit_by_gate,
        gate_scope="contracted_token_nodes",
        unit_scores=gate_scores,
        sink=sink,
        diagnostics={
            "edge_weight_semantics": semantics,
            "target_objective": target_objective,
            "capacity_field": (
                "contribution"
                if capacity_mode == "raw"
                else "backward_conserved_flow"
            ),
            "normalization": capacity_mode,
            "projection": "layered_copy_group_relaxation",
            "input_graph_edges": len(graph.get("edges", [])),
            "active_edges": len(active_edges),
            "active_nodes": len(active_nodes),
            "editable_layer_copies": len(token_nodes_by_gate),
            "editable_token_groups": len(copies_per_unit),
            "maximum_active_group_rank": max(copies_per_unit.values()),
            "mean_active_group_rank": (
                sum(copies_per_unit.values()) / len(copies_per_unit)
            ),
            "max_copies_per_unit": max_copies_per_unit,
            "uncapped_editable_layer_copies": sum(
                uncapped_copies_per_unit.values()
            ),
            "excluded_layer_copies": (
                sum(uncapped_copies_per_unit.values())
                - len(token_nodes_by_gate)
            ),
            "uncapped_copy_throughput": uncapped_copy_throughput,
            "excluded_copy_throughput": excluded_copy_throughput,
            "excluded_copy_throughput_fraction": (
                excluded_copy_throughput / uncapped_copy_throughput
                if uncapped_copy_throughput > 0
                else 0.0
            ),
            "copies_per_unit": dict(copies_per_unit),
            "discarded_nonpositive_or_tiny_edges": discarded_nonpositive,
            "minimum_active_capacity": min(capacities),
            "maximum_active_capacity": max(capacities),
            "total_active_capacity": sum(capacities),
            **capacity_diagnostics,
        },
    )


def augment_with_unary_support(
    network: RawContributionNetwork,
    unary_scores: dict[str, float],
    *,
    graph_weight: float,
) -> RawContributionNetwork:
    """Mix normalized graph flow with independent unary token-support paths."""

    if not 0 <= graph_weight <= 1 or not isfinite(graph_weight):
        raise ValueError("graph_weight must be finite and in [0, 1]")
    scores = {}
    for unit_id, value in unary_scores.items():
        score = float(value)
        if not isfinite(score) or score < 0:
            raise ValueError("unary scores must be finite and non-negative")
        if score > 0:
            scores[str(unit_id)] = score
    unary_total = sum(scores.values())

    graph_flow = (
        remaining_support_flow(network, frozenset())
        if network.status == "ok"
        else 0.0
    )
    graph_available = graph_flow > 0
    unary_available = unary_total > 0
    if not graph_available and not unary_available:
        return _empty_network(
            "no_graph_or_unary_support",
            {
                "parent_network_status": network.status,
                "graph_weight_requested": graph_weight,
            },
        )

    if graph_available and unary_available:
        effective_graph_weight = graph_weight
        effective_unary_weight = 1.0 - graph_weight
    elif graph_available:
        effective_graph_weight = 1.0
        effective_unary_weight = 0.0
    else:
        effective_graph_weight = 0.0
        effective_unary_weight = 1.0
    graph_scale = (
        effective_graph_weight / graph_flow if graph_available else 0.0
    )
    unary_scale = (
        effective_unary_weight / unary_total if unary_available else 0.0
    )

    identity_token_gates = (
        graph_available
        and network.gate_scope == "contracted_token_nodes"
        and all(
            network.selection_unit_by_gate.get(gate_id) == gate_id
            for gate_id in network.token_nodes_by_unit
        )
    )
    if identity_token_gates or not graph_available:
        return _augment_identity_token_support(
            network,
            scores,
            graph_available=graph_available,
            graph_scale=graph_scale,
            unary_scale=unary_scale,
            graph_weight=graph_weight,
            effective_graph_weight=effective_graph_weight,
            effective_unary_weight=effective_unary_weight,
            graph_flow=graph_flow,
            unary_total=unary_total,
        )

    nodes = set(network.nodes) if graph_available else {"answer_target"}
    edges = [
        (src, dst, capacity * graph_scale)
        for src, dst, capacity in network.edges
        if graph_scale > 0 and capacity > 0
    ]
    roots_by_gate = dict(network.roots_by_unit) if graph_available else {}
    token_nodes_by_gate = (
        dict(network.token_nodes_by_unit) if graph_available else {}
    )
    selection_unit_by_gate = (
        dict(network.selection_unit_by_gate) if graph_available else {}
    )
    # Root-gate scores are used only for the matched unary baseline.  Keeping
    # graph-copy scores at zero makes that baseline exactly the supplied unary
    # ranking instead of a blend chosen by the proposed method.
    gate_scores = {gate_id: 0.0 for gate_id in token_nodes_by_gate}
    for index, (unit_id, score) in enumerate(sorted(scores.items())):
        capacity = score * unary_scale
        if capacity <= 0:
            continue
        node = f"unary::{index}"
        gate_id = f"unary-gate::{index}"
        nodes.add(node)
        edges.append((node, "answer_target", capacity))
        roots_by_gate[gate_id] = (node,)
        token_nodes_by_gate[gate_id] = node
        selection_unit_by_gate[gate_id] = unit_id
        gate_scores[gate_id] = capacity

    copies_per_unit: dict[str, int] = defaultdict(int)
    for unit_id in selection_unit_by_gate.values():
        copies_per_unit[unit_id] += 1
    capacities = [capacity for _, _, capacity in edges]
    parent_diagnostics = network.diagnostics if graph_available else {}
    return RawContributionNetwork(
        status="ok",
        nodes=frozenset(nodes),
        edges=tuple(edges),
        roots_by_unit=roots_by_gate,
        token_nodes_by_unit=token_nodes_by_gate,
        selection_unit_by_gate=selection_unit_by_gate,
        gate_scope="contracted_token_nodes",
        unit_scores=gate_scores,
        sink="answer_target",
        diagnostics={
            **parent_diagnostics,
            "projection": "layered_copy_group_with_unary_support",
            "parent_network_status": network.status,
            "graph_weight_requested": graph_weight,
            "effective_graph_weight": effective_graph_weight,
            "effective_unary_weight": effective_unary_weight,
            "graph_flow_before_normalization": graph_flow,
            "unary_score_before_normalization": unary_total,
            "unary_support_tokens": len(scores),
            "active_edges": len(edges),
            "active_nodes": len(nodes),
            "editable_layer_copies": len(token_nodes_by_gate),
            "editable_token_groups": len(copies_per_unit),
            "maximum_active_group_rank": max(copies_per_unit.values()),
            "mean_active_group_rank": (
                sum(copies_per_unit.values()) / len(copies_per_unit)
            ),
            "copies_per_unit": dict(copies_per_unit),
            "minimum_active_capacity": min(capacities),
            "maximum_active_capacity": max(capacities),
            "total_active_capacity": sum(capacities),
        },
    )


def _augment_identity_token_support(
    network: RawContributionNetwork,
    scores: dict[str, float],
    *,
    graph_available: bool,
    graph_scale: float,
    unary_scale: float,
    graph_weight: float,
    effective_graph_weight: float,
    effective_unary_weight: float,
    graph_flow: float,
    unary_total: float,
) -> RawContributionNetwork:
    """Attach unary support to the same gate used by a projected token."""

    sink = network.sink if graph_available else "answer_target"
    nodes = set(network.nodes) if graph_available else {sink}
    edges = [
        (src, dst, capacity * graph_scale)
        for src, dst, capacity in network.edges
        if graph_scale > 0 and capacity > 0
    ]
    roots = dict(network.roots_by_unit) if graph_available else {}
    token_nodes = dict(network.token_nodes_by_unit) if graph_available else {}
    selection = (
        dict(network.selection_unit_by_gate) if graph_available else {}
    )
    gate_scores = {gate_id: 0.0 for gate_id in token_nodes}
    for index, (unit_id, score) in enumerate(sorted(scores.items())):
        capacity = score * unary_scale
        if capacity <= 0:
            continue
        node = token_nodes.get(unit_id, f"unary-token::{index}")
        nodes.add(node)
        token_nodes[unit_id] = node
        selection[unit_id] = unit_id
        roots[unit_id] = (node,)
        edges.append((node, sink, capacity))
        gate_scores[unit_id] = capacity

    capacities = [capacity for _, _, capacity in edges]
    parent_diagnostics = network.diagnostics if graph_available else {}
    return RawContributionNetwork(
        status="ok",
        nodes=frozenset(nodes),
        edges=tuple(edges),
        roots_by_unit=roots,
        token_nodes_by_unit=token_nodes,
        selection_unit_by_gate=selection,
        gate_scope="contracted_token_nodes",
        unit_scores=gate_scores,
        sink=sink,
        diagnostics={
            **parent_diagnostics,
            "projection": "projected_token_graph_with_shared_unary_gate",
            "parent_network_status": network.status,
            "graph_weight_requested": graph_weight,
            "effective_graph_weight": effective_graph_weight,
            "effective_unary_weight": effective_unary_weight,
            "graph_flow_before_normalization": graph_flow,
            "unary_score_before_normalization": unary_total,
            "unary_support_tokens": len(scores),
            "active_edges": len(edges),
            "active_nodes": len(nodes),
            "editable_token_groups": len(token_nodes),
            "maximum_active_group_rank": 1,
            "mean_active_group_rank": 1.0 if token_nodes else 0.0,
            "copies_per_unit": {
                unit_id: 1 for unit_id in token_nodes
            },
            "minimum_active_capacity": min(capacities),
            "maximum_active_capacity": max(capacities),
            "total_active_capacity": sum(capacities),
        },
    )


def restrict_group_editable_units(
    network: RawContributionNetwork,
    allowed_unit_ids: set[str] | frozenset[str],
) -> RawContributionNetwork:
    """Keep invalid token copies in the DAG while making them uneditable.

    The source topology and contribution capacities are unchanged.  Only gates
    whose shared input-token label is present in ``allowed_unit_ids`` may be
    purchased by the grouped interdiction solver.  This is the graph-side
    contract used by an answer-blind replacement registry.
    """

    if network.status != "ok":
        return network
    allowed = {str(unit_id) for unit_id in allowed_unit_ids}
    selection = {
        gate_id: unit_id
        for gate_id, unit_id in network.selection_unit_by_gate.items()
        if unit_id in allowed
    }
    diagnostics = {
        **network.diagnostics,
        "replacement_registry_applied": True,
        "registry_allowed_token_groups": len(set(selection.values())),
        "registry_excluded_token_groups": len(
            set(network.selection_unit_by_gate.values()) - allowed
        ),
        "registry_editable_layer_copies": len(selection),
    }
    if not selection:
        return replace(
            network,
            status="no_registry_editable_units",
            selection_unit_by_gate={},
            diagnostics=diagnostics,
        )
    return replace(
        network,
        selection_unit_by_gate=selection,
        diagnostics=diagnostics,
    )


def _projected_edge_capacities(
    edges: list[dict],
    *,
    mode: str,
    minimum_capacity: float,
) -> tuple[list[float], dict]:
    raw = []
    for edge in edges:
        capacity = float(edge.get("contribution", 0.0))
        if not isfinite(capacity):
            raise ValueError("contribution graph contains a non-finite capacity")
        raw.append(capacity)
    if mode == "raw":
        return raw, {}

    incoming: dict[str, list[int]] = defaultdict(list)
    node_stage: dict[str, int] = {}
    positive_indices = []
    for index, (edge, capacity) in enumerate(zip(edges, raw)):
        if capacity <= minimum_capacity:
            continue
        src = str(edge["src"])
        dst = str(edge["dst"])
        src_stage = int(edge.get("src_layer", -1))
        dst_stage = int(edge.get("dst_layer", -1))
        if src_stage >= dst_stage:
            raise ValueError("backward conservation requires a directed acyclic graph")
        if src in node_stage and node_stage[src] != src_stage:
            raise ValueError(f"inconsistent stage for contribution node {src}")
        if dst in node_stage and node_stage[dst] != dst_stage:
            raise ValueError(f"inconsistent stage for contribution node {dst}")
        node_stage[src] = src_stage
        node_stage[dst] = dst_stage
        incoming[dst].append(index)
        positive_indices.append(index)

    sink = "answer_target"
    relevance: dict[str, float] = defaultdict(float)
    relevance[sink] = 1.0
    conserved = [0.0] * len(edges)
    terminal_mass: dict[str, float] = {}
    for node in sorted(node_stage, key=lambda item: (node_stage[item], item), reverse=True):
        mass = relevance.get(node, 0.0)
        if mass <= 0:
            continue
        predecessors = incoming.get(node, [])
        total = sum(raw[index] for index in predecessors)
        if total <= minimum_capacity:
            terminal_mass[node] = terminal_mass.get(node, 0.0) + mass
            continue
        for index in predecessors:
            flow = mass * raw[index] / total
            conserved[index] = flow
            relevance[str(edges[index]["src"])] += flow

    incoming_flow: dict[str, float] = defaultdict(float)
    outgoing_flow: dict[str, float] = defaultdict(float)
    for index in positive_indices:
        flow = conserved[index]
        if flow <= 0:
            continue
        outgoing_flow[str(edges[index]["src"])] += flow
        incoming_flow[str(edges[index]["dst"])] += flow
    internal_nodes = set(incoming_flow) & set(outgoing_flow)
    conservation_error = max(
        (
            abs(incoming_flow[node] - outgoing_flow[node])
            for node in internal_nodes
        ),
        default=0.0,
    )
    input_terminal_mass = sum(
        mass for node, mass in terminal_mass.items() if node_stage.get(node) == 0
    )
    return conserved, {
        "conservation_target_mass": 1.0,
        "conservation_sink_inflow": incoming_flow.get(sink, 0.0),
        "conservation_terminal_mass": sum(terminal_mass.values()),
        "conservation_input_terminal_mass": input_terminal_mass,
        "conservation_unexplained_internal_mass": (
            sum(terminal_mass.values()) - input_terminal_mass
        ),
        "conservation_terminal_nodes": len(terminal_mass),
        "conservation_positive_raw_edges": len(positive_indices),
        "conservation_positive_flow_edges": sum(flow > 0 for flow in conserved),
        "maximum_internal_conservation_error": conservation_error,
    }


def solve_mixed_cut(
    network: RawContributionNetwork,
    token_cost: float,
) -> dict:
    """Exactly solve lambda * edited tokens + residual edge cut capacity."""

    if network.status != "ok":
        return _empty_cut(network.status, token_cost)
    if token_cost < 0 or not isfinite(token_cost):
        raise ValueError("token_cost must be finite and non-negative")
    state = _run_flow(
        network,
        {
            gate_id: token_cost for gate_id in _purchasable_gate_ids(network)
        },
    )
    selected = sorted(
        unit_id
        for unit_id, (gate_in, gate_out) in state["gates"].items()
        if gate_in in state["reachable"] and gate_out not in state["reachable"]
    )
    residual_cut = sum(
        capacity
        for src, dst, capacity in network.edges
        if state["cut_src_ids"][src] in state["reachable"]
        and state["node_ids"][dst] not in state["reachable"]
    )
    objective = token_cost * len(selected) + residual_cut
    return {
        "status": "optimal",
        "lambda": token_cost,
        "selected_ids": selected,
        "n_selected": len(selected),
        "residual_cut_capacity": residual_cut,
        "objective_value": objective,
        "max_flow_value": state["flow"],
        "cut_identity_error": abs(objective - state["flow"]),
        "solver": "float_dinic_mixed_node_edge_cut",
    }


def solve_weighted_mixed_cut(
    network: RawContributionNetwork,
    gate_costs: dict[str, float],
) -> dict:
    """Exactly solve a mixed cut with an independent cost for each gate."""

    if network.status != "ok":
        return {
            **_empty_cut(network.status, 0.0),
            "gate_costs": {},
            "selected_gate_cost": 0.0,
        }
    editable = _editable_unit_ids(network)
    if set(gate_costs) != editable:
        missing = sorted(editable - set(gate_costs))
        extra = sorted(set(gate_costs) - editable)
        raise ValueError(
            f"gate_costs must match editable gates; missing={missing}, extra={extra}"
        )
    costs = {str(gate_id): float(cost) for gate_id, cost in gate_costs.items()}
    if any(cost < 0 or not isfinite(cost) for cost in costs.values()):
        raise ValueError("gate costs must be finite and non-negative")

    state = _run_flow(network, costs)
    selected = sorted(
        gate_id
        for gate_id, (gate_in, gate_out) in state["gates"].items()
        if gate_in in state["reachable"] and gate_out not in state["reachable"]
    )
    residual_cut = sum(
        capacity
        for src, dst, capacity in network.edges
        if state["cut_src_ids"][src] in state["reachable"]
        and state["node_ids"][dst] not in state["reachable"]
    )
    selected_gate_cost = sum(costs[gate_id] for gate_id in selected)
    objective = selected_gate_cost + residual_cut
    return {
        "status": "optimal",
        "selected_ids": selected,
        "n_selected": len(selected),
        "gate_costs": costs,
        "selected_gate_cost": selected_gate_cost,
        "residual_cut_capacity": residual_cut,
        "objective_value": objective,
        "max_flow_value": state["flow"],
        "cut_identity_error": abs(objective - state["flow"]),
        "solver": "float_dinic_weighted_mixed_node_edge_cut",
    }


def remaining_support_flow(
    network: RawContributionNetwork,
    removed_ids: set[str] | frozenset[str],
) -> float:
    """Return max contribution flow after removing a fixed token set."""

    if network.status != "ok":
        return 0.0
    removed = {str(unit_id) for unit_id in removed_ids}
    total_capacity = sum(capacity for _, _, capacity in network.edges)
    fixed_open = total_capacity + 1.0
    state = _run_flow(
        network,
        {
            unit_id: 0.0 if unit_id in removed else fixed_open
            for unit_id in _editable_unit_ids(network)
        },
    )
    return float(state["flow"])


def sweep_mixed_cuts(
    network: RawContributionNetwork,
    *,
    points: int = 64,
    dynamic_range: float = 1e6,
) -> dict:
    """Enumerate distinct supported mixed cuts over a geometric lambda sweep."""

    if points < 2:
        raise ValueError("points must be at least two")
    if dynamic_range <= 1:
        raise ValueError("dynamic_range must be greater than one")
    if network.status != "ok":
        return {
            "status": network.status,
            "initial_flow": 0.0,
            "unary_order": [],
            "candidates": [],
            "diagnostics": {},
        }

    initial_flow = remaining_support_flow(network, frozenset())
    if initial_flow <= 0:
        return {
            "status": "zero_initial_flow",
            "initial_flow": initial_flow,
            "unary_order": [],
            "candidates": [],
            "diagnostics": {},
        }
    high = initial_flow * 2.0
    low = max(high / dynamic_range, 1e-12)
    log_high = log(high)
    log_low = log(low)
    lambdas = [
        exp(log_high + (log_low - log_high) * index / (points - 1))
        for index in range(points)
    ]

    by_selection: dict[tuple[str, ...], dict] = {}
    cardinalities = []
    for token_cost in lambdas:
        result = solve_mixed_cut(network, token_cost)
        selection = tuple(result["selected_ids"])
        cardinalities.append(result["n_selected"])
        if selection not in by_selection:
            by_selection[selection] = {
                **result,
                "lambda_max": token_cost,
                "lambda_min": token_cost,
            }
        else:
            by_selection[selection]["lambda_min"] = token_cost

    editable = _purchasable_gate_ids(network)
    unary_order = sorted(
        editable,
        key=lambda unit_id: (-network.unit_scores[unit_id], unit_id),
    )
    candidates = []
    for selection, result in by_selection.items():
        selected = set(selection)
        size = len(selected)
        residual_flow = remaining_support_flow(network, selected)
        unary_ids = unary_order[:size]
        unary_flow = remaining_support_flow(network, set(unary_ids))
        union = selected | set(unary_ids)
        candidates.append({
            **result,
            "remaining_support_flow": residual_flow,
            "remaining_support_fraction": residual_flow / initial_flow,
            "unary_matched_ids": unary_ids,
            "unary_remaining_support_flow": unary_flow,
            "unary_remaining_support_fraction": unary_flow / initial_flow,
            "differs_from_unary": list(selection) != sorted(unary_ids),
            "flow_improvement_over_unary": unary_flow - residual_flow,
            "jaccard_distance_from_unary": (
                0.0 if not union else 1.0 - len(selected & set(unary_ids)) / len(union)
            ),
        })
    candidates.sort(
        key=lambda row: (
            row["n_selected"],
            row["remaining_support_flow"],
            row["selected_ids"],
        )
    )

    return {
        "status": "ok",
        "initial_flow": initial_flow,
        "unary_order": unary_order,
        "unary_scores": network.unit_scores,
        "candidates": candidates,
        "diagnostics": {
            "lambda_points": points,
            "lambda_max": high,
            "lambda_min": low,
            "distinct_candidate_sets": len(candidates),
            "distinct_nonempty_candidate_sets": sum(
                candidate["n_selected"] > 0 for candidate in candidates
            ),
            "cardinality_monotone_over_descending_lambda": all(
                left <= right
                for left, right in zip(cardinalities, cardinalities[1:])
            ),
            "candidate_cardinalities": sorted({
                candidate["n_selected"] for candidate in candidates
            }),
        },
    }


def solve_fixed_mixed_cut(
    network: RawContributionNetwork,
    *,
    token_cost: float,
) -> dict:
    """Return the single supported cut for a fixed token price."""

    if network.status != "ok":
        return {
            "status": network.status,
            "initial_flow": 0.0,
            "unary_order": [],
            "unary_scores": {},
            "candidates": [],
            "diagnostics": {"solver": "fixed_lambda_mincut"},
        }
    initial_flow = remaining_support_flow(network, frozenset())
    if initial_flow <= 0:
        return {
            "status": "zero_initial_flow",
            "initial_flow": initial_flow,
            "unary_order": [],
            "unary_scores": network.unit_scores,
            "candidates": [],
            "diagnostics": {"solver": "fixed_lambda_mincut"},
        }

    result = solve_mixed_cut(network, token_cost)
    selected = set(result["selected_ids"])
    editable = _purchasable_gate_ids(network)
    unary_order = sorted(
        editable,
        key=lambda unit_id: (-network.unit_scores[unit_id], unit_id),
    )
    unary_ids = unary_order[:len(selected)]
    residual_flow = remaining_support_flow(network, selected)
    unary_flow = remaining_support_flow(network, set(unary_ids))
    union = selected | set(unary_ids)
    candidate = {
        **result,
        "lambda_min": token_cost,
        "lambda_max": token_cost,
        "remaining_support_flow": residual_flow,
        "remaining_support_fraction": residual_flow / initial_flow,
        "unary_matched_ids": unary_ids,
        "unary_remaining_support_flow": unary_flow,
        "unary_remaining_support_fraction": unary_flow / initial_flow,
        "differs_from_unary": result["selected_ids"] != sorted(unary_ids),
        "flow_improvement_over_unary": unary_flow - residual_flow,
        "jaccard_distance_from_unary": (
            0.0
            if not union
            else 1.0 - len(selected & set(unary_ids)) / len(union)
        ),
    }
    return {
        "status": "ok",
        "initial_flow": initial_flow,
        "unary_order": unary_order,
        "unary_scores": network.unit_scores,
        "candidates": [candidate],
        "diagnostics": {
            "solver": "fixed_lambda_mincut",
            "lambda": token_cost,
            "mincut_calls": 1,
            "distinct_candidate_sets": 1,
            "distinct_nonempty_candidate_sets": int(bool(selected)),
            "candidate_cardinalities": [len(selected)],
            "cardinality_monotone_over_descending_lambda": True,
        },
    }


def search_mixed_cut_threshold(
    network: RawContributionNetwork,
    *,
    beta: float,
    iterations: int = 32,
) -> dict:
    """Search the largest token price whose supported cut meets a flow bound.

    Every fixed-price problem is solved exactly by ``solve_mixed_cut``. The
    search is only over supported Lagrangian cuts; it is not an exact solver
    for cardinality-constrained flow interdiction.
    """

    if not 0 < beta < 1:
        raise ValueError("beta must be strictly between zero and one")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if network.status != "ok":
        return {
            "status": network.status,
            "initial_flow": 0.0,
            "unary_order": [],
            "candidates": [],
            "strict_candidate": None,
            "diagnostics": {},
        }

    initial_flow = remaining_support_flow(network, frozenset())
    if initial_flow <= 0:
        return {
            "status": "zero_initial_flow",
            "initial_flow": initial_flow,
            "unary_order": [],
            "candidates": [],
            "strict_candidate": None,
            "diagnostics": {},
        }

    threshold = beta * initial_flow
    evaluated: list[dict] = []

    def evaluate(token_cost: float) -> dict:
        result = solve_mixed_cut(network, token_cost)
        evaluated.append(result)
        return result

    def feasible(result: dict) -> bool:
        return float(result["residual_cut_capacity"]) <= threshold + 1e-12

    low = 0.0
    low_result = evaluate(low)
    if not feasible(low_result):
        return {
            "status": "no_supported_cut_meets_threshold",
            "initial_flow": initial_flow,
            "unary_order": [],
            "candidates": [],
            "strict_candidate": None,
            "diagnostics": {
                "solver": "binary_lambda_supported_mincut",
                "beta": beta,
                "strict_flow_threshold": threshold,
                "mincut_calls": len(evaluated),
                "reason": "minimum-residual supported cut exceeds threshold",
            },
        }

    high = initial_flow * 2.0
    high_result = evaluate(high)
    bracket_expansions = 0
    while feasible(high_result) and bracket_expansions < 64:
        high *= 2.0
        high_result = evaluate(high)
        bracket_expansions += 1
    if feasible(high_result):
        raise RuntimeError("failed to bracket an infeasible token price")

    for _ in range(iterations):
        middle = (low + high) / 2.0
        middle_result = evaluate(middle)
        if feasible(middle_result):
            low = middle
            low_result = middle_result
        else:
            high = middle
            high_result = middle_result

    by_selection: dict[tuple[str, ...], dict] = {}
    for result in evaluated:
        selection = tuple(result["selected_ids"])
        token_cost = float(result["lambda"])
        previous = by_selection.get(selection)
        if previous is None:
            by_selection[selection] = {
                **result,
                "lambda_min": token_cost,
                "lambda_max": token_cost,
            }
        else:
            previous["lambda_min"] = min(previous["lambda_min"], token_cost)
            previous["lambda_max"] = max(previous["lambda_max"], token_cost)

    editable = _purchasable_gate_ids(network)
    unary_order = sorted(
        editable,
        key=lambda unit_id: (-network.unit_scores[unit_id], unit_id),
    )
    candidates = []
    for selection, result in by_selection.items():
        selected = set(selection)
        size = len(selected)
        residual_flow = remaining_support_flow(network, selected)
        unary_ids = unary_order[:size]
        unary_flow = remaining_support_flow(network, set(unary_ids))
        union = selected | set(unary_ids)
        candidates.append({
            **result,
            "remaining_support_flow": residual_flow,
            "remaining_support_fraction": residual_flow / initial_flow,
            "strict_feasible": residual_flow <= threshold + 1e-12,
            "unary_matched_ids": unary_ids,
            "unary_remaining_support_flow": unary_flow,
            "unary_remaining_support_fraction": unary_flow / initial_flow,
            "differs_from_unary": list(selection) != sorted(unary_ids),
            "flow_improvement_over_unary": unary_flow - residual_flow,
            "jaccard_distance_from_unary": (
                0.0
                if not union
                else 1.0 - len(selected & set(unary_ids)) / len(union)
            ),
        })
    candidates.sort(
        key=lambda row: (
            row["n_selected"],
            row["remaining_support_flow"],
            row["selected_ids"],
        )
    )
    final_selection = tuple(low_result["selected_ids"])
    strict_candidate = next(
        candidate
        for candidate in candidates
        if tuple(candidate["selected_ids"]) == final_selection
    )
    observed_by_lambda = sorted(
        evaluated,
        key=lambda row: float(row["lambda"]),
    )
    return {
        "status": "ok",
        "initial_flow": initial_flow,
        "unary_order": unary_order,
        "unary_scores": network.unit_scores,
        "candidates": candidates,
        "strict_candidate": strict_candidate,
        "diagnostics": {
            "solver": "binary_lambda_supported_mincut",
            "beta": beta,
            "strict_flow_threshold": threshold,
            "binary_iterations": iterations,
            "bracket_expansions": bracket_expansions,
            "lambda_feasible_lower": low,
            "lambda_infeasible_upper": high,
            "lambda_bracket_width": high - low,
            "mincut_calls": len(evaluated),
            "distinct_candidate_sets": len(candidates),
            "distinct_nonempty_candidate_sets": sum(
                candidate["n_selected"] > 0 for candidate in candidates
            ),
            "cardinality_monotone_over_ascending_lambda": all(
                int(left["n_selected"]) >= int(right["n_selected"])
                for left, right in zip(
                    observed_by_lambda,
                    observed_by_lambda[1:],
                )
            ),
            "residual_cut_monotone_over_ascending_lambda": all(
                float(left["residual_cut_capacity"])
                <= float(right["residual_cut_capacity"]) + 1e-10
                for left, right in zip(
                    observed_by_lambda,
                    observed_by_lambda[1:],
                )
            ),
            "guarantee_scope": (
                "exact for every fixed-lambda min-cut; binary search returns "
                "the largest threshold-feasible supported cut"
            ),
        },
    }


def solve_bicriteria_flow_interdiction(
    network: RawContributionNetwork,
    *,
    beta: float,
    eta: float = 1.0,
    max_k_guess: int | None = None,
    gamma: float | None = None,
) -> dict:
    """Run the theorem-aligned cardinality guesses for residual-flow interdiction.

    For threshold ``B = beta * Phi(empty)`` and each integer guess ``g``, the
    fixed Lagrangian uses ``mu = g / (eta * B)``.  ``solve_mixed_cut`` scales
    this objective by ``1 / mu``, so its token gate cost is ``eta * B / g``.
    If the strict-threshold optimum has size ``k`` and ``k`` is among the
    guesses, the smallest returned set with residual flow at most
    ``(1 + eta) * B`` satisfies the residual-flow side of the bicriteria
    bound. When ``gamma`` is provided, guesses follow a geometric grid with
    ratio ``1 + gamma``. The tightened cardinality factor is
    ``1 + (1 + gamma) / eta``: fixed-price optimality preserves the optimum's
    ``k`` token cost and then pays at most ``B / lambda``.
    """

    if not 0 < beta < 1:
        raise ValueError("beta must be strictly between zero and one")
    if eta <= 0 or not isfinite(eta):
        raise ValueError("eta must be finite and positive")
    if max_k_guess is not None and max_k_guess <= 0:
        raise ValueError("max_k_guess must be positive when provided")
    if gamma is not None and (gamma <= 0 or not isfinite(gamma)):
        raise ValueError("gamma must be finite and positive when provided")
    if network.status != "ok":
        return {
            "status": network.status,
            "initial_flow": 0.0,
            "unary_order": [],
            "candidates": [],
            "strict_candidate": None,
            "bicriteria_candidate": None,
            "diagnostics": {},
        }

    initial_flow = remaining_support_flow(network, frozenset())
    if initial_flow <= 0:
        return {
            "status": "zero_initial_flow",
            "initial_flow": initial_flow,
            "unary_order": [],
            "candidates": [],
            "strict_candidate": None,
            "bicriteria_candidate": None,
            "diagnostics": {},
        }

    editable = _purchasable_gate_ids(network)
    guess_limit = min(max_k_guess or len(editable), len(editable))
    guesses: list[float | int]
    if gamma is None:
        guesses = list(range(1, guess_limit + 1))
    else:
        guesses = _geometric_cardinality_guesses(guess_limit, gamma)
    threshold = beta * initial_flow
    relaxed_threshold = (1.0 + eta) * threshold
    unary_order = sorted(
        editable,
        key=lambda unit_id: (-network.unit_scores[unit_id], unit_id),
    )

    by_selection: dict[tuple[str, ...], dict] = {}
    for guessed_k in guesses:
        flow_penalty = guessed_k / (eta * threshold)
        token_cost = 1.0 / flow_penalty
        result = solve_mixed_cut(network, token_cost)
        selection = tuple(result["selected_ids"])
        if selection not in by_selection:
            by_selection[selection] = {
                **result,
                "guessed_k_values": [guessed_k],
                "flow_penalty_mu_values": [flow_penalty],
            }
        else:
            by_selection[selection]["guessed_k_values"].append(guessed_k)
            by_selection[selection]["flow_penalty_mu_values"].append(
                flow_penalty
            )

    candidates = []
    for selection, result in by_selection.items():
        selected = set(selection)
        size = len(selected)
        residual_flow = remaining_support_flow(network, selected)
        unary_ids = unary_order[:size]
        unary_flow = remaining_support_flow(network, set(unary_ids))
        union = selected | set(unary_ids)
        candidates.append({
            **result,
            "remaining_support_flow": residual_flow,
            "remaining_support_fraction": residual_flow / initial_flow,
            "strict_feasible": residual_flow <= threshold + 1e-12,
            "bicriteria_feasible": residual_flow <= relaxed_threshold + 1e-12,
            "unary_matched_ids": unary_ids,
            "unary_remaining_support_flow": unary_flow,
            "unary_remaining_support_fraction": unary_flow / initial_flow,
            "differs_from_unary": list(selection) != sorted(unary_ids),
            "flow_improvement_over_unary": unary_flow - residual_flow,
            "jaccard_distance_from_unary": (
                0.0
                if not union
                else 1.0 - len(selected & set(unary_ids)) / len(union)
            ),
        })
    candidates.sort(
        key=lambda row: (
            row["n_selected"],
            row["remaining_support_flow"],
            row["selected_ids"],
        )
    )

    def best(feasibility_field: str) -> dict | None:
        return min(
            (
                candidate
                for candidate in candidates
                if candidate[feasibility_field]
            ),
            key=lambda row: (
                row["n_selected"],
                row["remaining_support_flow"],
                row["selected_ids"],
            ),
            default=None,
        )

    return {
        "status": "ok",
        "initial_flow": initial_flow,
        "unary_order": unary_order,
        "unary_scores": network.unit_scores,
        "candidates": candidates,
        "strict_candidate": best("strict_feasible"),
        "bicriteria_candidate": best("bicriteria_feasible"),
        "diagnostics": {
            "solver": (
                "integer_k_guessing_mincut"
                if gamma is None
                else "geometric_k_guessing_mincut"
            ),
            "beta": beta,
            "eta": eta,
            "strict_flow_threshold": threshold,
            "bicriteria_flow_threshold": relaxed_threshold,
            "max_k_guess": guess_limit,
            "gamma": gamma,
            "guess_scales": guesses,
            "mincut_calls": len(guesses),
            "distinct_candidate_sets": len(candidates),
            "distinct_nonempty_candidate_sets": sum(
                candidate["n_selected"] > 0 for candidate in candidates
            ),
            "strict_candidate_found": best("strict_feasible") is not None,
            "bicriteria_candidate_found": (
                best("bicriteria_feasible") is not None
            ),
            "cardinality_factor": (
                1.0
                + (1.0 if gamma is None else 1.0 + gamma) / eta
            ),
            "residual_flow_factor": 1.0 + eta,
            "guarantee_scope": (
                "strict-threshold optimum when its cardinality is no larger "
                "than max_k_guess; guarantee applies to graph residual flow only"
            ),
        },
    }


def _geometric_cardinality_guesses(
    guess_limit: int,
    gamma: float,
) -> list[float]:
    """Return 1, (1 + gamma), ... with the final scale capped at the limit."""

    if guess_limit <= 0:
        return []
    guesses = [1.0]
    ratio = 1.0 + gamma
    while guesses[-1] < guess_limit:
        next_guess = guesses[-1] * ratio
        if next_guess >= guess_limit:
            guesses.append(float(guess_limit))
            break
        guesses.append(next_guess)
    return guesses


def remaining_group_support_flow(
    network: RawContributionNetwork,
    selected_unit_ids: set[str] | frozenset[str],
) -> float:
    """Remove every layer-copy gate carrying one of the selected token labels."""

    selected = {str(unit_id) for unit_id in selected_unit_ids}
    removed_gates = {
        gate_id
        for gate_id, unit_id in network.selection_unit_by_gate.items()
        if unit_id in selected
    }
    return remaining_support_flow(network, removed_gates)


def prune_feasible_group_selection(
    network: RawContributionNetwork,
    selected_unit_ids: set[str] | frozenset[str],
    *,
    flow_threshold: float,
    flow_cache: dict[frozenset[str], float] | None = None,
) -> dict:
    """Delete redundant labels while preserving a residual-flow threshold.

    Labels are tried in increasing order of their one-label restoration flow.
    One pass is sufficient for inclusion minimality in that order: residual
    max-flow can only increase as labels are removed, so a rejected deletion
    cannot become feasible later.
    """

    if flow_threshold < 0 or not isfinite(flow_threshold):
        raise ValueError("flow_threshold must be finite and non-negative")
    current = frozenset(str(unit_id) for unit_id in selected_unit_ids)
    cache = flow_cache if flow_cache is not None else {}
    flow_calls = 0

    def measured(selection: frozenset[str]) -> float:
        nonlocal flow_calls
        if selection not in cache:
            cache[selection] = remaining_group_support_flow(
                network,
                selection,
            )
            flow_calls += 1
        return cache[selection]

    initial_flow = measured(current)
    if initial_flow > flow_threshold + 1e-12:
        return {
            "status": "infeasible_input",
            "selected_ids": sorted(current),
            "n_selected": len(current),
            "remaining_support_flow": initial_flow,
            "flow_calls": flow_calls,
        }

    order = sorted(
        current,
        key=lambda unit_id: (
            measured(current - {unit_id}),
            unit_id,
        ),
    )
    for unit_id in order:
        trial = current - {unit_id}
        if measured(trial) <= flow_threshold + 1e-12:
            current = trial
    return {
        "status": "feasible_inclusion_minimal",
        "selected_ids": sorted(current),
        "n_selected": len(current),
        "remaining_support_flow": measured(current),
        "flow_calls": flow_calls,
        "deletion_order": order,
    }


def _reverse_delete_budget_candidates(
    network: RawContributionNetwork,
    candidates: list[dict],
    unary_order: list[str],
    *,
    initial_flow: float,
    threshold: float,
    relaxed_threshold: float,
    maximum_budget: int,
    flow_cache: dict[frozenset[str], float] | None = None,
) -> tuple[list[dict], int]:
    """Repair supported group cuts into a nested candidate for every budget."""

    cache = flow_cache if flow_cache is not None else {}
    for candidate in candidates:
        cache.setdefault(
            frozenset(candidate["selected_ids"]),
            float(candidate["remaining_support_flow"]),
        )
    repair_flow_calls = 0

    def cached_group_flow(selection: frozenset[str]) -> float:
        nonlocal repair_flow_calls
        if selection not in cache:
            cache[selection] = remaining_group_support_flow(
                network,
                selection,
            )
            repair_flow_calls += 1
        return cache[selection]

    reverse_chains = []
    for candidate in candidates:
        selection = frozenset(candidate["selected_ids"])
        if not selection:
            continue
        chain = {len(selection): selection}
        current = selection
        while len(current) > 1:
            current = min(
                (current - {unit_id} for unit_id in sorted(current)),
                key=lambda trial: (
                    cached_group_flow(trial),
                    sorted(trial),
                ),
            )
            chain[len(current)] = current
        reverse_chains.append((candidate, chain))

    budget_candidates = []
    for budget in range(1, maximum_budget + 1):
        options = []
        for source_candidate, chain in reverse_chains:
            size = min(budget, max(chain))
            selection = chain[size]
            residual_flow = cached_group_flow(selection)
            unary_ids = unary_order[:size]
            unary_flow = remaining_group_support_flow(network, set(unary_ids))
            selected = set(selection)
            union = selected | set(unary_ids)
            options.append({
                "status": "optimal_supported_then_reverse_delete",
                "budget": budget,
                "selected_ids": sorted(selection),
                "n_selected": size,
                "remaining_support_flow": residual_flow,
                "remaining_support_fraction": residual_flow / initial_flow,
                "strict_feasible": residual_flow <= threshold + 1e-12,
                "bicriteria_feasible": (
                    residual_flow <= relaxed_threshold + 1e-12
                ),
                "unary_matched_ids": unary_ids,
                "unary_remaining_support_flow": unary_flow,
                "unary_remaining_support_fraction": unary_flow / initial_flow,
                "differs_from_unary": sorted(selection) != sorted(unary_ids),
                "flow_improvement_over_unary": unary_flow - residual_flow,
                "jaccard_distance_from_unary": (
                    0.0
                    if not union
                    else 1.0 - len(selected & set(unary_ids)) / len(union)
                ),
                "source_candidate_ids": source_candidate["selected_ids"],
                "solver": "supported_group_cut_reverse_delete_budget_repair",
            })
        if options:
            budget_candidates.append(min(
                options,
                key=lambda row: (
                    row["remaining_support_flow"],
                    row["n_selected"],
                    row["selected_ids"],
                ),
            ))
    return budget_candidates, repair_flow_calls


def solve_group_bicriteria_flow_interdiction(
    network: RawContributionNetwork,
    *,
    beta: float,
    eta: float = 1.0,
    max_k_guess: int | None = None,
) -> dict:
    """Approximate grouped layer-copy interdiction by min-cut and label rounding.

    Each copy of token ``i`` receives relaxation cost ``1 / r_i``, where
    ``r_i`` is its active number of layer copies.  A min-cut selects copies;
    rounding selects every token label represented by a cut copy and removes
    all copies carrying that label.  This preserves the original layered DAG
    during optimization and incurs a rank-dependent cardinality factor.
    """

    if not 0 < beta < 1:
        raise ValueError("beta must be strictly between zero and one")
    if eta <= 0 or not isfinite(eta):
        raise ValueError("eta must be finite and positive")
    if max_k_guess is not None and max_k_guess <= 0:
        raise ValueError("max_k_guess must be positive when provided")
    if network.status != "ok":
        return {
            "status": network.status,
            "initial_flow": 0.0,
            "unary_order": [],
            "candidates": [],
            "strict_candidate": None,
            "bicriteria_candidate": None,
            "diagnostics": {},
        }
    if not network.token_nodes_by_unit or not network.selection_unit_by_gate:
        raise ValueError("group interdiction requires labeled layer-copy gates")

    gates_by_unit: dict[str, set[str]] = defaultdict(set)
    for gate_id, unit_id in network.selection_unit_by_gate.items():
        gates_by_unit[unit_id].add(gate_id)
    rank_by_unit = {
        unit_id: len(gate_ids) for unit_id, gate_ids in gates_by_unit.items()
    }
    maximum_rank = max(rank_by_unit.values())
    initial_flow = remaining_support_flow(network, frozenset())
    if initial_flow <= 0:
        return {
            "status": "zero_initial_flow",
            "initial_flow": initial_flow,
            "unary_order": [],
            "candidates": [],
            "strict_candidate": None,
            "bicriteria_candidate": None,
            "diagnostics": {},
        }

    guess_limit = min(max_k_guess or len(gates_by_unit), len(gates_by_unit))
    threshold = beta * initial_flow
    relaxed_threshold = (1.0 + eta) * threshold
    shared_flow_cache = {frozenset(): initial_flow}
    candidate_flow_calls = 0

    def cached_group_flow(selection) -> float:
        nonlocal candidate_flow_calls
        key = frozenset(str(unit_id) for unit_id in selection)
        if key not in shared_flow_cache:
            shared_flow_cache[key] = remaining_group_support_flow(
                network,
                key,
            )
            candidate_flow_calls += 1
        return shared_flow_cache[key]

    root_gate_ids = set(network.roots_by_unit)
    unary_scores: dict[str, float] = defaultdict(float)
    for gate_id in root_gate_ids:
        unit_id = network.selection_unit_by_gate.get(gate_id)
        if unit_id is None:
            continue
        unary_scores[unit_id] += network.unit_scores.get(gate_id, 0.0)
    for unit_id in gates_by_unit:
        unary_scores.setdefault(unit_id, 0.0)
    unary_order = sorted(
        unary_scores,
        key=lambda unit_id: (-unary_scores[unit_id], unit_id),
    )

    by_selection: dict[tuple[str, ...], dict] = {}
    for guessed_k in range(1, guess_limit + 1):
        flow_penalty = guessed_k / (eta * threshold)
        relaxation_scale = 1.0 / flow_penalty
        immutable_gate_cost = sum(
            capacity for _, _, capacity in network.edges
        ) + 1.0
        gate_costs = {
            gate_id: (
                relaxation_scale
                / rank_by_unit[network.selection_unit_by_gate[gate_id]]
                if gate_id in network.selection_unit_by_gate
                else immutable_gate_cost
            )
            for gate_id in network.token_nodes_by_unit
        }
        copy_result = solve_weighted_mixed_cut(network, gate_costs)
        selected_copy_ids = copy_result["selected_ids"]
        if any(
            gate_id not in network.selection_unit_by_gate
            for gate_id in selected_copy_ids
        ):
            raise RuntimeError("weighted min-cut selected an immutable token gate")
        selection = tuple(sorted({
            network.selection_unit_by_gate[gate_id]
            for gate_id in selected_copy_ids
        }))
        rounded_flow = cached_group_flow(selection)
        unscaled_copy_cost = sum(
            1.0 / rank_by_unit[network.selection_unit_by_gate[gate_id]]
            for gate_id in selected_copy_ids
        )
        candidate = {
            "status": "optimal_rounded",
            "selected_ids": list(selection),
            "selected_copy_ids": selected_copy_ids,
            "n_selected": len(selection),
            "n_selected_copies": len(selected_copy_ids),
            "guessed_k_values": [guessed_k],
            "flow_penalty_mu_values": [flow_penalty],
            "copy_relaxation_cost": unscaled_copy_cost,
            "copy_cut_remaining_support_flow": copy_result[
                "residual_cut_capacity"
            ],
            "remaining_support_flow": rounded_flow,
            "remaining_support_fraction": rounded_flow / initial_flow,
            "strict_feasible": rounded_flow <= threshold + 1e-12,
            "bicriteria_feasible": rounded_flow <= relaxed_threshold + 1e-12,
            "copy_cut_identity_error": copy_result["cut_identity_error"],
            "solver": "weighted_copy_mincut_then_group_rounding",
        }
        previous = by_selection.get(selection)
        if previous is None:
            by_selection[selection] = candidate
        else:
            previous["guessed_k_values"].append(guessed_k)
            previous["flow_penalty_mu_values"].append(flow_penalty)
            if (
                candidate["copy_relaxation_cost"],
                candidate["n_selected_copies"],
            ) < (
                previous["copy_relaxation_cost"],
                previous["n_selected_copies"],
            ):
                candidate["guessed_k_values"] = previous["guessed_k_values"]
                candidate["flow_penalty_mu_values"] = previous[
                    "flow_penalty_mu_values"
                ]
                by_selection[selection] = candidate

    candidates = []
    for candidate in by_selection.values():
        size = int(candidate["n_selected"])
        selected = set(candidate["selected_ids"])
        unary_ids = unary_order[:size]
        unary_flow = cached_group_flow(unary_ids)
        union = selected | set(unary_ids)
        candidates.append({
            **candidate,
            "unary_matched_ids": unary_ids,
            "unary_remaining_support_flow": unary_flow,
            "unary_remaining_support_fraction": unary_flow / initial_flow,
            "differs_from_unary": list(candidate["selected_ids"]) != sorted(unary_ids),
            "flow_improvement_over_unary": (
                unary_flow - candidate["remaining_support_flow"]
            ),
            "jaccard_distance_from_unary": (
                0.0
                if not union
                else 1.0 - len(selected & set(unary_ids)) / len(union)
            ),
        })
    candidates.sort(
        key=lambda row: (
            row["n_selected"],
            row["remaining_support_flow"],
            row["selected_ids"],
        )
    )

    def best(feasibility_field: str) -> dict | None:
        return min(
            (
                candidate
                for candidate in candidates
                if candidate[feasibility_field]
            ),
            key=lambda row: (
                row["n_selected"],
                row["remaining_support_flow"],
                row["selected_ids"],
            ),
            default=None,
        )

    threshold_repair_flow_calls = 0
    repaired_candidates = []
    for feasibility_field, flow_bound, repair_name in (
        ("strict_feasible", threshold, "strict"),
        ("bicriteria_feasible", relaxed_threshold, "bicriteria"),
    ):
        source_candidate = best(feasibility_field)
        if source_candidate is None:
            continue
        repair = prune_feasible_group_selection(
            network,
            set(source_candidate["selected_ids"]),
            flow_threshold=flow_bound,
            flow_cache=shared_flow_cache,
        )
        threshold_repair_flow_calls += int(repair["flow_calls"])
        selection = repair["selected_ids"]
        selected = set(selection)
        size = len(selection)
        unary_ids = unary_order[:size]
        unary_flow = cached_group_flow(unary_ids)
        union = selected | set(unary_ids)
        repaired_candidates.append({
            **source_candidate,
            "status": "optimal_rounded_then_threshold_pruned",
            "selected_ids": selection,
            "selected_copy_ids": [],
            "n_selected": size,
            "n_selected_copies": None,
            "remaining_support_flow": repair["remaining_support_flow"],
            "remaining_support_fraction": (
                repair["remaining_support_flow"] / initial_flow
            ),
            "strict_feasible": (
                repair["remaining_support_flow"] <= threshold + 1e-12
            ),
            "bicriteria_feasible": (
                repair["remaining_support_flow"] <= relaxed_threshold + 1e-12
            ),
            "unary_matched_ids": unary_ids,
            "unary_remaining_support_flow": unary_flow,
            "unary_remaining_support_fraction": unary_flow / initial_flow,
            "differs_from_unary": selection != sorted(unary_ids),
            "flow_improvement_over_unary": (
                unary_flow - repair["remaining_support_flow"]
            ),
            "jaccard_distance_from_unary": (
                0.0
                if not union
                else 1.0 - len(selected & set(unary_ids)) / len(union)
            ),
            "source_candidate_ids": source_candidate["selected_ids"],
            "threshold_repair": repair_name,
            "threshold_repair_flow_calls": repair["flow_calls"],
            "solver": "weighted_copy_mincut_group_rounding_threshold_pruning",
        })
    by_repaired_selection = {tuple(row["selected_ids"]): row for row in candidates}
    for candidate in repaired_candidates:
        selection = tuple(candidate["selected_ids"])
        previous = by_repaired_selection.get(selection)
        if previous is None or (
            candidate["remaining_support_flow"],
            candidate["n_selected"],
        ) < (
            previous["remaining_support_flow"],
            previous["n_selected"],
        ):
            by_repaired_selection[selection] = candidate
    candidates = sorted(
        by_repaired_selection.values(),
        key=lambda row: (
            row["n_selected"],
            row["remaining_support_flow"],
            row["selected_ids"],
        ),
    )

    budget_candidates, repair_flow_calls = _reverse_delete_budget_candidates(
        network,
        candidates,
        unary_order,
        initial_flow=initial_flow,
        threshold=threshold,
        relaxed_threshold=relaxed_threshold,
        maximum_budget=guess_limit,
        flow_cache=shared_flow_cache,
    )
    strict_candidate = best("strict_feasible")
    bicriteria_candidate = best("bicriteria_feasible")
    return {
        "status": "ok",
        "initial_flow": initial_flow,
        "unary_order": unary_order,
        "unary_scores": dict(unary_scores),
        "candidates": candidates,
        "budget_candidates": budget_candidates,
        "strict_candidate": strict_candidate,
        "bicriteria_candidate": bicriteria_candidate,
        "diagnostics": {
            "solver": "group_k_guessing_copy_mincut_rounding",
            "beta": beta,
            "eta": eta,
            "strict_flow_threshold": threshold,
            "bicriteria_flow_threshold": relaxed_threshold,
            "max_k_guess": guess_limit,
            "mincut_calls": guess_limit,
            "candidate_flow_calls": candidate_flow_calls,
            "threshold_repair_flow_calls": threshold_repair_flow_calls,
            "budget_repair_flow_calls": repair_flow_calls,
            "budget_candidates": len(budget_candidates),
            "distinct_candidate_sets": len(candidates),
            "distinct_nonempty_candidate_sets": sum(
                candidate["n_selected"] > 0 for candidate in candidates
            ),
            "strict_candidate_found": strict_candidate is not None,
            "bicriteria_candidate_found": bicriteria_candidate is not None,
            "maximum_active_group_rank": maximum_rank,
            "cardinality_factor": maximum_rank * (1.0 + 1.0 / eta),
            "residual_flow_factor": 1.0 + eta,
            "guarantee_scope": (
                "grouped layered-copy residual-flow optimum when its cardinality "
                "is no larger than max_k_guess; guarantee applies to the graph only"
            ),
        },
    }


def direct_support_source_prizes(
    graph_row: dict,
    units: list[dict],
    network: RawContributionNetwork,
) -> dict[str, float]:
    """Aggregate raw positive target-logit support onto editable source words."""

    if network.status != "ok":
        return {}
    position_to_unit = _position_to_unit(graph_row, units)
    prizes = {unit_id: 0.0 for unit_id in network.roots_by_unit}
    for token in graph_row.get("context_token_supports", []):
        support = float(token.get("support", 0.0))
        if not isfinite(support):
            raise ValueError("context support contains a non-finite value")
        if support <= 0:
            continue
        unit_id = position_to_unit.get(int(token.get("position", -1)))
        if unit_id in prizes:
            prizes[str(unit_id)] += support
    return prizes


def solve_prize_collecting_separator(
    network: RawContributionNetwork,
    source_prizes: dict[str, float],
    token_cost: float,
) -> dict:
    """Exactly solve a prize-collecting token separator for fixed lambda.

    The objective is lambda times the number of edited words plus the raw
    contribution prize of source words that remain connected to the answer.
    Internal contribution edges enforce reachability but cannot themselves be
    bought by the cut.
    """

    if network.status != "ok":
        return _empty_cut(network.status, token_cost)
    if network.gate_scope != "contracted_token_nodes":
        return _empty_cut("prize_separator_requires_token_projection", token_cost)
    if token_cost < 0 or not isfinite(token_cost):
        raise ValueError("token_cost must be finite and non-negative")
    prizes = _validated_source_prizes(network, source_prizes)
    total_prize = sum(prizes.values())
    if total_prize <= 0:
        return _empty_cut("no_positive_source_prize", token_cost)

    state = _run_prize_flow(network, prizes, token_cost)
    selected = sorted(
        unit_id
        for unit_id, (gate_in, gate_out) in state["gates"].items()
        if gate_in in state["reachable"] and gate_out not in state["reachable"]
    )
    covered_ids = sorted(
        unit_id
        for unit_id in prizes
        if state["gates"][unit_id][0] in state["reachable"]
    )
    covered_prize = sum(prizes[unit_id] for unit_id in covered_ids)
    uncovered_prize = total_prize - covered_prize
    objective = token_cost * len(selected) + uncovered_prize
    return {
        "status": "optimal",
        "lambda": token_cost,
        "selected_ids": selected,
        "n_selected": len(selected),
        "covered_source_ids": covered_ids,
        "covered_source_prize": covered_prize,
        "covered_source_fraction": covered_prize / total_prize,
        "uncovered_source_prize": uncovered_prize,
        "total_source_prize": total_prize,
        "net_benefit": covered_prize - token_cost * len(selected),
        "objective_value": objective,
        "max_flow_value": state["flow"],
        "cut_identity_error": abs(objective - state["flow"]),
        "solver": "float_dinic_prize_collecting_token_separator",
    }


def covered_source_prize(
    network: RawContributionNetwork,
    source_prizes: dict[str, float],
    removed_ids: set[str] | frozenset[str],
) -> float:
    """Measure source prize separated from the answer by a fixed token set."""

    if network.status != "ok":
        return 0.0
    prizes = _validated_source_prizes(network, source_prizes)
    removed_nodes = {
        network.token_nodes_by_unit[unit_id]
        for unit_id in removed_ids
        if unit_id in network.token_nodes_by_unit
    }
    reverse: dict[str, set[str]] = defaultdict(set)
    for src, dst, _ in network.edges:
        if src in removed_nodes or dst in removed_nodes:
            continue
        reverse[dst].add(src)
    can_reach_sink = _reachable({network.sink}, reverse)
    return sum(
        prize
        for unit_id, prize in prizes.items()
        if network.token_nodes_by_unit[unit_id] not in can_reach_sink
    )


def sweep_prize_collecting_separators(
    network: RawContributionNetwork,
    source_prizes: dict[str, float],
    *,
    points: int = 64,
    dynamic_range: float = 1e6,
) -> dict:
    """Enumerate supported prize-collecting separators over lambda."""

    if points < 2:
        raise ValueError("points must be at least two")
    if dynamic_range <= 1:
        raise ValueError("dynamic_range must be greater than one")
    if network.status != "ok":
        return {
            "status": network.status,
            "total_source_prize": 0.0,
            "unary_order": [],
            "candidates": [],
            "diagnostics": {},
        }
    prizes = _validated_source_prizes(network, source_prizes)
    total_prize = sum(prizes.values())
    if total_prize <= 0:
        return {
            "status": "no_positive_source_prize",
            "total_source_prize": 0.0,
            "unary_order": [],
            "candidates": [],
            "diagnostics": {},
        }

    high = total_prize * 2.0
    low = max(high / dynamic_range, 1e-12)
    log_high = log(high)
    log_low = log(low)
    lambdas = [
        exp(log_high + (log_low - log_high) * index / (points - 1))
        for index in range(points)
    ]
    by_selection: dict[tuple[str, ...], dict] = {}
    cardinalities = []
    for token_cost in lambdas:
        result = solve_prize_collecting_separator(network, prizes, token_cost)
        selection = tuple(result["selected_ids"])
        cardinalities.append(result["n_selected"])
        if selection not in by_selection:
            by_selection[selection] = {
                **result,
                "lambda_max": token_cost,
                "lambda_min": token_cost,
            }
        else:
            by_selection[selection]["lambda_min"] = token_cost

    unary_order = sorted(
        network.token_nodes_by_unit,
        key=lambda unit_id: (-prizes.get(unit_id, 0.0), unit_id),
    )
    candidates = []
    for selection, result in by_selection.items():
        selected = set(selection)
        size = len(selected)
        covered = covered_source_prize(network, prizes, selected)
        unary_ids = unary_order[:size]
        unary_covered = covered_source_prize(network, prizes, set(unary_ids))
        union = selected | set(unary_ids)
        candidates.append({
            **result,
            "covered_source_prize": covered,
            "covered_source_fraction": covered / total_prize,
            "unary_matched_ids": unary_ids,
            "unary_covered_source_prize": unary_covered,
            "unary_covered_source_fraction": unary_covered / total_prize,
            "differs_from_unary": list(selection) != sorted(unary_ids),
            "coverage_improvement_over_unary": covered - unary_covered,
            "jaccard_distance_from_unary": (
                0.0 if not union else 1.0 - len(selected & set(unary_ids)) / len(union)
            ),
        })
    candidates.sort(
        key=lambda row: (
            row["n_selected"],
            -row["covered_source_prize"],
            row["selected_ids"],
        )
    )
    return {
        "status": "ok",
        "total_source_prize": total_prize,
        "source_prizes": prizes,
        "unary_order": unary_order,
        "candidates": candidates,
        "diagnostics": {
            "objective": "lambda_times_tokens_plus_uncovered_source_prize",
            "internal_edge_capacity": "infinite_topology_constraint",
            "source_prize_semantics": "raw_positive_target_logit_support",
            "lambda_points": points,
            "lambda_max": high,
            "lambda_min": low,
            "distinct_candidate_sets": len(candidates),
            "distinct_nonempty_candidate_sets": sum(
                candidate["n_selected"] > 0 for candidate in candidates
            ),
            "cardinality_monotone_over_descending_lambda": all(
                left <= right
                for left, right in zip(cardinalities, cardinalities[1:])
            ),
            "candidate_cardinalities": sorted({
                candidate["n_selected"] for candidate in candidates
            }),
        },
    }


def _run_flow(
    network: RawContributionNetwork,
    gate_capacities: dict[str, float],
) -> dict:
    dinic = Dinic()
    source = dinic.node()
    token_node_to_unit = {
        node: unit_id for unit_id, node in network.token_nodes_by_unit.items()
    }
    node_ids = {
        node: dinic.node()
        for node in sorted(network.nodes)
        if node not in token_node_to_unit
    }
    finite_total = sum(capacity for _, _, capacity in network.edges)
    finite_total += sum(gate_capacities.values())
    infinite_capacity = finite_total + 1.0
    gates = {}
    if network.gate_scope == "contracted_token_nodes":
        for unit_id in sorted(network.token_nodes_by_unit):
            gate_in = dinic.node()
            gate_out = dinic.node()
            gates[unit_id] = (gate_in, gate_out)
            dinic.add_edge(
                gate_in,
                gate_out,
                gate_capacities.get(unit_id, infinite_capacity),
            )

        def src_id(node: str) -> int:
            unit_id = token_node_to_unit.get(node)
            return gates[unit_id][1] if unit_id is not None else node_ids[node]

        def dst_id(node: str) -> int:
            unit_id = token_node_to_unit.get(node)
            return gates[unit_id][0] if unit_id is not None else node_ids[node]

        for src, dst, capacity in network.edges:
            dinic.add_edge(src_id(src), dst_id(dst), capacity)
        for unit_id in network.roots_by_unit:
            dinic.add_edge(source, gates[unit_id][0], infinite_capacity)
        expanded_node_ids = {
            node: (
                gates[token_node_to_unit[node]][0]
                if node in token_node_to_unit
                else node_ids[node]
            )
            for node in network.nodes
        }
        cut_src_ids = {
            node: (
                gates[token_node_to_unit[node]][1]
                if node in token_node_to_unit
                else node_ids[node]
            )
            for node in network.nodes
        }
    else:
        for src, dst, capacity in network.edges:
            dinic.add_edge(node_ids[src], node_ids[dst], capacity)
        for unit_id, roots in sorted(network.roots_by_unit.items()):
            gate_in = dinic.node()
            gate_out = dinic.node()
            gates[unit_id] = (gate_in, gate_out)
            dinic.add_edge(source, gate_in, infinite_capacity)
            dinic.add_edge(
                gate_in,
                gate_out,
                gate_capacities.get(unit_id, infinite_capacity),
            )
            for root in roots:
                dinic.add_edge(gate_out, node_ids[root], infinite_capacity)
        expanded_node_ids = node_ids
        cut_src_ids = node_ids

    sink = expanded_node_ids[network.sink]
    flow = dinic.max_flow(source, sink)
    return {
        "flow": flow,
        "reachable": dinic.reachable(source),
        "node_ids": expanded_node_ids,
        "cut_src_ids": cut_src_ids,
        "gates": gates,
    }


def _run_prize_flow(
    network: RawContributionNetwork,
    source_prizes: dict[str, float],
    token_cost: float,
) -> dict:
    dinic = Dinic()
    source = dinic.node()
    token_node_to_unit = {
        node: unit_id for unit_id, node in network.token_nodes_by_unit.items()
    }
    node_ids = {
        node: dinic.node()
        for node in sorted(network.nodes)
        if node not in token_node_to_unit
    }
    finite_total = sum(source_prizes.values())
    finite_total += token_cost * len(network.token_nodes_by_unit)
    infinite_capacity = finite_total + 1.0
    gates = {}
    for unit_id in sorted(network.token_nodes_by_unit):
        gate_in = dinic.node()
        gate_out = dinic.node()
        gates[unit_id] = (gate_in, gate_out)
        dinic.add_edge(gate_in, gate_out, token_cost)

    def src_id(node: str) -> int:
        unit_id = token_node_to_unit.get(node)
        return gates[unit_id][1] if unit_id is not None else node_ids[node]

    def dst_id(node: str) -> int:
        unit_id = token_node_to_unit.get(node)
        return gates[unit_id][0] if unit_id is not None else node_ids[node]

    for src, dst, _ in network.edges:
        dinic.add_edge(src_id(src), dst_id(dst), infinite_capacity)
    for unit_id, prize in source_prizes.items():
        if prize > 0:
            dinic.add_edge(source, gates[unit_id][0], prize)

    flow = dinic.max_flow(source, dst_id(network.sink))
    return {
        "flow": flow,
        "reachable": dinic.reachable(source),
        "gates": gates,
    }


def _validated_source_prizes(
    network: RawContributionNetwork,
    source_prizes: dict[str, float],
) -> dict[str, float]:
    prizes = {}
    for unit_id in network.roots_by_unit:
        prize = float(source_prizes.get(unit_id, 0.0))
        if not isfinite(prize) or prize < 0:
            raise ValueError("source prizes must be finite and non-negative")
        prizes[unit_id] = prize
    return prizes


def _reachable(starts: set[str], adjacency: dict[str, set[str]]) -> set[str]:
    seen = set(starts)
    stack = list(starts)
    while stack:
        node = stack.pop()
        for neighbor in adjacency.get(node, set()):
            if neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return seen


def _empty_network(status: str, diagnostics: dict) -> RawContributionNetwork:
    return RawContributionNetwork(
        status=status,
        nodes=frozenset(),
        edges=(),
        roots_by_unit={},
        token_nodes_by_unit={},
        selection_unit_by_gate={},
        gate_scope="none",
        unit_scores={},
        sink="answer_target",
        diagnostics=diagnostics,
    )


def _empty_cut(status: str, token_cost: float) -> dict:
    return {
        "status": status,
        "lambda": token_cost,
        "selected_ids": [],
        "n_selected": 0,
        "residual_cut_capacity": 0.0,
        "objective_value": 0.0,
        "max_flow_value": 0.0,
        "cut_identity_error": 0.0,
        "solver": "none",
    }


def _editable_unit_ids(network: RawContributionNetwork) -> set[str]:
    if network.gate_scope == "contracted_token_nodes":
        return set(network.token_nodes_by_unit)
    return set(network.roots_by_unit)


def _purchasable_gate_ids(network: RawContributionNetwork) -> set[str]:
    return set(network.selection_unit_by_gate)
