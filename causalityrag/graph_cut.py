"""Project attribution-token support to editable words and enumerate cut sets."""

from __future__ import annotations

import heapq
import itertools
import math
from collections import defaultdict, deque
from dataclasses import dataclass

from causalityrag.io import retrieved_contexts
from causalityrag.rules import ONTO_TYPES, TypedRuleLibrary
from causalityrag.token_units import STOPWORDS, TOKEN_RE, classify_token
from causalityrag.max_flow import Dinic, INF


@dataclass
class _CostFlowEdge:
    to: int
    rev: int
    cap: int
    cost: float
    original: int
    weight: float


class _MinCostUnitFlow:
    """Successive shortest paths for the small unit-capacity token graphs."""

    def __init__(self) -> None:
        self.graph: list[list[_CostFlowEdge]] = []

    def node(self) -> int:
        self.graph.append([])
        return len(self.graph) - 1

    def add_edge(self, source: int, target: int, cost: float, weight: float) -> None:
        forward = _CostFlowEdge(
            to=target,
            rev=len(self.graph[target]),
            cap=1,
            cost=float(cost),
            original=1,
            weight=float(weight),
        )
        reverse = _CostFlowEdge(
            to=source,
            rev=len(self.graph[source]),
            cap=0,
            cost=-float(cost),
            original=0,
            weight=0.0,
        )
        self.graph[source].append(forward)
        self.graph[target].append(reverse)

    def min_cost_max_flow(self, source: int, target: int) -> tuple[int, float]:
        flow = 0
        total_cost = 0.0
        n_nodes = len(self.graph)
        while True:
            distance = [math.inf] * n_nodes
            predecessor: list[tuple[int, int] | None] = [None] * n_nodes
            in_queue = [False] * n_nodes
            distance[source] = 0.0
            queue = deque([source])
            in_queue[source] = True
            while queue:
                node = queue.popleft()
                in_queue[node] = False
                for edge_index, edge in enumerate(self.graph[node]):
                    candidate = distance[node] + edge.cost
                    if edge.cap <= 0 or candidate + 1e-12 >= distance[edge.to]:
                        continue
                    distance[edge.to] = candidate
                    predecessor[edge.to] = (node, edge_index)
                    if not in_queue[edge.to]:
                        queue.append(edge.to)
                        in_queue[edge.to] = True
            if predecessor[target] is None:
                return flow, total_cost

            node = target
            while node != source:
                previous, edge_index = predecessor[node]
                edge = self.graph[previous][edge_index]
                edge.cap -= 1
                self.graph[node][edge.rev].cap += 1
                node = previous
            flow += 1
            total_cost += distance[target]


def project_graph_support_to_editable_units(
    record: dict,
    graph_row: dict,
    library: TypedRuleLibrary,
    *,
    k: int = 5,
    nlp=None,
    require_positive_support: bool = True,
) -> list[dict]:
    """Aggregate subword rollout mass into exact-offset editable word units."""

    contexts = retrieved_contexts(record)[:k]
    graph_method = str(graph_row.get("method", "attribution_graph"))
    support_rows = graph_row.get("context_token_supports") or graph_row.get("top_context_tokens") or []
    by_chunk: dict[str, list[dict]] = {}
    for row in support_rows:
        chunk_id = str(row.get("chunk_id", ""))
        start = int(row.get("chunk_char_start", -1))
        end = int(row.get("chunk_char_end", -1))
        if chunk_id and start >= 0 and end > start:
            by_chunk.setdefault(chunk_id, []).append(row)

    units = []
    for context in contexts:
        chunk_id = context["chunk_id"]
        token_rows = by_chunk.get(chunk_id, [])
        if not token_rows:
            continue
        text = context["text"]
        entities = list(nlp(text).ents) if nlp is not None else []
        for match in TOKEN_RE.finditer(text):
            word = match.group(0)
            unit_type = classify_token(word)
            if unit_type in {"PUNCT", "STOPWORD"} or word.lower() in STOPWORDS:
                continue
            entity = next(
                (ent for ent in entities if _overlaps(match.start(), match.end(), ent.start_char, ent.end_char)),
                None,
            )
            if entity is not None and entity.label_.upper() in ONTO_TYPES:
                unit_type = entity.label_.upper()
            elif unit_type not in {"NUMBER", "MONEY", "RELATION_CUE"}:
                continue
            support = sum(
                float(row.get("support", 0.0))
                for row in token_rows
                if _overlaps(
                    match.start(),
                    match.end(),
                    int(row["chunk_char_start"]),
                    int(row["chunk_char_end"]),
                )
            )
            if require_positive_support and support <= 0:
                continue
            replacement = library.replacement_for_token(word, unit_type, text)
            if not replacement.get("ok"):
                continue
            units.append({
                "unit_id": f"attr:{chunk_id}:{match.start()}:{match.end()}",
                "text": word,
                "type": unit_type,
                "chunk_id": chunk_id,
                "chunk_rank": context["rank"],
                "chunk_char_start": match.start(),
                "chunk_char_end": match.end(),
                "support": support,
                "cost": 1.0,
                "replacement_preview": replacement["new"],
                "replacement_type": replacement["inferred_type"],
                "sources": [graph_method, "subword_to_word"],
            })
    return sorted(units, key=lambda unit: (-unit["support"], unit["unit_id"]))


def project_source_target_graph(
    record: dict,
    graph_row: dict,
    library: TypedRuleLibrary | None,
    *,
    k: int = 5,
    nlp=None,
    all_context_tokens: bool = False,
) -> tuple[list[dict], dict[str, float], dict[tuple[str, str], float], dict[str, float], dict]:
    """Project direct activation edges to a query-source / target-score graph.

    The projection never compares answer text or answer token identities with
    chunk text.  ``query_source`` and ``answer_target`` are fixed terminals;
    all returned units are editable context words.  Positive context-to-query
    OV writes are reoriented as ``query_source -> context`` support edges,
    while positive context-to-answer-predictor OV writes become
    ``context -> answer_target`` support edges.
    """

    if all_context_tokens:
        units = project_all_context_word_units(record, graph_row, k=k)
    else:
        units = project_graph_support_to_editable_units(
            record,
            graph_row,
            library,
            k=k,
            nlp=nlp,
            require_positive_support=False,
        )
    position_to_unit = _position_to_unit(graph_row, units)
    graph = graph_row.get("graph", {})
    partitions = graph.get("token_partitions", {})
    query_positions = {int(position) for position in partitions.get("query", [])}
    target_predictors = {
        int(position) - 1 for position in graph.get("target_positions", []) if int(position) > 0
    }

    source_edges: dict[str, float] = defaultdict(float)
    interactions: dict[tuple[str, str], float] = defaultdict(float)
    target_edges: dict[str, float] = defaultdict(float)
    source_edge_count = 0
    target_edge_count = 0
    interaction_edge_count = 0
    for edge in graph.get("edges", []):
        if edge.get("kind") != "attention_ov_write":
            continue
        weight = max(0.0, float(edge.get("contribution", 0.0)))
        if weight <= 0:
            continue
        src_position = int(edge["src_position"])
        dst_position = int(edge["dst_position"])
        src_unit = position_to_unit.get(src_position)
        if not src_unit:
            continue
        if dst_position in query_positions:
            source_edges[src_unit] += weight
            source_edge_count += 1
            continue
        if dst_position in target_predictors:
            target_edges[src_unit] += weight
            target_edge_count += 1
            continue
        dst_unit = position_to_unit.get(dst_position)
        if dst_unit and dst_unit != src_unit:
            interactions[(src_unit, dst_unit)] += weight
            interaction_edge_count += 1

    return (
        units,
        dict(source_edges),
        dict(interactions),
        dict(target_edges),
        {
            "query_source": "fixed_query_source",
            "answer_target": "fixed_clean_answer_score",
            "source_edge_count": source_edge_count,
            "interaction_edge_count": interaction_edge_count,
            "target_edge_count": target_edge_count,
            "source_units": len(source_edges),
            "target_units": len(target_edges),
            "uses_answer_chunk_matching": False,
            "unit_domain": "all_context_words" if all_context_tokens else "typed_editable_words",
        },
    )


def project_cached_units_source_target_graph(
    graph_row: dict,
    units: list[dict],
) -> tuple[dict[str, float], dict[tuple[str, str], float], dict[str, float], dict]:
    """Project direct attribution edges onto a supplied editable-unit cache.

    Unlike :func:`project_source_target_graph`, this helper preserves the
    cached unit identifiers used by a frozen replacement registry.  It is
    useful when every proposed token must have a pre-approved replacement.
    """

    position_to_unit = _position_to_unit(graph_row, units)
    graph = graph_row.get("graph", {})
    partitions = graph.get("token_partitions", {})
    query_positions = {int(position) for position in partitions.get("query", [])}
    target_predictors = {
        int(position) - 1 for position in graph.get("target_positions", []) if int(position) > 0
    }
    source_edges: dict[str, float] = defaultdict(float)
    interactions: dict[tuple[str, str], float] = defaultdict(float)
    target_edges: dict[str, float] = defaultdict(float)
    counts = {"source": 0, "interaction": 0, "target": 0}
    for edge in graph.get("edges", []):
        if edge.get("kind") != "attention_ov_write":
            continue
        weight = max(0.0, float(edge.get("contribution", 0.0)))
        if weight <= 0:
            continue
        source = position_to_unit.get(int(edge["src_position"]))
        if source is None:
            continue
        destination_position = int(edge["dst_position"])
        if destination_position in query_positions:
            source_edges[source] += weight
            counts["source"] += 1
            continue
        if destination_position in target_predictors:
            target_edges[source] += weight
            counts["target"] += 1
            continue
        destination = position_to_unit.get(destination_position)
        if destination is not None and destination != source:
            interactions[(source, destination)] += weight
            counts["interaction"] += 1

    # If the direct projection misses one terminal side, recover a token-level
    # shortcut from the raw positive path DAG.  The shortcut weight is the
    # conserved positive flow leaving an input context root whose downstream
    # path reaches answer_target; it is not an arbitrary edge weight.
    if not source_edges or not target_edges:
        forward: dict[str, set[str]] = defaultdict(set)
        reverse: dict[str, set[str]] = defaultdict(set)
        raw_edges: list[tuple[str, str, float, dict]] = []
        for edge in graph.get("edges", []):
            weight = max(0.0, float(edge.get("contribution", 0.0)))
            src = str(edge.get("src", ""))
            dst = str(edge.get("dst", ""))
            if weight <= 0 or not src or not dst or src.startswith("background::"):
                continue
            raw_edges.append((src, dst, weight, edge))
            forward[src].add(dst)
            reverse[dst].add(src)
        can_reach_answer = {"answer_target"}
        frontier = ["answer_target"]
        while frontier:
            node = frontier.pop()
            for predecessor in reverse.get(node, ()):
                if predecessor not in can_reach_answer:
                    can_reach_answer.add(predecessor)
                    frontier.append(predecessor)
        root_flow: dict[str, float] = defaultdict(float)
        for src, dst, weight, edge in raw_edges:
            if int(edge.get("src_layer", -1)) != 0 or dst not in can_reach_answer:
                continue
            unit_id = position_to_unit.get(int(edge.get("src_position", -1)))
            if unit_id is not None:
                root_flow[unit_id] += weight
        # ``context_token_supports`` is the closed-flow mass that was traced
        # backward from the answer objective to the input context.  It is a
        # valid path-compressed token -> answer weight even when the retained
        # edge list does not expose a direct terminal edge.
        support_flow: dict[str, float] = defaultdict(float)
        for row in graph_row.get("context_token_supports", []):
            unit_id = position_to_unit.get(int(row.get("position", -1)))
            support = max(0.0, float(row.get("support", 0.0)))
            if unit_id is not None and support > 0:
                support_flow[unit_id] += support
        shortcut_flow = root_flow or support_flow
        if shortcut_flow:
            for unit_id, weight in shortcut_flow.items():
                if not source_edges:
                    source_edges[unit_id] += weight
                    counts["source"] += 1
                if not target_edges:
                    target_edges[unit_id] += weight
                    counts["target"] += 1
        projection_fallback = bool(shortcut_flow)
    else:
        projection_fallback = False
    return dict(source_edges), dict(interactions), dict(target_edges), {
        "source_edge_count": counts["source"],
        "interaction_edge_count": counts["interaction"],
        "target_edge_count": counts["target"],
        "uses_answer_chunk_matching": False,
        "unit_domain": "cached_editable_units",
        "path_shortcut_fallback": projection_fallback,
    }


def project_all_context_word_units(record: dict, graph_row: dict, *, k: int = 5) -> list[dict]:
    """Return every non-punctuation surface word token as an editable unit."""

    support_rows = graph_row.get("context_token_supports", [])
    by_chunk: dict[str, list[dict]] = defaultdict(list)
    for row in support_rows:
        chunk_id = str(row.get("chunk_id", ""))
        start = int(row.get("chunk_char_start", -1))
        end = int(row.get("chunk_char_end", -1))
        if chunk_id and start >= 0 and end > start:
            by_chunk[chunk_id].append(row)

    units = []
    for context in retrieved_contexts(record)[:k]:
        chunk_id = str(context["chunk_id"])
        token_rows = by_chunk.get(chunk_id, [])
        for match in TOKEN_RE.finditer(context["text"]):
            text = match.group(0)
            unit_type = classify_token(text)
            if unit_type == "PUNCT":
                continue
            support = sum(
                float(row.get("support", 0.0))
                for row in token_rows
                if _overlaps(
                    match.start(),
                    match.end(),
                    int(row["chunk_char_start"]),
                    int(row["chunk_char_end"]),
                )
            )
            units.append({
                "unit_id": f"token:{chunk_id}:{match.start()}:{match.end()}",
                "text": text,
                "type": unit_type,
                "chunk_id": chunk_id,
                "chunk_rank": context["rank"],
                "chunk_char_start": match.start(),
                "chunk_char_end": match.end(),
                "support": support,
                "cost": 1.0,
                "sources": ["all_context_words"],
            })
    return units


def solve_source_target_vertex_cut(
    units: list[dict],
    source_edges: dict[str, float],
    interactions: dict[tuple[str, str], float],
    target_edges: dict[str, float],
) -> dict:
    """Solve the exact minimum-cost editable vertex cut between fixed terminals.

    Each editable context unit is split into an ``in -> out`` edge with its
    edit cost.  All contribution edges have infinite capacity, so the cut can
    contain only editable chunk tokens.  Model-derived contribution weights
    define the source-target support topology; they are not confused with
    token edit costs.
    """

    unit_by_id = {str(unit["unit_id"]): unit for unit in units}
    if not unit_by_id:
        return _empty_source_target_cut("no_editable_units")

    valid_source = {
        unit_id: float(weight)
        for unit_id, weight in source_edges.items()
        if unit_id in unit_by_id and float(weight) > 0
    }
    valid_target = {
        unit_id: float(weight)
        for unit_id, weight in target_edges.items()
        if unit_id in unit_by_id and float(weight) > 0
    }
    valid_interactions = {
        (left, right): float(weight)
        for (left, right), weight in interactions.items()
        if left in unit_by_id and right in unit_by_id and left != right and float(weight) > 0
    }
    if not valid_source or not valid_target:
        return _empty_source_target_cut("no_source_target_support")

    dinic = Dinic()
    source = dinic.node()
    target = dinic.node()
    split_nodes: dict[str, tuple[int, int]] = {}
    for unit_id, unit in unit_by_id.items():
        node_in = dinic.node()
        node_out = dinic.node()
        cost = float(unit.get("cost", 1.0))
        if cost <= 0:
            raise ValueError(f"editable unit {unit_id} has non-positive cost")
        dinic.add_edge(node_in, node_out, cost)
        split_nodes[unit_id] = (node_in, node_out)
    for unit_id in valid_source:
        dinic.add_edge(source, split_nodes[unit_id][0], INF)
    for (left, right) in valid_interactions:
        dinic.add_edge(split_nodes[left][1], split_nodes[right][0], INF)
    for unit_id in valid_target:
        dinic.add_edge(split_nodes[unit_id][1], target, INF)

    flow = dinic.max_flow(source, target)
    if flow <= 1e-9:
        return _empty_source_target_cut(
            "no_source_target_path",
            source_units=len(valid_source),
            target_units=len(valid_target),
            interaction_edges=len(valid_interactions),
        )
    reachable = dinic.reachable(source)
    selected_ids = sorted(
        unit_id
        for unit_id, (node_in, node_out) in split_nodes.items()
        if node_in in reachable and node_out not in reachable
    )
    selected_cost = sum(float(unit_by_id[unit_id].get("cost", 1.0)) for unit_id in selected_ids)
    return {
        "status": "optimal",
        "selected_ids": selected_ids,
        "cut_cost": selected_cost,
        "max_flow": flow,
        "source_units": len(valid_source),
        "target_units": len(valid_target),
        "interaction_edges": len(valid_interactions),
    }


def solve_contribution_weighted_vertex_cut(
    units: list[dict],
    source_edges: dict[str, float],
    interactions: dict[tuple[str, str], float],
    target_edges: dict[str, float],
) -> dict:
    """Solve a contribution-weighted, token-gated min-cut.

    Unlike the hard vertex cut, contribution edges are finite and retain their
    learned weights.  The cut therefore minimizes

        token-edit cost + residual contribution-edge cost.

    A token gate has unit cost.  A contribution edge has cost
    ``1 + weight / mean_weight``: the unit baseline makes a token gate
    preferable to cutting a positive support edge, while the normalized weight
    retains the relative importance of the information flow without an
    external lambda.
    Only token-gate edges are returned as editable tokens; contribution edges
    crossing the cut are reported as residual-support penalties.
    """

    unit_by_id = {str(unit["unit_id"]): unit for unit in units}
    if not unit_by_id:
        return _empty_source_target_cut("no_editable_units")
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
        return _empty_source_target_cut(
            "no_source_target_support",
            source_units=len(valid_source),
            target_units=len(valid_target),
            interaction_edges=len(valid_interactions),
        )

    contribution_mass = [*valid_source.values(), *valid_target.values(), *valid_interactions.values()]
    mean_edge_weight = sum(contribution_mass) / len(contribution_mass)
    if mean_edge_weight <= 1e-12:
        return _empty_source_target_cut("no_positive_contribution")

    dinic = Dinic()
    source = dinic.node()
    target = dinic.node()
    split_nodes: dict[str, tuple[int, int]] = {}
    for unit_id, unit in unit_by_id.items():
        node_in = dinic.node()
        node_out = dinic.node()
        cost = float(unit.get("cost", 1.0))
        if cost <= 0:
            raise ValueError(f"editable unit {unit_id} has non-positive cost")
        dinic.add_edge(node_in, node_out, cost)
        split_nodes[unit_id] = (node_in, node_out)

    contribution_arcs: list[tuple[int, int, float, str]] = []

    def add_contribution_arc(src: int, dst: int, weight: float, label: str) -> None:
        capacity = 1.0 + float(weight) / mean_edge_weight
        dinic.add_edge(src, dst, capacity)
        contribution_arcs.append((src, dst, capacity, label))

    for unit_id, weight in valid_source.items():
        add_contribution_arc(source, split_nodes[unit_id][0], weight, f"source:{unit_id}")
    for (left, right), weight in valid_interactions.items():
        add_contribution_arc(
            split_nodes[left][1],
            split_nodes[right][0],
            weight,
            f"interaction:{left}:{right}",
        )
    for unit_id, weight in valid_target.items():
        add_contribution_arc(split_nodes[unit_id][1], target, weight, f"target:{unit_id}")

    max_flow = dinic.max_flow(source, target)
    if max_flow <= 1e-9:
        return _empty_source_target_cut(
            "no_source_target_path",
            source_units=len(valid_source),
            target_units=len(valid_target),
            interaction_edges=len(valid_interactions),
        )

    reachable = dinic.reachable(source)
    selected_ids = sorted(
        unit_id
        for unit_id, (node_in, node_out) in split_nodes.items()
        if node_in in reachable and node_out not in reachable
    )
    selected_cost = sum(float(unit_by_id[unit_id].get("cost", 1.0)) for unit_id in selected_ids)
    residual_edge_cost = sum(
        capacity for src, dst, capacity, _ in contribution_arcs
        if src in reachable and dst not in reachable
    )
    return {
        "status": "optimal",
        "solver": "dinic_contribution_weighted_vertex_cut",
        "selected_ids": selected_ids,
        "selected_tokens": [str(unit_by_id[unit_id].get("text", "")) for unit_id in selected_ids],
        "cut_cost": float(selected_cost + residual_edge_cost),
        "token_cost": float(selected_cost),
        "residual_edge_cost": float(residual_edge_cost),
        "max_flow": float(max_flow),
        "mean_edge_weight": float(mean_edge_weight),
        "source_units": len(valid_source),
        "target_units": len(valid_target),
        "interaction_edges": len(valid_interactions),
        "n_selected": len(selected_ids),
    }


def solve_raw_token_gated_contribution_cut(
    graph_row: dict,
    units: list[dict],
) -> dict:
    """Cut a raw stage-aware contribution DAG with gates on input tokens.

    This keeps all positive closed-flow messages in their original stage-aware
    form.  Only input context roots are gated by editable tokens; therefore a
    token replacement is charged once even when its information is propagated
    through several attention, residual, or MLP stages.  Finite raw-edge costs
    retain contribution weights, while the returned selected gates are always
    legal chunk-token replacements.
    """

    unit_by_id = {str(unit["unit_id"]): unit for unit in units}
    if not unit_by_id:
        return _empty_source_target_cut("no_editable_units")
    graph = graph_row.get("graph", {})
    positive_edges: list[tuple[str, str, float, dict]] = []
    root_by_unit: dict[str, set[str]] = defaultdict(set)
    raw_nodes: set[str] = set()
    position_to_unit = _position_to_unit(graph_row, units)
    for edge in graph.get("edges", []):
        weight = max(0.0, float(edge.get("contribution", 0.0)))
        src = str(edge.get("src", ""))
        dst = str(edge.get("dst", ""))
        if weight <= 0 or not src or not dst or src.startswith("background::"):
            continue
        positive_edges.append((src, dst, weight, edge))
        raw_nodes.update((src, dst))
        if int(edge.get("src_layer", -1)) == 0:
            unit_id = position_to_unit.get(int(edge.get("src_position", -1)))
            if unit_id in unit_by_id:
                root_by_unit[unit_id].add(src)

    if not positive_edges or "answer_target" not in raw_nodes:
        return _empty_source_target_cut(
            "no_positive_context_flow",
            raw_edges=len(positive_edges),
            root_tokens=len(root_by_unit),
        )
    if not root_by_unit:
        return _empty_source_target_cut(
            "no_editable_context_root",
            raw_edges=len(positive_edges),
            root_tokens=0,
        )

    mean_edge_weight = sum(weight for _, _, weight, _ in positive_edges) / len(positive_edges)
    dinic = Dinic()
    source = dinic.node()
    sink = dinic.node()
    raw_node_ids = {node: dinic.node() for node in sorted(raw_nodes) if node != "answer_target"}
    gate_nodes: dict[str, tuple[int, int]] = {}
    for unit_id in sorted(root_by_unit):
        gate_in = dinic.node()
        gate_out = dinic.node()
        dinic.add_edge(gate_in, gate_out, float(unit_by_id[unit_id].get("cost", 1.0)))
        gate_nodes[unit_id] = (gate_in, gate_out)
        for root in root_by_unit[unit_id]:
            dinic.add_edge(gate_out, raw_node_ids[root], INF)

    raw_arcs: list[tuple[int, int, float, str]] = []
    for src, dst, weight, edge in positive_edges:
        if src == "answer_target":
            continue
        dst_node = sink if dst == "answer_target" else raw_node_ids[dst]
        capacity = 1.0 + weight / mean_edge_weight
        dinic.add_edge(raw_node_ids[src], dst_node, capacity)
        raw_arcs.append((raw_node_ids[src], dst_node, capacity, str(edge.get("kind", ""))))

    for unit_id, (gate_in, _gate_out) in gate_nodes.items():
        dinic.add_edge(source, gate_in, INF)

    max_flow = dinic.max_flow(source, sink)
    if max_flow <= 1e-9:
        return _empty_source_target_cut(
            "no_source_target_path",
            raw_edges=len(positive_edges),
            root_tokens=len(root_by_unit),
        )
    reachable = dinic.reachable(source)
    selected_ids = sorted(
        unit_id
        for unit_id, (gate_in, gate_out) in gate_nodes.items()
        if gate_in in reachable and gate_out not in reachable
    )
    selected_cost = sum(float(unit_by_id[unit_id].get("cost", 1.0)) for unit_id in selected_ids)
    residual_edge_cost = sum(
        capacity for src, dst, capacity, _ in raw_arcs
        if src in reachable and dst not in reachable
    )
    return {
        "status": "optimal",
        "solver": "dinic_raw_stage_token_gated_cut",
        "selected_ids": selected_ids,
        "selected_tokens": [str(unit_by_id[unit_id].get("text", "")) for unit_id in selected_ids],
        "cut_cost": float(selected_cost + residual_edge_cost),
        "token_cost": float(selected_cost),
        "residual_edge_cost": float(residual_edge_cost),
        "max_flow": float(max_flow),
        "mean_edge_weight": float(mean_edge_weight),
        "raw_edges": len(positive_edges),
        "raw_nodes": len(raw_nodes),
        "root_tokens": len(root_by_unit),
        "n_selected": len(selected_ids),
    }


def solve_raw_flow_traffic_token_cut(
    graph_row: dict,
    units: list[dict],
) -> dict:
    """Use raw positive flow to rank input-token gates, then cut token gates.

    All stage-aware contribution edges remain infinite-capacity in the cut
    network.  A first max-flow computes how much answer-directed flow each
    editable input token carries.  A second min-cut uses token-gate costs
    ``|V| + 1 - normalized_traffic``; consequently cardinality is the primary
    objective and the learned flow traffic is the tie-breaker.  No residual,
    MLP, or attention message can be selected as an edit.
    """

    unit_by_id = {str(unit["unit_id"]): unit for unit in units}
    if not unit_by_id:
        return _empty_source_target_cut("no_editable_units")
    graph = graph_row.get("graph", {})
    position_to_unit = _position_to_unit(graph_row, units)
    positive_edges: list[tuple[str, str, float]] = []
    root_by_unit: dict[str, set[str]] = defaultdict(set)
    raw_nodes: set[str] = set()
    for edge in graph.get("edges", []):
        weight = max(0.0, float(edge.get("contribution", 0.0)))
        src = str(edge.get("src", ""))
        dst = str(edge.get("dst", ""))
        if weight <= 0 or not src or not dst or src.startswith("background::"):
            continue
        positive_edges.append((src, dst, weight))
        raw_nodes.update((src, dst))
        if int(edge.get("src_layer", -1)) == 0:
            unit_id = position_to_unit.get(int(edge.get("src_position", -1)))
            if unit_id in unit_by_id:
                root_by_unit[unit_id].add(src)
    if not positive_edges or "answer_target" not in raw_nodes:
        return _empty_source_target_cut("no_positive_context_flow", raw_edges=len(positive_edges))
    if not root_by_unit:
        return _empty_source_target_cut("no_editable_context_root", raw_edges=len(positive_edges))

    def build_network(
        gate_costs: dict[str, float],
        *,
        weighted_edges: bool,
    ) -> tuple[Dinic, int, int, dict[str, tuple[int, int]], dict[str, tuple[int, int]]]:
        dinic = Dinic()
        source = dinic.node()
        sink = dinic.node()
        node_ids = {
            node: dinic.node()
            for node in sorted(raw_nodes)
            if node != "answer_target"
        }
        gates: dict[str, tuple[int, int]] = {}
        source_refs: dict[str, tuple[int, int]] = {}
        for unit_id in sorted(root_by_unit):
            gate_in = dinic.node()
            gate_out = dinic.node()
            gate_index = len(dinic.graph[gate_in])
            dinic.add_edge(gate_in, gate_out, float(gate_costs.get(unit_id, INF)))
            gates[unit_id] = (gate_in, gate_out)
            for root in root_by_unit[unit_id]:
                dinic.add_edge(gate_out, node_ids[root], INF)
            source_index = len(dinic.graph[source])
            dinic.add_edge(source, gate_in, INF)
            source_refs[unit_id] = (source, source_index)
        for src, dst, _weight in positive_edges:
            if src == "answer_target":
                continue
            dinic.add_edge(
                node_ids[src],
                sink if dst == "answer_target" else node_ids[dst],
                _weight if weighted_edges else INF,
            )
        return dinic, source, sink, gates, source_refs

    initial, source, sink, _, source_refs = build_network({}, weighted_edges=True)
    maximum_flow = initial.max_flow(source, sink)
    if maximum_flow <= 1e-9:
        return _empty_source_target_cut(
            "no_source_target_path",
            raw_edges=len(positive_edges),
            root_tokens=len(root_by_unit),
        )
    traffic = {}
    for unit_id, (node, edge_index) in source_refs.items():
        edge = initial.graph[node][edge_index]
        traffic[unit_id] = float(edge.original - edge.cap)
    max_traffic = max(traffic.values(), default=0.0)
    scale = float(len(unit_by_id) + 1)
    gate_costs = {
        unit_id: scale - (traffic.get(unit_id, 0.0) / max_traffic if max_traffic > 1e-12 else 0.0)
        for unit_id in root_by_unit
    }
    cut_network, cut_source, cut_sink, gates, _ = build_network(gate_costs, weighted_edges=False)
    cut_value = cut_network.max_flow(cut_source, cut_sink)
    reachable = cut_network.reachable(cut_source)
    selected_ids = sorted(
        unit_id
        for unit_id, (gate_in, gate_out) in gates.items()
        if gate_in in reachable and gate_out not in reachable
    )
    return {
        "status": "optimal",
        "solver": "raw_maxflow_traffic_token_gate_cut",
        "selected_ids": selected_ids,
        "selected_tokens": [str(unit_by_id[unit_id].get("text", "")) for unit_id in selected_ids],
        "cut_cost": float(cut_value),
        "selected_cardinality": len(selected_ids),
        "maximum_flow": float(maximum_flow),
        "selected_traffic": float(sum(traffic.get(unit_id, 0.0) for unit_id in selected_ids)),
        "token_traffic": traffic,
        "cardinality_scale": scale,
        "raw_edges": len(positive_edges),
        "raw_nodes": len(raw_nodes),
        "root_tokens": len(root_by_unit),
        "n_selected": len(selected_ids),
    }


def solve_flow_aware_lexicographic_vertex_cut(
    units: list[dict],
    source_edges: dict[str, float],
    interactions: dict[tuple[str, str], float],
    target_edges: dict[str, float],
) -> dict:
    """Find a minimum token separator, then maximize carried support flow.

    The first coordinate is the number of editable token gates.  Among cuts
    with identical cardinality, the second coordinate is the sum of token
    traffic in one maximum contribution flow.  A gate capacity ``M-a_v``
    realizes this lexicographic objective because ``M=|V|+1`` dominates the
    normalized flow bonus ``a_v in [0, 1]``.
    """

    witness_flow = decompose_source_target_maxflow_witnesses(
        units,
        source_edges,
        interactions,
        target_edges,
    )
    if witness_flow.get("status") != "optimal":
        return {
            **witness_flow,
            "solver": "maxflow_then_lexicographic_vertex_cut",
        }

    traffic: dict[str, float] = defaultdict(float)
    for witness in witness_flow["witnesses"]:
        path_flow = float(witness["flow"])
        for unit_id in witness["unit_ids"]:
            traffic[str(unit_id)] += path_flow
    maximum_traffic = max(traffic.values(), default=0.0)
    n_units = len(units)
    cardinality_scale = float(n_units + 1)
    weighted_units = []
    for unit in units:
        unit_id = str(unit["unit_id"])
        normalized_traffic = (
            traffic.get(unit_id, 0.0) / maximum_traffic
            if maximum_traffic > 1e-12
            else 0.0
        )
        weighted_units.append({
            **unit,
            "cost": cardinality_scale - normalized_traffic,
        })

    cut = solve_source_target_vertex_cut(
        weighted_units,
        source_edges,
        interactions,
        target_edges,
    )
    selected_ids = list(cut.get("selected_ids", []))
    return {
        **cut,
        "selected_cardinality": len(selected_ids),
        "selected_traffic": sum(traffic.get(unit_id, 0.0) for unit_id in selected_ids),
        "maximum_flow": float(witness_flow["max_flow"]),
        "n_flow_routes": int(witness_flow["n_witnesses"]),
        "token_traffic": traffic,
        "cardinality_scale": cardinality_scale,
        "solver": "maxflow_then_lexicographic_vertex_cut",
    }


def solve_residual_maxflow_ratio(
    units: list[dict],
    source_edges: dict[str, float],
    interactions: dict[tuple[str, str], float],
    target_edges: dict[str, float],
    *,
    max_iterations: int = 100,
    tolerance: float = 1e-9,
) -> dict:
    """Maximize blocked contribution flow per edited token without path expansion.

    For a token set ``S``, let ``Phi(S)`` be the residual maximum flow after
    setting the gates of tokens in ``S`` to zero. The objective is

        max_{S != empty} (Phi(empty) - Phi(S)) / cost(S).

    At a fixed Dinkelbach price ``lambda``, the exact subproblem minimizes
    ``Phi(S) + lambda * cost(S)``. It is represented by one flow network whose
    token-gate capacities are ``lambda * cost`` and whose contribution-edge
    capacities are the learned graph weights. No flow-path decomposition is
    constructed.
    """

    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    unit_by_id = {str(unit["unit_id"]): unit for unit in units}
    if not unit_by_id:
        return _empty_source_target_cut("no_editable_units")
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
        return _empty_source_target_cut(
            "no_source_target_support",
            source_units=len(valid_source),
            target_units=len(valid_target),
            interaction_edges=len(valid_interactions),
        )

    def build_network(
        *,
        gate_price: float | None = None,
        removed: frozenset[str] = frozenset(),
    ) -> tuple[Dinic, int, int, dict[str, tuple[int, int]]]:
        dinic = Dinic()
        source = dinic.node()
        target = dinic.node()
        gates = {}
        for unit_id, unit in unit_by_id.items():
            node_in = dinic.node()
            node_out = dinic.node()
            if gate_price is None:
                capacity = 0.0 if unit_id in removed else INF
            else:
                capacity = gate_price * float(unit.get("cost", 1.0))
            dinic.add_edge(node_in, node_out, capacity)
            gates[unit_id] = (node_in, node_out)
        for unit_id, weight in valid_source.items():
            dinic.add_edge(source, gates[unit_id][0], weight)
        for (left, right), weight in valid_interactions.items():
            dinic.add_edge(gates[left][1], gates[right][0], weight)
        for unit_id, weight in valid_target.items():
            dinic.add_edge(gates[unit_id][1], target, weight)
        return dinic, source, target, gates

    def residual_flow(selected: frozenset[str]) -> float:
        dinic, source, target, _ = build_network(removed=selected)
        return float(dinic.max_flow(source, target))

    def priced_cut(price: float) -> tuple[list[str], float]:
        dinic, source, target, gates = build_network(gate_price=price)
        objective = float(dinic.max_flow(source, target))
        reachable = dinic.reachable(source)
        selected = sorted(
            unit_id
            for unit_id, (node_in, node_out) in gates.items()
            if node_in in reachable and node_out not in reachable
        )
        return selected, objective

    initial_flow = residual_flow(frozenset())
    if initial_flow <= tolerance:
        return _empty_source_target_cut(
            "no_positive_context_flow",
            source_units=len(valid_source),
            target_units=len(valid_target),
            interaction_edges=len(valid_interactions),
        )

    price = 0.0
    best_ids: list[str] = []
    best_density = 0.0
    best_residual = initial_flow
    history = []
    for iteration in range(1, max_iterations + 1):
        selected_ids, priced_objective = priced_cut(price)
        if not selected_ids:
            history.append({
                "iteration": iteration,
                "lambda": price,
                "selected_count": 0,
                "priced_objective": priced_objective,
                "residual_flow": initial_flow,
                "flow_reduction": 0.0,
                "density": 0.0,
                "dinkelbach_gap": 0.0,
            })
            break
        selected = frozenset(selected_ids)
        residual = residual_flow(selected)
        reduction = max(0.0, initial_flow - residual)
        selected_cost = sum(
            float(unit_by_id[unit_id].get("cost", 1.0))
            for unit_id in selected_ids
        )
        if selected_cost <= 0:
            raise ValueError("selected token set has non-positive total cost")
        density = reduction / selected_cost
        gap = reduction - price * selected_cost
        history.append({
            "iteration": iteration,
            "lambda": price,
            "selected_count": len(selected_ids),
            "selected_cost": selected_cost,
            "priced_objective": priced_objective,
            "residual_flow": residual,
            "flow_reduction": reduction,
            "density": density,
            "dinkelbach_gap": gap,
        })
        if density > best_density + tolerance or (
            abs(density - best_density) <= tolerance
            and selected_cost
            < sum(
                float(unit_by_id[unit_id].get("cost", 1.0))
                for unit_id in best_ids
            )
        ):
            best_ids = selected_ids
            best_density = density
            best_residual = residual
        if gap <= tolerance or abs(density - price) <= tolerance:
            break
        price = density

    return {
        "status": "optimal" if best_ids else "no_positive_reduction",
        "solver": "dinkelbach_residual_maxflow_ratio",
        "selected_ids": best_ids,
        "selected_tokens": [
            str(unit_by_id[unit_id].get("text", "")) for unit_id in best_ids
        ],
        "n_selected": len(best_ids),
        "initial_flow": initial_flow,
        "residual_flow": best_residual,
        "flow_reduction": initial_flow - best_residual,
        "density": best_density,
        "iterations": len(history),
        "history": history,
        "uses_path_decomposition": False,
        "source_units": len(valid_source),
        "target_units": len(valid_target),
        "interaction_edges": len(valid_interactions),
    }


def solve_maxflow_surplus_intervention(
    units: list[dict],
    source_edges: dict[str, float],
    interactions: dict[tuple[str, str], float],
    target_edges: dict[str, float],
    *,
    tolerance: float = 1e-9,
    contribution_aware_gates: bool = False,
    gate_scores: dict[str, float] | None = None,
) -> dict:
    """Select tokens whose flow reduction exceeds the full-cut average.

    Let ``Phi(S)`` be the residual maximum contribution flow after removing
    token set ``S``, ``Phi_0 = Phi(empty)``, and ``kappa`` be the minimum
    cardinality of a token vertex cut.  The graph-derived price

        lambda_q = Phi_0 / kappa

    is the average flow reduction required from each token in a complete cut.
    This routine exactly maximizes the flow surplus

        Phi_0 - Phi(S) - lambda_q * |S|

    through one token-gated minimum cut. When ``contribution_aware_gates`` is
    enabled, the same max-flow construction replaces the uniform gate price
    with a graph-derived gate price that decreases with a token's incident
    contribution. ``gate_scores`` may instead provide an externally computed
    non-negative effectiveness score for each token; this is used by the
    replacement-calibrated ablation without changing the default solver. If no
    proper set has positive
    surplus, the minimum vertex cut is returned: the graph contains no
    strictly above-average partial intervention under this objective.
    """

    unit_by_id = {str(unit["unit_id"]): unit for unit in units}
    if not unit_by_id:
        return _empty_source_target_cut("no_editable_units")
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
        return _empty_source_target_cut(
            "no_source_target_support",
            source_units=len(valid_source),
            target_units=len(valid_target),
            interaction_edges=len(valid_interactions),
        )

    hard_cut = solve_source_target_vertex_cut(
        units,
        valid_source,
        valid_interactions,
        valid_target,
    )
    if hard_cut.get("status") != "optimal":
        return {
            **hard_cut,
            "solver": "maxflow_surplus_intervention",
        }
    hard_ids = sorted(set(hard_cut.get("selected_ids", [])))
    kappa = sum(
        float(unit_by_id[unit_id].get("cost", 1.0))
        for unit_id in hard_ids
    )
    if kappa <= tolerance:
        return _empty_source_target_cut("zero_cardinality_cut")

    def build_network(
        *,
        gate_capacity: float,
        gate_capacities: dict[str, float] | None = None,
        removed: frozenset[str] = frozenset(),
    ) -> tuple[Dinic, int, int, dict[str, tuple[int, int]]]:
        dinic = Dinic()
        source = dinic.node()
        target = dinic.node()
        gates: dict[str, tuple[int, int]] = {}
        for unit_id, unit in unit_by_id.items():
            node_in = dinic.node()
            node_out = dinic.node()
            capacity = (
                0.0
                if unit_id in removed
                else (
                    float(gate_capacities[unit_id])
                    if gate_capacities is not None
                    else gate_capacity * float(unit.get("cost", 1.0))
                )
            )
            dinic.add_edge(node_in, node_out, capacity)
            gates[unit_id] = (node_in, node_out)
        for unit_id, weight in valid_source.items():
            dinic.add_edge(source, gates[unit_id][0], weight)
        for (left, right), weight in valid_interactions.items():
            dinic.add_edge(gates[left][1], gates[right][0], weight)
        for unit_id, weight in valid_target.items():
            dinic.add_edge(gates[unit_id][1], target, weight)
        return dinic, source, target, gates

    def residual_flow(selected: frozenset[str]) -> float:
        dinic, source, target, _ = build_network(
            gate_capacity=INF,
            removed=selected,
        )
        return float(dinic.max_flow(source, target))

    initial_flow = residual_flow(frozenset())
    if initial_flow <= tolerance:
        return _empty_source_target_cut("no_positive_context_flow")
    token_price = initial_flow / kappa

    # Assign every unit of graph contribution exactly once. Source/terminal
    # edges belong to their token endpoint, while an internal token-token edge
    # is shared equally by its two endpoints.
    incident_contribution = {
        unit_id: valid_source.get(unit_id, 0.0) + valid_target.get(unit_id, 0.0)
        for unit_id in unit_by_id
    }
    for (left, right), weight in valid_interactions.items():
        incident_contribution[left] += 0.5 * weight
        incident_contribution[right] += 0.5 * weight
    positive_incident = [
        value for value in incident_contribution.values() if value > tolerance
    ]
    mean_incident = (
        sum(positive_incident) / len(positive_incident)
        if positive_incident
        else 0.0
    )
    nonnegative_gate_scores = (
        {
            unit_id: max(0.0, float(gate_scores.get(unit_id, 0.0)))
            for unit_id in unit_by_id
        }
        if gate_scores is not None
        else None
    )
    positive_gate_scores = (
        [
            value
            for value in nonnegative_gate_scores.values()
            if value > tolerance
        ]
        if nonnegative_gate_scores is not None
        else []
    )
    mean_gate_score = (
        sum(positive_gate_scores) / len(positive_gate_scores)
        if positive_gate_scores
        else 0.0
    )
    if nonnegative_gate_scores is not None and mean_gate_score > tolerance:
        gate_capacities = {
            unit_id: token_price
            * mean_gate_score
            / (mean_gate_score + nonnegative_gate_scores[unit_id])
            for unit_id in unit_by_id
        }
        gate_cost_mode = "external_effect_score"
    elif contribution_aware_gates and mean_incident > tolerance:
        gate_capacities = {
            unit_id: token_price * mean_incident / (mean_incident + contribution)
            for unit_id, contribution in incident_contribution.items()
        }
        gate_cost_mode = "conserved_incident_contribution"
    else:
        gate_capacities = None
        gate_cost_mode = "uniform"

    dinic, source, target, gates = build_network(
        gate_capacity=token_price,
        gate_capacities=gate_capacities,
    )
    priced_objective = float(dinic.max_flow(source, target))
    reachable = dinic.reachable(source)
    selected_ids = sorted(
        unit_id
        for unit_id, (node_in, node_out) in gates.items()
        if node_in in reachable and node_out not in reachable
    )
    selected_cost = sum(
        float(unit_by_id[unit_id].get("cost", 1.0))
        for unit_id in selected_ids
    )
    selected_gate_penalty = sum(
        float(gate_capacities[unit_id])
        if gate_capacities is not None
        else token_price * float(unit_by_id[unit_id].get("cost", 1.0))
        for unit_id in selected_ids
    )
    residual = residual_flow(frozenset(selected_ids))
    reduction = max(0.0, initial_flow - residual)
    surplus = reduction - selected_gate_penalty

    fallback_to_cut = surplus <= tolerance or not selected_ids
    if fallback_to_cut:
        selected_ids = hard_ids
        selected_cost = kappa
        selected_gate_penalty = sum(
            float(gate_capacities[unit_id])
            if gate_capacities is not None
            else token_price * float(unit_by_id[unit_id].get("cost", 1.0))
            for unit_id in selected_ids
        )
        residual = 0.0
        reduction = initial_flow
        surplus = 0.0

    return {
        "status": "optimal",
        "solver": "maxflow_surplus_intervention",
        "selected_ids": selected_ids,
        "selected_tokens": [
            str(unit_by_id[unit_id].get("text", "")) for unit_id in selected_ids
        ],
        "n_selected": len(selected_ids),
        "selected_cost": selected_cost,
        "selected_gate_penalty": selected_gate_penalty,
        "initial_flow": initial_flow,
        "residual_flow": residual,
        "flow_reduction": reduction,
        "flow_reduction_fraction": reduction / initial_flow,
        "token_price": token_price,
        "gate_cost_mode": gate_cost_mode,
        "mean_gate_score": mean_gate_score,
        "mean_incident_contribution": mean_incident,
        "mean_selected_incident_contribution": (
            sum(incident_contribution[unit_id] for unit_id in selected_ids)
            / max(1, len(selected_ids))
        ),
        "vertex_cut_cardinality": kappa,
        "hard_cut_ids": hard_ids,
        "hard_cut_size": len(hard_ids),
        "flow_surplus": surplus,
        "fallback_to_hard_cut": fallback_to_cut,
        "priced_objective": priced_objective,
        "uses_external_threshold": False,
        "source_units": len(valid_source),
        "target_units": len(valid_target),
        "interaction_edges": len(valid_interactions),
    }


def solve_gate_calibrated_lagrangian_intervention(
    units: list[dict],
    source_edges: dict[str, float],
    interactions: dict[tuple[str, str], float],
    target_edges: dict[str, float],
    *,
    eta: float = 1.0,
    gamma: float = 1.0,
    tolerance: float = 1e-9,
) -> dict:
    """Run the geometric Lagrangian search with a cut-derived residual budget.

    The contribution-aware flow cut supplies a reference intervention without
    an externally chosen flow budget.  If its residual flow is ``F_gate`` and
    the original flow is ``F_max``, this routine uses

        beta_q = F_gate / F_max.

    The old Lagrangian objective and its geometric cardinality guesses are
    otherwise unchanged.  Thus the cut calibrates the *instance-specific*
    residual target rather than becoming an extra heuristic selector.
    """

    if eta < 0:
        raise ValueError("eta must be nonnegative")
    if gamma <= 0:
        raise ValueError("gamma must be positive")

    reference = solve_maxflow_surplus_intervention(
        units,
        source_edges,
        interactions,
        target_edges,
        tolerance=tolerance,
        contribution_aware_gates=True,
    )
    if reference.get("status") != "optimal":
        return {**reference, "solver": "gate_calibrated_lagrangian"}

    unit_by_id = {str(unit["unit_id"]): unit for unit in units}
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

    def build_network(
        *,
        gate_capacity: float,
        removed: frozenset[str] = frozenset(),
    ) -> tuple[Dinic, int, int, dict[str, tuple[int, int]]]:
        dinic = Dinic()
        source = dinic.node()
        target = dinic.node()
        gates: dict[str, tuple[int, int]] = {}
        for unit_id, unit in unit_by_id.items():
            node_in = dinic.node()
            node_out = dinic.node()
            capacity = (
                0.0
                if unit_id in removed
                else gate_capacity * float(unit.get("cost", 1.0))
            )
            dinic.add_edge(node_in, node_out, capacity)
            gates[unit_id] = (node_in, node_out)
        for unit_id, weight in valid_source.items():
            dinic.add_edge(source, gates[unit_id][0], weight)
        for (left, right), weight in valid_interactions.items():
            dinic.add_edge(gates[left][1], gates[right][0], weight)
        for unit_id, weight in valid_target.items():
            dinic.add_edge(gates[unit_id][1], target, weight)
        return dinic, source, target, gates

    def residual_flow(selected: frozenset[str]) -> float:
        dinic, source, target, _ = build_network(
            gate_capacity=INF,
            removed=selected,
        )
        return float(dinic.max_flow(source, target))

    initial_flow = float(reference.get("initial_flow", 0.0))
    gate_residual = float(reference.get("residual_flow", 0.0))
    if initial_flow <= tolerance:
        return _empty_source_target_cut("no_positive_context_flow")
    if gate_residual <= tolerance:
        return {
            **reference,
            "solver": "gate_calibrated_lagrangian",
            "beta_q": 0.0,
            "residual_budget": 0.0,
            "calibration_selected_ids": list(reference.get("selected_ids", [])),
            "calibration_n_selected": int(reference.get("n_selected", 0)),
            "calibration_is_exact_cut": True,
        }

    beta_q = gate_residual / initial_flow
    residual_budget = (1.0 + eta) * gate_residual
    guesses: list[float] = []
    cardinality = 1.0
    n_units = len(unit_by_id)
    while cardinality < n_units:
        guesses.append(cardinality)
        cardinality *= 1.0 + gamma
    guesses.append(float(n_units))

    candidates: list[tuple[list[str], float, float, float]] = []
    for guess in sorted(set(guesses)):
        # This is the original prize-collecting cut at the cut-derived beta.
        gate_price = eta * gate_residual / guess
        dinic, source, target, gates = build_network(gate_capacity=gate_price)
        priced_objective = float(dinic.max_flow(source, target))
        reachable = dinic.reachable(source)
        selected_ids = sorted(
            unit_id
            for unit_id, (node_in, node_out) in gates.items()
            if node_in in reachable and node_out not in reachable
        )
        remaining = residual_flow(frozenset(selected_ids))
        candidates.append((selected_ids, remaining, guess, priced_objective))

    feasible = [candidate for candidate in candidates if candidate[1] <= residual_budget + tolerance]
    if feasible:
        selected_ids, residual, guess, priced_objective = min(
            feasible,
            key=lambda candidate: (len(candidate[0]), candidate[1], candidate[0]),
        )
        feasible_at_budget = True
    else:
        selected_ids, residual, guess, priced_objective = min(
            candidates,
            key=lambda candidate: (candidate[1], len(candidate[0]), candidate[0]),
        )
        feasible_at_budget = False

    return {
        "status": "optimal",
        "solver": "gate_calibrated_lagrangian",
        "selected_ids": selected_ids,
        "selected_tokens": [
            str(unit_by_id[unit_id].get("text", "")) for unit_id in selected_ids
        ],
        "n_selected": len(selected_ids),
        "initial_flow": initial_flow,
        "residual_flow": residual,
        "flow_reduction": max(0.0, initial_flow - residual),
        "flow_reduction_fraction": max(0.0, initial_flow - residual) / initial_flow,
        "beta_q": beta_q,
        "residual_budget": residual_budget,
        "eta": eta,
        "gamma": gamma,
        "selected_cardinality_guess": guess,
        "priced_objective": priced_objective,
        "feasible_at_budget": feasible_at_budget,
        "calibration_selected_ids": list(reference.get("selected_ids", [])),
        "calibration_n_selected": int(reference.get("n_selected", 0)),
        "calibration_residual_flow": gate_residual,
        "calibration_gate_cost_mode": reference.get("gate_cost_mode"),
        "uses_external_threshold": False,
        "source_units": len(valid_source),
        "target_units": len(valid_target),
        "interaction_edges": len(valid_interactions),
    }


def solve_weighted_edge_disjoint_backbone_cover(
    units: list[dict],
    source_edges: dict[str, float],
    interactions: dict[tuple[str, str], float],
    target_edges: dict[str, float],
) -> dict:
    """Cover every route in a contribution-aware edge-disjoint backbone.

    Contribution arcs have unit capacity, so the first phase computes a
    maximum-cardinality family of edge-disjoint source-target routes.  Arc
    costs are negative log normalized contributions; among maximum-cardinality
    families, min-cost max-flow therefore favors routes made from stronger
    contribution edges.  The second phase repeatedly selects the chunk token
    covering the largest remaining bottleneck-weighted route mass until every
    backbone route is hit.  No budget or residual-flow threshold is used.
    """

    unit_by_id = {str(unit["unit_id"]): unit for unit in units}
    if not unit_by_id:
        return _empty_source_target_cut("no_editable_units")
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
        return _empty_source_target_cut(
            "no_source_target_support",
            source_units=len(valid_source),
            target_units=len(valid_target),
            interaction_edges=len(valid_interactions),
        )

    all_weights = [
        *valid_source.values(),
        *valid_target.values(),
        *valid_interactions.values(),
    ]
    max_weight = max(all_weights, default=0.0)
    if max_weight <= 0:
        return _empty_source_target_cut("no_positive_contribution")

    network = _MinCostUnitFlow()
    source = network.node()
    target = network.node()
    token_node = {unit_id: network.node() for unit_id in sorted(unit_by_id)}
    node_token = {node: unit_id for unit_id, node in token_node.items()}

    def contribution_cost(weight: float) -> float:
        normalized = min(1.0, max(float(weight) / max_weight, 1e-15))
        return -math.log(normalized)

    for unit_id, weight in valid_source.items():
        network.add_edge(
            source,
            token_node[unit_id],
            contribution_cost(weight),
            weight,
        )
    for (left, right), weight in valid_interactions.items():
        network.add_edge(
            token_node[left],
            token_node[right],
            contribution_cost(weight),
            weight,
        )
    for unit_id, weight in valid_target.items():
        network.add_edge(
            token_node[unit_id],
            target,
            contribution_cost(weight),
            weight,
        )

    flow_value, flow_cost = network.min_cost_max_flow(source, target)
    if flow_value <= 0:
        return _empty_source_target_cut(
            "no_source_target_path",
            source_units=len(valid_source),
            target_units=len(valid_target),
            interaction_edges=len(valid_interactions),
        )

    carried: dict[tuple[int, int], int] = {}
    for node, edges in enumerate(network.graph):
        for edge_index, edge in enumerate(edges):
            if edge.original > 0 and edge.cap < edge.original:
                carried[(node, edge_index)] = edge.original - edge.cap

    def find_carried_path(
        node: int,
        visited: set[int],
    ) -> list[tuple[int, int]] | None:
        if node == target:
            return []
        visited.add(node)
        for edge_index, edge in enumerate(network.graph[node]):
            if carried.get((node, edge_index), 0) <= 0 or edge.to in visited:
                continue
            suffix = find_carried_path(edge.to, visited)
            if suffix is not None:
                visited.remove(node)
                return [(node, edge_index), *suffix]
        visited.remove(node)
        return None

    routes = []
    while True:
        edge_path = find_carried_path(source, set())
        if edge_path is None:
            break
        token_path = []
        edge_weights = []
        path_cost = 0.0
        for node, edge_index in edge_path:
            edge = network.graph[node][edge_index]
            carried[(node, edge_index)] -= 1
            if edge.to in node_token:
                token_path.append(node_token[edge.to])
            edge_weights.append(edge.weight)
            path_cost += edge.cost
        if token_path:
            routes.append({
                "unit_ids": token_path,
                "weight": min(edge_weights),
                "cost": path_cost,
                "edge_count": len(edge_path),
            })

    if not routes:
        return _empty_source_target_cut(
            "no_decomposable_backbone_route",
            source_units=len(valid_source),
            target_units=len(valid_target),
            interaction_edges=len(valid_interactions),
        )

    uncovered = set(range(len(routes)))
    selected_ids = []
    coverage_trace = []
    while uncovered:
        gains: dict[str, float] = defaultdict(float)
        route_counts: dict[str, int] = defaultdict(int)
        for route_index in uncovered:
            route = routes[route_index]
            for unit_id in set(route["unit_ids"]):
                gains[unit_id] += float(route["weight"])
                route_counts[unit_id] += 1
        if not gains:
            break
        selected = max(
            gains,
            key=lambda unit_id: (
                gains[unit_id] / float(unit_by_id[unit_id].get("cost", 1.0)),
                route_counts[unit_id],
                float(unit_by_id[unit_id].get("support", 0.0)),
                unit_id,
            ),
        )
        covered_now = {
            route_index
            for route_index in uncovered
            if selected in routes[route_index]["unit_ids"]
        }
        selected_ids.append(selected)
        coverage_trace.append({
            "unit_id": selected,
            "gain": gains[selected],
            "routes_covered": len(covered_now),
            "routes_remaining": len(uncovered) - len(covered_now),
        })
        uncovered.difference_update(covered_now)

    total_route_weight = sum(float(route["weight"]) for route in routes)
    return {
        "status": "optimal" if not uncovered else "incomplete_cover",
        "solver": "min_cost_edge_disjoint_backbone_greedy_cover",
        "selected_ids": selected_ids,
        "selected_tokens": [
            str(unit_by_id[unit_id].get("text", "")) for unit_id in selected_ids
        ],
        "n_selected": len(selected_ids),
        "edge_disjoint_routes": len(routes),
        "unit_max_flow": flow_value,
        "min_cost_flow_cost": flow_cost,
        "total_route_weight": total_route_weight,
        "covered_route_weight": total_route_weight if not uncovered else (
            total_route_weight
            - sum(float(routes[index]["weight"]) for index in uncovered)
        ),
        "routes": routes,
        "coverage_trace": coverage_trace,
        "uses_path_decomposition": True,
        "uses_external_threshold": False,
        "source_units": len(valid_source),
        "target_units": len(valid_target),
        "interaction_edges": len(valid_interactions),
    }


def decompose_source_target_maxflow_witnesses(
    units: list[dict],
    source_edges: dict[str, float],
    interactions: dict[tuple[str, str], float],
    target_edges: dict[str, float],
    *,
    selected_ids: set[str] | frozenset[str] = frozenset(),
) -> dict:
    """Decompose the residual maximum flow into editable token paths."""

    unit_by_id = {str(unit["unit_id"]): unit for unit in units}
    selected = set(selected_ids) & set(unit_by_id)
    if not unit_by_id:
        return _empty_flow_witnesses("no_editable_units")

    valid_source = {
        unit_id: float(weight)
        for unit_id, weight in source_edges.items()
        if unit_id in unit_by_id and float(weight) > 0
    }
    valid_target = {
        unit_id: float(weight)
        for unit_id, weight in target_edges.items()
        if unit_id in unit_by_id and float(weight) > 0
    }
    valid_interactions = {
        (left, right): float(weight)
        for (left, right), weight in interactions.items()
        if (
            left in unit_by_id
            and right in unit_by_id
            and left != right
            and float(weight) > 0
        )
    }
    if not valid_source or not valid_target:
        return _empty_flow_witnesses(
            "no_source_target_support",
            source_units=len(valid_source),
            target_units=len(valid_target),
            interaction_edges=len(valid_interactions),
        )

    dinic = Dinic()
    source = dinic.node()
    target = dinic.node()
    split_nodes: dict[str, tuple[int, int]] = {}
    gate_unit: dict[tuple[int, int], str] = {}
    for unit_id in sorted(unit_by_id):
        node_in = dinic.node()
        node_out = dinic.node()
        dinic.add_edge(node_in, node_out, 0.0 if unit_id in selected else INF)
        split_nodes[unit_id] = (node_in, node_out)
        gate_unit[(node_in, node_out)] = unit_id
    for unit_id, weight in valid_source.items():
        dinic.add_edge(source, split_nodes[unit_id][0], weight)
    for (left, right), weight in valid_interactions.items():
        dinic.add_edge(split_nodes[left][1], split_nodes[right][0], weight)
    for unit_id, weight in valid_target.items():
        dinic.add_edge(split_nodes[unit_id][1], target, weight)

    max_flow = dinic.max_flow(source, target)
    if max_flow <= 1e-9:
        return _empty_flow_witnesses(
            "no_residual_source_target_flow",
            source_units=len(valid_source),
            target_units=len(valid_target),
            interaction_edges=len(valid_interactions),
        )

    remaining: dict[tuple[int, int], float] = {}
    for node, edges in enumerate(dinic.graph):
        for edge_index, edge in enumerate(edges):
            if edge.original <= 0:
                continue
            carried = edge.original - edge.cap
            if carried > 1e-9:
                remaining[(node, edge_index)] = carried

    def find_positive_path(
        node: int,
        visited: set[int],
    ) -> list[tuple[int, int]] | None:
        if node == target:
            return []
        visited.add(node)
        for edge_index, edge in enumerate(dinic.graph[node]):
            if remaining.get((node, edge_index), 0.0) <= 1e-9:
                continue
            if edge.to in visited:
                continue
            suffix = find_positive_path(edge.to, visited)
            if suffix is not None:
                visited.remove(node)
                return [(node, edge_index), *suffix]
        visited.remove(node)
        return None

    path_flow: dict[tuple[str, ...], float] = defaultdict(float)
    while True:
        edge_path = find_positive_path(source, set())
        if edge_path is None:
            break
        amount = min(remaining[edge_key] for edge_key in edge_path)
        token_path = []
        for node, edge_index in edge_path:
            edge = dinic.graph[node][edge_index]
            unit_id = gate_unit.get((node, edge.to))
            if unit_id is not None:
                token_path.append(unit_id)
            remaining[(node, edge_index)] -= amount
        if token_path:
            path_flow[tuple(token_path)] += amount

    witnesses = [
        {"unit_ids": list(path), "flow": flow}
        for path, flow in sorted(
            path_flow.items(),
            key=lambda item: (-item[1], len(item[0]), item[0]),
        )
        if flow > 1e-9
    ]
    return {
        "status": "optimal",
        "max_flow": max_flow,
        "decomposed_flow": sum(row["flow"] for row in witnesses),
        "witnesses": witnesses,
        "n_witnesses": len(witnesses),
        "source_units": len(valid_source),
        "target_units": len(valid_target),
        "interaction_edges": len(valid_interactions),
    }


def solve_maxflow_route_coverage_ratio(
    units: list[dict],
    source_edges: dict[str, float],
    interactions: dict[tuple[str, str], float],
    target_edges: dict[str, float],
    *,
    tolerance: float = 1e-9,
) -> dict:
    """Exactly maximize fixed-maxflow route coverage per token cost.

    A maximum flow is first decomposed into weighted source-target routes.
    For a token set ``T``, ``H(T)`` is the total weight of routes containing
    at least one token in ``T``.  This weighted coverage function is normalized,
    monotone, and submodular.  Consequently,

        H(T) / cost(T) <= max_{i in T} H({i}) / cost(i),

    so an optimal nonempty solution is always a singleton.  We compute that
    exact optimum directly rather than running an unnecessary ratio iteration.
    """

    unit_by_id = {str(unit["unit_id"]): unit for unit in units}
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
            **_empty_flow_witnesses(
                "no_source_target_support",
                source_units=len(valid_source),
                target_units=len(valid_target),
                interaction_edges=len(valid_interactions),
            ),
            "solver": "exact_maxflow_route_coverage_ratio",
            "selected_ids": [],
            "n_selected": 0,
        }

    dinic = Dinic()
    source = dinic.node()
    target = dinic.node()
    gate_edges: dict[str, tuple[int, int]] = {}
    flow_upper_bound = sum(valid_source.values())
    gate_capacity = flow_upper_bound + 1.0
    split_nodes: dict[str, tuple[int, int]] = {}
    for unit_id in sorted(unit_by_id):
        node_in = dinic.node()
        node_out = dinic.node()
        edge_index = len(dinic.graph[node_in])
        dinic.add_edge(node_in, node_out, gate_capacity)
        gate_edges[unit_id] = (node_in, edge_index)
        split_nodes[unit_id] = (node_in, node_out)
    for unit_id, weight in valid_source.items():
        dinic.add_edge(source, split_nodes[unit_id][0], weight)
    for (left, right), weight in valid_interactions.items():
        dinic.add_edge(split_nodes[left][1], split_nodes[right][0], weight)
    for unit_id, weight in valid_target.items():
        dinic.add_edge(split_nodes[unit_id][1], target, weight)

    max_flow = float(dinic.max_flow(source, target))
    if max_flow <= tolerance:
        return {
            **_empty_flow_witnesses(
                "no_positive_context_flow",
                source_units=len(valid_source),
                target_units=len(valid_target),
                interaction_edges=len(valid_interactions),
            ),
            "solver": "exact_maxflow_route_coverage_ratio",
            "selected_ids": [],
            "n_selected": 0,
        }

    candidates = []
    for unit_id, (node_in, edge_index) in gate_edges.items():
        edge = dinic.graph[node_in][edge_index]
        coverage = float(edge.original - edge.cap)
        if coverage <= tolerance:
            continue
        cost = float(unit_by_id[unit_id].get("cost", 1.0))
        if cost <= 0:
            raise ValueError(f"editable unit {unit_id} has non-positive cost")
        candidates.append((
            coverage / cost,
            coverage,
            unit_id,
        ))
    if not candidates:
        return {
            "status": "empty",
            "reason": "no_token_carries_maximum_flow",
            "solver": "exact_maxflow_route_coverage_ratio",
            "selected_ids": [],
            "n_selected": 0,
        }

    ratio, coverage, selected_id = max(
        candidates,
        key=lambda row: (row[0], row[1], row[2]),
    )
    return {
        "status": "optimal",
        "solver": "exact_maxflow_route_coverage_ratio",
        "selected_ids": [selected_id],
        "selected_tokens": [
            str(unit_by_id[selected_id].get("text", ""))
        ],
        "n_selected": 1,
        "max_flow": max_flow,
        "covered_flow": coverage,
        "covered_flow_fraction": coverage / max_flow,
        "ratio": ratio,
        "uses_path_decomposition": False,
        "route_coverage_derived_from_gate_flow": True,
        "is_exact_submodular_ratio_optimum": True,
        "source_units": len(valid_source),
        "target_units": len(valid_target),
        "interaction_edges": len(valid_interactions),
    }


def _empty_flow_witnesses(
    reason: str,
    *,
    source_units: int = 0,
    target_units: int = 0,
    interaction_edges: int = 0,
) -> dict:
    return {
        "status": "empty",
        "reason": reason,
        "max_flow": 0.0,
        "decomposed_flow": 0.0,
        "witnesses": [],
        "n_witnesses": 0,
        "source_units": source_units,
        "target_units": target_units,
        "interaction_edges": interaction_edges,
    }


def solve_source_target_densest_subgraph(
    units: list[dict],
    source_edges: dict[str, float],
    interactions: dict[tuple[str, str], float],
    target_edges: dict[str, float],
    *,
    iterations: int = 40,
) -> dict:
    """Solve the fixed-terminal source-target densest chunk subgraph.

    Query/source and answer/target are fixed terminals, not selectable nodes.
    Their incident contribution edges are rewards for selecting a chunk token:

    ``(W(Q,S) + W(E[S]) + W(S,Y)) / |S|``.

    The parametric maximum-closure reduction gives the exact optimum for this
    positive-weight objective.  No answer-text occurrence matching is used.
    """

    unit_ids = {str(unit["unit_id"]) for unit in units}
    node_weights = {
        unit_id: max(0.0, float(source_edges.get(unit_id, 0.0)))
        + max(0.0, float(target_edges.get(unit_id, 0.0)))
        for unit_id in unit_ids
    }
    weighted_edges = [
        (left, right, float(weight))
        for (left, right), weight in interactions.items()
        if left in unit_ids and right in unit_ids and left != right and float(weight) > 0
    ]
    upper = sum(node_weights.values()) + sum(weight for _, _, weight in weighted_edges)
    if upper <= 0:
        return {
            "status": "no_positive_support",
            "selected_ids": [],
            "density": 0.0,
            "source_weight": 0.0,
            "target_weight": 0.0,
            "internal_weight": 0.0,
            "interaction_edges": 0,
        }

    lower = 0.0
    best: set[str] = set()
    for _ in range(iterations):
        midpoint = (lower + upper) / 2.0
        candidate, value = _maximum_weight_closure_with_node_weights(
            node_weights, weighted_edges, midpoint
        )
        if candidate and value > 1e-12:
            lower = midpoint
            best = candidate
        else:
            upper = midpoint

    selected, _ = _maximum_weight_closure_with_node_weights(node_weights, weighted_edges, lower)
    if selected:
        best = selected
    source_weight = sum(max(0.0, float(source_edges.get(unit_id, 0.0))) for unit_id in best)
    target_weight = sum(max(0.0, float(target_edges.get(unit_id, 0.0))) for unit_id in best)
    internal_weight = sum(
        weight for left, right, weight in weighted_edges if left in best and right in best
    )
    return {
        "status": "optimal" if best else "no_positive_subgraph",
        "selected_ids": sorted(best),
        "density": (source_weight + target_weight + internal_weight) / len(best) if best else 0.0,
        "source_weight": source_weight,
        "target_weight": target_weight,
        "internal_weight": internal_weight,
        "interaction_edges": sum(
            1 for left, right, _ in weighted_edges if left in best and right in best
        ),
    }


def solve_source_target_supermodular_ratio(
    units: list[dict],
    source_edges: dict[str, float],
    interactions: dict[tuple[str, str], float],
    target_edges: dict[str, float],
    *,
    max_iterations: int = 100,
    tolerance: float = 1e-9,
) -> dict:
    """Solve the unrestricted supermodular ratio problem.

    The objective is

        max_{nonempty S} F(S) / |S|

    where ``F(S)`` is the sum of source/context support, target/context
    support, and positive pairwise interaction weights induced by ``S``.
    Dinkelbach iterations reduce the ratio problem to maximum-weight closure
    instances, which are solved exactly by the internal max-flow routine.

    There is no cardinality budget.  The denominator is the edit-count cost.
    ``source_edges`` and ``target_edges`` are fixed-terminal rewards; terminal
    nodes themselves are never selectable.
    """

    unit_ids = {str(unit["unit_id"]) for unit in units}
    node_weights = {
        unit_id: max(0.0, float(source_edges.get(unit_id, 0.0)))
        + max(0.0, float(target_edges.get(unit_id, 0.0)))
        for unit_id in unit_ids
    }
    normalized_edges: dict[tuple[str, str], float] = defaultdict(float)
    for (left, right), weight in interactions.items():
        left, right = str(left), str(right)
        weight = float(weight)
        if left not in unit_ids or right not in unit_ids or left == right or weight <= 0:
            continue
        normalized_edges[tuple(sorted((left, right)))] += weight
    weighted_edges = [
        (left, right, weight)
        for (left, right), weight in normalized_edges.items()
    ]

    if not unit_ids or (not any(node_weights.values()) and not weighted_edges):
        return {
            "status": "no_positive_support",
            "selected_ids": [],
            "density": 0.0,
            "source_weight": 0.0,
            "target_weight": 0.0,
            "internal_weight": 0.0,
            "iterations": 0,
            "history": [],
        }

    def score(selected: set[str]) -> tuple[float, float, float]:
        source_weight = sum(
            max(0.0, float(source_edges.get(unit_id, 0.0)))
            for unit_id in selected
        )
        target_weight = sum(
            max(0.0, float(target_edges.get(unit_id, 0.0)))
            for unit_id in selected
        )
        internal_weight = sum(
            weight
            for left, right, weight in weighted_edges
            if left in selected and right in selected
        )
        return source_weight, target_weight, internal_weight

    density = 0.0
    best: set[str] = set()
    history = []
    for iteration in range(1, max_iterations + 1):
        selected, surrogate_value = _maximum_weight_closure_with_node_weights(
            node_weights,
            weighted_edges,
            density,
        )
        if not selected or surrogate_value <= tolerance:
            break
        source_weight, target_weight, internal_weight = score(selected)
        numerator = source_weight + target_weight + internal_weight
        next_density = numerator / len(selected)
        history.append({
            "iteration": iteration,
            "lambda": density,
            "selected_count": len(selected),
            "surrogate_value": surrogate_value,
            "numerator": numerator,
            "density": next_density,
        })
        best = selected
        if abs(next_density - density) <= tolerance:
            density = next_density
            break
        density = next_density

    if not best:
        selected, _ = _maximum_weight_closure_with_node_weights(
            node_weights,
            weighted_edges,
            density,
        )
        best = selected

    source_weight, target_weight, internal_weight = score(best)
    numerator = source_weight + target_weight + internal_weight
    return {
        "status": "optimal" if best else "no_positive_subgraph",
        "selected_ids": sorted(best),
        "density": numerator / len(best) if best else 0.0,
        "source_weight": source_weight,
        "target_weight": target_weight,
        "internal_weight": internal_weight,
        "interaction_edges": sum(
            1 for left, right, _ in weighted_edges if left in best and right in best
        ),
        "iterations": len(history),
        "history": history,
    }


def solve_supermodular_contract_frontier(
    units: list[dict],
    node_rewards: dict[str, float],
    interactions: dict[tuple[str, str], float],
    *,
    costs: dict[str, float] | None = None,
    tolerance: float = 1e-9,
    max_depth: int = 256,
    price_scale: float = 1.0,
    analytic_endpoint: bool = False,
) -> dict:
    """Enumerate the supported sets of a graph-supermodular contract.

    The valuation is

        f(T) = sum_i a_i 1[i in T] +
               sum_(i,j) b_ij 1[i,j in T],  with a_i,b_ij >= 0.

    For a raw contract price ``t``, the demand set maximizes
    ``t * f(T) - c(T)``.  Its graph-supermodular form makes the demand query a
    maximum-closure/min-cut problem.  The breakpoint recursion follows the
    single-agent supermodular-contract construction: it returns every
    supported demand set, not an arbitrary discretization of ``t``.

    This is exact for the contract surrogate.  It does not claim that the
    supported family contains the optimum of the original black-box reader
    intervention problem.

    ``price_scale`` changes coordinates only: the reported price is
    ``alpha = price_scale * t`` while the demand objective remains unchanged.
    When ``analytic_endpoint`` is enabled, a closed-form sufficient endpoint
    replaces exponential bracketing.  This is useful when an external graph
    construction provides a natural contribution-per-token price scale.
    """

    unit_ids = sorted({str(unit["unit_id"]) for unit in units})
    if not unit_ids:
        return {"status": "empty", "frontier": [], "demand_calls": 0}
    costs = costs or {unit_id: 1.0 for unit_id in unit_ids}
    costs = {
        unit_id: max(0.0, float(costs.get(unit_id, 1.0)))
        for unit_id in unit_ids
    }
    rewards = {
        unit_id: max(0.0, float(node_rewards.get(unit_id, 0.0)))
        for unit_id in unit_ids
    }
    pair_weights: dict[tuple[str, str], float] = defaultdict(float)
    for (left, right), weight in interactions.items():
        left, right = str(left), str(right)
        weight = float(weight)
        if left not in rewards or right not in rewards or left == right or weight <= 0:
            continue
        pair_weights[tuple(sorted((left, right)))] += weight
    weighted_pairs = [(left, right, weight) for (left, right), weight in pair_weights.items()]
    active = {
        unit_id for unit_id, weight in rewards.items() if weight > tolerance
    }
    active.update(left for left, _, weight in weighted_pairs if weight > tolerance)
    active.update(right for _, right, weight in weighted_pairs if weight > tolerance)
    if not active:
        return {
            "status": "empty",
            "frontier": [{"price": 0.0, "selected_ids": [], "reward": 0.0, "cost": 0.0}],
            "demand_calls": 0,
        }

    price_scale = float(price_scale)
    if price_scale <= tolerance:
        price_scale = 1.0

    cache: dict[float, tuple[set[str], float, float]] = {}
    demand_calls = 0

    def score(selected: set[str]) -> tuple[float, float]:
        reward = sum(rewards[unit_id] for unit_id in selected)
        reward += sum(
            weight
            for left, right, weight in weighted_pairs
            if left in selected and right in selected
        )
        cost = sum(costs[unit_id] for unit_id in selected)
        return reward, cost

    def demand(price: float) -> tuple[set[str], float, float]:
        nonlocal demand_calls
        key = round(float(price), 12)
        if key in cache:
            return cache[key]
        demand_calls += 1
        raw_price = price / price_scale
        net_node_rewards = {
            unit_id: raw_price * rewards[unit_id] - costs[unit_id]
            for unit_id in unit_ids
        }
        scaled_pairs = [
            (left, right, raw_price * weight)
            for left, right, weight in weighted_pairs
        ]
        selected, _ = _maximum_weight_closure_with_node_weights(
            net_node_rewards,
            scaled_pairs,
            0.0,
        )
        selected &= active
        reward, cost = score(selected)
        result = (selected, reward, cost)
        cache[key] = result
        return result

    left = demand(0.0)
    endpoint_mode = "doubling"
    if analytic_endpoint:
        full_reward, full_cost = score(active)
        marginal_at_full = [
            full_reward - score(active - {unit_id})[0]
            for unit_id in active
        ]
        min_marginal = min(marginal_at_full, default=0.0)
        if min_marginal > tolerance:
            # For every S strictly contained in active, f(active)-f(S) is at
            # least min_marginal and c(active)-c(S) is at most full_cost.
            raw_high = (full_cost / min_marginal) * (1.0 + 1e-8)
            high = raw_high * price_scale
            right = demand(high)
            endpoint_mode = "analytic"
        else:
            high = price_scale
            right = demand(high)
    else:
        high = price_scale
        right = demand(high)

    # The analytic bound is sufficient for positive graph-supermodular
    # valuations.  Retain a numerical fallback for degenerate inputs.
    while right[0] != active and high < 1e12 * price_scale:
        high *= 2.0
        right = demand(high)
        endpoint_mode = "analytic_with_doubling_fallback" if analytic_endpoint else "doubling"

    # If the final active set is not demanded at the numerical upper bound,
    # use the largest demanded set we found. Zero-reward tokens are irrelevant.
    frontier: list[dict] = []
    seen: set[frozenset[str]] = set()

    def add_frontier(price: float, result: tuple[set[str], float, float]) -> None:
        selected, reward, cost = result
        frozen = frozenset(selected)
        if frozen in seen:
            return
        seen.add(frozen)
        frontier.append({
            "price": float(price),
            "raw_price": float(price / price_scale),
            "selected_ids": sorted(selected),
            "reward": float(reward),
            "cost": float(cost),
        })

    add_frontier(0.0, left)

    def recurse(
        left_result: tuple[set[str], float, float],
        right_result: tuple[set[str], float, float],
        depth: int,
    ) -> None:
        if depth > max_depth:
            return
        left_set, left_reward, left_cost = left_result
        right_set, right_reward, right_cost = right_result
        if right_reward <= left_reward + tolerance:
            return
        denominator = right_reward - left_reward
        raw_intersection = (right_cost - left_cost) / denominator
        if raw_intersection <= 0:
            raw_intersection = max(1e-12, raw_intersection)
        intersection = raw_intersection * price_scale
        probe = intersection * (1.0 + 1e-8) + 1e-12
        middle = demand(probe)
        if middle[1] >= right_reward - tolerance:
            add_frontier(intersection, right_result)
            return
        recurse(left_result, middle, depth + 1)
        recurse(middle, right_result, depth + 1)

    recurse(left, right, 0)
    add_frontier(high, right)
    frontier.sort(key=lambda row: (len(row["selected_ids"]), row["reward"], row["price"]))
    # Keep the empty set first and collapse any numerical duplicate cardinality
    # only at the reporting layer; all distinct supported sets remain available.
    return {
        "status": "optimal",
        "frontier": frontier,
        "demand_calls": demand_calls,
        "active_units": len(active),
        "pair_edges": len(weighted_pairs),
        "theoretical_frontier_bound": len(active) + 1,
        "price_scale": price_scale,
        "endpoint_mode": endpoint_mode,
        "endpoint_price": high,
        "endpoint_raw_price": high / price_scale,
    }


def solve_partial_edge_residual_density(
    units: list[dict],
    source_edges: dict[str, float],
    interactions: dict[tuple[str, str], float],
    target_edges: dict[str, float],
    *,
    selected_ids: set[str] | frozenset[str] = frozenset(),
    endpoint_disruption: float = 0.35,
    max_iterations: int = 100,
    tolerance: float = 1e-9,
) -> dict:
    """Find a highest-density block for the DR-submodular residual objective.

    For an editable interaction edge ``(u,v)`` of contribution ``c``, its
    residual fraction after editing set ``S`` is 1, ``1-alpha``,
    ``1-alpha``, or 0 according as zero, only ``u``, only ``v``, or both
    endpoints are edited.  ``2 * alpha < 1`` makes the residual contribution
    strictly DR-submodular.  The induced disruption gain is therefore a
    positive node-and-pair supermodular function, whose best gain-per-token
    block is solved exactly by Dinkelbach maximum closure.
    """

    if not 0.0 <= endpoint_disruption < 0.5:
        raise ValueError("endpoint_disruption must be in [0, 0.5)")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    unit_ids = {str(unit["unit_id"]) for unit in units}
    edited = {str(unit_id) for unit_id in selected_ids} & unit_ids
    available = unit_ids - edited
    normalized_edges: dict[tuple[str, str], float] = defaultdict(float)
    for (left, right), weight in interactions.items():
        left, right = str(left), str(right)
        weight = float(weight)
        if left in unit_ids and right in unit_ids and left != right and weight > 0:
            normalized_edges[tuple(sorted((left, right)))] += weight
    weighted_edges = [
        (left, right, weight) for (left, right), weight in normalized_edges.items()
    ]

    def residual(selection: set[str]) -> float:
        value = sum(
            max(0.0, float(source_edges.get(unit_id, 0.0)))
            + max(0.0, float(target_edges.get(unit_id, 0.0)))
            for unit_id in unit_ids
            if unit_id not in selection
        )
        for left, right, weight in weighted_edges:
            left_selected = left in selection
            right_selected = right in selection
            if left_selected and right_selected:
                continue
            if left_selected or right_selected:
                value += (1.0 - endpoint_disruption) * weight
            else:
                value += weight
        return value

    initial_residual = residual(edited)
    if not available or initial_residual <= tolerance:
        return {
            "status": "no_available_positive_residual",
            "selected_ids": [],
            "n_selected": 0,
            "initial_residual": initial_residual,
            "remaining_residual": initial_residual,
            "residual_reduction": 0.0,
            "density": 0.0,
            "iterations": 0,
            "history": [],
            "endpoint_disruption": endpoint_disruption,
        }

    # Convert the marginal residual reduction into node prizes plus a positive
    # internal-edge prize.  This is exact for the piecewise edge residual.
    node_weights: dict[str, float] = {
        unit_id: max(0.0, float(source_edges.get(unit_id, 0.0)))
        + max(0.0, float(target_edges.get(unit_id, 0.0)))
        for unit_id in available
    }
    marginal_pairs: list[tuple[str, str, float]] = []
    for left, right, weight in weighted_edges:
        left_selected = left in edited
        right_selected = right in edited
        if left_selected and right_selected:
            continue
        if left_selected:
            if right in available:
                node_weights[right] += (1.0 - endpoint_disruption) * weight
            continue
        if right_selected:
            if left in available:
                node_weights[left] += (1.0 - endpoint_disruption) * weight
            continue
        if left not in available or right not in available:
            continue
        node_weights[left] += endpoint_disruption * weight
        node_weights[right] += endpoint_disruption * weight
        bonus = (1.0 - 2.0 * endpoint_disruption) * weight
        if bonus > tolerance:
            marginal_pairs.append((left, right, bonus))

    density = 0.0
    best: set[str] = set()
    history = []
    for iteration in range(1, max_iterations + 1):
        candidate, value = _maximum_weight_closure_with_node_weights(
            node_weights, marginal_pairs, density
        )
        if not candidate or value <= tolerance:
            break
        remaining = residual(edited | candidate)
        reduction = max(0.0, initial_residual - remaining)
        if reduction <= tolerance:
            break
        next_density = reduction / len(candidate)
        history.append({
            "iteration": iteration,
            "eta": density,
            "selected_count": len(candidate),
            "residual_reduction": reduction,
            "density": next_density,
            "dinkelbach_residual": reduction - density * len(candidate),
        })
        best = candidate
        if abs(next_density - density) <= tolerance:
            density = next_density
            break
        density = next_density

    if not best:
        return {
            "status": "no_positive_marginal_block",
            "selected_ids": [],
            "n_selected": 0,
            "initial_residual": initial_residual,
            "remaining_residual": initial_residual,
            "residual_reduction": 0.0,
            "density": 0.0,
            "iterations": len(history),
            "history": history,
            "endpoint_disruption": endpoint_disruption,
        }
    remaining = residual(edited | best)
    reduction = initial_residual - remaining
    return {
        "status": "optimal",
        "selected_ids": sorted(best),
        "n_selected": len(best),
        "initial_residual": initial_residual,
        "remaining_residual": remaining,
        "residual_reduction": reduction,
        "residual_reduction_fraction": reduction / initial_residual,
        "density": reduction / len(best),
        "iterations": len(history),
        "history": history,
        "endpoint_disruption": endpoint_disruption,
        "marginal_pair_edges": len(marginal_pairs),
        "solver": "dinkelbach_dr_submodular_residual",
    }


def decompose_source_target_densest_subgraphs(
    units: list[dict],
    source_edges: dict[str, float],
    interactions: dict[tuple[str, str], float],
    target_edges: dict[str, float],
    *,
    max_rounds: int = 0,
) -> list[dict]:
    """Peel disjoint fixed-terminal densest chunk subgraphs in descending order."""

    remaining = {str(unit["unit_id"]) for unit in units}
    by_id = {str(unit["unit_id"]): unit for unit in units}
    rounds = []
    while remaining and (max_rounds <= 0 or len(rounds) < max_rounds):
        candidate_units = [by_id[unit_id] for unit_id in sorted(remaining)]
        result = solve_source_target_densest_subgraph(
            candidate_units, source_edges, interactions, target_edges
        )
        selected = set(result["selected_ids"])
        if not selected:
            break
        rounds.append({"round": len(rounds) + 1, **result})
        remaining -= selected
    return rounds


def top_budgeted_cut_sets(
    units: list[dict],
    budget: int,
    *,
    limit: int = 3,
    pool_extra: int = 6,
) -> list[list[dict]]:
    """Return high-support, unit-cost cut candidates for a fixed edit budget."""

    if budget <= 0 or budget > len(units) or limit <= 0:
        return []
    ranked = sorted(units, key=lambda unit: (-float(unit["support"]), unit["unit_id"]))
    pool = ranked[: min(len(ranked), budget + pool_extra)]
    candidates = heapq.nlargest(
        limit,
        itertools.combinations(pool, budget),
        key=lambda selected: (
            sum(float(unit["support"]) for unit in selected),
            tuple(unit["unit_id"] for unit in selected),
        ),
    )
    return [list(selected) for selected in candidates]


def minimum_support_cut(units: list[dict], *, remaining_tau: float = 0.2) -> list[dict]:
    """Return the minimum unit-cost set that removes enough editable support.

    With unit edit costs and additive projected support, sorting by support is
    the exact minimum-cardinality solution for this graph-side relaxation.
    """

    if not 0.0 <= remaining_tau < 1.0:
        raise ValueError("remaining_tau must be in [0, 1)")
    ranked = sorted(units, key=lambda unit: (-float(unit["support"]), unit["unit_id"]))
    total_support = sum(float(unit["support"]) for unit in ranked)
    target = (1.0 - remaining_tau) * total_support
    chosen = []
    removed = 0.0
    for unit in ranked:
        if removed >= target:
            break
        chosen.append(unit)
        removed += float(unit["support"])
    return chosen


def project_graph_interactions(
    record: dict,
    graph_row: dict,
    library: TypedRuleLibrary,
    *,
    k: int = 5,
    nlp=None,
) -> tuple[list[dict], dict[tuple[str, str], float]]:
    """Project layer-token attribution edges to an editable-word graph."""

    units = project_graph_support_to_editable_units(record, graph_row, library, k=k, nlp=nlp)
    position_to_unit = _position_to_unit(graph_row, units)

    interactions: dict[tuple[str, str], float] = defaultdict(float)
    for edge in graph_row.get("graph", {}).get("edges", []):
        left = position_to_unit.get(int(edge["src_position"]))
        right = position_to_unit.get(int(edge["dst_position"]))
        if not left or not right or left == right:
            continue
        key = tuple(sorted((left, right)))
        interactions[key] += max(0.0, float(edge.get("contribution", 0.0)))
    return units, dict(interactions)


def _position_to_unit(graph_row: dict, units: list[dict]) -> dict[int, str]:
    units_by_chunk: dict[str, list[dict]] = defaultdict(list)
    for unit in units:
        units_by_chunk[str(unit["chunk_id"])].append(unit)

    position_to_unit: dict[int, str] = {}
    for token in graph_row.get("context_token_supports", []):
        start = int(token.get("chunk_char_start", -1))
        end = int(token.get("chunk_char_end", -1))
        if start < 0 or end <= start:
            continue
        for unit in units_by_chunk.get(str(token.get("chunk_id", "")), []):
            if _overlaps(start, end, int(unit["chunk_char_start"]), int(unit["chunk_char_end"])):
                position_to_unit[int(token["position"])] = str(unit["unit_id"])
                break
    return position_to_unit


def solve_weighted_densest_subgraph(
    units: list[dict],
    interactions: dict[tuple[str, str], float],
    *,
    iterations: int = 40,
) -> dict:
    """Solve ``max_S sum_{e in E[S]} w_e / |S|`` by parametric max-flow.

    The reduction is maximum-weight closure: every interaction edge is a
    positive-profit node that can be selected only when both endpoint word
    nodes are selected. For a fixed density parameter lambda, word nodes have
    profit ``-lambda``. Binary search over lambda yields the exact optimum up
    to floating-point tolerance for this projected graph objective.
    """

    unit_ids = {unit["unit_id"] for unit in units}
    weighted_edges = [
        (left, right, float(weight))
        for (left, right), weight in interactions.items()
        if left in unit_ids and right in unit_ids and weight > 0
    ]
    if not weighted_edges:
        return {
            "status": "no_interactions",
            "selected_ids": [],
            "density": 0.0,
            "internal_weight": 0.0,
            "interaction_edges": 0,
        }

    upper = sum(weight for _, _, weight in weighted_edges)
    lower = 0.0
    best: set[str] = set()
    for _ in range(iterations):
        candidate, value = _maximum_weight_closure(unit_ids, weighted_edges, (lower + upper) / 2.0)
        if candidate and value > 1e-12:
            lower = (lower + upper) / 2.0
            best = candidate
        else:
            upper = (lower + upper) / 2.0

    selected, _ = _maximum_weight_closure(unit_ids, weighted_edges, lower)
    if selected:
        best = selected
    internal = sum(weight for left, right, weight in weighted_edges if left in best and right in best)
    return {
        "status": "optimal" if best else "no_positive_subgraph",
        "selected_ids": sorted(best),
        "density": internal / len(best) if best else 0.0,
        "internal_weight": internal,
        "interaction_edges": sum(1 for left, right, _ in weighted_edges if left in best and right in best),
    }


def _maximum_weight_closure(
    unit_ids: set[str],
    weighted_edges: list[tuple[str, str, float]],
    density: float,
) -> tuple[set[str], float]:
    dinic = Dinic()
    source = dinic.node()
    sink = dinic.node()
    vertices = {unit_id: dinic.node() for unit_id in unit_ids}
    positive_profit = 0.0
    for unit_id, node in vertices.items():
        dinic.add_edge(node, sink, density)
    for left, right, weight in weighted_edges:
        edge_node = dinic.node()
        dinic.add_edge(source, edge_node, weight)
        dinic.add_edge(edge_node, vertices[left], INF)
        dinic.add_edge(edge_node, vertices[right], INF)
        positive_profit += weight
    flow = dinic.max_flow(source, sink)
    reachable = dinic.reachable(source)
    selected = {unit_id for unit_id, node in vertices.items() if node in reachable}
    return selected, positive_profit - flow


def _maximum_weight_closure_with_node_weights(
    node_weights: dict[str, float],
    weighted_edges: list[tuple[str, str, float]],
    density: float,
) -> tuple[set[str], float]:
    if len(weighted_edges) >= 2_000:
        try:
            return _maximum_weight_closure_with_node_weights_scipy(
                node_weights, weighted_edges, density
            )
        except (ImportError, OverflowError, ValueError):
            pass
    dinic = Dinic()
    source = dinic.node()
    sink = dinic.node()
    vertices = {unit_id: dinic.node() for unit_id in node_weights}
    positive_profit = 0.0
    for unit_id, node in vertices.items():
        profit = float(node_weights[unit_id]) - density
        if profit > 0:
            dinic.add_edge(source, node, profit)
            positive_profit += profit
        elif profit < 0:
            dinic.add_edge(node, sink, -profit)
    for left, right, weight in weighted_edges:
        edge_node = dinic.node()
        dinic.add_edge(source, edge_node, weight)
        dinic.add_edge(edge_node, vertices[left], INF)
        dinic.add_edge(edge_node, vertices[right], INF)
        positive_profit += weight
    flow = dinic.max_flow(source, sink)
    reachable = dinic.reachable(source)
    selected = {unit_id for unit_id, node in vertices.items() if node in reachable}
    return selected, positive_profit - flow


def _maximum_weight_closure_with_node_weights_scipy(
    node_weights: dict[str, float],
    weighted_edges: list[tuple[str, str, float]],
    density: float,
    *,
    capacity_scale: float = 1e9,
) -> tuple[set[str], float]:
    """Solve a large closure graph with SciPy's compiled sparse max-flow."""

    import numpy as np
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import maximum_flow

    unit_ids = list(node_weights)
    vertex_index = {unit_id: index for index, unit_id in enumerate(unit_ids)}
    source = len(unit_ids) + len(weighted_edges)
    sink = source + 1
    node_count = sink + 1
    rows: list[int] = []
    columns: list[int] = []
    capacities: list[int] = []
    positive_capacity = 0
    raw_positive_total = sum(
        max(0.0, float(weight) - density) for weight in node_weights.values()
    ) + sum(max(0.0, float(weight)) for _, _, weight in weighted_edges)
    effective_scale = min(
        capacity_scale,
        2_000_000_000.0 / max(raw_positive_total, 1e-12),
    )

    def scaled(value: float) -> int:
        if value <= 0:
            return 0
        result = max(1, int(round(value * effective_scale)))
        if result >= np.iinfo(np.int64).max // 4:
            raise OverflowError("scaled max-flow capacity exceeds int64")
        return result

    for unit_id, index in vertex_index.items():
        profit = float(node_weights[unit_id]) - density
        if profit > 0:
            capacity = scaled(profit)
            rows.append(source)
            columns.append(index)
            capacities.append(capacity)
            positive_capacity += capacity
        elif profit < 0:
            rows.append(index)
            columns.append(sink)
            capacities.append(scaled(-profit))
    edge_capacities = []
    for _, _, weight in weighted_edges:
        capacity = scaled(float(weight))
        edge_capacities.append(capacity)
        positive_capacity += capacity
    infinite_capacity = positive_capacity + 1
    if infinite_capacity >= np.iinfo(np.int64).max // 4:
        raise OverflowError("closure graph total capacity exceeds int64")
    for edge_offset, ((left, right, _), capacity) in enumerate(
        zip(weighted_edges, edge_capacities)
    ):
        edge_node = len(unit_ids) + edge_offset
        rows.extend((source, edge_node, edge_node))
        columns.extend((edge_node, vertex_index[left], vertex_index[right]))
        capacities.extend((capacity, infinite_capacity, infinite_capacity))

    capacity_graph = csr_matrix(
        (np.asarray(capacities, dtype=np.int64), (rows, columns)),
        shape=(node_count, node_count),
        dtype=np.int64,
    )
    flow = maximum_flow(capacity_graph, source, sink).flow
    residual = (capacity_graph - flow).tocsr()
    residual.eliminate_zeros()
    reachable = {source}
    stack = [source]
    while stack:
        node = stack.pop()
        begin, end = residual.indptr[node], residual.indptr[node + 1]
        for position in range(begin, end):
            if residual.data[position] <= 0:
                continue
            neighbor = int(residual.indices[position])
            if neighbor not in reachable:
                reachable.add(neighbor)
                stack.append(neighbor)
    selected = {
        unit_id for unit_id, index in vertex_index.items() if index in reachable
    }
    value = sum(float(node_weights[unit_id]) - density for unit_id in selected)
    value += sum(
        float(weight)
        for left, right, weight in weighted_edges
        if left in selected and right in selected
    )
    return selected, value


def _empty_source_target_cut(status: str, **counts: int) -> dict:
    return {
        "status": status,
        "selected_ids": [],
        "cut_cost": 0.0,
        "max_flow": 0.0,
        "source_units": counts.get("source_units", 0),
        "target_units": counts.get("target_units", 0),
        "interaction_edges": counts.get("interaction_edges", 0),
    }


def _overlaps(start: int, end: int, other_start: int, other_end: int) -> bool:
    return start < other_end and end > other_start
