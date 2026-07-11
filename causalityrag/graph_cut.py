"""Project attribution-token support to editable words and enumerate cut sets."""

from __future__ import annotations

import heapq
import itertools
from collections import defaultdict

from causalityrag.io import retrieved_contexts
from causalityrag.rules import ONTO_TYPES, TypedRuleLibrary
from causalityrag.token_units import STOPWORDS, TOKEN_RE, classify_token
from causalityrag.max_flow import Dinic, INF


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
