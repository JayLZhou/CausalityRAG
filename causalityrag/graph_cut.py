"""Projection from answer-conditioned attribution edges to token-flow edges."""

from __future__ import annotations

from collections import defaultdict


def project_cached_units_source_target_graph(
    graph_row: dict,
    units: list[dict],
) -> tuple[dict[str, float], dict[tuple[str, str], float], dict[str, float], dict]:
    """Project positive contribution edges onto frozen editable token units.

    The query source and answer target are fixed terminals.  Returned edges
    preserve only positive answer-conditioned contributions.  The optional
    shortcut repairs an artifact whose retained edge list omits one terminal
    side while its traced context support proves a context-to-answer route.
    """

    position_to_unit = _position_to_unit(graph_row, units)
    graph = graph_row.get("graph", {})
    partitions = graph.get("token_partitions", {})
    query_positions = {int(position) for position in partitions.get("query", [])}
    target_predictors = {
        int(position) - 1
        for position in graph.get("target_positions", [])
        if int(position) > 0
    }
    source_edges: dict[str, float] = defaultdict(float)
    interactions: dict[tuple[str, str], float] = defaultdict(float)
    target_edges: dict[str, float] = defaultdict(float)
    counts = {"source": 0, "interaction": 0, "target": 0}

    for edge in graph.get("edges", []):
        if edge.get("kind") != "attention_ov_write":
            continue
        weight = max(0.0, float(edge.get("contribution", 0.0)))
        if weight <= 0.0:
            continue
        source = position_to_unit.get(int(edge["src_position"]))
        if source is None:
            continue
        destination_position = int(edge["dst_position"])
        if destination_position in query_positions:
            source_edges[source] += weight
            counts["source"] += 1
        elif destination_position in target_predictors:
            target_edges[source] += weight
            counts["target"] += 1
        else:
            destination = position_to_unit.get(destination_position)
            if destination is not None and destination != source:
                interactions[(source, destination)] += weight
                counts["interaction"] += 1

    if not source_edges or not target_edges:
        shortcut = _context_support_shortcut(graph_row, position_to_unit)
        for unit_id, weight in shortcut.items():
            if not source_edges:
                source_edges[unit_id] += weight
                counts["source"] += 1
            if not target_edges:
                target_edges[unit_id] += weight
                counts["target"] += 1
        used_shortcut = bool(shortcut)
    else:
        used_shortcut = False

    return dict(source_edges), dict(interactions), dict(target_edges), {
        "source_edge_count": counts["source"],
        "interaction_edge_count": counts["interaction"],
        "target_edge_count": counts["target"],
        "uses_answer_chunk_matching": False,
        "unit_domain": "cached_editable_units",
        "path_shortcut_fallback": used_shortcut,
    }


def _context_support_shortcut(
    graph_row: dict,
    position_to_unit: dict[int, str],
) -> dict[str, float]:
    support_by_unit: dict[str, float] = defaultdict(float)
    for row in graph_row.get("context_token_supports", []):
        unit_id = position_to_unit.get(int(row.get("position", -1)))
        support = max(0.0, float(row.get("support", 0.0)))
        if unit_id is not None and support > 0.0:
            support_by_unit[unit_id] += support
    return dict(support_by_unit)


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


def _overlaps(start: int, end: int, other_start: int, other_end: int) -> bool:
    return start < other_end and other_start < end
