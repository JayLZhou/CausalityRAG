"""End-to-end construction of the ReFlow contribution graph."""

from __future__ import annotations

from collections import defaultdict
from causalityrag.message_flow import _ClosedMessageFlowExtractor
from causalityrag.io import record_id
from causalityrag.token_units import all_context_word_units


class ContributionGraphBuilder:
    """Build the final token-level graph in one model execution."""

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cuda",
        dtype: str = "bfloat16",
        max_receivers_per_layer: int = 48,
    ) -> None:
        self._extractor = _ClosedMessageFlowExtractor(
            model_path,
            device=device,
            dtype=dtype,
            max_context_tokens=0,
            max_length=0,
            max_receivers_per_layer=max_receivers_per_layer,
            target_objective="mean-answer-logit",
        )

    def build(
        self,
        record: dict,
        target_answer: str,
        *,
        k: int = 5,
    ) -> dict:
        """Run attribution, close the flow, and contract it to token labels."""

        token_units = all_context_word_units(record, k=k)
        message_flow = self._extractor.extract(
            record,
            target_answer,
            k=k,
        )
        status = str(message_flow.get("status", "invalid_message_flow"))
        if status != "ok":
            return _empty_graph_row(
                record,
                target_answer,
                status,
                message_flow,
                token_units,
            )

        source, interactions, target, diagnostics = _contract_token_labels(
            message_flow,
            token_units,
        )
        if not source:
            status = "no_source_support"
        elif not target:
            status = "no_answer_support"
        else:
            status = "ok"

        raw_graph = message_flow.get("graph", {})
        return {
            "id": record_id(record),
            "question": str(record.get("question", "")),
            "gold_answer": str(record.get("answer", "")),
            "target_answer": target_answer,
            "status": status,
            "method": "closed_flow_token_contribution_graph",
            "contribution_graph": {
                "source": "query_context_source",
                "target": "answer_target",
                "token_nodes": sorted(
                    str(unit["unit_id"]) for unit in token_units
                ),
                "source_edges": [
                    {"token_id": unit_id, "capacity": weight}
                    for unit_id, weight in sorted(source.items())
                ],
                "interaction_edges": [
                    {"source": left, "target": right, "capacity": weight}
                    for (left, right), weight in sorted(interactions.items())
                ],
                "target_edges": [
                    {"token_id": unit_id, "capacity": weight}
                    for unit_id, weight in sorted(target.items())
                ],
                "diagnostics": diagnostics,
            },
            "message_flow_diagnostics": {
                "sequence_tokens": raw_graph.get("sequence_tokens", 0),
                "layers": raw_graph.get("layers", 0),
                "stages": raw_graph.get("stages", 0),
                "receiver_beam": raw_graph.get("receiver_beam", 0),
                "flow": raw_graph.get("flow_diagnostics", {}),
            },
        }


def contribution_graph_edges(
    graph_row: dict,
) -> tuple[dict[str, float], dict[tuple[str, str], float], dict[str, float]]:
    """Read the three edge families consumed by the ReFlow optimizer."""

    graph = graph_row.get("contribution_graph", {})
    source = {
        str(edge["token_id"]): float(edge["capacity"])
        for edge in graph.get("source_edges", [])
        if float(edge.get("capacity", 0.0)) > 0.0
    }
    interactions = {
        (str(edge["source"]), str(edge["target"])): float(edge["capacity"])
        for edge in graph.get("interaction_edges", [])
        if (
            str(edge.get("source", "")) != str(edge.get("target", ""))
            and float(edge.get("capacity", 0.0)) > 0.0
        )
    }
    target = {
        str(edge["token_id"]): float(edge["capacity"])
        for edge in graph.get("target_edges", [])
        if float(edge.get("capacity", 0.0)) > 0.0
    }
    return source, interactions, target


def _contract_token_labels(
    message_flow: dict,
    units: list[dict],
) -> tuple[dict[str, float], dict[tuple[str, str], float], dict[str, float], dict]:
    """Contract closed Transformer-stage flow by retrieved-token label."""

    position_to_unit = _position_to_unit(message_flow, units)
    graph = message_flow.get("graph", {})
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
    counts = {
        "fixed_source": 0,
        "query_anchor": 0,
        "interaction": 0,
        "target": 0,
        "contracted_internal": 0,
    }
    mass = {key: 0.0 for key in counts}
    retained_kinds: dict[str, int] = defaultdict(int)

    for edge in graph.get("edges", []):
        weight = max(0.0, float(edge.get("contribution", 0.0)))
        source_node = str(edge.get("src", ""))
        target_node = str(edge.get("dst", ""))
        if weight <= 0.0 or source_node.startswith("background::"):
            continue

        source_position = int(edge.get("src_position", -1))
        target_position = int(edge.get("dst_position", -1))
        source_unit = position_to_unit.get(source_position)
        target_unit = position_to_unit.get(target_position)
        edge_kind = str(edge.get("kind", "unknown"))

        if source_unit is not None and (
            target_node == "answer_target" or target_position in target_predictors
        ):
            target_edges[source_unit] += weight
            bucket = "target"
        elif source_unit is not None and target_position in query_positions:
            source_edges[source_unit] += weight
            bucket = "query_anchor"
        elif source_unit is not None and target_unit is not None:
            if source_unit == target_unit:
                bucket = "contracted_internal"
            else:
                interactions[(source_unit, target_unit)] += weight
                bucket = "interaction"
        elif source_unit is None and target_unit is not None:
            source_edges[target_unit] += weight
            bucket = "fixed_source"
        else:
            continue

        counts[bucket] += 1
        mass[bucket] += weight
        retained_kinds[edge_kind] += 1

    if not source_edges or not target_edges:
        shortcut = _context_support_shortcut(message_flow, position_to_unit)
        for unit_id, weight in shortcut.items():
            if not source_edges:
                source_edges[unit_id] += weight
            if not target_edges:
                target_edges[unit_id] += weight
        used_shortcut = bool(shortcut)
    else:
        used_shortcut = False

    return dict(source_edges), dict(interactions), dict(target_edges), {
        "graph_type": "answer_conditioned_token_contribution_graph",
        "construction": "closed_flow_token_label_contraction",
        "source_edge_count": len(source_edges),
        "interaction_edge_count": len(interactions),
        "target_edge_count": len(target_edges),
        "raw_edge_counts": counts,
        "raw_edge_mass": mass,
        "retained_edge_kinds": dict(retained_kinds),
        "contracted_token_labels": len(set(position_to_unit.values())),
        "uses_answer_chunk_matching": False,
        "unit_domain": "all_non_punctuation_context_tokens",
        "replacement_protocol": "on_demand_after_selection",
        "path_shortcut_fallback": used_shortcut,
    }


def _empty_graph_row(
    record: dict,
    target_answer: str,
    status: str,
    message_flow: dict,
    units: list[dict],
) -> dict:
    raw_graph = message_flow.get("graph", {})
    return {
        "id": record_id(record),
        "question": str(record.get("question", "")),
        "gold_answer": str(record.get("answer", "")),
        "target_answer": target_answer,
        "status": status,
        "method": "closed_flow_token_contribution_graph",
        "contribution_graph": {
            "source": "query_context_source",
            "target": "answer_target",
            "token_nodes": sorted(str(unit["unit_id"]) for unit in units),
            "source_edges": [],
            "interaction_edges": [],
            "target_edges": [],
            "diagnostics": {
                "graph_type": "answer_conditioned_token_contribution_graph",
                "construction": "closed_flow_token_label_contraction",
                "unit_domain": "all_non_punctuation_context_tokens",
                "replacement_protocol": "on_demand_after_selection",
            },
        },
        "message_flow_diagnostics": {
            "sequence_tokens": raw_graph.get("sequence_tokens", 0),
            "layers": raw_graph.get("layers", 0),
            "stages": raw_graph.get("stages", 0),
            "receiver_beam": raw_graph.get("receiver_beam", 0),
            "flow": raw_graph.get("flow_diagnostics", {}),
        },
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
