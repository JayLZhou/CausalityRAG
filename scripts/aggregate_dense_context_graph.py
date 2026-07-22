"""Contract an unpruned model-token context graph into lexical token nodes."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import re


def load_row(path: str, row_id: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if not row_id or str(row.get("id")) == row_id:
                return row
    raise ValueError(f"record {row_id!r} not found in {path}")


def overlap(left_start: int, left_end: int, right_start: int, right_end: int) -> int:
    return max(0, min(left_end, right_end) - max(left_start, right_start))


def context_position_map(context_meta: list[dict], units: list[dict]) -> dict[int, str]:
    by_chunk: dict[str, list[dict]] = defaultdict(list)
    for unit in units:
        by_chunk[str(unit["chunk_id"])].append(unit)

    mapping = {}
    for token in context_meta:
        candidates = by_chunk.get(str(token["chunk_id"]), [])
        ranked = [
            (
                overlap(
                    int(token["chunk_char_start"]),
                    int(token["chunk_char_end"]),
                    int(unit["chunk_char_start"]),
                    int(unit["chunk_char_end"]),
                ),
                -int(unit["chunk_char_start"]),
                str(unit["unit_id"]),
            )
            for unit in candidates
        ]
        if ranked:
            best_overlap, _, unit_id = max(ranked)
            if best_overlap > 0:
                mapping[int(token["position"])] = unit_id
    return mapping


def answer_units(target_outputs: list[dict]) -> tuple[list[dict], dict[int, str]]:
    pieces = []
    cursor = 0
    for output in target_outputs:
        text = str(output["text"])
        pieces.append({
            "answer_position": int(output["answer_position"]),
            "start": cursor,
            "end": cursor + len(text),
        })
        cursor += len(text)
    answer_text = "".join(str(output["text"]) for output in target_outputs)
    units = [
        {
            "unit_id": f"answer:{match.start()}:{match.end()}",
            "text": match.group(0),
            "char_start": match.start(),
            "char_end": match.end(),
        }
        for match in re.finditer(r"[\w]+(?:[-'][\w]+)*", answer_text)
    ]
    mapping = {}
    for piece in pieces:
        ranked = [
            (
                overlap(
                    piece["start"],
                    piece["end"],
                    int(unit["char_start"]),
                    int(unit["char_end"]),
                ),
                str(unit["unit_id"]),
            )
            for unit in units
        ]
        if ranked:
            best_overlap, unit_id = max(ranked)
            if best_overlap > 0:
                mapping[piece["answer_position"]] = unit_id
    return units, mapping


def empty_edge(src: str, dst: str, kind: str) -> dict:
    return {
        "src": src,
        "dst": dst,
        "kind": kind,
        "positive_contribution": 0.0,
        "negative_contribution": 0.0,
        "signed_contribution": 0.0,
        "model_pair_count": 0,
    }


def aggregate(graph_row: dict, unit_row: dict, top_k: int) -> dict:
    dense = graph_row["graph"]["dense_context_graph"]
    context_meta = [
        token
        for token in graph_row["context_token_supports"]
        if int(token["chunk_rank"]) <= top_k
    ]
    chunk_ids = {str(token["chunk_id"]) for token in context_meta}
    units = [
        unit
        for unit in unit_row["units"]
        if str(unit["chunk_id"]) in chunk_ids and int(unit["chunk_rank"]) <= top_k
    ]
    units.sort(
        key=lambda unit: (
            int(unit["chunk_rank"]),
            int(unit["chunk_char_start"]),
            int(unit["chunk_char_end"]),
        )
    )
    context_map = context_position_map(context_meta, units)
    answers, answer_map = answer_units(graph_row["target_outputs"])

    edges: dict[tuple[str, str], dict] = {}
    for dst_index, dst in enumerate(units):
        for src in units[: dst_index + 1]:
            kind = (
                "within_chunk"
                if str(src["chunk_id"]) == str(dst["chunk_id"])
                else "cross_chunk"
            )
            edge = empty_edge(str(src["unit_id"]), str(dst["unit_id"]), kind)
            edges[(edge["src"], edge["dst"])] = edge
    for src in units:
        for dst in answers:
            edge = empty_edge(
                str(src["unit_id"]),
                str(dst["unit_id"]),
                "context_to_answer",
            )
            edges[(edge["src"], edge["dst"])] = edge

    unmapped_model_pairs = 0
    for model_edge in dense["edges"]:
        src = context_map.get(int(model_edge["src_position"]))
        if model_edge["dst_region"] == "context":
            dst = context_map.get(int(model_edge["dst_position"]))
        else:
            dst = answer_map.get(int(model_edge["answer_position"]))
        if src is None or dst is None or (src, dst) not in edges:
            unmapped_model_pairs += 1
            continue
        edge = edges[(src, dst)]
        edge["positive_contribution"] += float(model_edge["positive_contribution"])
        edge["negative_contribution"] += float(model_edge["negative_contribution"])
        edge["signed_contribution"] += float(model_edge["signed_contribution"])
        edge["model_pair_count"] += 1

    output_edges = []
    for edge in edges.values():
        for field in (
            "positive_contribution",
            "negative_contribution",
            "signed_contribution",
        ):
            edge[field] = round(float(edge[field]), 12)
        output_edges.append(edge)

    counts = defaultdict(int)
    for edge in output_edges:
        counts[edge["kind"]] += 1
    return {
        "id": graph_row["id"],
        "question": graph_row["question"],
        "target_answer": graph_row["target_answer"],
        "top_k": top_k,
        "pruned": False,
        "contraction": "sum over layers and overlapping model subwords",
        "context_tokens": units,
        "answer_tokens": answers,
        "candidate_edges": len(output_edges),
        "edge_breakdown": dict(sorted(counts.items())),
        "positive_edges": sum(
            edge["positive_contribution"] > 0 for edge in output_edges
        ),
        "negative_edges": sum(
            edge["negative_contribution"] > 0 for edge in output_edges
        ),
        "unmapped_model_pairs": unmapped_model_pairs,
        "edges": output_edges,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", required=True)
    parser.add_argument("--token-units", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--id", default="")
    parser.add_argument("--top-k", type=int, default=2)
    args = parser.parse_args()

    graph_row = load_row(args.graph, args.id)
    unit_row = load_row(args.token_units, str(graph_row["id"]))
    result = aggregate(graph_row, unit_row, args.top_k)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({key: result[key] for key in (
        "id",
        "context_tokens",
        "answer_tokens",
        "candidate_edges",
        "edge_breakdown",
        "positive_edges",
        "negative_edges",
        "unmapped_model_pairs",
    ) if key not in {"context_tokens", "answer_tokens"}}))


if __name__ == "__main__":
    main()
