"""Experimental analysis: collect pilot contribution-flow observations."""

from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import islice
import json
import math
from pathlib import Path

import numpy as np


LOCAL_KINDS = {
    "attention_ov_write",
    "attention_residual",
    "mlp_output",
    "mlp_residual",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graphs", required=True)
    parser.add_argument("--verification", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--top-context-positions", type=int, default=20)
    return parser.parse_args()


def percentile(values: list[float], levels: list[float]) -> list[float]:
    return np.percentile(np.asarray(values, dtype=np.float64), levels).tolist()


def flip_summary(rows: list[dict]) -> dict:
    stages = ["attention", "raw_r", "conserved_f", "aggregated_c"]
    budgets = [1, 3, 5]
    rates = {}
    for stage in stages:
        rates[stage] = {}
        for budget in budgets:
            key = f"budget_b{budget}:{stage}"
            values = [bool(row["methods"][key]["flip"]) for row in rows]
            rates[stage][str(budget)] = {
                "queries": len(values),
                "flips": sum(values),
                "flip_rate": sum(values) / len(values),
            }

    paired = {}
    for budget in budgets:
        attention_key = f"budget_b{budget}:attention"
        graph_key = f"budget_b{budget}:aggregated_c"
        pairs = [
            (
                bool(row["methods"][attention_key]["flip"]),
                bool(row["methods"][graph_key]["flip"]),
            )
            for row in rows
        ]
        paired[str(budget)] = {
            "attention_only": sum(left and not right for left, right in pairs),
            "graph_only": sum(right and not left for left, right in pairs),
            "both": sum(left and right for left, right in pairs),
            "neither": sum(not left and not right for left, right in pairs),
        }
    return {
        "queries": len(rows),
        "budgets": budgets,
        "rates": rates,
        "attention_vs_aggregated_c": paired,
    }


def graph_observations(path: str, n: int, top_positions: int) -> dict:
    receiver_log_mass: list[float] = []
    receiver_by_layer: dict[int, list[float]] = defaultdict(list)
    outside_dominant: list[float] = []
    active_layers: list[int] = []
    query_outside_median: list[float] = []
    query_layers_median: list[float] = []
    successful = 0

    with open(path, encoding="utf-8") as handle:
        for row in islice((json.loads(line) for line in handle), n):
            if row.get("status") != "ok":
                continue
            successful += 1
            graph = row["graph"]
            receiver_edges: dict[tuple[str, int], list[float]] = defaultdict(list)
            for edge in graph["edges"]:
                if edge.get("kind") not in LOCAL_KINDS:
                    continue
                value = max(
                    0.0,
                    float(edge.get("weight", edge.get("raw_signed_contribution", 0.0))),
                )
                if value <= 0.0:
                    continue
                receiver_edges[(
                    str(edge["dst"]),
                    int(edge.get("model_layer", -1)),
                )].append(value)
            for (_, layer), values in receiver_edges.items():
                if len(values) < 2:
                    continue
                log_mass = math.log10(sum(values))
                receiver_log_mass.append(log_mass)
                receiver_by_layer[layer].append(log_mass)

            context_positions = set(
                int(value)
                for value in graph["token_partitions"].get("context", [])
            )
            position_layers: dict[int, dict[int, float]] = defaultdict(
                lambda: defaultdict(float)
            )
            for edge in graph["edges"]:
                if edge.get("kind") != "attention_ov_write":
                    continue
                position = int(edge.get("src_position", -1))
                if position not in context_positions:
                    continue
                value = max(0.0, float(edge.get("contribution", 0.0)))
                if value > 0.0:
                    position_layers[position][
                        int(edge.get("model_layer", -1))
                    ] += value
            ranked = sorted(
                position_layers.values(),
                key=lambda layers: -sum(layers.values()),
            )[:top_positions]
            query_outside = []
            query_layers = []
            for layers in ranked:
                total = sum(layers.values())
                if total <= 0.0:
                    continue
                outside = 1.0 - max(layers.values()) / total
                outside_dominant.append(outside)
                active_layers.append(len(layers))
                query_outside.append(outside)
                query_layers.append(len(layers))
            if query_outside:
                query_outside_median.append(float(np.median(query_outside)))
                query_layers_median.append(float(np.median(query_layers)))

    layer_rows = []
    for layer in sorted(receiver_by_layer):
        values = receiver_by_layer[layer]
        if len(values) < 20:
            continue
        p05, p25, p50, p75, p95 = percentile(values, [5, 25, 50, 75, 95])
        layer_rows.append({
            "layer": layer + 1,
            "receivers": len(values),
            "p05": p05,
            "p25": p25,
            "median": p50,
            "p75": p75,
            "p95": p95,
        })

    overall = percentile(receiver_log_mass, [5, 25, 50, 75, 95])
    return {
        "requested_queries": n,
        "successful_graphs": successful,
        "receiver_scale": {
            "receivers": len(receiver_log_mass),
            "log10_mass_percentiles": {
                "p05": overall[0],
                "p25": overall[1],
                "median": overall[2],
                "p75": overall[3],
                "p95": overall[4],
            },
            "p05_to_p95_orders": overall[4] - overall[0],
            "by_layer": layer_rows,
        },
        "layer_fragmentation": {
            "top_context_positions_per_query": top_positions,
            "positions": len(outside_dominant),
            "queries": len(query_outside_median),
            "outside_dominant_layer": outside_dominant,
            "active_layers": active_layers,
            "query_median_outside_percentiles": dict(zip(
                ["p05", "p25", "median", "p75", "p95"],
                percentile(query_outside_median, [5, 25, 50, 75, 95]),
            )),
            "query_median_active_layers_percentiles": dict(zip(
                ["p05", "p25", "median", "p75", "p95"],
                percentile(query_layers_median, [5, 25, 50, 75, 95]),
            )),
        },
    }


def main() -> None:
    args = parse_args()
    verification = [
        json.loads(line)
        for line in open(args.verification, encoding="utf-8")
        if line.strip()
    ]
    payload = {
        "dataset": "HotpotQA",
        "reader": "Qwen2.5-7B-Instruct",
        "retrieval_top_k": 5,
        "selection": flip_summary(verification),
        "graphs": graph_observations(
            args.graphs,
            args.n,
            args.top_context_positions,
        ),
    }
    Path(args.out).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
