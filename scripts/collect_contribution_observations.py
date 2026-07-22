"""Collect real-data diagnostics for attention, contribution, and flow.

The script runs the direct-activation tracer once per query and records only
the compact statistics needed by the paper's contribution-graph observation:

* an aligned raw-attention/local-contribution/conserved-flow heatmap;
* the cumulative answer-conditioned attention-flow mass retained by Top-d
  source edges.

Full layer-by-token matrices are held only for the current query and are never
serialized.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import islice
import json
import os
import sys
import time
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.attribution_graph import (  # noqa: E402
    DirectActivationAttributionGraphBuilder,
)
from causalityrag.io import iter_records, record_id  # noqa: E402


class ObservationGraphBuilder(DirectActivationAttributionGraphBuilder):
    """Direct-activation builder that retains one query's dense diagnostics."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.raw_attention: dict[int, Any] = {}
        self.local_contribution: dict[int, Any] = {}
        self.attention_residual: dict[int, Any] = {}
        self._observation_trace: Any | None = None

    def reset_observations(self) -> None:
        self.raw_attention.clear()
        self.local_contribution.clear()
        self.attention_residual.clear()

    def _closed_flow_edges(
        self,
        attentions: tuple[Any, ...],
        trace: Any,
        token_meta: list[dict],
        target_positions: list[int],
    ) -> tuple[list[dict], dict]:
        self._observation_trace = trace
        try:
            return super()._closed_flow_edges(
                attentions,
                trace,
                token_meta,
                target_positions,
            )
        finally:
            self._observation_trace = None

    def _direct_attention_matrix(
        self,
        layer_index: int,
        attention: Any,
        attention_output_grad: Any,
        attention_input: Any,
    ) -> Any:
        matrix = super()._direct_attention_matrix(
            layer_index,
            attention,
            attention_output_grad,
            attention_input,
        )
        self.raw_attention[layer_index] = (
            attention[0].detach().float().mean(dim=0).cpu()
        )
        self.local_contribution[layer_index] = matrix.detach().float().cpu()

        trace = self._observation_trace
        if trace is None:
            raise RuntimeError("closed-flow trace is unavailable")
        layer_input = trace.layer_inputs[layer_index][0].detach().float()
        attention_grad = trace.attn_outputs[layer_index].grad[0].detach().float()
        self.attention_residual[layer_index] = (
            attention_grad * layer_input
        ).sum(dim=-1).cpu()
        return matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--model-path",
        default="/data1/yujia/models/Qwen2.5-7B-Instruct",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--heatmap-index", type=int, default=6)
    parser.add_argument("--edge-topk", type=int, default=64)
    parser.add_argument("--receiver-beam", type=int, default=96)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    return parser.parse_args()


def active_attention_receivers(edges: list[dict]) -> dict[tuple[int, int], float]:
    """Recover answer-conditioned mass entering each attention receiver."""

    mass: dict[tuple[int, int], float] = defaultdict(float)
    valid_kinds = {
        "attention_ov_write",
        "attention_residual",
        "background_attention_beam",
    }
    for edge in edges:
        if edge.get("kind") not in valid_kinds:
            continue
        key = (int(edge["model_layer"]), int(edge["dst_position"]))
        mass[key] += max(0.0, float(edge.get("contribution", 0.0)))
    return mass


def query_flow_curve(
    builder: ObservationGraphBuilder,
    graph: dict,
    d_values: list[int],
) -> tuple[list[float], dict[str, float]]:
    """Return the Top-d fraction of dense answer-conditioned attention flow."""

    receiver_mass = active_attention_receivers(graph["edges"])
    retained = np.zeros(len(d_values), dtype=np.float64)
    total_attention_flow = 0.0
    receiver_count = 0
    for (layer, receiver), demand in receiver_mass.items():
        if demand <= 0:
            continue
        local = builder.local_contribution[layer][receiver, : receiver + 1]
        positive = local.clamp_min(0).numpy()
        positive_total = float(positive.sum())
        residual = max(
            0.0,
            float(builder.attention_residual[layer][receiver]),
        )
        denominator = positive_total + residual
        if positive_total <= 0 or denominator <= 0:
            continue
        attention_flow = demand * positive_total / denominator
        sorted_values = np.sort(positive)[::-1]
        cumulative = np.cumsum(sorted_values)
        for index, d_value in enumerate(d_values):
            cutoff = min(d_value, len(cumulative))
            if cutoff:
                retained[index] += demand * cumulative[cutoff - 1] / denominator
        total_attention_flow += attention_flow
        receiver_count += 1

    curve = (
        (retained / total_attention_flow).clip(0.0, 1.0).tolist()
        if total_attention_flow > 0
        else [0.0] * len(d_values)
    )
    return curve, {
        "active_attention_receivers": receiver_count,
        "dense_attention_flow": total_attention_flow,
    }


def heatmap_payload(
    builder: ObservationGraphBuilder,
    row: dict,
) -> dict:
    """Aggregate aligned layer-by-context-token maps for one query."""

    graph = row["graph"]
    context_tokens = sorted(
        row["context_token_supports"],
        key=lambda token: int(token["position"]),
    )
    positions = [int(token["position"]) for token in context_tokens]
    position_to_column = {position: index for index, position in enumerate(positions)}
    layers = int(graph["layers"])
    shape = (layers, len(positions))
    attention_map = np.zeros(shape, dtype=np.float64)
    local_map = np.zeros(shape, dtype=np.float64)
    flow_map = np.zeros(shape, dtype=np.float64)
    receiver_mass = active_attention_receivers(graph["edges"])

    for (layer, receiver), demand in receiver_mass.items():
        if demand <= 0:
            continue
        valid_positions = [position for position in positions if position <= receiver]
        if not valid_positions:
            continue
        columns = [position_to_column[position] for position in valid_positions]
        attention_values = builder.raw_attention[layer][
            receiver,
            valid_positions,
        ].numpy()
        local_values = builder.local_contribution[layer][
            receiver,
            valid_positions,
        ].clamp_min(0).numpy()
        attention_map[layer, columns] += demand * attention_values
        local_map[layer, columns] += demand * local_values

    for edge in graph["edges"]:
        if edge.get("kind") != "attention_ov_write":
            continue
        position = int(edge["src_position"])
        column = position_to_column.get(position)
        if column is None:
            continue
        layer = int(edge["model_layer"])
        flow_map[layer, column] += max(
            0.0,
            float(edge.get("contribution", 0.0)),
        )

    maps = {
        "raw_attention": normalize_mass(attention_map),
        "local_r": normalize_mass(local_map),
        "conserved_f": normalize_mass(flow_map),
    }
    flat_attention = maps["raw_attention"].ravel()
    flat_local = maps["local_r"].ravel()
    nonzero = (flat_attention > 0) | (flat_local > 0)
    pearson = float(np.corrcoef(
        flat_attention[nonzero],
        flat_local[nonzero],
    )[0, 1]) if int(nonzero.sum()) > 1 else 0.0
    top_count = max(1, int(np.ceil(0.01 * len(flat_attention))))
    attention_top = set(np.argpartition(flat_attention, -top_count)[-top_count:])
    local_top = set(np.argpartition(flat_local, -top_count)[-top_count:])

    return {
        "id": row["id"],
        "question": row["question"],
        "target_answer": row["target_answer"],
        "context_positions": positions,
        "context_tokens": [str(token.get("text", "")) for token in context_tokens],
        "chunk_ranks": [int(token.get("chunk_rank", 0)) for token in context_tokens],
        "layers": list(range(layers)),
        "maps": {name: values.tolist() for name, values in maps.items()},
        "attention_r_pearson": pearson,
        "attention_r_top_1pct_overlap": len(attention_top & local_top) / top_count,
    }


def normalize_mass(values: np.ndarray) -> np.ndarray:
    total = float(values.sum())
    return values / total if total > 0 else values


def bootstrap_mean_interval(
    values: np.ndarray,
    *,
    samples: int,
    seed: int = 20260722,
) -> tuple[np.ndarray, np.ndarray]:
    if not len(values):
        empty = np.zeros(values.shape[1] if values.ndim == 2 else 0)
        return empty, empty
    rng = np.random.default_rng(seed)
    estimates = np.empty((samples, values.shape[1]), dtype=np.float64)
    for sample in range(samples):
        indices = rng.integers(0, len(values), size=len(values))
        estimates[sample] = values[indices].mean(axis=0)
    return (
        np.quantile(estimates, 0.025, axis=0),
        np.quantile(estimates, 0.975, axis=0),
    )


def main() -> None:
    args = parse_args()
    if args.n <= 0:
        raise ValueError("n must be positive")
    d_values = [1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64]
    stop = args.start + args.n
    records = islice(iter_records(args.input), args.start, stop)
    targets = islice(iter_records(args.targets), args.start, stop)
    builder = ObservationGraphBuilder(
        args.model_path,
        device=args.device,
        dtype=args.dtype,
        edge_topk=args.edge_topk,
        max_receivers_per_layer=args.receiver_beam,
        max_edges=1,
        closed_flow=True,
    )

    curves = []
    per_query = []
    heatmap = None
    started = time.monotonic()
    for offset, (record, target_row) in enumerate(zip(records, targets)):
        index = args.start + offset
        identifier = record_id(record)
        if str(target_row.get("id")) != identifier:
            raise ValueError(
                f"input/target mismatch at {index}: "
                f"{identifier} != {target_row.get('id')}"
            )
        builder.reset_observations()
        query_started = time.monotonic()
        row = builder.build(
            record,
            str(target_row.get("target_answer", "")),
            k=args.k,
        )
        if row.get("status") != "ok":
            per_query.append({
                "index": index,
                "id": identifier,
                "status": row.get("status"),
            })
            continue
        curve, diagnostics = query_flow_curve(builder, row["graph"], d_values)
        if diagnostics["dense_attention_flow"] > 0:
            curves.append(curve)
        per_query.append({
            "index": index,
            "id": identifier,
            "status": "ok",
            "top_d_flow_fraction": curve,
            **diagnostics,
        })
        if index == args.heatmap_index:
            heatmap = heatmap_payload(builder, row)
        print(
            f"[contribution-observation] {offset + 1}/{args.n} "
            f"index={index} d6={curve[d_values.index(6)]:.4f} "
            f"seconds={time.monotonic() - query_started:.3f}",
            flush=True,
        )

    curve_array = np.asarray(curves, dtype=np.float64)
    if not len(curve_array):
        raise RuntimeError("no successful queries")
    lower, upper = bootstrap_mean_interval(
        curve_array,
        samples=args.bootstrap_samples,
    )
    payload = {
        "dataset": "HotpotQA",
        "retrieval_top_k": args.k,
        "model": "Qwen2.5-7B-Instruct",
        "query_start": args.start,
        "requested_queries": args.n,
        "successful_queries": sum(
            row.get("status") == "ok" for row in per_query
        ),
        "flow_valid_queries": len(curves),
        "zero_positive_attention_flow_queries": sum(
            row.get("status") == "ok"
            and float(row.get("dense_attention_flow", 0.0)) <= 0
            for row in per_query
        ),
        "edge_topk": args.edge_topk,
        "receiver_beam": args.receiver_beam,
        "d_values": d_values,
        "mean_top_d_flow_fraction": curve_array.mean(axis=0).tolist(),
        "median_top_d_flow_fraction": np.median(curve_array, axis=0).tolist(),
        "bootstrap_95_ci_lower": lower.tolist(),
        "bootstrap_95_ci_upper": upper.tolist(),
        "heatmap": heatmap,
        "per_query": per_query,
        "elapsed_seconds": time.monotonic() - started,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as output:
        json.dump(payload, output, ensure_ascii=False)
        output.write("\n")
    print(
        "[contribution-observation summary] "
        + json.dumps({
            key: payload[key]
            for key in (
                "successful_queries",
                "d_values",
                "mean_top_d_flow_fraction",
                "elapsed_seconds",
            )
        }),
        flush=True,
    )


if __name__ == "__main__":
    main()
