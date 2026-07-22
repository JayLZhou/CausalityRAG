"""Attach raw-r, conserved-f, and aggregated-c scores to intervention data."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.io import iter_records, load_records, record_id  # noqa: E402
from causalityrag.mixed_cut import (  # noqa: E402
    build_projected_token_contribution_network,
    build_raw_contribution_network,
)
from causalityrag.token_units import units_from_cache_row  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--records", required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--graphs", required=True)
    parser.add_argument("--units-cache", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def row_by_id(path: str, identifier: str) -> dict:
    for row in iter_records(path):
        if str(row.get("id", row.get("qid", ""))) == identifier:
            return row
    raise ValueError(f"missing row for {identifier} in {path}")


def top_effect_summary(rows: list[dict], score_key: str, top_k: int) -> dict:
    top_k = min(top_k, len(rows))
    ranked = sorted(
        rows,
        key=lambda row: (-float(row[score_key]), str(row["unit_id"])),
    )[:top_k]
    captured = sum(float(row["intervention_effect"]) for row in ranked)
    ideal = sum(
        sorted(
            (float(row["intervention_effect"]) for row in rows),
            reverse=True,
        )[:top_k]
    )
    return {
        "top_k": top_k,
        "captured_effect": captured,
        "ideal_effect": ideal,
        "captured_fraction": captured / ideal if ideal > 0.0 else 0.0,
        "selected_unit_ids": [str(row["unit_id"]) for row in ranked],
        "selected_tokens": [str(row["token"]) for row in ranked],
    }


def main() -> None:
    args = parse_args()
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    record = load_records(args.records)[args.index]
    identifier = record_id(record)
    if str(payload.get("id")) != identifier:
        raise ValueError("intervention payload does not match requested record")

    graph_row = row_by_id(args.graphs, identifier)
    units_row = row_by_id(args.units_cache, identifier)
    units = units_from_cache_row(record, units_row, k=args.k)

    networks = {
        "raw_r_score": build_raw_contribution_network(
            graph_row,
            units,
            capacity_mode="raw",
        ),
        "conserved_f_score": build_raw_contribution_network(
            graph_row,
            units,
            capacity_mode="backward-conserved",
        ),
        "aggregated_c_score": build_projected_token_contribution_network(
            graph_row,
            units,
            capacity_mode="backward-conserved",
        ),
    }
    for name, network in networks.items():
        if network.status != "ok":
            raise RuntimeError(f"{name} network failed: {network.status}")

    rows = payload["tokens"]
    for row in rows:
        unit_id = str(row["unit_id"])
        for name, network in networks.items():
            row[name] = float(network.unit_scores.get(unit_id, 0.0))

    stages = [
        ("attention", "attention_score"),
        ("raw_r", "raw_r_score"),
        ("conserved_f", "conserved_f_score"),
        ("aggregated_c", "aggregated_c_score"),
    ]
    payload["stage_alignment"] = {
        name: top_effect_summary(rows, score_key, args.top_k)
        for name, score_key in stages
    }
    payload["stage_definitions"] = {
        "attention": "mean raw attention allocated to each retrieved token",
        "raw_r": "positive answer-gradient dot attention-message contribution",
        "conserved_f": "answer-targeted backward-normalized contribution flow",
        "aggregated_c": "flow aggregated across layer copies of retrieved tokens",
        "intervention_effect": (
            "positive decrease in clean-answer target logit after one legal "
            "token replacement"
        ),
    }
    payload["graph_source"] = args.graphs
    payload["units_source"] = args.units_cache

    Path(args.out).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
