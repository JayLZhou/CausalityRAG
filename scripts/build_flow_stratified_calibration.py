"""Build fixed-budget intervention sets in prespecified residual-flow bands."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from causalityrag.contribution_graph import contribution_graph_edges
from causalityrag.flow_contract import remaining_contribution_flow
from causalityrag.shared_replacement_pool import (
    FrozenSharedReplacementPool,
    file_sha256,
)


BANDS = [
    ("flow_0_20", 0.0, 0.2),
    ("flow_20_40", 0.2, 0.4),
    ("flow_40_60", 0.4, 0.6),
    ("flow_60_80", 0.6, 0.8),
    ("flow_80_100", 0.8, 1.000000001),
]


def read_jsonl(path: Path, n: int) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if len(rows) == n:
                    break
    return rows


def stable_order(unit_ids: list[str], *, query_id: str, band: str) -> list[str]:
    return sorted(
        unit_ids,
        key=lambda unit_id: (
            hashlib.sha256(
                f"{query_id}\0{band}\0{unit_id}".encode("utf-8")
            ).digest(),
            unit_id,
        ),
    )


def graph_inputs(row: dict) -> tuple[list[dict], dict, dict, dict]:
    graph = row["contribution_graph"]
    units = [{"unit_id": str(unit_id)} for unit_id in graph["token_nodes"]]
    source, interactions, target = contribution_graph_edges(row)
    return units, source, interactions, target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphs", type=Path, required=True)
    parser.add_argument("--frontier", type=Path, required=True)
    parser.add_argument("--shared-pool", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path, required=True)
    parser.add_argument("--fixed-budget", type=int, default=20)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--capacity-mode", default="normalized")
    args = parser.parse_args()

    graph_rows = read_jsonl(args.graphs, args.n)
    frontier_rows = read_jsonl(args.frontier, args.n)
    if len(graph_rows) != len(frontier_rows):
        raise ValueError("graph and frontier row counts differ")
    pool = FrozenSharedReplacementPool(args.shared_pool)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    output_rows = {name: [] for name, _, _ in BANDS}
    selected_queries = []
    for graph_row, frontier_row in zip(graph_rows, frontier_rows):
        query_id = str(graph_row.get("id", ""))
        if query_id != str(frontier_row.get("id", "")):
            raise ValueError(f"misaligned query: {query_id}")
        empty_rows = {
            name: {"id": query_id, "ranked_ids": []}
            for name, _, _ in BANDS
        }
        if (
            graph_row.get("status") != "ok"
            or not frontier_row.get("frontier_candidates")
        ):
            for name in output_rows:
                output_rows[name].append(empty_rows[name])
            continue

        graph_units, source, interactions, target = graph_inputs(graph_row)
        units = [
            unit
            for unit in graph_units
            if pool.is_eligible(str(unit["unit_id"]))
        ]
        clean_flow = float(frontier_row.get("initial_support_flow", 0.0))
        if clean_flow <= 0.0:
            for name in output_rows:
                output_rows[name].append(empty_rows[name])
            continue

        active_ids = {
            str(unit_id)
            for unit_id, weight in source.items()
            if float(weight) > 0.0
        }
        active_ids.update(
            str(unit_id)
            for unit_id, weight in target.items()
            if float(weight) > 0.0
        )
        for (left, right), weight in interactions.items():
            if float(weight) > 0.0:
                active_ids.add(str(left))
                active_ids.add(str(right))
        neutral_ids = [
            str(unit["unit_id"])
            for unit in units
            if (
                str(unit["unit_id"]) not in active_ids
                and pool.is_eligible(str(unit["unit_id"]))
            )
        ]
        if len(neutral_ids) < args.fixed_budget:
            for name in output_rows:
                output_rows[name].append(empty_rows[name])
            continue

        candidates = [
            {
                "selected_ids": [],
                "n_selected": 0,
                "remaining_support_fraction": 1.0,
            }
        ]
        candidates.extend(
            candidate
            for candidate in frontier_row["frontier_candidates"]
            if int(candidate.get("n_selected", 0)) <= args.fixed_budget
        )
        chosen = {}
        for name, low, high in BANDS:
            in_band = [
                candidate
                for candidate in candidates
                if low
                <= float(candidate["remaining_support_fraction"])
                < high
            ]
            if not in_band:
                break
            midpoint = (low + min(high, 1.0)) / 2.0
            chosen[name] = min(
                in_band,
                key=lambda candidate: (
                    abs(
                        float(candidate["remaining_support_fraction"])
                        - midpoint
                    ),
                    int(candidate.get("n_selected", 0)),
                ),
            )
        if len(chosen) != len(BANDS):
            for name in output_rows:
                output_rows[name].append(empty_rows[name])
            continue

        query_rows = {}
        for name, low, high in BANDS:
            candidate = chosen[name]
            core_ids = [str(value) for value in candidate["selected_ids"]]
            padding = stable_order(
                neutral_ids,
                query_id=query_id,
                band=name,
            )[: args.fixed_budget - len(core_ids)]
            selected_ids = core_ids + padding
            if len(selected_ids) != args.fixed_budget:
                raise ValueError(f"failed to pad {query_id} in {name}")
            residual = remaining_contribution_flow(
                units,
                source,
                interactions,
                target,
                removed_ids=set(selected_ids),
                edge_capacity_mode=args.capacity_mode,
            ) / clean_flow
            expected = float(candidate["remaining_support_fraction"])
            if abs(residual - expected) > 1e-8:
                raise ValueError(
                    f"neutral padding changed residual flow for {query_id}: "
                    f"{expected} -> {residual}"
                )
            query_rows[name] = {
                "id": query_id,
                "ranked_ids": selected_ids,
                "residual_flow": residual,
                "residual_band": [low, min(high, 1.0)],
                "core_selected_ids": core_ids,
                "neutral_padding_ids": padding,
                "fixed_budget": args.fixed_budget,
            }
        selected_queries.append(query_id)
        for name in output_rows:
            output_rows[name].append(query_rows[name])

    score_files = {}
    for name, rows in output_rows.items():
        path = args.out_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        score_files[name] = {
            "path": str(path.resolve()),
            "sha256": file_sha256(str(path)),
        }
    manifest = {
        "schema": "causalityrag.flow_stratified_calibration.v1",
        "queries": len(graph_rows),
        "selected_queries": len(selected_queries),
        "fixed_budget": args.fixed_budget,
        "capacity_mode": args.capacity_mode,
        "bands": [
            {"name": name, "low": low, "high": min(high, 1.0)}
            for name, low, high in BANDS
        ],
        "selection": (
            "For each query, select the frontier candidate nearest the midpoint "
            "of every prespecified residual-flow band. Pad every candidate to "
            "the same token count using eligible token positions absent from "
            "all positive-capacity graph edges; verify padding leaves maximum "
            "flow unchanged."
        ),
        "graphs": {
            "path": str(args.graphs.resolve()),
            "sha256": file_sha256(str(args.graphs)),
        },
        "frontier": {
            "path": str(args.frontier.resolve()),
            "sha256": file_sha256(str(args.frontier)),
        },
        "shared_pool": {
            "path": str(args.shared_pool.resolve()),
            "sha256": file_sha256(str(args.shared_pool)),
        },
        "score_files": score_files,
    }
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
