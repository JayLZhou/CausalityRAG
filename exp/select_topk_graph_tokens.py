"""Select a fixed Top-K graph-local token-score baseline."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from itertools import islice

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.io import iter_records, load_records, record_id  # noqa: E402
from causalityrag.mixed_cut import (  # noqa: E402
    build_projected_token_contribution_network,
    remaining_support_flow,
    restrict_group_editable_units,
)
from causalityrag.token_units import units_from_context_row  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--graphs", required=True)
    parser.add_argument("--context-units", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", default="")
    parser.add_argument("--replacement-registry", default="")
    parser.add_argument(
        "--replacement-registry-policy",
        choices=("allow-only", "exclude-known-invalid"),
        default="allow-only",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--top-tokens", type=int, default=5)
    args = parser.parse_args()
    if args.start < 0 or args.n <= 0 or args.k <= 0 or args.top_tokens <= 0:
        parser.error("start must be non-negative and counts must be positive")

    units_by_id = {
        str(row["id"]): row
        for row in load_records(args.context_units)
    }
    registry_by_id = (
        {
            str(row["id"]): row
            for row in load_records(args.replacement_registry)
        }
        if args.replacement_registry
        else {}
    )
    records = islice(
        iter_records(args.input),
        args.start,
        args.start + args.n,
    )
    graphs = islice(
        iter_records(args.graphs),
        args.start,
        args.start + args.n,
    )

    rows = []
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as output:
        for offset, (record, graph_row) in enumerate(zip(records, graphs)):
            started = time.monotonic()
            index = args.start + offset
            identifier = record_id(record)
            if str(graph_row.get("id")) != identifier:
                raise ValueError(f"input/graph mismatch at row {index}")
            units_row = units_by_id.get(identifier)
            if units_row is None:
                raise ValueError(f"missing context units for {identifier}")
            units = units_from_context_row(record, units_row, k=args.k)
            by_id = {str(unit["unit_id"]): unit for unit in units}
            network = build_projected_token_contribution_network(
                graph_row,
                units,
                capacity_mode="raw",
            )

            known_registry_ids: set[str] | None = None
            if registry_by_id:
                registry_row = registry_by_id.get(identifier)
                if registry_row is None:
                    raise ValueError(
                        f"missing replacement registry row for {identifier}"
                    )
                replacements = registry_row.get("replacements", {})
                invalid_ids = {
                    str(unit_id)
                    for unit_id in registry_row.get("invalid", {})
                }
                known_registry_ids = {
                    str(unit_id)
                    for unit_id in replacements
                } | invalid_ids
                if args.replacement_registry_policy == "allow-only":
                    allowed_ids = {
                        str(unit_id)
                        for unit_id, replacement in replacements.items()
                        if isinstance(replacement, dict)
                        and replacement.get("ok")
                    }
                else:
                    allowed_ids = (
                        set(network.selection_unit_by_gate.values())
                        - invalid_ids
                    )
                network = restrict_group_editable_units(
                    network,
                    allowed_ids,
                )

            candidate = select_topk_candidate(
                network,
                by_id,
                top_tokens=args.top_tokens,
            )
            selected_ids = (
                set(candidate["selected_ids"])
                if candidate is not None
                else set()
            )
            registry_misses = sorted(
                selected_ids - known_registry_ids
                if known_registry_ids is not None
                else ()
            )
            if network.status != "ok":
                status = network.status
            elif candidate is None:
                status = "no_nonempty_graph_local_candidate"
            elif candidate["n_selected"] < args.top_tokens:
                status = "insufficient_editable_tokens"
            else:
                status = "ok"
            row = {
                "index": index,
                "id": identifier,
                "question": str(record.get("question", "")),
                "status": status,
                "baseline": "topk_graph_local_outgoing_contribution",
                "top_tokens": args.top_tokens,
                "projection": "layer-copy-token",
                "capacity_mode": "raw",
                "replacement_registry": args.replacement_registry,
                "replacement_registry_policy": (
                    args.replacement_registry_policy
                ),
                "registry_candidate_misses": registry_misses,
                "network_status": network.status,
                "network_diagnostics": network.diagnostics,
                "selection_candidate": candidate,
                "candidates": [candidate] if candidate is not None else [],
                "elapsed_seconds": round(
                    time.monotonic() - started,
                    6,
                ),
            }
            rows.append(row)
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            print(
                f"[topk-graph-token] {offset + 1}/{args.n} "
                f"status={status} "
                f"selected={candidate['n_selected'] if candidate else 0}",
                flush=True,
            )

    summary = summarize(
        rows,
        top_tokens=args.top_tokens,
        registry_path=args.replacement_registry,
    )
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    print("[topk-graph-token summary]", rendered)
    if args.summary_out:
        with open(args.summary_out, "w", encoding="utf-8") as output:
            output.write(rendered + "\n")


def select_topk_candidate(
    network,
    by_id: dict[str, dict],
    *,
    top_tokens: int,
) -> dict | None:
    """Rank editable token gates by graph-local outgoing contribution."""

    if network.status != "ok":
        return None
    available = sorted(set(network.selection_unit_by_gate.values()))
    ranked = sorted(
        available,
        key=lambda unit_id: (
            -float(network.unit_scores.get(unit_id, 0.0)),
            unit_id,
        ),
    )
    selected_ids = ranked[:top_tokens]
    if not selected_ids:
        return None
    initial_flow = remaining_support_flow(network, frozenset())
    remaining_flow = remaining_support_flow(
        network,
        frozenset(selected_ids),
    )
    return {
        "status": "fixed_topk",
        "selected_ids": selected_ids,
        "selected_tokens": [
            str(by_id[unit_id].get("text", ""))
            for unit_id in selected_ids
        ],
        "selected_scores": [
            float(network.unit_scores.get(unit_id, 0.0))
            for unit_id in selected_ids
        ],
        "n_selected": len(selected_ids),
        "initial_flow": initial_flow,
        "remaining_support_flow": remaining_flow,
        "remaining_support_fraction": (
            remaining_flow / initial_flow
            if initial_flow > 0
            else 0.0
        ),
        "score": "projected_token_outgoing_contribution",
    }


def summarize(
    rows: list[dict],
    *,
    top_tokens: int,
    registry_path: str,
) -> dict:
    selected_counts = [
        int(row["selection_candidate"]["n_selected"])
        for row in rows
        if row.get("selection_candidate")
    ]
    misses = {
        (str(row["id"]), str(unit_id))
        for row in rows
        for unit_id in row.get("registry_candidate_misses", [])
    }
    histogram: dict[str, int] = {}
    for row in rows:
        status = str(row["status"])
        histogram[status] = histogram.get(status, 0) + 1
    return {
        "baseline": "topk_graph_local_outgoing_contribution",
        "top_tokens": top_tokens,
        "queries": len(rows),
        "status_histogram": histogram,
        "queries_with_exact_topk": sum(
            count == top_tokens
            for count in selected_counts
        ),
        "queries_with_any_candidate": len(selected_counts),
        "mean_selected_tokens": (
            statistics.fmean(selected_counts)
            if selected_counts
            else None
        ),
        "replacement_registry": registry_path or None,
        "registry_fixed_point": bool(registry_path) and not misses,
        "evaluated_candidate_registry_misses": len(misses),
    }


if __name__ == "__main__":
    main()
