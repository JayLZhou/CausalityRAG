"""Run the no-reader structural gate for raw parametric mixed cuts."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from itertools import islice

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.arc_jsd import context_sentence_units
from causalityrag.io import iter_records, load_records, record_id
from causalityrag.linguistics import SpacyAnnotationClient
from causalityrag.mixed_cut import (
    augment_with_unary_support,
    build_layered_copy_contribution_network,
    build_projected_token_contribution_network,
    build_raw_contribution_network,
    restrict_group_editable_units,
    solve_bicriteria_flow_interdiction,
    solve_group_bicriteria_flow_interdiction,
    sweep_mixed_cuts,
)
from causalityrag.token_units import units_from_cache_row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--graphs", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", default="")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument(
        "--indices",
        default="",
        help="comma-separated zero-based row indices; overrides --start/--n",
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--lambda-points", type=int, default=64)
    parser.add_argument("--dynamic-range", type=float, default=1e6)
    parser.add_argument(
        "--solver",
        choices=("lambda-sweep", "k-guessing", "group-k-guessing"),
        default="lambda-sweep",
    )
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--max-k-guess", type=int, default=10)
    parser.add_argument("--max-copies-per-unit", type=int, default=0)
    parser.add_argument("--unary-score-rows", default="")
    parser.add_argument("--units-cache", default="")
    parser.add_argument("--replacement-registry", default="")
    parser.add_argument(
        "--replacement-registry-policy",
        choices=("allow-only", "exclude-known-invalid"),
        default="allow-only",
    )
    parser.add_argument("--graph-weight", type=float, default=1.0)
    parser.add_argument(
        "--projection",
        choices=("input-roots", "layer-copy-token", "layer-copy-rounding"),
        default="input-roots",
    )
    parser.add_argument(
        "--capacity-mode",
        choices=("raw", "backward-conserved"),
        default="raw",
    )
    parser.add_argument(
        "--spacy-base-url",
        default=os.environ.get(
            "CAUSALITYRAG_SPACY_BASE_URL",
            "http://127.0.0.1:8021",
        ),
    )
    args = parser.parse_args()

    requested_indices = sorted({
        int(value.strip())
        for value in args.indices.split(",")
        if value.strip()
    })
    if any(index < 0 for index in requested_indices):
        raise ValueError("indices must be non-negative")
    if requested_indices:
        requested = set(requested_indices)
        indexed_rows = []
        for index, (record, graph_row) in enumerate(zip(
            iter_records(args.input),
            iter_records(args.graphs),
        )):
            if index in requested:
                indexed_rows.append((index, record, graph_row))
            if index >= requested_indices[-1]:
                break
        if len(indexed_rows) != len(requested_indices):
            found = {index for index, _, _ in indexed_rows}
            raise ValueError(
                f"missing requested indices: {sorted(requested - found)}"
            )
        work_items = iter(indexed_rows)
        expected = len(indexed_rows)
    else:
        records = list(islice(
            iter_records(args.input),
            args.start,
            args.start + args.n,
        ))
        graph_rows = islice(
            iter_records(args.graphs),
            args.start,
            args.start + len(records),
        )
        work_items = (
            (args.start + offset, record, graph_row)
            for offset, (record, graph_row) in enumerate(
                zip(records, graph_rows)
            )
        )
        expected = len(records)
    unary_by_id = (
        {
            str(row.get("id")): row
            for row in load_records(args.unary_score_rows)
        }
        if args.unary_score_rows
        else {}
    )
    registry_by_id = (
        {
            str(row.get("id")): row
            for row in load_records(args.replacement_registry)
        }
        if args.replacement_registry
        else {}
    )
    units_by_id = (
        {
            str(row.get("id")): row
            for row in load_records(args.units_cache)
        }
        if args.units_cache
        else {}
    )
    nlp = None if units_by_id else SpacyAnnotationClient(args.spacy_base_url)
    if nlp is not None and not nlp.health().get("ok"):
        raise RuntimeError("spaCy annotation service is unhealthy")

    rows = []
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as output:
        processed = 0
        for processed, (global_index, record, graph_row) in enumerate(
            work_items,
            1,
        ):
            started = time.monotonic()
            identifier = record_id(record)
            if str(graph_row.get("id")) != identifier:
                raise ValueError(
                    "input and graph rows must be aligned: "
                    f"input={identifier} graph={graph_row.get('id')}"
                )
            if units_by_id:
                cached_units = units_by_id.get(identifier)
                if cached_units is None:
                    raise ValueError(f"missing token-units cache row for {identifier}")
                units = units_from_cache_row(record, cached_units, k=args.k)
            else:
                units, _ = context_sentence_units(record, k=args.k, nlp=nlp)
            by_id = {str(unit["unit_id"]): unit for unit in units}
            if args.projection == "layer-copy-token":
                network = build_projected_token_contribution_network(
                    graph_row,
                    units,
                    capacity_mode=args.capacity_mode,
                )
            elif args.projection == "layer-copy-rounding":
                network = build_layered_copy_contribution_network(
                    graph_row,
                    units,
                    capacity_mode=args.capacity_mode,
                    max_copies_per_unit=(
                        args.max_copies_per_unit or None
                    ),
                )
            else:
                network = build_raw_contribution_network(
                    graph_row,
                    units,
                    capacity_mode=args.capacity_mode,
                )
            if unary_by_id:
                unary_row = unary_by_id.get(identifier)
                if unary_row is None:
                    raise ValueError(f"missing unary scores for {identifier}")
                network = augment_with_unary_support(
                    network,
                    unary_scores_from_row(unary_row),
                    graph_weight=args.graph_weight,
                )
            if registry_by_id:
                registry_row = registry_by_id.get(identifier)
                if registry_row is None:
                    raise ValueError(f"missing replacement registry row for {identifier}")
                replacements = registry_row.get("replacements", {})
                valid_ids = {
                    str(unit_id)
                    for unit_id, replacement in replacements.items()
                    if isinstance(replacement, dict) and replacement.get("ok")
                }
                if args.replacement_registry_policy == "exclude-known-invalid":
                    invalid_ids = {
                        str(unit_id)
                        for unit_id in registry_row.get("invalid", {})
                    }
                    allowed_ids = (
                        set(network.selection_unit_by_gate.values()) - invalid_ids
                    )
                else:
                    allowed_ids = valid_ids
                network = restrict_group_editable_units(
                    network,
                    allowed_ids,
                )
            if (
                args.solver == "group-k-guessing"
                and args.projection != "layer-copy-rounding"
            ):
                raise ValueError(
                    "group-k-guessing requires layer-copy-rounding projection"
                )
            if (
                args.projection == "layer-copy-rounding"
                and args.solver != "group-k-guessing"
            ):
                raise ValueError(
                    "layer-copy-rounding requires group-k-guessing solver"
                )
            sweep = (
                solve_group_bicriteria_flow_interdiction(
                    network,
                    beta=args.beta,
                    eta=args.eta,
                    max_k_guess=args.max_k_guess,
                )
                if args.solver == "group-k-guessing"
                else solve_bicriteria_flow_interdiction(
                    network,
                    beta=args.beta,
                    eta=args.eta,
                    max_k_guess=args.max_k_guess,
                )
                if args.solver == "k-guessing"
                else sweep_mixed_cuts(
                    network,
                    points=args.lambda_points,
                    dynamic_range=args.dynamic_range,
                )
            )
            candidates = []
            for candidate in sweep["candidates"]:
                candidates.append({
                    **candidate,
                    "selected_tokens": [
                        str(by_id[unit_id].get("text", ""))
                        for unit_id in candidate["selected_ids"]
                    ],
                    "unary_matched_tokens": [
                        str(by_id[unit_id].get("text", ""))
                        for unit_id in candidate["unary_matched_ids"]
                    ],
                })
            budget_candidates = []
            for candidate in sweep.get("budget_candidates", []):
                budget_candidates.append({
                    **candidate,
                    "selected_tokens": [
                        str(by_id[unit_id].get("text", ""))
                        for unit_id in candidate["selected_ids"]
                    ],
                    "unary_matched_tokens": [
                        str(by_id[unit_id].get("text", ""))
                        for unit_id in candidate["unary_matched_ids"]
                    ],
                })
            row = {
                "index": global_index,
                "id": identifier,
                "question": str(record.get("question", "")),
                "graph_method": str(graph_row.get("method", "")),
                "projection": args.projection,
                "capacity_mode": args.capacity_mode,
                "solver": args.solver,
                "unary_score_rows": args.unary_score_rows,
                "units_cache": args.units_cache,
                "replacement_registry": args.replacement_registry,
                "replacement_registry_policy": args.replacement_registry_policy,
                "graph_weight": args.graph_weight,
                "elapsed_seconds": round(time.monotonic() - started, 6),
                "network_status": network.status,
                "network_diagnostics": network.diagnostics,
                **sweep,
                "candidates": candidates,
                "budget_candidates": budget_candidates,
            }
            rows.append(row)
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            print(
                f"[raw-mixed-cut] {processed}/{expected} "
                f"status={sweep['status']} "
                f"edges={network.diagnostics.get('active_edges', 0)} "
                f"sets={sweep['diagnostics'].get('distinct_candidate_sets', 0)} "
                f"different={any(c['differs_from_unary'] for c in candidates)} "
                f"seconds={row['elapsed_seconds']}",
                flush=True,
            )
        if processed != expected:
            raise ValueError(
                f"graph file ended after {processed} rows; expected {expected}"
            )

    summary = summarize(rows)
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    print("[raw-mixed-cut summary]", rendered)
    if args.summary_out:
        with open(args.summary_out, "w", encoding="utf-8") as output:
            output.write(rendered + "\n")


def summarize(rows: list[dict]) -> dict:
    ok = [row for row in rows if row["status"] == "ok"]
    nonempty_counts = [
        row["diagnostics"]["distinct_nonempty_candidate_sets"] for row in ok
    ]
    comparisons = [
        candidate
        for row in ok
        for candidate in row["candidates"]
        if candidate["n_selected"] > 0
    ]
    k_guessing = [
        row for row in ok
        if "k_guessing" in str(row["diagnostics"].get("solver", ""))
    ]
    elapsed = sorted(float(row.get("elapsed_seconds", 0.0)) for row in rows)
    return {
        "queries": len(rows),
        "ok_queries": len(ok),
        "status_histogram": _histogram(row["status"] for row in rows),
        "queries_with_multiple_nonempty_sets": sum(
            count >= 2 for count in nonempty_counts
        ),
        "queries_with_any_set_different_from_unary": sum(
            any(
                candidate["n_selected"] > 0
                and candidate["differs_from_unary"]
                for candidate in row["candidates"]
            )
            for row in ok
        ),
        "mean_distinct_nonempty_sets": (
            statistics.fmean(nonempty_counts) if nonempty_counts else None
        ),
        "candidate_comparisons": len(comparisons),
        "mixed_flow_better_than_unary": sum(
            candidate["flow_improvement_over_unary"] > 1e-9
            for candidate in comparisons
        ),
        "mixed_flow_equal_to_unary": sum(
            abs(candidate["flow_improvement_over_unary"]) <= 1e-9
            for candidate in comparisons
        ),
        "mixed_flow_worse_than_unary": sum(
            candidate["flow_improvement_over_unary"] < -1e-9
            for candidate in comparisons
        ),
        "all_cardinality_sweeps_monotone": all(
            row["diagnostics"].get(
                "cardinality_monotone_over_descending_lambda",
                True,
            )
            for row in ok
        ),
        "k_guessing_queries": len(k_guessing),
        "k_guessing_strict_candidate_queries": sum(
            bool(row["diagnostics"].get("strict_candidate_found"))
            for row in k_guessing
        ),
        "k_guessing_bicriteria_candidate_queries": sum(
            bool(row["diagnostics"].get("bicriteria_candidate_found"))
            for row in k_guessing
        ),
        "k_guessing_mincut_calls": sum(
            int(row["diagnostics"].get("mincut_calls", 0))
            for row in k_guessing
        ),
        "total_seconds": sum(elapsed),
        "mean_seconds": statistics.fmean(elapsed) if elapsed else None,
        "p90_seconds": (
            elapsed[min(len(elapsed) - 1, int(0.9 * len(elapsed)))]
            if elapsed
            else None
        ),
        "gate_pass": bool(ok) and all((
            any(count >= 2 for count in nonempty_counts),
            any(
                candidate["differs_from_unary"] for candidate in comparisons
            ),
            any(
                candidate["flow_improvement_over_unary"] > 1e-9
                for candidate in comparisons
            ),
        )),
    }


def _histogram(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[str(value)] = counts.get(str(value), 0) + 1
    return counts


def unary_scores_from_row(row: dict) -> dict[str, float]:
    """Read either a score map or the largest cached unary prefix."""

    direct = row.get("unary_scores")
    if isinstance(direct, dict):
        return {str(unit_id): float(score) for unit_id, score in direct.items()}
    prefixes = row.get("prefixes", {})
    numeric = [
        (int(key), value)
        for key, value in prefixes.items()
        if str(key).isdigit() and isinstance(value, dict)
    ]
    if not numeric:
        raise ValueError("unary score row has no score map or numeric prefix")
    _, prefix = max(numeric, key=lambda item: item[0])
    selected_ids = prefix.get("selected_ids", [])
    selected_scores = prefix.get("selected_scores", [])
    if len(selected_ids) != len(selected_scores):
        raise ValueError("unary selected IDs and scores have different lengths")
    return {
        str(unit_id): float(score)
        for unit_id, score in zip(selected_ids, selected_scores)
    }


if __name__ == "__main__":
    main()
