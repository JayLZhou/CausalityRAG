"""Experimental audit: compare grouped min-cut rounding with an exact MILP."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from itertools import islice

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.arc_jsd import context_sentence_units
from exp.group_flow_oracle import solve_exact_group_flow_blocker
from causalityrag.io import iter_records, load_records, record_id
from causalityrag.linguistics import SpacyAnnotationClient
from causalityrag.mixed_cut import (
    augment_with_unary_support,
    build_layered_copy_contribution_network,
    remaining_support_flow,
    restrict_group_editable_units,
)
from causalityrag.token_units import units_from_cache_row


def unary_scores_from_row(row: dict) -> dict[str, float]:
    """Read score maps produced by historical unary-support experiments."""

    direct = row.get("unary_scores")
    if isinstance(direct, dict):
        return {
            str(unit_id): float(score)
            for unit_id, score in direct.items()
        }
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
        raise ValueError(
            "unary selected IDs and scores have different lengths"
        )
    return {
        str(unit_id): float(score)
        for unit_id, score in zip(selected_ids, selected_scores)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--graphs", required=True)
    parser.add_argument("--gate", required=True)
    parser.add_argument("--unary-score-rows", required=True)
    parser.add_argument("--units-cache", default="")
    parser.add_argument("--replacement-registry", default="")
    parser.add_argument("--clean-reference", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", default="")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--max-k-guess", type=int, default=10)
    parser.add_argument("--graph-weight", type=float, default=0.5)
    parser.add_argument("--max-copies-per-unit", type=int, default=8)
    parser.add_argument("--capacity-mode", choices=("raw", "backward-conserved"), default="raw")
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--only-clean-correct", action="store_true")
    parser.add_argument(
        "--spacy-base-url",
        default=os.environ.get(
            "CAUSALITYRAG_SPACY_BASE_URL",
            "http://127.0.0.1:8021",
        ),
    )
    args = parser.parse_args()
    if not 0 < args.beta < 1:
        raise ValueError("beta must be strictly between zero and one")
    if args.eta <= 0:
        raise ValueError("eta must be positive")
    if args.max_k_guess <= 0:
        raise ValueError("max-k-guess must be positive")

    records = list(islice(iter_records(args.input), args.start, args.start + args.n))
    graph_rows = islice(iter_records(args.graphs), args.start, args.start + len(records))
    gate_by_id = _rows_by_id(args.gate)
    unary_by_id = _rows_by_id(args.unary_score_rows)
    registry_by_id = _rows_by_id(args.replacement_registry)
    reference_by_id = _rows_by_id(args.clean_reference)
    units_by_id = _rows_by_id(args.units_cache)
    nlp = None if units_by_id else SpacyAnnotationClient(args.spacy_base_url)
    if nlp is not None and not nlp.health().get("ok"):
        raise RuntimeError("spaCy annotation service is unhealthy")

    rows = []
    processed = 0
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as output:
        for offset, (record, graph_row) in enumerate(zip(records, graph_rows)):
            processed = offset + 1
            identifier = record_id(record)
            if str(graph_row.get("id")) != identifier:
                raise ValueError(
                    "input and graph rows must be aligned: "
                    f"input={identifier} graph={graph_row.get('id')}"
                )
            reference = reference_by_id.get(identifier, {})
            if args.only_clean_correct and not reference.get("clean_correct"):
                continue
            gate_row = gate_by_id.get(identifier)
            unary_row = unary_by_id.get(identifier)
            if gate_row is None or unary_row is None:
                raise ValueError(f"missing gate or unary row for {identifier}")

            cached_units = units_by_id.get(identifier)
            if cached_units is not None:
                units = units_from_cache_row(record, cached_units, k=args.k)
            else:
                units, _ = context_sentence_units(record, k=args.k, nlp=nlp)
            network = build_layered_copy_contribution_network(
                graph_row,
                units,
                capacity_mode=args.capacity_mode,
                max_copies_per_unit=args.max_copies_per_unit,
            )
            network = augment_with_unary_support(
                network,
                unary_scores_from_row(unary_row),
                graph_weight=args.graph_weight,
            )
            registry = registry_by_id.get(identifier)
            if registry is not None:
                invalid = {str(unit_id) for unit_id in registry.get("invalid", {})}
                allowed = set(network.selection_unit_by_gate.values()) - invalid
                network = restrict_group_editable_units(network, allowed)

            initial_flow = float(gate_row.get("initial_flow", 0.0))
            measured_initial = remaining_support_flow(network, frozenset())
            initial_flow_tolerance = 1e-4 * max(
                1.0,
                abs(initial_flow),
                abs(measured_initial),
            )
            if abs(initial_flow - measured_initial) > initial_flow_tolerance:
                raise ValueError(
                    f"reconstructed initial flow drift for {identifier}: "
                    f"gate={initial_flow} measured={measured_initial}"
                )
            strict_threshold = args.beta * measured_initial
            relaxed_threshold = (1.0 + args.eta) * strict_threshold
            exact_strict = solve_exact_group_flow_blocker(
                network,
                flow_threshold=strict_threshold,
                time_limit=args.time_limit,
            )
            exact_relaxed = solve_exact_group_flow_blocker(
                network,
                flow_threshold=relaxed_threshold,
                time_limit=args.time_limit,
            )
            algorithm = gate_row.get("bicriteria_candidate")
            algorithm_size = (
                int(algorithm.get("n_selected", 0)) if algorithm else None
            )
            algorithm_flow = (
                float(algorithm.get("remaining_support_flow", 0.0))
                if algorithm
                else None
            )
            rank = int(network.diagnostics.get("maximum_active_group_rank", 0))
            factor = rank * (1.0 + 1.0 / args.eta)
            strict_optimum = (
                exact_strict["n_selected"] if exact_strict["optimal"] else None
            )
            relaxed_optimum = (
                exact_relaxed["n_selected"] if exact_relaxed["optimal"] else None
            )
            theorem_applicable = bool(
                exact_strict.get("optimal")
                and strict_optimum is not None
                and strict_optimum <= args.max_k_guess
            )
            row = {
                "index": args.start + offset,
                "id": identifier,
                "question": str(record.get("question", "")),
                "clean_correct": reference.get("clean_correct"),
                "editable_labels": len(set(network.selection_unit_by_gate.values())),
                "layer_copy_gates": len(network.token_nodes_by_unit),
                "maximum_group_rank": rank,
                "initial_flow_from_gate": initial_flow,
                "normalized_channel_mass": measured_initial,
                "strict_threshold": strict_threshold,
                "relaxed_threshold": relaxed_threshold,
                "algorithm": algorithm,
                "exact_strict": exact_strict,
                "exact_relaxed": exact_relaxed,
                "same_threshold_size_ratio": (
                    algorithm_size / relaxed_optimum
                    if algorithm_size is not None and relaxed_optimum
                    else None
                ),
                "bicriteria_cardinality_factor": factor,
                "max_k_guess": args.max_k_guess,
                "theorem_applicable": theorem_applicable,
                "residual_bound_holds": (
                    algorithm_flow is not None
                    and algorithm_flow <= relaxed_threshold + 1e-8
                ),
                "cardinality_bound_holds": (
                    theorem_applicable
                    and algorithm_size is not None
                    and strict_optimum is not None
                    and algorithm_size <= factor * strict_optimum + 1e-8
                ) if theorem_applicable else None,
            }
            rows.append(row)
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            print(
                f"[group-flow-oracle] {processed}/{len(records)} "
                f"labels={row['editable_labels']} alg={algorithm_size} "
                f"optB={strict_optimum} optRelaxed={relaxed_optimum} "
                f"seconds={exact_strict.get('elapsed_seconds', 0) + exact_relaxed.get('elapsed_seconds', 0):.3f}",
                flush=True,
            )
    if processed != len(records):
        raise ValueError(f"graph file ended after {processed} rows; expected {len(records)}")

    summary = summarize(rows)
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    print("[group-flow-oracle summary]", rendered)
    if args.summary_out:
        with open(args.summary_out, "w", encoding="utf-8") as output:
            output.write(rendered + "\n")


def summarize(rows: list[dict]) -> dict:
    feasible_solved = [
        row
        for row in rows
        if row["exact_strict"]["optimal"] and row["exact_relaxed"]["optimal"]
    ]
    proven_infeasible = [
        row
        for row in rows
        if row["exact_strict"]["status"] == "infeasible"
        and row["exact_relaxed"]["status"] == "infeasible"
    ]
    ratios = [
        row["same_threshold_size_ratio"]
        for row in feasible_solved
        if row["same_threshold_size_ratio"] is not None
    ]
    strict_times = [row["exact_strict"].get("elapsed_seconds", 0.0) for row in rows]
    relaxed_times = [row["exact_relaxed"].get("elapsed_seconds", 0.0) for row in rows]
    theorem_rows = [
        row for row in feasible_solved if row.get("theorem_applicable")
    ]
    return {
        "queries": len(rows),
        "feasible_both_oracles_optimal": len(feasible_solved),
        "proven_infeasible": len(proven_infeasible),
        "algorithm_candidate_queries": sum(row["algorithm"] is not None for row in rows),
        "algorithm_same_threshold_optimal": sum(
            row["algorithm"] is not None
            and row["algorithm"]["n_selected"] == row["exact_relaxed"]["n_selected"]
            for row in feasible_solved
        ),
        "theorem_max_k_guess": max(
            (int(row.get("max_k_guess", 0)) for row in rows),
            default=None,
        ),
        "theorem_applicable_queries": len(theorem_rows),
        "theorem_residual_bound_violations": sum(
            not row["residual_bound_holds"] for row in theorem_rows
        ),
        "theorem_cardinality_bound_violations": sum(
            row["cardinality_bound_holds"] is False for row in theorem_rows
        ),
        "mean_same_threshold_size_ratio": (
            statistics.fmean(ratios) if ratios else None
        ),
        "median_same_threshold_size_ratio": (
            statistics.median(ratios) if ratios else None
        ),
        "maximum_same_threshold_size_ratio": max(ratios, default=None),
        "mean_strict_oracle_seconds": (
            statistics.fmean(strict_times) if strict_times else None
        ),
        "mean_relaxed_oracle_seconds": (
            statistics.fmean(relaxed_times) if relaxed_times else None
        ),
        "mean_editable_labels": (
            statistics.fmean(row["editable_labels"] for row in rows)
            if rows
            else None
        ),
        "mean_group_rank": (
            statistics.fmean(row["maximum_group_rank"] for row in rows)
            if rows
            else None
        ),
    }


def _rows_by_id(path: str) -> dict[str, dict]:
    if not path:
        return {}
    return {str(row.get("id")): row for row in load_records(path)}


if __name__ == "__main__":
    main()
