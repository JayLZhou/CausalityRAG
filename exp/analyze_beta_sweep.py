"""Analyze a frozen global-beta sweep and simulate submodular pilot selection.

This is an exploratory utility. Facility-location uses only graph/selection
features. Reader flips are consulted afterwards to simulate beta selection on
the chosen pilot set; they are never included in the query similarity.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-dir", required=True)
    parser.add_argument(
        "--scope",
        choices=("clean-exact", "all-queries"),
        default="clean-exact",
    )
    parser.add_argument("--pilot-sizes", default="10,25,50,100")
    parser.add_argument("--strata", type=int, default=10)
    parser.add_argument("--random-trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    pilot_sizes = sorted({
        int(value)
        for value in args.pilot_sizes.split(",")
        if value.strip()
    })
    if not pilot_sizes or min(pilot_sizes) <= 0:
        parser.error("pilot sizes must be positive")
    if args.random_trials < 0:
        parser.error("random trials must be non-negative")
    if args.strata <= 0:
        parser.error("strata must be positive")

    entries = discover_entries(Path(args.experiment_dir))
    dataset = build_dataset(entries, scope=args.scope)
    if max(pilot_sizes) > len(dataset["ids"]):
        parser.error("pilot size exceeds the selected query scope")

    standardized = standardize_features(dataset["features"])
    similarity, sigma_squared = rbf_similarity(standardized)
    pilot_order = facility_location_greedy(
        similarity,
        max(pilot_sizes),
    )
    full_metrics = metrics_by_beta(
        dataset["flips"],
        dataset["coverage"],
        dataset["edits"],
    )
    full_best = best_beta_index(full_metrics)
    rng = np.random.default_rng(args.seed)
    simulations = []
    for size in pilot_sizes:
        selected = pilot_order[:size]
        weights = facility_cluster_weights(similarity, selected)
        pilot_metrics = metrics_by_beta(
            dataset["flips"][selected],
            dataset["coverage"][selected],
            dataset["edits"][selected],
            weights=weights,
        )
        pilot_best = best_beta_index(pilot_metrics)
        random_choices = []
        stratified_choices = []
        strata_centers = pilot_order[: min(args.strata, size)]
        assignments = facility_assignments(
            similarity,
            strata_centers,
        )
        for _ in range(args.random_trials):
            random_selected = rng.choice(
                len(dataset["ids"]),
                size=size,
                replace=False,
            )
            random_metrics = metrics_by_beta(
                dataset["flips"][random_selected],
                dataset["coverage"][random_selected],
                dataset["edits"][random_selected],
            )
            random_choices.append(best_beta_index(random_metrics))
            stratified_selected, stratified_weights = (
                stratified_random_sample(
                    assignments,
                    size=size,
                    rng=rng,
                )
            )
            stratified_metrics = metrics_by_beta(
                dataset["flips"][stratified_selected],
                dataset["coverage"][stratified_selected],
                dataset["edits"][stratified_selected],
                weights=stratified_weights,
            )
            stratified_choices.append(
                best_beta_index(stratified_metrics)
            )
        simulations.append({
            "pilot_size": size,
            "pilot_fraction": size / len(dataset["ids"]),
            "selected_beta": entries[pilot_best]["beta"],
            "selected_remaining_flow_threshold": entries[pilot_best][
                "remaining_flow_threshold"
            ],
            "matches_full_best": pilot_best == full_best,
            "weighted_pilot_metrics": pilot_metrics[pilot_best],
            "full_metrics_at_selected_beta": full_metrics[pilot_best],
            "full_overall_flip_regret": (
                full_metrics[full_best]["overall_flip_rate"]
                - full_metrics[pilot_best]["overall_flip_rate"]
            ),
            "facility_objective": facility_objective(
                similarity,
                selected,
            ),
            "pilot_query_ids": [
                dataset["ids"][index]
                for index in selected
            ],
            "random_trials": args.random_trials,
            "random_match_rate": (
                sum(index == full_best for index in random_choices)
                / len(random_choices)
                if random_choices
                else None
            ),
            "random_mean_overall_flip_regret": (
                float(np.mean([
                    full_metrics[full_best]["overall_flip_rate"]
                    - full_metrics[index]["overall_flip_rate"]
                    for index in random_choices
                ]))
                if random_choices
                else None
            ),
            "random_selected_beta_histogram": {
                str(entries[index]["beta"]): random_choices.count(index)
                for index in sorted(set(random_choices))
            },
            "strata": len(strata_centers),
            "stratified_random_match_rate": (
                sum(index == full_best for index in stratified_choices)
                / len(stratified_choices)
                if stratified_choices
                else None
            ),
            "stratified_mean_overall_flip_regret": (
                float(np.mean([
                    full_metrics[full_best]["overall_flip_rate"]
                    - full_metrics[index]["overall_flip_rate"]
                    for index in stratified_choices
                ]))
                if stratified_choices
                else None
            ),
            "stratified_selected_beta_histogram": {
                str(entries[index]["beta"]): stratified_choices.count(index)
                for index in sorted(set(stratified_choices))
            },
        })

    result = {
        "experiment_dir": str(Path(args.experiment_dir).resolve()),
        "scope": args.scope,
        "queries": len(dataset["ids"]),
        "profile": (
            "[candidate_exists, selected/editable_units, remaining_flow]"
            "_per_beta"
        ),
        "similarity": "rbf_over_standardized_flow_profile",
        "rbf_sigma_squared": sigma_squared,
        "objective": (
            "maximize end-to-end flip rate; tie-break by lower edits/query"
        ),
        "warning": (
            "exploratory same-sample simulation; an independent random audit "
            "is required for a statistical certificate"
        ),
        "betas": [
            {
                "label": entry["label"],
                "beta": entry["beta"],
                "remaining_flow_threshold": entry[
                    "remaining_flow_threshold"
                ],
                "full_metrics": metric,
            }
            for entry, metric in zip(entries, full_metrics)
        ],
        "full_best_beta": entries[full_best]["beta"],
        "full_best_remaining_flow_threshold": entries[full_best][
            "remaining_flow_threshold"
        ],
        "full_best_metrics": full_metrics[full_best],
        "submodular_pilot_simulation": simulations,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as output:
        json.dump(result, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(json.dumps(
        {
            "scope": result["scope"],
            "queries": result["queries"],
            "full_best_beta": result["full_best_beta"],
            "full_best_metrics": result["full_best_metrics"],
            "pilots": [
                {
                    "pilot_size": row["pilot_size"],
                    "selected_beta": row["selected_beta"],
                    "matches_full_best": row["matches_full_best"],
                    "random_match_rate": row["random_match_rate"],
                    "full_overall_flip_regret": row[
                        "full_overall_flip_regret"
                    ],
                    "stratified_random_match_rate": row[
                        "stratified_random_match_rate"
                    ],
                    "stratified_mean_overall_flip_regret": row[
                        "stratified_mean_overall_flip_regret"
                    ],
                }
                for row in simulations
            ],
        },
        ensure_ascii=False,
        indent=2,
    ))


def discover_entries(experiment_dir: Path) -> list[dict]:
    flow_dir = experiment_dir / "flow" / "final"
    evaluation_dir = experiment_dir / "evaluation"
    entries = []
    for flow_path in sorted(flow_dir.glob("*.jsonl")):
        label = flow_path.stem
        evaluation_path = evaluation_dir / label / "all_queries.jsonl"
        summary_path = evaluation_dir / label / "all_queries.summary.json"
        if not evaluation_path.exists() or not summary_path.exists():
            continue
        beta = None
        eta = None
        with open(flow_path, encoding="utf-8") as source:
            for line in source:
                row = json.loads(line)
                diagnostics = row.get("diagnostics", {})
                if diagnostics.get("beta") is not None:
                    beta = float(diagnostics["beta"])
                    eta = float(diagnostics.get("eta", 1.0))
                    break
        if beta is None or eta is None:
            raise ValueError(f"cannot recover beta/eta from {flow_path}")
        summary = json.load(open(summary_path, encoding="utf-8"))
        remaining = float(summary["remaining_flow_threshold"])
        expected = (1.0 + eta) * beta
        if abs(remaining - expected) > 1e-9:
            raise ValueError(
                f"threshold mismatch for {label}: {remaining} != {expected}"
            )
        entries.append({
            "label": label,
            "beta": beta,
            "eta": eta,
            "remaining_flow_threshold": remaining,
            "flow_path": flow_path,
            "evaluation_path": evaluation_path,
        })
    if not entries:
        raise ValueError("no complete beta entries found")
    entries.sort(key=lambda row: row["beta"])
    return entries


def build_dataset(entries: list[dict], *, scope: str) -> dict:
    ids = None
    feature_columns = []
    flip_columns = []
    coverage_columns = []
    edit_columns = []
    for entry in entries:
        flow_by_id = {
            str(row["id"]): row
            for row in load_jsonl(entry["flow_path"])
        }
        evaluation_rows = load_jsonl(entry["evaluation_path"])
        if scope == "clean-exact":
            evaluation_rows = [
                row
                for row in evaluation_rows
                if row.get("clean_correct_exact")
            ]
        current_ids = [str(row["id"]) for row in evaluation_rows]
        if ids is None:
            ids = current_ids
        elif current_ids != ids:
            raise ValueError("evaluation rows are not aligned across betas")
        features = []
        flips = []
        coverage = []
        edits = []
        for row in evaluation_rows:
            identifier = str(row["id"])
            flow_row = flow_by_id[identifier]
            method = str(row["evaluated_method"])
            result = row["methods"][method]
            candidate = result["status"] != "no_candidate_under_selection_rule"
            editable_units = max(
                1,
                int(flow_row.get("network_diagnostics", {}).get(
                    "editable_units",
                    0,
                )),
            )
            n_selected = int(result.get("n_selected", 0))
            remaining = row.get("candidate_remaining_support_fraction")
            features.append([
                float(candidate),
                n_selected / editable_units,
                float(remaining) if remaining is not None else 1.0,
            ])
            flips.append(float(bool(result.get("flip"))))
            coverage.append(float(candidate))
            edits.append(float(n_selected))
        feature_columns.append(np.asarray(features, dtype=np.float64))
        flip_columns.append(np.asarray(flips, dtype=np.float64))
        coverage_columns.append(np.asarray(coverage, dtype=np.float64))
        edit_columns.append(np.asarray(edits, dtype=np.float64))
    return {
        "ids": ids or [],
        "features": np.concatenate(feature_columns, axis=1),
        "flips": np.stack(flip_columns, axis=1),
        "coverage": np.stack(coverage_columns, axis=1),
        "edits": np.stack(edit_columns, axis=1),
    }


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as source:
        return [json.loads(line) for line in source]


def standardize_features(features: np.ndarray) -> np.ndarray:
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return (features - mean) / scale


def rbf_similarity(features: np.ndarray) -> tuple[np.ndarray, float]:
    squared_norm = np.sum(features * features, axis=1)
    distance = (
        squared_norm[:, None]
        + squared_norm[None, :]
        - 2.0 * features @ features.T
    )
    distance = np.maximum(distance, 0.0)
    positive = distance[np.triu_indices(len(features), k=1)]
    positive = positive[positive > 1e-12]
    sigma_squared = float(np.median(positive)) if len(positive) else 1.0
    similarity = np.exp(-distance / sigma_squared)
    np.fill_diagonal(similarity, 1.0)
    return similarity, sigma_squared


def facility_location_greedy(
    similarity: np.ndarray,
    size: int,
) -> list[int]:
    if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1]:
        raise ValueError("similarity must be square")
    if not 0 < size <= len(similarity):
        raise ValueError("invalid facility set size")
    current = np.zeros(len(similarity), dtype=np.float64)
    selected = []
    available = np.ones(len(similarity), dtype=bool)
    for _ in range(size):
        improvement = np.maximum(
            similarity,
            current[:, None],
        ) - current[:, None]
        gains = improvement.sum(axis=0)
        gains[~available] = -np.inf
        choice = int(np.argmax(gains))
        selected.append(choice)
        available[choice] = False
        current = np.maximum(current, similarity[:, choice])
    return selected


def facility_cluster_weights(
    similarity: np.ndarray,
    selected: list[int],
) -> np.ndarray:
    assignments = np.argmax(similarity[:, selected], axis=1)
    return np.bincount(
        assignments,
        minlength=len(selected),
    ).astype(np.float64)


def facility_assignments(
    similarity: np.ndarray,
    selected: list[int],
) -> np.ndarray:
    if not selected:
        raise ValueError("at least one facility is required")
    return np.argmax(similarity[:, selected], axis=1)


def stratified_random_sample(
    assignments: np.ndarray,
    *,
    size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample randomly within facility strata and return design weights."""

    strata = [
        np.flatnonzero(assignments == stratum)
        for stratum in sorted(set(assignments.tolist()))
    ]
    strata = [indices for indices in strata if len(indices)]
    if size < len(strata) or size > len(assignments):
        raise ValueError("sample size must cover every nonempty stratum")
    allocation = np.ones(len(strata), dtype=int)
    remaining = size - len(strata)
    while remaining:
        available = [
            index
            for index, indices in enumerate(strata)
            if allocation[index] < len(indices)
        ]
        if not available:
            raise ValueError("stratified allocation exhausted unexpectedly")
        choice = max(
            available,
            key=lambda index: (
                len(strata[index]) / (allocation[index] + 1),
                -index,
            ),
        )
        allocation[choice] += 1
        remaining -= 1

    selected = []
    weights = []
    for indices, count in zip(strata, allocation):
        sampled = rng.choice(indices, size=int(count), replace=False)
        selected.extend(int(index) for index in sampled)
        weights.extend([len(indices) / count] * int(count))
    return (
        np.asarray(selected, dtype=int),
        np.asarray(weights, dtype=np.float64),
    )


def facility_objective(
    similarity: np.ndarray,
    selected: list[int],
) -> float:
    return float(similarity[:, selected].max(axis=1).sum())


def metrics_by_beta(
    flips: np.ndarray,
    coverage: np.ndarray,
    edits: np.ndarray,
    *,
    weights: np.ndarray | None = None,
) -> list[dict]:
    if weights is None:
        weights = np.ones(len(flips), dtype=np.float64)
    denominator = float(weights.sum())
    if denominator <= 0:
        raise ValueError("metric weights must have positive mass")
    metrics = []
    for index in range(flips.shape[1]):
        flip_count = float(weights @ flips[:, index])
        candidate_count = float(weights @ coverage[:, index])
        edit_count = float(weights @ edits[:, index])
        metrics.append({
            "queries": denominator,
            "candidate_queries": candidate_count,
            "flips": flip_count,
            "overall_flip_rate": flip_count / denominator,
            "candidate_coverage": candidate_count / denominator,
            "mean_edits_per_query": edit_count / denominator,
            "mean_edits_per_candidate": (
                edit_count / candidate_count
                if candidate_count
                else None
            ),
        })
    return metrics


def best_beta_index(metrics: list[dict]) -> int:
    return min(
        range(len(metrics)),
        key=lambda index: (
            -float(metrics[index]["overall_flip_rate"]),
            float(metrics[index]["mean_edits_per_query"]),
            index,
        ),
    )


if __name__ == "__main__":
    main()
