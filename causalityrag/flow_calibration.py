"""Calibration utilities for residual contribution-flow experiments.

The functions in this module deliberately operate on already reader-labelled
frontier candidates.  This keeps price selection, calibration, and test
reporting separate from the token replacement and reader execution pipeline.
"""

from __future__ import annotations

import hashlib
import math
import statistics
from collections import defaultdict
from typing import Iterable


def stable_calibration_split(
    identifier: str,
    *,
    fraction: float,
    seed: int,
) -> bool:
    """Return whether a query belongs to the calibration split.

    The assignment is deterministic by query ID, so re-running an experiment
    does not silently move examples between calibration and test cohorts.
    """

    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be strictly between zero and one")
    digest = hashlib.sha256(f"{seed}:{identifier}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(1 << 64)
    return value < fraction


def select_threshold_candidate(candidates: Iterable[dict], beta: float) -> dict | None:
    """Return the smallest labelled frontier candidate at residual ratio ``beta``."""

    if not 0.0 < beta <= 1.0:
        raise ValueError("beta must be in (0, 1]")
    eligible = []
    for candidate in candidates:
        if "answer_changed" not in candidate:
            continue
        fraction = candidate.get("remaining_support_fraction")
        if fraction is None or float(fraction) > beta + 1e-12:
            continue
        selected = [str(unit_id) for unit_id in candidate.get("selected_ids", [])]
        if not selected:
            continue
        eligible.append(candidate)
    if not eligible:
        return None
    return min(
        eligible,
        key=lambda row: (
            int(row.get("n_selected", len(row.get("selected_ids", [])))),
            float(row.get("remaining_support_fraction", 1.0)),
            float(row.get("lambda", 0.0)),
            tuple(str(unit_id) for unit_id in row.get("selected_ids", [])),
        ),
    )


def hoeffding_nonflip_upper_bound(
    nonflips: int,
    count: int,
    *,
    candidate_count: int,
    delta: float,
) -> float:
    """Family-wise Hoeffding upper bound for the non-flip probability."""

    if count <= 0:
        return float("inf")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be strictly between zero and one")
    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive")
    empirical = nonflips / count
    radius = math.sqrt(math.log(candidate_count / delta) / (2.0 * count))
    return min(1.0, empirical + radius)


def minimum_zero_nonflip_samples(
    *,
    epsilon: float,
    delta: float,
    candidate_count: int,
) -> int:
    """Return samples needed for a zero-nonflip Hoeffding certificate."""

    if not 0.0 < epsilon < 1.0:
        raise ValueError("epsilon must be strictly between zero and one")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be strictly between zero and one")
    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive")
    return math.ceil(math.log(candidate_count / delta) / (2.0 * epsilon * epsilon))


def beta_statistics(rows: Iterable[dict], beta: float) -> dict:
    """Summarize one deterministic one-shot candidate per query."""

    selected = []
    all_rows = list(rows)
    for row in all_rows:
        candidate = select_threshold_candidate(row.get("candidates", []), beta)
        if candidate is not None:
            selected.append((row, candidate))
    flips = sum(bool(candidate.get("answer_changed")) for _, candidate in selected)
    nonflips = len(selected) - flips
    sizes = [int(candidate.get("n_selected", len(candidate.get("selected_ids", [])))) for _, candidate in selected]
    return {
        "beta": float(beta),
        "queries": len(all_rows),
        "candidate_queries": len(selected),
        "candidate_coverage": len(selected) / max(1, len(all_rows)),
        "flips": flips,
        "nonflips": nonflips,
        "conditional_flip_rate": flips / max(1, len(selected)),
        "overall_flip_rate": flips / max(1, len(all_rows)),
        "mean_selected_tokens": statistics.fmean(sizes) if sizes else 0.0,
        "overall_mean_selected_tokens": sum(sizes) / max(1, len(all_rows)),
        "selected": [
            {
                "id": str(row["id"]),
                "answer_changed": bool(candidate.get("answer_changed")),
                "n_selected": int(candidate.get("n_selected", len(candidate.get("selected_ids", [])))),
                "remaining_support_fraction": float(candidate["remaining_support_fraction"]),
            }
            for row, candidate in selected
        ],
    }


def choose_calibrated_beta(
    rows: Iterable[dict],
    betas: Iterable[float],
    *,
    epsilon: float,
    delta: float,
) -> tuple[float | None, list[dict]]:
    """Choose the largest residual threshold meeting a Hoeffding reliability target."""

    if not 0.0 < epsilon < 1.0:
        raise ValueError("epsilon must be strictly between zero and one")
    ordered_betas = sorted({float(beta) for beta in betas})
    if not ordered_betas:
        raise ValueError("at least one beta is required")
    summaries = []
    for beta in ordered_betas:
        summary = beta_statistics(rows, beta)
        summary["nonflip_ucb"] = hoeffding_nonflip_upper_bound(
            int(summary["nonflips"]),
            int(summary["candidate_queries"]),
            candidate_count=len(ordered_betas),
            delta=delta,
        )
        summary["meets_target"] = bool(summary["nonflip_ucb"] <= epsilon)
        summaries.append(summary)
    accepted = [summary for summary in summaries if summary["meets_target"]]
    return (float(accepted[-1]["beta"]) if accepted else None), summaries


def residual_reliability_bins(rows: Iterable[dict], *, bins: int) -> list[dict]:
    """Create descriptive equal-width residual-flow bins from labelled candidates.

    Candidates from the same query are not independent.  These bins are meant
    for a diagnostic plot; the Hoeffding certificate uses ``beta_statistics``
    instead, which selects at most one candidate per query.
    """

    if bins <= 0:
        raise ValueError("bins must be positive")
    grouped: dict[int, list[tuple[str, bool]]] = defaultdict(list)
    for row in rows:
        identifier = str(row["id"])
        for candidate in row.get("candidates", []):
            if "answer_changed" not in candidate:
                continue
            fraction = candidate.get("remaining_support_fraction")
            if fraction is None:
                continue
            value = min(1.0, max(0.0, float(fraction)))
            index = min(bins - 1, int(value * bins))
            grouped[index].append((identifier, bool(candidate["answer_changed"])))
    result = []
    for index in range(bins):
        observations = grouped[index]
        flips = sum(changed for _, changed in observations)
        result.append({
            "lower": index / bins,
            "upper": (index + 1) / bins,
            "candidates": len(observations),
            "unique_queries": len({identifier for identifier, _ in observations}),
            "flip_rate": flips / max(1, len(observations)),
        })
    return result
