"""Calibrate a residual-flow threshold from reader-labelled frontier outputs.

Run ``run_contribution_aware_flow_contract_attack.py --evaluate-all-frontier``
first.  This script then uses a deterministic query-level split to select a
threshold on calibration queries and reports its held-out reliability.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.flow_calibration import (  # noqa: E402
    beta_statistics,
    choose_calibrated_beta,
    minimum_zero_nonflip_samples,
    residual_reliability_bins,
    stable_calibration_split,
)
from causalityrag.io import load_records  # noqa: E402


def parse_betas(value: str) -> list[float]:
    betas = sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    if not betas or any(beta <= 0.0 or beta > 1.0 for beta in betas):
        raise ValueError("--betas must contain comma-separated values in (0, 1]")
    return betas


def normalise_rows(raw_rows: list[dict]) -> list[dict]:
    rows = []
    for raw in raw_rows:
        attempts = {
            tuple(str(unit_id) for unit_id in attempt.get("selected_ids", [])): attempt
            for attempt in raw.get("attempts", [])
        }
        candidates = []
        missing_labels = []
        for candidate in raw.get("frontier_candidates", []):
            key = tuple(str(unit_id) for unit_id in candidate.get("selected_ids", []))
            attempt = attempts.get(key)
            if attempt is None:
                if key:
                    missing_labels.append(key)
                continue
            candidates.append({**candidate, "answer_changed": bool(attempt.get("answer_changed"))})
        if not raw.get("frontier_candidates"):
            candidates = [dict(attempt) for attempt in raw.get("attempts", [])]
        elif missing_labels:
            raise ValueError(
                f"frontier row {raw.get('id')} has {len(missing_labels)} unlabeled "
                "candidate(s); rerun with --evaluate-all-frontier and a large "
                "enough --max-verify"
            )
        rows.append({"id": str(raw["id"]), "candidates": candidates})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frontier-results", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--betas", default="0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50,0.60,0.70,0.80")
    parser.add_argument("--calibration-fraction", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--bins", type=int, default=10)
    args = parser.parse_args()

    raw_rows = load_records(args.frontier_results)
    rows = normalise_rows(raw_rows)
    calibration_rows = [
        row for row in rows
        if stable_calibration_split(
            row["id"], fraction=args.calibration_fraction, seed=args.seed
        )
    ]
    test_rows = [
        row for row in rows
        if not stable_calibration_split(
            row["id"], fraction=args.calibration_fraction, seed=args.seed
        )
    ]
    if not calibration_rows or not test_rows:
        raise ValueError("query split produced an empty calibration or test cohort")
    betas = parse_betas(args.betas)
    beta, calibration = choose_calibrated_beta(
        calibration_rows,
        betas,
        epsilon=args.epsilon,
        delta=args.delta,
    )
    test = beta_statistics(test_rows, beta) if beta is not None else None
    summary = {
        "protocol": {
            "candidate_unit": "one smallest labelled frontier candidate per query per beta",
            "split": "deterministic query-ID hash split",
            "calibration_fraction": args.calibration_fraction,
            "seed": args.seed,
            "epsilon": args.epsilon,
            "delta": args.delta,
            "betas": betas,
            "minimum_calibration_candidates_for_zero_nonflips": (
                minimum_zero_nonflip_samples(
                    epsilon=args.epsilon,
                    delta=args.delta,
                    candidate_count=len(betas),
                )
            ),
            "certificate": (
                "family-wise Hoeffding upper bound on the calibration non-flip "
                "probability; applies to exchangeable future queries"
            ),
        },
        "queries": {
            "all": len(rows),
            "calibration": len(calibration_rows),
            "test": len(test_rows),
        },
        "calibration_by_beta": calibration,
        "selected_beta": beta,
        "held_out_test": test,
        "descriptive_reliability_bins": residual_reliability_bins(rows, bins=args.bins),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as output:
        json.dump(summary, output, ensure_ascii=False, indent=2)
        output.write("\n")
    concise_test = None
    if test is not None:
        concise_test = {key: value for key, value in test.items() if key != "selected"}
    print(json.dumps({
        "selected_beta": beta,
        "calibration_queries": len(calibration_rows),
        "test_queries": len(test_rows),
        "held_out_test": concise_test,
        "out": args.out,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
