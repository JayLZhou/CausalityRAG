"""Audit that every executed method uses one frozen replacement pool."""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.io import load_records
from causalityrag.shared_replacement_pool import (
    FrozenSharedReplacementPool,
    file_sha256,
    stable_shared_candidate,
)


def _audit_edits(
    edits: list[dict],
    pool: FrozenSharedReplacementPool,
    *,
    seed: int,
    counters: dict[str, int],
) -> None:
    for edit in edits:
        counters["executed_edits"] += 1
        unit_id = str(edit.get("unit_id", ""))
        row = pool.get(unit_id)
        if row is None:
            counters["missing_pool_rows"] += 1
            continue
        expected = stable_shared_candidate(
            row["candidates"],
            unit_id=unit_id,
            seed=seed,
        )
        if str(edit.get("new", "")) != str(expected.get("new", "")):
            counters["replacement_mismatches"] += 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", required=True)
    parser.add_argument("--reflow-results", required=True)
    parser.add_argument("--baseline-results", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    pool = FrozenSharedReplacementPool(args.pool)
    counters = {
        "executed_edits": 0,
        "missing_pool_rows": 0,
        "replacement_mismatches": 0,
        "protocol_violations": 0,
    }
    for row in load_records(args.reflow_results):
        if str(row.get("evaluation_status", "")).startswith(
            "protocol_violation"
        ):
            counters["protocol_violations"] += 1
        for attempt in row.get("attempts", []):
            if str(attempt.get("candidate_status", "")).startswith(
                "protocol_violation"
            ):
                counters["protocol_violations"] += 1
            if attempt.get("reader_called"):
                _audit_edits(
                    attempt.get("edits", []),
                    pool,
                    seed=args.seed,
                    counters=counters,
                )
    for row in load_records(args.baseline_results):
        for method in row.get("methods", {}).values():
            if str(method.get("status", "")).startswith(
                "protocol_violation"
            ):
                counters["protocol_violations"] += 1
            _audit_edits(
                method.get("edits", []),
                pool,
                seed=args.seed,
                counters=counters,
            )

    report = {
        "pool": os.path.abspath(args.pool),
        "pool_sha256": file_sha256(args.pool),
        "replacement_seed": args.seed,
        **counters,
        "passed": all(
            counters[key] == 0
            for key in (
                "missing_pool_rows",
                "replacement_mismatches",
                "protocol_violations",
            )
        ),
    }
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit("shared replacement protocol audit failed")


if __name__ == "__main__":
    main()
