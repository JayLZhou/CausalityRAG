#!/usr/bin/env python3
"""Materialize one-shot and fixed-grid alternatives to the exact frontier."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load_rows(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def objective(candidate: dict, price: float) -> float:
    return float(candidate["remaining_support_flow"]) + price * int(
        candidate.get("n_selected", len(candidate.get("selected_ids", [])))
    )


def demand(row: dict, price: float) -> dict | None:
    candidates = [dict(candidate) for candidate in row.get("frontier_candidates", [])]
    empty_cost = float(row.get("initial_support_flow", 0.0))
    nonempty = [candidate for candidate in candidates if candidate.get("selected_ids")]
    if not nonempty:
        return None
    best = min(
        nonempty,
        key=lambda candidate: (
            objective(candidate, price),
            int(candidate.get("n_selected", 0)),
            tuple(str(value) for value in candidate.get("selected_ids", [])),
        ),
    )
    if objective(best, price) >= empty_cost - 1e-9:
        return None
    return {**best, "evaluation_price": price}


def protocol_row(row: dict, candidates: list[dict], protocol: str, calls: int) -> dict:
    candidates.sort(
        key=lambda candidate: (
            int(candidate.get("n_selected", 0)),
            float(candidate.get("remaining_support_flow", float("inf"))),
            tuple(str(value) for value in candidate.get("selected_ids", [])),
        )
    )
    return {
        **row,
        "method": protocol,
        "n_frontier": len(candidates),
        "frontier_candidates": candidates,
        "frontier_diagnostics": {
            "protocol": protocol,
            "fixed_price_min_cut_calls": calls,
            "candidate_source": "exact_supported_frontier",
        },
    }


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontier", type=Path, required=True)
    parser.add_argument("--one-shot-out", type=Path, required=True)
    parser.add_argument("--sweep-out", type=Path, required=True)
    parser.add_argument("--one-shot-price", type=float, default=1.0)
    parser.add_argument("--grid-min-exp", type=float, default=-8.0)
    parser.add_argument("--grid-max-exp", type=float, default=8.0)
    parser.add_argument("--grid-step", type=float, default=0.5)
    args = parser.parse_args()

    rows = load_rows(args.frontier)
    steps = int(round((args.grid_max_exp - args.grid_min_exp) / args.grid_step))
    prices = [
        2.0 ** (args.grid_min_exp + index * args.grid_step)
        for index in range(steps + 1)
    ]
    one_shot_rows = []
    sweep_rows = []
    for row in rows:
        selected = demand(row, args.one_shot_price)
        one_shot_rows.append(
            protocol_row(
                row,
                [selected] if selected is not None else [],
                "fixed_price_one_shot",
                1,
            )
        )
        by_set = {}
        for price in prices:
            candidate = demand(row, price)
            if candidate is None:
                continue
            key = tuple(sorted(str(value) for value in candidate["selected_ids"]))
            by_set.setdefault(key, candidate)
        sweep_rows.append(
            protocol_row(
                row,
                list(by_set.values()),
                "fixed_grid_price_sweep",
                len(prices),
            )
        )

    write_rows(args.one_shot_out, one_shot_rows)
    write_rows(args.sweep_out, sweep_rows)
    print(json.dumps({
        "queries": len(rows),
        "one_shot_price": args.one_shot_price,
        "sweep_prices": prices,
        "mean_one_shot_candidates": sum(
            len(row["frontier_candidates"]) for row in one_shot_rows
        ) / max(1, len(rows)),
        "mean_sweep_candidates": sum(
            len(row["frontier_candidates"]) for row in sweep_rows
        ) / max(1, len(rows)),
    }, indent=2))


if __name__ == "__main__":
    main()
