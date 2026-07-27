"""Merge ReFlow JSONL shards and compute one validated experiment summary."""

from __future__ import annotations

import argparse
import json
import os
import statistics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--expected-rows", type=int, default=0)
    args = parser.parse_args()

    rows = []
    for path in args.inputs:
        with open(path, encoding="utf-8") as source:
            rows.extend(json.loads(line) for line in source if line.strip())
    rows.sort(key=lambda row: int(row["index"]))
    indices = [int(row["index"]) for row in rows]
    if len(indices) != len(set(indices)):
        raise ValueError("duplicate query indices across ReFlow shards")
    if args.expected_rows and indices != list(range(args.expected_rows)):
        missing = sorted(set(range(args.expected_rows)) - set(indices))
        unexpected = sorted(set(indices) - set(range(args.expected_rows)))
        raise ValueError(
            f"incomplete shard coverage: missing={missing[:10]}, "
            f"unexpected={unexpected[:10]}"
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    temporary = args.out + ".tmp"
    with open(temporary, "w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, args.out)

    flips = [row for row in rows if row.get("verified_flip")]
    terminal_sizes = []
    cumulative_sizes = []
    for row in rows:
        sizes = [
            len(attempt.get("selected_ids", []))
            for attempt in row.get("attempts", [])
        ]
        cumulative_sizes.append(sum(sizes))
        terminal_sizes.append(
            int(row.get("n_selected", 0))
            if row.get("verified_flip")
            else (max(sizes) if sizes else 0)
        )
    summary = {
        "queries": len(rows),
        "query_index_min": min(indices) if indices else None,
        "query_index_max": max(indices) if indices else None,
        "unique_query_indices": len(set(indices)),
        "flips": len(flips),
        "overall_flip_rate": len(flips) / max(1, len(rows)),
        "success_mean_tokens": (
            statistics.fmean(int(row["n_selected"]) for row in flips)
            if flips
            else 0.0
        ),
        "terminal_mean_tokens": (
            statistics.fmean(terminal_sizes) if terminal_sizes else 0.0
        ),
        "cumulative_attempted_tokens_mean": (
            statistics.fmean(cumulative_sizes) if cumulative_sizes else 0.0
        ),
        "mean_reader_calls": statistics.fmean(
            int(row.get("reader_calls", 0)) for row in rows
        ) if rows else 0.0,
        "total_reader_calls": sum(
            int(row.get("reader_calls", 0)) for row in rows
        ),
        "mean_frontier": statistics.fmean(
            int(row.get("n_frontier", 0)) for row in rows
        ) if rows else 0.0,
        "editor_llm_calls": sum(
            int(row.get("editor_llm_calls", 0)) for row in rows
        ),
        "queries_without_frontier": sum(
            not row.get("frontier_candidates") for row in rows
        ),
        "graph_type": "answer_conditioned_token_contribution_graph",
        "frontier_mode": "exact_breakpoint_hull",
        "replacement_protocol": "on_demand_after_selection",
        "inputs": [os.path.abspath(path) for path in args.inputs],
        "out": os.path.abspath(args.out),
    }
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    os.makedirs(
        os.path.dirname(os.path.abspath(args.summary_out)),
        exist_ok=True,
    )
    with open(args.summary_out, "w", encoding="utf-8") as output:
        output.write(rendered + "\n")
    print("[merge-reflow-shards summary] " + rendered, flush=True)


if __name__ == "__main__":
    main()
