#!/usr/bin/env python3
"""Merge ReFlow frontier shards in frozen retrieval order."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.io import iter_records, record_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval", required=True, type=Path)
    parser.add_argument("--shard", action="append", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--summary-out", required=True, type=Path)
    parser.add_argument("--n", type=int, default=1000)
    args = parser.parse_args()

    expected_ids = [
        record_id(row) for row in iter_records(args.retrieval, limit=args.n)
    ]
    if len(expected_ids) != args.n or len(set(expected_ids)) != args.n:
        raise ValueError(f"retrieval does not contain {args.n} unique IDs")

    rows_by_id: dict[str, dict] = {}
    for path in args.shard:
        for row in iter_records(path):
            identifier = record_id(row)
            if not identifier or identifier in rows_by_id:
                raise ValueError(f"invalid or duplicate frontier row: {identifier!r}")
            rows_by_id[identifier] = row
    missing = [identifier for identifier in expected_ids if identifier not in rows_by_id]
    extra = sorted(set(rows_by_id).difference(expected_ids))
    if missing or extra:
        raise ValueError(
            f"frontier alignment failure: missing={missing[:5]} extra={extra[:5]}"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for identifier in expected_ids:
            output.write(json.dumps(rows_by_id[identifier], ensure_ascii=False) + "\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, args.out)

    ordered = [rows_by_id[identifier] for identifier in expected_ids]
    statuses = Counter(str(row.get("frontier_status", "")) for row in ordered)
    summary = {
        "queries": args.n,
        "queries_with_frontier": sum(
            bool(row.get("frontier_candidates")) for row in ordered
        ),
        "mean_frontier_size": statistics.fmean(
            int(row.get("n_frontier", 0)) for row in ordered
        ),
        "mean_editable_tokens": statistics.fmean(
            int(row.get("editable_token_units", 0)) for row in ordered
        ),
        "status_histogram": dict(sorted(statuses.items())),
        "capacity_normalization": "mean-positive-edge",
        "parts": [str(path) for path in args.shard],
        "out": str(args.out),
    }
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    temporary_summary = args.summary_out.with_suffix(
        args.summary_out.suffix + ".tmp"
    )
    temporary_summary.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_summary, args.summary_out)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
