#!/usr/bin/env python3
"""Merge contribution-graph shards in frozen retrieval order."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.io import iter_records, record_id


def merge(
    retrieval_path: Path,
    shard_paths: list[Path],
    output_path: Path,
    summary_path: Path,
    *,
    n: int,
) -> dict:
    expected_ids = [record_id(row) for row in iter_records(retrieval_path, limit=n)]
    if len(expected_ids) != n or len(set(expected_ids)) != n:
        raise ValueError(f"retrieval does not contain {n} unique IDs")

    rows_by_id: dict[str, dict] = {}
    for path in shard_paths:
        for row in iter_records(path):
            identifier = record_id(row)
            if not identifier or identifier in rows_by_id:
                raise ValueError(f"invalid or duplicate graph row: {identifier!r}")
            rows_by_id[identifier] = row
    missing = [identifier for identifier in expected_ids if identifier not in rows_by_id]
    extra = sorted(set(rows_by_id).difference(expected_ids))
    if missing or extra:
        raise ValueError(
            f"graph alignment failure: missing={missing[:5]} extra={extra[:5]}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for identifier in expected_ids:
            output.write(json.dumps(rows_by_id[identifier], ensure_ascii=False) + "\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, output_path)

    statuses = Counter(str(rows_by_id[value].get("status", "")) for value in expected_ids)
    ok_rows = [rows_by_id[value] for value in expected_ids if rows_by_id[value].get("status") == "ok"]
    summary = {
        "records": n,
        "ok": statuses.get("ok", 0),
        "failed": n - statuses.get("ok", 0),
        "status_histogram": dict(sorted(statuses.items())),
        "avg_seconds": (
            sum(float(row.get("elapsed_seconds", 0.0)) for row in ok_rows)
            / max(1, len(ok_rows))
        ),
        "method": "closed_flow_token_contribution_graph",
        "target_objective": "mean-answer-logit",
        "target_source": "frozen_vllm_results",
        "receiver_beam": 48,
        "parts": [str(path) for path in shard_paths],
        "out": str(output_path),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_summary = summary_path.with_suffix(summary_path.suffix + ".tmp")
    temporary_summary.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_summary, summary_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval", required=True, type=Path)
    parser.add_argument("--shard", action="append", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--summary-out", required=True, type=Path)
    parser.add_argument("--n", type=int, default=1000)
    args = parser.parse_args()
    print(json.dumps(merge(
        args.retrieval,
        args.shard,
        args.out,
        args.summary_out,
        n=args.n,
    ), indent=2), flush=True)


if __name__ == "__main__":
    main()
