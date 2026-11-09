#!/usr/bin/env python3
"""Prepare aligned data shards and merge matched-budget evaluation outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare(args: argparse.Namespace) -> None:
    tables = {
        "input": read_jsonl(args.input),
        "units": read_jsonl(args.units),
        "reflow": read_jsonl(args.reflow),
        "scores": read_jsonl(args.scores),
    }
    lengths = {len(rows) for rows in tables.values()}
    if lengths != {args.shards * args.shard_size}:
        raise ValueError(f"unaligned or unexpected table lengths: {lengths}")
    for index in range(args.shards):
        start = index * args.shard_size
        stop = start + args.shard_size
        for name, rows in tables.items():
            write_jsonl(args.output / f"{name}_{index}.jsonl", rows[start:stop])


def merge_scores(args: argparse.Namespace) -> None:
    rows = []
    for index in range(args.shards):
        rows.extend(read_jsonl(args.score_root / f"shard_{index}.jsonl"))
    rows.sort(key=lambda row: int(row["index"]))
    expected = args.shards * args.shard_size
    if len(rows) != expected or [int(row["index"]) for row in rows] != list(range(expected)):
        raise ValueError("score shards do not cover aligned query indices 0..999")
    write_jsonl(args.output / "scores_1000.jsonl", rows)


def selected_manifests(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.output / "evaluation_1000.jsonl")
    expected = args.shards * args.shard_size
    if len(rows) != expected:
        raise ValueError(f"expected {expected} factual evaluation rows")
    selected = [
        {
            "id": str(row["id"]),
            "selected_ids": row.get("methods", {})
            .get("token_leverage_shap", {})
            .get("selected_ids", []),
        }
        for row in rows
    ]
    for index in range(args.shards):
        start = index * args.shard_size
        write_jsonl(
            args.output / f"selected_{index}.jsonl",
            selected[start : start + args.shard_size],
        )


def merge_pools(args: argparse.Namespace) -> None:
    if args.existing_pool is None:
        raise ValueError("merge-pools requires --existing-pool")
    by_id = {}
    for row in read_jsonl(args.existing_pool):
        if row.get("unit_id") and row.get("candidates"):
            by_id[str(row["unit_id"])] = row
    for index in range(args.shards):
        for row in read_jsonl(args.output / f"paraphrase_pool_{index}.jsonl"):
            if row.get("unit_id") and row.get("candidates"):
                by_id[str(row["unit_id"])] = row
    path = args.output / "paraphrase_pool_shapley.jsonl"
    write_jsonl(path, [by_id[unit_id] for unit_id in sorted(by_id)])
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (args.output / "paraphrase_pool_shapley.sha256").write_text(
        digest + "\n",
        encoding="utf-8",
    )
    print(digest, flush=True)


def merge_controls(args: argparse.Namespace) -> None:
    rows = []
    for index in range(args.shards):
        shard = read_jsonl(args.output / f"control_{index}.jsonl")
        shard.sort(key=lambda row: int(row["index"]))
        if len(shard) != args.shard_size:
            raise ValueError(f"control shard {index} has {len(shard)} rows")
        for offset, row in enumerate(shard):
            row["index"] = index * args.shard_size + offset
            rows.append(row)
    if len({str(row["id"]) for row in rows}) != len(rows):
        raise ValueError("duplicate query identifiers in merged controls")
    write_jsonl(args.output / "control_1000.jsonl", rows)


def merge(args: argparse.Namespace) -> None:
    rows = []
    for index in range(args.shards):
        rows.extend(read_jsonl(args.output / f"evaluation_{index}.jsonl"))
    rows.sort(key=lambda row: int(row["index"]))
    if len(rows) != args.shards * args.shard_size:
        raise ValueError(f"expected 1000 evaluation rows, found {len(rows)}")
    identifiers = [str(row["id"]) for row in rows]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("duplicate query identifiers in merged evaluation")
    write_jsonl(args.output / "evaluation_1000.jsonl", rows)

    executed = []
    token_counts = []
    zero_budget = 0
    for row in rows:
        budget = int(row.get("matched_token_budget", 0))
        method = row.get("methods", {}).get("token_leverage_shap")
        if budget == 0:
            zero_budget += 1
            token_counts.append(0)
            continue
        if not isinstance(method, dict):
            raise ValueError(f"missing evaluated method for positive budget: {row['id']}")
        count = int(method.get("n_modified_tokens", -1))
        if count != budget:
            raise ValueError(
                f"budget mismatch for {row['id']}: expected {budget}, observed {count}"
            )
        token_counts.append(count)
        if method.get("reader_called"):
            executed.append(method)
    summary = {
        "queries": len(rows),
        "zero_reflow_budget_queries": zero_budget,
        "positive_budget_queries": len(rows) - zero_budget,
        "executed_queries": len(executed),
        "verified_flips": sum(bool(row.get("verified_flip")) for row in executed),
        "answer_flip_rate": (
            sum(bool(row.get("verified_flip")) for row in executed)
            / max(1, len(executed))
        ),
        "mean_modified_tokens_over_1000": sum(token_counts) / len(rows),
        "budget_audit": "every positive-budget query exactly matches ReFlow",
    }
    (args.output / "evaluation_1000.summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=(
            "scores",
            "prepare",
            "merge",
            "selected",
            "merge-pools",
            "merge-controls",
        ),
    )
    parser.add_argument("--input", type=Path)
    parser.add_argument("--units", type=Path)
    parser.add_argument("--reflow", type=Path)
    parser.add_argument("--scores", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--score-root", type=Path)
    parser.add_argument("--existing-pool", type=Path)
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--shard-size", type=int, default=250)
    args = parser.parse_args()
    if args.mode == "scores":
        if args.score_root is None:
            parser.error("scores requires --score-root")
        merge_scores(args)
    elif args.mode == "prepare":
        if any(value is None for value in (args.input, args.units, args.reflow, args.scores)):
            parser.error("prepare requires --input, --units, --reflow, and --scores")
        prepare(args)
    elif args.mode == "selected":
        selected_manifests(args)
    elif args.mode == "merge-pools":
        merge_pools(args)
    elif args.mode == "merge-controls":
        merge_controls(args)
    else:
        merge(args)


if __name__ == "__main__":
    main()
