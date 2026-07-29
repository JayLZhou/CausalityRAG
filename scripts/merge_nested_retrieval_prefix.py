"""Preserve an evaluated retrieval prefix while extending to a larger top-k."""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.io import load_records, record_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--extended", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--prefix-k", type=int, required=True)
    parser.add_argument("--target-k", type=int, required=True)
    parser.add_argument("--n", type=int, default=0)
    args = parser.parse_args()

    prefix_rows = load_records(args.prefix)
    extended_rows = load_records(args.extended)
    if args.n:
        prefix_rows = prefix_rows[: args.n]
        extended_rows = extended_rows[: args.n]
    if len(prefix_rows) != len(extended_rows):
        raise ValueError("retrieval files must contain aligned rows")

    with open(args.out, "w", encoding="utf-8") as output:
        for old, new in zip(prefix_rows, extended_rows):
            if record_id(old) != record_id(new):
                raise ValueError(
                    f"misaligned retrieval row: {record_id(old)} != "
                    f"{record_id(new)}"
                )
            merged = list(old.get("retrieved", []))[: args.prefix_k]
            seen = {str(chunk.get("chunk_id", "")) for chunk in merged}
            for chunk in new.get("retrieved", []):
                chunk_id = str(chunk.get("chunk_id", ""))
                if not chunk_id or chunk_id in seen:
                    continue
                merged.append(dict(chunk))
                seen.add(chunk_id)
                if len(merged) >= args.target_k:
                    break
            if len(merged) < args.target_k:
                raise ValueError(
                    f"{record_id(old)} has only {len(merged)} distinct chunks"
                )
            merged = [
                {**chunk, "rank": rank}
                for rank, chunk in enumerate(merged, start=1)
            ]
            row = dict(new)
            row["retrieved"] = merged
            output.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
