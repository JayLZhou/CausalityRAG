"""Merge validated typed replacement rows into one deterministic seed."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.shared_replacement_pool import file_sha256


def merge_seed_rows(paths: list[Path]) -> list[dict]:
    rows_by_key = {}
    for path in paths:
        with path.open(encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = str(row.get("typed_key", ""))
                if not key or not row.get("candidates"):
                    continue
                existing = rows_by_key.get(key)
                if existing is None or len(row["candidates"]) > len(
                    existing["candidates"]
                ):
                    rows_by_key[key] = row
    return [rows_by_key[key] for key in sorted(rows_by_key)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", action="append", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--manifest-out", required=True, type=Path)
    args = parser.parse_args()

    rows = merge_seed_rows(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, args.out)
    manifest = {
        "schema": "causalityrag.shared_replacement_seed.v1",
        "sources": [str(path.resolve()) for path in args.seed],
        "typed_keys": len(rows),
        "sha256": file_sha256(args.out),
    }
    args.manifest_out.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
