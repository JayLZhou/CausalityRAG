"""Seed typed candidates from an already frozen position-level pool."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.shared_replacement_pool import POOL_SCHEMA, file_sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positions", required=True)
    parser.add_argument("--position-pool", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--manifest-out", required=True)
    parser.add_argument("--max-candidates", type=int, default=20)
    args = parser.parse_args()

    positions = {
        str(row["unit_id"]): row
        for row in (
            json.loads(line)
            for line in open(args.positions, encoding="utf-8")
            if line.strip()
        )
    }
    candidates: dict[str, dict[str, dict]] = defaultdict(dict)
    metadata = {}
    source_positions = 0
    with open(args.position_pool, encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            position = positions.get(str(row.get("unit_id", "")))
            if position is None or not row.get("candidates"):
                continue
            source_positions += 1
            key = str(position["typed_key"])
            metadata.setdefault(key, position)
            for candidate in row["candidates"]:
                value = str(candidate.get("new", "")).strip()
                if value:
                    candidates[key].setdefault(value.casefold(), candidate)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as output:
        for key in sorted(candidates):
            position = metadata[key]
            output.write(json.dumps({
                "schema": POOL_SCHEMA,
                "row_kind": "typed_candidates",
                "typed_key": key,
                "surface": position["surface"],
                "type": position.get("type", ""),
                "pos": position.get("pos", ""),
                "tag": position.get("tag", ""),
                "morph": position.get("morph", ""),
                "candidates": list(candidates[key].values())[
                    : args.max_candidates
                ],
                "source": "frozen_position_pool",
            }, ensure_ascii=False) + "\n")

    manifest = {
        "schema": POOL_SCHEMA,
        "source_positions": source_positions,
        "seeded_typed_keys": len(candidates),
        "output": os.path.abspath(args.out),
        "sha256": file_sha256(args.out),
    }
    with open(args.manifest_out, "w", encoding="utf-8") as output:
        json.dump(manifest, output, indent=2)
        output.write("\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
