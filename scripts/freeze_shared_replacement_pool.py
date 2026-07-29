"""Expand typed candidates into one immutable position-level evaluation pool."""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.shared_replacement_pool import POOL_SCHEMA, file_sha256


def _load_rows(path: str, key: str) -> dict[str, dict]:
    rows = {}
    with open(path, encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            value = str(row.get(key, ""))
            if not value:
                raise ValueError(f"{path}:{line_number} misses {key}")
            previous = rows.get(value)
            if previous is not None and previous != row:
                raise ValueError(f"conflicting rows for {value}")
            rows[value] = row
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positions", required=True)
    parser.add_argument("--typed-candidates", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--manifest-out", required=True)
    parser.add_argument(
        "--existing-position-pool",
        default="",
        help="reuse already validated rows when extending the retrieval range",
    )
    parser.add_argument(
        "--exclude-unresolved",
        action="store_true",
        help=(
            "record uncovered typed keys as common exclusions; without this flag "
            "freezing fails closed"
        ),
    )
    args = parser.parse_args()

    positions = _load_rows(args.positions, "unit_id")
    typed = {
        key: row
        for key, row in _load_rows(
            args.typed_candidates,
            "typed_key",
        ).items()
        if row.get("candidates")
    }
    existing = (
        _load_rows(args.existing_position_pool, "unit_id")
        if args.existing_position_pool
        else {}
    )

    eligible = 0
    excluded = 0
    unresolved = []
    preserved = 0
    output_rows = []
    for unit_id in sorted(positions):
        position = positions[unit_id]
        previous = existing.get(unit_id)
        if previous is not None and (
            previous.get("candidates")
            or previous.get("row_kind") == "excluded_position"
        ):
            row = previous
            preserved += 1
        else:
            typed_row = typed.get(str(position["typed_key"]))
            if typed_row is None:
                unresolved.append(unit_id)
                if not args.exclude_unresolved:
                    continue
                row = {
                    **position,
                    "schema": POOL_SCHEMA,
                    "row_kind": "excluded_position",
                    "candidates": [],
                    "reason": "no_legal_counterfactual_under_contract",
                }
            else:
                row = {
                    **position,
                    "schema": POOL_SCHEMA,
                    "row_kind": "position_candidates",
                    "candidates": typed_row["candidates"],
                    "policy": typed_row.get(
                        "source",
                        "llm_typed_counterfactual_pool",
                    ),
                }
        if row.get("row_kind") == "excluded_position":
            excluded += 1
        else:
            eligible += 1
        output_rows.append(row)

    if unresolved and not args.exclude_unresolved:
        preview = ", ".join(unresolved[:5])
        raise ValueError(
            f"cannot freeze: {len(unresolved)} positions are unresolved ({preview})"
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    temporary = args.out + ".tmp"
    with open(temporary, "w", encoding="utf-8") as output:
        for row in output_rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, args.out)

    manifest = {
        "schema": POOL_SCHEMA,
        "positions": len(positions),
        "eligible_positions": eligible,
        "excluded_positions": excluded,
        "unresolved_typed_positions": len(unresolved),
        "coverage": len(output_rows) / max(1, len(positions)),
        "typed_keys_with_candidates": len(typed),
        "preserved_existing_positions": preserved,
        "positions_sha256": file_sha256(args.positions),
        "typed_candidates_sha256": file_sha256(args.typed_candidates),
        "existing_pool_sha256": (
            file_sha256(args.existing_position_pool)
            if args.existing_position_pool
            else ""
        ),
        "pool": os.path.abspath(args.out),
        "pool_sha256": file_sha256(args.out),
        "frozen": True,
    }
    with open(args.manifest_out, "w", encoding="utf-8") as output:
        json.dump(manifest, output, indent=2, ensure_ascii=False)
        output.write("\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
