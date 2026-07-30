"""Build the method-independent token inventory for a shared replacement pool."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.io import load_records, record_id, retrieved_contexts
from causalityrag.counterfactual_pool_generation import target_sentence
from causalityrag.shared_replacement_pool import (
    POOL_SCHEMA,
    file_sha256,
    is_editable_unit,
    typed_pool_key,
)
from causalityrag.token_units import units_from_cache_row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--units-cache", required=True)
    parser.add_argument("--positions-out", required=True)
    parser.add_argument("--typed-keys-out", required=True)
    parser.add_argument("--manifest-out", required=True)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--examples-per-key", type=int, default=3)
    args = parser.parse_args()

    records = load_records(args.input)[: args.n]
    unit_rows = load_records(args.units_cache)[: args.n]
    if len(records) != len(unit_rows):
        raise ValueError("input and unit cache must be aligned")

    positions: dict[str, dict] = {}
    typed_examples: dict[str, list[dict]] = defaultdict(list)
    typed_counts: dict[str, int] = defaultdict(int)
    occurrence_count = 0

    for record, unit_row in zip(records, unit_rows):
        query_id = record_id(record)
        if str(unit_row.get("id", "")) != query_id:
            raise ValueError(f"misaligned unit cache for {query_id}")
        contexts = {
            str(context["chunk_id"]): context
            for context in retrieved_contexts(record)[: args.k]
        }
        for unit in units_from_cache_row(record, unit_row, k=args.k):
            if not is_editable_unit(unit):
                continue
            occurrence_count += 1
            unit_id = str(unit["unit_id"])
            key = typed_pool_key(unit)
            context = str(contexts[str(unit["chunk_id"])]["text"])
            position_row = {
                "schema": POOL_SCHEMA,
                "unit_id": unit_id,
                "typed_key": key,
                "surface": str(unit.get("text", "")),
                "type": str(unit.get("type", "")),
                "pos": str(unit.get("pos", "")),
                "tag": str(unit.get("tag", "")),
                "morph": unit.get("morph", ""),
                "entity_text": str(unit.get("entity_text", "")),
                "entity_token_index": unit.get("entity_token_index"),
                "entity_token_count": unit.get("entity_token_count"),
                "chunk_id": str(unit.get("chunk_id", "")),
                "chunk_char_start": int(unit.get("chunk_char_start", -1)),
                "chunk_char_end": int(unit.get("chunk_char_end", -1)),
            }
            previous = positions.get(unit_id)
            if previous is not None and previous != position_row:
                raise ValueError(f"conflicting metadata for {unit_id}")
            positions[unit_id] = position_row
            typed_counts[key] += 1
            examples = typed_examples[key]
            if len(examples) < args.examples_per_key and all(
                example["unit_id"] != unit_id for example in examples
            ):
                sentence = target_sentence(unit, context)
                examples.append({
                    "unit_id": unit_id,
                    "query_id": query_id,
                    **sentence,
                    "entity_text": str(unit.get("entity_text", "")),
                    "entity_token_index": unit.get("entity_token_index"),
                    "entity_token_count": unit.get("entity_token_count"),
                })

    os.makedirs(os.path.dirname(os.path.abspath(args.positions_out)), exist_ok=True)
    with open(args.positions_out, "w", encoding="utf-8") as output:
        for unit_id in sorted(positions):
            output.write(json.dumps(positions[unit_id], ensure_ascii=False) + "\n")

    representatives = {}
    for row in positions.values():
        representatives.setdefault(row["typed_key"], row)
    with open(args.typed_keys_out, "w", encoding="utf-8") as output:
        for key in sorted(representatives):
            representative = representatives[key]
            output.write(json.dumps({
                "schema": POOL_SCHEMA,
                "typed_key": key,
                "surface": representative["surface"],
                "type": representative["type"],
                "pos": representative["pos"],
                "tag": representative["tag"],
                "morph": representative["morph"],
                "entity_token_index": representative["entity_token_index"],
                "entity_token_count": representative["entity_token_count"],
                "occurrences": typed_counts[key],
                "examples": typed_examples[key],
            }, ensure_ascii=False) + "\n")

    manifest = {
        "schema": POOL_SCHEMA,
        "queries": len(records),
        "top_k": args.k,
        "editable_occurrences": occurrence_count,
        "unique_positions": len(positions),
        "typed_keys": len(representatives),
        "positions_file": os.path.abspath(args.positions_out),
        "positions_sha256": file_sha256(args.positions_out),
        "typed_keys_file": os.path.abspath(args.typed_keys_out),
        "typed_keys_sha256": file_sha256(args.typed_keys_out),
    }
    with open(args.manifest_out, "w", encoding="utf-8") as output:
        json.dump(manifest, output, indent=2, ensure_ascii=False)
        output.write("\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
