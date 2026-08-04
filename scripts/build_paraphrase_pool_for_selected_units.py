#!/usr/bin/env python3
"""Build a frozen paraphrase pool for an explicit selected-unit manifest."""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.io import load_records, record_id
from causalityrag.paraphrase_control import (
    PARAPHRASE_POLICY,
    SentenceParaphraseClient,
    excluded_from_paraphrase_control,
)
from causalityrag.semantic_filter import ensure_wordnet_available
from causalityrag.shared_replacement_pool import file_sha256
from scripts.build_paraphrase_control_pool import generate_batch, target_from_unit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--units-cache", required=True)
    parser.add_argument("--selected-manifest", required=True)
    parser.add_argument("--existing-pool", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--manifest-out", required=True)
    parser.add_argument("--unresolved-out", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--attempts", type=int, default=6)
    parser.add_argument("--max-candidates", type=int, default=3)
    parser.add_argument("--llm-base-url", required=True)
    parser.add_argument("--llm-model", required=True)
    args = parser.parse_args()

    ensure_wordnet_available()
    records = {record_id(row): row for row in load_records(args.input)}
    unit_rows = {
        str(row.get("id", "")): row for row in load_records(args.units_cache)
    }
    selected_rows = load_records(args.selected_manifest)
    targets = {}
    for selected_row in selected_rows:
        query_id = str(selected_row.get("id", ""))
        if query_id not in records or query_id not in unit_rows:
            raise KeyError(f"selected query missing aligned inputs: {query_id}")
        unit_row = unit_rows[query_id]
        by_id = {
            str(unit["unit_id"]): unit for unit in unit_row.get("units", [])
        }
        sentences = {
            str(row["sentence_id"]): row for row in unit_row.get("sentences", [])
        }
        for unit_id in map(str, selected_row.get("selected_ids", [])):
            if unit_id not in by_id:
                raise KeyError(f"selected token missing from unit cache: {unit_id}")
            targets[unit_id] = target_from_unit(by_id[unit_id], sentences)

    excluded_targets = {
        unit_id: target
        for unit_id, target in targets.items()
        if excluded_from_paraphrase_control(target)
    }
    eligible_targets = {
        unit_id: target
        for unit_id, target in targets.items()
        if unit_id not in excluded_targets
    }

    existing_rows = {}
    if args.existing_pool and os.path.isfile(args.existing_pool):
        existing_rows = {
            str(row["unit_id"]): row
            for row in load_records(args.existing_pool)
            if row.get("unit_id") and row.get("candidates")
        }
    rows = [
        existing_rows[unit_id]
        for unit_id in sorted(eligible_targets)
        if unit_id in existing_rows
    ]
    missing = [
        eligible_targets[unit_id]
        for unit_id in sorted(eligible_targets)
        if unit_id not in existing_rows
    ]
    batches = [
        missing[start : start + args.batch_size]
        for start in range(0, len(missing), args.batch_size)
    ]
    client = SentenceParaphraseClient(
        base_url=args.llm_base_url,
        model=args.llm_model,
    )
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                generate_batch,
                batch,
                client=client,
                attempts=args.attempts,
                max_candidates=args.max_candidates,
            )
            for batch in batches
        ]
        for completed, future in enumerate(as_completed(futures), start=1):
            rows.extend(future.result())
            if completed % 25 == 0:
                print(
                    f"[paraphrase] {completed}/{len(futures)} batches calls={client.calls}",
                    flush=True,
                )
    rows.sort(key=lambda row: row["unit_id"])
    covered = [row for row in rows if row.get("candidates")]
    unresolved = [row for row in rows if not row.get("candidates")]
    if len(covered) + len(unresolved) != len(eligible_targets):
        raise ValueError("paraphrase output does not cover the selected token manifest")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        for row in covered:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(args.unresolved_out, "w", encoding="utf-8") as handle:
        for row in unresolved:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "schema": "causalityrag.selected_unit_paraphrase_pool.v1",
        "policy": PARAPHRASE_POLICY,
        "selected_positions": len(targets),
        "paraphrase_eligible_positions": len(eligible_targets),
        "excluded_numeric_date_positions": len(excluded_targets),
        "covered_positions": len(covered),
        "unresolved_positions": len(unresolved),
        "coverage": (
            len(covered) / len(eligible_targets)
            if eligible_targets
            else 1.0
        ),
        "reused_positions": len(rows) - len(missing),
        "generated_positions": len(missing),
        "llm_calls": client.calls,
        "source_sha256": {
            "input": file_sha256(args.input),
            "units": file_sha256(args.units_cache),
            "selected_manifest": file_sha256(args.selected_manifest),
            "existing_pool": file_sha256(args.existing_pool)
            if args.existing_pool and os.path.isfile(args.existing_pool)
            else None,
        },
    }
    with open(args.manifest_out, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print(json.dumps(manifest, indent=2))
    if unresolved:
        raise RuntimeError(f"{len(unresolved)} paraphrase positions remain unresolved")


if __name__ == "__main__":
    main()
