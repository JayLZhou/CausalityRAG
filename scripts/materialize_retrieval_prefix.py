#!/usr/bin/env python3
"""Materialize one nested retrieval prefix and its executable token pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.io import load_records, record_id, retrieved_contexts
from causalityrag.shared_replacement_pool import file_sha256
from causalityrag.token_units import units_from_cache_row


def atomic_write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(value, output, indent=2, ensure_ascii=False)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def replace_retrieved_prefix(record: dict, k: int) -> dict:
    contexts = retrieved_contexts(record)
    if len(contexts) < k:
        raise ValueError(
            f"{record_id(record)}: expected at least {k} retrieved chunks, "
            f"got {len(contexts)}"
        )
    clipped = dict(record)
    if isinstance(record.get("retrieved"), list):
        clipped["retrieved"] = list(record["retrieved"][:k])
    elif isinstance(record.get("contexts"), list):
        clipped["contexts"] = list(record["contexts"][:k])
    else:
        raise ValueError(f"{record_id(record)}: unsupported retrieval schema")
    return clipped


def materialize(
    input_path: Path,
    units_path: Path,
    shared_pool_path: Path,
    out_dir: Path,
    *,
    n: int,
    k: int,
    expected_source_pool_sha256: str = "",
    reuse_existing: bool = False,
) -> dict:
    if n <= 0 or k <= 0:
        raise ValueError("n and k must be positive")
    manifest_path = out_dir / "manifest.json"
    if reuse_existing and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != "causalityrag.retrieval_prefix.v1":
            raise ValueError(f"unsupported prefix manifest: {manifest_path}")
        if int(manifest.get("queries", 0)) != n:
            raise ValueError("existing prefix manifest has a different query count")
        if int(manifest.get("retrieval_depth", 0)) != k:
            raise ValueError("existing prefix manifest has a different retrieval depth")
        expected_sources = {
            "retrieval": input_path,
            "units": units_path,
            "shared_pool": shared_pool_path,
        }
        for name, path in expected_sources.items():
            if manifest["sources"].get(name) != str(path.resolve()):
                raise ValueError(f"existing prefix uses a different {name} source")
            if manifest["sources"].get(f"{name}_sha256") != file_sha256(str(path)):
                raise ValueError(f"existing prefix {name} source SHA changed")
        for name in ("retrieval", "units", "shared_pool"):
            path = Path(manifest["outputs"][name])
            if manifest["outputs"].get(f"{name}_sha256") != file_sha256(str(path)):
                raise ValueError(f"existing prefix {name} output SHA changed")
        return manifest
    actual_source_pool_sha256 = file_sha256(str(shared_pool_path))
    if (
        expected_source_pool_sha256
        and actual_source_pool_sha256 != expected_source_pool_sha256
    ):
        raise ValueError(
            "source replacement pool fingerprint mismatch: expected "
            f"{expected_source_pool_sha256}, got {actual_source_pool_sha256}"
        )

    records = load_records(input_path)[:n]
    unit_rows = load_records(units_path)[:n]
    if len(records) != n or len(unit_rows) != n:
        raise ValueError(
            f"expected {n} input/unit rows, got {len(records)}/{len(unit_rows)}"
        )
    record_ids = [record_id(row) for row in records]
    unit_ids = [record_id(row) for row in unit_rows]
    if any(not identifier for identifier in record_ids):
        raise ValueError("retrieval rows must have nonempty IDs")
    if len(set(record_ids)) != n:
        raise ValueError("retrieval row IDs must be unique")
    if unit_ids != record_ids:
        raise ValueError("token-unit rows are not aligned with retrieval rows")

    prefix_records = []
    prefix_unit_rows = []
    executable_unit_ids: set[str] = set()
    prefix_chunk_ids: set[str] = set()
    for record, cached_row in zip(records, unit_rows):
        prefix_record = replace_retrieved_prefix(record, k)
        selected_contexts = retrieved_contexts(prefix_record)
        selected_chunk_ids = {
            str(context["chunk_id"]) for context in selected_contexts
        }
        selected_units = units_from_cache_row(record, cached_row, k=k)
        selected_sentences = [
            dict(sentence)
            for sentence in cached_row.get("sentences", [])
            if str(sentence.get("chunk_id", "")) in selected_chunk_ids
        ]
        selected_hashes = {
            chunk_id: cached_row.get("context_sha256", {}).get(chunk_id)
            for chunk_id in selected_chunk_ids
        }
        if any(value is None for value in selected_hashes.values()):
            raise ValueError(
                f"{record_id(record)}: missing cached context hash in prefix"
            )
        prefix_records.append(prefix_record)
        prefix_unit_row = {
            **cached_row,
            "top_k": k,
            "context_sha256": selected_hashes,
            "units": selected_units,
        }
        if "sentences" in cached_row:
            prefix_unit_row["sentences"] = selected_sentences
        prefix_unit_rows.append(prefix_unit_row)
        executable_unit_ids.update(
            str(unit["unit_id"]) for unit in selected_units
        )
        prefix_chunk_ids.update(selected_chunk_ids)

    pool_rows = []
    source_pool_rows = 0
    with shared_pool_path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            source_pool_rows += 1
            row = json.loads(line)
            unit_id = str(row.get("unit_id", ""))
            if not unit_id or unit_id in executable_unit_ids:
                pool_rows.append(row)
    retained_pool_unit_ids = {
        str(row.get("unit_id", "")) for row in pool_rows if row.get("unit_id")
    }
    if not retained_pool_unit_ids:
        raise ValueError("materialized prefix has no replacement-pool positions")
    if not retained_pool_unit_ids.issubset(executable_unit_ids):
        raise RuntimeError("prefix pool contains a position outside the unit domain")

    retrieval_out = out_dir / "retrieval.jsonl"
    units_out = out_dir / "units.jsonl"
    pool_out = out_dir / "shared_pool.jsonl"
    atomic_write_jsonl(retrieval_out, prefix_records)
    atomic_write_jsonl(units_out, prefix_unit_rows)
    atomic_write_jsonl(pool_out, pool_rows)
    manifest = {
        "schema": "causalityrag.retrieval_prefix.v1",
        "queries": n,
        "retrieval_depth": k,
        "nested_from_top_k": int(unit_rows[0].get("top_k", k)),
        "query_id_sha256": hashlib.sha256(
            "\n".join(record_ids).encode("utf-8")
        ).hexdigest(),
        "prefix_chunks": len(prefix_chunk_ids),
        "executable_units": len(executable_unit_ids),
        "source_pool_rows": source_pool_rows,
        "prefix_pool_rows": len(pool_rows),
        "prefix_pool_positions": len(retained_pool_unit_ids),
        "sources": {
            "retrieval": str(input_path.resolve()),
            "retrieval_sha256": file_sha256(str(input_path)),
            "units": str(units_path.resolve()),
            "units_sha256": file_sha256(str(units_path)),
            "shared_pool": str(shared_pool_path.resolve()),
            "shared_pool_sha256": actual_source_pool_sha256,
        },
        "outputs": {
            "retrieval": str(retrieval_out.resolve()),
            "retrieval_sha256": file_sha256(str(retrieval_out)),
            "units": str(units_out.resolve()),
            "units_sha256": file_sha256(str(units_out)),
            "shared_pool": str(pool_out.resolve()),
            "shared_pool_sha256": file_sha256(str(pool_out)),
        },
    }
    atomic_write_json(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--units-cache", type=Path, required=True)
    parser.add_argument("--shared-pool", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--expected-source-pool-sha256", default="")
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()
    manifest = materialize(
        args.input,
        args.units_cache,
        args.shared_pool,
        args.out_dir,
        n=args.n,
        k=args.k,
        expected_source_pool_sha256=args.expected_source_pool_sha256,
        reuse_existing=args.reuse_existing,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
