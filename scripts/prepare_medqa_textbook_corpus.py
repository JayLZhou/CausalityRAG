#!/usr/bin/env python3
"""Normalize the frozen MedRAG Textbooks snapshot for MedQA retrieval."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


EXPECTED_BOOKS = 18


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_row(row: dict, *, source_file: str, source_line: int) -> dict:
    text = str(row.get("content") or row.get("text") or "").strip()
    title = str(row.get("title") or Path(source_file).stem).strip()
    if not text:
        raise ValueError(f"empty textbook content at {source_file}:{source_line}")
    identifier = str(row.get("id") or "").strip()
    if not identifier:
        material = f"{source_file}\0{source_line}\0{title}\0{text}".encode("utf-8")
        identifier = "medrag-" + hashlib.sha256(material).hexdigest()[:20]
    return {
        "id": identifier,
        "title": title,
        "text": text,
        "source_file": source_file,
        "source_line": source_line,
    }


def convert(
    chunk_dir: Path,
    output_path: Path,
    manifest_path: Path,
    *,
    source_revision: str,
) -> dict:
    source_files = sorted(chunk_dir.glob("*.jsonl"))
    if len(source_files) != EXPECTED_BOOKS:
        raise ValueError(
            f"expected {EXPECTED_BOOKS} MedRAG textbook files, found "
            f"{len(source_files)} in {chunk_dir}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    seen_ids: set[str] = set()
    rows = 0
    source_manifest = []
    with temporary.open("w", encoding="utf-8") as output:
        for source_path in source_files:
            source_rows = 0
            with source_path.open(encoding="utf-8") as source:
                for line_number, line in enumerate(source, 1):
                    if not line.strip():
                        continue
                    row = normalize_row(
                        json.loads(line),
                        source_file=source_path.name,
                        source_line=line_number,
                    )
                    if row["id"] in seen_ids:
                        raise ValueError(f"duplicate MedRAG snippet id: {row['id']}")
                    seen_ids.add(row["id"])
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
                    rows += 1
                    source_rows += 1
            source_manifest.append({
                "file": source_path.name,
                "rows": source_rows,
                "bytes": source_path.stat().st_size,
                "sha256": file_sha256(source_path),
            })
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, output_path)

    if rows < 100_000:
        raise ValueError(f"unexpectedly small MedRAG Textbooks corpus: {rows} rows")
    manifest = {
        "schema": "causalityrag.medqa_textbooks.v1",
        "source": "https://huggingface.co/datasets/MedRAG/textbooks",
        "source_revision": source_revision,
        "books": len(source_files),
        "snippets": rows,
        "files": source_manifest,
        "output": str(output_path.resolve()),
        "output_sha256": file_sha256(output_path),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_manifest, manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--manifest-out", required=True, type=Path)
    parser.add_argument("--source-revision", required=True)
    args = parser.parse_args()
    manifest = convert(
        args.chunk_dir,
        args.out,
        args.manifest_out,
        source_revision=args.source_revision,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
