#!/usr/bin/env python3
"""Finish and freeze the small unresolved tail of the PubMedQA pool."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def run(command: list[str], *, cwd: Path) -> None:
    print("[pubmedqa-pool] RUN " + " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def unresolved_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open(encoding="utf-8") as source:
        return sum(bool(line.strip()) for line in source)


def reset_forbidden(path: Path) -> None:
    backup = path.with_suffix(".pre_reset.jsonl")
    if not backup.exists():
        shutil.copy2(path, backup)
    rows = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            row["forbidden"] = []
            rows.append(row)
    with path.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[pubmedqa-pool] reset accumulated rejected proposals in {path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", default="/data1/yujia/CausalityRAG/out")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--max-passes", type=int, default=20)
    parser.add_argument("--llm-base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--llm-model", default="qwen2.5-7b")
    parser.add_argument("--spacy-base-url", default="http://127.0.0.1:8021")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    base = Path(args.out_root) / "pubmedqa"
    pool_dir = base / "replacements/shared_pool_top10_v1"
    unresolved = pool_dir / "unresolved.jsonl"
    previous = unresolved_count(unresolved)
    stalled = 0
    reset_done = False

    for generation_pass in range(1, args.max_passes + 1):
        if previous == 0:
            break
        run([
            args.python,
            "scripts/generate_shared_typed_replacement_pool.py",
            "--typed-keys", str(pool_dir / "typed_keys.jsonl"),
            "--seed", str(pool_dir / "typed_candidates.jsonl"),
            "--out", str(pool_dir / "typed_candidates.jsonl"),
            "--unresolved-out", str(unresolved),
            "--manifest-out", str(pool_dir / "generation.json"),
            "--workers", "24",
            "--batch-size", "1",
            "--max-candidates", "20",
            "--generation-rounds", "4",
            "--attempt-offset", str(1000 + generation_pass * 4),
            "--llm-base-url", args.llm_base_url,
            "--llm-model", args.llm_model,
            "--spacy-base-url", args.spacy_base_url,
        ], cwd=repo)
        current = unresolved_count(unresolved)
        print(
            f"[pubmedqa-pool] pass={generation_pass} unresolved={current}",
            flush=True,
        )
        stalled = stalled + 1 if current >= previous else 0
        previous = current
        if stalled >= 2 and not reset_done and current:
            reset_forbidden(unresolved)
            reset_done = True
            stalled = 0

    if previous:
        raise RuntimeError(
            f"PubMedQA pool remains unresolved for {previous} typed keys"
        )

    run([
        args.python,
        "scripts/freeze_shared_replacement_pool.py",
        "--positions", str(pool_dir / "positions.jsonl"),
        "--typed-candidates", str(pool_dir / "typed_candidates.jsonl"),
        "--out", str(pool_dir / "shared_pool.jsonl"),
        "--manifest-out", str(pool_dir / "shared_pool.manifest.json"),
    ], cwd=repo)
    run([
        args.python,
        "scripts/run_table3_frontiers.py",
        "--out-root", str(args.out_root),
        "--datasets", "pubmedqa",
    ], cwd=repo)
    print("[pubmedqa-pool] DONE", flush=True)


if __name__ == "__main__":
    main()
