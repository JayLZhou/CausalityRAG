#!/usr/bin/env python3
"""Build canonical ReFlow frontiers for the Table 3 datasets."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


DATASETS = ("timeqa", "finqa", "musique", "quartz", "qasper", "2wiki", "medqa")


def jsonl_is_complete(path: Path, expected: int = 1000) -> bool:
    if not path.is_file():
        return False
    identifiers = []
    try:
        with path.open(encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    identifiers.append(str(json.loads(line).get("id", "")))
    except (OSError, ValueError):
        return False
    return len(identifiers) == expected and len(set(identifiers)) == expected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", default="/data1/yujia/CausalityRAG/out")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--datasets", nargs="*", default=list(DATASETS))
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    out_root = Path(args.out_root)
    for dataset in args.datasets:
        base = out_root / dataset
        pool_dir = base / "replacements/shared_pool_top10_v1"
        manifest_path = pool_dir / "shared_pool.manifest.json"
        if not manifest_path.is_file():
            print(f"[table3-frontier] BLOCKED {dataset}: no frozen pool manifest", flush=True)
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not manifest.get("frozen") or float(manifest.get("coverage", 0.0)) != 1.0:
            print(f"[table3-frontier] BLOCKED {dataset}: pool is not fully frozen", flush=True)
            continue

        method_dir = base / "methods/reflow"
        method_dir.mkdir(parents=True, exist_ok=True)
        output = method_dir / "frontier_top5_1000.jsonl"
        summary = method_dir / "frontier_top5_1000.summary.json"
        if jsonl_is_complete(output, args.n) and summary.is_file():
            print(f"[table3-frontier] SKIP {dataset}: complete", flush=True)
            continue

        command = [
            sys.executable,
            str(repo / "exp/run_reflow.py"),
            "--input", str(base / "retrieval/top10_1000.jsonl"),
            "--graphs", str(base / "graphs/contribution_graph_top5_1000.jsonl"),
            "--units-cache", str(base / "inputs/token_units_top10_1000.jsonl"),
            "--shared-pool", str(pool_dir / "shared_pool.jsonl"),
            "--expected-pool-sha256", str(manifest["pool_sha256"]),
            "--out", str(output),
            "--summary-out", str(summary),
            "--n", str(args.n),
            "--k", str(args.k),
        ]
        print(f"[table3-frontier] START {dataset}", flush=True)
        subprocess.run(command, cwd=repo, check=True)
        if not jsonl_is_complete(output, args.n):
            raise RuntimeError(f"incomplete frontier output for {dataset}")
        print(f"[table3-frontier] DONE {dataset}", flush=True)


if __name__ == "__main__":
    main()
