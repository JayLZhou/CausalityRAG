#!/usr/bin/env python3
"""Run and merge the model-internal Table 3 attribution rankers."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.io import load_records, record_id


DATASETS = ("timeqa", "finqa", "musique", "quartz", "triviaqa", "2wiki", "pubmedqa")
METHODS = ("attention", "gradient_x_input", "integrated_gradients", "mirage", "arc_jsd")
FILENAMES = {
    "attention": "attention_top5_1000.jsonl",
    "gradient_x_input": "gradient_x_input_top5_1000.jsonl",
    "integrated_gradients": "integrated_gradients_top5_1000.jsonl",
    "mirage": "mirage_top5_1000.jsonl",
    "arc_jsd": "arc_jsd_top5_1000.jsonl",
}


def complete_ids(path: Path, expected_ids: list[str]) -> bool:
    if not path.is_file():
        return False
    try:
        ids = [record_id(row) for row in load_records(path)]
    except (OSError, ValueError):
        return False
    return ids == expected_ids


def clean_reference(base: Path) -> Path:
    candidates = (
        base / "inputs/clean_targets_top5_1000.jsonl",
        base / "inputs/clean_reference_top5_1000.jsonl",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"no clean top-5 reference under {base}")


def method_command(
    method: str,
    *,
    repo: Path,
    base: Path,
    reference: Path,
    output: Path,
    summary: Path,
    model_path: str,
    start: int,
    count: int,
) -> list[str]:
    common = [
        "--input", str(base / "retrieval/top10_1000.jsonl"),
        "--out", str(output),
        "--summary-out", str(summary),
        "--model-path", model_path,
        "--start", str(start),
        "--n", str(count),
        "--k", "5",
        "--device", "cuda",
        "--dtype", "bfloat16",
    ]
    if method == "attention":
        return [
            sys.executable,
            str(repo / "exp/run_attention.py"),
            *common,
            "--units-cache", str(base / "inputs/token_units_top10_1000.jsonl"),
        ]
    answer_conditioned = [
        *common,
        "--clean-reference", str(reference),
    ]
    if method in {"gradient_x_input", "integrated_gradients"}:
        return [
            sys.executable,
            str(repo / "exp/run_gradient_attribution_baselines.py"),
            *answer_conditioned,
            "--context-units", str(base / "inputs/token_units_top10_1000.jsonl"),
            "--method", method,
            "--top-k", "32",
            "--ig-steps", "16",
        ]
    if method == "mirage":
        return [
            sys.executable,
            str(repo / "exp/run_mirage.py"),
            *answer_conditioned,
            "--units-cache", str(base / "inputs/token_units_top10_1000.jsonl"),
        ]
    if method == "arc_jsd":
        return [
            sys.executable,
            str(repo / "exp/run_arc_jsd.py"),
            *answer_conditioned,
            "--batch-size", "4",
            "--feature-top-k", "64",
            "--routing-last-layers", "8",
            "--spacy-base-url", "http://127.0.0.1:8021",
        ]
    raise ValueError(f"unsupported method: {method}")


def merge_shards(
    shard_paths: list[Path],
    output: Path,
    summary: Path,
    expected_ids: list[str],
    method: str,
) -> None:
    by_id = {}
    for path in shard_paths:
        for row in load_records(path):
            identifier = record_id(row)
            if identifier in by_id:
                raise ValueError(f"duplicate ranker row: {identifier}")
            by_id[identifier] = row
    missing = [identifier for identifier in expected_ids if identifier not in by_id]
    extra = sorted(set(by_id).difference(expected_ids))
    if missing or extra:
        raise ValueError(f"ranker alignment failure: missing={len(missing)} extra={len(extra)}")
    ordered = [by_id[identifier] for identifier in expected_ids]
    with output.open("w", encoding="utf-8") as target:
        for row in ordered:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
    payload = {
        "method": method,
        "queries": len(ordered),
        "ranked_queries": sum(row.get("status") == "ok" for row in ordered),
        "total_seconds": sum(float(row.get("elapsed_seconds", 0.0)) for row in ordered),
        "shards": [str(path) for path in shard_paths],
        "out": str(output),
    }
    summary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", default="/data1/yujia/CausalityRAG/out")
    parser.add_argument("--model-path", default="/data1/yujia/models/Qwen2.5-7B-Instruct")
    parser.add_argument("--datasets", nargs="*", default=list(DATASETS))
    parser.add_argument("--methods", nargs="*", default=list(METHODS))
    parser.add_argument("--n", type=int, default=1000)
    args = parser.parse_args()

    if args.n != 1000:
        raise ValueError("the formal Table 3 runner requires n=1000")
    repo = Path(__file__).resolve().parents[1]
    for dataset in args.datasets:
        base = Path(args.out_root) / dataset
        records = load_records(base / "retrieval/top10_1000.jsonl")[: args.n]
        expected_ids = [record_id(row) for row in records]
        if len(expected_ids) != args.n or len(set(expected_ids)) != args.n:
            raise ValueError(f"invalid frozen retrieval IDs for {dataset}")
        reference = clean_reference(base)
        method_dir = base / "methods/baselines"
        shard_dir = method_dir / "shards"
        log_dir = base / "logs/table3_rankers"
        method_dir.mkdir(parents=True, exist_ok=True)
        shard_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        for method in args.methods:
            if method not in METHODS:
                raise ValueError(f"unsupported method: {method}")
            output = method_dir / FILENAMES[method]
            summary = output.with_suffix(".summary.json")
            if complete_ids(output, expected_ids) and summary.is_file():
                print(f"[table3-ranker] SKIP {dataset}/{method}: complete", flush=True)
                continue

            processes = []
            shard_paths = []
            log_handles = []
            for shard, (start, count) in enumerate(((0, 500), (500, 500))):
                shard_path = shard_dir / f"{method}_{start:04d}_{start + count - 1:04d}.jsonl"
                shard_summary = shard_path.with_suffix(".summary.json")
                shard_paths.append(shard_path)
                expected_shard = expected_ids[start : start + count]
                if complete_ids(shard_path, expected_shard) and shard_summary.is_file():
                    continue
                command = method_command(
                    method,
                    repo=repo,
                    base=base,
                    reference=reference,
                    output=shard_path,
                    summary=shard_summary,
                    model_path=args.model_path,
                    start=start,
                    count=count,
                )
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(shard)
                env["PYTHONUNBUFFERED"] = "1"
                env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
                log_handle = (log_dir / f"{method}_{start:04d}_{start + count - 1:04d}.log").open("a", encoding="utf-8")
                log_handles.append(log_handle)
                print(f"[table3-ranker] START {dataset}/{method} shard={shard}", flush=True)
                processes.append((subprocess.Popen(command, cwd=repo, env=env, stdout=log_handle, stderr=subprocess.STDOUT), shard))

            failures = []
            for process, shard in processes:
                code = process.wait()
                if code:
                    failures.append((shard, code))
            for handle in log_handles:
                handle.close()
            if failures:
                raise RuntimeError(f"ranker failures for {dataset}/{method}: {failures}")
            merge_shards(shard_paths, output, summary, expected_ids, method)
            print(f"[table3-ranker] DONE {dataset}/{method}", flush=True)


if __name__ == "__main__":
    main()
