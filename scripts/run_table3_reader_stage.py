#!/usr/bin/env python3
"""Run the frozen-pool reader stages required to populate Table 3."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.io import load_records, record_id
from causalityrag.shared_replacement_pool import file_sha256


DATASETS = (
    "hotpotqa",
    "timeqa",
    "finqa",
    "musique",
    "quartz",
    "qasper",
    "2wiki",
    "pubmedqa",
)
SCORE_FILES = {
    "attention": "attention_top5_1000.jsonl",
    "gradient_x_input": "gradient_x_input_top5_1000.jsonl",
    "integrated_gradients": "integrated_gradients_top5_1000.jsonl",
    "mirage": "mirage_top5_1000.jsonl",
    "arc_jsd": "arc_jsd_top5_1000.jsonl",
}


def run(command: list[str], *, cwd: Path) -> None:
    print("[table3-reader] RUN " + " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def complete_jsonl(path: Path, expected_ids: list[str]) -> bool:
    if not path.is_file():
        return False
    try:
        return [record_id(row) for row in load_records(path)] == expected_ids
    except (OSError, ValueError):
        return False


def require_complete_jsonl(path: Path, expected_ids: list[str]) -> None:
    if not complete_jsonl(path, expected_ids):
        raise RuntimeError(f"missing or misaligned prerequisite: {path}")


def require_frozen_pool(pool_dir: Path) -> tuple[Path, str]:
    pool = pool_dir / "shared_pool.jsonl"
    manifest_path = pool_dir / "shared_pool.manifest.json"
    if not manifest_path.is_file() or not pool.is_file():
        raise RuntimeError(f"replacement pool is not frozen: {pool_dir}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("frozen") or float(manifest.get("coverage", 0.0)) != 1.0:
        raise RuntimeError(f"replacement pool failed closed audit: {manifest}")
    expected_sha = str(manifest.get("pool_sha256", ""))
    actual_sha = file_sha256(pool)
    if not expected_sha or actual_sha != expected_sha:
        raise RuntimeError(
            f"replacement pool fingerprint mismatch: {expected_sha} != {actual_sha}"
        )
    return pool, actual_sha


def build_paraphrase_pool(
    *,
    repo: Path,
    python: str,
    base: Path,
    retrieval: Path,
    units: Path,
    reflow_results: Path,
    baseline_results: Path,
    max_passes: int,
    llm_base_url: str,
    llm_model: str,
) -> tuple[Path, str]:
    controls = base / "controls"
    controls.mkdir(parents=True, exist_ok=True)
    pool = controls / "paraphrase_pool_top5_v2.jsonl"
    manifest_path = controls / "paraphrase_pool_top5_v2.manifest.json"
    unresolved = controls / "paraphrase_pool_top5_v2.unresolved.jsonl"

    for generation_pass in range(1, max_passes + 1):
        if manifest_path.is_file() and pool.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                int(manifest.get("unresolved_positions", -1)) == 0
                and float(manifest.get("coverage", 0.0)) == 1.0
                and str(manifest.get("pool_sha256", "")) == file_sha256(pool)
            ):
                return pool, file_sha256(pool)

        command = [
            python,
            "scripts/build_paraphrase_control_pool.py",
            "--input", str(retrieval),
            "--units-cache", str(units),
            "--reflow-results", str(reflow_results),
            "--baseline-results", str(baseline_results),
            "--out", str(pool),
            "--manifest-out", str(manifest_path),
            "--unresolved-out", str(unresolved),
            "--n", "1000",
            "--workers", "32",
            "--batch-size", "16",
            "--attempts", "8",
            "--max-candidates", "3",
            "--llm-base-url", llm_base_url,
            "--llm-model", llm_model,
        ]
        if pool.is_file():
            command.extend(["--existing-pool", str(pool)])
        run(command, cwd=repo)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        print(
            f"[table3-reader] paraphrase pass={generation_pass} "
            f"coverage={manifest.get('coverage')} "
            f"unresolved={manifest.get('unresolved_positions')}",
            flush=True,
        )

    raise RuntimeError(f"paraphrase pool remains unresolved for {base.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", default="/data1/yujia/CausalityRAG/out")
    parser.add_argument("--datasets", nargs="*", default=list(DATASETS))
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--workers", type=int, default=48)
    parser.add_argument("--max-paraphrase-passes", type=int, default=20)
    parser.add_argument("--llm-base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--llm-model", default="qwen2.5-7b")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    root = Path(args.out_root)
    for dataset in args.datasets:
        base = root / dataset
        retrieval = base / "retrieval/top10_1000.jsonl"
        units = base / "inputs/token_units_top10_1000.jsonl"
        expected_ids = [record_id(row) for row in load_records(retrieval)[:1000]]
        if len(expected_ids) != 1000 or len(set(expected_ids)) != 1000:
            raise RuntimeError(f"invalid frozen query manifest for {dataset}")
        require_complete_jsonl(units, expected_ids)

        frontier = base / "methods/reflow/frontier_top5_1000.jsonl"
        require_complete_jsonl(frontier, expected_ids)
        score_paths = {
            method: base / "methods/baselines" / filename
            for method, filename in SCORE_FILES.items()
        }
        for score_path in score_paths.values():
            require_complete_jsonl(score_path, expected_ids)
        pool, pool_sha = require_frozen_pool(
            base / "replacements/shared_pool_top10_v1"
        )

        audit = base / "audits/final_top10pool_k5"
        audit.mkdir(parents=True, exist_ok=True)
        reflow_results = base / "methods/reflow/results_top5_1000.jsonl"
        reflow_summary = base / "methods/reflow/results_top5_1000.summary.json"
        if not (complete_jsonl(reflow_results, expected_ids) and reflow_summary.is_file()):
            run([
                args.python,
                "scripts/evaluate_reflow.py",
                "--input", str(retrieval),
                "--units-cache", str(units),
                "--frontier", str(frontier),
                "--shared-pool", str(pool),
                "--expected-pool-sha256", pool_sha,
                "--out", str(reflow_results),
                "--summary-out", str(reflow_summary),
                "--n", "1000",
                "--k", "5",
                "--workers", str(args.workers),
                "--replacement-seed", "0",
                "--llm-base-url", args.llm_base_url,
                "--llm-model", args.llm_model,
            ], cwd=repo)
        require_complete_jsonl(reflow_results, expected_ids)

        baseline_results = audit / "baselines_1000.jsonl"
        baseline_summary = audit / "baselines_1000.summary.json"
        if not (complete_jsonl(baseline_results, expected_ids) and baseline_summary.is_file()):
            command = [
                args.python,
                "scripts/evaluate_matched_budget_baselines.py",
                "--input", str(retrieval),
                "--units-cache", str(units),
                "--reflow-results", str(reflow_results),
                "--shared-pool", str(pool),
                "--expected-pool-sha256", pool_sha,
                "--out", str(baseline_results),
                "--summary-out", str(baseline_summary),
                "--random-seeds", "0,1,2,3,4",
                "--replacement-seed", "0",
                "--n", "1000",
                "--k", "5",
                "--workers", str(args.workers),
                "--llm-base-url", args.llm_base_url,
                "--llm-model", args.llm_model,
            ]
            for method, path in score_paths.items():
                command.extend(["--scores", f"{method}={path}"])
            run(command, cwd=repo)
        require_complete_jsonl(baseline_results, expected_ids)

        factual_metrics = audit / "factual_metrics_1000.json"
        run([
            args.python,
            "scripts/summarize_clean_anchor_flips.py",
            "--reflow", str(reflow_results),
            "--baselines", str(baseline_results),
            "--out", str(factual_metrics),
        ], cwd=repo)

        paraphrase_pool, paraphrase_sha = build_paraphrase_pool(
            repo=repo,
            python=args.python,
            base=base,
            retrieval=retrieval,
            units=units,
            reflow_results=reflow_results,
            baseline_results=baseline_results,
            max_passes=args.max_paraphrase_passes,
            llm_base_url=args.llm_base_url,
            llm_model=args.llm_model,
        )
        paraphrase_results = base / "controls/paraphrase_results_top5_1000_v2.jsonl"
        paraphrase_summary = base / "controls/paraphrase_results_top5_1000_v2.summary.json"
        if not (
            complete_jsonl(paraphrase_results, expected_ids)
            and paraphrase_summary.is_file()
        ):
            run([
                args.python,
                "scripts/evaluate_paraphrase_controls.py",
                "--input", str(retrieval),
                "--units-cache", str(units),
                "--reflow-results", str(reflow_results),
                "--baseline-results", str(baseline_results),
                "--paraphrase-pool", str(paraphrase_pool),
                "--expected-pool-sha256", paraphrase_sha,
                "--out", str(paraphrase_results),
                "--summary-out", str(paraphrase_summary),
                "--n", "1000",
                "--k", "5",
                "--workers", str(args.workers),
                "--replacement-seed", "0",
                "--llm-base-url", args.llm_base_url,
                "--llm-model", args.llm_model,
            ], cwd=repo)
        require_complete_jsonl(paraphrase_results, expected_ids)
        run([
            args.python,
            "scripts/summarize_paraphrase_controls.py",
            "--input", str(paraphrase_results),
            "--out", str(paraphrase_summary),
        ], cwd=repo)

        adjusted = audit / "table3_metrics_1000_v2.json"
        run([
            args.python,
            "scripts/summarize_adjusted_flips.py",
            "--factual", str(factual_metrics),
            "--meaning-preserving", str(paraphrase_summary),
            "--out", str(adjusted),
        ], cwd=repo)
        print(f"[table3-reader] DONE {dataset}: {adjusted}", flush=True)

    run([
        args.python,
        "scripts/render_table3_rows.py",
        "--metrics-root", str(root),
        "--out", str(root / "table3_rows_v2.tex"),
    ], cwd=repo)
    print(f"[table3-reader] TABLE READY: {root / 'table3_rows_v2.tex'}", flush=True)


if __name__ == "__main__":
    main()
