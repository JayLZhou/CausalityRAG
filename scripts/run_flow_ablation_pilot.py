#!/usr/bin/env python3
"""Resume a frozen-cache, matched-budget flow ablation on one local GPU."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from causalityrag.evaluation_metrics import (
    answer_changed, clean_correct, correctness_lost, gold_answer_spec, valid_answer,
)
from causalityrag.flow_ablation import METHODS, rank_graph_tokens
from causalityrag.flow_contract import require_complete_graph_domain
from causalityrag.io import load_records, record_id, retrieved_contexts
from causalityrag.reader import reader_answers_exact_match
from causalityrag.shared_replacement_pool import FrozenSharedReplacementPool, file_sha256
from causalityrag.token_units import units_from_cache_row
from scripts.evaluate_matched_budget_baselines import LocalReaderAdapter, evaluate_query


def keyed(path: str) -> dict:
    rows = load_records(path)
    table = {record_id(row): row for row in rows}
    if "" in table or len(rows) != len(table):
        raise ValueError(f"missing/duplicate IDs: {path}")
    return table


def freeze_json(path: Path, value: dict) -> None:
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"resume manifest mismatch: {path}")
    else:
        with path.open("x") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)


def summarize(rows: list[dict], records: dict, dataset: str) -> dict:
    metrics = ("answer", "f1", "em", "acc")
    denominators = dict.fromkeys(metrics, 0)
    counts = {name: dict.fromkeys(metrics, 0) for name in METHODS}
    for row in rows:
        clean = row["clean_answer"]
        if not valid_answer(clean):
            continue
        spec = gold_answer_spec(records[row["id"]], dataset)
        for metric in metrics:
            eligible = metric == "answer" or clean_correct(metric, clean, spec, reader_mode="short_answer")
            if not eligible:
                continue
            denominators[metric] += 1
            for name in METHODS:
                result = row.get("methods", {}).get(name, {})
                edited = result.get("edited_answer", "")
                executed = result.get("reader_called") and valid_answer(edited)
                if not executed or str(result.get("status", "")).startswith("protocol_violation"):
                    continue
                counts[name][metric] += int(
                    answer_changed(clean, edited) if metric == "answer"
                    else correctness_lost(metric, clean, edited, spec, reader_mode="short_answer")
                )
    return {
        "queries": len(rows), "denominators": denominators,
        "numerators": counts,
        "rates": {
            name: {metric: counts[name][metric] / denominators[metric] if denominators[metric] else None
                   for metric in metrics} for name in METHODS
        },
        "reader_calls": sum(row.get("reader_calls", 0) for row in rows),
        "controls_status": "not_run_no_cfr_claim",
        "method_failures_retained_in_denominator": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20261023)
    args = parser.parse_args()
    if args.n <= 0:
        parser.error("--n must be positive")
    config = json.loads(Path(args.config).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(config, out, args.n, args.seed)


def run(config: dict, out: Path, n: int, seed: int) -> None:
    # This pilot uses the existing short-answer HF reader only, not label-mode QA.
    if config["dataset"] == "quartz":
        raise ValueError("QuaRTz requires a separately validated label-mode reader")
    paths = {name: config[name] for name in ("input", "units", "graphs", "reflow", "pool")}
    shas = {name: file_sha256(path) for name, path in paths.items()}
    for name, expected in config["expected_sha256"].items():
        if shas[name] != expected:
            raise ValueError(f"frozen {name} SHA mismatch")
    tables = {name: keyed(paths[name]) for name in ("input", "units", "graphs", "reflow")}
    records, units, graphs, reflow = (tables[name] for name in ("input", "units", "graphs", "reflow"))
    if any(set(table) != set(records) for table in tables.values()):
        raise ValueError("query ID sets differ")
    ids = [qid for qid in records if valid_answer(reflow[qid].get("clean_answer", ""))]
    ids.sort(key=lambda qid: (hashlib.sha256(f"{seed}:{qid}".encode()).hexdigest(), qid))
    if len(ids) < n:
        raise ValueError("not enough valid-clean queries for declared pilot")
    ids = ids[:n]
    code_files = [*Path("causalityrag").glob("*.py"), Path(__file__),
                  Path("scripts/evaluate_matched_budget_baselines.py"), Path("scripts/evaluate_reflow.py")]
    manifest = {
        "config": config, "artifact_sha256": shas, "ids": ids, "seed": seed,
        "code_sha256": {str(p): file_sha256(str(p)) for p in sorted(code_files)},
        "selection": "stable_hash_of_valid_clean_ids_no_outcome_filter",
        "graph_domain": "editable_projection",
        "visible_device": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    freeze_json(out / "manifest.json", manifest)
    pool = FrozenSharedReplacementPool(paths["pool"])
    result_path = out / "factual.jsonl"
    done = keyed(str(result_path)) if result_path.exists() else {}
    if not set(done) <= set(ids):
        raise ValueError("output contains unexpected query IDs")
    print(f"PREFLIGHT passed: {len(records)} aligned IDs, pilot={n}, complete={len(done)}", flush=True)
    if len(done) == n:
        freeze_json(out / "summary.json", summarize(list(done.values()), records, config["dataset"]))
        return
    reader = LocalReaderAdapter(config["model"], device="cuda", dtype="bfloat16", reader_mode="short_answer")
    checks = out / "reader_compatibility.json"
    if not checks.exists():
        rows = []
        for qid in ids[:3]:
            record = records[qid]
            current = reader.answer(record["question"], retrieved_contexts(record)[:config["k"]])
            old = reflow[qid]["clean_answer"]
            rows.append({"id": qid, "cached": old, "current": current,
                         "matches": reader_answers_exact_match(record["question"], old, current, reader_mode="short_answer")})
        freeze_json(checks, {"rows": rows, "reader_calls": len(rows)})
    if not all(row["matches"] for row in json.loads(checks.read_text())["rows"]):
        raise ValueError("reader differs from cached clean responses; stop before ablation verification")
    print("Reader compatibility: 3/3 matched cached clean answers", flush=True)
    for index, qid in enumerate(ids):
        if qid in done:
            continue
        started = time.perf_counter()
        record, unit_row, graph = records[qid], units[qid], graphs[qid]
        cache = out / f"ranking_{index:04d}.json"
        if cache.exists():
            rankings = json.loads(cache.read_text())
        else:
            all_units = units_from_cache_row(record, unit_row, k=config["k"])
            require_complete_graph_domain(graph, all_units)
            eligible = {str(u["unit_id"]) for u in all_units if pool.is_eligible(str(u["unit_id"]))}
            rankings = {method: rank_graph_tokens(graph, eligible, method=method, graph_domain="editable_projection")
                        for method in METHODS}
            freeze_json(cache, rankings)
        print(f"RANKED {index + 1}/{n} {qid}", flush=True)
        result = evaluate_query(record, unit_row, reflow[qid], rankings, pool, reader,
                                random_seeds=[], replacement_seed=config["replacement_seed"], k=config["k"])
        result["elapsed_seconds"] = time.perf_counter() - started
        result["rankings_file"] = cache.name
        with result_path.open("a") as stream:
            stream.write(json.dumps(result, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        done[qid] = result
        print(f"VERIFIED {len(done)}/{n} {qid} calls={result['reader_calls']}", flush=True)
    freeze_json(out / "summary.json", summarize(list(done.values()), records, config["dataset"]))
    print("PILOT FACTUAL COMPLETE; controls and full eight-dataset evaluation remain pending", flush=True)


if __name__ == "__main__":
    main()
