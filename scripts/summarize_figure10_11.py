#!/usr/bin/env python3
"""Audit and aggregate the frozen eight-dataset Figure 10/11 artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from causalityrag.evaluation_metrics import answer_changed, valid_answer


DATASETS = (
    "hotpotqa",
    "timeqa",
    "finqa",
    "musique",
    "quartz",
    "triviaqa",
    "2wiki",
    "popqa",
)
DISPLAY_NAMES = {
    "hotpotqa": "HotpotQA",
    "timeqa": "TimeQA",
    "finqa": "FinQA",
    "musique": "MuSiQue",
    "quartz": "QuARTz",
    "triviaqa": "TriviaQA",
    "2wiki": "2Wiki",
    "popqa": "PopQA",
}
MAIN_DATASETS = ("hotpotqa", "2wiki", "musique", "timeqa")
APPENDIX_DATASETS = ("finqa", "quartz", "triviaqa", "popqa")
POOL_SHA = {
    "hotpotqa": "5f161ed2405becbda7cd39517d1d1291c562574a404c975e42f2d91ad8b31bef",
    "timeqa": "01f19f09fe2842006b036823ca820d649ce14fd23d41a74793f7f21f62fe8b44",
    "finqa": "e2276863d19c5c4f9148016eef4550128a4516baaa039e1e36e62db03ccdadb6",
    "musique": "335bc1027e62c72973c97f57f4b20fc769601b16698411b42be2dd3319eac31e",
    "quartz": "724115a67b2b8af90f74cfaeea47270e0412f7250b75a66d421f232d6cd148e6",
    "triviaqa": "3703c4fa8e05a41b6d67f2dc247705d14b410867f56fe6ad8cf918c521d14d0f",
    "2wiki": "968488f69f032fbea66405b385fe8c227ce92c018632812d6d6486ddfe2be19d",
    "popqa": "42dd41818cdefac31ce788a1bf13402b07353ed032806c9a874f5fb8b19313e7",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def results_path(
    dataset: str,
    protocol: str,
    *,
    run_root: Path,
    sentence_root: Path,
    reuse_root: Path,
    popqa_root: Path,
) -> Path:
    if dataset == "popqa":
        filename = "exact_1000.jsonl" if protocol == "frontier" else f"{protocol}_1000.jsonl"
        return popqa_root / "protocol" / filename
    if protocol == "frontier":
        return sentence_root / dataset / "reflow_results_top5_1000.jsonl"
    if dataset == "hotpotqa":
        return reuse_root / dataset / "protocols" / f"{protocol}_results.jsonl"
    return run_root / "out" / dataset / "protocols" / f"{protocol}_results.jsonl"


def fidelity_path(
    dataset: str,
    *,
    run_root: Path,
    reuse_root: Path,
    popqa_root: Path,
) -> Path:
    if dataset == "popqa":
        return popqa_root / "fidelity/summary.json"
    if dataset == "2wiki":
        return reuse_root / dataset / "fidelity" / "summary.json"
    return run_root / "out" / dataset / "fidelity" / "summary.json"


def protocol_metrics(path: Path, *, dataset: str, protocol: str) -> dict:
    rows = read_jsonl(path)
    require(len(rows) == 1000, f"{dataset}/{protocol}: expected 1000 rows, got {len(rows)}")
    identifiers = [str(row.get("id", "")) for row in rows]
    require(all(identifiers), f"{dataset}/{protocol}: empty query id")
    require(
        len(set(identifiers)) == len(identifiers),
        f"{dataset}/{protocol}: duplicate query ids",
    )
    require(
        all(int(row.get("replacement_seed", -1)) == 0 for row in rows),
        f"{dataset}/{protocol}: replacement seed changed",
    )
    violations = sum(
        str(row.get("evaluation_status", "")).startswith("protocol_violation")
        for row in rows
    )
    valid_rows = []
    invalid_clean = 0
    invalid_edited = 0
    for row in rows:
        status = str(row.get("evaluation_status", ""))
        if not valid_answer(row.get("clean_answer", "")) or status in {
            "invalid_clean_answer",
            "protocol_violation_invalid_clean_or_gold_answer",
        }:
            invalid_clean += 1
            continue
        called = int(row.get("reader_calls", 0)) > 0
        if status.startswith("protocol_violation") or (
            called and not valid_answer(row.get("edited_answer", ""))
        ):
            invalid_edited += 1
            continue
        valid_rows.append(row)
    require(valid_rows, f"{dataset}/{protocol}: empty valid answer population")
    flips = sum(
        int(row.get("reader_calls", 0)) > 0
        and answer_changed(
            row.get("clean_answer", ""), row.get("edited_answer", "")
        )
        for row in valid_rows
    )
    return {
        "queries": len(rows),
        "valid_answer_queries": len(valid_rows),
        "invalid_clean_queries": invalid_clean,
        "invalid_edited_queries": invalid_edited,
        "answer_flip_rate": flips / len(valid_rows),
        "mean_modified_tokens": statistics.fmean(
            int(row.get("n_modified_tokens", 0)) for row in valid_rows
        ),
        "mean_reader_calls": statistics.fmean(
            int(row.get("reader_calls", 0)) for row in valid_rows
        ),
        "verified_flips": flips,
        "protocol_violations": violations,
        "sha256": file_sha256(path),
        "path": str(path.resolve()),
        "ids": set(identifiers),
    }


def audit_fidelity(path: Path, seed_audit_path: Path, *, dataset: str) -> dict:
    summary = read_json(path)
    seed_audit = read_json(seed_audit_path)
    require(seed_audit.get("seed") == 0, f"{dataset}: fidelity seed is not zero")
    require(seed_audit.get("matched") is True, f"{dataset}: fidelity seed audit failed")
    require(
        seed_audit.get("expected_probes") == 3000,
        f"{dataset}: seed audit probe count changed",
    )
    require(
        summary.get("schema") == "causalityrag.residual_flow_fidelity.v1",
        f"{dataset}: unexpected fidelity schema",
    )
    require(summary.get("queries") == 100, f"{dataset}: fidelity queries != 100")
    require(summary.get("budgets") == [1, 3, 5], f"{dataset}: fidelity budgets changed")
    require(
        summary.get("trials_per_query_budget") == 10,
        f"{dataset}: fidelity trials changed",
    )
    require(
        summary.get("reader_executions") == 3000,
        f"{dataset}: fidelity reader executions != 3000",
    )
    deciles = summary.get("deciles_low_to_high_residual", [])
    require(len(deciles) == 10, f"{dataset}: expected ten residual ranks")
    require(
        all(int(row.get("count", -1)) == 300 for row in deciles),
        f"{dataset}: each residual rank must contain 300 probes",
    )
    require(
        summary.get("source_sha256", {}).get("shared_pool") == POOL_SHA[dataset],
        f"{dataset}: fidelity pool SHA mismatch",
    )
    return {
        "ranks": [int(row["decile"]) for row in deciles],
        "answer_flip_rate": [float(row["answer_flip_rate"]) for row in deciles],
        "mean_residual_flow_ratio": [
            float(row["mean_residual_flow_ratio"]) for row in deciles
        ],
        "rank_counts": [int(row["count"]) for row in deciles],
        "auroc_removed_flow_predicting_answer_flip": summary.get(
            "auroc_removed_flow_predicting_answer_flip"
        ),
        "equal_budget_pairwise_concordance": summary.get(
            "equal_budget_pairwise_concordance"
        ),
        "sha256": file_sha256(path),
        "path": str(path.resolve()),
        "seed_audit_sha256": file_sha256(seed_audit_path),
        "seed_audit_path": str(seed_audit_path.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--sentence-root", type=Path)
    parser.add_argument("--reuse-root", type=Path)
    parser.add_argument(
        "--base-aggregate",
        type=Path,
        help=(
            "inherit the seven non-PopQA datasets from a frozen aggregate; "
            "their raw artifacts and SHAs remain unchanged"
        ),
    )
    parser.add_argument("--popqa-root", type=Path, required=True)
    parser.add_argument("--popqa-pool", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path, required=True)
    args = parser.parse_args()

    inherited_datasets: list[str] = []
    if args.base_aggregate:
        base = read_json(args.base_aggregate)
        require(
            base.get("schema") in {
                "causalityrag.figure10_11.v2",
                "causalityrag.figure10_11.v3",
            },
            "unexpected base Figure 10/11 schema",
        )
        inherited_datasets = [dataset for dataset in DATASETS if dataset != "popqa"]
        require(
            all(dataset in base.get("fidelity", {}) for dataset in inherited_datasets),
            "base aggregate is missing inherited fidelity datasets",
        )
        require(
            all(
                dataset in base.get("selection_protocols", {})
                for dataset in inherited_datasets
            ),
            "base aggregate is missing inherited protocol datasets",
        )
        fidelity = {
            dataset: base["fidelity"][dataset]
            for dataset in inherited_datasets
        }
        protocols = {
            dataset: base["selection_protocols"][dataset]
            for dataset in inherited_datasets
        }
        datasets_to_compute = ("popqa",)
        settings = dict(base.get("settings", {}))
    else:
        require(args.run_root is not None, "--run-root is required without --base-aggregate")
        require(
            args.sentence_root is not None,
            "--sentence-root is required without --base-aggregate",
        )
        fidelity = {}
        protocols = {}
        datasets_to_compute = DATASETS
        settings = {
            "reader": "Qwen2.5-7B-Instruct",
            "temperature": 0,
            "retrieval_depth": 5,
            "replacement_seed": 0,
            "fidelity_queries_per_dataset": 100,
            "fidelity_budgets": [1, 3, 5],
            "fidelity_trials_per_query_budget": 10,
            "protocol_queries_per_dataset": 1000,
            "one_shot_price": 1.0,
            "grid_price_exponents": [-8.0, 8.0, 0.5],
        }
    reuse_root = (
        args.reuse_root
        or (args.run_root / "reuse" if args.run_root is not None else Path("."))
    )

    for dataset in datasets_to_compute:
        pool_path = (
            args.popqa_pool
            if dataset == "popqa"
            else args.run_root / "inputs" / dataset / "shared_pool.jsonl"
        )
        require(pool_path.is_file(), f"{dataset}: frozen pool missing")
        require(
            file_sha256(pool_path) == POOL_SHA[dataset],
            f"{dataset}: frozen pool SHA mismatch",
        )
        fidelity[dataset] = audit_fidelity(
            fidelity_path(
                dataset,
                run_root=args.run_root,
                reuse_root=reuse_root,
                popqa_root=args.popqa_root,
            ),
            (
                args.popqa_root / "fidelity/seed_audit.json"
                if dataset == "popqa"
                else args.run_root / "out" / dataset / "fidelity/seed_audit.json"
            ),
            dataset=dataset,
        )
        metrics = {}
        for protocol in ("one_shot", "fixed_grid", "frontier"):
            metrics[protocol] = protocol_metrics(
                results_path(
                    dataset,
                    protocol,
                    run_root=args.run_root,
                    sentence_root=args.sentence_root,
                    reuse_root=reuse_root,
                    popqa_root=args.popqa_root,
                ),
                dataset=dataset,
                protocol=protocol,
            )
        reference_ids = metrics["frontier"]["ids"]
        for protocol, values in metrics.items():
            require(
                values.pop("ids") == reference_ids,
                f"{dataset}: {protocol} does not use the same query ids as Frontier",
            )
        protocols[dataset] = metrics

    result = {
        "schema": "causalityrag.figure10_11.v3",
        "dataset_order": list(DATASETS),
        "display_names": DISPLAY_NAMES,
        "figure11_groups": {
            "main": list(MAIN_DATASETS),
            "appendix": list(APPENDIX_DATASETS),
        },
        "settings": settings,
        "provenance": {
            "inherited_datasets": inherited_datasets,
            "base_aggregate": (
                {
                    "path": str(args.base_aggregate.resolve()),
                    "sha256": file_sha256(args.base_aggregate),
                }
                if args.base_aggregate
                else None
            ),
            "recomputed_datasets": list(datasets_to_compute),
        },
        "fidelity": fidelity,
        "selection_protocols": protocols,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "schema": "causalityrag.figure10_11.manifest.v1",
        "aggregate": str(args.out.resolve()),
        "aggregate_sha256": file_sha256(args.out),
        "pool_sha256": POOL_SHA,
        "inherited_datasets": inherited_datasets,
        "base_aggregate": (
            {
                "path": str(args.base_aggregate.resolve()),
                "sha256": file_sha256(args.base_aggregate),
            }
            if args.base_aggregate
            else None
        ),
        "all_audits_passed": True,
        "paper_render_command": (
            "python3 scripts/render_reflow_analysis.py "
            "--analysis figures/data/reflow_analysis_v3.json "
            "--figure10-11 figures/data/figure10_11_v3.json "
            "--out sections/6_Experiments/reflow_analysis_values.tex"
        ),
    }
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
