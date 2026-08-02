"""Aggregate Acc-CFR by ReFlow's realized per-query intervention size."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from causalityrag.reader import answers_match


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
METHODS = ("mirage", "arc_jsd", "reflow")
STRATA: tuple[tuple[str, Callable[[int], bool]], ...] = (
    ("1", lambda budget: budget == 1),
    ("2-3", lambda budget: 2 <= budget <= 3),
    ("4-8", lambda budget: 4 <= budget <= 8),
    (">=9", lambda budget: budget >= 9),
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def method_result(
    baseline_row: dict[str, Any],
    reflow_by_id: dict[str, dict[str, Any]],
    method: str,
) -> tuple[bool, str]:
    if method == "reflow":
        result = reflow_by_id[str(baseline_row.get("id", ""))]
        return int(result.get("reader_calls", 0)) > 0, str(
            result.get("edited_answer", "")
        )
    result = baseline_row.get("methods", {}).get(method, {})
    return bool(result.get("reader_called", False)), str(
        result.get("edited_answer", "")
    )


def aggregate_dataset(base: Path) -> dict[str, Any]:
    metrics_path = base / "audits/final_top10pool_k5/factual_metrics_1000.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    reflow_path = Path(metrics["sources"]["reflow"])
    baselines_path = Path(metrics["sources"]["baselines"])
    reflow_rows = load_jsonl(reflow_path)
    baseline_rows = load_jsonl(baselines_path)
    reflow_by_id = {str(row.get("id", "")): row for row in reflow_rows}
    if len(reflow_by_id) != 1000 or len(baseline_rows) != 1000:
        raise ValueError(f"expected 1,000 aligned rows under {base}")

    zero_budget_queries = 0
    for row in baseline_rows:
        query_id = str(row.get("id", ""))
        reflow_row = reflow_by_id[query_id]
        matched_budget = int(row.get("matched_token_budget", 0))
        realized_budget = int(reflow_row.get("n_modified_tokens", 0))
        if matched_budget != realized_budget:
            raise ValueError(
                f"budget mismatch for {query_id}: {matched_budget} != {realized_budget}"
            )
        zero_budget_queries += int(matched_budget == 0)

    output: dict[str, Any] = {
        "queries": len(baseline_rows),
        "zero_budget_queries": zero_budget_queries,
        "strata": {},
    }
    for label, contains in STRATA:
        rows = [
            row
            for row in baseline_rows
            if contains(int(row.get("matched_token_budget", 0)))
        ]
        clean_rows = [
            row
            for row in rows
            if answers_match(
                str(row.get("clean_answer", "")), str(row.get("gold_answer", ""))
            )
        ]
        method_values: dict[str, Any] = {}
        for method in METHODS:
            flips = 0
            for row in clean_rows:
                reader_called, edited_answer = method_result(
                    row, reflow_by_id, method
                )
                gold_answer = str(row.get("gold_answer", ""))
                flips += int(
                    reader_called
                    and bool(edited_answer.strip())
                    and not answers_match(edited_answer, gold_answer)
                )
            method_values[method] = {
                "acc_cfr_count": flips,
                "acc_cfr_percent": (
                    round(100.0 * flips / len(clean_rows), 1) if clean_rows else None
                ),
            }
        output["strata"][label] = {
            "all_queries": len(rows),
            "clean_acc_queries": len(clean_rows),
            "methods": method_values,
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", default="out")
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    root = Path(args.out_root)
    dataset_names = tuple(name.strip() for name in args.datasets.split(",") if name.strip())
    output = {
        "schema": "reflow.matched_budget_stratified_acc_cfr.v1",
        "metric": "Acc-CFR",
        "strata": [label for label, _ in STRATA],
        "stratum_contract": (
            "Queries are grouped by ReFlow's realized n_modified_tokens. "
            "Within each group, every baseline uses the same per-query matched budget."
        ),
        "datasets": {
            dataset: aggregate_dataset(root / dataset) for dataset in dataset_names
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
