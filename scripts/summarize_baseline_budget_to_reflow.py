"""Find the baseline token budget required to match ReFlow's Acc-CFR."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from causalityrag.reader import answers_match


METHODS = ("mirage", "arc_jsd")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summarize_dataset(base: Path, sweep_name: str) -> dict[str, Any]:
    factual_path = base / "audits/final_top10pool_k5/factual_metrics_1000.json"
    sweep_path = base / f"audits/baseline_budget_to_reflow/{sweep_name}"
    factual = json.loads(factual_path.read_text(encoding="utf-8"))
    sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
    target = float(factual["methods"]["reflow"]["acc_flip_ratio"])

    reflow_rows = load_jsonl(Path(factual["sources"]["reflow"]))
    baseline_rows = load_jsonl(Path(factual["sources"]["baselines"]))
    baseline_by_id = {str(row.get("id", "")): row for row in baseline_rows}
    clean_ids = {
        identifier
        for identifier, row in baseline_by_id.items()
        if answers_match(
            str(row.get("clean_answer", "")), str(row.get("gold_answer", ""))
        )
    }
    if len(clean_ids) != int(factual["methods"]["reflow"]["acc_clean_correct_queries"]):
        raise ValueError(f"clean-accuracy population mismatch under {base}")
    reflow_mean = sum(
        int(row.get("n_modified_tokens", 0))
        for row in reflow_rows
        if str(row.get("id", "")) in clean_ids
    ) / max(1, len(clean_ids))

    methods: dict[str, Any] = {
        "reflow": {
            "target_acc_cfr": target,
            "mean_modified_tokens": reflow_mean,
        }
    }
    for method in METHODS:
        curve = sweep["methods"][method]
        budgets = sorted(int(value) for value in curve)
        matched_budget = next(
            (
                budget
                for budget in budgets
                if float(curve[str(budget)]["acc_cfr"]) + 1e-12 >= target
            ),
            None,
        )
        if matched_budget is None:
            best_budget = max(
                budgets,
                key=lambda budget: (
                    float(curve[str(budget)]["acc_cfr"]),
                    -budget,
                ),
            )
            best_point = curve[str(best_budget)]
            methods[method] = {
                "status": "not_reached",
                "max_tested_budget": max(budgets),
                "best_top_k": best_budget,
                "best_acc_cfr": float(best_point["acc_cfr"]),
                "best_mean_modified_tokens": float(
                    best_point["mean_modified_tokens"]
                ),
                "target_shortfall": target - float(best_point["acc_cfr"]),
            }
            continue
        point = curve[str(matched_budget)]
        previous_budget = max(
            (budget for budget in budgets if budget < matched_budget),
            default=None,
        )
        methods[method] = {
            "status": "matched",
            "minimum_top_k": matched_budget,
            "acc_cfr": float(point["acc_cfr"]),
            "executed_queries": int(point["executed_queries"]),
            "mean_modified_tokens": float(point["mean_modified_tokens"]),
            "token_ratio_over_reflow": (
                float(point["mean_modified_tokens"]) / max(reflow_mean, 1e-12)
            ),
        }
        if previous_budget is not None:
            methods[method]["previous_top_k"] = previous_budget
            methods[method]["previous_acc_cfr"] = float(
                curve[str(previous_budget)]["acc_cfr"]
            )
    return {
        "queries": int(sweep["queries"]),
        "clean_acc_queries": len(clean_ids),
        "reflow_acc_cfr_target": target,
        "methods": methods,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", default="out")
    parser.add_argument("--datasets", default="hotpotqa,finqa")
    parser.add_argument("--sweep-name", default="summary_k1_32_1000.json")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    root = Path(args.out_root)
    datasets = [value.strip() for value in args.datasets.split(",") if value.strip()]
    output = {
        "schema": "reflow.baseline_budget_to_match_acc_cfr.v1",
        "metric": "Acc-CFR",
        "matching_rule": (
            "smallest fixed baseline top-k whose Acc-CFR is at least ReFlow's "
            "Table 3 Acc-CFR on the identical clean-accuracy population"
        ),
        "datasets": {
            dataset: summarize_dataset(root / dataset, args.sweep_name)
            for dataset in datasets
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
