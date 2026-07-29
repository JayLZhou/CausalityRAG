"""Combine factual and meaning-preserving control flip metrics."""

from __future__ import annotations

import argparse
import json
import os
import statistics


METRICS = {
    "answer": ("answer_flip_ratio", "answer_flip_rate_itt"),
    "f1": ("f1_flip_ratio", "f1_flip_rate_itt"),
    "em": ("em_flip_ratio", "em_flip_rate_itt"),
    "acc": ("acc_flip_ratio", "acc_flip_rate_itt"),
}
RANDOM_METHODS = [f"random_seed{seed}" for seed in range(5)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factual", required=True)
    parser.add_argument("--meaning-preserving", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def metric_rows(
    factual_methods: dict,
    control_methods: dict,
    method_names: list[str],
) -> dict:
    result = {}
    for metric, (factual_key, control_key) in METRICS.items():
        factual = statistics.fmean(
            float(factual_methods[name][factual_key])
            for name in method_names
        )
        control = statistics.fmean(
            float(control_methods[name][control_key])
            for name in method_names
        )
        adjusted = factual - control
        if adjusted < -1e-12:
            raise ValueError(
                f"negative adjusted {metric} flip for {method_names}: "
                f"factual={factual}, control={control}"
            )
        result[metric] = {
            "factual_flip_ratio": factual,
            "meaning_preserving_flip_ratio": control,
            "adjusted_flip_ratio": max(0.0, adjusted),
        }
    return result


def main() -> None:
    args = parse_args()
    factual = load_json(args.factual)
    control = load_json(args.meaning_preserving)
    factual_methods = factual["methods"]
    control_methods = control["methods"]
    names = sorted(
        (set(factual_methods) & set(control_methods)) - set(RANDOM_METHODS)
    )
    methods = {
        name: metric_rows(factual_methods, control_methods, [name])
        for name in names
    }
    methods["random"] = metric_rows(
        factual_methods,
        control_methods,
        RANDOM_METHODS,
    )
    result = {
        "population": (
            "All 1,000 queries; unsupported or unexecuted interventions "
            "contribute zero flips."
        ),
        "adjustment": (
            "factual counterfactual flip ratio minus the flip ratio produced "
            "by meaning-preserving replacements at the same selected positions"
        ),
        "sources": {
            "factual": os.path.abspath(args.factual),
            "meaning_preserving": os.path.abspath(args.meaning_preserving),
        },
        "methods": methods,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
