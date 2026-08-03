#!/usr/bin/env python3
"""Merge previously executed fixed-budget answers into a reusable seed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    roots = [
        dataset_dir / "audits/fixed_budget_acc",
        dataset_dir / "audits/baseline_budget_to_reflow",
    ]
    files = []
    for root in roots:
        if root.is_dir():
            files.extend(root.rglob("*.jsonl"))
    files = sorted(
        path for path in files
        if "exact_points_v2" not in path.parts
        and path != Path(args.out)
    )

    merged: dict[str, dict] = {}
    for path in files:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                identifier = str(row.get("id", ""))
                if not identifier or not isinstance(row.get("methods"), dict):
                    continue
                target = merged.setdefault(identifier, {
                    "index": int(row.get("index", -1)),
                    "id": identifier,
                    "methods": {},
                })
                for method, points in row["methods"].items():
                    if not isinstance(points, dict):
                        continue
                    target_points = target["methods"].setdefault(method, {})
                    for budget, point in points.items():
                        if isinstance(point, dict) and point.get("edited_answer"):
                            target_points[str(budget)] = {
                                "status": point.get("status", ""),
                                "selected_ids": point.get("selected_ids", []),
                                "n_modified_tokens": int(
                                    point.get("n_modified_tokens", 0)
                                ),
                                "edited_answer": point["edited_answer"],
                                "reader_called": False,
                                "acc_flip": bool(point.get("acc_flip")),
                            }

    rows = sorted(merged.values(), key=lambda row: int(row["index"]))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({
        "sources": len(files),
        "rows": len(rows),
        "out": str(out_path),
    }))


if __name__ == "__main__":
    main()
