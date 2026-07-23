"""Legacy experiment: run the token-level ILP selector over retrieval records.

Example on Server A:

  /data1/yujia/envs/spacyner/bin/python exp/run_token_ilp.py \
    --input /data1/yujia/RAGData/hotpotqa-exp/results/retrieval_hotpotqa_vdb.jsonl \
    --out /data1/yujia/CausalityRAG/out/token_ilp_smoke.jsonl \
    --n 10 --k 5 --tau-graph 0.2
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.io import iter_records
from causalityrag.token_units import run_token_ilp_record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="JSON/JSONL records with question, answer, retrieved contexts")
    parser.add_argument("--out", required=True, help="output JSONL audit path")
    parser.add_argument("--n", type=int, default=0, help="limit examples; 0 means all")
    parser.add_argument("--k", type=int, default=5, help="retrieved contexts per example")
    parser.add_argument("--objective", choices=["min-cost", "budgeted"], default="min-cost")
    parser.add_argument("--tau-graph", type=float, default=0.2)
    parser.add_argument("--target-support", type=float, default=0.0)
    parser.add_argument("--budget", type=float, default=5.0)
    parser.add_argument("--solver", choices=["auto", "scipy", "dp", "branch"], default="auto")
    parser.add_argument("--support-mode", choices=["proxy", "precomputed"], default="proxy")
    parser.add_argument("--cost-mode", choices=["unit", "chars"], default="unit")
    parser.add_argument("--include-stopwords", action="store_true")
    parser.add_argument("--include-units", action="store_true")
    parser.add_argument("--top-candidates", type=int, default=20)
    parser.add_argument("--answer-boost", type=float, default=8.0)
    parser.add_argument("--question-boost", type=float, default=2.0)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    rows = []
    with open(args.out, "w", encoding="utf-8") as f:
        for record in iter_records(args.input, args.n):
            row = run_token_ilp_record(
                record,
                k=args.k,
                objective=args.objective,
                tau_graph=args.tau_graph,
                target_support=args.target_support if args.target_support > 0 else None,
                budget=args.budget,
                solver=args.solver,
                support_mode=args.support_mode,
                cost_mode=args.cost_mode,
                include_stopwords=args.include_stopwords,
                include_units=args.include_units,
                top_candidates=args.top_candidates,
                answer_boost=args.answer_boost,
                question_boost=args.question_boost,
            )
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    solved = [row for row in rows if row["status"] == "optimal"]
    summary = {
        "records": len(rows),
        "optimal": len(solved),
        "avg_cut_cost": round(sum(row["cut_cost"] for row in solved) / max(1, len(solved)), 3),
        "avg_support_removed": round(sum(row["support_removed"] for row in solved) / max(1, len(solved)), 3),
        "out": args.out,
    }
    print("[token-ilp summary]", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
