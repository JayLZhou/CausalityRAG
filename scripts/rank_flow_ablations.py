#!/usr/bin/env python3
"""Produce ranking JSONL for matched-budget evaluation; never modify graphs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from causalityrag.flow_ablation import METHODS, rank_graph_tokens
from causalityrag.io import iter_records
from causalityrag.shared_replacement_pool import FrozenSharedReplacementPool, file_sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphs", required=True)
    parser.add_argument("--shared-pool", required=True)
    parser.add_argument("--expected-pool-sha256", required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, default=0, help="0 means all graph rows")
    args = parser.parse_args()
    if args.n < 0:
        parser.error("--n must be nonnegative")
    pool_sha = file_sha256(args.shared_pool)
    if pool_sha != args.expected_pool_sha256:
        raise ValueError("frozen pool SHA mismatch")
    pool = FrozenSharedReplacementPool(args.shared_pool)
    graph_sha = file_sha256(args.graphs)
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    # Exclusive creation prevents accidental replacement of previous results.
    with output.open("x", encoding="utf-8") as destination:
        for index, graph in enumerate(iter_records(args.graphs, limit=args.n)):
            query_id = str(graph.get("id", ""))
            if not query_id or query_id in seen:
                raise ValueError(f"missing or duplicate graph query ID: {query_id}")
            seen.add(query_id)
            started = time.perf_counter()
            eligible = {
                str(unit_id) for unit_id in graph.get("contribution_graph", {}).get("token_nodes", [])
                if pool.is_eligible(str(unit_id))
            }
            result = rank_graph_tokens(graph, eligible, method=args.method)
            result.update({
                "index": index,
                "selection_seconds": time.perf_counter() - started,
                "pool_sha256": pool_sha,
                "graphs_sha256": graph_sha,
                "graph_row_sha256": hashlib.sha256(
                    json.dumps(graph, sort_keys=True, ensure_ascii=True).encode()
                ).hexdigest(),
                "schema": "causalityrag.flow_ablation_ranking.v1",
            })
            destination.write(json.dumps(result, allow_nan=False) + "\n")
            destination.flush()
            print(f"{args.method}: {index + 1} queries ranked; reader verification not run", flush=True)


if __name__ == "__main__":
    main()
