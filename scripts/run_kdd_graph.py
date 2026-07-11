"""Build target-specific attribution DAGs for a small RAG dataset."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.attribution_graph import (
    AttentionAttributionGraphBuilder,
    DirectActivationAttributionGraphBuilder,
    NativeMLPAttributionGraphBuilder,
)
from causalityrag.io import iter_records, retrieved_contexts
from causalityrag.reader import ReaderClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--target", choices=["gold", "reader"], default="reader")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-context-tokens", type=int, default=800)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--edge-topk", type=int, default=6)
    parser.add_argument("--max-receivers-per-layer", type=int, default=48)
    parser.add_argument("--max-edges", type=int, default=5000)
    parser.add_argument("--top-tokens", type=int, default=50)
    parser.add_argument(
        "--graph-method",
        choices=["direct-activation", "native-mlp", "attention-rollout"],
        default="direct-activation",
    )
    args = parser.parse_args()

    records = list(iter_records(args.input, args.n))
    reader = ReaderClient() if args.target == "reader" else None
    targets = []
    for record in records:
        if reader:
            targets.append(reader.answer(str(record.get("question", "")), retrieved_contexts(record)[: args.k]))
        else:
            targets.append(str(record.get("answer") or record.get("clean_answer") or ""))

    builder_cls = {
        "direct-activation": DirectActivationAttributionGraphBuilder,
        "native-mlp": NativeMLPAttributionGraphBuilder,
        "attention-rollout": AttentionAttributionGraphBuilder,
    }[args.graph_method]
    builder = builder_cls(
        args.model_path,
        device=args.device,
        max_context_tokens=args.max_context_tokens,
        max_length=args.max_length,
        edge_topk=args.edge_topk,
        max_receivers_per_layer=args.max_receivers_per_layer,
        max_edges=args.max_edges,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    rows = []
    with open(args.out, "w", encoding="utf-8") as output:
        for index, (record, target) in enumerate(zip(records, targets), 1):
            started = time.monotonic()
            row = builder.build(record, target, k=args.k, top_tokens=args.top_tokens)
            row["elapsed_seconds"] = round(time.monotonic() - started, 3)
            rows.append(row)
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            print(
                f"[attribution-graph] {index}/{len(records)} status={row['status']} "
                f"tokens={row['graph'].get('sequence_tokens', 0)} "
                f"seconds={row['elapsed_seconds']}",
                flush=True,
            )

    ok = [row for row in rows if row["status"] == "ok"]
    print("[attribution-graph summary]", json.dumps({
        "records": len(rows),
        "ok": len(ok),
        "avg_seconds": round(sum(row["elapsed_seconds"] for row in ok) / max(1, len(ok)), 3),
        "method": builder.method,
        "out": args.out,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
