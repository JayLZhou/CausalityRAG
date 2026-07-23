"""Build target-specific contribution DAGs for RAG answer resilience."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from itertools import islice

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.attribution_graph import (
    AttentionAttributionGraphBuilder,
    DirectActivationAttributionGraphBuilder,
    NativeMLPAttributionGraphBuilder,
)
from causalityrag.io import iter_records, record_id, retrieved_contexts
from causalityrag.reader import (
    ReaderClient,
    answers_exact_match,
    parse_json_object,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", default="")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--target", choices=["gold", "reader", "results"], default="reader")
    parser.add_argument("--target-results", nargs="+", default=[])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--max-context-tokens", type=int, default=800)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--edge-topk", type=int, default=6)
    parser.add_argument("--max-receivers-per-layer", type=int, default=48)
    parser.add_argument("--max-edges", type=int, default=5000)
    parser.add_argument("--top-tokens", type=int, default=50)
    parser.add_argument("--closed-flow", action="store_true")
    parser.add_argument("--absorbing-flow", action="store_true")
    parser.add_argument(
        "--graph-method",
        choices=["direct-activation", "native-mlp", "attention-rollout"],
        default="direct-activation",
    )
    args = parser.parse_args()
    if args.start < 0 or args.n <= 0 or args.k <= 0:
        parser.error("--start must be non-negative; --n and --k must be positive")
    if args.closed_flow and args.graph_method != "direct-activation":
        parser.error("--closed-flow requires --graph-method direct-activation")
    if args.absorbing_flow and args.graph_method != "direct-activation":
        parser.error("--absorbing-flow requires --graph-method direct-activation")
    if args.closed_flow and args.absorbing_flow:
        parser.error("--closed-flow and --absorbing-flow are mutually exclusive")

    records = list(islice(
        iter_records(args.input),
        args.start,
        args.start + args.n,
    ))
    reader = ReaderClient() if args.target == "reader" else None
    if args.target == "results":
        if not args.target_results:
            parser.error("--target-results is required with --target results")
        target_by_id = load_targets_by_id(args.target_results)
        missing = [
            record_id(record)
            for record in records
            if record_id(record) not in target_by_id
        ]
        if missing:
            raise ValueError(f"missing cached clean targets: {missing[:5]}")
        targets = [target_by_id[record_id(record)] for record in records]
    else:
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
        dtype=args.dtype,
        max_context_tokens=args.max_context_tokens,
        max_length=args.max_length,
        edge_topk=args.edge_topk,
        max_receivers_per_layer=args.max_receivers_per_layer,
        max_edges=args.max_edges,
        closed_flow=args.closed_flow,
        absorbing_flow=args.absorbing_flow,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    rows = []
    with open(args.out, "w", encoding="utf-8") as output:
        for index, (record, target) in enumerate(zip(records, targets), 1):
            started = time.monotonic()
            row = builder.build(record, target, k=args.k, top_tokens=args.top_tokens)
            row["clean_answer"] = target
            row["clean_correct"] = answers_exact_match(
                target,
                str(record.get("answer", "")),
            )
            row["elapsed_seconds"] = round(time.monotonic() - started, 3)
            rows.append(row)
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            print(
                f"[contribution-graph] {index}/{len(records)} status={row['status']} "
                f"tokens={row['graph'].get('sequence_tokens', 0)} "
                f"seconds={row['elapsed_seconds']}",
                flush=True,
            )

    ok = [row for row in rows if row["status"] == "ok"]
    summary = {
        "records": len(rows),
        "ok": len(ok),
        "avg_seconds": round(sum(row["elapsed_seconds"] for row in ok) / max(1, len(ok)), 3),
        "method": builder.method,
        "out": args.out,
    }
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    print("[contribution-graph summary]", rendered)
    if args.summary_out:
        with open(args.summary_out, "w", encoding="utf-8") as output:
            output.write(rendered + "\n")


def answer_from_response(response: str) -> str:
    parsed = parse_json_object(response)
    if isinstance(parsed, dict):
        return str(parsed.get("answer", "")).strip()
    return response.strip()


def answer_from_result_row(row: dict) -> str:
    """Read a cached clean target without issuing another reader call."""

    response = str(row.get("clean_response", "")).strip()
    if response:
        return answer_from_response(response)
    for key in ("clean_answer", "stored_clean_answer", "target_answer", "answer"):
        value = str(row.get(key, "")).strip()
        if value:
            return value
    raise ValueError(f"target result row {row.get('id')} has no clean answer")


def load_targets_by_id(paths: list[str]) -> dict[str, str]:
    targets = {}
    for path in paths:
        for row in iter_records(path):
            identifier = record_id(row)
            if identifier:
                targets[identifier] = answer_from_result_row(row)
    return targets


if __name__ == "__main__":
    main()
