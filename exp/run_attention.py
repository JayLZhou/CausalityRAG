"""Rank retrieved tokens by prompt-only last-layer attention mass."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.io import load_records, record_id, retrieved_contexts
from causalityrag.token_units import units_from_cache_row
from exp.arc_jsd import ArcJsdModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--units-cache", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    records = load_records(args.input)[args.start : args.start + args.n]
    unit_rows = {
        record_id(row): row for row in load_records(args.units_cache)
    }
    model = ArcJsdModel(
        args.model_path,
        device=args.device,
        dtype=args.dtype,
        attn_implementation="eager",
    )
    rows = []
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as output:
        for offset, record in enumerate(records):
            started = time.monotonic()
            index = args.start + offset
            identifier = record_id(record)
            units_row = unit_rows.get(identifier)
            if units_row is None:
                raise ValueError(f"missing frozen artifact for {identifier}")
            base = {
                "index": index,
                "id": identifier,
                "method": "prompt_only_last_layer_attention",
                "uses_answer_tokens": False,
                "layers": 1,
            }
            question = str(record.get("question", ""))
            contexts = retrieved_contexts(record)[: args.k]
            units = units_from_cache_row(record, units_row, k=args.k)
            scores = model.prompt_last_position_attention(
                question,
                contexts,
                units,
            )
            token_scores = {
                str(unit["unit_id"]): float(score)
                for unit, score in zip(units, scores)
            }
            ranked = sorted(
                token_scores,
                key=lambda unit_id: (-token_scores[unit_id], unit_id),
            )
            row = {
                **base,
                "status": "ok" if ranked else "no_ranked_tokens",
                "token_scores": token_scores,
                "ranked_ids": ranked,
                "prompt_tokens_only": True,
            }
            row["elapsed_seconds"] = round(time.monotonic() - started, 3)
            rows.append(row)
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            print(
                f"[attention] {len(rows)}/{len(records)} "
                f"status={row['status']}",
                flush=True,
            )

    summary = {
        "method": "prompt_only_last_layer_attention",
        "queries": len(rows),
        "ranked_queries": sum(row["status"] == "ok" for row in rows),
        "uses_answer_tokens": False,
        "layers": 1,
        "total_seconds": sum(float(row["elapsed_seconds"]) for row in rows),
        "out": os.path.abspath(args.out),
    }
    with open(args.summary_out, "w", encoding="utf-8") as output:
        json.dump(summary, output, indent=2, ensure_ascii=False)
        output.write("\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
