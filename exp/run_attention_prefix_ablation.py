"""Experimental ablation: select tokens with response-to-context attention."""

from __future__ import annotations

import argparse
from itertools import islice
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.attribution_graph import (  # noqa: E402
    DirectActivationAttributionGraphBuilder,
)
from causalityrag.io import iter_records, load_records, record_id, retrieved_contexts  # noqa: E402
from causalityrag.token_units import units_from_cache_row  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--targets", required=True)
    parser.add_argument("--units-cache", required=True)
    parser.add_argument("--replacement-registry", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--model-path",
        default="/data1/yujia/models/Qwen2.5-7B-Instruct",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--budgets", default="1,3,5")
    parser.add_argument("--last-layers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    return parser.parse_args()


def attention_by_unit(
    units: list[dict],
    token_metadata: list[dict],
    token_scores,
) -> dict[str, float]:
    scores = {}
    for unit in units:
        value = 0.0
        for position, token in enumerate(token_metadata):
            if str(token.get("chunk_id", "")) != str(unit["chunk_id"]):
                continue
            if (
                int(token.get("chunk_char_start", -1))
                < int(unit["chunk_char_end"])
                and int(unit["chunk_char_start"])
                < int(token.get("chunk_char_end", -1))
            ):
                value += float(token_scores[position])
        scores[str(unit["unit_id"])] = value
    return scores


def score_attention(
    builder: DirectActivationAttributionGraphBuilder,
    record: dict,
    target_answer: str,
    units: list[dict],
    *,
    k: int,
    last_layers: int,
) -> dict[str, float]:
    torch = builder.torch
    contexts = builder._truncate_contexts(retrieved_contexts(record)[:k])
    text, spans = builder._render(record, contexts, target_answer)
    encoded = builder.tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_tensors="pt",
        truncation=True,
        max_length=builder.max_length,
    )
    input_ids = encoded["input_ids"].to(builder.device)
    offsets = [
        tuple(map(int, pair))
        for pair in encoded["offset_mapping"][0].tolist()
    ]
    metadata = builder._token_metadata(input_ids[0].tolist(), offsets, spans)
    answer_positions = [
        int(token["position"])
        for token in metadata
        if token["region"] == "answer"
    ]
    if not answer_positions:
        raise RuntimeError("answer tokens were truncated")

    with torch.no_grad():
        output = builder.model(
            input_ids=input_ids,
            output_attentions=True,
            use_cache=False,
        )
    attentions = output.attentions
    if not attentions or any(attention is None for attention in attentions):
        raise RuntimeError("model returned no eager attentions")
    layers = attentions[-min(last_layers, len(attentions)):]
    token_scores = torch.zeros(input_ids.shape[1], dtype=torch.float32)
    for attention in layers:
        matrix = attention[0].detach().float().mean(dim=0).cpu()
        token_scores += matrix[answer_positions].sum(dim=0)
    del output
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return attention_by_unit(units, metadata, token_scores.numpy())


def main() -> None:
    args = parse_args()
    budgets = sorted({int(value) for value in args.budgets.split(",") if value})
    if not budgets or budgets[0] <= 0:
        raise ValueError("budgets must contain positive integers")
    unit_rows = {str(row["id"]): row for row in load_records(args.units_cache)}
    registry_rows = {
        str(row["id"]): row for row in load_records(args.replacement_registry)
    }
    records = islice(iter_records(args.input), args.start, args.start + args.n)
    targets = islice(iter_records(args.targets), args.start, args.start + args.n)
    builder = DirectActivationAttributionGraphBuilder(
        args.model_path,
        device=args.device,
        dtype=args.dtype,
        max_edges=1,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as output:
        for offset, (record, target_row) in enumerate(zip(records, targets)):
            started = time.monotonic()
            index = args.start + offset
            identifier = record_id(record)
            if str(target_row.get("id")) != identifier:
                raise ValueError(f"input/target mismatch at row {index}")
            units = units_from_cache_row(record, unit_rows[identifier], k=args.k)
            replacements = registry_rows[identifier].get("replacements", {})
            valid_ids = {
                str(unit_id)
                for unit_id, replacement in replacements.items()
                if isinstance(replacement, dict) and replacement.get("ok")
            }
            try:
                scores = score_attention(
                    builder,
                    record,
                    str(target_row.get("target_answer", "")),
                    units,
                    k=args.k,
                    last_layers=args.last_layers,
                )
                ranked = sorted(
                    (
                        (unit_id, score)
                        for unit_id, score in scores.items()
                        if unit_id in valid_ids
                    ),
                    key=lambda item: (-item[1], item[0]),
                )
                unit_by_id = {str(unit["unit_id"]): unit for unit in units}
                total = sum(score for _, score in ranked)
                prefixes = {}
                for budget in budgets:
                    if len(ranked) < budget:
                        continue
                    selected = ranked[:budget]
                    selected_ids = [unit_id for unit_id, _ in selected]
                    retained = total - sum(score for _, score in selected)
                    prefixes[str(budget)] = {
                        "budget": budget,
                        "selected_ids": selected_ids,
                        "n_selected": budget,
                        "remaining_support_flow": retained,
                        "remaining_support_fraction": retained / total if total else 0.0,
                        "selected_tokens": [
                            str(unit_by_id[unit_id].get("text", ""))
                            for unit_id in selected_ids
                        ],
                    }
                status = "ok" if str(max(budgets)) in prefixes else "insufficient_editable_tokens"
            except RuntimeError as error:
                scores = {}
                ranked = []
                prefixes = {}
                total = 0.0
                status = str(error).replace(" ", "_")

            row = {
                "index": index,
                "id": identifier,
                "question": str(record.get("question", "")),
                "variant": "attention",
                "status": status,
                "last_layers": args.last_layers,
                "initial_flow": total,
                "available_tokens": len(ranked),
                "ranked_ids": [unit_id for unit_id, _ in ranked],
                "ranked_scores": [float(score) for _, score in ranked],
                "prefixes": prefixes,
                "elapsed_seconds": time.monotonic() - started,
            }
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            print(
                f"[attention-ablation] {offset + 1}/{args.n} index={index} "
                f"status={status} seconds={row['elapsed_seconds']:.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
