"""Measure how token scores align with executed singleton interventions."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.attribution_graph import DirectActivationAttributionGraphBuilder
from causalityrag.io import load_records, record_id, retrieved_contexts
from causalityrag.revision import apply_token_replacements
from causalityrag.token_units import units_from_cache_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--attention-prefix", required=True)
    parser.add_argument("--flow-rows", required=True)
    parser.add_argument("--replacement-registry", required=True)
    parser.add_argument("--units-cache", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--model-path",
        default="/data1/yujia/models/Qwen2.5-7B-Instruct",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def row_by_id(path: str, identifier: str) -> dict:
    for row in load_records(path):
        if str(row.get("id", row.get("qid", ""))) == identifier:
            return row
    raise ValueError(f"missing row for {identifier} in {path}")


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2:
        return 0.0
    left_ranks = average_ranks(left)
    right_ranks = average_ranks(right)
    if np.std(left_ranks) == 0 or np.std(right_ranks) == 0:
        return 0.0
    return float(np.corrcoef(left_ranks, right_ranks)[0, 1])


def top_overlap(left: np.ndarray, right: np.ndarray, k: int) -> float:
    k = min(k, len(left))
    if k <= 0:
        return 0.0
    left_top = set(np.argsort(left)[-k:])
    right_top = set(np.argsort(right)[-k:])
    return len(left_top & right_top) / k


def ndcg(scores: np.ndarray, effects: np.ndarray, k: int) -> float:
    """Measure how much executed intervention effect appears near the top."""
    k = min(k, len(scores))
    if k <= 0:
        return 0.0
    ranked = np.argsort(scores)[::-1][:k]
    ideal = np.argsort(effects)[::-1][:k]

    def dcg(indices: np.ndarray) -> float:
        return sum(
            (2.0 ** float(effects[index]) - 1.0) / math.log2(rank + 2.0)
            for rank, index in enumerate(indices)
        )

    denominator = dcg(ideal)
    return dcg(ranked) / denominator if denominator > 0.0 else 0.0


def top_effect(scores: np.ndarray, effects: np.ndarray, k: int) -> float:
    k = min(k, len(scores))
    if k <= 0:
        return 0.0
    return float(effects[np.argsort(scores)[::-1][:k]].sum())


def attention_by_unit(
    units: list[dict],
    token_metadata: list[dict],
    attention: np.ndarray,
) -> dict[str, float]:
    positive = np.maximum(attention.astype(np.float64), 0.0)
    totals = positive.sum(axis=1, keepdims=True)
    normalized = np.divide(
        positive,
        totals,
        out=np.zeros_like(positive),
        where=totals > 0,
    )
    token_scores = normalized.sum(axis=0)
    scores = {}
    for unit in units:
        value = 0.0
        for column, token in enumerate(token_metadata):
            if str(token.get("chunk_id", "")) != str(unit["chunk_id"]):
                continue
            if (
                int(token.get("chunk_char_start", -1))
                < int(unit["chunk_char_end"])
                and int(unit["chunk_char_start"])
                < int(token.get("chunk_char_end", -1))
            ):
                value += float(token_scores[column])
        scores[str(unit["unit_id"])] = value
    return scores


def target_logit(
    builder: DirectActivationAttributionGraphBuilder,
    record: dict,
    contexts: list[dict],
    target_answer: str,
) -> float:
    torch = builder.torch
    contexts = builder._truncate_contexts(contexts)
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
    positions = [
        int(token["position"])
        for token in metadata
        if token["region"] == "answer" and int(token["position"]) > 0
    ]
    if not positions:
        raise RuntimeError("target answer was truncated")
    with torch.no_grad():
        output = builder.model(input_ids=input_ids, use_cache=False)
        values = [
            output.logits[0, position - 1, input_ids[0, position]].float()
            for position in positions
        ]
        score = float(torch.stack(values).mean().item())
    del output
    return score


def main() -> None:
    args = parse_args()
    record = load_records(args.input)[args.index]
    identifier = record_id(record)
    flow_row = row_by_id(args.flow_rows, identifier)
    registry_row = row_by_id(args.replacement_registry, identifier)
    units_row = row_by_id(args.units_cache, identifier)
    units = units_from_cache_row(record, units_row, k=args.k)
    unit_by_id = {str(unit["unit_id"]): unit for unit in units}

    attention_prefix = Path(args.attention_prefix)
    arrays = np.load(str(attention_prefix) + ".npz")
    attention_metadata = json.loads(
        Path(str(attention_prefix) + ".json").read_text(encoding="utf-8")
    )
    if str(attention_metadata.get("id")) != identifier:
        raise ValueError("attention payload does not match requested query")

    replacements = registry_row.get("replacements", {})
    flow_scores = {
        str(key): float(value)
        for key, value in flow_row.get("unary_scores", {}).items()
    }
    candidate_ids = sorted(
        set(replacements) & set(flow_scores) & set(unit_by_id),
        key=lambda unit_id: (
            int(unit_by_id[unit_id].get("chunk_rank", 0)),
            int(unit_by_id[unit_id].get("chunk_char_start", 0)),
            unit_id,
        ),
    )
    if not candidate_ids:
        raise RuntimeError("no aligned replacement candidates")
    candidates = [unit_by_id[unit_id] for unit_id in candidate_ids]
    attention_scores = attention_by_unit(
        candidates,
        attention_metadata["context_token_metadata"],
        arrays["raw_attention"],
    )

    builder = DirectActivationAttributionGraphBuilder(
        args.model_path,
        device=args.device,
        dtype=args.dtype,
        max_edges=1,
    )
    target_answer = str(flow_row.get("target_answer") or record.get("answer", ""))
    clean_score = target_logit(
        builder,
        record,
        retrieved_contexts(record)[: args.k],
        target_answer,
    )

    rows = []
    for index, unit_id in enumerate(candidate_ids, 1):
        unit = unit_by_id[unit_id]
        revised = apply_token_replacements(
            record,
            [unit],
            {unit_id: replacements[unit_id]},
            k=args.k,
        )
        if revised["n_edits"] != 1:
            continue
        edited_score = target_logit(
            builder,
            record,
            revised["edited_contexts"],
            target_answer,
        )
        signed_drop = clean_score - edited_score
        rows.append({
            "unit_id": unit_id,
            "token": str(unit["text"]),
            "replacement": str(replacements[unit_id]["new"]),
            "chunk_rank": int(unit.get("chunk_rank", 0)),
            "chunk_char_start": int(unit.get("chunk_char_start", 0)),
            "attention_score": attention_scores[unit_id],
            "reflow_score": flow_scores[unit_id],
            "edited_target_logit": edited_score,
            "signed_logit_drop": signed_drop,
            "intervention_effect": max(0.0, signed_drop),
        })
        print(
            f"[token-alignment] {index}/{len(candidate_ids)} "
            f"{unit['text']}->{replacements[unit_id]['new']} "
            f"drop={signed_drop:.4f}",
            flush=True,
        )

    attention = np.asarray([row["attention_score"] for row in rows])
    reflow = np.asarray([row["reflow_score"] for row in rows])
    effect = np.asarray([row["intervention_effect"] for row in rows])
    payload = {
        "id": identifier,
        "question": str(record.get("question", "")),
        "target_answer": target_answer,
        "clean_target_logit": clean_score,
        "candidate_tokens": len(rows),
        "effect_definition": "positive drop in mean clean-answer raw logit after one frozen legal replacement",
        "metrics": {
            "attention_spearman": spearman(attention, effect),
            "reflow_spearman": spearman(reflow, effect),
            f"attention_top_{args.top_k}_overlap": top_overlap(
                attention, effect, args.top_k
            ),
            f"reflow_top_{args.top_k}_overlap": top_overlap(
                reflow, effect, args.top_k
            ),
            f"attention_ndcg_at_{args.top_k}": ndcg(
                attention, effect, args.top_k
            ),
            f"reflow_ndcg_at_{args.top_k}": ndcg(
                reflow, effect, args.top_k
            ),
            f"attention_top_{args.top_k}_effect": top_effect(
                attention, effect, args.top_k
            ),
            f"reflow_top_{args.top_k}_effect": top_effect(
                reflow, effect, args.top_k
            ),
        },
        "tokens": rows,
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["metrics"], indent=2), flush=True)


if __name__ == "__main__":
    main()
