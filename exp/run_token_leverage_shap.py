"""Estimate retrieved-token Shapley values with LeverageSHAP sampling.

This implements the paired, without-replacement leverage-score regression of
Musco and Witter (ICLR 2025). Coalition sizes are allocated deterministically
to keep the requested evaluation budget fixed, as discussed in their paper.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.io import load_records, record_id, retrieved_contexts
from causalityrag.reader import reader_completion_text
from causalityrag.token_units import units_from_context_row
from exp.arc_jsd import ArcJsdModel
from exp.run_token_shapley import objective_batch
from exp.run_token_svarm import _coalition_players, _evaluate_cached_masks, _mask


def _allocate_pair_counts(n_players: int, n_pairs: int) -> dict[int, int]:
    """Allocate paired samples evenly over leverage-score coalition sizes."""

    sizes = list(range(1, n_players // 2 + 1))
    size_weights = {
        size: (0.5 if n_players % 2 == 0 and size == n_players // 2 else 1.0)
        for size in sizes
    }
    total_weight = sum(size_weights.values())
    raw = {size: n_pairs * size_weights[size] / total_weight for size in sizes}
    capacities = {
        size: (
            math.comb(n_players, size) // 2
            if n_players % 2 == 0 and size == n_players // 2
            else math.comb(n_players, size)
        )
        for size in sizes
    }
    counts = {
        size: min(int(math.floor(raw[size])), capacities[size]) for size in sizes
    }
    remaining = n_pairs - sum(counts.values())
    order = sorted(sizes, key=lambda item: (-(raw[item] - counts[item]), item))
    while remaining > 0:
        changed = False
        for size in order:
            if remaining == 0:
                break
            if counts[size] < capacities[size]:
                counts[size] += 1
                remaining -= 1
                changed = True
        if not changed:
            break
    return counts


def _paired_leverage_masks(
    n_players: int, budget: int, seed: int
) -> tuple[list[list[bool]], list[float], dict[int, int]]:
    """Sample paired coalitions without replacement at a fixed budget."""

    internal_budget = max(0, budget - 2)
    n_pairs = internal_budget // 2
    pair_counts = _allocate_pair_counts(n_players, n_pairs)
    rng = random.Random(seed)
    masks: list[list[bool]] = []
    seen_pairs: set[tuple[int, ...]] = set()

    for size, count in pair_counts.items():
        for _ in range(count):
            while True:
                subset = tuple(sorted(rng.sample(range(n_players), size)))
                if n_players % 2 == 0 and size == n_players // 2:
                    subset_set = set(subset)
                    complement = tuple(i for i in range(n_players) if i not in subset_set)
                    key = min(subset, complement)
                else:
                    key = subset
                if key not in seen_pairs:
                    seen_pairs.add(key)
                    break
            coalition = _mask(n_players, subset)
            masks.append(coalition)
            masks.append([not present for present in coalition])

    row_counts: dict[int, int] = {}
    for coalition in masks:
        size = sum(coalition)
        row_counts[size] = row_counts.get(size, 0) + 1
    weights = [
        1.0 / (size * (n_players - size) * row_counts[size])
        for size in (sum(coalition) for coalition in masks)
    ]
    return masks, weights, row_counts


def sampled_token_leverage_shap_scores(
    model: ArcJsdModel,
    question: str,
    contexts: list[dict],
    units: list[dict],
    response_ids: list[int],
    *,
    budget: int,
    batch_size: int,
    seed: int,
) -> tuple[list[float], dict]:
    """Estimate one Shapley value per retrieved-token unit."""

    torch = model.torch
    prompt_ids, positions_by_unit = model._prompt_context_token_positions(
        question, contexts, units
    )
    sequence = prompt_ids + list(response_ids)
    input_ids = torch.tensor([sequence], dtype=torch.long, device=model.device)
    base = model.model.get_input_embeddings()(input_ids).detach()
    player_members, context_positions, position_owners = _coalition_players(
        units, positions_by_unit
    )
    n_players = len(player_members)
    if n_players == 0:
        return [0.0] * len(units), {
            "method": "token_leverage_shap",
            "estimator": "LeverageSHAP",
            "coalition_players": 0,
            "unique_coalition_evaluations": 0,
        }
    if n_players == 1:
        effective_budget = 2
        internal_masks: list[list[bool]] = []
        regression_weights: list[float] = []
        row_counts: dict[int, int] = {}
    else:
        effective_budget = min(max(budget, n_players + 2), 2**n_players)
        if effective_budget % 2 == 1:
            effective_budget += 1
        internal_masks, regression_weights, row_counts = _paired_leverage_masks(
            n_players, effective_budget, seed
        )

    endpoints = [[False] * n_players, [True] * n_players]
    all_masks = internal_masks + endpoints
    all_values, unique_evaluations = _evaluate_cached_masks(
        model,
        base,
        len(prompt_ids),
        response_ids,
        context_positions,
        position_owners,
        all_masks,
        batch_size,
    )
    empty_value, full_value = all_values[-2:]
    gap = full_value - empty_value
    if n_players == 1:
        player_scores = [gap]
        rank = 1
        condition_number = 1.0
    else:
        coalitions = np.asarray(internal_masks, dtype=np.float64)
        values = np.asarray(all_values[:-2], dtype=np.float64) - empty_value
        sizes = np.sum(coalitions, axis=1)
        centered_design = coalitions - sizes[:, None] / n_players
        centered_values = values - gap * sizes / n_players
        sqrt_weights = np.sqrt(np.asarray(regression_weights, dtype=np.float64))
        weighted_design = centered_design * sqrt_weights[:, None]
        weighted_values = centered_values * sqrt_weights
        centered_scores, _, rank, singular_values = np.linalg.lstsq(
            weighted_design, weighted_values, rcond=None
        )
        centered_scores -= np.mean(centered_scores)
        player_scores = (centered_scores + gap / n_players).tolist()
        positive_singular = singular_values[singular_values > 1e-12]
        condition_number = (
            float(positive_singular[0] / positive_singular[-1])
            if len(positive_singular) > 0
            else float("inf")
        )

    scores = [0.0] * len(units)
    for player, members in enumerate(player_members):
        share = player_scores[player] / len(members)
        for owner in members:
            scores[owner] = share
    return scores, {
        "method": "token_leverage_shap",
        "estimator": "LeverageSHAP",
        "variant": "paired_without_replacement_fixed_budget",
        "reference": "Musco_Witter_ICLR_2025",
        "requested_budget": budget,
        "effective_budget": effective_budget,
        "unique_coalition_evaluations": unique_evaluations,
        "coalition_batch_size": batch_size,
        "context_baseline": "zero_embedding_only_context_positions",
        "prompt_tokens": len(prompt_ids),
        "response_tokens": len(response_ids),
        "context_units": len(units),
        "coalition_players": n_players,
        "overlap_groups": sum(len(members) > 1 for members in player_members),
        "model_subtokens": len(context_positions),
        "samples_per_coalition_size": row_counts,
        "regression_rank": int(rank),
        "regression_condition_number": condition_number,
        "efficiency_error": sum(player_scores) - gap,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--clean-reference", required=True)
    parser.add_argument("--context-units", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", default="")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--n", type=int, default=0)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--attention-candidates",
        type=int,
        default=128,
        help=(
            "Run LeverageSHAP only over this many highest-attention tokens; "
            "use 0 to retain the full-token game."
        ),
    )
    parser.add_argument("--budget-multiplier", type=float, default=5.0)
    parser.add_argument("--budget", type=int, default=0)
    parser.add_argument("--coalition-batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.start, args.n, args.budget, args.attention_candidates) < 0:
        parser.error(
            "start, n, budget, and attention-candidates must be non-negative"
        )
    if min(args.k, args.top_k, args.coalition_batch_size) <= 0:
        parser.error("k, top-k, and coalition-batch-size must be positive")
    if args.budget_multiplier <= 0:
        parser.error("budget-multiplier must be positive")

    records = load_records(args.input)
    references = {record_id(row): row for row in load_records(args.clean_reference)}
    units_by_id = {record_id(row): row for row in load_records(args.context_units)}
    end = len(records) if args.n == 0 else min(len(records), args.start + args.n)
    selected_records = list(enumerate(records))[args.start:end]
    if args.resume and os.path.isfile(args.out):
        existing_ids = {record_id(row) for row in load_records(args.out)}
        selected_records = [
            (index, record)
            for index, record in selected_records
            if record_id(record) not in existing_ids
        ]
    model = ArcJsdModel(
        args.model_path,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )

    rows: list[dict] = []
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    output_mode = "a" if args.resume else "w"
    with open(args.out, output_mode, encoding="utf-8") as output:
        for index, record in selected_records:
            started = time.monotonic()
            identifier = record_id(record)
            reference = references.get(identifier)
            units_row = units_by_id.get(identifier)
            if reference is None or units_row is None:
                raise ValueError(f"missing frozen artifact for {identifier}")
            contexts = retrieved_contexts(record)[: args.k]
            units = units_from_context_row(record, units_row, k=args.k)
            clean_answer = str(reference.get("clean_answer", "")).strip()
            row = {
                "index": index,
                "id": identifier,
                "question": str(record.get("question", "")),
                "baseline": (
                    "attention_filtered_token_leverage_shap"
                    if args.attention_candidates
                    else "token_leverage_shap"
                ),
                "method": (
                    "attention_filtered_token_leverage_shap"
                    if args.attention_candidates
                    else "token_leverage_shap"
                ),
                "top_k": args.top_k,
                "status": "ok",
                "objective": "mean_teacher_forced_clean_answer_log_likelihood",
                "attribution_backend": "matching_hf_reader",
                "selection_domain": (
                    f"top_{args.attention_candidates}_attention_context_tokens"
                    if args.attention_candidates
                    else "all_non_punctuation_context_tokens"
                ),
            }
            if not clean_answer:
                row.update({
                    "status": "reader_abstention_empty_answer",
                    "token_scores": {},
                    "selection_candidate": None,
                    "candidates": [],
                })
            else:
                response_text = reader_completion_text(
                    clean_answer,
                    reader_mode=os.environ.get("CAUSALITYRAG_READER_MODE", "short_answer"),
                    compact_json=True,
                )
                trajectory = model.trajectory_for_response(
                    str(record.get("question", "")), contexts, response_text
                )
                attention_scores = model.prompt_last_position_attention(
                    str(record.get("question", "")), contexts, units
                )
                attention_order = sorted(
                    range(len(units)),
                    key=lambda owner: (
                        -float(attention_scores[owner]),
                        str(units[owner]["unit_id"]),
                    ),
                )
                candidate_count = (
                    min(args.attention_candidates, len(units))
                    if args.attention_candidates
                    else len(units)
                )
                candidate_owners = attention_order[:candidate_count]
                candidate_units = [units[owner] for owner in candidate_owners]
                budget = args.budget or int(
                    math.ceil(args.budget_multiplier * len(candidate_units))
                )
                scores, diagnostics = sampled_token_leverage_shap_scores(
                    model,
                    str(record.get("question", "")),
                    contexts,
                    candidate_units,
                    trajectory.response_ids,
                    budget=budget,
                    batch_size=args.coalition_batch_size,
                    seed=args.seed + index,
                )
                score_map = {
                    str(unit["unit_id"]): float(score)
                    for unit, score in zip(candidate_units, scores)
                }
                shapley_order = sorted(
                    candidate_owners,
                    key=lambda owner: (
                        -score_map[str(units[owner]["unit_id"])],
                        str(units[owner]["unit_id"]),
                    ),
                )
                candidate_set = set(candidate_owners)
                ranked_owners = shapley_order + [
                    owner for owner in attention_order if owner not in candidate_set
                ]
                ranked_ids = [str(units[owner]["unit_id"]) for owner in ranked_owners]
                selected_owners = ranked_owners[: args.top_k]
                candidate = {
                    "status": "fixed_topk",
                    "selected_ids": [
                        str(units[owner]["unit_id"]) for owner in selected_owners
                    ],
                    "selected_tokens": [
                        str(units[owner].get("text", "")) for owner in selected_owners
                    ],
                    "selected_scores": [
                        score_map.get(
                            str(units[owner]["unit_id"]),
                            float(attention_scores[owner]),
                        )
                        for owner in selected_owners
                    ],
                    "n_selected": len(selected_owners),
                    "score": "attention_filtered_leverage_shap",
                } if selected_owners else None
                diagnostics.update({
                    "attention_filter": "prompt_only_last_layer_attention",
                    "attention_candidate_limit": args.attention_candidates,
                    "attention_candidates": candidate_count,
                    "full_context_units": len(units),
                    "ranking_tail": "prompt_only_last_layer_attention",
                })
                row.update({
                    "status": "ok" if candidate else "no_positive_editable_token",
                    "clean_answer": clean_answer,
                    "token_scores": score_map,
                    "attention_scores": {
                        str(unit["unit_id"]): float(score)
                        for unit, score in zip(units, attention_scores)
                    },
                    "ranked_ids": ranked_ids,
                    "selection_candidate": candidate,
                    "candidates": [candidate] if candidate else [],
                    "diagnostics": diagnostics,
                })
            row["elapsed_seconds"] = round(time.monotonic() - started, 3)
            rows.append(row)
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            if model.torch.cuda.is_available():
                model.torch.cuda.empty_cache()
            print(
                f"[token_leverage_shap] {len(rows)}/{len(selected_records)} "
                f"status={row['status']} seconds={row['elapsed_seconds']}",
                flush=True,
            )

    all_rows = load_records(args.out)
    summary = {
        "baseline": (
            "attention_filtered_token_leverage_shap"
            if args.attention_candidates
            else "token_leverage_shap"
        ),
        "method": (
            "attention_filtered_token_leverage_shap"
            if args.attention_candidates
            else "token_leverage_shap"
        ),
        "queries": len(all_rows),
        "queries_added": len(rows),
        "budget": args.budget,
        "budget_multiplier": args.budget_multiplier,
        "attention_candidates": args.attention_candidates,
        "coalition_batch_size": args.coalition_batch_size,
        "total_unique_coalition_evaluations": sum(
            int((row.get("diagnostics") or {}).get("unique_coalition_evaluations", 0))
            for row in all_rows
        ),
        "total_seconds": sum(float(row["elapsed_seconds"]) for row in all_rows),
        "mean_seconds": statistics.fmean(
            float(row["elapsed_seconds"]) for row in all_rows
        ) if all_rows else None,
    }
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    print(f"[token_leverage_shap summary] {rendered}")
    if args.summary_out:
        Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_out).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
