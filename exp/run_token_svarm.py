"""Estimate retrieved-token Shapley values with the SVARM sampler.

SVARM (Kolpaczki et al., AAAI 2024) estimates every player's Shapley
value from shared coalition evaluations.  One sampled coalition therefore
updates many token estimates at once, unlike permutation sampling, which
requires a complete path of ``n + 1`` coalitions per permutation.

The cooperative game is identical to ``run_token_shapley.py``: a coalition
keeps the embeddings of its retrieved-token members, all other retrieved
tokens are set to the zero reference, and utility is the mean teacher-forced
log-likelihood of the frozen clean response.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.io import load_records, record_id, retrieved_contexts
from causalityrag.reader import reader_completion_text
from causalityrag.token_units import units_from_context_row
from exp.arc_jsd import ArcJsdModel
from exp.run_gradient_attribution_baselines import ranked_candidate
from exp.run_token_shapley import objective_batch


def _coalition_players(
    units: list[dict], positions_by_unit: list[list[int]]
) -> tuple[list[list[int]], list[int], list[int]]:
    """Merge surface units that overlap the same model subtoken."""

    parents = list(range(len(units)))

    def find(owner: int) -> int:
        while parents[owner] != owner:
            parents[owner] = parents[parents[owner]]
            owner = parents[owner]
        return owner

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    first_owner_by_position: dict[int, int] = {}
    for owner, positions in enumerate(positions_by_unit):
        for position in positions:
            previous = first_owner_by_position.setdefault(position, owner)
            union(owner, previous)

    members_by_root: dict[int, list[int]] = {}
    for owner in range(len(units)):
        members_by_root.setdefault(find(owner), []).append(owner)
    player_members = sorted(members_by_root.values(), key=lambda group: group[0])
    unit_to_player = {
        owner: player
        for player, members in enumerate(player_members)
        for owner in members
    }
    player_by_position: dict[int, int] = {}
    for owner, positions in enumerate(positions_by_unit):
        for position in positions:
            player_by_position[position] = unit_to_player[owner]
    context_positions = sorted(player_by_position)
    position_owners = [player_by_position[position] for position in context_positions]
    return player_members, context_positions, position_owners


def _mask(n_players: int, members) -> list[bool]:
    coalition = [False] * n_players
    for member in members:
        coalition[member] = True
    return coalition


def _sample_svarm_positive(rng: random.Random, n_players: int) -> list[bool]:
    """Sample A+ from P+(A) proportional to 1 / |A|."""

    harmonic = [1.0 / size for size in range(1, n_players + 1)]
    size = rng.choices(range(1, n_players + 1), weights=harmonic, k=1)[0]
    return _mask(n_players, rng.sample(range(n_players), size))


def _sample_svarm_negative(rng: random.Random, n_players: int) -> list[bool]:
    """Sample A- from P-(A) proportional to 1 / (n - |A|)."""

    inverse_complement = [1.0 / (n_players - size) for size in range(n_players)]
    size = rng.choices(range(n_players), weights=inverse_complement, k=1)[0]
    return _mask(n_players, rng.sample(range(n_players), size))


def _svarm_draws(
    n_players: int, budget: int, seed: int
) -> tuple[list[list[bool]], list[tuple[str, list[int]]]]:
    """Return the SVARM warm-up and shared-sample update instructions."""

    if n_players == 0:
        return [], []
    if n_players == 1:
        return [[False], [True]], [("negative", [0]), ("positive", [0])]
    minimum_budget = 2 * n_players
    if budget < minimum_budget:
        raise ValueError(
            f"SVARM needs at least 2n={minimum_budget} coalition draws; got {budget}"
        )

    rng = random.Random(seed)
    masks: list[list[bool]] = []
    updates: list[tuple[str, list[int]]] = []

    # Warm-up: obtain one correctly distributed positive and negative sample
    # for every player. Coalition size is uniform under the Shapley weights.
    for player in range(n_players):
        others = [candidate for candidate in range(n_players) if candidate != player]
        size = rng.randrange(n_players)
        subset = rng.sample(others, size)
        negative = _mask(n_players, subset)
        positive = negative.copy()
        positive[player] = True
        masks.extend([positive, negative])
        updates.extend([("positive", [player]), ("negative", [player])])

    while len(masks) + 2 <= budget:
        positive = _sample_svarm_positive(rng, n_players)
        negative = _sample_svarm_negative(rng, n_players)
        masks.extend([positive, negative])
        updates.extend([
            ("positive", [i for i, present in enumerate(positive) if present]),
            ("negative", [i for i, present in enumerate(negative) if not present]),
        ])

    if len(masks) < budget:
        positive = _sample_svarm_positive(rng, n_players)
        masks.append(positive)
        updates.append(
            ("positive", [i for i, present in enumerate(positive) if present])
        )
    return masks, updates


def _evaluate_cached_masks(
    model: ArcJsdModel,
    base_embeddings,
    prompt_length: int,
    response_ids: list[int],
    context_positions: list[int],
    position_owners: list[int],
    masks: list[list[bool]],
    batch_size: int,
) -> tuple[list[float], int]:
    """Evaluate unique deterministic coalitions and restore draw order."""

    unique_masks: list[list[bool]] = []
    index_by_key: dict[bytes, int] = {}
    draw_indices: list[int] = []
    for coalition in masks:
        key = bytes(coalition)
        if key not in index_by_key:
            index_by_key[key] = len(unique_masks)
            unique_masks.append(coalition)
        draw_indices.append(index_by_key[key])

    unique_values: list[float] = []
    for start in range(0, len(unique_masks), batch_size):
        unique_values.extend(
            objective_batch(
                model,
                base_embeddings,
                prompt_length,
                response_ids,
                context_positions,
                position_owners,
                unique_masks[start : start + batch_size],
            )
        )
    return [unique_values[index] for index in draw_indices], len(unique_masks)


def sampled_token_svarm_scores(
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
    """Estimate one Shapley value per retrieved-token unit using SVARM."""

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
            "method": "token_svarm",
            "estimator": "SVARM",
            "coalition_draws": 0,
            "unique_coalition_evaluations": 0,
            "coalition_players": 0,
        }

    effective_budget = 2 if n_players == 1 else max(budget, 2 * n_players)
    masks, updates = _svarm_draws(n_players, effective_budget, seed)
    values, unique_evaluations = _evaluate_cached_masks(
        model,
        base,
        len(prompt_ids),
        response_ids,
        context_positions,
        position_owners,
        masks,
        batch_size,
    )

    positive_sum = [0.0] * n_players
    negative_sum = [0.0] * n_players
    positive_count = [0] * n_players
    negative_count = [0] * n_players
    for value, (side, players) in zip(values, updates):
        if side == "positive":
            for player in players:
                positive_sum[player] += value
                positive_count[player] += 1
        else:
            for player in players:
                negative_sum[player] += value
                negative_count[player] += 1
    raw_player_scores = [
        positive_sum[player] / positive_count[player]
        - negative_sum[player] / negative_count[player]
        for player in range(n_players)
    ]

    # Enforce the Shapley efficiency identity. The correction is identical for
    # every player and therefore does not alter their ranking.
    endpoints = [[False] * n_players, [True] * n_players]
    endpoint_values, endpoint_evaluations = _evaluate_cached_masks(
        model,
        base,
        len(prompt_ids),
        response_ids,
        context_positions,
        position_owners,
        endpoints,
        batch_size,
    )
    target_sum = endpoint_values[1] - endpoint_values[0]
    efficiency_error_before = sum(raw_player_scores) - target_sum
    correction = -efficiency_error_before / n_players
    player_scores = [score + correction for score in raw_player_scores]

    scores = [0.0] * len(units)
    for player, members in enumerate(player_members):
        share = player_scores[player] / len(members)
        for owner in members:
            scores[owner] = share
    return scores, {
        "method": "token_svarm",
        "estimator": "SVARM",
        "reference": "Kolpaczki_et_al_AAAI_2024",
        "coalition_draw_budget": effective_budget,
        "unique_coalition_evaluations": unique_evaluations,
        "endpoint_evaluations": endpoint_evaluations,
        "coalition_batch_size": batch_size,
        "context_baseline": "zero_embedding_only_context_positions",
        "prompt_tokens": len(prompt_ids),
        "response_tokens": len(response_ids),
        "context_units": len(units),
        "coalition_players": n_players,
        "overlap_groups": sum(len(members) > 1 for members in player_members),
        "model_subtokens": len(context_positions),
        "minimum_svarm_budget": 2 * n_players,
        "mean_positive_samples_per_player": statistics.fmean(positive_count),
        "mean_negative_samples_per_player": statistics.fmean(negative_count),
        "efficiency_error_before_correction": efficiency_error_before,
        "efficiency_error_after_correction": sum(player_scores) - target_sum,
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
    parser.add_argument("--budget", type=int, default=512)
    parser.add_argument("--coalition-batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.start, args.n) < 0:
        parser.error("start and n must be non-negative")
    if min(args.k, args.top_k, args.budget, args.coalition_batch_size) <= 0:
        parser.error("k, top-k, budget, and coalition-batch-size must be positive")

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
                "baseline": "token_svarm",
                "method": "token_svarm",
                "top_k": args.top_k,
                "status": "ok",
                "objective": "mean_teacher_forced_clean_answer_log_likelihood",
                "attribution_backend": "matching_hf_reader",
                "selection_domain": "all_non_punctuation_context_tokens",
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
                scores, diagnostics = sampled_token_svarm_scores(
                    model,
                    str(record.get("question", "")),
                    contexts,
                    units,
                    trajectory.response_ids,
                    budget=args.budget,
                    batch_size=args.coalition_batch_size,
                    seed=args.seed + index,
                )
                score_map = {
                    str(unit["unit_id"]): float(score)
                    for unit, score in zip(units, scores)
                }
                candidate = ranked_candidate(units, score_map, top_k=args.top_k)
                row.update({
                    "status": "ok" if candidate else "no_positive_editable_token",
                    "clean_answer": clean_answer,
                    "token_scores": score_map,
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
                f"[token_svarm] {len(rows)}/{len(selected_records)} "
                f"status={row['status']} seconds={row['elapsed_seconds']}",
                flush=True,
            )

    all_rows = load_records(args.out)
    summary = {
        "baseline": "token_svarm",
        "method": "token_svarm",
        "queries": len(all_rows),
        "queries_added": len(rows),
        "requested_budget": args.budget,
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
    print(f"[token_svarm summary] {rendered}")
    if args.summary_out:
        Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_out).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
