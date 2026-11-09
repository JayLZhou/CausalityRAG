"""Estimate retrieved-token Shapley values by sampled permutations.

Each retrieved token occurrence is a player.  A coalition keeps the model
embeddings of its members and replaces all other retrieved-token embeddings
with the zero reference used by the Integrated Gradients baseline.  Utility is
the mean teacher-forced log-likelihood of the frozen clean response.  Walking
from the empty coalition to the full coalition along a random permutation
therefore yields one marginal contribution for every token.

This implementation is intentionally faithful rather than cheap: one sampled
permutation evaluates ``n + 1`` coalitions for a query with ``n`` editable
tokens.  The command reports this cost explicitly so that a smoke test can
determine whether the baseline is practical at Table-3 scale.
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


def objective_batch(
    model: ArcJsdModel,
    base_embeddings,
    prompt_length: int,
    response_ids: list[int],
    context_positions: list[int],
    position_owners,
    coalition_masks: list[list[bool]],
) -> list[float]:
    """Return clean-response log-likelihood for several token coalitions."""

    torch = model.torch
    if not coalition_masks:
        return []
    batch_size = len(coalition_masks)
    embeddings = base_embeddings.expand(batch_size, -1, -1).clone()
    gates = torch.tensor(
        coalition_masks,
        dtype=embeddings.dtype,
        device=embeddings.device,
    )
    embeddings[:, context_positions] *= gates[:, position_owners, None]
    attention_mask = torch.ones(
        embeddings.shape[:2],
        dtype=torch.long,
        device=embeddings.device,
    )
    with torch.inference_mode():
        hidden = model.model.model(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        response_positions = (
            torch.arange(len(response_ids), device=embeddings.device)
            + prompt_length
            - 1
        )
        logits = model.model.lm_head(hidden[:, response_positions]).float()
        log_probs = torch.log_softmax(logits, dim=-1)
        targets = torch.tensor(
            response_ids,
            dtype=torch.long,
            device=embeddings.device,
        )
        rows = torch.arange(batch_size, device=embeddings.device)[:, None]
        columns = torch.arange(len(targets), device=embeddings.device)[None, :]
        values = log_probs[rows, columns, targets[None, :]].mean(dim=1)
    return [float(value) for value in values.cpu().tolist()]


def sampled_token_shapley_scores(
    model: ArcJsdModel,
    question: str,
    contexts: list[dict],
    units: list[dict],
    response_ids: list[int],
    *,
    permutations: int,
    batch_size: int,
    seed: int,
) -> tuple[list[float], dict]:
    """Estimate one Shapley value per unit using random permutation paths."""

    torch = model.torch
    prompt_ids, positions_by_unit = model._prompt_context_token_positions(
        question,
        contexts,
        units,
    )
    sequence = prompt_ids + list(response_ids)
    input_ids = torch.tensor(
        [sequence],
        dtype=torch.long,
        device=model.device,
    )
    base = model.model.get_input_embeddings()(input_ids).detach()

    # One model subtoken may cover multiple surface-token units. Those units
    # cannot be switched independently in embedding space, so merge each
    # connected overlap component into one coalition player and split the
    # resulting group value equally among its surface tokens.
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

    rng = random.Random(seed)
    n_units = len(units)
    n_players = len(player_members)
    totals = [0.0] * n_players
    evaluations = 0
    efficiency_errors: list[float] = []
    for _ in range(permutations):
        order = list(range(n_players))
        rng.shuffle(order)
        active = [False] * n_players
        masks = [active.copy()]
        for owner in order:
            active[owner] = True
            masks.append(active.copy())

        utilities: list[float] = []
        for start in range(0, len(masks), batch_size):
            utilities.extend(
                objective_batch(
                    model,
                    base,
                    len(prompt_ids),
                    response_ids,
                    context_positions,
                    position_owners,
                    masks[start : start + batch_size],
                )
            )
        evaluations += len(utilities)
        for step, owner in enumerate(order):
            totals[owner] += utilities[step + 1] - utilities[step]
        efficiency_errors.append(
            abs(sum(utilities[i + 1] - utilities[i] for i in range(n_players))
                - (utilities[-1] - utilities[0]))
        )

    player_scores = [value / permutations for value in totals]
    scores = [0.0] * n_units
    for player, members in enumerate(player_members):
        share = player_scores[player] / len(members)
        for owner in members:
            scores[owner] = share
    return scores, {
        "method": "token_shapley",
        "estimator": "sampled_permutations",
        "permutations": permutations,
        "coalition_evaluations": evaluations,
        "coalitions_per_permutation": n_players + 1,
        "coalition_batch_size": batch_size,
        "context_baseline": "zero_embedding_only_context_positions",
        "prompt_tokens": len(prompt_ids),
        "response_tokens": len(response_ids),
        "context_units": n_units,
        "coalition_players": n_players,
        "overlap_groups": sum(len(members) > 1 for members in player_members),
        "model_subtokens": len(context_positions),
        "max_efficiency_error": max(efficiency_errors, default=0.0),
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
    parser.add_argument("--permutations", type=int, default=4)
    parser.add_argument("--coalition-batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.start, args.n) < 0:
        parser.error("start and n must be non-negative")
    if min(args.k, args.top_k, args.permutations, args.coalition_batch_size) <= 0:
        parser.error("k, top-k, permutations, and coalition-batch-size must be positive")

    records = load_records(args.input)
    references = {record_id(row): row for row in load_records(args.clean_reference)}
    units_by_id = {
        record_id(row): row for row in load_records(args.context_units)
    }
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
                "baseline": "token_shapley",
                "method": "token_shapley",
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
                    reader_mode=os.environ.get(
                        "CAUSALITYRAG_READER_MODE", "short_answer"
                    ),
                    compact_json=True,
                )
                trajectory = model.trajectory_for_response(
                    str(record.get("question", "")),
                    contexts,
                    response_text,
                )
                scores, diagnostics = sampled_token_shapley_scores(
                    model,
                    str(record.get("question", "")),
                    contexts,
                    units,
                    trajectory.response_ids,
                    permutations=args.permutations,
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
                f"[token_shapley] {len(rows)}/{len(selected_records)} "
                f"status={row['status']} seconds={row['elapsed_seconds']}",
                flush=True,
            )

    all_rows = load_records(args.out)
    summary = {
        "baseline": "token_shapley",
        "method": "token_shapley",
        "queries": len(all_rows),
        "queries_added": len(rows),
        "permutations": args.permutations,
        "coalition_batch_size": args.coalition_batch_size,
        "total_coalition_evaluations": sum(
            int((row.get("diagnostics") or {}).get("coalition_evaluations", 0))
            for row in all_rows
        ),
        "total_seconds": sum(float(row["elapsed_seconds"]) for row in all_rows),
        "mean_seconds": statistics.fmean(
            float(row["elapsed_seconds"]) for row in all_rows
        ) if all_rows else None,
    }
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    print(f"[token_shapley summary] {rendered}")
    if args.summary_out:
        Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_out).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
