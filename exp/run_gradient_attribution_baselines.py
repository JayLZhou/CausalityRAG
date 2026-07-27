"""Rank editable context tokens with gradient attribution baselines.

The two supported methods are deliberately outside ReFlow's contribution
graph: Gradient x Input and Integrated Gradients use the clean-answer
teacher-forced log-likelihood of the matching Hugging Face reader.  The
resulting Top-K selections are later evaluated with the shared vLLM reader
and replacement registry.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.io import load_records, record_id, retrieved_contexts
from causalityrag.token_units import units_from_context_row
from exp.arc_jsd import ArcJsdModel


METHOD_NAMES = {
    "gradient_x_input": "gradient_x_input_top5",
    "integrated_gradients": "integrated_gradients_top5",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--clean-reference", required=True)
    parser.add_argument("--context-units", required=True)
    parser.add_argument("--replacement-registry", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", default="")
    parser.add_argument(
        "--method",
        choices=tuple(METHOD_NAMES),
        required=True,
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--n", type=int, default=0)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--ig-steps", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    args = parser.parse_args()
    if args.start < 0 or args.n < 0:
        parser.error("start and n must be non-negative")
    if args.k <= 0 or args.top_k <= 0:
        parser.error("k and top-k must be positive")
    if args.method == "integrated_gradients" and args.ig_steps <= 0:
        parser.error("ig-steps must be positive")

    records = load_records(args.input)
    references = {record_id(row): row for row in load_records(args.clean_reference)}
    units_by_id = {
        record_id(row): row for row in load_records(args.context_units)
    }
    registries = {
        record_id(row): row
        for row in load_records(args.replacement_registry)
    }
    end = len(records) if args.n == 0 else min(len(records), args.start + args.n)
    selected_records = list(enumerate(records))[args.start:end]
    model = ArcJsdModel(
        args.model_path,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )

    rows = []
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as output:
        for index, record in selected_records:
            started = time.monotonic()
            identifier = record_id(record)
            reference = references.get(identifier)
            units_row = units_by_id.get(identifier)
            registry = registries.get(identifier)
            if reference is None or units_row is None or registry is None:
                raise ValueError(f"missing frozen artifact for {identifier}")
            contexts = retrieved_contexts(record)[: args.k]
            units = units_from_context_row(record, units_row, k=args.k)
            valid_ids = {
                str(unit_id)
                for unit_id, replacement in registry.get(
                    "replacements", {}
                ).items()
                if isinstance(replacement, dict) and replacement.get("ok")
            }
            clean_answer = str(reference.get("clean_answer", "")).strip()
            row = {
                "index": index,
                "id": identifier,
                "question": str(record.get("question", "")),
                "baseline": METHOD_NAMES[args.method],
                "method": args.method,
                "top_k": args.top_k,
                "status": "ok",
                "objective": "mean_teacher_forced_clean_answer_log_likelihood",
                "attribution_backend": "matching_hf_reader",
                "replacement_registry": args.replacement_registry,
            }
            if not clean_answer:
                row.update({
                    "status": "reader_abstention_empty_answer",
                    "token_scores": {},
                    "selection_candidate": None,
                    "candidates": [],
                })
            else:
                response_text = json.dumps(
                    {"answer": clean_answer},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                trajectory = model.trajectory_for_response(
                    str(record.get("question", "")),
                    contexts,
                    response_text,
                )
                if args.method == "gradient_x_input":
                    scores, diagnostics = gradient_x_input_scores(
                        model,
                        str(record.get("question", "")),
                        contexts,
                        units,
                        trajectory.response_ids,
                    )
                else:
                    scores, diagnostics = integrated_gradients_scores(
                        model,
                        str(record.get("question", "")),
                        contexts,
                        units,
                        trajectory.response_ids,
                        steps=args.ig_steps,
                    )
                score_map = {
                    str(unit["unit_id"]): float(score)
                    for unit, score in zip(units, scores)
                }
                candidate = ranked_candidate(
                    units,
                    score_map,
                    valid_ids=valid_ids,
                    top_k=args.top_k,
                )
                if candidate is None:
                    status = "no_positive_editable_token"
                elif candidate["n_selected"] < args.top_k:
                    status = "insufficient_editable_tokens"
                else:
                    status = "ok"
                row.update({
                    "status": status,
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
                f"[{args.method}] {len(rows)}/{len(selected_records)} "
                f"status={row['status']} "
                f"selected={len((row.get('selection_candidate') or {}).get('selected_ids', []))} "
                f"seconds={row['elapsed_seconds']}",
                flush=True,
            )

    summary = summarize(rows, args)
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    print(f"[{args.method} summary] {rendered}")
    if args.summary_out:
        os.makedirs(
            os.path.dirname(os.path.abspath(args.summary_out)),
            exist_ok=True,
        )
        Path(args.summary_out).write_text(rendered + "\n", encoding="utf-8")


def teacher_forced_scores(
    model: ArcJsdModel,
    question: str,
    contexts: list[dict],
    units: list[dict],
    response_ids: list[int],
    *,
    context_embedding_builder,
) -> tuple[list[float], dict]:
    """Return per-unit attribution scores for one clean trajectory."""

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
    attributions, integration_diagnostics = context_embedding_builder(
        model,
        base,
        prompt_ids,
        response_ids,
        positions_by_unit,
    )
    scores = []
    for positions in positions_by_unit:
        values = attributions[0, positions].abs().sum(dim=-1)
        scores.append(float(values.mean().item()))
    diagnostics = {
        "prompt_tokens": len(prompt_ids),
        "response_tokens": len(response_ids),
        "context_units": len(units),
        "model_subtokens": sum(len(positions) for positions in positions_by_unit),
        **integration_diagnostics,
    }
    return scores, diagnostics


def _objective_and_gradient(
    model: ArcJsdModel,
    embeddings,
    prompt_length: int,
    response_ids: list[int],
):
    torch = model.torch
    attention_mask = torch.ones(
        embeddings.shape[:2],
        dtype=torch.long,
        device=embeddings.device,
    )
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
    objective = log_probs[0, torch.arange(len(targets)), targets].mean()
    gradient = torch.autograd.grad(
        objective,
        embeddings,
        retain_graph=False,
        create_graph=False,
    )[0]
    return gradient


def _gradient_x_input_builder(
    model,
    base,
    prompt_ids,
    response_ids,
    positions_by_unit,
):
    embeddings = base.clone().requires_grad_(True)
    gradient = _objective_and_gradient(
        model,
        embeddings,
        len(prompt_ids),
        response_ids,
    )
    return base * gradient, {
        "method": "gradient_x_input",
        "gradient_passes": 1,
        "context_baseline": "none",
    }


def _integrated_gradients_builder(
    steps: int,
):
    def builder(model, base, prompt_ids, response_ids, positions_by_unit):
        torch = model.torch
        context_positions = sorted({
            position
            for positions in positions_by_unit
            for position in positions
        })
        gradient_sum = torch.zeros_like(base)
        for step in range(1, steps + 1):
            alpha = step / steps
            embeddings = base.clone()
            embeddings[:, context_positions] *= alpha
            embeddings.requires_grad_(True)
            gradient = _objective_and_gradient(
                model,
                embeddings,
                len(prompt_ids),
                response_ids,
            )
            gradient_sum += gradient.detach()
        integrated = base * gradient_sum / steps
        return integrated, {
            "method": "integrated_gradients",
            "gradient_passes": steps,
            "ig_steps": steps,
            "context_baseline": "zero_embedding_only_context_positions",
            "riemann_rule": "right_endpoint",
        }

    return builder


def gradient_x_input_scores(
    model,
    question,
    contexts,
    units,
    response_ids,
):
    return teacher_forced_scores(
        model,
        question,
        contexts,
        units,
        response_ids,
        context_embedding_builder=_gradient_x_input_builder,
    )


def integrated_gradients_scores(
    model,
    question,
    contexts,
    units,
    response_ids,
    *,
    steps=16,
):
    return teacher_forced_scores(
        model,
        question,
        contexts,
        units,
        response_ids,
        context_embedding_builder=_integrated_gradients_builder(steps),
    )


def ranked_candidate(
    units: list[dict],
    scores: dict[str, float],
    *,
    valid_ids: set[str],
    top_k: int,
) -> dict | None:
    ranked = sorted(
        (
            unit
            for unit in units
            if (
                str(unit["unit_id"]) in valid_ids
                and float(scores.get(str(unit["unit_id"]), 0.0)) > 0
            )
        ),
        key=lambda unit: (
            -float(scores[str(unit["unit_id"])]),
            str(unit["unit_id"]),
        ),
    )
    selected = ranked[:top_k]
    if not selected:
        return None
    return {
        "status": "fixed_topk",
        "selected_ids": [str(unit["unit_id"]) for unit in selected],
        "selected_tokens": [str(unit.get("text", "")) for unit in selected],
        "selected_scores": [
            float(scores[str(unit["unit_id"])]) for unit in selected
        ],
        "n_selected": len(selected),
        "score": "embedding_attribution",
    }


def summarize(rows: list[dict], args) -> dict:
    counts = [
        int((row.get("selection_candidate") or {}).get("n_selected", 0))
        for row in rows
    ]
    histogram: dict[str, int] = {}
    for row in rows:
        status = str(row["status"])
        histogram[status] = histogram.get(status, 0) + 1
    return {
        "baseline": METHOD_NAMES[args.method],
        "method": args.method,
        "queries": len(rows),
        "status_histogram": histogram,
        "top_k": args.top_k,
        "ig_steps": args.ig_steps if args.method == "integrated_gradients" else None,
        "queries_with_any_candidate": sum(count > 0 for count in counts),
        "queries_with_exact_topk": sum(count == args.top_k for count in counts),
        "mean_selected_tokens": statistics.fmean(counts) if counts else None,
        "total_seconds": sum(float(row["elapsed_seconds"]) for row in rows),
        "mean_seconds": (
            statistics.fmean(float(row["elapsed_seconds"]) for row in rows)
            if rows
            else None
        ),
    }


if __name__ == "__main__":
    main()
