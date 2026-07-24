"""Score context tokens with the MIRAGE model-internals baseline."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from itertools import islice

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.io import iter_records, load_records, record_id, retrieved_contexts
from causalityrag.token_units import units_from_context_row
from exp.arc_jsd import ArcJsdModel, CleanTrajectory
from exp.mirage import MirageScorer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--clean-reference", required=True)
    parser.add_argument("--context-units", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", default="")
    parser.add_argument(
        "--model-path",
        default="/data1/yujia/models/Qwen2.5-7B-Instruct",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--top-tokens", type=int, default=5)
    parser.add_argument("--cti-std-threshold", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument(
        "--no-gradient-checkpointing",
        action="store_true",
    )
    args = parser.parse_args()
    if args.start < 0 or args.n <= 0 or args.k <= 0 or args.top_tokens <= 0:
        parser.error("start must be non-negative and counts must be positive")

    references = {
        str(row["id"]): row
        for row in load_records(args.clean_reference)
    }
    units_by_id = {
        str(row["id"]): row
        for row in load_records(args.context_units)
    }
    model = ArcJsdModel(
        args.model_path,
        device=args.device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    scorer = MirageScorer(
        model,
        cti_std_threshold=args.cti_std_threshold,
        gradient_checkpointing=not args.no_gradient_checkpointing,
    )
    records = islice(
        iter_records(args.input),
        args.start,
        args.start + args.n,
    )
    rows = []
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as output:
        for offset, record in enumerate(records):
            started = time.monotonic()
            index = args.start + offset
            identifier = record_id(record)
            reference = references.get(identifier)
            units_row = units_by_id.get(identifier)
            if reference is None or units_row is None:
                raise ValueError(f"missing frozen artifact for {identifier}")
            clean_answer = str(reference.get("clean_answer", "")).strip()
            if not clean_answer:
                row = {
                    "index": index,
                    "id": identifier,
                    "question": str(record.get("question", "")),
                    "status": "reader_abstention_empty_answer",
                    "method": "mirage_cti_contrastive_gradient",
                    "token_scores": {},
                    "selection_candidate": None,
                    "candidates": [],
                    "elapsed_seconds": round(
                        time.monotonic() - started,
                        6,
                    ),
                }
            else:
                question = str(record.get("question", ""))
                contexts = retrieved_contexts(record)[: args.k]
                units = units_from_context_row(
                    record,
                    units_row,
                    k=args.k,
                )
                trajectory = frozen_clean_trajectory(
                    model,
                    question,
                    contexts,
                    clean_answer,
                )
                token_scores, diagnostics = scorer.score(
                    question,
                    contexts,
                    units,
                    trajectory,
                )
                candidate = ranked_candidate(
                    units,
                    token_scores,
                    top_tokens=args.top_tokens,
                )
                row = {
                    "index": index,
                    "id": identifier,
                    "question": question,
                    "status": (
                        "ok"
                        if candidate is not None
                        else "no_positive_mirage_token"
                    ),
                    "method": "mirage_cti_contrastive_gradient",
                    "objective": (
                        "fixed_topk_by_mirage_unary_saliency"
                    ),
                    "trajectory_source": (
                        "frozen_vllm_answer_canonical_json_teacher_forcing"
                    ),
                    "clean_answer": clean_answer,
                    "mirage": diagnostics.to_dict(),
                    "token_scores": token_scores,
                    "selection_candidate": candidate,
                    "candidates": (
                        [candidate]
                        if candidate is not None
                        else []
                    ),
                    "elapsed_seconds": round(
                        time.monotonic() - started,
                        6,
                    ),
                }
            rows.append(row)
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            print(
                f"[mirage-score] {offset + 1}/{args.n} "
                f"status={row['status']} "
                f"selected={len((row.get('selection_candidate') or {}).get('selected_ids', []))} "
                f"seconds={row['elapsed_seconds']}",
                flush=True,
            )

    summary = summarize(rows, args)
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    print("[mirage-score summary]", rendered)
    if args.summary_out:
        with open(args.summary_out, "w", encoding="utf-8") as output:
            output.write(rendered + "\n")


def frozen_clean_trajectory(
    model: ArcJsdModel,
    question: str,
    contexts: list[dict],
    clean_answer: str,
) -> CleanTrajectory:
    """Teacher-force the frozen vLLM answer in the required JSON format."""

    response_text = json.dumps(
        {"answer": clean_answer},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    response_ids = model.tokenizer(
        response_text,
        add_special_tokens=False,
    )["input_ids"]
    if not response_ids:
        raise ValueError("frozen answer produced an empty response trajectory")
    prompt_ids = model._prompt_ids(question, contexts)
    reference = model._log_probs_for_prompt_ids(
        [prompt_ids],
        response_ids,
    )[0].cpu()
    return CleanTrajectory(response_text, response_ids, reference)


def ranked_candidate(
    units: list[dict],
    token_scores: dict[str, float],
    *,
    top_tokens: int,
    allowed_ids: set[str] | None = None,
) -> dict | None:
    ranked = sorted(
        (
            unit
            for unit in units
            if (
                allowed_ids is None
                or str(unit["unit_id"]) in allowed_ids
            )
            and float(
                token_scores.get(str(unit["unit_id"]), 0.0)
            ) > 0
        ),
        key=lambda unit: (
            -float(token_scores[str(unit["unit_id"])]),
            str(unit["unit_id"]),
        ),
    )
    selected = ranked[:top_tokens]
    if not selected:
        return None
    return {
        "status": "fixed_topk",
        "selected_ids": [
            str(unit["unit_id"])
            for unit in selected
        ],
        "selected_tokens": [
            str(unit.get("text", ""))
            for unit in selected
        ],
        "selected_scores": [
            float(token_scores[str(unit["unit_id"])])
            for unit in selected
        ],
        "n_selected": len(selected),
        "score": "mirage_cti_contrastive_embedding_gradient",
    }


def summarize(rows: list[dict], args) -> dict:
    elapsed = [float(row["elapsed_seconds"]) for row in rows]
    histogram: dict[str, int] = {}
    for row in rows:
        status = str(row["status"])
        histogram[status] = histogram.get(status, 0) + 1
    return {
        "method": "mirage_cti_contrastive_gradient",
        "queries": len(rows),
        "status_histogram": histogram,
        "top_tokens": args.top_tokens,
        "cti_std_threshold": args.cti_std_threshold,
        "trajectory_source": (
            "frozen_vllm_answer_canonical_json_teacher_forcing"
        ),
        "attn_implementation": args.attn_implementation,
        "gradient_checkpointing": not args.no_gradient_checkpointing,
        "total_seconds": sum(elapsed),
        "mean_seconds": (
            statistics.fmean(elapsed)
            if elapsed
            else None
        ),
    }


if __name__ == "__main__":
    main()
