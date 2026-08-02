"""Produce ARC-JSD sentence-to-token attribution rankings."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.io import load_records, record_id, retrieved_contexts
from causalityrag.linguistics import SpacyAnnotationClient
from exp.arc_jsd import (
    ArcJsdModel,
    ablate_context_sentence,
    context_sentence_units,
    lift_sentence_features_to_tokens,
    vector_supermodular_graph,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--clean-reference", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--feature-top-k", type=int, default=64)
    parser.add_argument("--routing-last-layers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--spacy-base-url", default="")
    args = parser.parse_args()

    records = load_records(args.input)[args.start : args.start + args.n]
    references = {
        record_id(row): row for row in load_records(args.clean_reference)
    }
    nlp = SpacyAnnotationClient(args.spacy_base_url or None)
    if not nlp.health().get("ok"):
        raise RuntimeError("spaCy annotation service is unhealthy")
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
            reference = references.get(identifier)
            if reference is None:
                raise ValueError(f"missing clean reference for {identifier}")
            clean_answer = str(reference.get("clean_answer", "")).strip()
            question = str(record.get("question", ""))
            contexts = retrieved_contexts(record)[: args.k]
            units, sentences = context_sentence_units(
                record,
                k=args.k,
                nlp=nlp,
            )
            base = {
                "index": index,
                "id": identifier,
                "method": "arc_jsd",
                "clean_answer": clean_answer,
            }
            if not clean_answer:
                row = {
                    **base,
                    "status": "reader_abstention_empty_answer",
                    "token_scores": {},
                    "ranked_ids": [],
                }
            else:
                response_text = json.dumps(
                    {"answer": clean_answer},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                trajectory = model.trajectory_for_response(
                    question,
                    contexts,
                    response_text,
                )
                variants = (
                    ablate_context_sentence(contexts, sentence)
                    for sentence in sentences
                )
                sentence_scores, sentence_features = (
                    model.score_context_variants_with_fisher_features(
                        question,
                        variants,
                        trajectory,
                        batch_size=args.batch_size,
                        top_k=args.feature_top_k,
                    )
                )
                routing = model._arc_jsd_response_to_context_routing(
                    question,
                    contexts,
                    units,
                    trajectory.response_ids,
                    last_layers=args.routing_last_layers,
                )
                token_features, diagnostics = (
                    lift_sentence_features_to_tokens(
                        units,
                        sentences,
                        sentence_scores,
                        sentence_features,
                        routing,
                        response_length=len(trajectory.response_ids),
                    )
                )
                node_weights, _ = vector_supermodular_graph(
                    units,
                    token_features,
                )
                token_scores = {
                    str(unit["unit_id"]): float(
                        node_weights.get(str(unit["unit_id"]), 0.0)
                    )
                    for unit in units
                }
                ranked = sorted(
                    token_scores,
                    key=lambda unit_id: (-token_scores[unit_id], unit_id),
                )
                row = {
                    **base,
                    "status": "ok",
                    "token_scores": token_scores,
                    "ranked_ids": ranked,
                    "n_context_tokens": len(units),
                    "n_sentences": len(sentences),
                    "lift_diagnostics": diagnostics,
                }
            row["elapsed_seconds"] = round(time.monotonic() - started, 3)
            rows.append(row)
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            print(
                f"[arc-jsd] {len(rows)}/{len(records)} "
                f"status={row['status']}",
                flush=True,
            )

    summary = {
        "method": "arc_jsd",
        "queries": len(rows),
        "ranked_queries": sum(row["status"] == "ok" for row in rows),
        "total_seconds": sum(
            float(row["elapsed_seconds"]) for row in rows
        ),
        "out": os.path.abspath(args.out),
    }
    with open(args.summary_out, "w", encoding="utf-8") as output:
        json.dump(summary, output, indent=2, ensure_ascii=False)
        output.write("\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
