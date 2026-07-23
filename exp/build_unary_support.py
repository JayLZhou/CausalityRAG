"""Experimental ARC-JSD unary-support builder for historical hybrid runs."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.arc_jsd import (
    ArcJsdModel,
    ablate_context_sentence,
    conservative_sentence_pair_graph,
    context_sentence_units,
    lift_sentence_features_to_tokens,
)
from causalityrag.io import load_records, record_id, retrieved_contexts
from causalityrag.linguistics import SpacyAnnotationClient
from causalityrag.reader import answers_exact_match, parse_json_object
from causalityrag.token_units import units_from_cache_row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary-out", default="")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--units-cache", default="")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--feature-top-k", type=int, default=64)
    parser.add_argument("--routing-last-layers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--spacy-base-url",
        default=os.environ.get(
            "CAUSALITYRAG_SPACY_BASE_URL",
            "http://127.0.0.1:8021",
        ),
    )
    args = parser.parse_args()
    if args.n <= 0 or args.k <= 0 or args.batch_size <= 0:
        raise ValueError("n, k, and batch-size must be positive")
    if args.feature_top_k <= 0 or args.routing_last_layers <= 0:
        raise ValueError("feature-top-k and routing-last-layers must be positive")

    records = load_records(args.input)[args.start:args.start + args.n]
    units_by_id = (
        {
            str(row.get("id")): row
            for row in load_records(args.units_cache)
        }
        if args.units_cache
        else {}
    )
    nlp = None if units_by_id else SpacyAnnotationClient(args.spacy_base_url)
    if nlp is not None and not nlp.health().get("ok"):
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
        for offset, record in enumerate(records, 1):
            started = time.monotonic()
            identifier = record_id(record)
            cached = units_by_id.get(identifier)
            if cached is None:
                units, sentences = context_sentence_units(
                    record,
                    k=args.k,
                    nlp=nlp,
                )
            else:
                units = units_from_cache_row(record, cached, k=args.k)
                sentences = list(cached.get("sentences", []))
                if not sentences:
                    raise ValueError(
                        f"token-units cache row has no sentences for {identifier}"
                    )

            question = str(record.get("question", ""))
            contexts = retrieved_contexts(record)[:args.k]
            trajectory = model.clean_trajectory(question, contexts)
            sentence_scores, sentence_features = (
                model.score_context_variants_with_fisher_features(
                    question,
                    (
                        ablate_context_sentence(contexts, sentence)
                        for sentence in sentences
                    ),
                    trajectory,
                    batch_size=args.batch_size,
                    top_k=args.feature_top_k,
                )
            )
            routing = model.response_to_context_attention_routing(
                question,
                contexts,
                units,
                trajectory.response_ids,
                last_layers=args.routing_last_layers,
            )
            token_features, lift_diagnostics = lift_sentence_features_to_tokens(
                units,
                sentences,
                sentence_scores,
                sentence_features,
                routing,
                response_length=len(trajectory.response_ids),
            )
            unary_scores, _unused_edges, conservation = (
                conservative_sentence_pair_graph(
                    units,
                    sentences,
                    sentence_scores,
                    token_features,
                    {},
                )
            )
            clean_answer = answer_from_response(trajectory.response_text)
            row = {
                "index": args.start + offset - 1,
                "id": identifier,
                "question": question,
                "gold_answer": str(record.get("answer", "")),
                "clean_response": trajectory.response_text,
                "clean_answer": clean_answer,
                "clean_correct": answers_exact_match(
                    clean_answer,
                    str(record.get("answer", "")),
                ),
                "method": "arc_jsd_sentence_ablation_conserved_token_unary",
                "top_k": args.k,
                "n_context_tokens": len(units),
                "n_sentences": len(sentences),
                "unary_scores": unary_scores,
                "sentence_scores": {
                    str(sentence["sentence_id"]): float(score)
                    for sentence, score in zip(sentences, sentence_scores)
                },
                "lift_diagnostics": lift_diagnostics,
                "conservation_diagnostics": conservation,
                "elapsed_seconds": round(time.monotonic() - started, 6),
            }
            rows.append(row)
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            print(
                f"[unary-support] {offset}/{len(records)} "
                f"tokens={len(units)} sentences={len(sentences)} "
                f"seconds={row['elapsed_seconds']}",
                flush=True,
            )

    summary = summarize(rows)
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    print("[unary-support summary]", rendered)
    if args.summary_out:
        with open(args.summary_out, "w", encoding="utf-8") as output:
            output.write(rendered + "\n")


def answer_from_response(response: str) -> str:
    parsed = parse_json_object(response)
    if isinstance(parsed, dict):
        return str(parsed.get("answer", "")).strip()
    return response.strip()


def summarize(rows: list[dict]) -> dict:
    elapsed = [float(row.get("elapsed_seconds", 0.0)) for row in rows]
    return {
        "queries": len(rows),
        "exact_clean_queries": sum(bool(row.get("clean_correct")) for row in rows),
        "context_tokens": sum(int(row.get("n_context_tokens", 0)) for row in rows),
        "sentences": sum(int(row.get("n_sentences", 0)) for row in rows),
        "mean_seconds": statistics.fmean(elapsed) if elapsed else None,
        "method": (
            rows[0].get("method")
            if rows
            else "arc_jsd_sentence_ablation_conserved_token_unary"
        ),
    }


if __name__ == "__main__":
    main()
