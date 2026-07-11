"""Lift sentence ARC-JSD to a token graph and run one ratio attack."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causalityrag.arc_jsd import (
    ArcJsdModel,
    ablate_context_sentence,
    context_sentence_units,
    lift_sentence_features_to_tokens,
    vector_supermodular_graph,
)
from causalityrag.graph_cut import solve_source_target_supermodular_ratio
from causalityrag.io import iter_records, record_id, retrieved_contexts
from causalityrag.linguistics import SpacyAnnotationClient
from causalityrag.reader import ReaderClient, answers_match, parse_json_object
from causalityrag.replacement import GenericReplacementClient, generate_valid_replacement
from causalityrag.revision import apply_token_replacements
from causalityrag.rules import TypedRuleLibrary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model-path", default="/data1/yujia/models/Qwen2.5-7B-Instruct")
    parser.add_argument("--cf-pools", required=True)
    parser.add_argument("--type-rules", default="")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--feature-top-k", type=int, default=64)
    parser.add_argument("--routing-last-layers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--spacy-base-url", default=os.environ.get("CAUSALITYRAG_SPACY_BASE_URL", "http://127.0.0.1:8021"))
    args = parser.parse_args()

    records = list(iter_records(args.input, args.start + args.n))[args.start:]
    nlp = SpacyAnnotationClient(args.spacy_base_url)
    health = nlp.health()
    if not health.get("ok"):
        raise RuntimeError(f"spaCy annotation service is unhealthy: {health}")
    model = ArcJsdModel(
        args.model_path,
        device=args.device,
        dtype=args.dtype,
        attn_implementation="eager",
    )
    reader = ReaderClient()
    library = TypedRuleLibrary.from_files(args.cf_pools, args.type_rules or None)
    generic_editor = GenericReplacementClient()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    with open(args.out, "w", encoding="utf-8") as output:
        for row_index, record in enumerate(records, 1):
            started = time.monotonic()
            question = str(record.get("question", ""))
            contexts = retrieved_contexts(record)[:args.k]
            units, sentences = context_sentence_units(record, k=args.k, nlp=nlp)
            trajectory = model.clean_trajectory(question, contexts)
            parsed = parse_json_object(trajectory.response_text)
            clean_answer = (
                str(parsed.get("answer", "")).strip()
                if isinstance(parsed, dict)
                else trajectory.response_text
            )

            sentence_variants = (
                ablate_context_sentence(contexts, sentence)
                for sentence in sentences
            )
            sentence_scores, sentence_features = model.score_context_variants_with_fisher_features(
                question,
                sentence_variants,
                trajectory,
                batch_size=args.batch_size,
                top_k=args.feature_top_k,
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
            node_weights, interactions = vector_supermodular_graph(units, token_features)
            by_id = {str(unit["unit_id"]): unit for unit in units}
            candidate, selected, replacements, rejected_units = solve_with_valid_replacements(
                units,
                node_weights,
                interactions,
                contexts,
                library,
                generic_editor,
                nlp,
            )
            revision = apply_token_replacements(record, selected, replacements, k=args.k)
            edited_answer = (
                reader.answer(question, revision["edited_contexts"])
                if selected else clean_answer
            )
            result = {
                "id": record_id(record),
                "question": question,
                "gold_answer": str(record.get("answer", "")),
                "clean_response": trajectory.response_text,
                "clean_answer": clean_answer,
                "edited_answer": edited_answer,
                "answer_changed": not answers_match(clean_answer, edited_answer),
                "method": "arc_jsd_conserving_sentence_to_token_lift_ratio",
                "lift_mode": "signed_shift_positive_pair_envelope",
                "objective": "max_nonempty_S F(S)/|S|",
                "n_context_tokens": len(units),
                "n_sentences": len(sentences),
                "n_pairs": len(units) * (len(units) - 1) // 2,
                "n_positive_pair_edges": len(interactions),
                "lift_diagnostics": lift_diagnostics,
                "sentence_scores": {
                    str(sentence["sentence_id"]): score
                    for sentence, score in zip(sentences, sentence_scores)
                },
                "candidate": candidate,
                "selected_units": selected,
                "n_selected": len(selected),
                "rejected_uneditable_units": rejected_units,
                "edits": revision["edits"],
                "n_edits": revision["n_edits"],
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            output.flush()
            print(
                f"[sentence-token-lift] {row_index}/{len(records)} "
                f"sentences={len(sentences)} tokens={len(units)} "
                f"pairs={result['n_pairs']} selected={len(selected)} "
                f"flip={result['answer_changed']} seconds={result['elapsed_seconds']}",
                flush=True,
            )


def solve_with_valid_replacements(
    units: list[dict],
    node_weights: dict[str, float],
    interactions: dict[tuple[str, str], float],
    contexts: list[dict],
    library: TypedRuleLibrary,
    generic_editor: GenericReplacementClient,
    nlp,
) -> tuple[dict, list[dict], dict[str, dict], list[dict]]:
    """Remove contextually invalid units and re-solve the exact ratio."""

    by_id = {str(unit["unit_id"]): unit for unit in units}
    remaining = set(by_id)
    rejected_units = []
    while remaining:
        candidate_units = [by_id[unit_id] for unit_id in remaining]
        candidate = solve_source_target_supermodular_ratio(
            candidate_units,
            {unit_id: weight for unit_id, weight in node_weights.items() if unit_id in remaining},
            {
                edge: weight for edge, weight in interactions.items()
                if edge[0] in remaining and edge[1] in remaining
            },
            {},
        )
        selected = [by_id[unit_id] for unit_id in candidate["selected_ids"]]
        if not selected:
            return candidate, [], {}, rejected_units
        replacements, rejected = build_selected_replacements(
            selected, contexts, library, generic_editor, nlp
        )
        if not rejected:
            return candidate, selected, replacements, rejected_units
        rejected_units.extend(rejected)
        remaining -= {str(unit["unit_id"]) for unit in rejected}
    return {"status": "no_valid_replacement_set", "selected_ids": []}, [], {}, rejected_units


def build_selected_replacements(
    selected: list[dict],
    contexts: list[dict],
    library: TypedRuleLibrary,
    generic_editor: GenericReplacementClient,
    nlp,
) -> tuple[dict[str, dict], list[dict]]:
    context_by_id = {str(context["chunk_id"]): str(context["text"]) for context in contexts}
    replacements = {}
    rejected = []
    for unit in selected:
        unit_id = str(unit["unit_id"])
        context = context_by_id[str(unit["chunk_id"])]
        replacement = generate_valid_replacement(
            unit, context, library, generic_editor, nlp
        )
        if replacement.get("ok"):
            replacements[unit_id] = replacement
        else:
            rejected.append({
                **unit,
                "replacement_failure": replacement,
            })
    return replacements, rejected


if __name__ == "__main__":
    main()
