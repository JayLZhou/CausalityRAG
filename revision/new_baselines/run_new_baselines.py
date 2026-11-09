"""Token rankings for the baselines added in the revision (R4 D11).

contextcite  ContextCite (Cohen-Wang et al., 2024): sentence sources, random
             ablations (each sentence kept with probability 0.5), Lasso on the
             logit-probability of the frozen clean response; sentence weights
             are projected onto tokens exactly as the ARC-JSD baseline does
             (prompt-attention shares within each sentence).
mice_rag     MiCE adapted to the shared pool: gradient-guided candidate
             selection (top-M tokens by Gradient x Input), then greedy search
             that at each step applies the pool replacement which most lowers
             the probability of the original response, up to ReFlow's budget.
             The greedy order is a ranking, so smaller budgets are prefixes.
rage         RAGE adapted to sentences and the token budget: sentence sources
             in fixed relevance order (retrieval rank, then position);
             combinations searched by size, then order; a combination is
             intervened on by replacing its top-attention eligible tokens (at
             most ReFlow's budget) with pool alternatives; the first
             combination whose generated answer changes is returned.
rage_native  RAGE's own operation for reference: delete the sentences of each
             combination, same search order, no token budget. Reports the
             deleted-token count; not comparable with matched-budget methods.

Ranking methods write rows in the score-file schema (ranked_ids /
token_scores) read by evaluate_matched_budget_baselines.py. Every forward pass
and generation is recorded. RAGE's search generates with HF (answers capped at
64 new tokens, compared with the HF clean answer); the final verification of
the returned tokens is the shared vLLM reader in the matched-budget evaluator.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
import time

sys.path.insert(0, "/data1/yujia/CausalityRAG_release_no_frontier_repair_20260810")  # code behind Table III

import numpy as np  # noqa: E402

from causalityrag.io import iter_records, load_records, record_id, retrieved_contexts  # noqa: E402
from causalityrag.reader import (  # noqa: E402
    canonicalize_reader_answer,
    parse_json_object,
    reader_answers_exact_match,
)
from causalityrag.shared_replacement_pool import (  # noqa: E402
    FrozenSharedReplacementPool,
    stable_shared_candidate,
)
from causalityrag.token_units import units_from_cache_row  # noqa: E402
from exp.arc_jsd import ArcJsdModel, ablate_context_sentences, replace_context_units  # noqa: E402


def ranked_map(row: dict) -> dict[str, float]:
    if isinstance(row.get("token_scores"), dict):
        return {str(k): float(v) for k, v in row["token_scores"].items()}
    return {str(k): float(v) for k, v in zip(row.get("ranked_ids", []), row.get("ranked_scores", []))}


def ranked_list(row: dict) -> list[str]:
    if isinstance(row.get("ranked_ids"), list):
        return [str(x) for x in row["ranked_ids"]]
    return [k for k, _ in sorted(ranked_map(row).items(), key=lambda kv: -kv[1])]


class ResponseScorer:
    """Teacher-forced log P(frozen clean response | contexts), batched."""

    def __init__(self, model: ArcJsdModel, batch_size: int) -> None:
        self.model, self.batch_size, self.passes = model, batch_size, 0

    def log_probs(self, question: str, variants: list[list[dict]], response_ids: list[int]) -> list[float]:
        torch = self.model.torch
        out = []
        target = torch.tensor(response_ids, device=self.model.device)
        for start in range(0, len(variants), self.batch_size):
            prompts = [self.model._prompt_ids(question, v) for v in variants[start:start + self.batch_size]]
            lp = self.model._log_probs_for_prompt_ids(prompts, response_ids)  # [B, T, V]
            picked = lp.gather(-1, target[None, :, None].expand(lp.shape[0], -1, 1)).squeeze(-1)
            out.extend(picked.sum(-1).float().cpu().tolist())
            self.passes += len(prompts)
        return out


class AnswerGenerator:
    """Greedy reader answers for context variants, batched, parsed like the reader."""

    def __init__(self, model: ArcJsdModel, batch_size: int, max_new_tokens: int, reader_mode: str) -> None:
        self.model, self.batch_size, self.max_new = model, batch_size, max_new_tokens
        self.mode, self.generations = reader_mode, 0

    def answers(self, question: str, variants: list[list[dict]]) -> list[str]:
        torch, m, out = self.model.torch, self.model, []
        for start in range(0, len(variants), self.batch_size):
            prompts = [m._prompt_ids(question, v) for v in variants[start:start + self.batch_size]]
            width = max(len(p) for p in prompts)
            ids = torch.full((len(prompts), width), m.tokenizer.pad_token_id, dtype=torch.long, device=m.device)
            mask = torch.zeros_like(ids)
            for row, p in enumerate(prompts):
                ids[row, width - len(p):] = torch.tensor(p, dtype=torch.long, device=m.device)
                mask[row, width - len(p):] = 1
            with torch.inference_mode():
                gen = m.model.generate(input_ids=ids, attention_mask=mask, do_sample=False,
                                       max_new_tokens=self.max_new, pad_token_id=m.tokenizer.pad_token_id,
                                       eos_token_id=m.tokenizer.eos_token_id)
            for seq in gen[:, width:].tolist():
                if m.tokenizer.eos_token_id in seq:
                    seq = seq[:seq.index(m.tokenizer.eos_token_id)]
                text = m.tokenizer.decode(seq, skip_special_tokens=True).strip()
                parsed = parse_json_object(text)
                answer = str(parsed["answer"]).strip() if isinstance(parsed, dict) and parsed.get("answer") is not None else text
                out.append(canonicalize_reader_answer(question, answer, reader_mode=self.mode))
            self.generations += len(prompts)
        return out


def ordered_sentences(sentences: list[dict]) -> list[dict]:
    return sorted(sentences, key=lambda s: (int(s["chunk_rank"]), int(s["sentence_index"])))


def sentence_combinations(n: int, max_size: int, max_combos: int):
    combos = itertools.chain.from_iterable(itertools.combinations(range(n), size) for size in range(1, max_size + 1))
    return list(itertools.islice(combos, max_combos))


def search_first_change(record, contexts, generator, combos, make_variant, clean, args):
    """Evaluate combinations in order (batched); return (index, answer) of the first changed answer."""
    for start in range(0, len(combos), args.batch_size):
        chunk = combos[start:start + args.batch_size]
        answers = generator.answers(record["question"], [make_variant(c) for c in chunk])
        for offset, answer in enumerate(answers):
            if not reader_answers_exact_match(record["question"], answer, clean, reader_mode=args.reader_mode):
                return start + offset, answer
    return None, None


def rage(record, units, sentences, contexts, generator, attention, pool, budget, args):
    order = ordered_sentences(sentences)
    by_sentence: dict[str, list[str]] = {}
    for u in units:
        uid = str(u["unit_id"])
        if pool.is_eligible(uid):
            by_sentence.setdefault(str(u.get("sentence_id", "")), []).append(uid)
    by_id = {str(u["unit_id"]): u for u in units}

    def tokens_for(combo):
        ids = [x for i in combo for x in by_sentence.get(order[i]["sentence_id"], [])]
        return sorted(ids, key=lambda x: (-attention.get(x, 0.0), x))[:budget]

    combos = [c for c in sentence_combinations(len(order), args.rage_max_size, args.rage_max_combos) if tokens_for(c)]
    all_ids = [x for c in combos for x in tokens_for(c)]
    rows = pool.require(sorted(set(all_ids)))
    repl = {x: stable_shared_candidate(rows[x]["candidates"], unit_id=x, seed=args.replacement_seed) for x in rows}
    clean = generator.answers(record["question"], [contexts])[0]
    hit, answer = search_first_change(
        record, contexts, generator, combos,
        lambda c: replace_context_units(contexts, [by_id[x] for x in tokens_for(c)], repl), clean, args)
    first = tokens_for(combos[hit]) if hit is not None else []
    ranked = first + [x for x in dict.fromkeys(all_ids) if x not in first]
    return {"ranked_ids": ranked, "budget": budget, "search_success": hit is not None,
            "winning_combo": [order[i]["sentence_id"] for i in combos[hit]] if hit is not None else [],
            "combos_tried": (hit + 1) if hit is not None else len(combos), "combos_available": len(combos),
            "hf_clean_answer": clean, "hf_edited_answer": answer}


def rage_native(record, units, sentences, contexts, generator, args):
    order = ordered_sentences(sentences)
    combos = sentence_combinations(len(order), args.rage_max_size, args.rage_max_combos)
    clean = generator.answers(record["question"], [contexts])[0]
    hit, answer = search_first_change(
        record, contexts, generator, combos,
        lambda c: ablate_context_sentences(contexts, [order[i] for i in c]), clean, args)
    deleted = {order[i]["sentence_id"] for i in combos[hit]} if hit is not None else set()
    return {"ranked_ids": [], "native_flip": hit is not None,
            "deleted_sentences": sorted(deleted),
            "deleted_tokens": sum(1 for u in units if str(u.get("sentence_id", "")) in deleted),
            "combos_tried": (hit + 1) if hit is not None else len(combos),
            "hf_clean_answer": clean, "hf_edited_answer": answer}


def logit_prob(log_p: float) -> float:
    p = min(math.exp(log_p), 1.0 - 1e-6)
    return log_p - math.log1p(-p)


def contextcite(record, units, sentences, contexts, scorer, response_ids, attention, args, index):
    from sklearn.linear_model import Lasso

    rng = np.random.default_rng(args.seed + index)
    masks = rng.random((args.ablations, len(sentences))) < args.keep_prob  # True = sentence kept
    variants = [ablate_context_sentences(contexts, [s for s, keep in zip(sentences, m) if not keep]) for m in masks]
    y = np.array([logit_prob(v) for v in scorer.log_probs(record["question"], variants, response_ids)])
    lasso = Lasso(alpha=args.lasso_alpha, max_iter=10000).fit(masks.astype(float), y)
    weights = dict(zip((s["sentence_id"] for s in sentences), lasso.coef_.tolist()))

    by_sentence: dict[str, list[dict]] = {}
    for unit in units:
        by_sentence.setdefault(str(unit.get("sentence_id", "")), []).append(unit)
    token_scores = {}
    for sid, sent_units in by_sentence.items():
        masses = [attention.get(str(u["unit_id"]), 0.0) for u in sent_units]
        total = sum(masses)
        shares = [m / total for m in masses] if total > 0 else [1.0 / len(sent_units)] * len(sent_units)
        for u, share in zip(sent_units, shares):
            token_scores[str(u["unit_id"])] = max(0.0, weights.get(sid, 0.0)) * share
    ranked = sorted(token_scores, key=lambda u: (-token_scores[u], -attention.get(u, 0.0), u))
    return {"ranked_ids": ranked, "token_scores": token_scores,
            "sentence_weights": weights, "lasso_intercept": float(lasso.intercept_),
            "n_sentences": len(sentences), "ablations": int(args.ablations)}


def mice_rag(record, units, contexts, scorer, response_ids, gxi_ranked, pool, budget, args):
    by_id = {str(u["unit_id"]): u for u in units}
    eligible = [u for u in gxi_ranked if u in by_id and pool.is_eligible(u)]
    candidates = eligible[: args.mice_candidates]
    rows = pool.require(candidates)
    repl = {u: stable_shared_candidate(rows[u]["candidates"], unit_id=u, seed=args.replacement_seed) for u in candidates}
    chosen, trace = [], []
    for _ in range(min(budget, len(candidates))):
        rest = [u for u in candidates if u not in chosen]
        variants = [replace_context_units(contexts, [by_id[x] for x in chosen + [u]], repl) for u in rest]
        scores = scorer.log_probs(record["question"], variants, response_ids)
        best = int(np.argmin(scores))
        chosen.append(rest[best])
        trace.append({"unit_id": rest[best], "log_p": scores[best]})
    ranked = chosen + [u for u in eligible if u not in chosen]
    return {"ranked_ids": ranked, "greedy_trace": trace, "budget": budget, "n_candidates": len(candidates)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["contextcite", "mice_rag", "rage", "rage_native"], required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--units-cache", required=True)
    ap.add_argument("--reflow-results", required=True, help="main-table ReFlow rows: clean answer and budget")
    ap.add_argument("--prompt-attention", default="")
    ap.add_argument("--gxi-scores", default="")
    ap.add_argument("--shared-pool", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model-path", default="/data1/yujia/models/Qwen2.5-7B-Instruct")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--ablations", type=int, default=64)
    ap.add_argument("--keep-prob", type=float, default=0.5)
    ap.add_argument("--lasso-alpha", type=float, default=0.01)
    ap.add_argument("--mice-candidates", type=int, default=32)
    ap.add_argument("--max-budget", type=int, default=10)
    ap.add_argument("--rage-max-size", type=int, default=2)
    ap.add_argument("--rage-max-combos", type=int, default=64)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--reader-mode", default=os.environ.get("CAUSALITYRAG_READER_MODE", "short_answer"))
    ap.add_argument("--replacement-seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    unit_rows = {str(r["id"]): r for r in load_records(args.units_cache)}
    reflow = {str(r["id"]): r for r in load_records(args.reflow_results)}
    attention = {str(r["id"]): ranked_map(r) for r in load_records(args.prompt_attention)} if args.prompt_attention else {}
    gxi = {str(r["id"]): ranked_list(r) for r in load_records(args.gxi_scores)} if args.gxi_scores else {}
    pool = FrozenSharedReplacementPool(args.shared_pool) if args.shared_pool else None
    done = set()
    if os.path.exists(args.out):
        done = {json.loads(l)["id"] for l in open(args.out) if l.strip()}
    model = ArcJsdModel(args.model_path, attn_implementation="sdpa")
    scorer = ResponseScorer(model, args.batch_size)
    generator = AnswerGenerator(model, args.batch_size, args.max_new_tokens, args.reader_mode)

    with open(args.out, "a", encoding="utf-8") as out:
        for offset, record in enumerate(list(iter_records(args.input))[args.start:args.start + args.n]):
            index, qid = args.start + offset, record_id(record)
            if qid in done:
                continue
            t0, passes0, gens0 = time.monotonic(), scorer.passes, generator.generations
            row = {"index": index, "id": qid, "method": args.method}
            clean = str(reflow.get(qid, {}).get("clean_answer", "")).strip()
            try:
                if not clean:
                    raise ValueError("no frozen clean answer")
                contexts = retrieved_contexts(record)[: args.k]
                units = units_from_cache_row(record, unit_rows[qid], k=args.k)
                chunk_ids = {str(c["chunk_id"]) for c in contexts}
                sentences = [s for s in unit_rows[qid]["sentences"] if str(s["chunk_id"]) in chunk_ids]
                response = json.dumps({"answer": clean}, ensure_ascii=False)
                response_ids = model.tokenizer(response, add_special_tokens=False)["input_ids"]
                if args.method == "contextcite":
                    row.update(contextcite(record, units, sentences, contexts, scorer, response_ids,
                                           attention.get(qid, {}), args, index))
                elif args.method == "rage_native":
                    row.update(rage_native(record, units, sentences, contexts, generator, args))
                else:
                    budget = max(1, min(args.max_budget, int(reflow[qid].get("n_modified_tokens") or 1)))
                    if args.method == "mice_rag":
                        row.update(mice_rag(record, units, contexts, scorer, response_ids, gxi[qid], pool, budget, args))
                    else:
                        row.update(rage(record, units, sentences, contexts, generator,
                                        attention.get(qid, {}), pool, budget, args))
                row["status"] = "ok"
            except Exception as exc:  # recorded per query; evaluator treats it as a method failure
                row.update(status="error", error=f"{type(exc).__name__}: {exc}", ranked_ids=[])
            row.update(forward_passes=scorer.passes - passes0, generations=generator.generations - gens0,
                       elapsed_seconds=time.monotonic() - t0)
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            print(f"[{args.method}] {index} {row['status']} passes={row['forward_passes']} "
                  f"gens={row['generations']} sec={row['elapsed_seconds']:.1f}", flush=True)


if __name__ == "__main__":
    main()
