"""A6: ERASER comprehensiveness and sufficiency at the matched token budget.

Model output: the geometric-mean token probability of the frozen clean response
({"answer": ...}) under teacher forcing, p = exp(mean log p(y_t | ...)).
  comprehensiveness = p(full) - p(context with the selected tokens deleted)   (higher is better)
  comprehensiveness_replace = p(full) - p(selected tokens replaced by the shared-pool
                      alternatives used in the main run), i.e. the paper's intervention
  sufficiency       = p(full) - p(only the sentences containing them)          (lower is better)
Selections are each method's matched-budget tokens from the main run (qwen_a5;
TS from the original-ranking re-verification).
Usage: run_eraser.py DATASET OUT [--n N]
"""
import argparse
import json
import math
import os
import re
import sys

sys.path.insert(0, "/data1/yujia/CausalityRAG_release_no_frontier_repair_20260810")
from causalityrag.io import load_records, record_id, retrieved_contexts  # noqa: E402
from causalityrag.revision import apply_token_replacements  # noqa: E402
from causalityrag.shared_replacement_pool import (  # noqa: E402
    FrozenSharedReplacementPool,
    stable_shared_candidate,
)
from causalityrag.token_units import units_from_cache_row  # noqa: E402
from exp.arc_jsd import ArcJsdModel  # noqa: E402

S = "/data1/yujia/CausalityRAG"
P = "/data1/yujia/CausalityRAG_reviewer_baselines_20261023_v1/pool_sensitivity"
R = "/data1/yujia/CausalityRAG_reviewer_baselines_20261023_v1"
BASES = {"hotpotqa": f"{S}/out/hotpotqa", "musique": f"{S}/out/musique", "finqa": f"{S}/out/finqa",
         "popqa": f"{S}/out_table3_final8_20260811/popqa", "2wiki": f"{S}/out/2wiki", "timeqa": f"{S}/out/timeqa",
         "triviaqa": f"{S}/out/triviaqa", "quartz": f"{S}/out/quartz"}
# For these datasets the main-table rows (ReFlow, original baselines) and the
# revision baselines (TS, ContextCite, MiCE-RAG, RAGE) were verified separately.
SPLIT = {"2wiki", "timeqa", "triviaqa", "quartz"}
UNITS = {"triviaqa": "context_units_top10_1000.jsonl"}
POOLS = {ds: f"{b}/replacements/shared_pool_top10_v1/"
             + ("shared_pool_top10_final.jsonl" if ds in ("hotpotqa", "triviaqa") else "shared_pool.jsonl")
         for ds, b in BASES.items()}


def load_selections(ds):
    """Return (reflow rows, {id: baseline row}, {id: TS row}) for a dataset.

    MiCE-RAG selections come from the first-flip run (qwen_a5_miceflip) when present.
    """
    reflow, base_rows, ts_rows = _load_selections(ds)
    flip = f"{P}/qwen_a5_miceflip/{ds}/baselines_1000.jsonl"
    if os.path.exists(flip):
        for r in load_records(flip):
            entry = r.get("methods", {}).get("mice_rag")
            if entry is not None and r["id"] in base_rows:
                base_rows[r["id"]].setdefault("methods", {})["mice_rag"] = entry
    else:
        for r in base_rows.values():
            r.get("methods", {}).pop("mice_rag", None)  # never score the superseded greedy variant
    return reflow, base_rows, ts_rows


def _load_selections(ds):
    if ds not in SPLIT:
        reflow = load_records(f"{P}/qwen_a5/{ds}/reflow_1000.jsonl")
        base_rows = {r["id"]: r for r in load_records(f"{P}/qwen_a5/{ds}/baselines_1000.jsonl")}
        ts_path = f"{P}/qwen_a5_tsorig/{ds}/baselines_1000.jsonl"
        # original-paper TS rankings where they were re-verified; otherwise the TS rows of this run
        ts_rows = {r["id"]: r for r in load_records(ts_path)} if os.path.exists(ts_path) else base_rows
        return reflow, base_rows, ts_rows
    main = f"{S}/out_no_frontier_repair_20260810/{ds}"
    reflow = load_records(f"{main}/results_1000.jsonl")
    base_rows = {r["id"]: r for r in load_records(f"{main}/table3_resume/baselines_1000.jsonl")}
    for r in load_records(f"{R}/a5_cfg/{ds}/baselines_1000.jsonl"):
        base_rows.setdefault(r["id"], {"methods": {}}).setdefault("methods", {}).update(
            {m: v for m, v in r.get("methods", {}).items() if not m.startswith("random")})
    return reflow, base_rows, base_rows
METHODS = ["random_seed0", "attention", "attention_prompt_last", "gradient_x_input", "mirage", "arc_jsd",
           "contextcite", "mice_rag", "rage"]


def delete_spans(contexts, units):
    by_chunk = {}
    for u in units:
        by_chunk.setdefault(str(u["chunk_id"]), []).append(u)
    out = []
    for c in contexts:
        text = str(c["text"])
        for u in sorted(by_chunk.get(str(c["chunk_id"]), []), key=lambda x: -int(x["chunk_char_start"])):
            text = text[: int(u["chunk_char_start"])] + text[int(u["chunk_char_end"]):]
        out.append(dict(c, text=re.sub(r"[ \t]{2,}", " ", text).strip()))
    return out


def keep_sentences(contexts, sentences, keep_ids):
    out = []
    for c in contexts:
        kept = [s for s in sorted(sentences, key=lambda s: int(s["chunk_char_start"]))
                if str(s["chunk_id"]) == str(c["chunk_id"]) and s["sentence_id"] in keep_ids]
        out.append(dict(c, text=" ".join(str(s["text"]) for s in kept)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--only", default="", help="comma-separated methods to score (default: all)")
    args = ap.parse_args()
    base = BASES[args.dataset]
    records = {record_id(r): r for r in load_records(f"{base}/retrieval/top10_1000.jsonl")}
    units_rows = {str(r["id"]): r for r in load_records(f"{base}/inputs/{UNITS.get(args.dataset, 'token_units_top10_1000.jsonl')}")}
    reflow, base_rows, ts_rows = load_selections(args.dataset)
    done = {json.loads(l)["id"] for l in open(args.out)} if os.path.exists(args.out) else set()
    pool = FrozenSharedReplacementPool(POOLS[args.dataset])
    model = ArcJsdModel("/data1/yujia/models/Qwen2.5-7B-Instruct", attn_implementation="sdpa")
    torch = model.torch

    def mean_logp(question, variants, response_ids):
        target = torch.tensor(response_ids, device=model.device)
        vals = []
        for i in range(0, len(variants), args.batch_size):
            prompts = [model._prompt_ids(question, v) for v in variants[i:i + args.batch_size]]
            lp = model._log_probs_for_prompt_ids(prompts, response_ids)
            picked = lp.gather(-1, target[None, :, None].expand(lp.shape[0], -1, 1)).squeeze(-1)
            vals.extend(picked.mean(-1).float().cpu().tolist())
        return vals

    with open(args.out, "a") as out:
        for rf in reflow[: args.n]:
            q = rf["id"]
            if q in done or rf.get("evaluation_status") == "invalid_clean_answer" or not str(rf.get("clean_answer", "")).strip():
                continue
            record = records[q]
            question = str(record["question"])
            contexts = retrieved_contexts(record)[:5]
            by_id = {str(u["unit_id"]): u for u in units_from_cache_row(record, units_rows[q], k=5)}
            chunk_ids = {str(c["chunk_id"]) for c in contexts}
            sentences = [s for s in units_rows[q]["sentences"] if str(s["chunk_id"]) in chunk_ids]
            selections = {"reflow": rf.get("selected_ids") or []}
            for m in METHODS:
                sel = base_rows.get(q, {}).get("methods", {}).get(m, {}).get("selected_ids")
                if sel:
                    selections["random" if m == "random_seed0" else m] = sel
            ts_sel = ts_rows.get(q, {}).get("methods", {}).get("ts", {}).get("selected_ids")
            if ts_sel:
                selections["ts"] = ts_sel
            if args.only:
                selections = {m: v for m, v in selections.items() if m in args.only.split(",")}
            names, variants = ["full"], [contexts]
            for m, sel in selections.items():
                units = [by_id[u] for u in sel if u in by_id]
                if not units:
                    continue
                names += [f"{m}|comp", f"{m}|suff"]
                variants += [delete_spans(contexts, units),
                             keep_sentences(contexts, sentences, {str(u.get("sentence_id", "")) for u in units})]
                ids = [str(u["unit_id"]) for u in units]
                if all(pool.is_eligible(u) for u in ids):
                    rows = pool.require(ids)
                    repl = {u: stable_shared_candidate(rows[u]["candidates"], unit_id=u, seed=0) for u in ids}
                    rev = apply_token_replacements(record, units, repl, k=5)
                    if int(rev["n_failed_edits"]) == 0:
                        names.append(f"{m}|repl")
                        variants.append(rev["edited_contexts"])
            response = json.dumps({"answer": str(rf["clean_answer"]).strip()}, ensure_ascii=False)
            response_ids = model.tokenizer(response, add_special_tokens=False)["input_ids"]
            probs = dict(zip(names, (math.exp(v) for v in mean_logp(question, variants, response_ids))))
            row = {"id": q, "p_full": probs["full"], "methods": {}}
            for m, sel in selections.items():
                if f"{m}|comp" in probs:
                    row["methods"][m] = {"n_tokens": len(sel),
                                         "comprehensiveness": probs["full"] - probs[f"{m}|comp"],
                                         "sufficiency": probs["full"] - probs[f"{m}|suff"]}
                    if f"{m}|repl" in probs:
                        row["methods"][m]["comprehensiveness_replace"] = probs["full"] - probs[f"{m}|repl"]
            out.write(json.dumps(row) + "\n")
            out.flush()


if __name__ == "__main__":
    main()
