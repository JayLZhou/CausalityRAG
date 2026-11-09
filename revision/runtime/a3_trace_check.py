"""A3: check that contribution tracing does not change the model's output.

For each query, compare (i) the next-token logits at the answer positions from a plain
forward pass and from the traced forward pass that graph construction runs (hooks
installed, gradients enabled), and (ii) greedy generations with and without the hooks.
"""
import argparse, json, sys
sys.path.insert(0, "/data1/yujia/CausalityRAG_release_model_sweep_20260807")
import torch
from causalityrag.io import load_records, retrieved_contexts
from causalityrag.message_flow import _ClosedMessageFlowExtractor

ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True); ap.add_argument("--clean-reference", required=True)
ap.add_argument("--model-path", required=True); ap.add_argument("--n", type=int, default=100)
ap.add_argument("--k", type=int, default=5); ap.add_argument("--out", required=True)
a = ap.parse_args()
ex = _ClosedMessageFlowExtractor(a.model_path, device="cuda", dtype="bfloat16", max_context_tokens=0,
                                 max_length=0, max_receivers_per_layer=48)
clean = {r["id"]: r["clean_answer"] for r in load_records(a.clean_reference)}
model, tok = ex.model, ex.tokenizer
agg = {"queries": 0, "answer_positions": 0, "argmax_agree": 0, "max_abs_logit_diff": 0.0, "generation_identical": 0}
with open(a.out, "w") as out:
    for r in load_records(a.input)[: a.n]:
        target = clean[r["id"]]
        contexts = ex._truncate_contexts(retrieved_contexts(r)[: a.k])
        text, spans = ex._render(r, contexts, target)
        ids = ex._encode_without_truncation(text)["input_ids"].to("cuda")
        prompt = text[: len(text) - len(json.dumps({"answer": target}, ensure_ascii=False))]
        pids = ex._encode_without_truncation(prompt)["input_ids"].to("cuda")
        first = pids.shape[1]  # positions >= first belong to the completion
        with torch.no_grad():
            plain = model(input_ids=ids, use_cache=False).logits[0, first - 1:-1].float()
            gen_plain = model.generate(pids, attention_mask=torch.ones_like(pids), do_sample=False, max_new_tokens=128)
        trace = ex._install_trace_hooks()
        try:
            with torch.enable_grad():
                emb = model.get_input_embeddings()(ids).detach().requires_grad_(True)
                traced = model(inputs_embeds=emb, use_cache=False).logits[0, first - 1:-1].detach().float()
            with torch.no_grad():
                gen_traced = model.generate(pids, attention_mask=torch.ones_like(pids), do_sample=False, max_new_tokens=128)
        finally:
            trace.close()
        model.zero_grad(set_to_none=True)
        agree = int((plain.argmax(-1) == traced.argmax(-1)).sum())
        diff = float((plain - traced).abs().max())
        same = bool(torch.equal(gen_plain, gen_traced))
        row = {"id": r["id"], "positions": plain.shape[0], "argmax_agree": agree, "max_abs_logit_diff": diff,
               "generation_identical": same,
               "generation": tok.decode(gen_plain[0, first:], skip_special_tokens=True)}
        out.write(json.dumps(row) + "\n")
        agg["queries"] += 1; agg["answer_positions"] += row["positions"]; agg["argmax_agree"] += agree
        agg["max_abs_logit_diff"] = max(agg["max_abs_logit_diff"], diff); agg["generation_identical"] += same
print(json.dumps(agg))
json.dump(agg, open(a.out.replace(".jsonl", ".summary.json"), "w"), indent=1)
