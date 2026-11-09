"""A8: accuracy of the first-order message contribution r (Eq. attention-contribution).

Per query, r^(l)(j,i) = <dS/d attn_out_i^(l), m_{j->i}^(l)> with m_{j->i} = W_O sum_h a^h_ij v^h_j,
computed exactly as graph construction does. We take the 200 messages with the largest |r| and the
100 largest whose source j is a retrieved-context token, remove each message alone in a real forward
pass (subtract m from the attention output of receiver i at layer l), and record the exact change dS
of the answer objective S (mean teacher-forced answer logit). First-order prediction: dS = -r.

Two stages, so each fits on one GPU:
  --stage r       bf16 model, forward/backward as graph construction; saves r and the messages (.pt).
  --stage remove  fp32 model; measures each removal. In bf16 the run-to-run rounding of S (~0.1) is
                  comparable to single-message effects, and TF32 changes dS by up to ~10 (activation
                  outliers), so removals run in full fp32. Removing a message at receiver i cannot
                  change positions < i (causal attention), so a batch of removals reuses the clean KV
                  cache up to its smallest receiver and recomputes only the positions after it.
"""
import argparse, json, os, sys
sys.path.insert(0, "/data1/yujia/CausalityRAG_release_model_sweep_20260807")
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--stage", choices=["r", "remove"], required=True)
ap.add_argument("--input"); ap.add_argument("--results")
ap.add_argument("--model-path", required=True); ap.add_argument("--messages", required=True)
ap.add_argument("--out")
ap.add_argument("--k", type=int, default=5); ap.add_argument("--n", type=int, default=100)
ap.add_argument("--top", type=int, default=200); ap.add_argument("--top-context", type=int, default=100)
ap.add_argument("--batch", type=int, default=12); ap.add_argument("--max-tokens", type=int, default=6000)
a = ap.parse_args()


def stage_r():
    from causalityrag.io import load_records, retrieved_contexts
    from causalityrag.message_flow import _ClosedMessageFlowExtractor
    ex = _ClosedMessageFlowExtractor(a.model_path, device="cuda", dtype="bfloat16", max_context_tokens=0,
                                     max_length=0, max_receivers_per_layer=48)
    model, decoder = ex.model, ex.decoder
    layers = decoder.layers
    cfg = layers[0].self_attn.config
    assert not getattr(cfg, "use_sliding_window", False), "sliding-window attention is not handled"
    H, KV = cfg.num_attention_heads, cfg.num_key_value_heads

    def layer_messages(l, hidden, grad, pos_emb):
        """Attention weights [H,T,T], per-head values [T,H,d] and r [T(receiver), T(source)] for layer l."""
        m = layers[l].self_attn
        T, d = hidden.shape[0], m.head_dim
        q = m.q_proj(hidden).view(1, T, H, d).transpose(1, 2)
        k = m.k_proj(hidden).view(1, T, KV, d).transpose(1, 2)
        cos, sin = (x.unsqueeze(1).to(q.dtype) for x in pos_emb)
        q, k = q * cos + ex._rotate_half(q) * sin, k * cos + ex._rotate_half(k) * sin
        k = k.repeat_interleave(H // KV, dim=1)
        s = torch.matmul(q, k.transpose(2, 3)) * m.scaling
        s.masked_fill_(torch.ones(T, T, dtype=torch.bool, device=s.device).triu(1), torch.finfo(s.dtype).min)
        attn = torch.softmax(s, dim=-1, dtype=torch.float32).to(q.dtype)[0]
        v = m.v_proj(hidden).view(T, KV, d).float().repeat_interleave(H // KV, dim=1)
        og = torch.matmul(grad.float(), m.o_proj.weight.float()).view(T, H, d)
        r = (torch.einsum("thd,shd->hts", og, v) * attn).sum(0)
        return attn, v, r

    records = {r["id"]: r for r in load_records(a.input)}
    results = [r for r in load_records(a.results) if r.get("eligible", True) and r.get("clean_answer")][: a.n]
    saved = []
    for res in results:
        rec = records[res["id"]]
        contexts = ex._truncate_contexts(retrieved_contexts(rec)[: a.k])
        text, spans = ex._render(rec, contexts, res["clean_answer"])
        enc = ex._encode_without_truncation(text)
        ids = enc["input_ids"].to("cuda")
        if ids.shape[1] > a.max_tokens:
            saved.append({"id": res["id"], "status": "too_long", "tokens": ids.shape[1]}); continue
        meta = ex._token_metadata(ids[0].tolist(), [tuple(map(int, p)) for p in enc["offset_mapping"][0].tolist()], spans)
        targets_pos = [m["position"] for m in meta if m["region"] == "answer" and m["position"] > 0]
        is_ctx = torch.tensor([m["region"] == "context" for m in meta], device="cuda")

        trace = ex._install_trace_hooks()
        try:
            with torch.enable_grad():
                emb = model.get_input_embeddings()(ids).detach().requires_grad_(True)
                logits = model(inputs_embeds=emb, use_cache=False).logits
                S, _, _ = ex._build_target_objective(logits, ids, targets_pos, meta)
                S.backward()
            del logits
            with torch.no_grad():
                pos_emb = decoder.rotary_emb(emb.detach(), torch.arange(ids.shape[1], device="cuda").unsqueeze(0))
                cands = []
                for l in range(len(layers)):
                    attn, v, r = layer_messages(l, trace.attn_inputs[l][0].detach(),
                                                trace.attn_outputs[l].grad[0].detach(), pos_emb)
                    T = r.shape[0]
                    w_o = layers[l].self_attn.o_proj.weight.float()
                    picks = [("top", r.abs().flatten().topk(a.top).indices),
                             ("context", (r.abs() * is_ctx.view(1, -1)).flatten().topk(a.top_context).indices)]
                    for kind, idx in picks:
                        for f in idx.tolist():
                            i, j = divmod(f, T)
                            vec = w_o @ (attn[:, i, j].float().unsqueeze(-1) * v[j]).reshape(-1)
                            cands.append((kind, float(r[i, j]), l, i, j, vec))
                    # keep only the messages that can still make the global cut
                    keep = []
                    for kind, n in (("top", a.top), ("context", a.top_context)):
                        keep += sorted((c for c in cands if c[0] == kind), key=lambda c: -abs(c[1]))[:n]
                    cands = keep
                    del attn, v, r
        finally:
            trace.close()
        model.zero_grad(set_to_none=True)
        # one entry per distinct message; a message can be in both sets
        uniq = {}
        for kind, r, l, i, j, vec in cands:
            uniq.setdefault((l, i, j), {"sets": [], "r": r, "vec": vec})["sets"].append(kind)
        keys = list(uniq)
        saved.append({"id": res["id"], "status": "ok", "tokens": ids.shape[1], "S": float(S.detach()),
                      "ids": ids[0].cpu(), "targets_pos": targets_pos,
                      "msgs": [{"layer": l, "receiver": i, "source": j, "source_region": meta[j]["region"],
                                "r": uniq[(l, i, j)]["r"], "sets": uniq[(l, i, j)]["sets"]} for l, i, j in keys],
                      "vecs": torch.stack([uniq[k]["vec"] for k in keys]).cpu()})
        del S, emb
        print(f"[r] {len(saved)}/{len(results)} tokens={ids.shape[1]} messages={len(keys)}", flush=True)
    torch.save(saved, a.messages + ".tmp")
    os.replace(a.messages + ".tmp", a.messages)  # the remove stage waits for this file


def stage_remove():
    from transformers import AutoConfig, AutoModelForCausalLM, DynamicCache
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = AutoConfig.from_pretrained(a.model_path)
    config.base_model_tp_plan = None
    ref = AutoModelForCausalLM.from_pretrained(a.model_path, config=config, torch_dtype=torch.float32,
                                               attn_implementation="sdpa").to("cuda").eval()
    ablate = {}  # layer -> list of (batch index, offset of the receiver in the recomputed suffix, message)

    def make_hook(l):
        def hook(_module, _inputs, output):
            if l not in ablate:
                return None
            out = output[0].clone()
            for b, offset, vec in ablate[l]:
                out[b, offset] -= vec
            return (out,) + tuple(output[1:])
        return hook
    for l, layer in enumerate(ref.model.layers):
        layer.self_attn.register_forward_hook(make_hook(l))

    done = set()
    try:
        done = {json.loads(l)["id"] for l in open(a.out)}
    except FileNotFoundError:
        pass
    with open(a.out, "a") as out, torch.no_grad():
        for q in torch.load(a.messages, weights_only=False):
            if q["id"] in done:
                continue
            if q["status"] != "ok":
                out.write(json.dumps({k: q[k] for k in ("id", "status", "tokens")}) + "\n"); continue
            ids = q["ids"].to("cuda").unsqueeze(0)
            T = ids.shape[1]
            preds = torch.tensor([p - 1 for p in q["targets_pos"]], device="cuda")
            targets = ids[0, q["targets_pos"]]
            vecs = q["vecs"].to("cuda")
            ablate.clear()
            clean = ref.model(input_ids=ids, use_cache=True)  # logits only at the answer predictors
            clean_vals = ref.lm_head(clean.last_hidden_state[0, preds]).gather(-1, targets.view(-1, 1)).squeeze(-1)
            S_ref = float(clean_vals.mean())
            legacy = clean.past_key_values.to_legacy_cache()
            del clean
            order = sorted(range(len(q["msgs"])), key=lambda idx: q["msgs"][idx]["receiver"])
            dS, slot0 = {}, []
            start = 0
            while start < len(order):
                i0 = q["msgs"][order[start]]["receiver"]  # smallest receiver in this batch
                # fp32 activations scale with batch x suffix length; keep that product bounded (40 GB GPUs)
                size = max(1, min(a.batch - 1, (a.batch - 1) * 1000 // max(1000, T - i0)))
                chunk = order[start: start + size]
                start += size
                n = len(chunk) + 1  # slot 0 = no removal, checks the cached path against S_ref
                late = preds >= i0  # predictors whose logits can change
                ablate.clear()
                for b, idx in enumerate(chunk, start=1):
                    m = q["msgs"][idx]
                    ablate.setdefault(m["layer"], []).append((b, m["receiver"] - i0, vecs[idx]))
                cache = DynamicCache.from_legacy_cache(tuple(
                    (k[:, :, :i0].expand(n, -1, -1, -1).contiguous(), v[:, :, :i0].expand(n, -1, -1, -1).contiguous())
                    for k, v in legacy))
                pos = torch.arange(i0, T, device="cuda")
                hidden = ref.model(input_ids=ids[:, i0:].expand(n, -1), past_key_values=cache,
                                   position_ids=pos.expand(n, -1), cache_position=pos, use_cache=True).last_hidden_state
                vals = clean_vals.expand(n, -1).clone()
                if late.any():
                    lg = ref.lm_head(hidden[:, preds[late] - i0])
                    vals[:, late] = lg.gather(-1, targets[late].view(1, -1, 1).expand(n, -1, 1)).squeeze(-1)
                s_b = vals.mean(-1)
                slot0.append(abs(float(s_b[0]) - S_ref))
                for b, idx in enumerate(chunk, start=1):
                    dS[idx] = float(s_b[b] - s_b[0])
                del cache, hidden
            ablate.clear()
            rows = [{"set": kind, **{k: m[k] for k in ("layer", "receiver", "source", "source_region", "r")}, "dS": dS[idx]}
                    for idx, m in enumerate(q["msgs"]) for kind in m["sets"]]
            out.write(json.dumps({"id": q["id"], "status": "ok", "tokens": T, "S": q["S"], "S_fp32": S_ref,
                                  "cache_path_max_abs_err": max(slot0), "messages": rows}) + "\n")
            out.flush()
            print(f"[remove] {q['id']} tokens={T} messages={len(q['msgs'])}", flush=True)


stage_r() if a.stage == "r" else stage_remove()
