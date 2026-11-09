"""A3 reference: plain greedy generation with the Table 3 reader prompt, no tracing."""
import argparse, json, sys, time
sys.path.insert(0, "/data1/yujia/CausalityRAG_release_no_frontier_repair_20260810")
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from causalityrag.io import load_records, retrieved_contexts
from causalityrag.reader import reader_prompt

ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True); ap.add_argument("--n", type=int, default=100)
ap.add_argument("--k", type=int, default=5); ap.add_argument("--model-path", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(a.model_path)
config = AutoConfig.from_pretrained(a.model_path)
# Same workaround as ArcJsdModel: transformers 4.52 validates the TP plan even on one GPU.
config.base_model_tp_plan = None
config.base_model_pp_plan = None
model = AutoModelForCausalLM.from_pretrained(a.model_path, config=config, torch_dtype=torch.bfloat16).to("cuda").eval()
with open(a.out, "w") as out:
    for r in load_records(a.input)[: a.n]:
        system, user = reader_prompt(r["question"], retrieved_contexts(r)[: a.k], reader_mode="short_answer")
        text = tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                       tokenize=False, add_generation_prompt=True)
        ids = tok(text, return_tensors="pt").to("cuda")
        t0 = time.time()
        with torch.inference_mode():
            gen = model.generate(**ids, do_sample=False, max_new_tokens=128)
        out.write(json.dumps({"id": r["id"], "prompt_tokens": ids["input_ids"].shape[1], "seconds": time.time() - t0,
                              "answer": tok.decode(gen[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)}) + "\n")
print("torch_peak_mib", torch.cuda.max_memory_allocated() // 2**20)
