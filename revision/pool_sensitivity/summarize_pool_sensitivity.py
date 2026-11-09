"""Collect FR / CFR per method for each (pool, dataset) next to the paper's main-table run."""
import json, os, sys
P = "/data1/yujia/CausalityRAG_reviewer_baselines_20261023_v1/pool_sensitivity"
PAPER = "/data1/yujia/CausalityRAG/out_no_frontier_repair_20260810/{}/table3_resume/table3_metrics_1000.json"
NAMES = {"random": "Random", "attention": "Attn", "attention_prompt_last": "Attn", "gradient_x_input": "GxI",
         "mirage": "MIRAGE", "arc_jsd": "ARC-JSD", "reflow": "ReFlow"}
out = {}
for ds in ("hotpotqa", "musique", "finqa"):
    for tag, path in [("paper", PAPER.format(ds))] + [(t, f"{P}/{t}/{ds}/table3_metrics_1000.json")
                                                       for t in ("qwen_original", "llama31_8b", "gpt56_sol")]:
        if not os.path.exists(path):
            continue
        d = json.load(open(path))
        row = {}
        for m, v in d["methods"].items():
            if m in NAMES:
                row[NAMES[m]] = {**{k: round(100 * v[k]["factual_flip_ratio"], 1) for k in ("answer", "em")},
                                 **{k + "_cfr": round(100 * v[k]["adjusted_flip_ratio"], 1) for k in ("answer", "em")}}
        out[f"{ds}/{tag}"] = {"population": d.get("population", {}).get("answer", "")[:60], "methods": row}
json.dump(out, open(f"{P}/pool_sensitivity_summary.json", "w"), indent=1)
cols = ["Random", "Attn", "GxI", "MIRAGE", "ARC-JSD", "ReFlow"]
for metric in ("answer", "answer_cfr", "em", "em_cfr"):
    print(f"\n### {metric}")
    print(f"{'dataset/pool':26s}" + "".join(f"{c:>9s}" for c in cols) + "   gain")
    for key, v in out.items():
        vals = [v["methods"].get(c, {}).get(metric) for c in cols]
        best = max(x for x in vals[:-1] if x is not None)
        print(f"{key:26s}" + "".join(f"{x:9.1f}" if x is not None else f"{'-':>9s}" for x in vals) + f"  {vals[-1]-best:+5.1f}")
print("\npopulations:", {k: v["population"] for k, v in out.items()})
