"""Table II with the revision baselines: FR / CFR per method, matched budget, Qwen pool.

Prints the recomputed run (qwen_a5) for each dataset; the paper's TS cells are
read from table3_metric_rows.tex (passed as text on stdin) to check that the
recomputed TS reproduces them.
"""
import json, os, re, sys

P = "/data1/yujia/CausalityRAG_reviewer_baselines_20261023_v1/pool_sensitivity"
ORDER = [("random", "Random"), ("attention", "Attn"), ("attention_prompt_last", "Attn"),
         ("gradient_x_input", "GxI"), ("ts", "TS"), ("mirage", "MIRAGE"), ("arc_jsd", "ARC-JSD"),
         ("contextcite", "CtxCite"), ("mice_rag", "MiCE-RAG"), ("rage", "RAGE"), ("reflow", "ReFlow")]
COLS = ["Random", "Attn", "GxI", "TS", "MIRAGE", "ARC-JSD", "CtxCite", "MiCE-RAG", "RAGE", "ReFlow"]
MACRO = {"hotpotqa": "HotpotQA", "musique": "MuSiQue", "popqa": "PopQA", "finqa": "FinQA"}

tex = sys.stdin.read()
paper_ts = {}
for ds, name in MACRO.items():
    m = re.search(r"\\" + name + r"MetricRows\}\{%(.*?)\n\}", tex, re.S)
    if not m:
        continue
    rows = {}
    for line in m.group(1).splitlines():
        cells = re.findall(r"\\rag(?:cell|best)\{[0-9A-Fa-f]{6}\}\{([^}]*)\}", line)
        label = re.search(r"&\s*([A-Za-z]+-C?FR)", line)
        if cells and label:
            rows[label.group(1)] = float(cells[3])  # column 4 is TS
    paper_ts[ds] = rows

summary = {}
for ds in ("hotpotqa", "musique", "popqa", "finqa"):
    path = f"{P}/qwen_a5/{ds}/table3_metrics_1000.json"
    if not os.path.exists(path):
        continue
    d = json.load(open(path))["methods"]
    row = {}
    for key, name in ORDER:
        if key in d:
            row[name] = {f"{k}{suffix}": round(100 * d[key][k][field], 1)
                         for k in ("answer", "em", "acc") if k in d[key]
                         for suffix, field in (("", "factual_flip_ratio"), ("_cfr", "adjusted_flip_ratio"))}
    summary[ds] = row

for metric in ("answer", "answer_cfr", "em", "em_cfr"):
    print(f"\n### {metric}")
    print(f"{'dataset':10s}" + "".join(f"{c:>9s}" for c in COLS) + "   best-baseline  gain")
    for ds, row in summary.items():
        vals = {c: row.get(c, {}).get(metric) for c in COLS}
        base = {c: v for c, v in vals.items() if c != "ReFlow" and v is not None}
        best = max(base, key=base.get)
        print(f"{ds:10s}" + "".join(f"{vals[c]:9.1f}" if vals[c] is not None else f"{'-':>9s}" for c in COLS)
              + f"   {best:>12s}  {vals['ReFlow'] - base[best]:+5.1f}")
print("\nTS recomputed vs paper (Ans-FR, Ans-CFR, EM-FR):")
for ds, row in summary.items():
    p = paper_ts.get(ds, {})
    r = row.get("TS", {})
    print(f"  {ds:9s} recomputed {r.get('answer')} / {r.get('answer_cfr')} / {r.get('em')}   "
          f"paper {p.get('Ans-FR')} / {p.get('Ans-CFR')} / {p.get('EM-FR')}")
json.dump(summary, open(f"{P}/a5_summary.json", "w"), indent=1)
